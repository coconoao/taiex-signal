"""Research-only event normalizer for TAIEX Signal.

This module reuses the existing RSS/news discovery in news.py. It does not
call an LLM, create a market forecast, write to Supabase, or alter production
outputs. The deterministic labels are candidate metadata for human review.
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import hashlib
import json
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import news

UTC = dt.timezone.utc
TZ = ZoneInfo("Asia/Taipei")
DEFAULT_WATCHLIST = Path(__file__).parent / "research/event_monitoring/people-watchlist-v1.json"
DEFAULT_OUTPUT = Path("research_data/event_monitor_history.jsonl")

EVENT_TERMS = {
    "policy_trade": ("tariff", "關稅", "trade", "貿易", "export control", "export", "出口", "出口管制", "sanction", "制裁"),
    "monetary_macro": ("fed", "federal reserve", "利率", "聯準會", "央行", "inflation", "通膨", "匯率", "gdp"),
    "geopolitics": ("taiwan strait", "台海", "geopolit", "外交", "國安", "military", "戰爭", "conflict"),
    "semiconductor_ai": ("semiconductor", "半導體", "chip", "晶片", "tsmc", "台積電", "nvidia", "黃仁勳", "gpu", "ai"),
    "corporate": ("earnings", "財報", "營收", "capex", "投資", "order", "訂單", "ceo", "董事長", "訴訟", "investigation"),
    "market_structure": ("index", "指數", "exchange", "交易所", "short selling", "融券", "熔斷"),
    "shock": ("earthquake", "地震", "typhoon", "颱風", "outage", "停電", "blackout", "disruption", "事故"),
}
TAIWAN_TERMS = (
    "taiwan", "台灣", "台股", "taiex", "加權指數", "tsmc", "台積電",
    "semiconductor", "半導體", "chip", "晶片", "supply chain", "供應鏈",
    "export control", "出口管制", "tariff", "關稅", "fed", "聯準會",
)
ALLEGATION_TERMS = ("alleged", "allegedly", "unverified", "rumor", "rumour", "reportedly",
                    "claims that", "傳出", "據報", "未證實", "指稱", "疑似")
REVIEW_TERMS = ("investigation", "probe", "spy", "intelligence", "間諜", "調查", "制裁", "軍事")


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def iso_utc(value: dt.datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("UTC timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_published(raw: str | None) -> tuple[str | None, str | None, str]:
    """Return (normalized UTC, source timezone label, quality); never guess a zone."""
    value = str(raw or "").strip()
    if not value:
        return None, None, "unknown"
    parsed: dt.datetime | None = None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        pass
    if parsed is None:
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None, None, "unknown"
    if parsed.tzinfo is None:
        return None, None, "source_timestamp_unzoned"
    zone = str(parsed.tzinfo)
    return iso_utc(parsed), zone, "source_timestamp_verified"


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\u4e00-\u9fff]+", " ", value.casefold())).strip()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _contains(text: str, term: str) -> bool:
    term = term.casefold()
    if re.fullmatch(r"[a-z0-9 .'-]+", term):
        return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text.casefold()) is not None
    return term in text


def match_entities(item: dict[str, Any], watchlist: dict[str, Any]) -> list[dict[str, Any]]:
    text = f"{item.get('title', '')} {item.get('description', '')}"
    low = text.casefold()
    matches = []
    for entity in watchlist.get("entities", []):
        found = None
        for alias in entity.get("aliases", []):
            if not _contains(low, alias):
                continue
            # "Trump" alone is ambiguous; require political context unless full name/title appears.
            if alias.casefold() == "trump" and not any(
                marker in low for marker in ("donald trump", "president trump", "white house", "川普", "特朗普")
            ):
                continue
            found = alias
            break
        if found:
            matches.append({
                "entity_id": entity["entity_id"],
                "name": entity["display_name"],
                "entity_type": "role" if entity.get("entity_type") == "role" else entity.get("entity_type", "other"),
                "match_method": "exact_alias",
                "confidence": 0.9 if len(found) > 4 else 0.75,
            })
    return matches


def classify_event(text: str) -> str:
    low = text.casefold()
    for category in ("policy_trade", "monetary_macro", "geopolitics", "semiconductor_ai",
                     "corporate", "market_structure", "shock"):
        if any(term in low for term in EVENT_TERMS[category]):
            return category
    return "other"


def source_class(domain: str, source_name: str) -> str:
    host = (domain or "").lower().removeprefix("www.")
    if host.endswith((".gov", ".gov.tw", ".gov.uk", ".gov.au")):
        return "official_government"
    if any(host == d or host.endswith("." + d) for d in ("tsmc.com", "nvidia.com", "amd.com", "intel.com")):
        return "official_company"
    if "x.com" == host or host.endswith(".x.com"):
        return "public_figure_statement"
    if source_name.lower() in ("google news", "bing news"):
        return "aggregator"
    return "news_media"


def _domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return ""


def build_event(item: dict[str, Any], watchlist: dict[str, Any], now: dt.datetime) -> dict[str, Any] | None:
    title = str(item.get("title") or "").strip()
    url = str(item.get("link") or "").strip()
    if not title or not url:
        return None
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
        return None

    raw_published = str(item.get("published") or "").strip()
    published_utc, published_timezone, timestamp_quality = parse_published(raw_published)
    domain = _domain(url)
    source_name = str(item.get("source") or item.get("tag") or domain or "unknown")
    source_kind = source_class(domain, source_name)
    combined = f"{title} {item.get('description', '')}"
    low = combined.casefold()
    evidence_status = "official_confirmed" if source_kind in ("official_government", "official_company") else "single_source_reporting"
    allegation = any(term in low for term in ALLEGATION_TERMS)
    review_required = allegation or any(term in low for term in REVIEW_TERMS)
    if allegation:
        evidence_status = "allegation"
    if source_kind in ("aggregator", "news_media") and not allegation:
        evidence_status = "single_source_reporting"

    taiwan_hits = sum(term in low for term in TAIWAN_TERMS)
    relevance = "high" if taiwan_hits >= 2 else "medium" if taiwan_hits == 1 else "low"
    if relevance == "low":
        relevance = "unknown"
    topics = [name for name, terms in EVENT_TERMS.items() if any(term in low for term in terms)]
    entity_matches = match_entities(item, watchlist)
    factual_identity = url.split("#", 1)[0].rstrip("/")
    event_id = _hash(factual_identity)
    cluster_id = _hash(_normalized_text(title))
    observed = iso_utc(now)
    return {
        "schema_version": "event-record-v1",
        "event_id": event_id,
        "cluster_id": cluster_id,
        "version": 1,
        "headline": title[:1000],
        "summary_factual": None,
        "raw_content_hash": _hash(title + " " + str(item.get("description") or "")),
        "source": {
            "name": source_name[:160],
            "domain": domain[:253],
            "url": url,
            "source_class": source_kind,
            "source_item_id": str(item.get("id")) if item.get("id") is not None else None,
            "language": None,
        },
        "timestamps": {
            "event_time_raw": None,
            "published_at_raw": raw_published or None,
            "published_timezone": published_timezone,
            "published_at_utc": published_utc,
            "first_seen_at_utc": observed,
            "last_seen_at_utc": observed,
            "forecast_eligible_at_utc": observed,
            "timestamp_quality": timestamp_quality,
        },
        "entities": entity_matches,
        "event_type": classify_event(combined),
        "evidence": {
            "status": evidence_status,
            "attribution": source_name,
            "supporting_source_urls": [url] if source_kind in ("official_government", "official_company") else [],
            "contradicting_source_urls": [],
            "review_required": review_required,
            "review_status": "needs_followup" if review_required else "not_reviewed",
        },
        "classification": {
            "topics": topics,
            "novelty": 0.5,
            "duplicate_cluster_confidence": 0.5,
            "is_material_update": False,
            "supersedes_event_version": None,
        },
        "taiwan_impact": {
            "relevance": relevance,
            "transmission_channels": [],
            "affected_sectors": [],
            "direction_scenario": "unknown",
            "horizon_sessions": None,
            "volatility_potential": "unknown",
            "confidence": 0.25,
            "rationale": "規則式初篩，尚未經 AI 語意判讀或人工確認。",
        },
        "extraction": {
            "method": "rules",
            "model_name": None,
            "model_version": "event-monitor-rules-v1",
            "prompt_version": None,
            "extracted_at_utc": observed,
        },
    }


def merge_history(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve prior versions when the same source URL materially changes."""
    by_version = {
        (row["event_id"], int(row.get("version", 1))): row
        for row in existing if isinstance(row, dict) and row.get("event_id")
    }
    latest_by_id: dict[str, dict[str, Any]] = {}
    for row in by_version.values():
        event_id = row["event_id"]
        if event_id not in latest_by_id or int(row.get("version", 1)) > int(latest_by_id[event_id].get("version", 1)):
            latest_by_id[event_id] = row

    for event in incoming:
        previous = latest_by_id.get(event["event_id"])
        if previous:
            old_version = int(previous.get("version", 1))
            old_hash = previous.get("raw_content_hash")
            new_hash = event.get("raw_content_hash")
            old_times = previous.get("timestamps", {})
            event["timestamps"]["first_seen_at_utc"] = old_times.get(
                "first_seen_at_utc", event["timestamps"]["first_seen_at_utc"]
            )
            event["timestamps"]["forecast_eligible_at_utc"] = old_times.get(
                "forecast_eligible_at_utc", event["timestamps"]["first_seen_at_utc"]
            )
            if old_hash and new_hash and old_hash != new_hash:
                event["version"] = old_version + 1
                event["classification"]["is_material_update"] = True
                event["classification"]["supersedes_event_version"] = old_version
                by_version[(event["event_id"], event["version"])] = event
            else:
                event["version"] = old_version
                event["timestamps"]["last_seen_at_utc"] = event["timestamps"]["last_seen_at_utc"]
                by_version[(event["event_id"], old_version)] = event
        else:
            event["version"] = 1
            by_version[(event["event_id"], 1)] = event
        latest_by_id[event["event_id"]] = event

    return sorted(
        by_version.values(),
        key=lambda row: (
            row.get("timestamps", {}).get("last_seen_at_utc", ""),
            row.get("event_id", ""),
            int(row.get("version", 1)),
        ),
        reverse=True,
    )

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            except json.JSONDecodeError:
                continue
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def run(output: Path = DEFAULT_OUTPUT, watchlist_path: Path = DEFAULT_WATCHLIST,
        max_items: int = 300, now: dt.datetime | None = None) -> dict[str, Any]:
    with watchlist_path.open(encoding="utf-8") as stream:
        watchlist = json.load(stream)
    observed = now or utc_now()
    if observed.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    items = news.headlines()
    candidates = []
    skipped = 0
    for item in items[:max_items]:
        event = build_event(item, watchlist, observed)
        if event is None:
            skipped += 1
            continue
        candidates.append(event)
    merged = merge_history(load_jsonl(output), candidates)
    atomic_write_jsonl(output, merged)
    return {
        "status": "research_only",
        "source_items_seen": len(items),
        "events_written": len(candidates),
        "items_skipped_missing_or_invalid_url": skipped,
        "history_rows": len(merged),
        "output": str(output),
        "note": "規則式候選事件；尚未經 AI/人工查證，不可作為預測特徵。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--max-items", type=int, default=300)
    args = parser.parse_args()
    if args.max_items < 1:
        parser.error("--max-items must be positive")
    print(json.dumps(run(args.output, args.watchlist, args.max_items), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
