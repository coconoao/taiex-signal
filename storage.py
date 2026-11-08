"""Supabase-backed persistence for generated market/news data.

The Render free filesystem is ephemeral. This module mirrors only public market
and news artifacts; authentication secrets and user credentials are never stored.
"""
import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request

FILES = (
    "signals.json", "weekly.json", "weekly_prob.csv", "market_history.csv",
    "chips.csv", "margin.csv", "futures.csv", "taiwan_data_status.json",
    "news.json", "news_history.jsonl", "policy_events.jsonl",
    "news_research.json", "news_research_history.jsonl",
    "news_gdelt_history.jsonl", "news_gdelt_research.json",
    "news_policy_archive.json", "news_policy_research.json",
    "news_mediacloud_history.jsonl", "news_mediacloud_research.json",
    "news_enhanced_model.json", "multi_market_research.json", "model_improvement_research.json", "strategy_research.json", "multi_horizon_model_research.json",
)
# These two artifacts are maintained by dedicated research workflows. Exclude
# them from routine Render saves so stale process-local copies cannot overwrite
# freshly computed research data in Supabase.
MANAGED_RESEARCH_FILES = ("news_policy_archive.json", "news_policy_research.json", "news_gdelt_history.jsonl", "news_gdelt_research.json", "news_mediacloud_history.jsonl", "news_mediacloud_research.json", "news_enhanced_model.json", "multi_market_research.json", "model_improvement_research.json", "strategy_research.json", "multi_horizon_model_research.json")
AUTO_SAVE_FILES = tuple(name for name in FILES if name not in MANAGED_RESEARCH_FILES)
POLICY_ARCHIVE_FILE = "news_policy_archive.json"
POLICY_RESEARCH_FILE = "news_policy_research.json"
GDELT_RESEARCH_FILES = ("news_gdelt_history.jsonl", "news_gdelt_research.json")
MEDIACLOUD_RESEARCH_FILES = ("news_mediacloud_history.jsonl", "news_mediacloud_research.json")
NEWS_ENHANCED_MODEL_FILE = "news_enhanced_model.json"
MULTI_MARKET_RESEARCH_FILE = "multi_market_research.json"
MODEL_IMPROVEMENT_RESEARCH_FILE = "model_improvement_research.json"
STRATEGY_RESEARCH_FILE = "strategy_research.json"
MULTI_HORIZON_MODEL_RESEARCH_FILE = "multi_horizon_model_research.json"
TABLE = "site_data"


def enabled():
    return (
        os.environ.get("TAIEX_STORAGE", "").strip().lower() == "supabase"
        and bool(os.environ.get("SUPABASE_URL"))
        and bool(os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
    )


def _request(method, path, payload=None, prefer=None):
    base = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not base or not key:
        raise RuntimeError("Supabase storage credentials are not configured")
    headers = {"apikey": key, "Accept": "application/json"}
    if not key.startswith("sb_secret_"):
        headers["Authorization"] = "Bearer " + key
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    else:
        body = None
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(
        base + "/rest/v1/" + path, data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else []
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            "Supabase data storage request failed (HTTP %s)" % exc.code
        ) from None


def restore_data(directory="."):
    """Restore known data files from Supabase; return number of files restored."""
    if not enabled():
        return 0
    rows = _request("GET", TABLE + "?select=key,content")
    allowed = set(FILES)
    restored = 0
    os.makedirs(directory, exist_ok=True)
    for row in rows:
        name = row.get("key")
        content = row.get("content")
        if name not in allowed or not isinstance(content, str):
            continue
        target = os.path.join(directory, name)
        tmp = target + ".restore.tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
        restored += 1
    return restored


def save_data(directory=".", names=None):
    """Upsert selected data files; routine saves omit dedicated research artifacts."""
    if not enabled():
        return 0
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = []
    for name in (AUTO_SAVE_FILES if names is None else tuple(names)):
        if name not in FILES:
            raise ValueError("Unknown data file: " + str(name))
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as stream:
                rows.append({"key": name, "content": stream.read(), "updated_at": now})
    if rows:
        _request(
            "POST",
            TABLE + "?on_conflict=key",
            rows,
            "resolution=merge-duplicates,return=minimal",
        )
    return len(rows)


if __name__ == "__main__":
    import sys
    if not enabled():
        raise SystemExit("Set TAIEX_STORAGE=supabase, SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "restore":
        print("Restored data files:", restore_data())
    elif action == "save":
        print("Saved data files:", save_data())
    elif action == "save-policy-archive":
        print("Saved policy archive files:", save_data(names=(POLICY_ARCHIVE_FILE,)))
    elif action == "save-policy-research":
        print("Saved policy research files:", save_data(names=(POLICY_RESEARCH_FILE,)))
    elif action == "save-gdelt-research":
        print("Saved GDELT research files:", save_data(names=GDELT_RESEARCH_FILES))
    elif action == "save-mediacloud-research":
        print("Saved Media Cloud research files:", save_data(names=MEDIACLOUD_RESEARCH_FILES))
    elif action == "save-news-enhanced-model":
        print("Saved news-enhanced model:", save_data(names=(NEWS_ENHANCED_MODEL_FILE,)))
    elif action == "save-multi-market-research":
        print("Saved multi-market research:", save_data(names=(MULTI_MARKET_RESEARCH_FILE,)))
    elif action == "save-model-improvement-research|save-strategy-research|save-multi-horizon-model-research":
        print("Saved model improvement research:", save_data(names=(MODEL_IMPROVEMENT_RESEARCH_FILE,)))
    elif action == "save-strategy-research":
        print("Saved strategy research:", save_data(names=(STRATEGY_RESEARCH_FILE,)))
    elif action == "save-multi-horizon-model-research":
        print("Saved multi-horizon model research:", save_data(names=(MULTI_HORIZON_MODEL_RESEARCH_FILE,)))
    else:
        raise SystemExit(
            "Usage: python storage.py restore|save|save-policy-archive|save-policy-research|save-gdelt-research|save-mediacloud-research|save-news-enhanced-model|save-multi-market-research|save-model-improvement-research"
        )
