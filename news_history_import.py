"""Backfill a separate GDELT-based Taiwan news research history.

This is a descriptive research dataset, not a trading signal. GDELT DOC 2.0
has a rolling ~3-month article-search window. We align each trading-day
snapshot to an 08:30 Asia/Taipei cutoff and keep GDELT separate from live RSS
snapshots to avoid mixing measurement methods.
"""
import argparse
import datetime as dt
import json
import os
import re
import time
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

API = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = "Taiwan OR TSMC OR 台灣 OR 台積電 OR TAIEX OR 加權指數"
TZ = ZoneInfo("Asia/Taipei")
CUTOFF = dt.time(8, 30)
MAX_RECORDS = 250
POS_ZH = ("上漲", "成長", "增加", "改善", "利多", "創高", "突破", "復甦", "降息", "回升", "獲利")
NEG_ZH = ("下跌", "衰退", "減少", "惡化", "利空", "暴跌", "制裁", "戰爭", "升息", "疲弱", "虧損")
POS_EN = ("surge", "rally", "gain", "rise", "beat", "boost", "recover", "bullish", "upgrade")
NEG_EN = ("fall", "drop", "slump", "plunge", "loss", "miss", "weak", "tariff", "sanction", "bearish", "crash")
TOPICS = {
    "semiconductor": ("半導體", "晶片", "晶圓", "台積電", "tsmc", "chip", "semiconductor", "ai"),
    "macro": ("利率", "通膨", "通胀", "央行", "聯準會", "fed", "inflation", "rate", "gdp", "匯率"),
    "trade": ("關稅", "貿易", "出口", "進口", "tariff", "trade", "export", "supply chain"),
    "geopolitics": ("台海", "地緣", "制裁", "戰爭", "中國", "taiwan strait", "sanction", "war", "china"),
    "domestic_policy": ("行政院", "經濟部", "政策", "法案", "預算", "政府", "policy", "government"),
}


def atomic_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_seen(value):
    value = str(value or "").strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
        try:
            return dt.datetime.strptime(value, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            pass
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(dt.timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def fetch_window(start_local, end_local):
    start_utc = start_local.astimezone(dt.timezone.utc)
    end_utc = end_local.astimezone(dt.timezone.utc)
    params = {
        "query": QUERY, "mode": "ArtList", "format": "json",
        "maxrecords": str(MAX_RECORDS), "sort": "DateAsc",
        "startdatetime": start_utc.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_utc.strftime("%Y%m%d%H%M%S"),
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "TAIEX-News-Research/1.0 (historical metadata analysis)",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    articles = payload.get("articles") or []
    normalized = []
    seen = set()
    for article in articles:
        title = re.sub(r"\s+", " ", str(article.get("title") or "")).strip()
        link = str(article.get("url") or "").strip()
        seen_at = parse_seen(article.get("seendate"))
        if not title or not seen_at:
            continue
        if not (start_utc < seen_at <= end_utc):
            continue
        key = link or title.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            "title": title[:500],
            "source": str(article.get("domain") or article.get("sourcecountry") or "GDELT")[:160],
            "language": str(article.get("language") or "")[:20],
            "source_country": str(article.get("sourcecountry") or "")[:80],
            "published": seen_at.isoformat(),
            "link": link[:1000],
        })
    normalized.sort(key=lambda x: x["published"])
    return normalized, len(articles) >= MAX_RECORDS


def score_articles(articles):
    pos = neg = 0
    categories = {key: 0 for key in TOPICS}
    for article in articles:
        title = article["title"]
        low = title.lower()
        p = sum(word in title for word in POS_ZH) + sum(word in low for word in POS_EN)
        n = sum(word in title for word in NEG_ZH) + sum(word in low for word in NEG_EN)
        pos += int(p > n)
        neg += int(n > p)
        for key, words in TOPICS.items():
            if any(word in low for word in words):
                categories[key] += 1
    count = len(articles)
    sources = {a.get("source") for a in articles if a.get("source")}
    return {
        "headline_count": count,
        "category_counts": categories,
        "unique_source_count": len(sources),
        "published_time_coverage_pct": 100.0 if count else 0.0,
        "original_link_coverage_pct": round(100 * sum(bool(a.get("link")) for a in articles) / count, 1) if count else 0.0,
        "positive_headline_count": pos,
        "negative_headline_count": neg,
        "news_environment_score": round((pos - neg) / max(1, count), 4),
        "data_source": "GDELT DOC 2.0",
        "score_method": "標題字詞分類；不理解語意",
    }


def build_context(rows, latest):
    context = {
        "status": "insufficient", "sample_count": 0,
        "note": "GDELT 相似新聞環境的已標記歷史樣本不足 20 筆，暫不顯示統計。",
    }
    current = (latest.get("features") or {}).get("news_environment_score")
    if current is None:
        return context
    peers = []
    for row in rows:
        if row.get("asof") == latest.get("asof"):
            continue
        score = (row.get("features") or {}).get("news_environment_score")
        outcome = row.get("outcome") or {}
        value = outcome.get("next_session_return_pct")
        try:
            if score is not None and value is not None and abs(float(score) - float(current)) <= 0.20:
                peers.append(float(value))
        except (TypeError, ValueError):
            continue
    context["sample_count"] = len(peers)
    if len(peers) >= 20:
        arr = np.asarray(peers, dtype=float)
        context = {
            "status": "available", "sample_count": int(len(arr)),
            "positive_return_pct": round(float(np.mean(arr > 0) * 100), 1),
            "negative_return_pct": round(float(np.mean(arr < 0) * 100), 1),
            "average_return_pct": round(float(np.mean(arr)), 3),
            "median_return_pct": round(float(np.median(arr)), 3),
            "p10_return_pct": round(float(np.percentile(arr, 10)), 3),
            "p90_return_pct": round(float(np.percentile(arr, 90)), 3),
            "score_tolerance": 0.20,
            "note": "僅為 GDELT 相似新聞環境下的歷史描述統計，不是未來機率或買賣訊號。",
        }
    return context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90, choices=(30, 60, 90))
    args = parser.parse_args()
    if not os.path.isfile("market_history.csv"):
        raise SystemExit("缺少 market_history.csv；請先還原台股歷史行情。")
    market = pd.read_csv("market_history.csv", index_col=0, parse_dates=True)
    if "TAIEX" not in market:
        raise SystemExit("market_history.csv 缺少 TAIEX 欄位。")
    closes = market["TAIEX"].dropna()
    dates = sorted({pd.Timestamp(x).date() for x in closes.index})
    if len(dates) < 2:
        raise SystemExit("加權指數歷史資料不足。")

    cutoff_date = dt.datetime.now(TZ).date() - dt.timedelta(days=args.days)
    selected = [d for d in dates if d >= cutoff_date]
    if not selected:
        raise SystemExit("指定期間沒有可用交易日。")
    first_idx = dates.index(selected[0])
    if first_idx == 0:
        raise SystemExit("缺少第一個回補日期之前的交易日，無法建立對齊視窗。")
    sessions = dates[first_idx - 1:]
    close_by_date = {pd.Timestamp(k).date(): float(v) for k, v in closes.items()}

    rows_by_date = {}
    if os.path.isfile("news_gdelt_history.jsonl"):
        with open("news_gdelt_history.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("asof"):
                        rows_by_date[row["asof"]] = row
                except Exception:
                    continue

    capped_dates = []
    fetched = 0
    for idx in range(1, len(sessions)):
        asof = sessions[idx]
        if asof.isoformat() in rows_by_date:
            continue
        previous = sessions[idx - 1]
        start_local = dt.datetime.combine(previous, CUTOFF, TZ)
        end_local = dt.datetime.combine(asof, CUTOFF, TZ)
        try:
            articles, capped = fetch_window(start_local, end_local)
        except Exception as exc:
            print(f"GDELT {asof} 擷取失敗：{type(exc).__name__}: {str(exc)[:160]}", flush=True)
            time.sleep(5.2)
            continue
        if capped:
            capped_dates.append(asof.isoformat())
        features = score_articles(articles)
        current_close = close_by_date.get(asof)
        next_close = close_by_date.get(sessions[idx + 1]) if idx + 1 < len(sessions) else None
        outcome = None
        if current_close and next_close and current_close != 0:
            outcome = {
                "next_session_date": sessions[idx + 1].isoformat(),
                "next_session_return_pct": round((next_close / current_close - 1) * 100, 4),
            }
        rows_by_date[asof.isoformat()] = {
            "asof": asof.isoformat(),
            "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": "GDELT DOC 2.0",
            "cutoff_local": end_local.isoformat(),
            "window_start_local": start_local.isoformat(),
            "features": features,
            "articles": articles,
            "outcome": outcome,
            "query_capped": bool(capped),
        }
        fetched += 1
        print(f"GDELT {asof}: {len(articles)} 篇標題；{'達到單次上限' if capped else '未達單次上限'}", flush=True)
        time.sleep(5.2)

    rows = sorted(rows_by_date.values(), key=lambda x: x.get("asof", ""))[-1500:]
    with open("news_gdelt_history.jsonl.tmp", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace("news_gdelt_history.jsonl.tmp", "news_gdelt_history.jsonl")

    latest = rows[-1] if rows else None
    labeled = sum(isinstance(row.get("outcome"), dict) for row in rows)
    context = build_context(rows, latest) if latest else {"status": "insufficient", "sample_count": 0}
    summary = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "GDELT DOC 2.0",
        "attribution": "Data source: GDELT Project, https://www.gdeltproject.org/",
        "coverage_note": "DOC 2.0 僅提供約三個月的滾動新聞搜尋範圍；非台灣全部媒體的完整新聞庫。",
        "history_count": len(rows),
        "labeled_count": labeled,
        "fetched_this_run": fetched,
        "query_capped_dates": capped_dates,
        "latest": latest,
        "historical_context": context,
        "note": "新聞分數是標題字詞研究特徵；歷史報酬僅為事後描述統計，不代表未來結果。",
    }
    atomic_json("news_gdelt_research.json", summary)
    print(
        f"GDELT 歷史研究完成：新增 {fetched} 個交易日；總快照 {len(rows)}；"
        f"已標記隔日結果 {labeled}；達上限日期 {len(capped_dates)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
