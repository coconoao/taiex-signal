"""Collect an independent Media Cloud news-index snapshot.

Stores only story metadata (title, source, date, URL, language). Media Cloud
does not provide story full text to ordinary API users. This is a descriptive
news-coverage dataset, not a market forecast or trading signal.
"""
import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

API = "https://search.mediacloud.org/api/search/story-list"
TZ = ZoneInfo("Asia/Taipei")
QUERY = 'Taiwan OR 台灣 OR 台積電 OR TSMC OR TAIEX OR 加權指數'
PAGE_SIZE = 100
HISTORY_FILE = "news_mediacloud_history.jsonl"
SUMMARY_FILE = "news_mediacloud_research.json"
TOPICS = {
    "semiconductor": ("半導體", "晶片", "晶圓", "台積電", "tsmc", "chip", "semiconductor"),
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


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
            return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def load_history():
    rows = {}
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("asof"):
                        rows[row["asof"]] = row
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def fetch_stories(api_key, start_date, end_date):
    params = {
        "q": QUERY,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "platform": "onlinenews-mediacloud",
        "page_size": str(PAGE_SIZE),
        "sort_order": "desc",
    }
    req = urllib.request.Request(
        API + "?" + urllib.parse.urlencode(params),
        headers={
            "Authorization": "Token " + api_key,
            "Accept": "application/json",
            "User-Agent": "TAIEX-News-Research/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    stories = payload.get("stories") or []
    normalized, seen = [], set()
    for story in stories:
        title = re.sub(r"\s+", " ", str(story.get("title") or "")).strip()
        url = str(story.get("url") or "").strip()
        if not title:
            continue
        key = url or title.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            "title": title[:500],
            "source": str(story.get("media_name") or "")[:160],
            "source_url": str(story.get("media_url") or "")[:500],
            "language": str(story.get("language") or "")[:20],
            "published_date": str(story.get("publish_date") or "")[:30],
            "indexed_date": str(story.get("indexed_date") or "")[:50],
            "url": url[:1000],
        })
    return normalized, bool(payload.get("pagination_token"))


def features(stories):
    languages, topics = {}, {name: 0 for name in TOPICS}
    sources = set()
    for story in stories:
        language = story.get("language") or "unknown"
        languages[language] = languages.get(language, 0) + 1
        if story.get("source"):
            sources.add(story["source"])
        text = (story.get("title") or "").lower()
        for name, words in TOPICS.items():
            if any(word in text for word in words):
                topics[name] += 1
    return {
        "headline_count": len(stories),
        "unique_source_count": len(sources),
        "language_counts": languages,
        "topic_counts": topics,
        "original_link_coverage_pct": round(
            100 * sum(bool(x.get("url")) for x in stories) / len(stories), 1
        ) if stories else 0.0,
        "score_method": "只統計標題主題與來源；不計算多空分數、不產生市場方向。",
    }


def main():
    now = dt.datetime.now(TZ)
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    previous = load_json(SUMMARY_FILE)
    api_key = os.environ.get("MEDIACLOUD_API_KEY", "").strip()
    history = load_history()

    if not api_key:
        summary = {
            **previous,
            "status": "needs_api_key",
            "updated_at": now_utc,
            "current_error": None,
            "note": "尚未設定 MEDIACLOUD_API_KEY。加入 GitHub Actions secret 後才會向 Media Cloud 擷取資料；既有快照會保留。",
            "history_count": len(history),
            "source": "Media Cloud Online News Archive",
        }
        atomic_json(SUMMARY_FILE, summary)
        print("Media Cloud skipped: MEDIACLOUD_API_KEY is not configured.", flush=True)
        return

    start_date = now.date() - dt.timedelta(days=3)
    end_date = now.date()
    attempted_at = now_utc
    try:
        stories, has_more = fetch_stories(api_key, start_date, end_date)
        snapshot = {
            "asof": now.date().isoformat(),
            "captured_at": now_utc,
            "window_start": start_date.isoformat(),
            "window_end": end_date.isoformat(),
            "features": features(stories),
            "stories": stories,
            "has_more_results": has_more,
        }
        history[snapshot["asof"]] = snapshot
        rows = sorted(history.values(), key=lambda x: x.get("asof", ""))[-1500:]
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, HISTORY_FILE)
        summary = {
            "status": "ok" if stories else "empty",
            "updated_at": now_utc,
            "last_successful_at": now_utc,
            "last_attempted_at": attempted_at,
            "source": "Media Cloud Online News Archive",
            "source_url": "https://www.mediacloud.org/",
            "query": QUERY,
            "current": snapshot,
            "history_count": len(rows),
            "current_error": None,
            "has_more_results": has_more,
            "coverage_note": "只保存 Media Cloud 提供的標題與中繼資料，不保存新聞全文。來源收錄起始日不一，並非完整台灣新聞庫；每次最多讀取一頁，若有下一頁會明確標示。",
            "note": "主題計數為標題關鍵字分類，不代表語意理解、因果關係、未來市場結果或買賣訊號。",
        }
        atomic_json(SUMMARY_FILE, summary)
        print(
            f"Media Cloud: {len(stories)} metadata records; "
            f"{snapshot['features']['unique_source_count']} sources; "
            f"more_pages={has_more}; history_days={len(rows)}",
            flush=True,
        )
    except Exception as exc:
        summary = {
            **previous,
            "status": "error" if not previous.get("current") else "stale",
            "updated_at": now_utc,
            "last_attempted_at": attempted_at,
            "current_error": f"{type(exc).__name__}: {str(exc)[:180]}",
            "history_count": len(history),
            "source": "Media Cloud Online News Archive",
            "note": "本次擷取失敗，保留最近一次成功資料；請檢查 API Key、配額與服務狀態。",
        }
        atomic_json(SUMMARY_FILE, summary)
        print("Media Cloud update failed; previous data preserved: " + summary["current_error"], flush=True)


if __name__ == "__main__":
    main()
