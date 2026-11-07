"""Backfill a bounded archive of official Executive Yuan news metadata.

Only titles, publication dates and source URLs are retained. Article bodies are
not redistributed. This archive is kept separate from multi-source daily
sentiment snapshots because its source scope is government releases only.
"""
import datetime as dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://opendata.ey.gov.tw/api/ExecutiveYuan/NewsEy"
START = dt.date(2026, 1, 28)
END = dt.date(2026, 11, 5)
MAX_SIZE = 1000
USER_AGENT = "TAIEXSignal/1.0 (official public-data archive)"


def fetch_range(start, end):
    params = urllib.parse.urlencode({
        "StartDate": start.strftime("%Y/%m/%d"),
        "EndDate": end.strftime("%Y/%m/%d"),
        "MaxSize": MAX_SIZE,
        "IsRemoveHtmlTag": "true",
    })
    request = urllib.request.Request(
        API + "?" + params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8-sig"))
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                for key in ("data", "Data", "items", "Items", "result", "Result"):
                    if isinstance(payload.get(key), list):
                        return payload[key]
            raise RuntimeError("Unexpected Executive Yuan API response shape")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def parse_day(value):
    raw = str(value or "").strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(raw, fmt).date()
        except ValueError:
            pass
    return None


def normalize(row):
    if not isinstance(row, dict):
        return None
    title = str(row.get("標題") or row.get("title") or "").strip()
    day = parse_day(row.get("上版日期") or row.get("published") or row.get("date"))
    link = str(row.get("來源網址") or row.get("url") or "").strip()
    if not title or not day:
        return None
    return {
        "date": day.isoformat(),
        "title": title[:500],
        "source": "行政院",
        "url": link[:1000],
    }


def main():
    items, seen = [], set()
    cursor = START
    while cursor <= END:
        chunk_end = min(cursor + dt.timedelta(days=13), END)
        rows = fetch_range(cursor, chunk_end)
        # Split dense ranges to avoid silently accepting the API's 1,000-row cap.
        if len(rows) >= MAX_SIZE and cursor < chunk_end:
            midpoint = cursor + (chunk_end - cursor) // 2
            ranges = ((cursor, midpoint), (midpoint + dt.timedelta(days=1), chunk_end))
        else:
            ranges = ((cursor, chunk_end),)
        if len(ranges) == 1:
            for raw in rows:
                item = normalize(raw)
                if not item:
                    continue
                key = item["url"] or (item["date"] + "|" + item["title"])
                if key not in seen:
                    seen.add(key)
                    items.append(item)
        else:
            for left, right in ranges:
                for raw in fetch_range(left, right):
                    item = normalize(raw)
                    if not item:
                        continue
                    key = item["url"] or (item["date"] + "|" + item["title"])
                    if key not in seen:
                        seen.add(key)
                        items.append(item)
                time.sleep(0.15)
        cursor = chunk_end + dt.timedelta(days=1)
        time.sleep(0.15)

    items.sort(key=lambda x: (x["date"], x["title"]))
    if not items:
        raise SystemExit("No historical records returned; existing archive was not overwritten.")
    daily = {}
    for item in items:
        daily[item["date"]] = daily.get(item["date"], 0) + 1
    archive = {
        "dataset": "行政院即時新聞",
        "source": "行政院全球資訊網 Open API",
        "source_url": "https://opendata.ey.gov.tw/api/index.html",
        "license": "政府資料開放授權條款",
        "scope": "官方政府新聞稿標題與來源連結；非完整媒體新聞資料庫",
        "date_start": items[0]["date"],
        "date_end": items[-1]["date"],
        "record_count": len(items),
        "date_count": len(daily),
        "daily_counts": daily,
        "imported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "items": items,
    }
    tmp = "news_policy_archive.json.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(archive, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
    import os
    os.replace(tmp, "news_policy_archive.json")
    print(
        "Official historical news archive:",
        len(items), "records across", len(daily), "publication dates;",
        items[0]["date"], "to", items[-1]["date"],
        flush=True,
    )


if __name__ == "__main__":
    main()
