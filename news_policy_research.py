"""Build a descriptive alignment study of sampled Executive Yuan headlines and TAIEX closes.

This is not an ML training or forecasting job. It pairs publication dates with the
following observed TAIEX session close-to-close return, without inferring causality.
"""
import csv
import datetime as dt
import json
import math
import os
import statistics

ARCHIVE = "news_policy_archive.json"
MARKET = "market_history.csv"
OUTPUT = "news_policy_research.json"


def read_market():
    with open(MARKET, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        date_key = fields[0] if fields else None
        if not date_key or "TAIEX" not in fields:
            raise ValueError("market_history.csv 缺少日期或 TAIEX 欄位")
        values = {}
        for row in reader:
            day = str(row.get(date_key) or "")[:10]
            try:
                close = float(str(row.get("TAIEX") or "").replace(",", ""))
            except ValueError:
                continue
            if day and math.isfinite(close) and close > 0:
                values[day] = close
    return sorted(values.items())


def stats(rows):
    values = [x["next_session_return_pct"] for x in rows]
    if not values:
        return {"sample_count": 0}
    return {
        "sample_count": len(values),
        "mean_return_pct": round(statistics.mean(values), 4),
        "median_return_pct": round(statistics.median(values), 4),
        "positive_days_pct": round(100 * sum(x > 0 for x in values) / len(values), 1),
        "p10_return_pct": round(sorted(values)[max(0, math.ceil(.1 * len(values)) - 1)], 4),
        "p90_return_pct": round(sorted(values)[max(0, math.ceil(.9 * len(values)) - 1)], 4),
    }


def main():
    if not os.path.isfile(ARCHIVE) or not os.path.isfile(MARKET):
        raise SystemExit("缺少官方新聞索引或 market_history.csv；未產生研究結果。")
    with open(ARCHIVE, encoding="utf-8") as stream:
        archive = json.load(stream)
    closes = read_market()
    if len(closes) < 2:
        raise SystemExit("台股收盤資料不足，未產生研究結果。")
    close_map = dict(closes)
    next_close = {closes[i][0]: closes[i + 1] for i in range(len(closes) - 1)}

    daily = {}
    for item in archive.get("items", []):
        day = str(item.get("date") or "")[:10]
        try:
            dt.date.fromisoformat(day)
        except ValueError:
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        daily.setdefault(day, []).append(title)

    rows = []
    for day, titles in sorted(daily.items()):
        following = next_close.get(day)
        start = close_map.get(day)
        if start is None or following is None:
            continue
        next_day, end = following
        rows.append({
            "date": day,
            "headline_count": len(titles),
            "next_session_date": next_day,
            "next_session_return_pct": round((end / start - 1) * 100, 4),
        })

    if not rows:
        raise SystemExit("沒有可與台股收盤日配對的新聞日期；保留既有研究資料。")

    split = max(1, min(len(rows) - 1, int(len(rows) * 0.7))) if len(rows) > 1 else len(rows)
    train, holdout = rows[:split], rows[split:]
    result = {
        "status": "descriptive_only",
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": archive.get("dataset", "行政院官方新聞索引（抽樣）"),
        "archive_record_count": len(archive.get("items", [])),
        "archive_date_count": archive.get("date_count", len(daily)),
        "archive_date_start": archive.get("date_start"),
        "archive_date_end": archive.get("date_end"),
        "market_date_start": closes[0][0],
        "market_date_end": closes[-1][0],
        "matched_date_count": len(rows),
        "unmatched_news_date_count": max(0, len(daily) - len(rows)),
        "matched_headline_count": sum(x["headline_count"] for x in rows),
        "chronological_split": {
            "method": "依日期排序，前 70% 僅作歷史描述，後 30% 保留作時間外觀察",
            "historical_period": {
                "date_start": train[0]["date"], "date_end": train[-1]["date"],
                **stats(train),
            },
            "later_period": ({
                "date_start": holdout[0]["date"], "date_end": holdout[-1]["date"],
                **stats(holdout),
            } if holdout else {"sample_count": 0}),
        },
        "all_matched": stats(rows),
        "latest_matches": rows[-12:][::-1],
        "method_note": "以官方索引發布日期聚合標題，配對該日加權指數收盤至下一交易日收盤的已觀察變化。未取得發布時刻，故僅為日期層級描述性對照；抽樣官方政策標題不代表完整新聞環境，不代表因果關係、預測能力或未來機率。",
        "model_training": False,
        "forecasting": False,
    }
    tmp = OUTPUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, OUTPUT)
    print(
        f"官方政策新聞歷史對照完成：標題 {result['archive_record_count']}；"
        f"可配對發布日 {len(rows)}；歷史期 {len(train)}；後段觀察期 {len(holdout)}。"
        "此流程為描述性研究，沒有訓練預測模型。",
        flush=True,
    )


if __name__ == "__main__":
    main()
