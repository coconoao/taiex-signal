"""Label existing news snapshots only after the next TAIEX session has closed."""
import csv
import datetime as dt
import json
import math
import os

HISTORY = "news_research_history.jsonl"
SUMMARY = "news_research.json"
MARKET = "market_history.csv"


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return round(ordered[lo], 3)
    return round(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo), 3)


def load_closes():
    if not os.path.isfile(MARKET):
        raise SystemExit("缺少 market_history.csv；本次只保留既有新聞快照。")
    with open(MARKET, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        date_key = fields[0] if fields else None
        if not date_key or "TAIEX" not in fields:
            raise SystemExit("market_history.csv 缺少日期或 TAIEX 欄位。")
        closes = {}
        for row in reader:
            day = str(row.get(date_key) or "").strip()[:10]
            raw = str(row.get("TAIEX") or "").replace(",", "").strip()
            try:
                value = float(raw)
                if day and math.isfinite(value) and value > 0:
                    closes[day] = value
            except (TypeError, ValueError):
                continue
    return sorted(closes.items())


def atomic_write(path, content):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def main():
    if not os.path.isfile(HISTORY):
        print("尚無新聞歷史快照，略過結果標記。", flush=True)
        return
    closes = load_closes()
    if len(closes) < 2:
        print("加權指數收盤資料不足，略過結果標記。", flush=True)
        return
    date_to_index = {day: i for i, (day, _) in enumerate(closes)}
    with open(HISTORY, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    changed = 0
    for row in rows:
        if row.get("outcome") is not None:
            continue
        idx = date_to_index.get(str(row.get("asof", "")))
        if idx is None or idx + 1 >= len(closes):
            continue
        start_day, start_close = closes[idx]
        next_day, next_close = closes[idx + 1]
        row["outcome"] = {
            "next_session_date": next_day,
            "next_session_return_pct": round((next_close / start_close - 1) * 100, 4),
        }
        changed += 1

    if changed:
        atomic_write(HISTORY, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    latest = rows[-1] if rows else None
    context = {
        "status": "insufficient",
        "sample_count": 0,
        "note": "相似新聞環境的已標記歷史樣本不足 20 筆，暫不顯示統計比例。",
    }
    current_score = ((latest or {}).get("features") or {}).get("news_environment_score")
    peers = []
    if current_score is not None:
        for row in rows[:-1]:
            score = (row.get("features") or {}).get("news_environment_score")
            outcome = row.get("outcome") or {}
            value = outcome.get("next_session_return_pct")
            try:
                if score is not None and value is not None and abs(float(score) - float(current_score)) <= 0.20:
                    peers.append(float(value))
            except (TypeError, ValueError):
                continue
    context["sample_count"] = len(peers)
    if len(peers) >= 20:
        context = {
            "status": "available",
            "sample_count": len(peers),
            "positive_return_pct": round(100 * sum(x > 0 for x in peers) / len(peers), 1),
            "negative_return_pct": round(100 * sum(x < 0 for x in peers) / len(peers), 1),
            "average_return_pct": round(sum(peers) / len(peers), 3),
            "median_return_pct": percentile(peers, 0.5),
            "p10_return_pct": percentile(peers, 0.1),
            "p90_return_pct": percentile(peers, 0.9),
            "score_tolerance": 0.20,
            "note": "僅為相似新聞環境下的歷史描述統計，不是未來機率或買賣訊號。",
        }

    summary = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "latest": latest,
        "history_count": len(rows),
        "labeled_count": sum(isinstance(row.get("outcome"), dict) for row in rows),
        "historical_context": context,
        "note": "新聞特徵僅作資料研究；歷史結果只在後續交易日收盤資料到齊後標記，不回填未來資訊。",
    }
    atomic_write(SUMMARY, json.dumps(summary, ensure_ascii=False))
    print(
        f"新聞研究維護完成：快照 {len(rows)} 個；本次新增隔日結果 {changed}；"
        f"已標記 {summary['labeled_count']}；相似樣本 {context['sample_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
