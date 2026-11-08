"""News-enhanced TAIEX research classifier.

Trains only on chronologically ordered, fully labelled snapshots that also
contain the contemporaneous original-model context. It never changes signals.json.
"""
import datetime as dt
import json
import math
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HISTORY = "news_research_history.jsonl"
SIGNALS = "signals.json"
OUTPUT = "news_enhanced_model.json"
MIN_SAMPLES = 100
HOLDOUT_MIN = 20


def read_json(path):
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_history():
    rows = []
    if not os.path.isfile(HISTORY):
        return rows
    with open(HISTORY, encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except ValueError:
                continue
    return sorted(rows, key=lambda row: str(row.get("asof", "")))


def write_history(rows):
    tmp = HISTORY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        for row in rows[-1500:]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, HISTORY)


def vector(features, market_context):
    categories = features.get("category_counts") or {}
    total = max(1, int(features.get("headline_count") or 0))
    try:
        original_score = float(market_context["prob_up"])
        if not math.isfinite(original_score) or not 0 <= original_score <= 1:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    positive = max(0, int(features.get("positive_headline_count") or 0))
    negative = max(0, int(features.get("negative_headline_count") or 0))
    missing = max(0, min(4, int(market_context.get("missing_layer_count") or 0)))
    values = [
        original_score,
        float(features.get("news_environment_score") or 0),
        min(total, 300) / 300.0,
        max(0, int(categories.get("market") or 0)) / total,
        max(0, int(categories.get("taiwan") or 0)) / total,
        max(0, int(categories.get("policy") or 0)) / total,
        max(0, int(categories.get("analyst") or 0)) / total,
        (positive - negative) / total,
        float(features.get("published_time_coverage_pct") or 0) / 100.0,
        missing / 4.0,
    ]
    result = np.asarray(values, dtype=float)
    return result if np.isfinite(result).all() else None


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, max_iter=1000, class_weight="balanced", random_state=17),
    )


def write_output(payload):
    payload["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    tmp = OUTPUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, OUTPUT)


def main():
    history = read_history()
    training_x, training_y, training_dates, original_scores = [], [], [], []
    for row in history:
        outcome = row.get("outcome") or {}
        market_context = row.get("market_context") or {}
        features = row.get("features") or {}
        x = vector(features, market_context)
        value = outcome.get("next_session_return_pct")
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if x is None or not math.isfinite(value):
            continue
        training_x.append(x)
        training_y.append(int(value > 0))
        training_dates.append(str(row.get("asof", "")))
        original_scores.append(float(market_context["prob_up"]))

    signals = read_json(SIGNALS)
    signal_asof = str(signals.get("asof") or "")
    current_context = {}
    try:
        probability = float(signals.get("prob_up"))
        if signal_asof and 0 <= probability <= 1:
            current_context = {
                "prob_up": probability,
                "missing_layer_count": len(signals.get("missing_layers") or []),
            }
    except (TypeError, ValueError):
        current_context = {}

    history_changed = False
    current_row = None
    for row in history:
        if str(row.get("asof") or "") == signal_asof:
            current_row = row
            if current_context and row.get("market_context") != current_context:
                row["market_context"] = current_context
                history_changed = True
            break
    if history_changed:
        write_history(history)

    current_features = (current_row or {}).get("features") or {}
    current_vector = vector(current_features, current_context) if current_context else None

    payload = {
        "status": "insufficient",
        "signal": "資料累積中",
        "score_pct": None,
        "asof": signal_asof or None,
        "history_count": len(history),
        "sample_count": len(training_x),
        "minimum_samples": MIN_SAMPLES,
        "training_samples": 0,
        "holdout_samples": 0,
        "metrics": None,
        "feature_summary": None,
        "note": (
            "新聞增強模型需要至少 100 筆同時具備原始模型當時分數、新聞特徵及後續實際收盤結果的交易日樣本；"
            "目前只累積符合條件的資料，不以未驗證規則產生訊號。"
        ),
    }

    if current_vector is None:
        payload["status"] = "current_data_unavailable"
        payload["signal"] = "目前資料不足"
        payload["note"] = "目前新聞或原始模型資料不足，無法產生當日研究分類。"
        write_output(payload)
        print("News-enhanced model: current inputs unavailable.", flush=True)
        return

    if len(training_x) < MIN_SAMPLES:
        payload["feature_summary"] = {
            "headline_count": current_features.get("headline_count", 0),
            "news_environment_score": current_features.get("news_environment_score", 0),
            "original_model_score_pct": round(float(signals["prob_up"]) * 100, 1),
        }
        write_output(payload)
        print(
            f"News-enhanced model collecting samples: {len(training_x)}/{MIN_SAMPLES}.",
            flush=True,
        )
        return

    X = np.asarray(training_x, dtype=float)
    y = np.asarray(training_y, dtype=int)
    holdout_n = max(HOLDOUT_MIN, int(math.ceil(len(y) * 0.2)))
    split = len(y) - holdout_n
    if split < 60 or len(np.unique(y[:split])) < 2 or len(np.unique(y[split:])) < 2:
        payload["status"] = "insufficient_class_coverage"
        payload["note"] = "時間外測試期或訓練期的正負樣本類別不足，暫不產生模型訊號。"
        write_output(payload)
        return

    enhanced_test = make_model().fit(X[:split], y[:split])
    enhanced_pred = enhanced_test.predict_proba(X[split:])[:, 1]
    baseline_pred = np.asarray(original_scores[split:], dtype=float)
    actual = y[split:]
    enhanced_brier = float(np.mean((enhanced_pred - actual) ** 2))
    baseline_brier = float(np.mean((baseline_pred - actual) ** 2))
    accuracy = float(np.mean((enhanced_pred >= 0.5) == actual))
    full_model = make_model().fit(X, y)
    score = float(full_model.predict_proba(current_vector.reshape(1, -1))[0, 1])
    signal = "偏正向" if score >= 0.55 else "偏負向" if score <= 0.45 else "中性"
    payload.update({
        "status": "ready",
        "signal": signal,
        "score_pct": round(score * 100, 1),
        "training_samples": int(split),
        "holdout_samples": int(holdout_n),
        "metrics": {
            "holdout_accuracy_pct": round(accuracy * 100, 1),
            "enhanced_brier": round(enhanced_brier, 4),
            "original_model_brier": round(baseline_brier, 4),
            "brier_improved_vs_original": enhanced_brier < baseline_brier,
            "holdout_start": training_dates[split],
            "holdout_end": training_dates[-1],
        },
        "feature_summary": {
            "headline_count": current_features.get("headline_count", 0),
            "news_environment_score": current_features.get("news_environment_score", 0),
            "original_model_score_pct": round(float(signals["prob_up"]) * 100, 1),
        },
        "note": (
            "此分數是研究模型的正向分類分數，不是經校準的未來上漲機率。"
            "偏正向／中性／偏負向僅描述模型分類，不是買賣指令；時間外測試結果不代表未來表現。"
        ),
    })
    write_output(payload)
    print(
        f"News-enhanced model ready: samples={len(X)}, holdout={holdout_n}, "
        f"signal={signal}, score={score:.3f}, brier={enhanced_brier:.4f}, "
        f"original_brier={baseline_brier:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
