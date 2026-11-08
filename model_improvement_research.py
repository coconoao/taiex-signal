"""Research-only challenger experiments for the Taiwan model.

Never writes signals.json and never changes the production model. All model
comparisons use identical chronological out-of-sample dates where possible.
No candidate is automatically promoted based on a single aggregate score.
"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pipeline

STEP = 5
MIN_TRAIN = 500
RANDOM_STATE = 42


def load_inputs():
    market = pd.read_csv("market_history.csv", index_col=0, parse_dates=True).sort_index()
    market.index = pd.to_datetime(market.index).normalize()
    def optional(name):
        try:
            frame = pd.read_csv(name, index_col=0, parse_dates=True).sort_index()
            frame.index = pd.to_datetime(frame.index).normalize()
            return frame
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return None
    return market, optional("chips.csv"), optional("margin.csv"), optional("futures.csv")


def candidates():
    return {
        "logistic_current_step5": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=500)
        ),
        "logistic_stronger_regularization": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=500)
        ),
        "logistic_weaker_regularization": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=2.0, max_iter=500)
        ),
        "random_forest_conservative": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=3, min_samples_leaf=20,
            max_features=0.8, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=2,
        ),
        "hist_gradient_boosting_conservative": lambda: HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=7, min_samples_leaf=30,
            learning_rate=0.05, l2_regularization=5.0,
            random_state=RANDOM_STATE,
        ),
        "logistic_market_features_only": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=500)
        ),
    }


def evaluate_predictions(prob, actual, baseline_prob):
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(actual, dtype=int)
    bp = np.clip(np.asarray(baseline_prob, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {}
    brier = float(np.mean((p - y) ** 2))
    base_brier = float(np.mean((bp - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {
        "sample_count": int(len(y)),
        "direction_accuracy_pct": round(float(np.mean((p >= 0.5) == (y == 1)) * 100), 2),
        "baseline_direction_accuracy_pct": round(float(np.mean((bp >= 0.5) == (y == 1)) * 100), 2),
        "brier": round(brier, 5),
        "baseline_brier": round(base_brier, 5),
        "brier_skill": round(1 - brier / base_brier, 5) if base_brier > 0 else None,
        "log_loss": round(logloss, 5),
    }


def walk_forward_candidate(X, y, factory, step=STEP):
    valid = X.notna().all(axis=1) & y.notna()
    idx = X.index[valid]
    n = len(idx)
    if n < 100:
        return None, None, {"status": "insufficient", "available_samples": n}
    first_test = min(max(MIN_TRAIN, 31), max(31, n // 2))
    pred = pd.Series(np.nan, index=idx, dtype=float)
    baseline = pd.Series(np.nan, index=idx, dtype=float)
    for i in range(first_test, n, step):
        train_idx = idx[:i]
        test_idx = idx[i:i + step]
        y_train = y.loc[train_idx].astype(int)
        if len(y_train) < 30 or y_train.nunique() < 2:
            continue
        model = factory()
        model.fit(X.loc[train_idx], y_train)
        pred.loc[test_idx] = model.predict_proba(X.loc[test_idx])[:, 1]
        # Expanding historical prevalence is a transparent, non-model baseline.
        baseline.loc[test_idx] = float(y_train.mean())
    keep = pred.notna() & baseline.notna()
    pred, baseline = pred.loc[keep], baseline.loc[keep]
    actual = y.loc[pred.index].astype(int)
    if len(pred) < 30:
        return pred, baseline, {"status": "insufficient", "sample_count": int(len(pred))}
    result = {"status": "available", **evaluate_predictions(pred, actual, baseline)}
    result.update({
        "sample_start": str(pred.index.min().date()),
        "sample_end": str(pred.index.max().date()),
        "refit_every_sessions": int(step),
        "training_minimum_target": int(MIN_TRAIN),
        "yearly": {},
    })
    for year in sorted(set(pred.index.year)):
        mask = pred.index.year == year
        if int(mask.sum()) >= 10:
            result["yearly"][str(year)] = evaluate_predictions(
                pred.loc[mask], actual.loc[mask], baseline.loc[mask]
            )
    return pred, baseline, result


def run():
    market, chips, margin, futures = load_inputs()
    X, y, _ = pipeline.build(market, chips, margin, futures=futures)
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < 100:
        raise RuntimeError(f"完整特徵樣本不足 100 筆：{len(X)}")

    market_cols = [c for c in X.columns if not c.startswith(("籌碼:", "信用:", "期貨:"))]
    experiments = {}
    specs = candidates()
    for name, factory in specs.items():
        use = X[market_cols] if name == "logistic_market_features_only" else X
        step = 20 if name == "logistic_current_step5" and False else STEP
        pred, base, result = walk_forward_candidate(use, y, factory, step=step)
        experiments[name] = result

    # Exact production LogisticRegression(C=0.5) cadence reference.
    _, _, experiments["production_cadence_step20"] = walk_forward_candidate(
        X, y, specs["logistic_current_step5"], step=20
    )
    # Feature coverage and the calendar are part of the result, not hidden.
    feature_coverage = {
        str(col): round(float(X[col].notna().mean() * 100), 1)
        for col in X.columns
    }
    return {
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only",
        "target": "TAIEX next-session direction",
        "available_complete_samples": int(len(X)),
        "feature_count": int(X.shape[1]),
        "feature_coverage_pct": feature_coverage,
        "method": "Expanding-window chronological walk-forward; first test after at least 500 training rows where available, refit every 5 sessions (production cadence comparator every 20); no random shuffle; identical chronological target labels; yearly subperiod metrics.",
        "experiments": experiments,
        "promotion_policy": "No automatic promotion. Require repeatable improvement across multiple time blocks, improved Brier/log loss versus baseline, adequate coverage, and a later untouched holdout before any production change.",
        "disclaimer": "Historical research only. Metrics are not a guarantee of future performance and are not investment advice or trading instructions.",
    }


if __name__ == "__main__":
    result = run()
    tmp = "model_improvement_research.json.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, "model_improvement_research.json")
    for name, metrics in result["experiments"].items():
        print(name, metrics.get("status"), metrics.get("sample_count"),
              metrics.get("direction_accuracy_pct"), metrics.get("brier_skill"))
