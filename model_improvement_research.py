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
MIN_TRAIN = 100
RANDOM_STATE = 42


def prepare_market_frame(market):
    """Match the production pipeline's trading-calendar and limited-fill preprocessing."""
    market = market[~market.index.duplicated(keep="last")].sort_index()
    if "TAIEX" in market:
        market = market.dropna(subset=["TAIEX"]).ffill(limit=3)
    return market


def load_inputs():
    market = pd.read_csv("market_history.csv", index_col=0, parse_dates=True).sort_index()
    market.index = pd.to_datetime(market.index).normalize()
    market = prepare_market_frame(market)
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


def walk_forward_candidate(X, y, factory, step=STEP, train_window=None):
    valid = X.notna().all(axis=1) & y.notna()
    idx = X.index[valid]
    n = len(idx)
    if n < 100:
        return None, None, {"status": "insufficient", "available_samples": n}
    first_test = MIN_TRAIN
    if n <= first_test + 30:
        return None, None, {"status": "insufficient", "available_samples": n,
                            "minimum_training_rows": MIN_TRAIN,
                            "note": "不足訓練資料與至少 30 筆樣本外評估資料。"}
    pred = pd.Series(np.nan, index=idx, dtype=float)
    baseline = pd.Series(np.nan, index=idx, dtype=float)
    for i in range(first_test, n, step):
        train_start = max(0, i - int(train_window)) if train_window else 0
        train_idx = idx[train_start:i]
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
        "training_window_sessions": int(train_window) if train_window else None,
        "yearly": {},
    })
    for year in sorted(set(pred.index.year)):
        mask = pred.index.year == year
        if int(mask.sum()) >= 10:
            result["yearly"][str(year)] = evaluate_predictions(
                pred.loc[mask], actual.loc[mask], baseline.loc[mask]
            )
    return pred, baseline, result


def current_research_judgment(X, y, factory):
    """Fit on labels strictly before the latest complete feature snapshot."""
    feature_complete = X.notna().all(axis=1)
    latest_data_date = X.index.max()
    latest_missing = (
        X.columns[X.loc[latest_data_date].isna()].tolist()
        if latest_data_date in X.index else []
    )
    if not feature_complete.any():
        return {"status": "insufficient", "asof": None, "training_samples": 0,
                "latest_data_date": str(latest_data_date.date()),
                "missing_features_at_latest_data_date": latest_missing,
                "note": "目前沒有完整的最新特徵列。"}
    # The raw calendar can contain trailing rows with unavailable source layers.
    # Select the latest row that this candidate can actually evaluate.
    current_date = X.index[feature_complete].max()
    current = X.loc[[current_date]]
    freshness_note = (
        f"最新原始資料日 {latest_data_date.date()} 缺少特徵："
        + "、".join(latest_missing)
        + f"；因此採用最新完整特徵日 {current_date.date()}。"
        if latest_missing else
        f"最新原始資料日與完整特徵日均為 {current_date.date()}。"
    )
    valid = feature_complete & y.notna()
    train_idx = X.index[valid & (X.index < current_date)]
    if len(train_idx) < MIN_TRAIN:
        return {"status": "insufficient", "asof": str(current_date.date()),
                "latest_data_date": str(latest_data_date.date()),
                "missing_features_at_latest_data_date": latest_missing,
                "training_samples": int(len(train_idx)),
                "note": freshness_note + " 歷史標記樣本尚未達訓練門檻。"}
    y_train = y.loc[train_idx].astype(int)
    if y_train.nunique() < 2:
        return {"status": "insufficient", "asof": str(current_date.date()),
                "latest_data_date": str(latest_data_date.date()),
                "missing_features_at_latest_data_date": latest_missing,
                "training_samples": int(len(train_idx)),
                "note": freshness_note + " 歷史標記僅包含單一類別。"}
    model = factory()
    model.fit(X.loc[train_idx], y_train)
    p = float(model.predict_proba(current)[:, 1][0])
    category = "偏正向研究分類" if p >= 0.55 else "偏負向研究分類" if p <= 0.45 else "中性研究分類"
    return {"status": "available", "asof": str(current_date.date()),
            "latest_data_date": str(latest_data_date.date()),
            "missing_features_at_latest_data_date": latest_missing,
            "training_samples": int(len(train_idx)), "category": category,
            "note": freshness_note + " 訓練標籤僅取判讀日之前的資料；非盤中即時行情、非買賣指示。"}


def paired_block_bootstrap(candidate_prob, incumbent_prob, actual, block_size=20, repeats=1000):
    """Paired moving-block bootstrap; negative Brier delta favors candidate."""
    cp = np.clip(np.asarray(candidate_prob, dtype=float), 1e-6, 1 - 1e-6)
    ip = np.clip(np.asarray(incumbent_prob, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(actual, dtype=int)
    n = len(y)
    if n < block_size or len(cp) != n or len(ip) != n:
        return {"status": "insufficient"}
    brier_delta = (cp - y) ** 2 - (ip - y) ** 2
    accuracy_delta = ((cp >= 0.5) == (y == 1)).astype(float) - ((ip >= 0.5) == (y == 1)).astype(float)
    rng = np.random.default_rng(RANDOM_STATE)
    n_blocks = int(np.ceil(n / block_size))
    brier_samples = np.empty(repeats)
    accuracy_samples = np.empty(repeats)
    max_start = n - block_size
    for i in range(repeats):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        brier_samples[i] = float(np.mean(brier_delta[indices]))
        accuracy_samples[i] = float(np.mean(accuracy_delta[indices]) * 100)
    return {
        "status": "available",
        "block_size_sessions": int(block_size),
        "bootstrap_repeats": int(repeats),
        "brier_delta_candidate_minus_incumbent": round(float(np.mean(brier_delta)), 5),
        "brier_delta_95pct_ci": [round(float(x), 5) for x in np.quantile(brier_samples, [0.025, 0.975])],
        "accuracy_delta_percentage_points": round(float(np.mean(accuracy_delta) * 100), 2),
        "accuracy_delta_95pct_ci": [round(float(x), 2) for x in np.quantile(accuracy_samples, [0.025, 0.975])],
        "interpretation": "Negative Brier delta favors candidate; positive accuracy delta favors candidate. Confidence intervals are moving-block bootstrap estimates, not guarantees.",
    }


def paired_feature_sets(X_raw, market_cols, paired_index):
    """Feature-layer ablations on one identical complete-case calendar."""
    sets = {"market_only": list(market_cols)}
    for key, prefix in (
        ("market_plus_chips", "籌碼:"),
        ("market_plus_margin", "信用:"),
        ("market_plus_futures", "期貨:"),
    ):
        layer = [col for col in X_raw.columns if col.startswith(prefix)]
        if layer:
            sets[key] = list(market_cols) + layer
    sets["all_features"] = list(X_raw.columns)
    return {
        name: X_raw.loc[paired_index, cols]
        for name, cols in sets.items()
    }


def run():
    market, chips, margin, futures = load_inputs()
    X_raw, y_raw, _ = pipeline.build(market, chips, margin, futures=futures)

    # Report feature coverage before complete-case filtering; otherwise every
    # reported feature would misleadingly appear to have 100% coverage.
    feature_coverage = {
        str(col): round(float(X_raw[col].notna().mean() * 100), 1)
        for col in X_raw.columns
    }
    # Each candidate gets its own complete-case calendar. This prevents sparse
    # chip/credit/futures columns from suppressing a market-only model's usable
    # history. Results on different calendars are explicitly non-comparable.
    market_cols = [c for c in X_raw.columns
                   if not c.startswith(("籌碼:", "信用:", "期貨:"))]
    experiments = {}
    specs = candidates()
    for name, factory in specs.items():
        cols = market_cols if name == "logistic_market_features_only" else list(X_raw.columns)
        X_candidate = X_raw[cols]
        _, _, result = walk_forward_candidate(X_candidate, y_raw, factory, step=STEP)
        result["current_judgment"] = current_research_judgment(X_candidate, y_raw, factory)
        result["feature_count"] = len(cols)
        result["sample_calendar"] = "market_features_only" if name == "logistic_market_features_only" else "all_features_complete_cases"
        experiments[name] = result

    # Test each Taiwan data layer separately so sparse sources do not
    # force every other source onto the same complete-case date calendar.
    layer_cols = {
        "logistic_market_plus_chips": market_cols + [c for c in X_raw.columns if c.startswith("籌碼:")],
        "logistic_market_plus_margin": market_cols + [c for c in X_raw.columns if c.startswith("信用:")],
        "logistic_market_plus_futures": market_cols + [c for c in X_raw.columns if c.startswith("期貨:")],
    }
    for name, cols in layer_cols.items():
        _, _, result = walk_forward_candidate(
            X_raw[cols], y_raw, specs["logistic_market_features_only"], step=STEP
        )
        result["current_judgment"] = current_research_judgment(X_raw[cols], y_raw, specs["logistic_market_features_only"])
        result["feature_count"] = len(cols)
        result["sample_calendar"] = name.removeprefix("logistic_")
        experiments[name] = result

    # Paired comparison: force all model families onto the exact same
    # all-feature complete-case calendar, labels and walk-forward schedule.
    # This isolates algorithm/feature-set effects from missing-data coverage.
    paired_valid = X_raw.notna().all(axis=1) & y_raw.notna()
    paired_index = X_raw.index[paired_valid]
    paired_y = y_raw.loc[paired_index]
    paired_experiments = {}
    paired_sets = paired_feature_sets(X_raw, market_cols, paired_index)
    paired_specs = {
        "incumbent_all_features_logistic": (paired_sets["all_features"], specs["logistic_current_step5"]),
        "market_only_logistic": (paired_sets["market_only"], specs["logistic_market_features_only"]),
        "market_only_random_forest": (paired_sets["market_only"], specs["random_forest_conservative"]),
        "market_only_hist_gradient_boosting": (paired_sets["market_only"], specs["hist_gradient_boosting_conservative"]),
    }
    # Ablate one Taiwan data layer at a time while holding the dates, labels,
    # walk-forward folds and logistic estimator constant. This isolates
    # incremental layer value from missing-data calendar effects.
    for layer_name in ("market_plus_chips", "market_plus_margin", "market_plus_futures"):
        if layer_name in paired_sets:
            paired_specs[layer_name + "_logistic"] = (
                paired_sets[layer_name], specs["logistic_market_features_only"]
            )
    incumbent_pred = None
    for paired_name, (paired_X, paired_factory) in paired_specs.items():
        paired_pred, _, paired_result = walk_forward_candidate(
            paired_X, paired_y, paired_factory, step=STEP
        )
        if paired_name == "incumbent_all_features_logistic":
            incumbent_pred = paired_pred
        elif incumbent_pred is not None and paired_pred is not None:
            common_pred = paired_pred.index.intersection(incumbent_pred.index)
            paired_result["paired_uncertainty_vs_incumbent"] = paired_block_bootstrap(
                paired_pred.loc[common_pred],
                incumbent_pred.loc[common_pred],
                paired_y.loc[common_pred],
            )
        paired_result["current_judgment"] = current_research_judgment(
            paired_X, paired_y, paired_factory
        )
        paired_result["feature_count"] = int(paired_X.shape[1])
        paired_result["sample_calendar"] = "identical_all_features_complete_case_calendar"
        paired_result["comparison_start"] = (
            str(paired_y.index.min().date()) if len(paired_y) else None
        )
        paired_result["comparison_end"] = (
            str(paired_y.index.max().date()) if len(paired_y) else None
        )
        paired_experiments[paired_name] = paired_result

    # Test whether a finite rolling training history adapts better than the
    # expanding window. All variants use the same complete-case dates, labels,
    # test folds and refit cadence; only the amount of training history differs.
    rolling_specs = {
        "all_features_rolling_252": (paired_sets["all_features"], specs["logistic_current_step5"], 252),
        "all_features_rolling_504": (paired_sets["all_features"], specs["logistic_current_step5"], 504),
        "market_only_rolling_252": (paired_sets["market_only"], specs["logistic_market_features_only"], 252),
        "market_only_rolling_504": (paired_sets["market_only"], specs["logistic_market_features_only"], 504),
    }
    for rolling_name, (rolling_X, rolling_factory, window) in rolling_specs.items():
        rolling_pred, _, rolling_result = walk_forward_candidate(
            rolling_X, paired_y, rolling_factory, step=STEP, train_window=window
        )
        if rolling_pred is not None and incumbent_pred is not None:
            common_pred = rolling_pred.index.intersection(incumbent_pred.index)
            rolling_result["paired_uncertainty_vs_expanding_incumbent"] = paired_block_bootstrap(
                rolling_pred.loc[common_pred],
                incumbent_pred.loc[common_pred],
                paired_y.loc[common_pred],
            )
        rolling_result["feature_count"] = int(rolling_X.shape[1])
        rolling_result["sample_calendar"] = "identical_all_features_complete_case_calendar"
        rolling_result["training_window_sessions"] = int(window)
        rolling_result["comparison_start"] = (
            str(paired_y.index.min().date()) if len(paired_y) else None
        )
        rolling_result["comparison_end"] = (
            str(paired_y.index.max().date()) if len(paired_y) else None
        )
        paired_experiments[rolling_name] = rolling_result

    # Production-cadence reference uses the same full-feature sample calendar.
    _, _, production_result = walk_forward_candidate(
        X_raw, y_raw, specs["logistic_current_step5"], step=20
    )
    production_result["current_judgment"] = current_research_judgment(X_raw, y_raw, specs["logistic_current_step5"])
    production_result["feature_count"] = int(X_raw.shape[1])
    production_result["sample_calendar"] = "all_features_complete_cases"
    experiments["production_cadence_step20"] = production_result

    full_valid = X_raw.notna().all(axis=1) & y_raw.notna()
    complete_samples = int(full_valid.sum())
    market_valid = X_raw[market_cols].notna().all(axis=1) & y_raw.notna()
    market_complete_samples = int(market_valid.sum())
    any_available = any(item.get("status") == "available" for item in experiments.values())

    return {
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated" if any_available else "insufficient",
        "target": "TAIEX next-session direction",
        "available_complete_samples": complete_samples,
        "market_only_complete_samples": market_complete_samples,
        "minimum_training_rows": MIN_TRAIN,
        "minimum_oos_samples": 30,
        "feature_count": int(X_raw.shape[1]),
        "feature_coverage_pct": feature_coverage,
        "method": "Expanding-window chronological walk-forward; each candidate uses its own complete-case feature calendar; at least 100 training rows and 30 out-of-sample rows required; refit every 5 sessions (production cadence comparator every 20); no random shuffle; yearly subperiod metrics. Results on different calendars must not be compared as if they used identical dates.",
        "experiments": experiments,
        "same_calendar_comparison": {
            "calendar": "Exact all-feature complete-case intersection; identical labels, chronological walk-forward schedule and expanding-prevalence baseline.",
            "sample_count": int(len(paired_y)),
            "experiments": paired_experiments,
            "note": "Use this paired block for direct model comparisons. Broader per-feature-calendar experiments are coverage context only and are not directly rankable.",
        },
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
