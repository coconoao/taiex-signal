"""Research-only four-regime router and specialist forecasts for TAIEX.

All targets are forward close-to-close returns. Features are supplied by
pipeline.build() and must be available as of each forecast origin. Walk-forward
folds purge the final h labelled rows before each test block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REGIME_NAMES = ("low", "normal", "high", "extreme")
REGIME_LABELS = {
    "low": "低波動",
    "normal": "正常波動",
    "high": "高波動",
    "extreme": "極高波動",
}
# Initial research boundaries applied to the maximum absolute one-session
# close-to-close move within each forecast horizon. The 7.5% threshold is user-specified.
LOW_MAX_RETURN = 0.005
NORMAL_MAX_RETURN = 0.01
EXTREME_MIN_RETURN = 0.075
MIN_TRAIN_SAMPLES = 180
MIN_EXPERT_SAMPLES = {0: 60, 1: 60, 2: 45, 3: 40}
EXPERT_ALPHA = {0: 25.0, 1: 25.0, 2: 25.0, 3: 25.0}
CLASS_IDS = (0, 1, 2, 3)


def classify_regimes(future_max_daily_moves):
    """Map future maximum absolute one-day moves to mutually exclusive regimes."""
    values = np.asarray(future_max_daily_moves, dtype=float)
    magnitude = np.abs(values)
    labels = np.full(values.shape, -1, dtype=int)
    valid = np.isfinite(values)
    labels[valid & (magnitude <= LOW_MAX_RETURN)] = 0
    labels[valid & (magnitude > LOW_MAX_RETURN) & (magnitude <= NORMAL_MAX_RETURN)] = 1
    labels[valid & (magnitude > NORMAL_MAX_RETURN) & (magnitude < EXTREME_MIN_RETURN)] = 2
    labels[valid & (magnitude >= EXTREME_MIN_RETURN)] = 3
    if isinstance(future_max_daily_moves, pd.Series):
        return pd.Series(labels, index=future_max_daily_moves.index, name="regime")
    return labels


def _future_max_daily_move(close, horizon):
    """Maximum absolute one-session close-to-close move in the next h sessions."""
    prices = pd.to_numeric(close, errors="coerce").sort_index()
    daily_returns = prices.pct_change()
    future_moves = pd.concat(
        [daily_returns.shift(-offset).abs().rename(f"D+{offset}") for offset in range(1, int(horizon) + 1)],
        axis=1,
    )
    return future_moves.max(axis=1)


def _lag_base_cross_market_features(X):
    """Shift pre-existing international/regional returns to Taiwan-close availability.

    pipeline.build() is shared with an overnight forecast path where same-date
    US closes are valid for the next Taiwan session. The router's forecast
    origin is instead the Taiwan cash close, so those same-date values are not
    yet observable and must be lagged here.
    """
    features = X.copy()
    columns = [
        column for column in features.columns
        if str(column).startswith(("國際:", "區域:"))
    ]
    if columns:
        features.loc[:, columns] = features.loc[:, columns].shift(1)
    return features


def _lag_post_close_source_frame(frame, trading_index):
    """Align post-close Taiwan data to the full cash-session calendar, then lag one session.

    Institutional flow, margin/short-sale balances, and futures open interest are
    published after the Taiwan cash close. Reindex before shifting so missing
    source dates never compress the session lag.
    """
    if frame is None:
        return None
    aligned = frame.copy()
    aligned.index = pd.to_datetime(aligned.index).tz_localize(None).normalize()
    aligned = aligned[~aligned.index.duplicated(keep="last")].sort_index()
    calendar = pd.DatetimeIndex(trading_index).tz_localize(None).normalize()
    return aligned.reindex(calendar).shift(1)


def _lag_post_close_features(X):
    """Lag post-close Taiwan institutional, credit, and futures features one session."""
    features = X.copy()
    columns = [
        column for column in features.columns
        if str(column).startswith(("籌碼:", "信用:", "期貨:"))
    ]
    if columns:
        features.loc[:, columns] = features.loc[:, columns].shift(1)
    return features


def _augment_router_features(X, market_data):
    """Add point-in-time volatility and cross-market level/momentum features."""
    if market_data is None:
        return X
    market = market_data.reindex(X.index).apply(pd.to_numeric, errors="coerce")
    features = X.copy()
    # Canonical forecast origin is the Taiwan cash close. Cross-market closes
    # from the same calendar label may occur after that close (US session), so
    # lag every non-Taiwan market input by one Taiwan trading session. This keeps
    # historical OOS rows aligned with what production actually knew at 16:15.
    cross_market = (
        ("SOX", "國際:SOX"),
        ("TSM", "國際:TSM"),
        ("VIX", "國際:VIX"),
        ("US10Y", "國際:US10Y"),
        ("N225", "區域:N225"),
        ("KOSPI", "區域:KOSPI"),
        ("USDTWD", "區域:USDTWD"),
    )
    for ticker, feature_name in cross_market:
        if ticker in market:
            features[feature_name] = market[ticker].pct_change().shift(1)
    if "TAIEX" in market:
        close = market["TAIEX"]
        daily = close.pct_change()
        features["router:TAIEX_realized_vol_5d"] = daily.rolling(5, min_periods=5).std()
        features["router:TAIEX_realized_vol_20d"] = daily.rolling(20, min_periods=15).std()
        features["router:TAIEX_abs_move_max_10d"] = daily.abs().rolling(10, min_periods=8).max()
        features["router:TAIEX_abs_return_5d"] = close.pct_change(5).abs()
        features["router:TAIEX_drawdown_20d"] = close / close.rolling(20, min_periods=15).max() - 1.0
    for ticker in ("SOX", "TSM", "N225", "KOSPI", "USDTWD"):
        if ticker in market:
            features[f"router:{ticker}_momentum_5d"] = market[ticker].pct_change(5).shift(1)
    if "VIX" in market:
        vix = market["VIX"]
        features["router:VIX_level_vs_60d_mean"] = (
            vix / vix.rolling(60, min_periods=20).mean() - 1.0
        ).shift(1)
        features["router:VIX_change_5d"] = vix.pct_change(5).shift(1)
    if "US10Y" in market:
        features["router:US10Y_change_5d"] = market["US10Y"].diff(5).shift(1)
    return features


def _fit_router(X, y):
    labels = np.asarray(y, dtype=int)
    classes = np.sort(np.unique(labels))
    if len(classes) < 2:
        return None
    # Blend a balanced linear classifier with a regularized nonlinear tree
    # booster. The latter captures volatility interactions; the former keeps
    # a stable baseline. Both remain subject to OOS calibration and evaluation.
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, max_iter=900, class_weight="balanced", random_state=17),
    )
    logistic.fit(X, labels)
    counts = np.bincount(labels, minlength=4)
    class_weights = {
        class_id: min(4.0, len(labels) / (max(1, len(classes)) * max(1, counts[class_id])))
        for class_id in classes
    }
    sample_weights = np.asarray([class_weights[int(label)] for label in labels], dtype=float)
    boosted = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=12,
        max_leaf_nodes=7,
        min_samples_leaf=30,
        l2_regularization=8.0,
        early_stopping=False,
        random_state=23,
    )
    boosted.fit(X, labels, sample_weight=sample_weights)
    return {"logistic": logistic, "boosted": boosted}


def _router_probabilities(estimator, X, fallback_class=None):
    probabilities = np.zeros((len(X), 4), dtype=float)
    if estimator is None:
        if fallback_class is not None:
            probabilities[:, int(fallback_class)] = 1.0
        else:
            probabilities[:, 1] = 1.0
        return probabilities
    for model in (estimator["logistic"], estimator["boosted"]):
        raw = model.predict_proba(X)
        classes = model[-1].classes_ if hasattr(model, "steps") else model.classes_
        aligned = np.zeros((len(X), 4), dtype=float)
        for column, class_id in enumerate(classes):
            aligned[:, int(class_id)] = raw[:, column]
        probabilities += aligned / 2.0
    row_sums = probabilities.sum(axis=1, keepdims=True)
    return np.divide(probabilities, row_sums, out=np.zeros_like(probabilities), where=row_sums > 0)


def _fit_probability_calibrator(probability_history, actual_history):
    """Fit a point-in-time calibrator using only earlier OOS router predictions."""
    if len(actual_history) < 100 or len(set(actual_history)) < 2:
        return None
    features = np.log(np.clip(np.asarray(probability_history, dtype=float), 1e-7, 1.0))
    calibrator = LogisticRegression(C=1.0, max_iter=800, random_state=29)
    calibrator.fit(features, np.asarray(actual_history, dtype=int))
    return calibrator


def _apply_probability_calibrator(calibrator, raw_probabilities):
    raw = np.asarray(raw_probabilities, dtype=float)
    if calibrator is None:
        return raw
    features = np.log(np.clip(raw, 1e-7, 1.0))
    raw_calibrated = calibrator.predict_proba(features)
    calibrated = np.zeros((len(raw), 4), dtype=float)
    for column, class_id in enumerate(calibrator.classes_):
        calibrated[:, int(class_id)] = raw_calibrated[:, column]
    totals = calibrated.sum(axis=1, keepdims=True)
    return np.divide(calibrated, totals, out=np.zeros_like(calibrated), where=totals > 0)




def _purged_training_indices(feature_index, candidate_train_idx, first_test_date, horizon):
    """Keep only training origins whose full forward label ends by test origin.

    Use positions in the complete trading-session index, not positions in the
    filtered valid-sample array: missing feature rows must not weaken the purge.
    """
    candidate_train_idx = pd.Index(candidate_train_idx)
    positions = feature_index.get_indexer(candidate_train_idx)
    test_position = feature_index.get_loc(first_test_date)
    return candidate_train_idx[positions <= int(test_position) - int(horizon)]


def _eligible_calibration_records(records, current_position, horizon):
    """Only use prior OOS outcomes whose target close is already observable."""
    return [record for record in records if record[0] + int(horizon) <= int(current_position)]


def _best_f2_threshold(
    scores, actual_positive, default, min_positives, precision_floor, max_rate_multiplier
):
    scores = np.asarray(scores, dtype=float)
    actual_positive = np.asarray(actual_positive, dtype=bool)
    positives = int(actual_positive.sum())
    if positives < min_positives:
        return float(default)
    prevalence = float(actual_positive.mean())
    max_predicted_rate = min(0.95, prevalence * float(max_rate_multiplier))
    best_threshold, best_score = float(default), -1.0
    for threshold in np.arange(0.025, 0.976, 0.025):
        predicted = scores >= threshold
        predicted_count = int(predicted.sum())
        if predicted_count == 0 or predicted_count / len(scores) > max_predicted_rate:
            continue
        true_positive = int(np.sum(predicted & actual_positive))
        precision = true_positive / predicted_count
        recall = true_positive / positives
        if precision < precision_floor:
            continue
        f2 = 5.0 * precision * recall / max(1e-12, 4.0 * precision + recall)
        if f2 > best_score or (np.isclose(f2, best_score) and threshold > best_threshold):
            best_threshold, best_score = float(threshold), float(f2)
    if best_score >= 0:
        return best_threshold
    # If precision and prevalence constraints cannot both be met, rank by score
    # and predict approximately the historical base rate instead of routing
    # nearly every session to high volatility or suppressing all risk upgrades.
    return float(np.quantile(scores, 1.0 - prevalence, method="higher"))


def _select_risk_thresholds(probability_history, actual_history):
    """Tune risk-first escalation using only earlier calibrated OOS probabilities."""
    if len(actual_history) < 100:
        return {
            "high_extreme": 0.40, "extreme": 0.20,
            "source": "default_insufficient_calibrated_oos",
            "training_oos_samples": int(len(actual_history)),
        }
    probs = np.asarray(probability_history, dtype=float)
    actual = np.asarray(actual_history, dtype=int)
    high_extreme = _best_f2_threshold(
        probs[:, 2] + probs[:, 3], actual >= 2,
        default=0.40, min_positives=20, precision_floor=0.30, max_rate_multiplier=1.0,
    )
    extreme = _best_f2_threshold(
        probs[:, 3], actual == 3,
        default=0.20, min_positives=12, precision_floor=0.20, max_rate_multiplier=2.0,
    )
    return {
        "high_extreme": round(float(high_extreme), 4),
        "extreme": round(float(extreme), 4),
        "source": "prior_calibrated_oos_f2_with_prevalence_guard",
        "training_oos_samples": int(len(actual)),
    }


def _routing_reason(probabilities, thresholds, selected_class):
    probabilities = np.asarray(probabilities, dtype=float)
    high_risk = float(probabilities[2] + probabilities[3])
    if int(selected_class) == 3:
        return (
            f"極高波動分數 {probabilities[3] * 100:.1f}% 達到升級門檻 "
            f"{float(thresholds['extreme']) * 100:.1f}%"
        )
    if int(selected_class) == 2:
        return (
            f"高／極高合計分數 {high_risk * 100:.1f}% 達到升級門檻 "
            f"{float(thresholds['high_extreme']) * 100:.1f}%"
        )
    return (
        f"高／極高合計分數 {high_risk * 100:.1f}% 未達升級門檻 "
        f"{float(thresholds['high_extreme']) * 100:.1f}%，由低／正常波動分數決定"
    )


def _route_regime(probabilities, thresholds):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities[3] >= float(thresholds["extreme"]):
        return 3
    if probabilities[2] + probabilities[3] >= float(thresholds["high_extreme"]):
        return 2
    return 0 if probabilities[0] >= probabilities[1] else 1


def _fit_experts(X, cumulative_returns, max_daily_moves, regimes):
    global_model = make_pipeline(StandardScaler(), Ridge(alpha=25.0))
    global_model.fit(X, np.asarray(cumulative_returns, dtype=float))
    global_magnitude_model = make_pipeline(StandardScaler(), Ridge(alpha=25.0))
    global_magnitude_model.fit(X, np.asarray(max_daily_moves, dtype=float))
    models, magnitude_models, counts = {}, {}, {}
    for class_id in CLASS_IDS:
        mask = np.asarray(regimes, dtype=int) == class_id
        count = int(mask.sum())
        counts[class_id] = count
        if count >= MIN_EXPERT_SAMPLES[class_id]:
            selected_X = X.iloc[np.flatnonzero(mask)]
            model = make_pipeline(StandardScaler(), Ridge(alpha=EXPERT_ALPHA[class_id]))
            model.fit(selected_X, np.asarray(cumulative_returns, dtype=float)[mask])
            magnitude_model = make_pipeline(StandardScaler(), Ridge(alpha=EXPERT_ALPHA[class_id]))
            magnitude_model.fit(selected_X, np.asarray(max_daily_moves, dtype=float)[mask])
            models[class_id] = model
            magnitude_models[class_id] = magnitude_model
    return global_model, models, global_magnitude_model, magnitude_models, counts


def _select_expert(requested_class, global_model, models, counts):
    requested_class = int(requested_class)
    if requested_class in models:
        return models[requested_class], REGIME_NAMES[requested_class], counts[requested_class], None
    if requested_class == 3 and 2 in models:
        return (
            models[2], "high",
            counts[2],
            "極高波動樣本不足，暫由高波動專家備援",
        )
    return (
        global_model, "global_fallback",
        sum(counts.values()),
        f"{REGIME_LABELS[REGIME_NAMES[requested_class]]}專家樣本不足，暫由全域模型備援",
    )


def _safe_mae(predicted, actual):
    if not len(predicted):
        return None
    return round(float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual)))), 2)


def _historical_magnitude_profile(cumulative_returns, max_daily_moves, base_close):
    """Summarize past cumulative returns and maximum daily moves in the selected regime."""
    returns = np.asarray(cumulative_returns, dtype=float)
    moves = np.asarray(max_daily_moves, dtype=float)
    valid = np.isfinite(returns) & np.isfinite(moves)
    returns, moves = returns[valid], moves[valid]
    if len(returns) < 20:
        return {"status": "insufficient", "samples": int(len(returns))}
    return_quantiles = {
        "p10_return_pct": round(float(np.percentile(returns, 10) * 100), 3),
        "p50_return_pct": round(float(np.percentile(returns, 50) * 100), 3),
        "p90_return_pct": round(float(np.percentile(returns, 90) * 100), 3),
    }
    move_quantiles = {
        "median_abs_move_pct": round(float(np.percentile(moves, 50) * 100), 3),
        "p90_abs_move_pct": round(float(np.percentile(moves, 90) * 100), 3),
    }
    exceedance = {
        str(threshold): round(float(np.mean(moves >= threshold / 100.0) * 100), 2)
        for threshold in (1.0, 2.0, 3.0, 5.0, 7.5)
    }
    return {
        "status": "available",
        "samples": int(len(returns)),
        **return_quantiles,
        **move_quantiles,
        "historical_abs_move_exceedance_pct": exceedance,
        "scenario_p10_close": round(float(base_close * (1.0 + np.percentile(returns, 10))), 2),
        "scenario_p50_close": round(float(base_close * (1.0 + np.percentile(returns, 50))), 2),
        "scenario_p90_close": round(float(base_close * (1.0 + np.percentile(returns, 90))), 2),
        "note": "最大單日幅度為未來區間內最大單日漲跌；點位情境為累積報酬歷史分布。僅供歷史參考，不是未來機率。",
    }


def _expert_performance_blend_weight(
    records,
    expert_name,
    current_position,
    horizon,
    expert_prediction_key,
    global_prediction_key,
    actual_key,
    min_samples=60,
    max_history=500,
    full_weight_improvement=0.05,
):
    """Use only matured prior OOS errors to decide expert-vs-global shrinkage."""
    if expert_name == "global_fallback":
        return {
            "weight": 0.0, "samples": 0, "expert_mae": None,
            "global_mae": None, "relative_improvement": None,
            "reason": "expert_fell_back_to_global_for_training_sample_sufficiency",
        }
    eligible = []
    for row in records:
        if row.get("requested_expert") != expert_name:
            continue
        position = row.get("forecast_position")
        if position is None or int(position) + int(horizon) > int(current_position):
            continue
        values = (
            row.get(expert_prediction_key),
            row.get(global_prediction_key),
            row.get(actual_key),
        )
        if not all(value is not None and np.isfinite(float(value)) for value in values):
            continue
        eligible.append(row)
    eligible.sort(key=lambda row: int(row["forecast_position"]))
    eligible = eligible[-int(max_history):]
    samples = len(eligible)
    if samples < int(min_samples):
        return {
            "weight": 0.0, "samples": samples, "expert_mae": None,
            "global_mae": None, "relative_improvement": None,
            "reason": "matured_prior_oos_samples_insufficient",
        }
    expert_mae = float(np.mean([
        abs(float(row[expert_prediction_key]) - float(row[actual_key]))
        for row in eligible
    ]))
    global_mae = float(np.mean([
        abs(float(row[global_prediction_key]) - float(row[actual_key]))
        for row in eligible
    ]))
    relative_improvement = (
        (global_mae - expert_mae) / max(global_mae, 1e-12)
    )
    weight = float(np.clip(
        relative_improvement / max(float(full_weight_improvement), 1e-12),
        0.0,
        1.0,
    ))
    return {
        "weight": round(weight, 4),
        "samples": int(samples),
        "expert_mae": round(expert_mae, 4),
        "global_mae": round(global_mae, 4),
        "relative_improvement": round(relative_improvement, 4),
        "reason": (
            "expert_beats_global_on_matured_prior_oos"
            if weight > 0 else "global_ridge_not_worse_on_matured_prior_oos"
        ),
    }




def _paired_block_bootstrap_delta(
    rows, predicted_key, global_key, actual_key,
    seed=20261110, block_size=20, replicates=500,
):
    """Block-bootstrap paired absolute-error differences against global Ridge.

    Negative values favor the expert policy. Contiguous blocks preserve
    short-run dependence in the time-ordered out-of-sample errors.
    """
    valid = [
        row for row in rows
        if all(
            row.get(key) is not None and np.isfinite(float(row[key]))
            for key in (predicted_key, global_key, actual_key)
        )
    ]
    n = len(valid)
    if n < 30:
        return {"status": "insufficient", "samples": n}
    delta = np.asarray([
        abs(float(row[predicted_key]) - float(row[actual_key]))
        - abs(float(row[global_key]) - float(row[actual_key]))
        for row in valid
    ], dtype=float)
    size = min(int(block_size), n)
    blocks_per_sample = int(np.ceil(n / size))
    rng = np.random.default_rng(int(seed))
    boot_means = np.empty(int(replicates), dtype=float)
    offsets = np.arange(size)
    for i in range(int(replicates)):
        starts = rng.integers(0, n, size=blocks_per_sample)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        boot_means[i] = float(np.mean(delta[indices]))
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return {
        "status": "available",
        "samples": n,
        "block_size": size,
        "replicates": int(replicates),
        "observed_mae_delta_vs_global": round(float(np.mean(delta)), 4),
        "ci_95_lower": round(float(lower), 4),
        "ci_95_upper": round(float(upper), 4),
        "bootstrap_probability_expert_better": round(float(np.mean(boot_means < 0.0)), 4),
        "significant_improvement_at_95pct": bool(upper < 0.0),
    }



def _gate_policy_sensitivity(rows, horizon, early_fraction=0.70):
    """Compare predeclared expert-shrinkage policies with a chronological holdout.

    Every policy is recalculated at the start of each 60-session OOS block using
    only outcomes from earlier blocks that have matured by that block's origin.
    The first 70% is a policy-selection window; the final 30% is a holdout.
    This diagnostic never changes the forecast used by the main backtest.
    """
    ordered = sorted(rows, key=lambda row: int(row["forecast_position"]))
    if not ordered:
        return {"status": "insufficient", "oos_samples": 0}

    policies = {
        "global_only": {"min_samples": 10**9, "full_weight_improvement": 0.05},
        "gate_60_2pct": {"min_samples": 60, "full_weight_improvement": 0.02},
        "gate_60_5pct": {"min_samples": 60, "full_weight_improvement": 0.05},
        "gate_60_10pct": {"min_samples": 60, "full_weight_improvement": 0.10},
        "gate_120_5pct": {"min_samples": 120, "full_weight_improvement": 0.05},
    }
    blocks = sorted({int(row["oos_block_start_position"]) for row in ordered})

    def summarize(items):
        if not items:
            return {"oos_samples": 0}
        predicted = np.asarray([row["policy_predicted_close"] for row in items], dtype=float)
        actual = np.asarray([row["actual_close"] for row in items], dtype=float)
        predicted_move = np.asarray([row["policy_predicted_max_daily_move"] for row in items], dtype=float)
        actual_move = np.asarray([row["actual_max_daily_move"] for row in items], dtype=float)
        large = actual_move >= 0.01
        global_predicted = np.asarray([row["global_predicted_close"] for row in items], dtype=float)
        global_move = np.asarray([row["global_predicted_max_daily_move"] for row in items], dtype=float)
        return {
            "oos_samples": int(len(items)),
            "point_mae": _safe_mae(predicted, actual),
            "point_mae_delta_vs_global": round(
                _safe_mae(predicted, actual) - _safe_mae(global_predicted, actual), 4
            ),
            "large_move_point_mae": _safe_mae(predicted[large], actual[large]),
            "large_move_point_mae_delta_vs_global": round(
                _safe_mae(predicted[large], actual[large])
                - _safe_mae(global_predicted[large], actual[large]), 4
            ) if large.any() else None,
            "max_daily_move_mae_pct": _safe_mae(predicted_move * 100.0, actual_move * 100.0),
            "max_daily_move_mae_delta_vs_global_pct": round(
                _safe_mae(predicted_move * 100.0, actual_move * 100.0)
                - _safe_mae(global_move * 100.0, actual_move * 100.0), 4
            ),
            "direction_accuracy": round(float(np.mean([
                np.sign(row["policy_predicted_close"] - row["base_close"])
                == np.sign(row["actual_close"] - row["base_close"])
                for row in items
            ])), 4),
            "large_move_samples": int(large.sum()),
        }

    results = {}
    for policy_name, config in policies.items():
        policy_rows = []
        for block_position in blocks:
            prior = [
                row for row in ordered
                if int(row["oos_block_start_position"]) < block_position
            ]
            batch = [
                row for row in ordered
                if int(row["oos_block_start_position"]) == block_position
            ]
            point_gates, magnitude_gates = {}, {}
            for row in batch:
                expert_name = row["expert_used"]
                if expert_name not in point_gates:
                    point_gates[expert_name] = _expert_performance_blend_weight(
                        prior, expert_name, block_position, horizon,
                        "raw_expert_predicted_close", "global_predicted_close",
                        "actual_close", min_samples=config["min_samples"],
                        full_weight_improvement=config["full_weight_improvement"],
                    )
                    magnitude_gates[expert_name] = _expert_performance_blend_weight(
                        prior, expert_name, block_position, horizon,
                        "raw_expert_predicted_max_daily_move",
                        "global_predicted_max_daily_move", "actual_max_daily_move",
                        min_samples=config["min_samples"],
                        full_weight_improvement=config["full_weight_improvement"],
                    )
                point_weight = float(point_gates[expert_name]["weight"])
                magnitude_weight = float(magnitude_gates[expert_name]["weight"])
                policy_rows.append({
                    **row,
                    "policy_predicted_close": (
                        point_weight * float(row["raw_expert_predicted_close"])
                        + (1.0 - point_weight) * float(row["global_predicted_close"])
                    ),
                    "policy_predicted_max_daily_move": max(
                        0.0,
                        magnitude_weight * float(row["raw_expert_predicted_max_daily_move"])
                        + (1.0 - magnitude_weight) * float(row["global_predicted_max_daily_move"]),
                    ),
                })
        policy_rows.sort(key=lambda row: int(row["forecast_position"]))
        split = max(1, min(len(policy_rows) - 1, int(len(policy_rows) * early_fraction)))
        early = summarize(policy_rows[:split])
        late = summarize(policy_rows[split:])
        results[policy_name] = {
            "parameters": config,
            "full_oos": summarize(policy_rows),
            "early_selection_window": early,
            "late_holdout": late,
            "late_holdout_point_mae_delta_ci_vs_global": _paired_block_bootstrap_delta(
                policy_rows[split:],
                "policy_predicted_close",
                "global_predicted_close",
                "actual_close",
                seed=20261110 + int(horizon) * 101 + len(policy_name),
            ),
        }

    best_policy = min(
        results,
        key=lambda name: (
            results[name]["early_selection_window"].get("point_mae", float("inf")),
            0 if name == "global_only" else 1,
        ),
    )
    return {
        "status": "available",
        "oos_samples": int(len(ordered)),
        "early_fraction": float(early_fraction),
        "selection_rule": "lowest point MAE in first chronological 70%; final 30% remains untouched holdout",
        "best_policy_on_early_window": best_policy,
        "selected_policy_late_holdout": results[best_policy]["late_holdout"],
        "policies": results,
        "note": "僅為門檻敏感度研究；不改變主回測輸出，也不自動選擇正式模型。",
    }

def _prior_regime_probabilities(training_regimes, smoothing=1.0):
    """Return a smoothed class-frequency baseline from matured training labels only."""
    values = np.asarray(training_regimes, dtype=int)
    counts = np.bincount(values, minlength=len(CLASS_IDS)).astype(float)
    alpha = max(0.0, float(smoothing))
    probabilities = counts + alpha
    total = float(probabilities.sum())
    if total <= 0:
        return np.full(len(CLASS_IDS), 1.0 / len(CLASS_IDS), dtype=float)
    return probabilities / total


def _router_metrics(actual_classes, predicted_classes, probabilities, prior_probabilities=None):
    if not actual_classes:
        return {
            "status": "insufficient",
            "oos_samples": 0,
            "accuracy": None,
            "accuracy_threshold_gate": {
                "metric": "four_class_regime_oos_accuracy",
                "threshold_exclusive": 0.70,
                "observed_accuracy": None,
                "eligible_by_accuracy_only": False,
                "status": "not_eligible",
                "baseline_accuracy": None,
                "accuracy_above_prior_baseline": None,
                "scope": "insufficient OOS samples",
                "production_enabled": False,
            },
        }
    actual = np.asarray(actual_classes, dtype=int)
    predicted = np.asarray(predicted_classes, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    matrix = confusion_matrix(actual, predicted, labels=CLASS_IDS)
    per_class = {}
    for class_id, name in enumerate(REGIME_NAMES):
        tp = int(matrix[class_id, class_id])
        support = int(matrix[class_id].sum())
        predicted_count = int(matrix[:, class_id].sum())
        per_class[name] = {
            "precision": round(tp / predicted_count, 4) if predicted_count else None,
            "recall": round(tp / support, 4) if support else None,
            "support": support,
        }
    high_mask = actual >= 2
    high_predicted = predicted >= 2
    high_missed = high_mask & ~high_predicted
    high_true_positive = int(np.sum(high_mask & high_predicted))
    high_precision = high_true_positive / int(high_predicted.sum()) if high_predicted.any() else None
    extreme_mask = actual == 3
    extreme_predicted = predicted == 3
    one_hot = np.eye(len(CLASS_IDS), dtype=float)[actual]
    clipped = np.clip(probs, 1e-8, 1.0)
    # The operational router is deliberately risk-first and may override argmax.
    # Report conventional four-class classification separately so its accuracy
    # is not conflated with the risk-escalation decision policy.
    argmax_predicted = np.argmax(probs, axis=1)
    argmax_accuracy = round(float(accuracy_score(actual, argmax_predicted)), 4)
    result = {
        "status": "available",
        "oos_samples": int(len(actual)),
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "risk_first_accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "argmax_accuracy": argmax_accuracy,
        "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
        "macro_f1": round(float(f1_score(actual, predicted, labels=CLASS_IDS, average="macro", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "high_extreme_recall": round(float(np.sum(high_mask & high_predicted) / high_mask.sum()), 4) if high_mask.any() else None,
        "high_extreme_precision": round(float(high_precision), 4) if high_precision is not None else None,
        "high_extreme_predicted_rate": round(float(high_predicted.mean()), 4),
        "high_extreme_actual_rate": round(float(high_mask.mean()), 4),
        "high_extreme_missed": int(high_missed.sum()),
        "high_extreme_missed_rate": round(float(high_missed.sum() / high_mask.sum()), 4) if high_mask.any() else None,
        "extreme_recall": round(float(np.sum(extreme_mask & extreme_predicted) / extreme_mask.sum()), 4) if extreme_mask.any() else None,
        "extreme_precision": round(float(np.sum(extreme_mask & extreme_predicted) / extreme_predicted.sum()), 4) if extreme_predicted.any() else None,
        "extreme_predicted_rate": round(float(extreme_predicted.mean()), 4),
        "extreme_actual_rate": round(float(extreme_mask.mean()), 4),
        "multiclass_brier": round(float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))), 5),
        "log_loss": round(float(log_loss(actual, clipped, labels=list(CLASS_IDS))), 5),
        "probabilities_calibrated": False,
    }
    if prior_probabilities is not None and len(prior_probabilities) == len(actual):
        prior_probs = np.asarray(prior_probabilities, dtype=float)
        prior_probs = np.clip(prior_probs, 1e-8, 1.0)
        prior_probs = prior_probs / prior_probs.sum(axis=1, keepdims=True)
        prior_predicted = np.argmax(prior_probs, axis=1)
        prior_matrix = confusion_matrix(actual, prior_predicted, labels=CLASS_IDS)
        prior_per_class = {}
        for class_id, name in enumerate(REGIME_NAMES):
            tp = int(prior_matrix[class_id, class_id])
            support = int(prior_matrix[class_id].sum())
            predicted_count = int(prior_matrix[:, class_id].sum())
            prior_per_class[name] = {
                "precision": round(tp / predicted_count, 4) if predicted_count else None,
                "recall": round(tp / support, 4) if support else None,
                "support": support,
            }
        result["prequential_prior_frequency_baseline"] = {
            "source": "blockwise_expanding_matured_training_class_frequency_laplace_smoothing",
            "oos_samples": int(len(actual)),
            "accuracy": round(float(accuracy_score(actual, prior_predicted)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(actual, prior_predicted)), 4),
            "macro_f1": round(float(f1_score(actual, prior_predicted, labels=CLASS_IDS, average="macro", zero_division=0)), 4),
            "per_class": prior_per_class,
            "confusion_matrix": prior_matrix.tolist(),
            "multiclass_brier": round(float(np.mean(np.sum((prior_probs - one_hot) ** 2, axis=1))), 5),
            "log_loss": round(float(log_loss(actual, prior_probs, labels=list(CLASS_IDS))), 5),
        }
    # Strict user-defined gate: accuracy must be greater than 70%, not equal.
    # This is a research eligibility flag only; production replacement remains disabled.
    accuracy = result.get("accuracy")
    threshold = 0.70
    baseline = result.get("prequential_prior_frequency_baseline", {})
    result["accuracy_threshold_gate"] = {
        "metric": "four_class_regime_argmax_oos_accuracy",
        "threshold_exclusive": threshold,
        "observed_accuracy": result.get("argmax_accuracy"),
        "eligible_by_accuracy_only": bool(result.get("argmax_accuracy") is not None and result["argmax_accuracy"] > threshold),
        "status": (
            "eligible_by_accuracy_only"
            if result.get("argmax_accuracy") is not None and result["argmax_accuracy"] > threshold
            else "not_eligible"
        ),
        "baseline_accuracy": baseline.get("accuracy"),
        "accuracy_above_prior_baseline": (
            bool(result["argmax_accuracy"] > baseline["accuracy"])
            if result.get("argmax_accuracy") is not None and baseline.get("accuracy") is not None
            else None
        ),
        "scope": "argmax of OOS class probabilities; pooled chronological walk-forward OOS, not an untouched prospective holdout",
        "risk_first_accuracy_reported_separately": True,
        "production_enabled": False,
    }
    return result


def _forecast_metrics(rows):
    if not rows:
        return {"status": "insufficient", "oos_samples": 0}
    predicted = np.asarray([row["predicted_close"] for row in rows], dtype=float)
    actual = np.asarray([row["actual_close"] for row in rows], dtype=float)
    actual_returns = np.asarray([row["actual_return"] for row in rows], dtype=float)
    predicted_returns = np.asarray([row["predicted_return"] for row in rows], dtype=float)
    actual_max_moves = np.asarray([row["actual_max_daily_move"] for row in rows], dtype=float)
    predicted_max_moves = np.asarray([row["predicted_max_daily_move"] for row in rows], dtype=float)
    global_predicted = np.asarray([row["global_predicted_close"] for row in rows], dtype=float)
    global_max_moves = np.asarray([row["global_predicted_max_daily_move"] for row in rows], dtype=float)
    large = actual_max_moves >= 0.01
    extreme = actual_max_moves >= EXTREME_MIN_RETURN
    by_actual = {}
    for class_id, name in enumerate(REGIME_NAMES):
        subset = [row for row in rows if row["actual_regime"] == class_id]
        by_actual[name] = {
            "samples": len(subset),
            "routed_mae_points": _safe_mae(
                [x["predicted_close"] for x in subset],
                [x["actual_close"] for x in subset],
            ),
            "global_ridge_mae_points": _safe_mae(
                [x["global_predicted_close"] for x in subset],
                [x["actual_close"] for x in subset],
            ),
        }
    used = {}
    by_expert = {}
    for row in rows:
        expert_name = row["expert_used"]
        used[expert_name] = used.get(expert_name, 0) + 1
    for expert_name in sorted(used):
        subset = [row for row in rows if row["expert_used"] == expert_name]
        by_expert[expert_name] = {
            "samples": len(subset),
            "routed_mae_points": _safe_mae(
                [x["predicted_close"] for x in subset],
                [x["actual_close"] for x in subset],
            ),
            "global_ridge_mae_points": _safe_mae(
                [x["global_predicted_close"] for x in subset],
                [x["actual_close"] for x in subset],
            ),
            "routed_direction_accuracy": round(float(np.mean([
                np.sign(x["predicted_return"]) == np.sign(x["actual_return"])
                for x in subset
            ])), 4) if subset else None,
        }
    return {
        "status": "available",
        "oos_samples": int(len(rows)),
        "oos_start_date": min((row["forecast_asof"] for row in rows if row.get("forecast_asof")), default=None),
        "oos_end_date": max((row["forecast_asof"] for row in rows if row.get("forecast_asof")), default=None),
        "routed_mae_points": _safe_mae(predicted, actual),
        "global_ridge_mae_points": _safe_mae(global_predicted, actual),
        "routed_direction_accuracy": round(float(np.mean(np.sign(predicted_returns) == np.sign(actual_returns))), 4),
        "large_move_samples": int(large.sum()),
        "large_move_routed_mae_points": _safe_mae(predicted[large], actual[large]),
        "large_move_global_mae_points": _safe_mae(global_predicted[large], actual[large]),
        "max_daily_move_mae_pct": _safe_mae(predicted_max_moves * 100.0, actual_max_moves * 100.0),
        "global_max_daily_move_mae_pct": _safe_mae(global_max_moves * 100.0, actual_max_moves * 100.0),
        "large_move_max_daily_move_mae_pct": _safe_mae(predicted_max_moves[large] * 100.0, actual_max_moves[large] * 100.0),
        "extreme_move_samples": int(extreme.sum()),
        "extreme_move_routed_mae_points": _safe_mae(predicted[extreme], actual[extreme]),
        "by_actual_regime": by_actual,
        "by_selected_expert": by_expert,
        "expert_usage": used,
        "point_model_usage": {
            name: sum(row.get("point_model_used") == name for row in rows)
            for name in sorted({row.get("point_model_used", "unknown") for row in rows})
        },
        "magnitude_model_usage": {
            name: sum(row.get("magnitude_model_used") == name for row in rows)
            for name in sorted({row.get("magnitude_model_used", "unknown") for row in rows})
        },
    }


def build_regime_router_forecasts(features_by_horizon, close, target_meta=None, market_data=None, post_close_features_lagged=False, allowed_sample_index_by_horizon=None, include_gate_sensitivity=True):
    """Train and evaluate an independent router + four specialists per horizon.

    features_by_horizon maps horizon -> (feature_frame, cumulative forward-return series).
    Regimes use the maximum absolute one-day return in the next h sessions.
    OOS folds are expanding-window; the last h labelled observations are purged.
    This module is research-only and does not replace the production Ridge forecast.
    """
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    target_meta = target_meta or {}
    sessions = []
    for horizon in (1, 2, 3):
        if horizon not in features_by_horizon:
            sessions.append({
                "horizon_sessions": horizon, "status": "insufficient",
                "reason": "缺少該期限的特徵資料",
            })
            continue
        X, forward_returns = features_by_horizon[horizon]
        X = _lag_base_cross_market_features(X)
        if not post_close_features_lagged:
            X = _lag_post_close_features(X)
        X = _augment_router_features(X, market_data)
        forward_returns = pd.to_numeric(forward_returns, errors="coerce").reindex(X.index)
        numeric_features = X.apply(pd.to_numeric, errors="coerce")
        valid_features = pd.Series(
            np.isfinite(numeric_features.to_numpy(dtype=float)).all(axis=1), index=X.index
        )
        if allowed_sample_index_by_horizon and horizon in allowed_sample_index_by_horizon:
            allowed_dates = pd.DatetimeIndex(
                allowed_sample_index_by_horizon[horizon]
            ).tz_localize(None).normalize()
            valid_features &= pd.Series(X.index.isin(allowed_dates), index=X.index)
        latest_candidates = X.index[valid_features & close.reindex(X.index).notna()]
        if not len(latest_candidates):
            sessions.append({
                "horizon_sessions": horizon, "status": "insufficient",
                "reason": "沒有完整的最新特徵資料",
            })
            continue
        latest = latest_candidates[-1]
        future_max_daily_move = _future_max_daily_move(close, horizon).reindex(X.index)
        train_idx = X.index[
            valid_features & forward_returns.notna() & future_max_daily_move.notna() & (X.index < latest)
        ]
        if len(train_idx) < MIN_TRAIN_SAMPLES:
            sessions.append({
                "horizon_sessions": horizon, "status": "insufficient",
                "asof": str(latest.date()),
                "training_samples": int(len(train_idx)),
                "minimum_samples": MIN_TRAIN_SAMPLES,
                "reason": "分流器有效訓練樣本不足",
            })
            continue

        oos_actual, oos_predicted, oos_probabilities, oos_prior_probabilities, oos_forecasts = [], [], [], [], []
        calibration_records = []
        calibrated_oos_samples = 0
        first_test = min(max(300, horizon + 30), max(horizon + 30, len(train_idx) // 2))
        # Evaluate the most recent 2,000 labelled origins (or all available
        # origins for shorter histories), while retaining an expanding train set.
        # Refit every 60 sessions to keep scheduled refresh cost bounded.
        oos_start = max(first_test, len(train_idx) - 2000)
        for start in range(oos_start, len(train_idx), 60):
            test_idx = train_idx[start:start + 60]
            if not len(test_idx):
                continue
            fit_idx = _purged_training_indices(
                X.index, train_idx[:start], test_idx[0], horizon
            )
            if len(fit_idx) < MIN_TRAIN_SAMPLES:
                continue
            fit_regimes = classify_regimes(future_max_daily_move.loc[fit_idx]).astype(int)
            prior_regime_probs = _prior_regime_probabilities(fit_regimes, smoothing=1.0)
            router = _fit_router(X.loc[fit_idx], fit_regimes)
            fallback_class = int(pd.Series(fit_regimes).mode().iloc[0])
            raw_probs = _router_probabilities(router, X.loc[test_idx], fallback_class=fallback_class)
            eligible_records = _eligible_calibration_records(
                calibration_records, X.index.get_loc(test_idx[0]), horizon
            )
            calibrator = _fit_probability_calibrator(
                [record[1] for record in eligible_records],
                [record[3] for record in eligible_records],
            )
            probs = _apply_probability_calibrator(calibrator, raw_probs)
            calibrated_records = [record for record in eligible_records if record[4]]
            decision_thresholds = _select_risk_thresholds(
                [record[2] for record in calibrated_records],
                [record[3] for record in calibrated_records],
            )
            predicted_classes = np.asarray(
                [_route_regime(row, decision_thresholds) for row in probs], dtype=int
            )
            batch_calibration_records = []
            global_model, experts, global_magnitude_model, magnitude_experts, expert_counts = _fit_experts(
                X.loc[fit_idx],
                forward_returns.loc[fit_idx].to_numpy(dtype=float),
                future_max_daily_move.loc[fit_idx].to_numpy(dtype=float),
                fit_regimes,
            )
            global_returns = global_model.predict(X.loc[test_idx])
            global_max_moves = np.maximum(0.0, global_magnitude_model.predict(X.loc[test_idx]))
            batch_position = X.index.get_loc(test_idx[0])
            point_blend_by_expert = {
                name: _expert_performance_blend_weight(
                    oos_forecasts, name, batch_position, horizon,
                    "raw_expert_predicted_close", "global_predicted_close", "actual_close",
                )
                for name in (*REGIME_NAMES, "global_fallback")
            }
            magnitude_blend_by_expert = {
                name: _expert_performance_blend_weight(
                    oos_forecasts, name, batch_position, horizon,
                    "raw_expert_predicted_max_daily_move",
                    "global_predicted_max_daily_move",
                    "actual_max_daily_move",
                )
                for name in (*REGIME_NAMES, "global_fallback")
            }
            base_points = close.reindex(test_idx).to_numpy(dtype=float)
            actual_points = close.shift(-horizon).reindex(test_idx).to_numpy(dtype=float)
            actual_returns = forward_returns.reindex(test_idx).to_numpy(dtype=float)
            actual_max_moves = future_max_daily_move.reindex(test_idx).to_numpy(dtype=float)
            for offset, asof_date in enumerate(test_idx):
                requested = int(predicted_classes[offset])
                expert, expert_used, expert_samples, fallback_reason = _select_expert(
                    requested, global_model, experts, expert_counts
                )
                magnitude_expert, magnitude_expert_used, _, _ = _select_expert(
                    requested, global_magnitude_model, magnitude_experts, expert_counts
                )
                raw_expert_return = float(expert.predict(X.loc[[asof_date]])[0])
                raw_expert_max_move = max(
                    0.0, float(magnitude_expert.predict(X.loc[[asof_date]])[0])
                )
                global_return = float(global_returns[offset])
                global_max_move = float(global_max_moves[offset])
                point_gate = point_blend_by_expert.get(expert_used, {"weight": 0.0})
                magnitude_gate = magnitude_blend_by_expert.get(
                    magnitude_expert_used, {"weight": 0.0}
                )
                point_weight = float(point_gate.get("weight", 0.0))
                magnitude_weight = float(magnitude_gate.get("weight", 0.0))
                predicted_return = (
                    point_weight * raw_expert_return
                    + (1.0 - point_weight) * global_return
                )
                predicted_max_move = max(
                    0.0,
                    magnitude_weight * raw_expert_max_move
                    + (1.0 - magnitude_weight) * global_max_move,
                )
                base_point = float(base_points[offset])
                actual_point = float(actual_points[offset])
                actual_return = float(actual_returns[offset])
                actual_max_move = float(actual_max_moves[offset])
                if not all(np.isfinite(x) for x in (base_point, actual_point, actual_return, actual_max_move, predicted_return, predicted_max_move)):
                    continue
                if calibrator is not None:
                    calibrated_oos_samples += 1
                actual_class = int(classify_regimes([actual_max_move])[0])
                oos_actual.append(actual_class)
                oos_predicted.append(requested)
                oos_probabilities.append(probs[offset].tolist())
                oos_prior_probabilities.append(prior_regime_probs.tolist())
                batch_calibration_records.append((
                    X.index.get_loc(asof_date), raw_probs[offset].tolist(),
                    probs[offset].tolist(), actual_class, calibrator is not None,
                ))
                oos_forecasts.append({
                    "forecast_asof": str(asof_date.date()),
                    "forecast_position": int(X.index.get_loc(asof_date)),
                    "oos_block_start_position": int(batch_position),
                    "requested_expert": expert_used,
                    "point_model_used": "expert_blend" if point_weight > 0 else "global_ridge",
                    "point_expert_blend_weight": point_weight,
                    "magnitude_model_used": "expert_blend" if magnitude_weight > 0 else "global_ridge_magnitude",
                    "magnitude_expert_blend_weight": magnitude_weight,
                    "raw_expert_predicted_close": base_point * (1.0 + raw_expert_return),
                    "raw_expert_predicted_max_daily_move": raw_expert_max_move,
                    "predicted_close": base_point * (1.0 + predicted_return),
                    "actual_close": actual_point,
                    "base_close": base_point,
                    "predicted_return": predicted_return,
                    "actual_return": actual_return,
                    "global_predicted_close": base_point * (1.0 + global_return),
                    "predicted_max_daily_move": predicted_max_move,
                    "actual_max_daily_move": actual_max_move,
                    "global_predicted_max_daily_move": global_max_move,
                    "actual_regime": actual_class,
                    "requested_regime": REGIME_NAMES[requested],
                    "expert_used": expert_used,
                })
            # Outcomes from this block are retained with target timing; later
            # folds may calibrate only after each outcome's target close is known.
            calibration_records.extend(batch_calibration_records)

        # The latest feature origin can lag the latest available close when
        # optional inputs are missing. Purge labels relative to the actual
        # forecast origin, not merely relative to the end of the price series.
        final_train_idx = _purged_training_indices(
            X.index, train_idx, latest, horizon
        )
        final_regimes = classify_regimes(
            future_max_daily_move.loc[final_train_idx]
        ).astype(int)
        router = _fit_router(X.loc[final_train_idx], final_regimes)
        fallback_class = int(pd.Series(final_regimes).mode().iloc[0])
        latest_raw_probs = _router_probabilities(router, X.loc[[latest]], fallback_class=fallback_class)
        latest_position = X.index.get_loc(latest)
        eligible_records = _eligible_calibration_records(
            calibration_records, latest_position, horizon
        )
        final_calibrator = _fit_probability_calibrator(
            [record[1] for record in eligible_records],
            [record[3] for record in eligible_records],
        )
        latest_probs = _apply_probability_calibrator(final_calibrator, latest_raw_probs)[0]
        calibrated_records = [record for record in eligible_records if record[4]]
        final_decision_thresholds = _select_risk_thresholds(
            [record[2] for record in calibrated_records],
            [record[3] for record in calibrated_records],
        )
        requested = _route_regime(latest_probs, final_decision_thresholds)
        global_model, experts, global_magnitude_model, magnitude_experts, expert_counts = _fit_experts(
            X.loc[final_train_idx],
            forward_returns.loc[final_train_idx].to_numpy(dtype=float),
            future_max_daily_move.loc[final_train_idx].to_numpy(dtype=float),
            final_regimes,
        )
        expert, expert_used, expert_samples, fallback_reason = _select_expert(
            requested, global_model, experts, expert_counts
        )
        magnitude_expert, magnitude_expert_used, _, _ = _select_expert(
            requested, global_magnitude_model, magnitude_experts, expert_counts
        )
        raw_expert_return = float(expert.predict(X.loc[[latest]])[0])
        raw_expert_max_move = max(0.0, float(magnitude_expert.predict(X.loc[[latest]])[0]))
        global_return = float(global_model.predict(X.loc[[latest]])[0])
        global_max_move = max(0.0, float(global_magnitude_model.predict(X.loc[[latest]])[0]))
        point_gate = _expert_performance_blend_weight(
            oos_forecasts, expert_used, latest_position, horizon,
            "raw_expert_predicted_close", "global_predicted_close", "actual_close",
        )
        magnitude_gate = _expert_performance_blend_weight(
            oos_forecasts, magnitude_expert_used, latest_position, horizon,
            "raw_expert_predicted_max_daily_move",
            "global_predicted_max_daily_move",
            "actual_max_daily_move",
        )
        point_weight = float(point_gate.get("weight", 0.0))
        magnitude_weight = float(magnitude_gate.get("weight", 0.0))
        predicted_return = point_weight * raw_expert_return + (1.0 - point_weight) * global_return
        predicted_max_move = max(
            0.0,
            magnitude_weight * raw_expert_max_move + (1.0 - magnitude_weight) * global_max_move,
        )
        base_close = float(close.loc[latest])
        forecast_close = base_close * (1.0 + predicted_return)
        if point_weight <= 0:
            fallback_reason = (
                str(fallback_reason) + "；成熟 OOS 誤差未證明專家優於全域 Ridge，點位採全域 Ridge"
            )
        elif point_weight < 1:
            fallback_reason = (
                str(fallback_reason) + f"；點位依成熟 OOS MAE 收縮，專家權重 {point_weight:.2f}"
            )
        if magnitude_weight <= 0:
            fallback_reason = (
                str(fallback_reason) + "；幅度專家未優於全域基準，幅度採全域模型"
            )
        elif magnitude_weight < 1:
            fallback_reason = (
                str(fallback_reason) + f"；幅度依成熟 OOS MAE 收縮，專家權重 {magnitude_weight:.2f}"
            )
        if expert_used in REGIME_NAMES:
            selected_class = REGIME_NAMES.index(expert_used)
            selected_mask = final_regimes.to_numpy(dtype=int) == selected_class
            selected_returns = forward_returns.loc[final_train_idx].to_numpy(dtype=float)[selected_mask]
            selected_moves = future_max_daily_move.loc[final_train_idx].to_numpy(dtype=float)[selected_mask]
        else:
            selected_returns = forward_returns.loc[final_train_idx].to_numpy(dtype=float)
            selected_moves = future_max_daily_move.loc[final_train_idx].to_numpy(dtype=float)
        magnitude_profile = _historical_magnitude_profile(selected_returns, selected_moves, base_close)
        meta = target_meta.get(horizon, {})
        sessions.append({
            "horizon_sessions": horizon,
            "label": f"D+{horizon}",
            "status": "research",
            "asof": str(latest.date()),
            "target_date": meta.get("target_date"),
            "trading_calendar_source": meta.get("trading_calendar_source"),
            "base_close": round(base_close, 2),
            "forecast_close": round(float(forecast_close), 2),
            "predicted_return_pct": round(predicted_return * 100, 4),
            "predicted_max_daily_move_pct": round(predicted_max_move * 100, 4),
            "training_samples": int(len(final_train_idx)),
            "candidate_training_samples": int(len(train_idx)),
            "unmatured_training_samples_excluded": int(len(train_idx) - len(final_train_idx)),
             "latest_eligible_training_origin_date": str(
                X.index[max(0, latest_position - horizon)].date()
            ),
            "latest_included_training_origin_date": (
                str(final_train_idx[-1].date()) if len(final_train_idx) else None
            ),
            "regime": REGIME_NAMES[requested],
            "regime_label": REGIME_LABELS[REGIME_NAMES[requested]],
            "regime_probabilities": {
                name: round(float(latest_probs[class_id]), 5)
                for class_id, name in enumerate(REGIME_NAMES)
            },
            "router_decision_thresholds": final_decision_thresholds,
            "routing_reason": _routing_reason(latest_probs, final_decision_thresholds, requested),
            "probabilities_calibrated": final_calibrator is not None,
            "probability_calibration": {
                "method": "prequential_multinomial_logistic" if final_calibrator is not None else "not_available",
                "training_oos_samples": int(len(eligible_records)),
                "prequential_calibrated_oos_samples": int(calibrated_oos_samples),
                "oos_validation_fully_calibrated": bool(oos_actual and calibrated_oos_samples == len(oos_actual)),
                "oos_validation_calibration_coverage_pct": round(100.0 * calibrated_oos_samples / len(oos_actual), 2) if oos_actual else 0.0,
            },
            "expert_used": expert_used,
            "point_model_used": "expert_blend" if point_weight > 0 else "global_ridge",
            "point_expert_blend_weight": round(point_weight, 4),
            "point_expert_performance": point_gate,
            "magnitude_model_used": "expert_blend" if magnitude_weight > 0 else "global_ridge_magnitude",
            "magnitude_expert_blend_weight": round(magnitude_weight, 4),
            "magnitude_expert_performance": magnitude_gate,
            "expert_training_samples": int(expert_samples),
            "requested_expert_training_samples": int(expert_counts.get(requested, 0)),
            "fallback_reason": fallback_reason,
            "router_status": "available" if router is not None else "single_class_history",
            "router_training_regime_counts": {
                name: int(expert_counts.get(class_id, 0))
                for class_id, name in enumerate(REGIME_NAMES)
            },
            "router_validation": {
                **_router_metrics(oos_actual, oos_predicted, oos_probabilities, oos_prior_probabilities),
                "prequential_calibrated_oos_samples": int(calibrated_oos_samples),
                "probabilities_calibrated": bool(oos_actual and calibrated_oos_samples == len(oos_actual)),
                "calibration_coverage_pct": round(100.0 * calibrated_oos_samples / len(oos_actual), 2) if oos_actual else 0.0,
            },
            "routed_forecast_validation": _forecast_metrics(oos_forecasts),
            "gate_policy_sensitivity": (
                _gate_policy_sensitivity(oos_forecasts, horizon)
                if include_gate_sensitivity else {
                    "status": "skipped",
                    "reason": "僅對主要價格特徵組進行門檻敏感度與保留樣本檢驗",
                }
            ),
            "historical_magnitude_profile": magnitude_profile,
            "note": (
                "此為研究候選；分流依未來區間最大單日絕對漲跌幅，點位預測另為基準至 D+N 收盤的累積報酬。專家僅在成熟先前 OOS MAE 優於全域 Ridge 時參與點位與幅度輸出，否則自動回退或收縮至全域模型。"
                "分流機率使用先前可得的樣本外預測逐段校準；校準樣本不足時保留原始分流分數。"
                "極端專家樣本不足時會退回高波動或全域模型。"
            ),
        })

    return {
        "status": "research" if any(x.get("status") == "research" for x in sessions) else "insufficient",
        "model_version": "four-regime-router-moE-performance-gated-research-v2",
        "routing_priority": "router_first",
        "regime_definitions": {
            "low": {"max_future_max_daily_move_pct": 0.5},
            "normal": {"min_future_max_daily_move_pct_exclusive": 0.5, "max_future_max_daily_move_pct": 1.0},
            "high": {"min_future_max_daily_move_pct_exclusive": 1.0, "max_future_max_daily_move_pct_exclusive": 7.5},
            "extreme": {"min_future_max_daily_move_pct": 7.5},
            "regime_target_definition": "maximum absolute one-day close-to-close return within the next N trading sessions",
            "point_forecast_definition": "cumulative close-to-close return from forecast origin to the Nth session close",
        },
        "sessions": sessions,
        "probability_note": "使用先前樣本外分流預測進行逐段時間序列校準；校準樣本不足時保留原始分流分數並明確標示。Brier 與 log-loss 以樣本外資料計算。",
        "validation_method": "purged_expanding_walk_forward_oos_recent_2000_sessions",
        "feature_timing_policy": "Taiwan cash-close origin; non-Taiwan closes and Taiwan institutional, margin, and futures features lagged one Taiwan session; source frames are reindexed before lagging to prevent missing-date compression",
        "production_replacement": False,
    }
