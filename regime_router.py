"""Research-only four-regime router and specialist forecasts for TAIEX.

All targets are forward close-to-close returns. Features are supplied by
pipeline.build() and must be available as of each forecast origin. Walk-forward
folds purge the final h labelled rows before each test block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
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
# Initial research boundaries, applied to absolute cumulative close-to-close
# returns for each horizon. The 7.5% extreme threshold is user-specified.
LOW_MAX_RETURN = 0.005
NORMAL_MAX_RETURN = 0.01
EXTREME_MIN_RETURN = 0.075
MIN_TRAIN_SAMPLES = 180
MIN_EXPERT_SAMPLES = {0: 60, 1: 60, 2: 45, 3: 40}
EXPERT_ALPHA = {0: 25.0, 1: 15.0, 2: 5.0, 3: 1.0}
CLASS_IDS = (0, 1, 2, 3)


def classify_regimes(forward_returns):
    """Map signed forward returns to mutually exclusive absolute-volatility regimes."""
    values = np.asarray(forward_returns, dtype=float)
    magnitude = np.abs(values)
    labels = np.full(values.shape, -1, dtype=int)
    valid = np.isfinite(values)
    labels[valid & (magnitude <= LOW_MAX_RETURN)] = 0
    labels[valid & (magnitude > LOW_MAX_RETURN) & (magnitude <= NORMAL_MAX_RETURN)] = 1
    labels[valid & (magnitude > NORMAL_MAX_RETURN) & (magnitude < EXTREME_MIN_RETURN)] = 2
    labels[valid & (magnitude >= EXTREME_MIN_RETURN)] = 3
    if isinstance(forward_returns, pd.Series):
        return pd.Series(labels, index=forward_returns.index, name="regime")
    return labels


def _fit_router(X, y):
    classes = np.sort(pd.Series(y).dropna().astype(int).unique())
    if len(classes) < 2:
        return None
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.35, max_iter=800, random_state=17),
    )
    estimator.fit(X, np.asarray(y, dtype=int))
    return estimator


def _router_probabilities(estimator, X, fallback_class=None):
    probabilities = np.zeros((len(X), 4), dtype=float)
    if estimator is None:
        if fallback_class is not None:
            probabilities[:, int(fallback_class)] = 1.0
        else:
            probabilities[:, 1] = 1.0
        return probabilities
    raw = estimator.predict_proba(X)
    for column, class_id in enumerate(estimator[-1].classes_):
        probabilities[:, int(class_id)] = raw[:, column]
    row_sums = probabilities.sum(axis=1, keepdims=True)
    return np.divide(probabilities, row_sums, out=np.zeros_like(probabilities), where=row_sums > 0)


def _fit_experts(X, returns, regimes):
    global_model = make_pipeline(StandardScaler(), Ridge(alpha=25.0))
    global_model.fit(X, np.asarray(returns, dtype=float))
    models = {}
    counts = {}
    for class_id in CLASS_IDS:
        mask = np.asarray(regimes, dtype=int) == class_id
        count = int(mask.sum())
        counts[class_id] = count
        if count >= MIN_EXPERT_SAMPLES[class_id]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=EXPERT_ALPHA[class_id]))
            model.fit(X.iloc[np.flatnonzero(mask)], np.asarray(returns, dtype=float)[mask])
            models[class_id] = model
    return global_model, models, counts


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


def _router_metrics(actual_classes, predicted_classes, probabilities):
    if not actual_classes:
        return {"status": "insufficient", "oos_samples": 0}
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
    high_missed = high_mask & (predicted < 2)
    one_hot = np.eye(4, dtype=float)[actual]
    clipped = np.clip(probs, 1e-8, 1.0)
    return {
        "status": "available",
        "oos_samples": int(len(actual)),
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
        "macro_f1": round(float(f1_score(actual, predicted, labels=CLASS_IDS, average="macro", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "high_extreme_recall": round(float(np.mean(predicted[high_mask] >= 2)), 4) if high_mask.any() else None,
        "high_extreme_missed": int(high_missed.sum()),
        "high_extreme_missed_rate": round(float(high_missed.sum() / high_mask.sum()), 4) if high_mask.any() else None,
        "multiclass_brier": round(float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))), 5),
        "log_loss": round(float(log_loss(actual, clipped, labels=list(CLASS_IDS))), 5),
        "probabilities_calibrated": False,
    }


def _forecast_metrics(rows):
    if not rows:
        return {"status": "insufficient", "oos_samples": 0}
    predicted = np.asarray([row["predicted_close"] for row in rows], dtype=float)
    actual = np.asarray([row["actual_close"] for row in rows], dtype=float)
    base = np.asarray([row["base_close"] for row in rows], dtype=float)
    actual_returns = np.asarray([row["actual_return"] for row in rows], dtype=float)
    predicted_returns = np.asarray([row["predicted_return"] for row in rows], dtype=float)
    global_predicted = np.asarray([row["global_predicted_close"] for row in rows], dtype=float)
    large = np.abs(actual_returns) >= 0.01
    extreme = np.abs(actual_returns) >= EXTREME_MIN_RETURN
    by_actual = {}
    for class_id, name in enumerate(REGIME_NAMES):
        subset = [row for row in rows if row["actual_regime"] == class_id]
        by_actual[name] = {
            "samples": len(subset),
            "routed_mae_points": _safe_mae(
                [x["predicted_close"] for x in subset],
                [x["actual_close"] for x in subset],
            ),
        }
    used = {}
    for row in rows:
        used[row["expert_used"]] = used.get(row["expert_used"], 0) + 1
    return {
        "status": "available",
        "oos_samples": int(len(rows)),
        "routed_mae_points": _safe_mae(predicted, actual),
        "global_ridge_mae_points": _safe_mae(global_predicted, actual),
        "routed_direction_accuracy": round(float(np.mean(np.sign(predicted_returns) == np.sign(actual_returns))), 4),
        "large_move_samples": int(large.sum()),
        "large_move_routed_mae_points": _safe_mae(predicted[large], actual[large]),
        "large_move_global_mae_points": _safe_mae(global_predicted[large], actual[large]),
        "extreme_move_samples": int(extreme.sum()),
        "extreme_move_routed_mae_points": _safe_mae(predicted[extreme], actual[extreme]),
        "by_actual_regime": by_actual,
        "expert_usage": used,
    }


def build_regime_router_forecasts(features_by_horizon, close, target_meta=None):
    """Train and evaluate an independent router + four specialists per horizon.

    features_by_horizon maps horizon -> (feature_frame, forward_return_series).
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
        forward_returns = pd.to_numeric(forward_returns, errors="coerce").reindex(X.index)
        valid_features = X.notna().all(axis=1)
        latest_candidates = X.index[valid_features & close.reindex(X.index).notna()]
        if not len(latest_candidates):
            sessions.append({
                "horizon_sessions": horizon, "status": "insufficient",
                "reason": "沒有完整的最新特徵資料",
            })
            continue
        latest = latest_candidates[-1]
        train_idx = X.index[valid_features & forward_returns.notna() & (X.index < latest)]
        if len(train_idx) < MIN_TRAIN_SAMPLES:
            sessions.append({
                "horizon_sessions": horizon, "status": "insufficient",
                "asof": str(latest.date()),
                "training_samples": int(len(train_idx)),
                "minimum_samples": MIN_TRAIN_SAMPLES,
                "reason": "分流器有效訓練樣本不足",
            })
            continue

        train_regimes = classify_regimes(forward_returns.loc[train_idx]).astype(int)
        oos_actual, oos_predicted, oos_probabilities, oos_forecasts = [], [], [], []
        first_test = min(max(300, horizon + 30), max(horizon + 30, len(train_idx) // 2))
        for start in range(first_test, len(train_idx), 30):
            fit_idx = train_idx[:start - horizon + 1]
            test_idx = train_idx[start:start + 30]
            if len(fit_idx) < MIN_TRAIN_SAMPLES or not len(test_idx):
                continue
            fit_regimes = classify_regimes(forward_returns.loc[fit_idx]).astype(int)
            router = _fit_router(X.loc[fit_idx], fit_regimes)
            fallback_class = int(pd.Series(fit_regimes).mode().iloc[0])
            probs = _router_probabilities(router, X.loc[test_idx], fallback_class=fallback_class)
            predicted_classes = probs.argmax(axis=1)
            global_model, experts, expert_counts = _fit_experts(
                X.loc[fit_idx], forward_returns.loc[fit_idx].to_numpy(dtype=float), fit_regimes
            )
            global_returns = global_model.predict(X.loc[test_idx])
            base_points = close.reindex(test_idx).to_numpy(dtype=float)
            actual_points = close.shift(-horizon).reindex(test_idx).to_numpy(dtype=float)
            actual_returns = forward_returns.reindex(test_idx).to_numpy(dtype=float)
            for offset, asof_date in enumerate(test_idx):
                requested = int(predicted_classes[offset])
                expert, expert_used, expert_samples, fallback_reason = _select_expert(
                    requested, global_model, experts, expert_counts
                )
                predicted_return = float(expert.predict(X.loc[[asof_date]])[0])
                base_point = float(base_points[offset])
                actual_point = float(actual_points[offset])
                actual_return = float(actual_returns[offset])
                if not all(np.isfinite(x) for x in (base_point, actual_point, actual_return, predicted_return)):
                    continue
                oos_actual.append(int(classify_regimes([actual_return])[0]))
                oos_predicted.append(requested)
                oos_probabilities.append(probs[offset].tolist())
                oos_forecasts.append({
                    "predicted_close": base_point * (1.0 + predicted_return),
                    "actual_close": actual_point,
                    "base_close": base_point,
                    "predicted_return": predicted_return,
                    "actual_return": actual_return,
                    "global_predicted_close": base_point * (1.0 + float(global_returns[offset])),
                    "actual_regime": int(classify_regimes([actual_return])[0]),
                    "expert_used": expert_used,
                })

        final_regimes = classify_regimes(forward_returns.loc[train_idx]).astype(int)
        router = _fit_router(X.loc[train_idx], final_regimes)
        fallback_class = int(pd.Series(final_regimes).mode().iloc[0])
        latest_probs = _router_probabilities(router, X.loc[[latest]], fallback_class=fallback_class)[0]
        requested = int(np.argmax(latest_probs))
        global_model, experts, expert_counts = _fit_experts(
            X.loc[train_idx], forward_returns.loc[train_idx].to_numpy(dtype=float), final_regimes
        )
        expert, expert_used, expert_samples, fallback_reason = _select_expert(
            requested, global_model, experts, expert_counts
        )
        predicted_return = float(expert.predict(X.loc[[latest]])[0])
        base_close = float(close.loc[latest])
        forecast_close = base_close * (1.0 + predicted_return)
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
            "training_samples": int(len(train_idx)),
            "regime": REGIME_NAMES[requested],
            "regime_label": REGIME_LABELS[REGIME_NAMES[requested]],
            "regime_probabilities": {
                name: round(float(latest_probs[class_id]), 5)
                for class_id, name in enumerate(REGIME_NAMES)
            },
            "probabilities_calibrated": False,
            "expert_used": expert_used,
            "expert_training_samples": int(expert_samples),
            "requested_expert_training_samples": int(expert_counts.get(requested, 0)),
            "fallback_reason": fallback_reason,
            "router_status": "available" if router is not None else "single_class_history",
            "router_training_regime_counts": {
                name: int(expert_counts.get(class_id, 0))
                for class_id, name in enumerate(REGIME_NAMES)
            },
            "router_validation": _router_metrics(oos_actual, oos_predicted, oos_probabilities),
            "routed_forecast_validation": _forecast_metrics(oos_forecasts),
            "note": (
                "此為研究候選；D+N 報酬為基準收盤至第 N 個交易日收盤的累積報酬。"
                "分流機率尚未校準，極端專家樣本不足時會退回高波動或全域模型。"
            ),
        })

    return {
        "status": "research" if any(x.get("status") == "research" for x in sessions) else "insufficient",
        "model_version": "four-regime-router-moE-research-v1",
        "routing_priority": "router_first",
        "regime_definitions": {
            "low": {"max_abs_return_pct": 0.5},
            "normal": {"min_abs_return_pct_exclusive": 0.5, "max_abs_return_pct": 1.0},
            "high": {"min_abs_return_pct_exclusive": 1.0, "max_abs_return_pct_exclusive": 7.5},
            "extreme": {"min_abs_return_pct": 7.5},
            "return_definition": "absolute cumulative close-to-close return from forecast origin to each horizon close",
        },
        "sessions": sessions,
        "probability_note": "LogisticRegression 原始分流機率，尚未完成獨立時間序列校準；Brier 與 log-loss 僅作校準診斷。",
        "validation_method": "purged_expanding_walk_forward_oos",
        "production_replacement": False,
    }
