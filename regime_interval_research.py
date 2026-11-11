"""Research-only expanding-quantile regimes and rolling conformal intervals.

The functions are independent of production forecasting. Callers must supply
one horizon at a time and chronological, point-in-time OOS predictions.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

REGIME_NAMES = ("low", "normal", "high", "extreme")


def expanding_quantile_regimes(outcomes: Iterable[float], *, min_history: int = 252,
                               quantiles: tuple[float, float, float] = (0.25, 0.75, 0.95),
                               origin_positions: Iterable[int] | None = None,
                               horizon: int = 1) -> dict:
    """Label each outcome using quantiles from strictly earlier outcomes only."""
    values = np.asarray(list(outcomes), dtype=float)
    if min_history < 1:
        raise ValueError("min_history must be positive")
    if len(quantiles) != 3 or not (0 < quantiles[0] < quantiles[1] < quantiles[2] < 1):
        raise ValueError("quantiles must be three strictly increasing values in (0, 1)")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    positions = np.arange(len(values), dtype=int) if origin_positions is None else np.asarray(list(origin_positions), dtype=int)
    if positions.shape != values.shape or (len(positions) > 1 and np.any(np.diff(positions) <= 0)):
        raise ValueError("origin_positions must be strictly increasing and match outcomes")
    labels = np.full(len(values), -1, dtype=int)
    thresholds = np.full((len(values), 3), np.nan, dtype=float)
    history_counts = np.zeros(len(values), dtype=int)
    for i, value in enumerate(values):
        eligible = (np.arange(len(values)) < i) & (positions + int(horizon) <= positions[i])
        history = np.abs(values[eligible])
        history = history[np.isfinite(history)]
        history_counts[i] = len(history)
        if len(history) < min_history:
            continue
        q25, q75, q95 = np.quantile(history, quantiles, method="higher")
        thresholds[i] = (q25, q75, q95)
        # Thresholds are knowable at the forecast origin even when the
        # corresponding future outcome has not matured yet. Keep the label
        # unknown, but expose the latest point-in-time cutoffs for diagnostics.
        if not np.isfinite(value):
            continue
        magnitude = abs(float(value))
        labels[i] = 0 if magnitude <= q25 else 1 if magnitude <= q75 else 2 if magnitude <= q95 else 3
    return {
        "labels": labels, "thresholds": thresholds, "history_counts": history_counts,
        "quantiles": list(quantiles), "min_history": int(min_history),
        "threshold_policy": "expanding_prior_matured_outcomes_only",
    }


def _finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    ordered = ordered[np.isfinite(ordered)]
    if not len(ordered):
        return float("nan")
    rank = min(len(ordered), int(math.ceil((len(ordered) + 1) * coverage)))
    return float(ordered[max(0, rank - 1)])


def rolling_conformal_intervals(predictions: Iterable[float], actuals: Iterable[float], *,
                                horizon: int, origin_positions: Iterable[int] | None = None,
                                coverages: tuple[float, ...] = (0.50, 0.80, 0.90),
                                min_calibration_samples: int = 60,
                                regime_labels: Iterable[int] | None = None,
                                min_regime_samples: int = 80,
                                method: str = "absolute_residual") -> dict:
    """Create intervals from matured prior OOS residuals, separately per horizon.

    absolute_residual creates symmetric intervals from absolute errors.
    signed_residual creates asymmetric intervals from lower/upper residual
    quantiles. Both variants use only matured prior OOS forecasts.
    """
    pred = np.asarray(list(predictions), dtype=float)
    actual = np.asarray(list(actuals), dtype=float)
    if pred.shape != actual.shape:
        raise ValueError("predictions and actuals must have equal lengths")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if min_calibration_samples < 1 or min_regime_samples < 1:
        raise ValueError("minimum calibration sample counts must be positive")
    if method not in ("absolute_residual", "signed_residual"):
        raise ValueError("method must be absolute_residual or signed_residual")
    if not coverages or any(not 0 < c < 1 for c in coverages):
        raise ValueError("coverages must be values in (0, 1)")
    n = len(pred)
    positions = np.arange(n, dtype=int) if origin_positions is None else np.asarray(list(origin_positions), dtype=int)
    if positions.shape != pred.shape or (len(positions) > 1 and np.any(np.diff(positions) <= 0)):
        raise ValueError("origin_positions must be strictly increasing and match predictions")
    regimes = None if regime_labels is None else np.asarray(list(regime_labels), dtype=int)
    if regimes is not None and regimes.shape != pred.shape:
        raise ValueError("regime_labels must match predictions")
    residuals = actual - pred
    absolute_residuals = np.abs(residuals)
    rows = []
    for i in range(n):
        eligible = np.flatnonzero(np.isfinite(residuals) & (np.arange(n) < i)
                                  & (positions + int(horizon) <= positions[i]))
        selected = eligible
        source = "same_horizon_global_matured_oos"
        if regimes is not None and regimes[i] >= 0:
            conditional = eligible[regimes[eligible] == regimes[i]]
            if len(conditional) >= min_regime_samples:
                selected = conditional
                source = "same_regime_matured_oos"
            else:
                source = "same_horizon_global_fallback_insufficient_regime_samples"
        count = int(len(selected))
        row = {"index": i, "origin_position": int(positions[i]),
               "calibration_samples": count, "calibration_source": source, "intervals": {}}
        if count >= min_calibration_samples and np.isfinite(pred[i]):
            for coverage in coverages:
                key = str(int(round(float(coverage) * 100)))
                if method == "absolute_residual":
                    radius = _finite_sample_quantile(
                        absolute_residuals[selected], float(coverage)
                    )
                    row["intervals"][key] = {
                        "coverage": float(coverage), "lower": float(pred[i] - radius),
                        "upper": float(pred[i] + radius), "radius": float(radius),
                    }
                else:
                    tail_coverage = 1.0 - (1.0 - float(coverage)) / 2.0
                    lower_offset = -_finite_sample_quantile(
                        -residuals[selected], tail_coverage
                    )
                    upper_offset = _finite_sample_quantile(
                        residuals[selected], tail_coverage
                    )
                    row["intervals"][key] = {
                        "coverage": float(coverage),
                        "lower": float(pred[i] + lower_offset),
                        "upper": float(pred[i] + upper_offset),
                        "lower_offset": float(lower_offset),
                        "upper_offset": float(upper_offset),
                    }
        rows.append(row)
    return {
        "horizon_sessions": int(horizon), "nominal_coverages": [float(c) for c in coverages],
        "min_calibration_samples": int(min_calibration_samples),
        "min_regime_samples": int(min_regime_samples), "rows": rows,
        "method": method,
        "calibration_policy": f"rolling_{method}_conformal_matured_oos_only",
    }


def interval_metrics(interval_result: dict, actuals: Iterable[float]) -> dict:
    """Summarize empirical coverage, mean width and interval score by nominal level."""
    actual = np.asarray(list(actuals), dtype=float)
    summary = {}
    coverages = interval_result.get("nominal_coverages", [])
    for coverage in coverages:
        key = str(int(round(float(coverage) * 100)))
        scored = []
        for row in interval_result.get("rows", []):
            i = int(row["index"])
            interval = row.get("intervals", {}).get(key)
            if interval and i < len(actual) and np.isfinite(actual[i]):
                scored.append((float(actual[i]), float(interval["lower"]), float(interval["upper"])))
        if not scored:
            summary[key] = {"nominal_coverage": float(coverage), "empirical_coverage": None,
                            "mean_width": None, "mean_interval_score": None, "samples": 0}
            continue
        y = np.asarray([x[0] for x in scored])
        lower = np.asarray([x[1] for x in scored])
        upper = np.asarray([x[2] for x in scored])
        alpha = 1.0 - float(coverage)
        score = (upper - lower) + (2.0 / alpha) * (np.maximum(lower - y, 0.0) + np.maximum(y - upper, 0.0))
        summary[key] = {
            "nominal_coverage": float(coverage),
            "empirical_coverage": round(float(np.mean((y >= lower) & (y <= upper))), 4),
            "mean_width": round(float(np.mean(upper - lower)), 6),
            "mean_interval_score": round(float(np.mean(score)), 6), "samples": int(len(scored)),
        }
    return summary
