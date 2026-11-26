"""Leakage-safe primitives for the final TAIEX OOS experiments.

This module contains pure, testable time-series transformations. It deliberately
does not fetch data, write storage, deploy models, or change production weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


FIXED_LIMITS = (0.005, 0.01, 0.075)
ATR_RATIO_LIMITS = (0.5, 1.0, 2.0)


def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing simple ATR divided by close; requires a complete window."""
    c = pd.to_numeric(close, errors="coerce").sort_index()
    h = pd.to_numeric(high, errors="coerce").reindex(c.index)
    l = pd.to_numeric(low, errors="coerce").reindex(c.index)
    previous = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - previous).abs(), (l - previous).abs()],
        axis=1,
    ).max(axis=1, skipna=False)
    return tr.rolling(int(window), min_periods=int(window)).mean().div(c.where(c > 0))


def classify_fixed_and_atr(
    future_max_abs_move: pd.Series,
    atr_pct: pd.Series,
    fixed_limits: tuple[float, float, float] = FIXED_LIMITS,
    atr_limits: tuple[float, float, float] = ATR_RATIO_LIMITS,
) -> pd.DataFrame:
    """Return fixed and ATR-standardized labels; ATR warmup falls back to fixed.

    Future movement is a target. ATR is trailing-only and known at the forecast
    origin. Labels are -1 when the future target itself is unavailable.
    """
    future = pd.to_numeric(future_max_abs_move, errors="coerce")
    scale = pd.to_numeric(atr_pct, errors="coerce").reindex(future.index)
    fixed = _classify(future, fixed_limits)
    ratio = future.div(scale.where(scale > 0))
    atr_labels = _classify(ratio, atr_limits)
    warmup = scale.isna() | (scale <= 0)
    atr_labels.loc[warmup] = fixed.loc[warmup]
    return pd.DataFrame({
        "fixed": fixed.astype("int64"),
        "atr": atr_labels.astype("int64"),
        "atr_warmup_fallback": warmup.astype(bool),
        "atr_pct": scale,
    }, index=future.index)


def _classify(values: pd.Series, limits: tuple[float, float, float]) -> pd.Series:
    low, normal, extreme = (float(x) for x in limits)
    labels = pd.Series(-1, index=values.index, dtype="int64")
    valid = values.notna() & np.isfinite(values)
    labels.loc[valid & (values <= low)] = 0
    labels.loc[valid & (values > low) & (values <= normal)] = 1
    labels.loc[valid & (values > normal) & (values < extreme)] = 2
    labels.loc[valid & (values >= extreme)] = 3
    return labels


def purged_prior_dates(
    full_trading_index: pd.DatetimeIndex,
    candidate_dates: pd.DatetimeIndex,
    first_test_date: pd.Timestamp,
    horizon_sessions: int,
) -> pd.DatetimeIndex:
    """Keep candidate origins whose complete forward labels mature by test start."""
    calendar = pd.DatetimeIndex(full_trading_index)
    candidates = pd.DatetimeIndex(candidate_dates)
    first_pos = calendar.get_loc(pd.Timestamp(first_test_date))
    positions = calendar.get_indexer(candidates)
    if (positions < 0).any():
        raise ValueError("candidate date is absent from the complete trading calendar")
    return candidates[positions + int(horizon_sessions) <= int(first_pos)]


def fit_expanding_platt(
    raw_oos: pd.Series,
    actual_oos: pd.Series,
    full_trading_index: pd.DatetimeIndex,
    horizon_sessions: int = 21,
    block_size: int = 21,
    min_samples: int = 300,
    min_per_class: int = 50,
    c_value: float = 1.0,
) -> pd.DataFrame:
    """Prequential Platt probabilities using only matured earlier OOS records.

    The input raw_oos must itself be generated walk-forward. A calibrator is
    refit at each test block from earlier OOS rows whose labels matured before
    that block. Insufficient history returns raw probabilities for that block
    and records an explicit status; no holdout is used for fitting.
    """
    raw = pd.to_numeric(raw_oos, errors="coerce").dropna().sort_index()
    actual = pd.to_numeric(actual_oos, errors="coerce").reindex(raw.index)
    valid = actual.notna() & raw.between(0, 1)
    raw, actual = raw.loc[valid], actual.loc[valid].astype(int)
    calendar = pd.DatetimeIndex(full_trading_index)
    if not raw.index.isin(calendar).all():
        raise ValueError("OOS prediction date missing from full trading calendar")

    output = []
    dates = pd.DatetimeIndex(raw.index)
    for start in range(0, len(dates), int(block_size)):
        test_dates = dates[start:start + int(block_size)]
        if not len(test_dates):
            continue
        eligible = purged_prior_dates(
            calendar, dates[:start], test_dates[0], horizon_sessions
        )
        train_y = actual.reindex(eligible).dropna().astype(int)
        train_x = raw.reindex(train_y.index)
        counts = train_y.value_counts()
        enough = (
            len(train_y) >= int(min_samples)
            and int(counts.get(0, 0)) >= int(min_per_class)
            and int(counts.get(1, 0)) >= int(min_per_class)
        )
        if enough:
            safe = np.clip(train_x.to_numpy(dtype=float), 1e-8, 1 - 1e-8)
            logits = np.log(safe / (1 - safe)).reshape(-1, 1)
            model = LogisticRegression(C=float(c_value), max_iter=500, random_state=37)
            model.fit(logits, train_y.to_numpy(dtype=int))
            test_safe = np.clip(raw.loc[test_dates].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
            test_logits = np.log(test_safe / (1 - test_safe)).reshape(-1, 1)
            calibrated = model.predict_proba(test_logits)[:, 1]
            status = "platt_fitted"
        else:
            calibrated = raw.loc[test_dates].to_numpy(dtype=float)
            model = None
            status = "raw_insufficient_matured_history"
        for date, probability in zip(test_dates, calibrated):
            output.append({
                "date": date,
                "raw_probability": float(raw.loc[date]),
                "probability": float(probability),
                "actual": int(actual.loc[date]),
                "calibration_status": status,
                "calibration_samples": int(len(train_y)),
                "calibration_fit_end": str(eligible.max().date()) if enough and len(eligible) else None,
                "calibrator_c": float(c_value) if enough else None,
            })
    return pd.DataFrame(output).set_index("date")


def binary_metrics(predictions: pd.DataFrame) -> dict:
    """Summarize prequential raw and calibrated predictions without refitting."""
    y = predictions["actual"].to_numpy(dtype=int)
    raw = np.clip(predictions["raw_probability"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
    calibrated = np.clip(predictions["probability"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
    baseline = np.full(len(y), float(y.mean()))
    brier = float(np.mean((calibrated - y) ** 2))
    base_brier = float(np.mean((baseline - y) ** 2))
    return {
        "samples": int(len(y)),
        "brier_score": brier,
        "baseline_brier_score": base_brier,
        "brier_skill": 1.0 - brier / base_brier if base_brier > 0 else None,
        "log_loss": float(log_loss(y, calibrated, labels=[0, 1])),
        "raw_brier_score": float(np.mean((raw - y) ** 2)),
        "direction_accuracy_pct": float(np.mean((calibrated >= 0.5) == (y == 1)) * 100),
        "raw_direction_accuracy_pct": float(np.mean((raw >= 0.5) == (y == 1)) * 100),
        "calibrated_blocks": int((predictions["calibration_status"] == "platt_fitted").sum()),
    }
