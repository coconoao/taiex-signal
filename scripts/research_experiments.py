"""Offline, point-in-time volatility and probability-calibration experiments.

All routines are read-only. Thresholds and calibrators are evaluated on historical
holdout data only; no model parameters or production decision rules are changed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stock_model_research import MIN_CALIBRATION, _walk_forward


def _future_max_abs_move(close: pd.Series, horizon: int = 3) -> pd.Series:
    daily = pd.to_numeric(close, errors="coerce").sort_index().pct_change(fill_method=None).abs()
    frame = pd.concat(
        [daily.shift(-step).rename(f"D+{step}") for step in range(1, int(horizon) + 1)],
        axis=1,
    )
    return frame.max(axis=1).where(frame.notna().all(axis=1))


def _classify_with_row_thresholds(values, low, normal, extreme):
    values = pd.Series(values, copy=False, dtype=float)
    low = pd.Series(low, index=values.index, dtype=float)
    normal = pd.Series(normal, index=values.index, dtype=float)
    extreme = pd.Series(extreme, index=values.index, dtype=float)
    valid = values.notna() & low.notna() & normal.notna() & extreme.notna()
    labels = pd.Series(pd.NA, index=values.index, dtype="Int64")
    labels.loc[valid & (values <= low)] = 0
    labels.loc[valid & (values > low) & (values <= normal)] = 1
    labels.loc[valid & (values > normal) & (values < extreme)] = 2
    labels.loc[valid & (values >= extreme)] = 3
    return labels


def _annual_rows(classes, threshold_set, threshold_columns=None):
    frame = pd.DataFrame({"year": classes.index.year, "regime": classes})
    for name, values in (threshold_columns or {}).items():
        frame[name] = pd.Series(values, index=classes.index)
    frame = frame.dropna(subset=["regime"])
    rows = []
    names = ("low", "normal", "high", "extreme")
    for year, group in frame.groupby("year"):
        counts = group["regime"].astype(int).value_counts()
        n = len(group)
        row = {
            "threshold_set": threshold_set,
            "year": int(year),
            "samples": int(n),
            **{f"{name}_n": int(counts.get(i, 0)) for i, name in enumerate(names)},
            **{f"{name}_pct": round(100 * counts.get(i, 0) / n, 2)
               for i, name in enumerate(names)},
        }
        for column in (threshold_columns or {}):
            row[f"{column}_median"] = round(float(group[column].median()), 6)
        rows.append(row)
    return rows


def annual_regime_experiments(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    current=(0.005, 0.01, 0.075),
    candidate=(0.005, 0.012, 0.025),
) -> list[dict]:
    """Compare fixed, trailing-quantile, realized-vol and ATR-scaled regimes.

    Rolling quantiles use the trailing 252 observations of historical 3-session
    maximum absolute daily moves, available at each origin. ATR and realized
    volatility use information through the forecast origin only.
    """
    px = pd.to_numeric(close, errors="coerce").sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[~px.index.duplicated(keep="last")]
    daily = px.pct_change(fill_method=None)
    future = _future_max_abs_move(px, 3)
    rows = []

    for label, limits in (("fixed_current", current), ("fixed_candidate", candidate)):
        low_limit, normal_limit, extreme_limit = limits
        low_series = pd.Series(low_limit, index=px.index)
        normal_series = pd.Series(normal_limit, index=px.index)
        extreme_series = pd.Series(extreme_limit, index=px.index)
        classes = _classify_with_row_thresholds(
            future, low_series, normal_series, extreme_series
        )
        rows.extend(_annual_rows(classes, label, {
            "low_threshold_pct": low_series * 100,
            "normal_threshold_pct": normal_series * 100,
            "extreme_threshold_pct": extreme_series * 100,
        }))

    # Compare the future three-session maximum against the trailing distribution
    # of historical three-session maxima. The rolling window ends at the origin.
    historical_three_day_max = pd.concat(
        [daily.abs().shift(offset).rename(str(offset)) for offset in range(3)],
        axis=1,
    ).max(axis=1).where(
        pd.concat([daily.abs().shift(offset) for offset in range(3)], axis=1).notna().all(axis=1)
    )
    rolling = historical_three_day_max.rolling(252, min_periods=126)
    q50 = rolling.quantile(0.50)
    q75 = rolling.quantile(0.75)
    q95 = rolling.quantile(0.95)
    quantile_classes = _classify_with_row_thresholds(future, q50, q75, q95)
    rows.extend(_annual_rows(quantile_classes, "rolling_quantile_252", {
        "low_threshold_pct": q50 * 100,
        "normal_threshold_pct": q75 * 100,
        "extreme_threshold_pct": q95 * 100,
    }))

    # Close-to-close realized volatility is used when a reliable OHLC history is
    # unavailable. These are standardized units, not percentage-point limits.
    rv20 = daily.rolling(20, min_periods=15).std()
    rv_ratio = future / rv20.replace(0, np.nan)
    rv_limits = (0.75, 1.25, 2.0)
    rv_classes = _classify_with_row_thresholds(
        rv_ratio,
        pd.Series(rv_limits[0], index=px.index),
        pd.Series(rv_limits[1], index=px.index),
        pd.Series(rv_limits[2], index=px.index),
    )
    rows.extend(_annual_rows(rv_classes, "realized_vol_20d_standardized", {
        "low_threshold_atr_units": pd.Series(rv_limits[0], index=px.index),
        "normal_threshold_atr_units": pd.Series(rv_limits[1], index=px.index),
        "extreme_threshold_atr_units": pd.Series(rv_limits[2], index=px.index),
    }))

    if high is not None and low is not None:
        hi = pd.to_numeric(high, errors="coerce").reindex(px.index)
        lo = pd.to_numeric(low, errors="coerce").reindex(px.index)
        previous = px.shift(1)
        true_range = pd.concat(
            [(hi - lo).abs(), (hi - previous).abs(), (lo - previous).abs()],
            axis=1,
        ).max(axis=1).where(pd.concat(
            [(hi - lo).notna(), (hi - previous).notna(), (lo - previous).notna()],
            axis=1,
        ).all(axis=1))
        atr_pct = true_range.rolling(20, min_periods=15).mean() / px
        atr_ratio = future / atr_pct.replace(0, np.nan)
        atr_limits = (0.5, 1.0, 2.0)
        atr_classes = _classify_with_row_thresholds(
            atr_ratio,
            pd.Series(atr_limits[0], index=px.index),
            pd.Series(atr_limits[1], index=px.index),
            pd.Series(atr_limits[2], index=px.index),
        )
        rows.extend(_annual_rows(atr_classes, "atr_20d_standardized", {
            "low_threshold_atr_units": pd.Series(atr_limits[0], index=px.index),
            "normal_threshold_atr_units": pd.Series(atr_limits[1], index=px.index),
            "extreme_threshold_atr_units": pd.Series(atr_limits[2], index=px.index),
        }))
    return rows


def _matured_development_dates(walk_forward, feature_index, horizon):
    """Development OOS labels known before the first untouched holdout origin."""
    development = pd.DatetimeIndex(walk_forward["development_index"])
    holdout = pd.DatetimeIndex(walk_forward["holdout_index"])
    if not len(development) or not len(holdout):
        return pd.DatetimeIndex([])
    index = pd.DatetimeIndex(feature_index)
    positions = pd.Series(np.arange(len(index)), index=index)
    first_holdout_position = int(positions.loc[holdout.min()])
    eligible = [
        day for day in development
        if int(positions.loc[day]) + int(horizon) <= first_holdout_position
    ]
    return pd.DatetimeIndex(eligible)


def _probability_metrics(probability, actual, baseline):
    p = np.asarray(probability, dtype=float)
    y = np.asarray(actual, dtype=int)
    b = np.asarray(baseline, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    baseline_brier = float(np.mean((b - y) ** 2))
    return {
        "samples": int(len(y)),
        "brier_score": round(brier, 6),
        "baseline_brier_score": round(baseline_brier, 6),
        "brier_skill": round(1.0 - brier / baseline_brier, 6) if baseline_brier > 0 else None,
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 6),
        "baseline_log_loss": round(float(log_loss(y, b, labels=[0, 1])), 6),
        "direction_accuracy_pct": round(float(np.mean((p >= 0.5) == (y == 1)) * 100), 2),
        "baseline_direction_accuracy_pct": round(float(np.mean((b >= 0.5) == (y == 1)) * 100), 2),
    }


def _fit_platt(raw, labels):
    raw = np.asarray(raw, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(raw) < MIN_CALIBRATION or len(np.unique(labels)) < 2:
        return None
    # Clipping here is solely for a finite logit transform; calibrated outputs
    # are not clipped or otherwise forced into an artificial probability band.
    safe = np.clip(raw, 1e-8, 1.0 - 1e-8)
    logits = np.log(safe / (1.0 - safe)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=37)
    calibrator.fit(logits, labels)
    return calibrator


def _fit_isotonic(raw, labels):
    raw = np.asarray(raw, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(raw) < MIN_CALIBRATION or len(np.unique(labels)) < 2 or len(np.unique(raw)) < 10:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw, labels)
    return calibrator


def _apply_platt(calibrator, raw):
    values = np.asarray(raw, dtype=float)
    safe = np.clip(values, 1e-8, 1.0 - 1e-8)
    logits = np.log(safe / (1.0 - safe)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _annual_probability_metrics(probability, actual, baseline):
    p = pd.Series(np.asarray(probability, dtype=float), index=actual.index)
    rows = {}
    for year, dates in actual.groupby(actual.index.year).groups.items():
        if len(dates) < 20:
            continue
        rows[str(year)] = _probability_metrics(
            p.loc[dates], actual.loc[dates], baseline.loc[dates]
        )
    return rows


def probability_calibration_experiment(features: pd.DataFrame, close: pd.Series, horizon: int):
    """Evaluate scaled/unscaled logistic C grid and Platt/isotonic calibration.

    Calibration is fitted only on matured development OOS predictions. The
    chronological holdout is used once for reporting and never for fitting,
    threshold selection, or choosing a regularization setting.
    """
    from stock_model_research import STEP

    forward_return = close.reindex(features.index).shift(-int(horizon)) / close.reindex(features.index) - 1
    labels = (forward_return > 0).where(forward_return.notna()).astype(float)
    valid = features.notna().all(axis=1) & labels.notna()
    configs = [
        {"scaling": scaling, "C": c}
        for scaling in ("standard", "none")
        for c in (0.01, 0.1, 1.0, 10.0)
    ]
    outputs = []
    for config in configs:
        c = float(config["C"])
        scaling = config["scaling"]

        def factory(c=c, scaling=scaling):
            estimator = LogisticRegression(C=c, max_iter=700, solver="lbfgs", random_state=41)
            return make_pipeline(StandardScaler(), estimator) if scaling == "standard" else estimator

        wf = _walk_forward(features.loc[valid], labels.loc[valid], int(horizon), factory)
        if wf is None or not len(wf["holdout_index"]):
            outputs.append({
                **config, "status": "insufficient",
                "reason": "not enough chronological OOS predictions for a separated holdout",
            })
            continue

        hold = pd.DatetimeIndex(wf["holdout_index"])
        dev_dates = _matured_development_dates(wf, features.index, int(horizon))
        dev_raw = wf["raw"].reindex(dev_dates).dropna()
        dev_actual = wf["actual"].reindex(dev_raw.index).astype(int)
        hold_raw = wf["raw"].reindex(hold).astype(float)
        hold_actual = wf["actual"].reindex(hold).astype(int)
        hold_baseline = wf["baseline"].reindex(hold).astype(float)

        raw_metrics = _probability_metrics(hold_raw, hold_actual, hold_baseline)
        methods = {
            "raw": {
                "status": "available",
                "holdout_metrics": raw_metrics,
                "annual_holdout_metrics": _annual_probability_metrics(hold_raw, hold_actual, hold_baseline),
                "calibration_samples": 0,
                "calibration_fit_end": None,
            }
        }

        platt = _fit_platt(dev_raw.to_numpy(), dev_actual.to_numpy()) if len(dev_raw) else None
        if platt is not None:
            calibrated = _apply_platt(platt, hold_raw.to_numpy())
            calibrated_series = pd.Series(calibrated, index=hold)
            methods["platt_scaling"] = {
                "status": "available",
                "holdout_metrics": _probability_metrics(calibrated, hold_actual, hold_baseline),
                "annual_holdout_metrics": _annual_probability_metrics(calibrated_series, hold_actual, hold_baseline),
                "calibration_samples": int(len(dev_raw)),
                "calibration_fit_end": str(dev_raw.index.max().date()),
            }
        else:
            methods["platt_scaling"] = {
                "status": "insufficient_matured_development_oos",
                "calibration_samples": int(len(dev_raw)),
            }

        isotonic = _fit_isotonic(dev_raw.to_numpy(), dev_actual.to_numpy()) if len(dev_raw) else None
        if isotonic is not None:
            calibrated = isotonic.predict(hold_raw.to_numpy())
            calibrated_series = pd.Series(calibrated, index=hold)
            methods["isotonic_regression"] = {
                "status": "available",
                "holdout_metrics": _probability_metrics(calibrated, hold_actual, hold_baseline),
                "annual_holdout_metrics": _annual_probability_metrics(calibrated_series, hold_actual, hold_baseline),
                "calibration_samples": int(len(dev_raw)),
                "calibration_fit_end": str(dev_raw.index.max().date()),
            }
        else:
            methods["isotonic_regression"] = {
                "status": "insufficient_matured_development_oos",
                "calibration_samples": int(len(dev_raw)),
            }

        outputs.append({
            **config,
            "status": "available",
            "holdout_start": str(hold.min().date()),
            "holdout_end": str(hold.max().date()),
            "holdout_samples": int(len(hold)),
            "calibration_training_samples": int(len(dev_raw)),
            "methods": methods,
        })
    return {
        "status": "available" if any(row.get("status") == "available" for row in outputs) else "insufficient",
        "horizon_sessions": int(horizon),
        "evaluation": "fixed chronological 20% holdout; all calibrators fit only on matured development OOS predictions",
        "regularization_grid": [0.01, 0.1, 1.0, 10.0],
        "feature_scaling_variants": ["standard", "none"],
        "experiments": outputs,
    }
