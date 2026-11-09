"""Per-stock medium/long-horizon research model.

Targets are positive total returns over 21/63/126 actual trading sessions.
Features are trailing-only price, volume and market-relative measures. The
module uses purged chronological walk-forward evaluation and never places
orders or modifies the production TAIEX model.
"""
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = {"1m": 21, "3m": 63, "6m": 126}
MIN_TRAIN = 500
MIN_CALIBRATION = 80
# Refit quarterly rather than monthly. Each fold still predicts a contiguous
# block of 63 sessions, with the same horizon-specific purge; this cuts repeated
# model fits substantially on the free Render CPU without dropping OOS dates.
STEP = 63
RANDOM_STATE = 42


def build_features(stock_close, market_close, volume=None):
    """Create point-in-time features; each row uses data available on that date."""
    px = pd.to_numeric(stock_close, errors="coerce").sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[~px.index.duplicated(keep="last")]
    market = pd.to_numeric(market_close, errors="coerce").sort_index()
    market.index = pd.to_datetime(market.index).tz_localize(None).normalize()
    market = market[~market.index.duplicated(keep="last")]
    frame = pd.concat([px.rename("stock"), market.rename("market")], axis=1).sort_index()
    ret = frame.pct_change(fill_method=None)
    out = pd.DataFrame(index=frame.index)

    for window in (1, 5, 21, 63, 126):
        out[f"stock_return_{window}"] = frame["stock"].pct_change(window, fill_method=None)
        out[f"market_return_{window}"] = frame["market"].pct_change(window, fill_method=None)
        out[f"relative_return_{window}"] = (
            out[f"stock_return_{window}"] - out[f"market_return_{window}"]
        )

    for window in (21, 63, 126):
        out[f"stock_volatility_{window}"] = ret["stock"].rolling(
            window, min_periods=max(10, window // 2)
        ).std() * np.sqrt(252)
        out[f"market_volatility_{window}"] = ret["market"].rolling(
            window, min_periods=max(10, window // 2)
        ).std() * np.sqrt(252)
        average = frame["stock"].rolling(window, min_periods=max(10, window // 2)).mean()
        out[f"price_to_sma_{window}"] = frame["stock"] / average - 1
        rolling_high = frame["stock"].rolling(window, min_periods=max(10, window // 2)).max()
        out[f"drawdown_{window}"] = frame["stock"] / rolling_high - 1

    delta = frame["stock"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    if volume is not None:
        vol = pd.to_numeric(volume, errors="coerce").copy()
        vol.index = pd.to_datetime(vol.index).tz_localize(None).normalize()
        vol = vol[~vol.index.duplicated(keep="last")].sort_index().reindex(frame.index)
        vol_mean = vol.rolling(20, min_periods=10).mean()
        out["volume_ratio_20"] = vol / vol_mean.replace(0, np.nan)
        out["volume_change_5"] = vol.pct_change(5, fill_method=None)

    return out.replace([np.inf, -np.inf], np.nan)


def _factories():
    return {
        "regularized_logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=700)
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=7, min_samples_leaf=30,
            learning_rate=0.04, l2_regularization=6.0,
            random_state=RANDOM_STATE,
        ),
    }


def _block_bootstrap_accuracy_ci(probability, actual, block_length, seed=RANDOM_STATE, draws=1000):
    """Moving-block bootstrap CI; preserves short-range dependence in overlapping labels."""
    p = np.asarray(probability, dtype=float)
    y = np.asarray(actual, dtype=int)
    n = len(y)
    if n < max(100, block_length * 4):
        return None
    correct = ((p >= 0.5) == (y == 1)).astype(float)
    rng = np.random.default_rng(seed)
    block = min(max(1, int(block_length)), max(1, n // 4))
    starts = np.arange(0, n - block + 1)
    estimates = np.empty(draws, dtype=float)
    blocks_needed = int(np.ceil(n / block))
    for i in range(draws):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([correct[start:start + block] for start in chosen])[:n]
        estimates[i] = sample.mean() * 100
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"lower_95_pct": round(float(low), 2), "upper_95_pct": round(float(high), 2),
            "method": "moving-block bootstrap", "block_length_sessions": int(block),
            "bootstrap_draws": int(draws)}


def _annual_metrics(probability, actual, baseline):
    frame = pd.DataFrame({
        "probability": np.asarray(probability, dtype=float),
        "actual": np.asarray(actual, dtype=int),
        "baseline": np.asarray(baseline, dtype=float),
    }, index=pd.DatetimeIndex(actual.index))
    output = {}
    for year, group in frame.groupby(frame.index.year):
        if len(group) < 20:
            continue
        output[str(year)] = _metrics(
            group["probability"], group["actual"], group["baseline"]
        )
    return output


def _metrics(probability, actual, baseline, horizon=21):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(actual, dtype=int)
    b = np.clip(np.asarray(baseline, dtype=float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    baseline_brier = float(np.mean((b - y) ** 2))
    return {
        "sample_count": int(len(y)),
        "direction_accuracy_pct": round(float(np.mean((p >= 0.5) == (y == 1)) * 100), 2),
        "baseline_direction_accuracy_pct": round(float(np.mean((b >= 0.5) == (y == 1)) * 100), 2),
        "accuracy_95pct_ci": _block_bootstrap_accuracy_ci(p, y, horizon),
        "brier": round(brier, 5),
        "baseline_brier": round(baseline_brier, 5),
        "brier_skill": round(1 - brier / baseline_brier, 5) if baseline_brier else None,
    }


def _holdout_passes(metrics):
    """Require a meaningful untouched holdout that beats the baseline Brier score."""
    if not isinstance(metrics, dict):
        return False
    count = metrics.get("sample_count", 0)
    skill = metrics.get("brier_skill")
    return (
        isinstance(count, int) and count >= 20
        and isinstance(skill, (int, float))
        and np.isfinite(skill) and skill > 0
    )


def _fit_calibrator(raw, labels):
    if len(raw) < MIN_CALIBRATION or len(np.unique(labels)) < 2:
        return None
    model = LogisticRegression(C=1.0, max_iter=300)
    model.fit(np.log(np.clip(raw, 1e-5, 1 - 1e-5) /
                     (1 - np.clip(raw, 1e-5, 1 - 1e-5))).reshape(-1, 1),
              np.asarray(labels, dtype=int))
    return model


def _calibrate(calibrator, raw):
    raw = np.clip(np.asarray(raw, dtype=float), 1e-5, 1 - 1e-5)
    if calibrator is None:
        return raw
    logits = np.log(raw / (1 - raw)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _walk_forward(X, y, horizon, factory):
    y = y.reindex(X.index)
    valid = X.notna().all(axis=1) & y.notna()
    index = X.index[valid]
    if len(index) < MIN_TRAIN + horizon + 100:
        return None

    raw = pd.Series(np.nan, index=index, dtype=float)
    baseline = pd.Series(np.nan, index=index, dtype=float)
    for offset in range(MIN_TRAIN + horizon, len(index), STEP):
        # Purge labels whose forward-return windows overlap the test block.
        train_index = index[:offset - horizon]
        test_index = index[offset:offset + STEP]
        labels = y.loc[train_index].astype(int)
        if len(labels) < MIN_TRAIN or labels.nunique() < 2 or len(test_index) == 0:
            continue
        model = factory()
        model.fit(X.loc[train_index], labels)
        raw.loc[test_index] = model.predict_proba(X.loc[test_index])[:, 1]
        baseline.loc[test_index] = float(labels.mean())

    keep = raw.notna() & baseline.notna()
    raw, baseline = raw.loc[keep], baseline.loc[keep]
    actual = y.loc[raw.index].astype(int)
    if len(actual) < 100:
        return None
    split = max(1, int(len(actual) * 0.8))
    return {
        "raw": raw,
        "baseline": baseline,
        "actual": actual,
        "development_index": actual.index[:split],
        "holdout_index": actual.index[split:],
    }


def _one_horizon(features, close, horizon):
    forward_return = close.shift(-horizon) / close - 1
    labels = (forward_return > 0).where(forward_return.notna()).astype("float")
    valid = features.notna().all(axis=1) & labels.notna()
    factories = _factories()
    results = {}
    predictions = {}

    for name, factory in factories.items():
        wf = _walk_forward(features.loc[valid], labels.loc[valid], horizon, factory)
        if wf is None:
            continue
        dev = wf["development_index"]
        hold = wf["holdout_index"]
        dev_metrics = _metrics(
            wf["raw"].loc[dev], wf["actual"].loc[dev], wf["baseline"].loc[dev], horizon
        )
        hold_metrics = _metrics(
            wf["raw"].loc[hold], wf["actual"].loc[hold], wf["baseline"].loc[hold], horizon
        ) if len(hold) else None
        results[name] = {
            "development_brier": dev_metrics["brier"],
            "untouched_holdout": hold_metrics,
            "annual_holdout_metrics": _annual_metrics(
                wf["raw"].loc[hold], wf["actual"].loc[hold], wf["baseline"].loc[hold]
            ) if len(hold) else {},
            "holdout_start": str(hold.min().date()) if len(hold) else None,
            "holdout_end": str(hold.max().date()) if len(hold) else None,
        }
        predictions[name] = wf

    if not results:
        return {"status": "insufficient", "note": "此期間的完整特徵或樣本外資料不足。"}

    # Candidate selection uses development data only; the final holdout is untouched.
    selected_name = min(results, key=lambda key: results[key]["development_brier"])
    selected_factory = factories[selected_name]
    selected_wf = predictions[selected_name]
    holdout_metrics = results[selected_name]["untouched_holdout"]

    latest_date = features.dropna().index.max() if features.notna().any(axis=1).any() else None
    if latest_date is None:
        return {"status": "insufficient", "note": "缺少完整的最新特徵列。"}
    matured_train = valid & (features.index < latest_date)
    train_index = features.index[matured_train]
    if len(train_index) < MIN_TRAIN or labels.loc[train_index].nunique() < 2:
        return {
            "status": "insufficient", "asof": str(latest_date.date()),
            "training_samples": int(len(train_index)), "note": "成熟的歷史標籤不足。",
        }

    model = selected_factory()
    model.fit(features.loc[train_index], labels.loc[train_index].astype(int))
    raw_current = float(model.predict_proba(features.loc[[latest_date]])[:, 1][0])

    # Calibrate only with prior OOS predictions whose forward labels matured by latest_date.
    positions = pd.Series(np.arange(len(features.index)), index=features.index)
    latest_pos = int(positions.loc[latest_date])
    matured_dates = [
        day for day in selected_wf["raw"].index
        if int(positions.loc[day]) + horizon <= latest_pos
    ]
    calibrator = None
    if matured_dates:
        calibrator = _fit_calibrator(
            selected_wf["raw"].loc[matured_dates].to_numpy(),
            selected_wf["actual"].loc[matured_dates].to_numpy(),
        )
    calibrated = float(_calibrate(calibrator, [raw_current])[0]) if calibrator else None

    # A fitted calibrator is not evidence of predictive value. Only expose a
    # probability when the untouched chronological holdout beats the baseline
    # on the proper scoring rule (Brier score).
    estimate_available = calibrated is not None and _holdout_passes(holdout_metrics)

    return {
        "status": "available" if estimate_available else (
            "uncalibrated" if calibrated is None else "not_validated"
        ),
        "asof": str(latest_date.date()),
        "horizon_sessions": int(horizon),
        "training_samples": int(len(train_index)),
        "selected_research_model": selected_name,
        "estimated_positive_return_probability_pct": round(calibrated * 100, 2) if estimate_available else None,
        "research_category": (
            "偏正向研究分類" if calibrated >= 0.60 else
            "偏負向研究分類" if calibrated <= 0.40 else "中性研究分類"
        ) if estimate_available else (
            "尚未校準" if calibrated is None else "尚未通過樣本外驗證"
        ),
        "untouched_holdout": holdout_metrics,
        "annual_holdout_metrics": results[selected_name]["annual_holdout_metrics"],
        "candidate_comparison": results,
        "note": (
            "目標為該標的在指定交易日數後的累積報酬是否為正；這是歷史樣本外研究模型估計，"
            "不是價格目標、不是買賣指示，也不保證未來結果。"
            if estimate_available else (
                "目前沒有足夠成熟的樣本外預測可校準；暫不顯示機率，只累積資料。"
                if calibrated is None else
                "未觸發預測機率顯示：樣本外測試樣本數不足或 Brier Skill 未優於基準；"
                "暫不顯示機率，保留研究資料供後續驗證。"
            )
        ),
    }


def analyze(stock_close, market_close, volume=None):
    """Return 1m/3m/6m stock research estimates and chronological holdout metrics."""
    px = pd.to_numeric(stock_close, errors="coerce").dropna().sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[~px.index.duplicated(keep="last")]
    market = pd.to_numeric(market_close, errors="coerce").dropna().sort_index()
    market.index = pd.to_datetime(market.index).tz_localize(None).normalize()
    market = market[~market.index.duplicated(keep="last")]
    features = build_features(px, market, volume).reindex(px.index)
    output = {}
    for name, horizon in HORIZONS.items():
        output[name] = _one_horizon(features, px, horizon)
    return {
        "history_start": str(px.index.min().date()) if len(px) else None,
        "history_end": str(px.index.max().date()) if len(px) else None,
        "history_sessions": int(len(px)),
        "history_years_requested": 4,
        "horizons": output,
        "disclaimer": "研究用途；歷史樣本外結果不代表未來表現，不構成投資建議或交易指示。",
    }
