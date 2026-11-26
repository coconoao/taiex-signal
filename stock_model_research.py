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

    stock_returns = ret["stock"]
    market_returns = ret["market"]
    for window in (20, 60):
        min_periods = max(10, window // 2)
        covariance = stock_returns.rolling(window, min_periods=min_periods).cov(market_returns)
        market_variance = market_returns.rolling(window, min_periods=min_periods).var()
        beta = covariance / market_variance.replace(0, np.nan)
        out[f"rolling_beta_{window}"] = beta
        stock_mean = stock_returns.rolling(window, min_periods=min_periods).mean()
        market_mean = market_returns.rolling(window, min_periods=min_periods).mean()
        out[f"rolling_alpha_{window}"] = (stock_mean - beta * market_mean) * 252.0

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


def _matured_training_dates(feature_index, valid_mask, latest_date, horizon):
    """Return training origins whose full forward label matured by latest_date."""
    index = pd.DatetimeIndex(feature_index)
    latest_pos = index.get_loc(pd.Timestamp(latest_date))
    valid = pd.Series(np.asarray(valid_mask, dtype=bool), index=index)
    positions = np.arange(len(index))
    return index[valid.to_numpy() & (positions <= int(latest_pos) - int(horizon))]


def _development_calibration_dates(walk_forward, positions, latest_position, horizon):
    """Restrict calibration to matured development OOS rows, never final holdout."""
    development = set(walk_forward["development_index"])
    return [
        day for day in walk_forward["raw"].index
        if day in development
        and int(positions.loc[day]) + int(horizon) <= int(latest_position)
    ]


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
    train_index = _matured_training_dates(
        features.index, valid.to_numpy(), latest_date, horizon
    )
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
    matured_dates = _development_calibration_dates(
        selected_wf, positions, latest_pos, horizon
    )
    calibrator = None
    if matured_dates:
        calibrator = _fit_calibrator(
            selected_wf["raw"].loc[matured_dates].to_numpy(),
            selected_wf["actual"].loc[matured_dates].to_numpy(),
        )
    calibrated = float(_calibrate(calibrator, [raw_current])[0]) if calibrator else None

    return {
        "status": "available" if calibrated is not None else "uncalibrated",
        "asof": str(latest_date.date()),
        "horizon_sessions": int(horizon),
        "training_samples": int(len(train_index)),
        "selected_research_model": selected_name,
        "estimated_positive_return_probability_pct": round(calibrated * 100, 2) if calibrated is not None else None,
        "research_category": (
            "偏正向研究分類" if calibrated >= 0.60 else
            "偏負向研究分類" if calibrated <= 0.40 else "中性研究分類"
        ) if calibrated is not None else "尚未校準",
        "untouched_holdout": holdout_metrics,
        "annual_holdout_metrics": results[selected_name]["annual_holdout_metrics"],
        "candidate_comparison": results,
        "note": (
            "目標為該標的在指定交易日數後的累積報酬是否為正；這是歷史樣本外研究模型估計，"
            "不是價格目標、不是買賣指示，也不保證未來結果。"
            if calibrated is not None else
            "目前沒有足夠成熟的樣本外預測可校準；暫不顯示機率，只累積資料。"
        ),
    }


def _recent_direction_signal(stock_close, market_close, horizon):
    """Rule-based recent trend snapshot; deliberately not a probability forecast."""
    px = pd.to_numeric(stock_close, errors="coerce").dropna().sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[~px.index.duplicated(keep="last")]
    minimum_sessions = int(horizon) + 1
    asof = str(px.index[-1].date()) if len(px) else None
    if len(px) < minimum_sessions:
        return {
            "status": "insufficient", "asof": asof,
            "sessions_used": int(len(px)), "minimum_sessions": minimum_sessions,
            "signal": None, "components": [],
            "note": "近期有效交易日不足，暫不產生此期間訊號。",
        }

    current = float(px.iloc[-1])
    momentum = float(px.iloc[-1] / px.iloc[-1 - horizon] - 1)
    average = float(px.iloc[-horizon:].mean())
    trend_gap = current / average - 1
    delta = px.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_value = float((100 - 100 / (1 + rs)).iloc[-1]) if pd.notna(rs.iloc[-1]) else None

    thresholds = {21: (0.015, 0.005, 0.01), 63: (0.04, 0.005, 0.02),
                  126: (0.08, 0.005, 0.04)}
    momentum_threshold, trend_threshold, relative_threshold = thresholds.get(
        int(horizon), (0.02, 0.005, 0.02)
    )
    components = []

    def add_component(name, value, positive, negative, unit=None):
        direction = 1 if value > positive else -1 if value < negative else 0
        components.append({
            "name": name,
            "value": round(float(value) * 100, 2) if unit == "pct" else round(float(value), 1),
            "unit": "%" if unit == "pct" else "",
            "direction": "偏多" if direction > 0 else "偏空" if direction < 0 else "中性",
            "score": direction,
        })

    add_component("期間動能", momentum, momentum_threshold, -momentum_threshold, "pct")
    add_component("價格相對均線", trend_gap, trend_threshold, -trend_threshold, "pct")

    market = pd.to_numeric(market_close, errors="coerce").dropna().sort_index()
    market.index = pd.to_datetime(market.index).tz_localize(None).normalize()
    market = market[~market.index.duplicated(keep="last")]
    aligned_market = market.reindex(px.index)
    if aligned_market.notna().sum() >= minimum_sessions and pd.notna(aligned_market.iloc[-1]):
        market_base = aligned_market.iloc[-1 - horizon]
        if pd.notna(market_base) and float(market_base) != 0:
            relative = momentum - (float(aligned_market.iloc[-1]) / float(market_base) - 1)
            add_component("相對大盤動能", relative, relative_threshold, -relative_threshold, "pct")

    if rsi_value is not None:
        rsi_direction = 1 if rsi_value >= 55 else -1 if rsi_value <= 45 else 0
        components.append({
            "name": "RSI(14)", "value": round(rsi_value, 1), "unit": "",
            "direction": "偏多" if rsi_direction > 0 else "偏空" if rsi_direction < 0 else "中性",
            "score": rsi_direction,
        })

    score = float(np.mean([item["score"] for item in components])) if components else 0.0
    signal = "偏多" if score >= 0.34 else "偏空" if score <= -0.34 else "中性"
    return {
        "status": "available", "asof": asof, "horizon_sessions": int(horizon),
        "sessions_used": int(len(px)), "minimum_sessions": minimum_sessions,
        "signal": signal, "score": round(score, 3), "components": components,
        "note": "依近期價格趨勢、動能、相對大盤表現與 RSI 規則整理；未經樣本外驗證，不是模型機率或買賣指示。",
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
        "history_years_requested": 10,
        "horizons": output,
        "recent_signals": {
            name: _recent_direction_signal(px, market, horizon)
            for name, horizon in HORIZONS.items()
        },
        "disclaimer": "研究用途；歷史樣本外結果不代表未來表現，不構成投資建議或交易指示。",
    }



def prequential_platt_candidates(features, close, horizon=21, oos_start="2024-01-01"):
    """Compare unscaled logistic C=.1/.01 with expanding, matured-OOS Platt.

    Walk-forward training and calibration purges use positions on the complete
    feature/trading calendar, never the compressed valid-row index.
    """
    from scripts.final_oos_methods import fit_expanding_platt, binary_metrics

    px = pd.to_numeric(close, errors="coerce").reindex(features.index)
    forward_return = px.shift(-int(horizon)) / px - 1.0
    labels = (forward_return > 0).where(forward_return.notna()).astype("float")
    full_index = pd.DatetimeIndex(features.index)
    positions = pd.Series(np.arange(len(full_index)), index=full_index)
    valid = features.notna().all(axis=1) & labels.notna()
    eligible_dates = full_index[valid.to_numpy()]
    start_date = pd.Timestamp(oos_start)
    oos_dates = eligible_dates[eligible_dates >= start_date]
    raw_results = {}

    for base_c in (0.1, 0.01):
        raw_rows = []
        for offset in range(0, len(oos_dates), STEP):
            test_dates = oos_dates[offset:offset + STEP]
            if not len(test_dates):
                continue
            first_pos = int(positions.loc[test_dates[0]])
            candidate_train = eligible_dates[
                (positions.reindex(eligible_dates).to_numpy() + int(horizon)) <= first_pos
            ]
            if len(candidate_train) < MIN_TRAIN or labels.loc[candidate_train].nunique() < 2:
                continue
            model = LogisticRegression(C=float(base_c), max_iter=700, random_state=RANDOM_STATE)
            model.fit(features.loc[candidate_train], labels.loc[candidate_train].astype(int))
            raw_prob = model.predict_proba(features.loc[test_dates])[:, 1]
            train_prior = float(labels.loc[candidate_train].mean())
            for date, probability in zip(test_dates, raw_prob):
                raw_rows.append({
                    "date": date,
                    "raw_probability": float(probability),
                    "actual": int(labels.loc[date]),
                    "baseline_probability": train_prior,
                    "training_samples": int(len(candidate_train)),
                    "training_end": str(candidate_train.max().date()),
                })
        if not raw_rows:
            raw_results[str(base_c)] = {"status": "insufficient", "samples": 0}
            continue

        raw_frame = pd.DataFrame(raw_rows).set_index("date").sort_index()
        raw_series = raw_frame["raw_probability"]
        actual_series = raw_frame["actual"]
        calibrated = fit_expanding_platt(
            raw_series, actual_series, full_index,
            horizon_sessions=21, block_size=21,
            min_samples=300, min_per_class=50, platt_c=1.0,
        )
        joined = calibrated.join(raw_frame[["baseline_probability"]], how="inner")
        if joined.empty:
            raw_results[str(base_c)] = {"status": "insufficient", "samples": 0}
            continue

        def summarize(frame):
            y = frame["actual"].to_numpy(dtype=int)
            p = np.clip(frame["probability"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
            raw_p = np.clip(frame["raw_probability"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
            baseline = np.clip(frame["baseline_probability"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
            brier = float(np.mean((p - y) ** 2))
            base_brier = float(np.mean((baseline - y) ** 2))
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            if len(np.unique(y)) == 2 and len(y) >= 20:
                calibration_model = LogisticRegression(C=1e6, max_iter=500)
                calibration_model.fit(logits, y)
                calibration_slope = float(calibration_model.coef_[0, 0])
                calibration_intercept = float(calibration_model.intercept_[0])
            else:
                calibration_slope, calibration_intercept = None, None
            bins = np.array_split(np.argsort(p), min(10, max(1, len(p))))
            ece = float(sum(
                len(ix) / len(p) * abs(float(p[ix].mean()) - float(y[ix].mean()))
                for ix in bins if len(ix)
            ))
            brier_skill_ci = None
            log_loss_delta_ci = None
            if len(y) >= 84:
                rng_ci = np.random.default_rng(20261126)
                block = 21
                starts = np.arange(0, len(y) - block + 1)
                draws = np.empty((500, 2), dtype=float)
                blocks_needed = int(np.ceil(len(y) / block))
                for draw in range(len(draws)):
                    chosen = rng_ci.choice(starts, size=blocks_needed, replace=True)
                    ix = np.concatenate([np.arange(s, s + block) for s in chosen])[:len(y)]
                    pb, bb, yy = p[ix], baseline[ix], y[ix]
                    b0 = float(np.mean((bb - yy) ** 2))
                    draws[draw, 0] = 1.0 - float(np.mean((pb - yy) ** 2)) / b0 if b0 > 0 else np.nan
                    llp = -(yy * np.log(pb) + (1 - yy) * np.log(1 - pb))
                    llb = -(yy * np.log(bb) + (1 - yy) * np.log(1 - bb))
                    draws[draw, 1] = float(np.mean(llp - llb))
                brier_skill_ci = [float(x) for x in np.nanquantile(draws[:, 0], [0.025, 0.975])]
                log_loss_delta_ci = [float(x) for x in np.nanquantile(draws[:, 1], [0.025, 0.975])]
            return {
                "samples": int(len(frame)),
                "start_date": str(frame.index.min().date()),
                "end_date": str(frame.index.max().date()),
                "brier_score": brier,
                "baseline_brier_score": base_brier,
                "brier_skill": float(1.0 - brier / base_brier) if base_brier > 0 else None,
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "calibration_slope": calibration_slope,
                "calibration_intercept": calibration_intercept,
                "ece_10_bins": ece,
                "brier_skill_95pct_block_bootstrap_ci": brier_skill_ci,
                "log_loss_delta_95pct_block_bootstrap_ci": log_loss_delta_ci,
                "raw_brier_score": float(np.mean((raw_p - y) ** 2)),
                "raw_log_loss": float(log_loss(y, raw_p, labels=[0, 1])),
                "direction_accuracy_pct": float(np.mean((p >= 0.5) == (y == 1)) * 100.0),
                "calibrated_blocks": int((frame["calibration_status"] == "platt_fitted").sum()),
                "calibration_coverage_pct": float((frame["calibration_status"] == "platt_fitted").mean() * 100.0),
            }

        annual = {}
        for year, group in joined.groupby(joined.index.year):
            annual[str(year)] = summarize(group)
        raw_results[str(base_c)] = {
            "status": "available",
            "base_model": "LogisticRegression_no_scaling",
            "base_model_c": float(base_c),
            "platt_regularization_c": 1.0,
            "purge_sessions": 21,
            "full_calendar_sessions": int(len(full_index)),
            "minimum_calibration_samples": 300,
            "minimum_each_class": 50,
            "overall": summarize(joined),
            "annual": annual,
            "calibration_status_counts": {
                str(k): int(v) for k, v in joined["calibration_status"].value_counts().items()
            },
        }
    return {
        "status": "available" if any(x.get("status") == "available" for x in raw_results.values()) else "insufficient",
        "target_horizon_sessions": int(horizon),
        "oos_start_requested": str(start_date.date()),
        "oos_start_actual": min((v.get("overall", {}).get("start_date") for v in raw_results.values() if v.get("overall")), default=None),
        "oos_end_actual": max((v.get("overall", {}).get("end_date") for v in raw_results.values() if v.get("overall")), default=None),
        "candidates": raw_results,
    }
