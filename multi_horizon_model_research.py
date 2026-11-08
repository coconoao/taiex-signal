"""Multi-horizon TAIEX research challenger.

Targets are positive forward returns over 21/63/126 sessions, not next-day
direction. Uses purged expanding-window walk-forward predictions, past-only
probability calibration, development-only candidate selection, and a final
chronological holdout. Never writes production signals.json.
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

import model_improvement_research as mir
import pipeline

HORIZONS = {"1m": 21, "3m": 63, "6m": 126}
MIN_TRAIN = 500
MIN_CALIBRATION = 80
STEP = 20
DEV_FRACTION = 0.80
RANDOM_STATE = 42


def factories():
    return {
        "logistic_baseline": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=700)
        ),
        "logistic_regularized": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=700)
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=250, max_depth=4, min_samples_leaf=25,
            max_features=0.8, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=2,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=7, min_samples_leaf=35,
            learning_rate=0.04, l2_regularization=8.0,
            random_state=RANDOM_STATE,
        ),
    }


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _fit_calibrator(raw, labels):
    if len(raw) < MIN_CALIBRATION or len(np.unique(labels)) < 2:
        return None
    cal = LogisticRegression(C=1.0, max_iter=300)
    cal.fit(_logit(raw), np.asarray(labels, dtype=int))
    return cal


def _apply_calibrator(cal, raw):
    raw = np.asarray(raw, dtype=float)
    if cal is None:
        return np.clip(raw, 1e-5, 1 - 1e-5)
    return cal.predict_proba(_logit(raw))[:, 1]


def _metrics(prob, y, baseline_probability=None):
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    actual = np.asarray(y, dtype=int)
    prevalence = float(np.mean(actual))
    baseline = (np.full(len(actual), prevalence) if baseline_probability is None
                else np.clip(np.asarray(baseline_probability, dtype=float), 1e-6, 1 - 1e-6))
    brier = float(np.mean((p - actual) ** 2))
    base_brier = float(np.mean((baseline - actual) ** 2))
    return {
        "sample_count": int(len(actual)),
        "positive_rate_pct": round(prevalence * 100, 2),
        "brier": round(brier, 5),
        "prevalence_baseline_brier": round(base_brier, 5),
        "brier_skill_vs_prevalence": round(1 - brier / base_brier, 5) if base_brier else None,
        "log_loss": round(float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p))), 5),
        "direction_accuracy_pct": round(float(np.mean((p >= 0.5) == (actual == 1)) * 100), 2),
        "mean_estimated_probability_pct": round(float(np.mean(p) * 100), 2),
    }


def _reliability_bins(prob, y, minimum=20):
    p = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001]
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n >= minimum:
            result.append({
                "probability_band_pct": [round(lo * 100, 1), round(min(hi, 1) * 100, 1)],
                "sample_count": n,
                "mean_estimated_probability_pct": round(float(p[mask].mean() * 100), 2),
                "observed_positive_rate_pct": round(float(y[mask].mean() * 100), 2),
            })
    return result


def _return_context(prob, fwd):
    frame = pd.DataFrame({"prob": prob, "return": fwd.reindex(prob.index)}).dropna()
    edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        part = frame[(frame.prob >= lo) & (frame.prob < hi)]
        if len(part) >= 20:
            rows.append({
                "probability_band_pct": [round(lo * 100, 1), round(min(hi, 1) * 100, 1)],
                "sample_count": int(len(part)),
                "observed_positive_rate_pct": round(float((part["return"] > 0).mean() * 100), 2),
                "mean_forward_return_pct": round(float(part["return"].mean() * 100), 3),
                "median_forward_return_pct": round(float(part["return"].median() * 100), 3),
                "tenth_percentile_return_pct": round(float(part["return"].quantile(0.10) * 100), 3),
            })
    return rows


def walk_forward(X, y, horizon, factory):
    # Feature-set comparisons may intentionally use a strict common-date
    # subset. Align labels to that exact index before building the mask.
    y = y.reindex(X.index)
    valid = X.notna().all(axis=1) & y.notna()
    idx = X.index[valid]
    if len(idx) < MIN_TRAIN + horizon + 100:
        return None
    raw_pred = pd.Series(np.nan, index=idx, dtype=float)
    calibrated_pred = pd.Series(np.nan, index=idx, dtype=float)
    baseline_pred = pd.Series(np.nan, index=idx, dtype=float)

    for i in range(MIN_TRAIN + horizon, len(idx), STEP):
        # Purge all training labels whose forward-return windows overlap this test block.
        train_end = i - horizon
        train_idx = idx[:train_end]
        test_idx = idx[i:i + STEP]
        if len(train_idx) < MIN_TRAIN or y.loc[train_idx].nunique() < 2:
            continue
        model = factory()
        model.fit(X.loc[train_idx], y.loc[train_idx].astype(int))
        raw_test = model.predict_proba(X.loc[test_idx])[:, 1]

        # Calibrator sees only prior OOS predictions whose outcome windows
        # have fully matured before the current test block.
        matured_cutoff = i - horizon
        past_idx = idx[:matured_cutoff]
        past = raw_pred.loc[past_idx].dropna()
        past_labels = y.loc[past.index].astype(int) if len(past) else pd.Series(dtype=int)
        calibrator = _fit_calibrator(past.to_numpy(), past_labels.to_numpy()) if len(past) else None
        calibrated_test = _apply_calibrator(calibrator, raw_test)
        raw_pred.loc[test_idx] = raw_test
        calibrated_pred.loc[test_idx] = calibrated_test
        baseline_pred.loc[test_idx] = float(y.loc[train_idx].mean())

    keep = calibrated_pred.notna() & baseline_pred.notna()
    raw_pred = raw_pred.loc[keep]
    calibrated_pred = calibrated_pred.loc[keep]
    baseline_pred = baseline_pred.loc[keep]
    actual = y.loc[calibrated_pred.index].astype(int)
    if len(actual) < 100:
        return None

    split = max(1, int(len(actual) * DEV_FRACTION))
    dev_idx = actual.index[:split]
    hold_idx = actual.index[split:]
    if len(hold_idx) < 50:
        return None

    return {
        "raw_probability": raw_pred,
        "probability": calibrated_pred,
        "baseline_probability": baseline_pred,
        "actual": actual,
        "development_index": dev_idx,
        "holdout_index": hold_idx,
    }


def _promotion_quality_check(hold_metrics, yearly_blocks, holdout_count, horizon):
    """Conservative gate accounting for overlapping forward-return labels."""
    independent_windows = int(holdout_count // horizon)
    year_skills = [
        item.get("brier_skill_vs_prevalence")
        for item in yearly_blocks.values()
        if item.get("brier_skill_vs_prevalence") is not None
    ]
    reasons = []
    if (hold_metrics.get("brier_skill_vs_prevalence") or 0) <= 0:
        reasons.append("整體樣本外 Brier Skill 未優於基準")
    if independent_windows < 20:
        reasons.append("扣除重疊視窗後的獨立期間少於 20")
    if len(year_skills) < 3:
        reasons.append("可評估年度區塊少於 3 個")
    if year_skills and any(value <= 0 for value in year_skills):
        reasons.append("至少一個年度區塊未優於基準")
    return {
        "eligible": not reasons,
        "effective_nonoverlap_windows": independent_windows,
        "minimum_nonoverlap_windows": 20,
        "evaluated_year_blocks": len(year_skills),
        "minimum_year_blocks": 3,
        "positive_year_blocks": sum(value > 0 for value in year_skills),
        "reasons": reasons,
        "note": "同一交易日的多日遠期報酬標籤高度重疊；原始預測筆數不等於獨立樣本數。",
    }


def evaluate_candidate(X, y, fwd, horizon, name, factory):
    result = walk_forward(X, y, horizon, factory)
    if result is None:
        return {"status": "insufficient", "note": "有效樣本外資料不足。"}
    dev = result["development_index"]
    hold = result["holdout_index"]
    prob = result["probability"]
    actual = result["actual"]
    base = result["baseline_probability"]
    hold_metrics = _metrics(prob.loc[hold], actual.loc[hold], base.loc[hold])
    baseline_metrics = _metrics(base.loc[hold], actual.loc[hold], base.loc[hold])
    hold_fwd = fwd.reindex(hold)
    yearly_blocks = {
        str(year): _metrics(
            prob.loc[hold][prob.loc[hold].index.year == year],
            actual.loc[hold][actual.loc[hold].index.year == year],
            base.loc[hold][base.loc[hold].index.year == year],
        )
        for year in sorted(set(hold.year))
        if int((hold.year == year).sum()) >= 20
    }
    quality_check = _promotion_quality_check(
        hold_metrics, yearly_blocks, len(hold), horizon
    )
    return {
        "status": "evaluated",
        "candidate": name,
        "development": _metrics(prob.loc[dev], actual.loc[dev], base.loc[dev]),
        "untouched_holdout": hold_metrics,
        "holdout_prevalence_baseline": baseline_metrics,
        "holdout_reliability": _reliability_bins(prob.loc[hold], actual.loc[hold]),
        "holdout_return_context": _return_context(prob.loc[hold], hold_fwd),
        "development_start": str(dev.min().date()),
        "development_end": str(dev.max().date()),
        "holdout_start": str(hold.min().date()),
        "holdout_end": str(hold.max().date()),
        "holdout_prediction_count": int(len(hold)),
        "holdout_brier_skill_vs_prevalence": hold_metrics["brier_skill_vs_prevalence"],
        "holdout_yearly_blocks": yearly_blocks,
        "promotion_quality_check": quality_check,
    }


def latest_estimate(X, y, horizon, factory, oos_probability, oos_actual):
    complete = X.notna().all(axis=1)
    if not complete.any():
        return {"status": "insufficient", "note": "沒有完整的最新特徵列。"}
    asof = X.index[complete].max()
    # y.notna() means the entire forward-return window has matured in the
    # observed trading calendar; never approximate sessions with business days.
    valid = complete & y.notna() & (X.index < asof)
    idx = X.index[valid]
    if len(idx) < MIN_TRAIN or y.loc[idx].nunique() < 2:
        return {"status": "insufficient", "asof": str(asof.date()),
                "training_samples": int(len(idx)), "note": "成熟的歷史標籤不足。"}
    model = factory()
    model.fit(X.loc[idx], y.loc[idx].astype(int))
    raw = float(model.predict_proba(X.loc[[asof]])[:, 1][0])
    # Use the actual Taiwan trading-session calendar, not business-day offsets:
    # holidays and market closures must not count as forecast sessions.
    asof_pos = int(X.index.get_indexer([asof])[0])
    mature_count = max(0, asof_pos - horizon + 1)
    matured_calendar = X.index[:mature_count]
    matured_oos = oos_probability.index.intersection(matured_calendar)
    matured_oos = matured_oos.intersection(oos_actual.index)
    matured_oos = matured_oos[oos_probability.loc[matured_oos].notna()]
    cal = _fit_calibrator(oos_probability.loc[matured_oos].to_numpy(),
                          oos_actual.loc[matured_oos].astype(int).to_numpy()) if len(matured_oos) else None
    calibrated = float(_apply_calibrator(cal, [raw])[0])
    return {
        "status": "available",
        "asof": str(asof.date()),
        "training_samples": int(len(idx)),
        "estimated_positive_return_probability_pct": round(calibrated * 100, 2) if cal is not None else None,
        "calibration_status": "past_oos_calibrated" if cal is not None else "uncalibrated_model_score",
        "research_category": ("偏正向" if calibrated >= 0.60 else "中性" if calibrated >= 0.40 else "偏負向") if cal is not None else "尚未校準",
        "note": "這是指定期間正報酬的模型估計，不是保證、不是持有/賣出指示；若校準樣本不足，數值不可視為可靠機率。",
    }



def build_macro_proxy_features(market):
    """Build point-in-time daily macro/market proxies from the existing price panel.

    These are market-implied proxies, not historical macroeconomic releases.
    Every feature at date t uses observations at or before t only.
    """
    frame = market.copy().sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    returns = frame.pct_change(fill_method=None)
    out = pd.DataFrame(index=frame.index)

    # Global technology cycle and regional risk appetite.
    for col in ("SOX", "TSM", "N225", "KOSPI"):
        if col not in frame:
            continue
        for window in (5, 21, 63):
            out[f"宏觀代理:{col}_報酬{window}日"] = frame[col].pct_change(window, fill_method=None)
        out[f"宏觀代理:{col}_波動21日"] = returns[col].rolling(21, min_periods=15).std()

    # Volatility regime: level changes and a trailing-only standardized level.
    if "VIX" in frame:
        vix = frame["VIX"]
        out["宏觀代理:VIX變化5日"] = vix.diff(5)
        out["宏觀代理:VIX變化21日"] = vix.diff(21)
        vix_mean = vix.rolling(63, min_periods=40).mean()
        vix_std = vix.rolling(63, min_periods=40).std()
        out["宏觀代理:VIX滾動Z分數"] = (vix - vix_mean) / vix_std.replace(0, np.nan)

    # Interest-rate pressure proxy; source series is the market yield quote.
    if "US10Y" in frame:
        rate = frame["US10Y"]
        out["宏觀代理:美國10年債殖利率變化5日"] = rate.diff(5)
        out["宏觀代理:美國10年債殖利率變化21日"] = rate.diff(21)
        rate_mean = rate.rolling(63, min_periods=40).mean()
        rate_std = rate.rolling(63, min_periods=40).std()
        out["宏觀代理:美國10年債殖利率滾動Z分數"] = (
            (rate - rate_mean) / rate_std.replace(0, np.nan)
        )

    # FX pressure proxy; use the quoted USD/TWD market series as-is.
    if "USDTWD" in frame:
        fx = frame["USDTWD"]
        for window in (5, 21, 63):
            out[f"宏觀代理:USD_TWD變化{window}日"] = fx.pct_change(window, fill_method=None)
        out["宏觀代理:USD_TWD波動21日"] = returns["USDTWD"].rolling(21, min_periods=15).std()

    # Relative leadership versus Taiwan, rather than absolute index levels.
    if "TAIEX" in frame:
        taiex = frame["TAIEX"].pct_change(21, fill_method=None)
        for col in ("SOX", "KOSPI", "N225"):
            if col in frame:
                out[f"宏觀代理:{col}_相對TAIEX_21日"] = (
                    frame[col].pct_change(21, fill_method=None) - taiex
                )
        if "SOX" in returns:
            out["宏觀代理:SOX_TAIEX_相關21日"] = (
                returns["SOX"].rolling(21, min_periods=15).corr(returns["TAIEX"])
            )
    return out.replace([np.inf, -np.inf], np.nan)

def run():
    market, chips, margin, futures = mir.load_inputs()
    experiments = {}
    latest = {}
    for horizon_name, horizon in HORIZONS.items():
        Xh, yh, fwd = pipeline.build(market, chips, margin, futures=futures, h=horizon)
        market_cols = [col for col in Xh.columns
                       if not col.startswith(("籌碼:", "信用:", "期貨:"))]
        macro = build_macro_proxy_features(market).reindex(Xh.index)
        macro_cols = list(macro.columns)
        common = (
            Xh[market_cols].notna().all(axis=1)
            & macro.notna().all(axis=1)
            & yh.notna()
        )
        # Baseline and enriched proxy model share the exact same dates.
        market_common = Xh.loc[common, market_cols]
        macro_common = pd.concat(
            [market_common, macro.loc[common, macro_cols]], axis=1
        )
        feature_sets = {
            "market_only_common_calendar": market_common,
            "macro_proxy_enhanced": macro_common,
        }
        for layer, prefix in [
            ("market_plus_chips", "籌碼:"),
            ("market_plus_margin", "信用:"),
            ("market_plus_futures", "期貨:"),
        ]:
            extra = [col for col in Xh.columns if col.startswith(prefix)]
            if extra:
                feature_sets[layer] = Xh[market_cols + extra]
        if len(Xh.columns) > len(market_cols):
            feature_sets["all_features"] = Xh

        candidates_result = {}
        market_candidates = {}
        model_factories = factories()
        for feature_set_name, X_candidate in feature_sets.items():
            candidate_factories = (
                model_factories
                if feature_set_name == "market_only_common_calendar"
                else {
                    "logistic_baseline": model_factories["logistic_baseline"],
                    "logistic_regularized": model_factories["logistic_regularized"],
                }
            )
            for model_name, factory in candidate_factories.items():
                name = feature_set_name + "__" + model_name
                result = evaluate_candidate(X_candidate, yh, fwd, horizon, name, factory)
                result["feature_set"] = feature_set_name
                result["feature_count"] = int(X_candidate.shape[1])
                result["available_complete_samples"] = int(
                    (X_candidate.notna().all(axis=1) & yh.notna()).sum()
                )
                candidates_result[name] = result
                if feature_set_name in ("market_only_common_calendar", "macro_proxy_enhanced"):
                    market_candidates[name] = result

        viable_market = {k: v for k, v in market_candidates.items()
                         if v.get("status") == "evaluated"}
        if not viable_market:
            experiments[horizon_name] = {
                "status": "insufficient", "horizon_sessions": horizon,
                "candidates": candidates_result, "selected_candidate": None,
            }
            latest[horizon_name] = {"status": "insufficient", "note": "市場特徵完整樣本不足，尚不能選出候選模型。"}
            continue

        # Select only within the identical market-only calendar. Layer-specific
        # results are reported separately and are never ranked against a
        # different complete-case calendar.
        selected_name = min(
            viable_market,
            key=lambda k: viable_market[k]["development"]["brier"]
        )
        selected = viable_market[selected_name]
        selected_feature_set = selected.get("feature_set", "market_only_common_calendar")
        selected_X = feature_sets[selected_feature_set]
        # Keep the latest feature rows (whose forward labels are not mature yet)
        # for the current estimate. Only the historical evaluation calendar is
        # restricted to rows with realized outcomes.
        selected_X_full = (
            Xh[market_cols]
            if selected_feature_set == "market_only_common_calendar"
            else pd.concat([Xh[market_cols], macro[macro_cols].reindex(Xh.index)], axis=1)
        )
        selected_factory_name = selected_name.split("__", 1)[1]
        wf = walk_forward(selected_X, yh.reindex(selected_X.index), horizon, model_factories[selected_factory_name])
        current = latest_estimate(
            selected_X_full, yh, horizon, model_factories[selected_factory_name],
            wf["probability"], wf["actual"]
        ) if wf else {"status": "insufficient"}
        experiments[horizon_name] = {
            "status": "evaluated",
            "horizon_sessions": horizon,
            "selected_candidate": selected_name,
            "selection_basis": "market-only and macro-proxy candidates compared on the same complete-case calendar; selected on development Brier only; final chronological holdout not used for candidate selection",
            "macro_proxy_features": macro_cols,
            "macro_proxy_note": "Daily market-implied proxies from SOX, TSM, VIX, US10Y, USD/TWD, Nikkei and KOSPI. These are not historical CPI/GDP/jobs release series; no latest macro snapshot is backfilled into historical rows.",
            "comparison_calendar_samples": int(common.sum()),
            "calendar_warning": "Market-only and macro-proxy candidates share one complete-case calendar. Taiwan layer-specific models use different complete-case calendars and are not directly ranked against them.",
            "selected_candidate_results": selected,
            "candidates": candidates_result,
            "promotion_gate": {
                "requires_positive_holdout_brier_skill": True,
                "requires_reliability_bins_to_track_estimated_probabilities": True,
                "requires_at_least_3_year_blocks_all_positive": True,
                "requires_at_least_20_effective_nonoverlap_windows": True,
                "requires_comparison_to_existing_strategy_on_same_horizon": True,
                "requires_untouched_followup_period": True,
                "current_candidate_gate": selected.get("promotion_quality_check", {}),
            },
        }
        latest[horizon_name] = current

    return {
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated" if any(v.get("status") == "evaluated" for v in experiments.values()) else "insufficient",
        "target": "TAIEX positive forward return probability by 1m/3m/6m horizon",
        "horizons": HORIZONS,
        "minimum_training_samples": MIN_TRAIN,
        "walk_forward_step_sessions": STEP,
        "development_fraction": DEV_FRACTION,
        "purge_sessions": "equal to forecast horizon",
        "latest_estimates": latest,
        "experiments": experiments,
        "promotion_policy": "No automatic production replacement. A candidate must beat prevalence and incumbent strategy baselines on untouched same-horizon holdouts, show positive Brier skill, reliable calibration, stable year/regime blocks, acceptable drawdown under a separate execution simulation, and pass a later untouched follow-up period. Promotion requires explicit review.",
        "disclaimer": "General historical research only. Probabilities are estimates conditional on historical data and model assumptions, not guarantees, individualized hold/sell advice, or trade instructions.",
    }


if __name__ == "__main__":
    result = run()
    tmp = "multi_horizon_model_research.json.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, "multi_horizon_model_research.json")
    for horizon, item in result["experiments"].items():
        selected = item.get("selected_candidate_results", {}).get("untouched_holdout", {})
        print(horizon, item.get("selected_candidate"), selected.get("sample_count"),
              selected.get("brier"), selected.get("brier_skill_vs_prevalence"))
