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


def _metrics(prob, y):
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    actual = np.asarray(y, dtype=int)
    prevalence = float(np.mean(actual))
    baseline = np.full(len(actual), prevalence)
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

    # Candidate selection uses development period only.
    dev_scores = {}
    for name, factory_candidate in factories().items():
        # Walk-forward predictions above are for this candidate only; selection
        # is performed by the caller across candidate-specific result objects.
        pass
    return {
        "raw_probability": raw_pred,
        "probability": calibrated_pred,
        "baseline_probability": baseline_pred,
        "actual": actual,
        "development_index": dev_idx,
        "holdout_index": hold_idx,
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
    hold_metrics = _metrics(prob.loc[hold], actual.loc[hold])
    baseline_metrics = _metrics(base.loc[hold], actual.loc[hold])
    hold_fwd = fwd.reindex(hold)
    return {
        "status": "evaluated",
        "candidate": name,
        "development": _metrics(prob.loc[dev], actual.loc[dev]),
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
    }


def latest_estimate(X, y, horizon, factory, oos_probability, oos_actual):
    complete = X.notna().all(axis=1)
    if not complete.any():
        return {"status": "insufficient", "note": "沒有完整的最新特徵列。"}
    asof = X.index[complete].max()
    valid = complete & y.notna() & (X.index <= asof - pd.tseries.offsets.BDay(0))
    idx = X.index[valid]
    # Exclude rows whose forward label has not matured by the as-of date.
    idx = idx[idx <= asof - pd.offsets.BDay(horizon)]
    if len(idx) < MIN_TRAIN or y.loc[idx].nunique() < 2:
        return {"status": "insufficient", "asof": str(asof.date()),
                "training_samples": int(len(idx)), "note": "成熟的歷史標籤不足。"}
    model = factory()
    model.fit(X.loc[idx], y.loc[idx].astype(int))
    raw = float(model.predict_proba(X.loc[[asof]])[:, 1][0])
    matured_oos = oos_probability.index[oos_probability.index <= asof - pd.offsets.BDay(horizon)]
    matured_oos = matured_oos.intersection(oos_actual.index)
    matured_oos = matured_oos[oos_probability.loc[matured_oos].notna()]
    cal = _fit_calibrator(oos_probability.loc[matured_oos].to_numpy(),
                          oos_actual.loc[matured_oos].astype(int).to_numpy()) if len(matured_oos) else None
    calibrated = float(_apply_calibrator(cal, [raw])[0])
    return {
        "status": "available",
        "asof": str(asof.date()),
        "training_samples": int(len(idx)),
        "estimated_positive_return_probability_pct": round(calibrated * 100, 2),
        "calibration_status": "past_oos_calibrated" if cal is not None else "uncalibrated_model_score",
        "research_category": "偏正向" if calibrated >= 0.60 else "中性" if calibrated >= 0.40 else "偏負向",
        "note": "這是指定期間正報酬的模型估計，不是保證、不是持有/賣出指示；若校準樣本不足，數值不可視為可靠機率。",
    }


def run():
    market, chips, margin, futures = mir.load_inputs()
    X, _, _ = pipeline.build(market, chips, margin, futures=futures, h=1)
    experiments = {}
    latest = {}
    for horizon_name, horizon in HORIZONS.items():
        Xh, yh, fwd = pipeline.build(market, chips, margin, futures=futures, h=horizon)
        candidates_result = {}
        candidate_results = {}
        for name, factory in factories().items():
            result = evaluate_candidate(Xh, yh, fwd, horizon, name, factory)
            candidates_result[name] = result
            candidate_results[name] = result
        viable = {k: v for k, v in candidate_results.items()
                  if v.get("status") == "evaluated"}
        if not viable:
            experiments[horizon_name] = {
                "status": "insufficient", "horizon_sessions": horizon,
                "candidates": candidates_result, "selected_candidate": None,
            }
            latest[horizon_name] = {"status": "insufficient", "note": "候選模型樣本不足。"}
            continue
        selected_name = min(
            viable,
            key=lambda k: viable[k]["development"]["brier"]
        )
        selected = viable[selected_name]
        # Reconstruct selected candidate's full walk-forward OOS series for
        # current calibration, still only using matured historical outcomes.
        wf = walk_forward(Xh, yh, horizon, factories()[selected_name])
        current = latest_estimate(
            Xh, yh, horizon, factories()[selected_name],
            wf["probability"], wf["actual"]
        ) if wf else {"status": "insufficient"}
        experiments[horizon_name] = {
            "status": "evaluated",
            "horizon_sessions": horizon,
            "selected_candidate": selected_name,
            "selection_basis": "development Brier only; final chronological holdout not used for candidate selection",
            "selected_candidate_results": selected,
            "candidates": candidates_result,
            "promotion_gate": {
                "requires_positive_holdout_brier_skill": True,
                "requires_reliability_bins_to_track_estimated_probabilities": True,
                "requires_multiple_year_blocks": True,
                "requires_comparison_to_existing_strategy_on_same_horizon": True,
                "requires_untouched_followup_period": True,
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
