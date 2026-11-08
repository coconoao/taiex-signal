"""Isolated model-plus-holding-rule strategy research.

Signals are generated out-of-sample in chronological order. A positive research
classification schedules a hypothetical entry at the next session close. The
position exits when the close-to-entry gain reaches a tested threshold, or at
the fixed maximum holding period. No production files are modified.
"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

import model_improvement_research as mir
import pipeline

ENTRY_THRESHOLD = 0.55
PROFIT_TARGETS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05)
MAX_HOLD_SESSIONS = 20
COST_PER_SIDE = float(pipeline.COST)
MIN_TRAIN = mir.MIN_TRAIN
STEP = mir.STEP


def walk_forward_probabilities(X, y, factory, step=STEP):
    """Return strictly out-of-sample probabilities on the candidate calendar."""
    valid = X.notna().all(axis=1) & y.notna()
    idx = X.index[valid]
    if len(idx) <= MIN_TRAIN + 30:
        return pd.Series(dtype=float), {"status": "insufficient", "available_samples": int(len(idx))}
    pred = pd.Series(np.nan, index=idx, dtype=float)
    for i in range(MIN_TRAIN, len(idx), step):
        train_idx = idx[:i]
        test_idx = idx[i:i + step]
        y_train = y.loc[train_idx].astype(int)
        if y_train.nunique() < 2:
            continue
        model = factory()
        model.fit(X.loc[train_idx], y_train)
        pred.loc[test_idx] = model.predict_proba(X.loc[test_idx])[:, 1]
    pred = pred.dropna()
    if len(pred) < 30:
        return pred, {"status": "insufficient", "sample_count": int(len(pred))}
    return pred, {"status": "available", "sample_count": int(len(pred)),
                  "sample_start": str(pred.index.min().date()),
                  "sample_end": str(pred.index.max().date())}


def _curve_metrics(daily_returns):
    r = pd.Series(daily_returns, dtype=float).fillna(0.0)
    if r.empty:
        return {}
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    total = float(equity.iloc[-1] - 1)
    years = max(len(r) / 252.0, 1 / 252.0)
    annualized = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 and r.std(ddof=1) > 0 else None
    return {
        "total_return_pct": round(total * 100, 2),
        "annualized_return_pct": round(annualized * 100, 2),
        "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
        "annualized_volatility_pct": round(vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
    }


def simulate_profit_target(prob, close, target, max_hold=MAX_HOLD_SESSIONS):
    """Long-only event simulation; signals at t enter at t+1 close, no lookahead."""
    prob = prob.dropna().sort_index()
    close = close.dropna().sort_index()
    dates = close.index
    if len(prob) < 30:
        return {"status": "insufficient", "sample_count": int(len(prob))}
    first_signal = dates.get_indexer([prob.index.min()])[0]
    last_signal = dates.get_indexer([prob.index.max()])[0]
    if first_signal < 0 or last_signal < 0 or last_signal + 1 >= len(dates):
        return {"status": "insufficient", "sample_count": int(len(prob)),
                "note": "樣本外訊號日期無法完整對齊收盤資料。"}
    start_i, end_i = first_signal + 1, last_signal + 1
    period_dates = dates[start_i:end_i + 1]
    if len(period_dates) < 2:
        return {"status": "insufficient", "sample_count": int(len(prob))}
    daily = pd.Series(0.0, index=period_dates)
    signal_map = prob.to_dict()
    trades = []
    i = start_i - 1
    while i <= end_i - 1:
        signal_date = dates[i]
        p = signal_map.get(signal_date)
        if p is None or float(p) < ENTRY_THRESHOLD or i + 1 > end_i:
            i += 1
            continue
        entry_i = i + 1
        entry_price = float(close.iloc[entry_i])
        exit_i = min(end_i, entry_i + max_hold)
        reason = "最長持有期限"
        for k in range(entry_i + 1, exit_i + 1):
            gain = float(close.iloc[k] / entry_price - 1)
            if gain >= target:
                exit_i = k
                reason = "達到報酬門檻"
                break
        exit_price = float(close.iloc[exit_i])
        gross = exit_price / entry_price - 1
        net_factor = (1 - COST_PER_SIDE) * (exit_price / entry_price) * (1 - COST_PER_SIDE)
        net = net_factor - 1
        # Entry cost is charged at entry close; subsequent daily price changes
        # are marked to market; exit cost is charged at exit close.
        daily.loc[dates[entry_i]] -= COST_PER_SIDE
        for k in range(entry_i + 1, exit_i + 1):
            daily.loc[dates[k]] += float(close.iloc[k] / close.iloc[k - 1] - 1)
        daily.loc[dates[exit_i]] -= COST_PER_SIDE
        trades.append({
            "entry_date": str(dates[entry_i].date()),
            "exit_date": str(dates[exit_i].date()),
            "gross_return_pct": round(gross * 100, 3),
            "net_return_pct": round(net * 100, 3),
            "holding_sessions": int(exit_i - entry_i),
            "exit_reason": reason,
        })
        i = exit_i
    metrics = _curve_metrics(daily)
    bh = close.loc[period_dates]
    bh_daily = bh.pct_change().fillna(0.0)
    if len(bh_daily):
        bh_daily.iloc[0] -= COST_PER_SIDE
        bh_daily.iloc[-1] -= COST_PER_SIDE
    benchmark = _curve_metrics(bh_daily)
    wins = sum(t["net_return_pct"] > 0 for t in trades)
    return {
        "status": "available",
        "sample_count": int(len(prob)),
        "period_start": str(period_dates[0].date()),
        "period_end": str(period_dates[-1].date()),
        "period_sessions": int(len(period_dates)),
        "entry_threshold_category": "偏正向研究分類（內部門檻 0.55）",
        "profit_target_pct": round(target * 100, 2),
        "max_holding_sessions": int(max_hold),
        "cost_per_side_pct": round(COST_PER_SIDE * 100, 3),
        "trade_count": len(trades),
        "trade_win_rate_pct": round(wins / len(trades) * 100, 2) if trades else None,
        "average_holding_sessions": round(float(np.mean([t["holding_sessions"] for t in trades])), 2) if trades else None,
        "strategy": metrics,
        "buy_and_hold": benchmark,
        "trades": trades,
    }


def run():
    market, chips, margin, futures = mir.load_inputs()
    X_raw, y, _ = pipeline.build(market, chips, margin, futures=futures)
    close = market["TAIEX"].dropna().astype(float)
    market_cols = [c for c in X_raw.columns
                   if not c.startswith(("籌碼:", "信用:", "期貨:"))]
    specs = mir.candidates()
    feature_sets = {
        "logistic_current_all_features": (X_raw, specs["logistic_current_step5"]),
        "logistic_market_only": (X_raw[market_cols], specs["logistic_market_features_only"]),
        "logistic_stronger_regularization": (X_raw, specs["logistic_stronger_regularization"]),
        "logistic_weaker_regularization": (X_raw, specs["logistic_weaker_regularization"]),
        "random_forest_conservative": (X_raw, specs["random_forest_conservative"]),
        "hist_gradient_boosting_conservative": (X_raw, specs["hist_gradient_boosting_conservative"]),
        "market_plus_chips": (X_raw[market_cols + [c for c in X_raw if c.startswith("籌碼:")]], specs["logistic_market_features_only"]),
        "market_plus_margin": (X_raw[market_cols + [c for c in X_raw if c.startswith("信用:")]], specs["logistic_market_features_only"]),
        "market_plus_futures": (X_raw[market_cols + [c for c in X_raw if c.startswith("期貨:")]], specs["logistic_market_features_only"]),
    }
    experiments = {}
    for name, (features, factory) in feature_sets.items():
        pred, meta = walk_forward_probabilities(features, y, factory)
        if meta["status"] != "available":
            experiments[name] = {"status": "insufficient", **meta, "variants": []}
            continue
        variants = []
        for target in PROFIT_TARGETS:
            result = simulate_profit_target(pred, close, target)
            variants.append(result)
        experiments[name] = {
            "status": "evaluated",
            "prediction_samples": meta["sample_count"],
            "prediction_start": meta["sample_start"],
            "prediction_end": meta["sample_end"],
            "entry_rule": "僅在樣本外模型分類達偏正向時，於下一交易日收盤建立假設持有；不使用當日收盤前不可得資料。",
            "exit_rule": "收盤價相對進場收盤價達到測試門檻時退出；未達門檻則於最多 20 個交易日後退出。",
            "variants": variants,
        }
    return {
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated" if any(v.get("status") == "evaluated" for v in experiments.values()) else "insufficient",
        "target": "TAIEX model-plus-profit-target holding strategy",
        "entry_threshold_internal": ENTRY_THRESHOLD,
        "profit_targets_pct": [round(x * 100, 2) for x in PROFIT_TARGETS],
        "max_holding_sessions": MAX_HOLD_SESSIONS,
        "transaction_cost_per_side_pct": round(COST_PER_SIDE * 100, 3),
        "execution_assumption": "每日收盤資料研究；訊號於 t 日收盤後形成，最早於 t+1 收盤建立假設部位；不模擬盤中觸價、稅費差異、借券或融資。",
        "method": "Chronological expanding-window walk-forward; minimum 100 prior labeled samples; refit every 5 sessions; entry only on positive research classification; profit-target exit or fixed 20-session time exit; same-period buy-and-hold benchmark; transaction cost applied on entry and exit; no random shuffle.",
        "experiments": experiments,
        "disclaimer": "Historical strategy research only. Results are not forecasts, investment advice, or instructions to trade. Parameter sweeps can overfit historical data; untouched future holdout and robustness checks are required.",
    }


if __name__ == "__main__":
    result = run()
    tmp = "strategy_research.json.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, "strategy_research.json")
    for name, item in result["experiments"].items():
        print(name, item.get("status"), len(item.get("variants", [])))
