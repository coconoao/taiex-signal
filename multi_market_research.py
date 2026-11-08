"""Independent out-of-sample research for major equity indices.

This is a separate research experiment. It does not alter the Taiwan model,
its features, thresholds, signals, or backtest. All cross-market features are
lagged by one target-market session to avoid close-time look-ahead leakage.
"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MARKETS = {
    "SP500": {"ticker": "^GSPC", "label": "美國 S&P 500"},
    "NASDAQ": {"ticker": "^IXIC", "label": "美國 Nasdaq Composite"},
    "NIKKEI225": {"ticker": "^N225", "label": "日本 Nikkei 225"},
    "KOSPI": {"ticker": "^KS11", "label": "韓國 KOSPI"},
}
STEP = 5
MIN_TRAIN = 500


def extract_close(raw, tickers):
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance 沒有回傳價格資料")
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0))
        level1 = list(raw.columns.get_level_values(1))
        if "Close" in level0:
            close = raw["Close"]
        elif "Close" in level1:
            close = raw.xs("Close", axis=1, level=1)
        else:
            raise ValueError("下載資料缺少 Close 欄位")
    else:
        if len(tickers) != 1 or "Close" not in raw:
            raise ValueError("下載資料欄位格式不符")
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.apply(pd.to_numeric, errors="coerce")


def download_prices():
    import yfinance as yf
    tickers = [v["ticker"] for v in MARKETS.values()]
    raw = yf.download(
        tickers, period="max", auto_adjust=True, progress=False,
        group_by="column", threads=False,
    )
    close = extract_close(raw, tickers)
    rename = {v["ticker"]: k for k, v in MARKETS.items()}
    close = close.rename(columns=rename)
    return close.reindex(columns=list(MARKETS))


def make_dataset(prices, target):
    target_price = prices[target].dropna()
    target_index = target_price.index
    returns = prices.pct_change(fill_method=None)
    X = pd.DataFrame(index=target_index)
    own = returns[target]
    X["own_return_1d"] = own.reindex(target_index)
    X["own_return_5d"] = target_price.pct_change(5)
    X["own_volatility_10d"] = own.rolling(10).std().reindex(target_index)
    for market in MARKETS:
        if market == target:
            continue
        # Conservative one-session lag across markets with different closing times.
        lagged = returns[market].shift(1).reindex(target_index).ffill(limit=3)
        X[f"{market}_return_lag1"] = lagged
        X[f"{market}_momentum_5d_lag1"] = (
            prices[market].pct_change(5).shift(1).reindex(target_index).ffill(limit=3)
        )
    forward = target_price.shift(-1) / target_price - 1
    y = (forward > 0).astype(float).where(forward.notna())
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid], forward.loc[valid]


def evaluate_market(prices, market):
    X, y, forward = make_dataset(prices, market)
    n = len(X)
    if n <= MIN_TRAIN + 10:
        return {
            "status": "insufficient", "sample_count": 0, "available_rows": n,
            "minimum_training_rows": MIN_TRAIN,
            "note": "完整特徵資料不足，未降低樣本門檻。",
        }
    predictions = pd.Series(np.nan, index=X.index, dtype=float)
    # Refit every five target-market sessions. Each block uses only labels
    # available before its first prediction date.
    for i in range(MIN_TRAIN, n, STEP):
        train_idx = X.index[:i]
        test_idx = X.index[i:i + STEP]
        if len(test_idx) == 0 or y.loc[train_idx].nunique() < 2:
            continue
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=500),
        )
        estimator.fit(X.loc[train_idx], y.loc[train_idx].astype(int))
        predictions.loc[test_idx] = estimator.predict_proba(X.loc[test_idx])[:, 1]
    p = predictions.dropna()
    if len(p) < 30:
        return {
            "status": "insufficient", "sample_count": int(len(p)),
            "available_rows": n, "minimum_training_rows": MIN_TRAIN,
            "note": "樣本外預測不足 30 筆，不顯示評估指標。",
        }
    actual = y.loc[p.index].astype(int)
    baseline = actual.expanding(min_periods=1).mean().shift(1).fillna(0.5)
    brier = float(((p - actual) ** 2).mean())
    baseline_brier = float(((baseline - actual) ** 2).mean())
    clipped = np.clip(p.to_numpy(), 1e-6, 1 - 1e-6)
    log_loss = float(-(actual.to_numpy() * np.log(clipped) +
                       (1 - actual.to_numpy()) * np.log(1 - clipped)).mean())
    baseline_direction = float(((baseline > 0.5) == (actual > 0.5)).mean() * 100)
    return {
        "status": "available",
        "sample_count": int(len(p)),
        "available_rows": int(n),
        "train_minimum": MIN_TRAIN,
        "refit_every_sessions": STEP,
        "sample_start": str(p.index.min().date()),
        "sample_end": str(p.index.max().date()),
        "data_start": str(X.index.min().date()),
        "data_end": str(X.index.max().date()),
        "signal_coverage_pct": round(float(len(p) / n * 100), 1),
        "direction_accuracy_pct": round(float(((p > 0.5) == (actual > 0)).mean() * 100), 1),
        "baseline_direction_accuracy_pct": round(baseline_direction, 1),
        "brier": round(brier, 4),
        "baseline_brier": round(baseline_brier, 4),
        "brier_skill": round(1 - brier / baseline_brier, 4) if baseline_brier > 0 else None,
        "log_loss": round(log_loss, 4),
        "note": "樣本外研究分類；不是未來漲跌機率保證或交易訊號。",
    }


def run_research(prices):
    results = {}
    for market, meta in MARKETS.items():
        if market not in prices or prices[market].dropna().empty:
            results[market] = {
                "status": "unavailable", "sample_count": 0,
                "note": "此市場資料來源未回傳有效價格。",
            }
            continue
        try:
            results[market] = evaluate_market(prices, market)
        except Exception as exc:
            results[market] = {
                "status": "error", "sample_count": 0,
                "note": f"計算失敗：{type(exc).__name__}",
            }
    return {
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_source": "Yahoo Finance daily adjusted close",
        "method": "Expanding-window walk-forward; refit every 5 target sessions; 500-session minimum training window; next-session target; all cross-market features lagged one target session.",
        "markets": results,
        "disclaimer": "獨立歷史樣本外研究，不改動台股原始模型；統計結果不代表未來表現，不是投資建議或買賣訊號。",
    }


if __name__ == "__main__":
    prices = download_prices()
    result = run_research(prices)
    tmp = "multi_market_research.json.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, "multi_market_research.json")
    for key, value in result["markets"].items():
        print(f"{key}: {value.get('status')} | OOS samples={value.get('sample_count')} | accuracy={value.get('direction_accuracy_pct', '—')}")
