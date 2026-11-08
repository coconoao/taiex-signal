"""個股／ETF 官方歷史資料與中長期研究：python stock.py 2330

上市日線來自證交所、上櫃日線來自櫃買中心；大盤基準來自證交所加權指數歷史資料。
模型使用依官方漲跌價差重建的連續調整價格，避免除權息造成不合理跳空。
此資料介面提供歷史日線，不等同盤中即時行情。
"""
import sys
import json

import numpy as np
import pandas as pd

import tw_market_data

HISTORY_YEARS = 4


def _historical_horizon_summary(close, horizon):
    forward = (close.shift(-horizon) / close - 1).dropna()
    if forward.empty:
        return None
    return {
        "n": int(len(forward)),
        "positive_pct": round(float((forward > 0).mean()) * 100, 1),
        "average_return_pct": round(float(forward.mean()) * 100, 2),
        "median_return_pct": round(float(forward.median()) * 100, 2),
        "p10_pct": round(float(forward.quantile(0.10)) * 100, 2),
        "p90_pct": round(float(forward.quantile(0.90)) * 100, 2),
        "overlapping_windows": True,
    }


def analyze(code):
    code = str(code or "").strip().upper()
    exchange_hint = None
    for suffix, exchange in ((".TWO", "TPEX"), (".TW", "TWSE")):
        if code.endswith(suffix):
            code = code[:-len(suffix)]
            exchange_hint = exchange
            break
    if not code or not code[0].isdigit() or not code.isalnum() or len(code) > 12:
        raise ValueError("請輸入台股代號，例如 2330、0050 或 00679B")

    stock_data = tw_market_data.load_stock_history(code, exchange_hint, years=HISTORY_YEARS)

    import os
    if os.path.exists("signals.json"):
        try:
            with open("signals.json", encoding="utf-8") as stream:
                if json.load(stream).get("demo"):
                    raise ValueError("目前仍是示範行情，請先更新真實行情與訊號，再使用個股分析。")
        except json.JSONDecodeError:
            raise ValueError("行情資料檔案格式錯誤，請重新更新行情與訊號。")

    # Close is a continuous adjusted series reconstructed from the official
    # exchange's daily reference-price change; RawClose remains the quoted close.
    px = stock_data["Close"].dropna()
    raw_px = stock_data["RawClose"].dropna()
    volume = stock_data["Volume"].reindex(px.index)
    mk = tw_market_data.load_taiex_history(years=HISTORY_YEARS).dropna()

    f5 = px.shift(-5) / px - 1

    def summarize(values):
        values = values.dropna()
        if values.empty:
            return None
        return {
            "n": int(len(values)),
            "win": round(float((values > 0).mean()) * 100, 1),
            "avg": round(float(values.mean()) * 100, 2),
            "p10": round(float(values.quantile(0.1)) * 100, 2),
            "p90": round(float(values.quantile(0.9)) * 100, 2),
        }

    stock_expected = summarize(f5)
    returns = pd.concat(
        [px.pct_change(), mk.pct_change()], axis=1, keys=["s", "m"]
    ).dropna().tail(252)
    if len(returns) < 30:
        raise ValueError("此標的與加權指數的共同歷史資料不足 30 個交易日，暫時無法計算 Beta。")
    market_var = float(returns["m"].var())
    if not np.isfinite(market_var) or market_var <= 0:
        raise ValueError("無法計算 Beta：大盤報酬資料不足或變異數為零。")
    beta = round(float(returns["s"].cov(returns["m"]) / market_var), 2)

    asset_type = "ETF" if code.startswith(("00", "01", "02")) else "股票"

    import stock_model_research
    long_term = stock_model_research.analyze(px, mk, volume)
    historical_horizons = {
        "1m": _historical_horizon_summary(px, 21),
        "3m": _historical_horizon_summary(px, 63),
        "6m": _historical_horizon_summary(px, 126),
    }

    return {
        "code": code,
        "symbol": code,
        "asset_type": asset_type,
        "last": round(float(raw_px.iloc[-1]), 2),
        "adjusted_last": round(float(px.iloc[-1]), 4),
        "asof": str(raw_px.index[-1].date()),
        "history_asof": str(raw_px.index[-1].date()),
        "quote_time": None,
        "quote_source": stock_data.attrs.get("source", "TWSE/TPEx official"),
        "market_state": "CLOSED",
        "history_start": str(px.index[0].date()),
        "history_sessions": int(len(px)),
        "history_years_requested": HISTORY_YEARS,
        "market_data_source": stock_data.attrs.get("source", "TWSE/TPEx official"),
        "price_basis": "official_reference_price_chained_adjusted",
        "quote_realtime": False,
        "quote_status": "目前顯示交易所最近一個完整收盤日線價格；尚未接入獲授權的盤中行情來源。",
        "beta": beta,
        "stock_expected": stock_expected,
        "historical_horizons": historical_horizons,
        "long_term_model": long_term,
        "mom20": round(float(px.iloc[-1] / px.iloc[-21] - 1) * 100, 1),
        "vol60": round(float(px.pct_change().tail(60).std() * np.sqrt(252)) * 100, 1),
        "fundamentals": [],
    }


if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), ensure_ascii=False, indent=1))
