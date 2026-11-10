"""個股／ETF 官方歷史資料與中長期研究：python stock.py 2330

上市日線來自證交所、上櫃日線來自櫃買中心；大盤基準來自證交所加權指數歷史資料。
模型使用依官方漲跌價差重建的連續調整價格，避免除權息造成不合理跳空。
此資料介面提供歷史日線，不等同盤中即時行情。
"""
import sys
import json
import time

import numpy as np
import pandas as pd

import tw_market_data

HISTORY_YEARS = 4


def _timed_stage(stage, function, *args, **kwargs):
    """Log safe stage timing without symbols, request data, or exception messages."""
    started = time.perf_counter()
    try:
        result = function(*args, **kwargs)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        print(
            f"[stock-analysis] stage={stage} status=failed "
            f"elapsed_ms={elapsed_ms} error_class={type(exc).__name__}",
            flush=True,
        )
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    print(
        f"[stock-analysis] stage={stage} status=ok elapsed_ms={elapsed_ms}",
        flush=True,
    )
    return result


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

    stock_data = _timed_stage(
        "stock_history", tw_market_data.load_stock_history,
        code, exchange_hint, years=HISTORY_YEARS,
    )

    # Intraday data is supplemental; historical research calculations continue
    # to use the validated daily series, never an intraday quote.
    intraday_quote = _timed_stage(
        "intraday_quote", tw_market_data.load_intraday_quote, code
    )
    if intraday_quote.get("status") == "REALTIME":
        quote_status_text = "Fugle 即時行情；更新時間在 5 分鐘內。"
    elif intraday_quote.get("status") == "STALE":
        quote_status_text = "已取得 Fugle 報價，但資料過期或非當日，不能視為即時行情。"
    else:
        quote_status_text = {
            "NO_API_KEY": "尚未設定 Fugle API Key，僅提供歷史日線資料。",
            "AUTH_FAILED": "Fugle API 驗證失敗，僅提供歷史日線資料。",
            "RATE_LIMITED": "Fugle API 已達請求限制，僅提供歷史日線資料。",
            "TIMEOUT": "Fugle API 連線逾時，僅提供歷史日線資料。",
            "INVALID_SYMBOL": "股票代號格式不正確，無法取得即時報價。",
            "INVALID_DATA": "Fugle 回傳資料與查詢代號不符，已拒絕使用。",
            "INVALID_TIMESTAMP": "Fugle 報價缺少有效時間戳記，已拒絕標示為即時。",
            "NO_TRADE_DATA": "Fugle 尚未提供可用成交價格。",
        }.get(intraday_quote.get("status"), "Fugle 行情暫時無法取得，僅提供歷史日線資料。")

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
    mk = _timed_stage(
        "taiex_history", tw_market_data.load_taiex_history,
        years=HISTORY_YEARS,
    ).dropna()

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
    long_term = _timed_stage(
        "research_model", stock_model_research.analyze, px, mk, volume
    )
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
        "quote_time": intraday_quote.get("trade_time"),
        "quote_updated_at": intraday_quote.get("updated_at"),
        "quote_last": intraday_quote.get("price"),
        "quote_change": intraday_quote.get("change"),
        "quote_change_percent": intraday_quote.get("change_percent"),
        "quote_age_seconds": intraday_quote.get("age_seconds"),
        "quote_source": "Fugle" if intraday_quote.get("available") else stock_data.attrs.get("source", "TWSE/TPEx official"),
        "market_state": "REALTIME" if intraday_quote.get("realtime") else "CLOSED_OR_STALE",
        "history_start": str(px.index[0].date()),
        "history_sessions": int(len(px)),
        "history_years_requested": HISTORY_YEARS,
        "market_data_source": stock_data.attrs.get("source", "TWSE/TPEx official"),
        "price_basis": "official_reference_price_chained_adjusted",
        "quote_realtime": bool(intraday_quote.get("realtime")),
        "quote_status": quote_status_text,
        "intraday_quote": intraday_quote,
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
