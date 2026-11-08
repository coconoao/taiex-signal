"""個股／ETF 歷史資料與中長期研究：python stock.py 2330

預設取得可用的 10 年日線資料，提供 1m/3m/6m 的個股專屬研究模型。
歷史財報另列參考，不納入模型特徵，避免把缺乏歷史公告時點的財報欄位誤當成 point-in-time 資料。
"""
import sys
import json

import numpy as np
import pandas as pd


def _hist(yf, sym):
    last_error = None
    for attempt in range(2):
        try:
            # Bound each provider request so a stalled Yahoo response cannot occupy
            # the web worker indefinitely. Retry once for transient failures only.
            raw = yf.Ticker(sym).history(period="10y", auto_adjust=True, timeout=10)
            if raw is not None and not raw.empty and "Close" in raw:
                columns = ["Close"] + (["Volume"] if "Volume" in raw.columns else [])
                frame = raw[columns].dropna(subset=["Close"]).copy()
                frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
                frame = frame[~frame.index.duplicated(keep="last")].sort_index()
                if len(frame):
                    return frame
            last_error = ValueError("Yahoo Finance 回傳空行情")
        except Exception as exc:
            last_error = exc
            # Authentication/permission failures will not recover by retrying.
            if getattr(exc, "code", None) in (401, 403) or "HTTP Error 401" in str(exc) or "HTTP Error 403" in str(exc):
                break
        if attempt < 1:
            import time
            time.sleep(1.5)
    raise RuntimeError(
        f"{sym} 行情下載失敗；可能遇到 Yahoo Finance 限流（HTTP 429）。最後錯誤：{last_error}"
    )


def fundamentals(t):
    out, info = [], {}
    try:
        info = t.info or {}
    except Exception:
        pass

    def add(item, val, good=None, fmt="{:.1f}%"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return
        out.append({
            "item": item, "value": fmt.format(val),
            "flag": 0 if good is None else (1 if good else -1)
        })

    try:
        q = t.quarterly_income_stmt
        row = lambda *names: next((q.loc[name].dropna().sort_index() for name in names if name in q.index), None)
        rev = row("Total Revenue", "Operating Revenue")
        ni, gp, op = row("Net Income"), row("Gross Profit"), row("Operating Income")
        if rev is not None and len(rev) >= 5:
            yoy = (rev.iloc[-1] / rev.iloc[-5] - 1) * 100
            add("最近一季營收年增率", yoy, yoy > 0)
            for name, series in (("毛利率", gp), ("營業利益率", op), ("稅後淨利率", ni)):
                if series is not None and len(series) >= 5:
                    now = series.iloc[-1] / rev.iloc[-1] * 100
                    ago = series.iloc[-5] / rev.iloc[-5] * 100
                    add(f"{name}（去年同季 {ago:.1f}%）", now, now > ago)
    except Exception:
        pass

    add("本益比（近四季）", info.get("trailingPE"), None, "{:.1f}")
    add("預估本益比", info.get("forwardPE"), None, "{:.1f}")
    roe = info.get("returnOnEquity")
    add("股東權益報酬率", None if roe is None else roe * 100,
        None if roe is None else roe > 0.08)
    debt_equity = info.get("debtToEquity")
    add("負債／權益", debt_equity,
        None if debt_equity is None else debt_equity < 100)
    return out


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
    import yfinance as yf

    code = str(code or "").strip().upper()
    for suffix in (".TWO", ".TW"):
        if code.endswith(suffix):
            code = code[:-len(suffix)]
            break
    if not code or not code[0].isdigit() or not code.isalnum() or len(code) > 12:
        raise ValueError("請輸入台股代號，例如 2330、0050 或 00679B")

    stock_data = sym = None
    errors = []
    for suffix in (".TW", ".TWO"):
        try:
            history = _hist(yf, code + suffix)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        # 新上市標的可能沒有完整十年資料；至少保留 60 個交易日供基本歷史摘要。
        if len(history) >= 60:
            stock_data, sym = history, code + suffix
            break
    if stock_data is None:
        detail = "；".join(errors)
        raise ValueError(f"找不到 {code} 的足夠歷史行情，請確認代號或稍後重試。{detail}")

    import os
    if os.path.exists("signals.json"):
        try:
            with open("signals.json", encoding="utf-8") as stream:
                if json.load(stream).get("demo"):
                    raise ValueError("目前仍是示範行情，請先更新真實行情與訊號，再使用個股分析。")
        except json.JSONDecodeError:
            raise ValueError("行情資料檔案格式錯誤，請重新更新行情與訊號。")

    px = stock_data["Close"].dropna()
    mk = _hist(yf, "^TWII")["Close"].dropna()
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

    ticker = yf.Ticker(sym)
    try:
        quote_type = (ticker.info or {}).get("quoteType", "")
    except Exception:
        quote_type = ""
    asset_type = "ETF" if str(quote_type).upper() == "ETF" else "股票"

    import stock_model_research
    long_term = stock_model_research.analyze(
        px, mk, stock_data["Volume"] if "Volume" in stock_data else None
    )
    historical_horizons = {
        "1m": _historical_horizon_summary(px, 21),
        "3m": _historical_horizon_summary(px, 63),
        "6m": _historical_horizon_summary(px, 126),
    }

    return {
        "code": code,
        "symbol": sym,
        "asset_type": asset_type,
        "last": round(float(px.iloc[-1]), 2),
        "asof": str(px.index[-1].date()),
        "history_start": str(px.index[0].date()),
        "history_sessions": int(len(px)),
        "history_years_requested": 10,
        "beta": beta,
        "stock_expected": stock_expected,
        "historical_horizons": historical_horizons,
        "long_term_model": long_term,
        "mom20": round(float(px.iloc[-1] / px.iloc[-21] - 1) * 100, 1),
        "vol60": round(float(px.pct_change().tail(60).std() * np.sqrt(252)) * 100, 1),
        "fundamentals": [] if asset_type == "ETF" else fundamentals(ticker),
    }


if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), ensure_ascii=False, indent=1))
