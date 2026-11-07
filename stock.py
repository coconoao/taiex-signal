"""個股上漲幅度估計：python stock.py 2330
需先執行 pipeline.py（產生 weekly.json、weekly_prob.csv）。
方法：歷史上大盤出現「同樣一週訊號」時，這檔股票之後 5 日的實際報酬分布（平均、勝率、八成區間），不是點預測。
財報體檢另列參考：沒有「當時公布」的歷史財報可回測，所以不調整上面的區間。
"""
import sys, json
import numpy as np, pandas as pd

def _hist(yf, sym):
    last_error = None
    for attempt in range(3):
        try:
            raw = yf.Ticker(sym).history(period="5y", auto_adjust=True)
            if raw is not None and not raw.empty and "Close" in raw:
                h = raw["Close"].dropna()
                h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
                if len(h):
                    return h
            last_error = ValueError("Yahoo Finance 回傳空行情")
        except Exception as e:
            last_error = e
        if attempt < 2:
            import time
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{sym} 行情下載失敗；可能遇到 Yahoo Finance 限流（HTTP 429）。最後錯誤：{last_error}")

def fundamentals(t):
    out, info = [], {}
    try: info = t.info or {}
    except Exception: pass
    def add(item, val, good=None, fmt="{:.1f}%"):
        if val is None or (isinstance(val, float) and np.isnan(val)): return
        out.append({"item": item, "value": fmt.format(val), "flag": 0 if good is None else (1 if good else -1)})
    try:
        q = t.quarterly_income_stmt
        row = lambda *ns: next((q.loc[n].dropna().sort_index() for n in ns if n in q.index), None)
        rev, ni, gp, op = row("Total Revenue", "Operating Revenue"), row("Net Income"), row("Gross Profit"), row("Operating Income")
        if rev is not None and len(rev) >= 5:
            yoy = (rev.iloc[-1] / rev.iloc[-5] - 1) * 100
            add("最近一季營收年增率", yoy, yoy > 0)
            for nm, s in (("毛利率", gp), ("營業利益率", op), ("稅後淨利率", ni)):
                if s is not None and len(s) >= 5:
                    now, ago = s.iloc[-1] / rev.iloc[-1] * 100, s.iloc[-5] / rev.iloc[-5] * 100
                    add(f"{nm}（去年同季 {ago:.1f}%）", now, now > ago)
    except Exception:
        pass
    add("本益比（近四季）", info.get("trailingPE"), None, "{:.1f}")
    add("預估本益比", info.get("forwardPE"), None, "{:.1f}")
    roe = info.get("returnOnEquity")
    add("股東權益報酬率", None if roe is None else roe * 100, None if roe is None else roe > 0.08)
    de = info.get("debtToEquity")
    add("負債／權益", de, None if de is None else de < 100)
    return out

def analyze(code):
    import yfinance as yf
    code = str(code or "").strip().upper()
    for suffix in (".TWO", ".TW"):
        if code.endswith(suffix):
            code = code[:-len(suffix)]
            break
    if not code or not code[0].isdigit() or not code.isalnum() or len(code) > 12:
        raise ValueError("請輸入台股代號，例如 2330、0050 或 00679B")
    px = sym = None
    errors = []
    for suf in (".TW", ".TWO"):  # 上市、上櫃；其中一個市場代碼無資料時繼續嘗試另一個
        try:
            h = _hist(yf, code + suf)
        except RuntimeError as e:
            errors.append(str(e))
            continue
        # 新上市 ETF 可能沒有五年資料；至少保留 60 個交易日供基本歷史統計。
        if len(h) >= 60:
            px, sym = h, code + suf
            break
    if px is None:
        detail = "；".join(errors)
        raise ValueError(f"找不到 {code} 的足夠歷史行情，請確認代號或稍後重試。{detail}")
    import os
    if os.path.exists("signals.json"):
        try:
            if json.load(open("signals.json", encoding="utf-8")).get("demo"):
                raise ValueError("目前仍是示範行情，請先更新真實行情與訊號，再使用個股分析。")
        except json.JSONDecodeError:
            raise ValueError("行情資料檔案格式錯誤，請重新更新行情與訊號。")
    mk = _hist(yf, "^TWII")
    f5 = (px.shift(-5) / px - 1)
    q = lambda values, a: round(float(values.quantile(a)) * 100, 2)
    def summarize(values):
        values = values.dropna()
        if values.empty: return None
        return {"n": int(len(values)), "win": round(float((values > 0).mean()) * 100, 1),
                "avg": round(float(values.mean()) * 100, 2),
                "p10": q(values, .1), "p90": q(values, .9)}
    stock_expected = summarize(f5)
    r = pd.concat([px.pct_change(), mk.pct_change()], axis=1, keys=["s", "m"]).dropna().tail(252)
    if len(r) < 30:
        raise ValueError("此標的與加權指數的共同歷史資料不足 30 個交易日，暫時無法計算 Beta。")
    market_var = float(r["m"].var())
    if not np.isfinite(market_var) or market_var <= 0:
        raise ValueError("無法計算 Beta：大盤報酬資料不足或變異數為零。")
    beta = round(float(r["s"].cov(r["m"]) / market_var), 2)
    ticker = yf.Ticker(sym)
    try:
        quote_type = (ticker.info or {}).get("quoteType", "")
    except Exception:
        quote_type = ""
    asset_type = "ETF" if str(quote_type).upper() == "ETF" else "股票"
    return {"code": code, "symbol": sym, "asset_type": asset_type,
            "last": round(float(px.iloc[-1]), 2), "asof": str(px.index[-1].date()),
            "beta": beta, "stock_expected": stock_expected,
            "mom20": round(float(px.iloc[-1] / px.iloc[-21] - 1) * 100, 1),
            "vol60": round(float(px.pct_change().tail(60).std() * np.sqrt(252)) * 100, 1),
            "fundamentals": [] if asset_type == "ETF" else fundamentals(ticker)}

if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), ensure_ascii=False, indent=1))
