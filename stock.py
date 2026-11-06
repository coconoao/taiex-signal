"""個股上漲幅度估計：python stock.py 2330
需先執行 pipeline.py（產生 weekly.json、weekly_prob.csv）。
方法：歷史上大盤出現「同樣一週訊號」時，這檔股票之後 5 日的實際報酬分布（平均、勝率、八成區間），不是點預測。
財報體檢另列參考：沒有「當時公布」的歷史財報可回測，所以不調整上面的區間。
"""
import sys, json
import numpy as np, pandas as pd
from pipeline import BUY, SELL

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
    code = code.strip().upper().replace(".TWO", "").replace(".TW", "")
    if not code.isalnum(): raise ValueError("請輸入股票代號，例如 2330")
    px = sym = None
    for suf in (".TW", ".TWO"):  # 上市、上櫃
        h = _hist(yf, code + suf)
        if len(h) > 300: px, sym = h, code + suf; break
    if px is None: raise ValueError(f"找不到 {code} 的行情，請確認代號")
    import os
    if os.path.exists("signals.json"):
        try:
            if json.load(open("signals.json", encoding="utf-8")).get("demo"):
                raise ValueError("目前仍是示範行情，請先更新真實行情與訊號，再使用個股分析。")
        except json.JSONDecodeError:
            raise ValueError("行情資料檔案格式錯誤，請重新更新行情與訊號。")
    # 優先使用大盤 Pipeline 的週訊號分組；若尚未產生，仍可執行個股自身歷史分析。
    wk, p = None, None
    try:
        if os.path.exists("weekly.json") and os.path.exists("weekly_prob.csv") and os.path.getsize("weekly_prob.csv") > 0:
            wk = json.load(open("weekly.json", encoding="utf-8"))
            p = pd.read_csv("weekly_prob.csv", index_col=0, parse_dates=True)["prob"].dropna()
            p = p.iloc[::5]  # 與大盤週報一致，降低五日重疊造成的樣本膨脹
    except (ValueError, KeyError, OSError, pd.errors.ParserError) as e:
        print(f"大盤週資料不可用，改用個股自身歷史樣本：{e}")

    mk = _hist(yf, "^TWII")
    f5 = (px.shift(-5) / px - 1)
    q = lambda values, a: round(float(values.quantile(a)) * 100, 2)
    def summarize(values):
        values = values.dropna()
        if values.empty: return None
        return {"n": int(len(values)), "win": round(float((values > 0).mean()) * 100, 1),
                "avg": round(float(values.mean()) * 100, 2),
                "p10": q(values, .1), "p90": q(values, .9)}
    buckets = {}
    market_bucket = "未載入大盤條件"
    market_prob = None
    stock_expected = summarize(f5)
    exp = None
    if p is not None and wk is not None and wk.get("bucket"):
        aligned = f5.reindex(p.index)
        name = lambda v: "偏多" if v > BUY else "偏空" if v < SELL else "中性"
        d = pd.DataFrame({"b": p.map(name), "f": aligned}).dropna()
        buckets = {k: summarize(g["f"]) for k, g in d.groupby("b")}
        market_bucket = wk["bucket"]
        market_prob = wk.get("prob_up")
        stock_expected = buckets.get(market_bucket)
        exp = wk.get("expected")
    r = pd.concat([px.pct_change(), mk.pct_change()], axis=1, keys=["s", "m"]).dropna().tail(252)
    market_var = float(r["m"].var())
    if not np.isfinite(market_var) or market_var <= 0:
        raise ValueError("無法計算 Beta：大盤報酬資料不足或變異數為零。")
    beta = round(float(r["s"].cov(r["m"]) / market_var), 2)
    exp = wk.get("expected") if wk else None
    return {"code": code, "symbol": sym, "last": round(float(px.iloc[-1]), 2), "asof": str(px.index[-1].date()),
            "market_bucket": market_bucket, "market_prob": market_prob, "beta": beta,
            "beta_implied": round(beta * exp["avg"], 2) if exp and exp.get("avg") is not None else None,
            "stock_expected": stock_expected, "buckets": buckets,
            "mom20": round(float(px.iloc[-1] / px.iloc[-21] - 1) * 100, 1),
            "vol60": round(float(px.pct_change().tail(60).std() * np.sqrt(252)) * 100, 1),
            "fundamentals": fundamentals(yf.Ticker(sym))}

if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), ensure_ascii=False, indent=1))
