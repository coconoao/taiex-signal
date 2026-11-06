"""個股上漲幅度估計：python stock.py 2330
需先執行 pipeline.py（產生 weekly.json、weekly_prob.csv）。
方法：歷史上大盤出現「同樣一週訊號」時，這檔股票之後 5 日的實際報酬分布（平均、勝率、八成區間），不是點預測。
財報體檢另列參考：沒有「當時公布」的歷史財報可回測，所以不調整上面的區間。
"""
import sys, json
import numpy as np, pandas as pd
from pipeline import BUY, SELL

def _hist(yf, sym):
    h = yf.Ticker(sym).history(period="5y", auto_adjust=True)["Close"]
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    return h.dropna()

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
    if not (os.path.exists("weekly.json") and os.path.exists("weekly_prob.csv")):
        raise ValueError("請先到「設定」頁按「更新行情與訊號」（或執行 pipeline.py）")
    mk = _hist(yf, "^TWII")
    wk = json.load(open("weekly.json", encoding="utf-8"))
    p = pd.read_csv("weekly_prob.csv", index_col=0, parse_dates=True)["prob"].dropna()
    f5 = (px.shift(-5) / px - 1).reindex(p.index)
    name = lambda v: "偏多" if v > BUY else "偏空" if v < SELL else "中性"
    d = pd.DataFrame({"b": p.map(name), "f": f5}).dropna()
    q = lambda s, a: round(float(s.quantile(a)) * 100, 2)
    buckets = {k: {"n": int(len(g)), "win": round(float((g > 0).mean()) * 100, 1), "avg": round(float(g.mean()) * 100, 2),
                   "p10": q(g, .1), "p90": q(g, .9)} for k, g in d.groupby("b")["f"]}
    r = pd.concat([px.pct_change(), mk.pct_change()], axis=1, keys=["s", "m"]).dropna().tail(252)
    beta = round(float(r["s"].cov(r["m"]) / r["m"].var()), 2)
    exp = wk.get("expected")
    return {"code": code, "symbol": sym, "last": round(float(px.iloc[-1]), 2), "asof": str(px.index[-1].date()),
            "market_bucket": wk["bucket"], "market_prob": wk["prob_up"], "beta": beta,
            "beta_implied": round(beta * exp["avg"], 2) if exp else None,
            "stock_expected": buckets.get(wk["bucket"]), "buckets": buckets,
            "mom20": round(float(px.iloc[-1] / px.iloc[-21] - 1) * 100, 1),
            "vol60": round(float(px.pct_change().tail(60).std() * np.sqrt(252)) * 100, 1),
            "fundamentals": fundamentals(yf.Ticker(sym))}

if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), ensure_ascii=False, indent=1))
