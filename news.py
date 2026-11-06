"""新聞與機構觀點分析 → news.json（僅供參考，未納入回測模型）。
兩種模式並存：
  1. 內建規則：不需要任何金鑰，永遠執行（只比對字詞，不理解語意，準確度有限）。
  2. AI 模式：在儀表板「設定」頁（或環境變數）填入金鑰後，Claude / ChatGPT / Gemini 會同時分析，結果並排顯示。
     ANTHROPIC_API_KEY（CLAUDE_MODEL 預設 claude-sonnet-5-5）；OPENAI_API_KEY + OPENAI_MODEL；GEMINI_API_KEY + GEMINI_MODEL
資料來源：以英文通訊社標題為主（不用台灣媒體標題判斷）＋下列機構的近 7 日標題。
機構觀點只計入「標題明確寫出該機構對台股方向」者；每日結果累積到 news_history.jsonl，日後可算各機構命中率。
用法：python news.py   （時間同 pipeline.py：台股交易日 8:30 前後）
"""
import os, json, re, datetime as dt, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

MARKET = [("美股", "Wall Street stocks Nasdaq close when:1d"), ("半導體", "TSMC semiconductor chips export when:1d"),
          ("總經", "Fed rates inflation Treasury yields when:1d"), ("台灣總經", "Taiwan central bank OR GDP forecast when:3d"),
          ("地緣", "Taiwan Strait China tensions when:1d")]
# 機構清單可自行增刪；也可加入你信任的個人，例如 ("姓名", "姓名 台股 when:7d")
ANALYSTS = [(n, f"{q} Taiwan stocks when:7d") for n, q in [("高盛", "Goldman Sachs"), ("摩根士丹利", "Morgan Stanley"),
            ("摩根大通", "JPMorgan"), ("花旗", "Citi"), ("瑞銀", "UBS"), ("野村", "Nomura"), ("麥格理", "Macquarie")]]
STANCE = {"看多": 1, "看空": -1, "中性": 0}

POS = {"surge", "surges", "rally", "rallies", "gain", "gains", "rise", "rises", "rose", "beat", "beats", "record", "boost", "soar",
       "jump", "recover", "ceasefire", "easing", "bullish", "overweight", "upgrad", "raises", "optimis"}
NEG = {"fall", "falls", "fell", "drop", "drops", "slump", "plunge", "loss", "losses", "miss", "downgrad", "weak", "fear", "fears",
       "tension", "war", "wars", "tariff", "sanction", "curb", "ban", "bans", "recession", "sell-off", "selloff", "slide", "crash",
       "probe", "bearish", "underweight", "cuts"}

PROMPT = """你是台股分析助理。以下是今天蒐集的新聞標題（id|分類|標題），分成「市場新聞」與「機構觀點」兩部分。只根據標題判斷，不要引用標題以外的資訊；資訊不足時分數接近 0、信心調低。
只輸出 JSON：{"next_day": 數字, "next_week": 數字, "confidence": 0到1, "drivers": [{"id": 標題id, "impact": "正面"或"負面", "why": "20字內理由"}], "summary": "繁體中文，80字內", "analysts": [{"name": 機構名稱, "id": 標題id, "stance": "看多"或"看空"或"中性"或"無明確看法"}]}
分數代表對「台灣加權指數」的影響：-1 極偏空、0 中性、1 極偏多；next_day 指下一個交易日，next_week 指未來 5 個交易日。drivers 最多 4 則，來自市場新聞。
analysts 只列「標題明確寫出該機構對台股或台股大盤的方向性看法」的項目，其他不列，不要從標題推測。
"""

def load_settings():  # 儀表板「設定」頁存下的金鑰
    try: os.environ.update({k: v for k, v in json.load(open("settings.json", encoding="utf-8")).items() if v})
    except Exception: pass

def collect(groups, per, kind, out, seen):
    for tag, q in groups:
        url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(q) + "&hl=en-US&gl=US&ceid=US:en"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            titles = [(i.findtext("title") or "").strip() for i in ET.fromstring(urllib.request.urlopen(req, timeout=20).read()).iter("item")]
        except Exception as e:
            print("新聞來源失敗", tag, e); continue
        n = 0
        for t in titles:
            if t and t not in seen and n < per:
                seen.add(t); n += 1; out.append({"id": len(out), "kind": kind, "tag": tag, "title": t})

def headlines():
    out, seen = [], set()
    collect(MARKET, 8, "market", out, seen)
    collect(ANALYSTS, 3, "analyst", out, seen)
    return out

def tally(title):
    ws = re.findall(r"[a-z\-]+", title.lower())
    hit = lambda S: sum(any(w == s or (len(s) > 4 and w.startswith(s)) for s in S) for w in ws)
    return hit(POS), hit(NEG)

def lexicon(hl):
    p = n = 0
    for h in hl:
        if h["kind"] == "market":
            a, b = tally(h["title"]); p += a > b; n += b > a
    score = round((p - n) / (p + n + 3), 2)  # 標題少或中性時自然趨近 0
    calls = []
    for h in hl:
        if h["kind"] == "analyst":
            a, b = tally(h["title"])
            if a != b: calls.append({"name": h["tag"], "id": h["id"], "stance": "看多" if a > b else "看空"})
    return {"next_day": score, "next_week": score, "confidence": 0.3, "drivers": [],
            "summary": f"內建規則：{p} 則偏多、{n} 則偏空標題（只比對字詞，不理解語意，隔日與一週同分）。", "analysts": calls}

def post(url, headers, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    return json.load(urllib.request.urlopen(req, timeout=90))

def ask_claude(p):
    r = post("https://api.anthropic.com/v1/messages", {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
             {"model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"), "max_tokens": 2000, "messages": [{"role": "user", "content": p}]})
    return r["content"][0]["text"]

def ask_openai(p):
    r = post("https://api.openai.com/v1/chat/completions", {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
             {"model": os.environ["OPENAI_MODEL"], "messages": [{"role": "user", "content": p}], "response_format": {"type": "json_object"}})
    return r["choices"][0]["message"]["content"]

def ask_gemini(p):
    r = post(f"https://generativelanguage.googleapis.com/v1beta/models/{os.environ['GEMINI_MODEL']}:generateContent",
             {"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
             {"contents": [{"parts": [{"text": p}]}], "generationConfig": {"responseMimeType": "application/json"}})
    return "".join(x.get("text", "") for x in r["candidates"][0]["content"]["parts"])

MODELS = {"Claude": (ask_claude, ["ANTHROPIC_API_KEY"]), "ChatGPT": (ask_openai, ["OPENAI_API_KEY", "OPENAI_MODEL"]),
          "Gemini": (ask_gemini, ["GEMINI_API_KEY", "GEMINI_MODEL"])}
clip = lambda x: round(max(-1.0, min(1.0, float(x))), 2)

def run(name, fn, p):
    try:
        d = json.loads(re.search(r"\{.*\}", fn(p), re.S).group(0))
        return name, {"next_day": clip(d["next_day"]), "next_week": clip(d["next_week"]), "confidence": clip(abs(float(d.get("confidence", 0)))),
                      "drivers": d.get("drivers", [])[:4], "summary": str(d.get("summary", "")),
                      "analysts": [a for a in d.get("analysts", []) if isinstance(a, dict)][:15]}
    except Exception as e:
        return name, {"error": f"{type(e).__name__}: {e}"}

def main():
    load_settings()
    hl = headlines()
    if len(hl) < 5: raise SystemExit("抓到的標題太少，請檢查網路")
    sec = lambda k: "\n".join(f'{h["id"]}|{h["tag"]}|{h["title"]}' for h in hl if h["kind"] == k)
    p = PROMPT + "【市場新聞】\n" + sec("market") + "\n【機構觀點】\n" + sec("analyst")
    todo = {n: f for n, (f, keys) in MODELS.items() if all(os.environ.get(k) for k in keys)}
    with ThreadPoolExecutor() as ex:  # 有金鑰的 AI 同時分析
        res = {"內建規則": lexicon(hl)} | dict(ex.map(lambda kv: run(kv[0], kv[1], p), todo.items()))
    ok = {n: r for n, r in res.items() if "error" not in r}
    agg = {}
    for k in ("next_day", "next_week"):
        v = [r[k] for r in ok.values()]
        agg[k] = {"mean": round(sum(v) / len(v), 2), "spread": round(max(v) - min(v), 2), "n": len(v)}
    rows = {}
    for n, r in ok.items():
        for a in r["analysts"]:
            if a.get("stance") in STANCE:
                row = rows.setdefault(str(a.get("name", "?")), {"name": str(a.get("name", "?")), "id": a.get("id"), "stances": {}})
                row["stances"][n] = a["stance"]
    vals = [sum(STANCE[s] for s in r["stances"].values()) / len(r["stances"]) for r in rows.values()]
    out = {"asof": dt.date.today().isoformat(), "headlines": hl, "prompt": p, "models": res, "agg": agg,
           "missing": [n for n in MODELS if n not in todo],
           "analyst": {"rows": list(rows.values()), "score": round(sum(vals) / len(vals), 2) if vals else None}}
    json.dump(out, open("news.json", "w", encoding="utf-8"), ensure_ascii=False)
    with open("news_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"asof": out["asof"], "agg": agg, "analyst": [{"name": r["name"], "stances": r["stances"]} for r in rows.values()],
                            "models": {n: {k: r[k] for k in ("next_day", "next_week")} for n, r in ok.items()}}, ensure_ascii=False) + "\n")
    for n, r in res.items(): print(n, r.get("error") or f"隔日 {r['next_day']:+.2f}  一週 {r['next_week']:+.2f}")

if __name__ == "__main__":
    main()
