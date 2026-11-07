"""新聞與機構觀點分析 → news.json（僅供參考，未納入回測模型）。
兩種模式並存：
  1. 內建規則：不需要任何金鑰，永遠執行（只比對字詞，不理解語意，準確度有限）。
  2. AI 模式：在儀表板「設定」頁（或環境變數）填入金鑰後，Claude / ChatGPT / Gemini 會同時分析，結果並排顯示。
     ANTHROPIC_API_KEY（CLAUDE_MODEL 預設 claude-sonnet-5-5）；OPENAI_API_KEY + OPENAI_MODEL；GEMINI_API_KEY + GEMINI_MODEL
資料來源：國際市場新聞、中央社台灣新聞 RSS，以及台灣／美國政府政策與國際局勢的近 7 日公開資訊。
機構觀點只計入「標題明確寫出該機構對台股方向」者；每日結果累積到 news_history.jsonl，日後可算各機構命中率。
用法：python news.py   （時間同 pipeline.py：台股交易日 8:30 前後）
"""
import os, json, re, html, hashlib, datetime as dt, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

MARKET = [("美股", "Wall Street stocks Nasdaq close when:1d"), ("半導體", "TSMC semiconductor chips export when:1d"),
          ("總經", "Fed rates inflation Treasury yields when:1d"), ("台灣總經", "Taiwan central bank OR GDP forecast when:3d"),
          ("地緣", "Taiwan Strait China tensions when:1d")]
# 中央社 RSS 僅呈現標題、來源、時間與原文連結；遵守其個人非商業使用規範。
TAIWAN_FEEDS = [
    ("台灣政治", "https://feeds.feedburner.com/rsscna/politics"),
    ("台灣產經", "https://feeds.feedburner.com/rsscna/finance"),
    ("台灣科技", "https://feeds.feedburner.com/rsscna/technology"),
    ("台灣國際", "https://feeds.feedburner.com/rsscna/intworld"),
    ("台灣地方", "https://feeds.feedburner.com/rsscna/local"),
]
POLICY_FEEDS = [
    ("行政院新聞", "https://www.ey.gov.tw/RSS_Content.aspx?ModuleType=3"),
    ("經濟部新聞", "https://www.moea.gov.tw/Mns/Populace/news/NewsRSSdetail.aspx?Kind=1"),
]
POLICY = [
    ("台灣政府政策", "site:ey.gov.tw OR site:moea.gov.tw OR site:cbc.gov.tw OR site:dgbas.gov.tw OR site:fsc.gov.tw 台灣經濟產業政策 when:7d"),
    ("美國政府政策", "site:whitehouse.gov/news OR site:ustr.gov/about/policy-offices/press-office OR site:federalreserve.gov/newsevents 美國關稅貿易科技貨幣政策 when:7d"),
    ("國際局勢", "Taiwan Strait OR export controls OR sanctions OR Middle East energy supply chain trade when:7d"),
]
# 機構清單可自行增刪；也可加入你信任的個人，例如 ("姓名", "姓名 台股 when:7d")
ANALYSTS = [(n, f"{q} Taiwan stocks when:7d") for n, q in [("高盛", "Goldman Sachs"), ("摩根士丹利", "Morgan Stanley"),
            ("摩根大通", "JPMorgan"), ("花旗", "Citi"), ("瑞銀", "UBS"), ("野村", "Nomura"), ("麥格理", "Macquarie")]]
STANCE = {"看多": 1, "看空": -1, "中性": 0}

POS = {"surge", "surges", "rally", "rallies", "gain", "gains", "rise", "rises", "rose", "beat", "beats", "record", "boost", "soar",
       "jump", "recover", "ceasefire", "easing", "bullish", "overweight", "upgrad", "raises", "optimis"}
NEG = {"fall", "falls", "fell", "drop", "drops", "slump", "plunge", "loss", "losses", "miss", "downgrad", "weak", "fear", "fears",
       "tension", "war", "wars", "tariff", "sanction", "curb", "ban", "bans", "recession", "sell-off", "selloff", "slide", "crash",
       "probe", "bearish", "underweight", "cuts"}

PROMPT = """你是台灣市場新聞研究助理。以下資料包含國際市場新聞、台灣本地新聞、政府政策與國際局勢，以及機構觀點。只根據提供的標題與 RSS 摘要判斷，不要引用未提供的資訊；資訊不足時分數接近 0、信心調低。
分析政策時，請區分「已公告／已生效」與「提案／討論／媒體推測」，說明可能的傳導管道（例如關稅、出口、半導體供應鏈、能源、匯率、利率、財政支出），不要把政策立場直接等同市場結果。
只輸出 JSON：{"next_day": 數字, "next_week": 數字, "confidence": 0到1, "drivers": [{"id": 標題id, "impact": "正面"或"負面", "why": "20字內理由"}], "summary": "繁體中文，80字內", "analysts": [{"name": 機構名稱, "id": 標題id, "stance": "看多"或"看空"或"中性"或"無明確看法"}]}
分數代表對「台灣加權指數」的影響：-1 極偏空、0 中性、1 極偏多；next_day 指下一個交易日，next_week 指未來 5 個交易日。drivers 最多 4 則，可引用市場新聞、台灣新聞或政策新聞，並指出來源 id。
analysts 只列「標題明確寫出該機構對台股或台股大盤的方向性看法」的項目，其他不列，不要從標題推測。
"""

def load_settings():  # 儀表板「設定」頁存下的金鑰
    try: os.environ.update({k: v for k, v in json.load(open("settings.json", encoding="utf-8")).items() if v})
    except Exception: pass

def _fetch_feed(url, provider):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; TAIEXSignal/1.0)"})
    xml = urllib.request.urlopen(req, timeout=12).read()
    items = []
    for item in ET.fromstring(xml).iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        published = (item.findtext("pubDate") or item.findtext("date") or "").strip()
        source = (item.findtext("source") or provider).strip()
        description = item.findtext("description") or item.findtext("summary") or ""
        description = html.unescape(re.sub(r"<[^>]+>", " ", description))
        description = re.sub(r"\s+", " ", description).strip()[:360]
        if title:
            items.append({"title": title, "description": description, "link": link,
                          "published": published, "source": source})
    return items

def _collect_group(group, per):
    tag, q = group
    urls = [
        ("Google News", "https://news.google.com/rss/search?q=" + urllib.parse.quote(q) + "&hl=en-US&gl=US&ceid=US:en"),
        ("Bing News", "https://www.bing.com/news/search?q=" + urllib.parse.quote(q) + "&format=rss"),
    ]
    items = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(_fetch_feed, url, provider) for provider, url in urls]
        for fut in futures:
            try:
                items.extend(fut.result())
            except Exception as e:
                print("新聞來源失敗", tag, type(e).__name__, str(e)[:160])
    unique, seen_titles = [], set()
    for item in items:
        key = re.sub(r"\W+", "", item["title"].lower())
        if key and key not in seen_titles:
            seen_titles.add(key)
            unique.append(item)
        if len(unique) >= per:
            break
    return tag, unique

def collect(groups, per, kind, out, seen):
    # 不同新聞主題彼此獨立；有限併發可縮短等待，同時避免一次打爆 RSS 來源。
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda g: _collect_group(g, per), groups))
    for tag, items in results:
        n = 0
        for item in items:
            title = item["title"]
            key = re.sub(r"\W+", "", title.lower())
            if key and key not in seen and n < per:
                seen.add(key)
                n += 1
                out.append({"id": len(out), "kind": kind, "tag": tag, **item})

def collect_rss(feeds, kind, out, seen, per=8):
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(_fetch_feed, url, tag) for tag, url in feeds]
        for (tag, _), fut in zip(feeds, futures):
            try:
                items = fut.result()
            except Exception as e:
                print("RSS 新聞來源失敗", tag, type(e).__name__, str(e)[:160])
                continue
            count = 0
            for item in items:
                key = re.sub(r"\W+", "", item["title"].lower())
                if key and key not in seen and count < per:
                    seen.add(key)
                    out.append({"id": len(out), "kind": kind, "tag": tag, **item})
                    count += 1

def headlines():
    out, seen = [], set()
    collect(MARKET, 8, "market", out, seen)
    collect(POLICY, 6, "policy", out, seen)
    collect_rss(POLICY_FEEDS, "policy", out, seen, per=8)
    collect_rss(TAIWAN_FEEDS, "taiwan", out, seen, per=8)
    collect(ANALYSTS, 3, "analyst", out, seen)
    return out

def policy_events(hl):
    """將政策相關來源整理成可追溯事件清單；不推斷市場漲跌方向。"""
    channels = {
        "貿易／關稅": ("關稅", "貿易", "tariff", "trade", "ustr", "出口管制"),
        "半導體／科技": ("半導體", "晶片", "晶圓", "chip", "semiconductor", "科技", "ai", "人工智慧", "出口管制"),
        "利率／貨幣": ("利率", "央行", "聯準會", "fed", "貨幣", "匯率", "金融", "通膨", "通胀"),
        "財政／預算": ("預算", "財政", "補助", "稅", "支出", "建設", "投資抵減"),
        "能源／原物料": ("能源", "電價", "天然氣", "石油", "電力", "oil", "energy"),
        "外交／地緣": ("外交", "國安", "台海", "制裁", "sanction", "china", "taiwan", "國際"),
        "產業／供應鏈": ("產業", "供應鏈", "製造", "投資", "出口", "manufacturing", "supply chain"),
    }
    records, seen = [], set()
    for h in hl:
        if h.get("kind") != "policy":
            continue
        title = str(h.get("title", ""))
        desc = str(h.get("description", ""))
        blob = (title + " " + desc).lower()
        tag = str(h.get("tag", ""))
        link = str(h.get("link", ""))
        host = urllib.parse.urlparse(link).hostname or ""
        official = host.endswith(".gov.tw") or host.endswith(".gov") or host.endswith(".gov.uk")
        if any(x in blob for x in ("已生效", "正式實施", "開始實施", "生效日期")):
            status = "已生效／實施（依來源文字）"
        elif any(x in blob for x in ("草案", "提案", "研議", "徵求意見", "擬議", "proposal", "proposed")):
            status = "提案／研議（尚待確認）"
        elif any(x in blob for x in ("公告", "發布", "公布", "決議", "簽署", "通過", "press release", "announces")):
            status = "已發布／公告（不等於已生效）"
        else:
            status = "待確認政策階段"
        categories = [name for name, words in channels.items() if any(w in blob for w in words)]
        category = "、".join(categories[:3]) if categories else "一般政策"
        identity = link or (title + "|" + str(h.get("published", "")))
        event_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        if event_id in seen:
            continue
        seen.add(event_id)
        records.append({
            "event_id": event_id, "title": title, "description": desc,
            "category": category, "status": status,
            "source": str(h.get("source") or tag),
            "source_class": "政府官方網域" if official else "新聞／搜尋彙整，需交叉核對",
            "published": str(h.get("published", "")), "link": link,
            "tag": tag, "channels": categories[:3],
        })
    return records


def save_policy_event_history(events, path="policy_events.jsonl", limit=1200):
    """以事件網址雜湊去重，保留事件出現紀錄；不覆寫來源原始發布時間。"""
    old = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        old[row["event_id"]] = row
                    except Exception:
                        continue
        except OSError:
            pass
    today = dt.datetime.now(dt.timezone.utc).isoformat()
    for event in events:
        row = dict(event)
        row["first_seen"] = old.get(event["event_id"], {}).get("first_seen", today)
        row["last_seen"] = today
        old[event["event_id"]] = row
    rows = sorted(old.values(), key=lambda x: x.get("last_seen", ""), reverse=True)[:limit]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return rows


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
             {"model": os.environ.get("OPENAI_MODEL", "gpt-6-luna"), "messages": [{"role": "user", "content": p}], "response_format": {"type": "json_object"}})
    return r["choices"][0]["message"]["content"]

def ask_gemini(p):
    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    url = "https://generativelanguage.googleapis.com/v1beta/models/" + urllib.parse.quote(model, safe="") + ":generateContent"
    r = post(url, {"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
             {"contents": [{"parts": [{"text": p}]}], "generationConfig": {"responseMimeType": "application/json"}})
    return "".join(x.get("text", "") for x in r["candidates"][0]["content"]["parts"])

MODELS = {"Claude": (ask_claude, ["ANTHROPIC_API_KEY"]), "ChatGPT": (ask_openai, ["OPENAI_API_KEY"]),
          "Gemini": (ask_gemini, ["GEMINI_API_KEY"])}
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
    if len(hl) < 3: raise SystemExit(f"抓到的新聞標題只有 {len(hl)} 則；Google News 與 Bing News RSS 均不足，保留上一版資料。")
    sec = lambda k: "\n".join(
        f'{h["id"]}|{h["tag"]}|{h["title"]}|摘要：{h.get("description", "")}'
        for h in hl if h["kind"] == k
    )
    p = (PROMPT + "【國際市場新聞】\n" + sec("market") +
         "\n【台灣新聞】\n" + sec("taiwan") +
         "\n【台灣／美國政府政策與國際局勢】\n" + sec("policy") +
         "\n【機構觀點】\n" + sec("analyst"))
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
    events = policy_events(hl)
    event_history = save_policy_event_history(events)
    out = {"asof": dt.date.today().isoformat(), "demo": False, "headlines": hl, "policy_events": events,
           "policy_event_history_count": len(event_history), "prompt": p, "models": res, "agg": agg,
           "missing": [n for n in MODELS if n not in todo],
           "analyst": {"rows": list(rows.values()), "score": round(sum(vals) / len(vals), 2) if vals else None}}
    json.dump(out, open("news.json", "w", encoding="utf-8"), ensure_ascii=False)
    with open("news_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"asof": out["asof"], "agg": agg, "analyst": [{"name": r["name"], "stances": r["stances"]} for r in rows.values()],
                            "models": {n: {k: r[k] for k in ("next_day", "next_week")} for n, r in ok.items()}}, ensure_ascii=False) + "\n")
    for n, r in res.items(): print(n, r.get("error") or f"隔日 {r['next_day']:+.2f}  一週 {r['next_week']:+.2f}")

if __name__ == "__main__":
    main()
