"""新聞與機構觀點分析 → news.json（僅供參考，未納入回測模型）。
兩種模式並存：
  1. 內建規則：不需要任何金鑰，永遠執行（只比對字詞，不理解語意，準確度有限）。
  2. AI 模式：在儀表板「設定」頁（或環境變數）填入金鑰後，Claude / ChatGPT / Gemini 會同時分析，結果並排顯示。
     ANTHROPIC_API_KEY（CLAUDE_MODEL 預設 claude-sonnet-5-5）；OPENAI_API_KEY + OPENAI_MODEL；GEMINI_API_KEY + GEMINI_MODEL
資料來源：國際市場新聞、中央社台灣新聞 RSS，以及台灣／美國政府政策與國際局勢的近 7 日公開資訊。
機構觀點只計入「標題明確寫出該機構對台股方向」者；每日結果累積到 news_history.jsonl，日後可算各機構命中率。
用法：python news.py   （時間同 pipeline.py：台股交易日 8:30 前後）
"""
import os, json, re, html, hashlib, csv, datetime as dt, urllib.request, urllib.parse
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

PROMPT = """你是台灣加權指數的新聞研究助理。只根據提供的標題與 RSS 摘要判斷，不得補入未提供的事實；資訊不足、方向混合或傳導不清楚時，分數應接近 0 並降低信心。
先判斷新聞是否與台灣加權指數有合理關聯。排除與台股無關的一般政治、地方活動、其他國家股市及單一公司新聞；若單一公司新聞涉及台灣大型權值股、半導體供應鏈或明確的台灣產業傳導，才納入並說明原因。不要把「提到台灣企業」直接等同「台股方向」。
政策新聞須區分已公告／已生效、提案／研議及媒體推測，並說明關稅、出口、半導體供應鏈、能源、匯率、利率或財政支出的具體傳導管道。關稅、制裁、戰爭、升降息等詞本身不自動代表利空或利多，須判斷實際影響與雙向風險。
機構觀點必須同時符合：標題直接點名該機構、明確提到台股／台灣股市／TAIEX，且明確表達該機構對台股方向的看法。泛指美股、中國股市、全球科技股或台灣企業投資的文章不得列為台股機構觀點。
只輸出 JSON：{"next_day": 數字, "next_week": 數字, "confidence": 0到1, "drivers": [{"id": 標題id, "impact": "正面"或"負面", "why": "20字內理由"}], "summary": "繁體中文，80字內", "analysts": [{"name": 機構名稱, "id": 標題id, "stance": "看多"或"看空"或"中性"或"無明確看法"}]}
分數代表對台灣加權指數的研究方向：-1 極偏負向、0 中性、1 極偏正向；不是未來上漲機率。next_day 指下一個交易日，next_week 指未來 5 個交易日。drivers 最多 4 則，須引用實際標題 id 並簡述傳導原因。
analysts 不得依機構搜尋分類或單純情緒字詞推測；只有標題明確表達該機構對台股大盤方向的看法才列出，否則回傳空陣列。"""

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

MARKET_RELEVANCE_TERMS = (
    "台股", "台灣股市", "台灣股票", "加權指數", "台積電", "半導體", "晶片", "晶圓",
    "出口", "關稅", "貿易", "央行", "利率", "通膨", "通胀", "匯率", "gdp",
    "經濟成長", "景氣", "供應鏈", "能源", "油價", "科技", "人工智慧", "投資",
    "預算", "補助", "金融", "semiconductor", "chip", "tsmc", "taiwan stocks", "taiwan shares", "taiwan weighted", "taiex", "export", "tariff",
    "trade", "central bank", "interest rate", "inflation", "currency", "supply chain",
    "energy", "oil price", "manufacturing", "fiscal", "budget", "subsidy",
)
POSITIVE_ZH = ("上漲", "上揚", "走高", "反彈", "回升", "成長", "增加", "改善", "利多",
               "創高", "突破", "復甦", "降息", "獲利", "擴張", "強勁", "優於預期", "買超")
NEGATIVE_ZH = ("下跌", "走低", "回落", "下滑", "衰退", "減少", "惡化", "利空", "暴跌",
               "制裁", "戰爭", "升息", "疲弱", "虧損", "萎縮", "低於預期", "賣超")

TAIWAN_MARKET_RE = re.compile(
    r"(?:台股|台灣股市|台灣股票|台灣加權指數|加權指數|TAIEX|"
    r"Taiwan(?:ese)?\s+(?:stocks?|shares?|equities)|Taiwan Weighted)",
    re.I,
)
ANALYST_ALIASES = {
    "高盛": ("goldman sachs", "goldman"),
    "摩根士丹利": ("morgan stanley",),
    "摩根大通": ("j.p. morgan", "jp morgan", "jpmorgan"),
    "花旗": ("citigroup", "citi"),
    "瑞銀": ("ubs",),
    "野村": ("nomura",),
    "麥格理": ("macquarie",),
}
ANALYST_POSITIVE = ("bullish", "overweight", "upgrade", "outperform", "positive on",
                    "optimistic", "top pick", "看多", "看好", "增持", "加碼", "偏多")
ANALYST_NEGATIVE = ("bearish", "underweight", "downgrade", "underperform", "negative on",
                    "pessimistic", "看空", "看淡", "減持", "降評", "偏空")


def market_relevant_headline(item):
    kind = str(item.get("kind") or "")
    if kind == "market":
        return True
    if kind not in ("taiwan", "policy"):
        return False
    text = (str(item.get("title") or "") + " " + str(item.get("description") or "")).lower()
    return any(term in text for term in MARKET_RELEVANCE_TERMS)


def headline_sentiment(title):
    title = str(title or "")
    en_positive, en_negative = tally(title)
    positive = en_positive + sum(term in title for term in POSITIVE_ZH)
    negative = en_negative + sum(term in title for term in NEGATIVE_ZH)
    return int(positive), int(negative)


def explicit_analyst_view(item):
    if str(item.get("kind") or "") != "analyst":
        return None
    title = str(item.get("title") or "")
    lower = title.lower()
    tag = str(item.get("tag") or "")
    aliases = ANALYST_ALIASES.get(tag, ())
    if not aliases or not any(alias in lower for alias in aliases):
        return None
    if not TAIWAN_MARKET_RE.search(title):
        return None
    if re.search(r"(?:not|no longer|isn't|is not)\s+(?:bullish|bearish|overweight|underweight)|(?:不看多|不看空|並非看多|並非看空|不代表看多|不代表看空)", lower):
        return None
    positive = any(term in lower for term in ANALYST_POSITIVE)
    negative = any(term in lower for term in ANALYST_NEGATIVE)
    if positive == negative:
        return None
    attribution = re.search(
        r"\b(?:says?|said|sees?|expects?|forecast(?:s|ed)?|recommends?|rates?|"
        r"bullish|bearish|overweight|underweight|upgrade[sd]?|downgrade[sd]?)\b|"
        r"(?:表示|認為|看好|看淡|看多|看空|建議|調升|調降)",
        lower,
    )
    if not attribution:
        return None
    return "看多" if positive else "看空"


def validate_analyst_calls(calls, analyst_headlines):
    allowed = {
        str(item.get("id")): (item, explicit_analyst_view(item))
        for item in analyst_headlines
    }
    result, seen = [], set()
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, dict):
            continue
        item, expected = allowed.get(str(call.get("id")), (None, None))
        if not item or not expected or call.get("name") != item.get("tag"):
            continue
        if call.get("stance") != expected or str(item.get("id")) in seen:
            continue
        seen.add(str(item.get("id")))
        result.append({"name": item.get("tag"), "id": item.get("id"), "stance": expected})
    return result


def lexicon(hl):
    weighted = []
    positive_count = negative_count = 0
    weights = {"market": 1.0, "taiwan": 0.8, "policy": 0.7}
    for item in hl:
        if not market_relevant_headline(item):
            continue
        positive, negative = headline_sentiment(item.get("title"))
        if positive == negative:
            continue
        direction = 1 if positive > negative else -1
        positive_count += int(direction > 0)
        negative_count += int(direction < 0)
        weighted.append((weights.get(item.get("kind"), 0.0), direction, item))
    total_weight = sum(weight for weight, _, _ in weighted)
    raw_score = (
        sum(weight * direction for weight, direction, _ in weighted) / total_weight
        if total_weight else 0.0
    )
    score = round(raw_score * len(weighted) / (len(weighted) + 5), 2) if weighted else 0.0
    drivers = [{
        "id": item.get("id"),
        "impact": "正面" if direction > 0 else "負面",
        "why": "標題字詞規則命中；非語意判讀",
    } for _, direction, item in weighted[:4]]
    calls = []
    for item in hl:
        stance = explicit_analyst_view(item)
        if stance:
            calls.append({"name": item.get("tag"), "id": item.get("id"), "stance": stance})
    summary = (
        f"規則基準：納入 {len(weighted)} 則具方向字詞的市場相關標題，"
        f"正向 {positive_count}、負向 {negative_count}；僅依標題字詞，非語意模型，"
        "隔日與一週共用同一分數。"
    )
    return {"next_day": score, "next_week": score, "confidence": 0.3,
            "drivers": drivers, "summary": summary, "analysts": calls}

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

def run(name, fn, p, analyst_headlines=None):
    try:
        d = json.loads(re.search(r"\{.*\}", fn(p), re.S).group(0))
        return name, {"next_day": clip(d["next_day"]), "next_week": clip(d["next_week"]), "confidence": clip(abs(float(d.get("confidence", 0)))),
                      "drivers": d.get("drivers", [])[:4], "summary": str(d.get("summary", "")),
                      "analysts": validate_analyst_calls(d.get("analysts", []), analyst_headlines or [])}
    except Exception as e:
        return name, {"error": f"{type(e).__name__}: {e}"}

def research_features(hl):
    """Versioned, Taiwan-market-relevant news features; not a trading signal."""
    kinds = ("market", "taiwan", "policy", "analyst")
    counts = {kind: sum(1 for h in hl if h.get("kind") == kind) for kind in kinds}
    relevant = [item for item in hl if market_relevant_headline(item)]
    relevant_counts = {
        kind: sum(1 for item in relevant if item.get("kind") == kind)
        for kind in ("market", "taiwan", "policy")
    }
    sources = {str(h.get("source") or h.get("tag") or "").strip()
               for h in hl if str(h.get("source") or h.get("tag") or "").strip()}
    dated = sum(bool(str(h.get("published") or "").strip()) for h in hl)
    linked = sum(bool(str(h.get("link") or "").strip()) for h in hl)
    pos = neg = 0
    for item in relevant:
        positive, negative = headline_sentiment(item.get("title"))
        pos += int(positive > negative)
        neg += int(negative > positive)
    score = round((pos - neg) / max(1, len(relevant)), 4)
    return {
        "feature_version": 2,
        "headline_count": len(hl),
        "relevant_headline_count": len(relevant),
        "category_counts": counts,
        "relevant_category_counts": relevant_counts,
        "unique_source_count": len(sources),
        "published_time_coverage_pct": round(100 * dated / len(hl), 1) if hl else 0,
        "original_link_coverage_pct": round(100 * linked / len(hl), 1) if hl else 0,
        "positive_headline_count": pos,
        "negative_headline_count": neg,
        "news_environment_score": score,
    }

def latest_market_asof():
    """Use the latest completed TAIEX date, even when the full pipeline has not refreshed signals.json."""
    try:
        with open("market_history.csv", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            date_key = (reader.fieldnames or [None])[0]
            if not date_key or "TAIEX" not in (reader.fieldnames or []):
                return None
            latest = None
            for row in reader:
                date_value = str(row.get(date_key) or "").strip()[:10]
                close = str(row.get("TAIEX") or "").replace(",", "").strip()
                try:
                    if date_value and close and close.lower() not in ("nan", "none", "null"):
                        if float(close) > 0:
                            latest = date_value
                except (TypeError, ValueError):
                    continue
            return latest
    except (OSError, csv.Error):
        return None


def save_research_snapshot(hl):
    """保存每個交易日的原始快照；舊版快照只重算衍生特徵，不改動新聞原文。"""
    market_asof = latest_market_asof()
    if not market_asof:
        try:
            market_asof = json.load(open("signals.json", encoding="utf-8")).get("asof")
        except Exception:
            market_asof = None
    try:
        taipei_today = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    except Exception:
        taipei_today = dt.date.today().isoformat()
    asof = market_asof or taipei_today
    path = "news_research_history.jsonl"
    records = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        records.append(row)
                except Exception:
                    continue

    existing = next((row for row in records if str(row.get("asof")) == str(asof)), None)
    if existing:
        old_features = existing.get("features") or {}
        if int(old_features.get("feature_version") or 0) == 2:
            print(f"新聞研究快照 {asof} 已存在，保留原始紀錄。", flush=True)
            return
        # Rebuild only derived features from the immutable stored article list.
        stored_articles = existing.get("articles") or []
        existing["features"] = research_features(stored_articles)
        snapshot = existing
        print(f"新聞研究快照 {asof} 已由舊版衍生特徵升級至 v2；原始標題與時間不變。", flush=True)
    else:
        market_context = {}
        try:
            signal_data = json.load(open("signals.json", encoding="utf-8"))
            probability = float(signal_data.get("prob_up"))
            if signal_data.get("asof") == asof and 0 <= probability <= 1:
                market_context = {
                    "prob_up": probability,
                    "missing_layer_count": len(signal_data.get("missing_layers") or []),
                }
        except (OSError, ValueError, TypeError):
            pass
        snapshot = {
            "asof": asof, "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "market_context": market_context,
            "features": research_features(hl),
            "articles": [{
                "kind": str(h.get("kind", "")), "category": str(h.get("tag", "")),
                "title": str(h.get("title", ""))[:500],
                "source": str(h.get("source", ""))[:160],
                "published": str(h.get("published", ""))[:100],
                "link": str(h.get("link", ""))[:1000],
            } for h in hl],
            "outcome": None,
        }
        records.append(snapshot)

    records.sort(key=lambda row: str(row.get("asof", "")))
    with open(path + ".tmp", "w", encoding="utf-8") as stream:
        for row in records[-1500:]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(path + ".tmp", path)

    labeled = sum(isinstance(row.get("outcome"), dict) for row in records)
    context = {"status": "insufficient", "sample_count": 0,
               "note": "相似新聞環境的已標記樣本不足 20 筆，暫不顯示統計比例。"}
    current_score = (records[-1].get("features") or {}).get("news_environment_score") if records else None
    peers = []
    if current_score is not None:
        for row in records[:-1]:
            features, outcome = row.get("features") or {}, row.get("outcome") or {}
            score, value = features.get("news_environment_score"), outcome.get("next_session_return_pct")
            if score is None or value is None:
                continue
            try:
                if abs(float(score) - float(current_score)) <= 0.20:
                    peers.append(float(value))
            except (TypeError, ValueError):
                continue
    context["sample_count"] = len(peers)
    if len(peers) >= 20:
        ordered = sorted(peers)
        quantile = lambda q: round(ordered[min(len(ordered)-1, int((len(ordered)-1)*q))], 3)
        context = {
            "status": "available", "sample_count": len(ordered),
            "positive_return_pct": round(100 * sum(x > 0 for x in ordered) / len(ordered), 1),
            "negative_return_pct": round(100 * sum(x < 0 for x in ordered) / len(ordered), 1),
            "average_return_pct": round(sum(ordered) / len(ordered), 3),
            "median_return_pct": round((ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2]) / 2, 3),
            "p10_return_pct": quantile(0.10), "p90_return_pct": quantile(0.90),
            "score_tolerance": 0.20,
            "note": "僅為相似新聞環境下的歷史描述統計，不是未來機率或買賣訊號。",
        }
    summary = {
        "updated_at": records[-1].get("captured_at") if records else dt.datetime.now(dt.timezone.utc).isoformat(),
        "latest": records[-1] if records else None,
        "history_count": len(records), "labeled_count": labeled,
        "historical_context": context,
        "note": "新聞特徵僅作資料研究；歷史結果只在後續交易日收盤資料到齊後標記，不回填未來資訊。",
    }
    with open("news_research.json.tmp", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace("news_research.json.tmp", "news_research.json")

def main():
    load_settings()
    hl = headlines()
    if len(hl) < 3: raise SystemExit(f"抓到的新聞標題只有 {len(hl)} 則；Google News 與 Bing News RSS 均不足，保留上一版資料。")
    sec = lambda k: "\\n".join(
        f'{h["id"]}|{h["tag"]}|來源：{h.get("source", "")}|發布：{h.get("published", "")}|'
        f'{h["title"]}|摘要：{h.get("description", "")}'
        for h in hl if h["kind"] == k
    )
    p = (PROMPT + "【國際市場新聞】\\n" + sec("market") +
         "\\n【台灣新聞】\\n" + sec("taiwan") +
         "\\n【台灣／美國政府政策與國際局勢】\\n" + sec("policy") +
         "\\n【機構觀點】\\n" + sec("analyst"))
    todo = {n: f for n, (f, keys) in MODELS.items() if all(os.environ.get(k) for k in keys)}
    analyst_headlines = [h for h in hl if h.get("kind") == "analyst"]
    with ThreadPoolExecutor() as ex:  # 有金鑰的 AI 同時分析
        res = {"內建規則": lexicon(hl)} | dict(
            ex.map(lambda kv: run(kv[0], kv[1], p, analyst_headlines), todo.items())
        )
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
    save_research_snapshot(hl)
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
