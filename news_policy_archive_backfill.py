"""Build a small, clearly sampled index of historical Executive Yuan news.

The legacy OpenAPI endpoint returns HTTP 410. This importer therefore reads
only the public listing pages and stores short headlines, dates and source
links. It never downloads article bodies or summaries.
"""
import datetime as dt
from html.parser import HTMLParser
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

LIST_URL = "https://www.ey.gov.tw/Page/A2EC1FEF9BC39AD0"
ARTICLE_PREFIX = "/Page/9277F759E41CCD91/"
PAGE_SIZE = 15
SAMPLE_PAGES = [
    1, 36, 72, 108, 144, 180, 216, 252, 288, 324, 360, 396,
    432, 468, 504, 540, 576, 612, 648, 684, 720, 756, 792, 830,
]
DATE_RE = re.compile(r"(?<!\d)(\d{2,3})-(\d{1,2})-(\d{1,2})")


class NewsListParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.active = False
        self.href = ""
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag.lower() == "a":
            href = attrs.get("href", "")
            if ARTICLE_PREFIX in href:
                self.active = True
                self.href = href
                self.parts = []

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.active:
            text = re.sub(r"\s+", " ", " ".join(self.parts)).strip()
            self.links.append((self.href, text))
            self.active = False
            self.href = ""
            self.parts = []


def fetch_page(page):
    query = urllib.parse.urlencode({"PS": PAGE_SIZE, "page": page})
    request = urllib.request.Request(
        LIST_URL + "?" + query,
        headers={
            "User-Agent": "TAIEXSignal/1.0 (personal historical news index)",
            "Accept": "text/html",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                raw = response.read()
            return raw.decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def parse_item(href, text, page):
    match = DATE_RE.search(text)
    if not match:
        return None
    year, month, day = (int(x) for x in match.groups())
    if year < 1911:
        year += 1911
    try:
        published = dt.date(year, month, day).isoformat()
    except ValueError:
        return None
    title = html.unescape(text[:match.start()]).strip(" -　\t\r\n")
    title = re.sub(r"\s+", " ", title)
    if not title:
        return None
    url = urllib.parse.urljoin("https://www.ey.gov.tw", href)
    return {
        "date": published,
        "title": title[:180],
        "source": "行政院",
        "url": url,
        "index_page": page,
    }


def main():
    items, seen = [], set()
    successful_pages = []
    for page in SAMPLE_PAGES:
        raw = fetch_page(page)
        parser = NewsListParser()
        parser.feed(raw)
        page_count = 0
        for href, text in parser.links:
            item = parse_item(href, text, page)
            if not item or item["url"] in seen:
                continue
            seen.add(item["url"])
            items.append(item)
            page_count += 1
        if page_count:
            successful_pages.append(page)
        time.sleep(0.2)

    items.sort(key=lambda x: (x["date"], x["title"]))
    if len(items) < 10:
        raise SystemExit(
            f"Only {len(items)} dated index records parsed; existing archive preserved."
        )

    daily = {}
    for item in items:
        daily[item["date"]] = daily.get(item["date"], 0) + 1
    archive = {
        "dataset": "行政院一般新聞歷史索引（抽樣）",
        "source": "行政院全球資訊網",
        "source_url": LIST_URL,
        "rights_note": "僅保存短標題、日期及原文連結；原文及其他內容權利依行政院與原著作權人聲明辦理。",
        "scope": "跨年度分頁抽樣索引，非完整新聞庫；不含新聞全文或摘要。",
        "sampling": f"等距抽取 {len(successful_pages)} 個歷史分頁，每頁最多 {PAGE_SIZE} 筆。",
        "sampled_pages": successful_pages,
        "date_start": items[0]["date"],
        "date_end": items[-1]["date"],
        "record_count": len(items),
        "date_count": len(daily),
        "daily_counts": daily,
        "imported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "items": items,
    }
    tmp = "news_policy_archive.json.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(archive, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, "news_policy_archive.json")
    print(
        "Official historical news index:",
        len(items), "titles across", len(daily), "dates;",
        items[0]["date"], "to", items[-1]["date"],
        "| sampled pages", len(successful_pages),
        flush=True,
    )


if __name__ == "__main__":
    main()
