"""Official Taiwan stock-market historical data adapter.

TWSE listed shares: monthly STOCK_DAY endpoint.
TPEx OTC shares: monthly afterTrading/tradingStock endpoint.
TAIEX benchmark: monthly MI_5MINS_HIST endpoint.

The provider returns end-of-day history, not an intraday real-time feed.
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


class MarketDataError(RuntimeError):
    pass


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TaiwanMarketResearch/1.0)",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.twse.com.tw/",
}
_CACHE_LOCK = threading.Lock()
_STOCK_CACHE = {}
_INDEX_CACHE = {}
_CACHE_TTL_SECONDS = 4 * 60 * 60


def _request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers=_HEADERS)
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=18) as response:
                payload = response.read()
            parsed = json.loads(payload.decode("utf-8-sig"))
            if not isinstance(parsed, dict):
                raise MarketDataError("交易所回傳格式不是 JSON 物件")
            return parsed
        except Exception as exc:
            last_error = exc
            code = getattr(exc, "code", None)
            if code in (401, 403):
                break
            if code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(float(2 ** attempt))
                continue
            break

    # Keep diagnostics useful without recording full URLs, query strings, or
    # request payloads. The adapter only calls fixed public exchange hosts.
    host = urllib.parse.urlparse(url).hostname or "unknown-host"
    reason = getattr(last_error, "reason", None)
    if code is not None:
        detail = f"{type(last_error).__name__}(http={code})"
        if code in (301, 302, 303, 307, 308):
            headers = getattr(last_error, "headers", None)
            location = headers.get("Location") if headers is not None else None
            if not location:
                getheader = getattr(last_error, "getheader", None)
                if callable(getheader):
                    location = getheader("Location")
            if not location:
                info = getattr(last_error, "info", None)
                response_headers = info() if callable(info) else None
                if response_headers is not None:
                    location = response_headers.get("Location")
            if location:
                target = urllib.parse.urlparse(urllib.parse.urljoin(url, location))
                target_host = target.hostname or "unknown-host"
                target_path = target.path or "/"
                detail += f"(redirect={target_host}{target_path})"
    elif reason is not None:
        reason_type = type(reason).__name__
        reason_errno = getattr(reason, "errno", None)
        detail = f"{type(last_error).__name__}/{reason_type}"
        if reason_errno is not None:
            detail += f"(errno={reason_errno})"
    else:
        detail = type(last_error).__name__
    raise MarketDataError(f"官方行情來源連線失敗：{host} {detail}")


def _month_starts(years: int = 10):
    today = dt.date.today()
    start = dt.date(today.year - years, today.month, 1)
    current = dt.date(start.year, start.month, 1)
    end = dt.date(today.year, today.month, 1)
    values = []
    while current <= end:
        values.append(current)
        year, month = current.year, current.month + 1
        current = dt.date(year + (month == 13), 1 if month == 13 else month, 1)
    return values


def _roc_date(value: str) -> dt.date:
    raw = str(value or "").strip().replace("*", "").replace("＊", "")
    raw = raw.replace("年", "/").replace("月", "/").replace("日", "")
    parts = raw.split("/")
    if len(parts) != 3:
        raise ValueError("日期格式錯誤")
    year, month, day = (int(part) for part in parts)
    if year < 1911:
        year += 1911
    return dt.date(year, month, day)


def _number(value):
    raw = str(value if value is not None else "").strip()
    raw = raw.replace(",", "").replace("＋", "+").replace("－", "-")
    if raw in ("", "--", "—", "-", "X", "x"):
        return None
    # X-prefixed exchange values (for example X0.00) indicate a non-comparable
    # reference-price day; they are not a numeric zero change.
    if raw.startswith(("X", "x")):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _twse_month(code: str, month: dt.date):
    params = urllib.parse.urlencode({
        "date": month.strftime("%Y%m01"),
        "stockNo": code,
        "response": "json",
    })
    url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?" + params
    payload = _request_json(url)
    status = payload.get("stat")
    if status == "很抱歉，沒有符合條件的資料!":
        return []
    if status != "OK":
        raise MarketDataError(f"TWSE STOCK_DAY 回傳非成功狀態：{status or 'missing stat'}")
    rows = []
    for item in payload.get("data") or []:
        if len(item) < 8:
            continue
        try:
            day = _roc_date(item[0])
            volume = _number(item[1])
            open_price, high, low, close = (_number(x) for x in item[3:7])
            change = _number(item[7])
            if close is None or volume is None:
                continue
            rows.append({
                "date": day, "open": open_price, "high": high, "low": low,
                "raw_close": close, "change": change, "volume": volume,
            })
        except (ValueError, TypeError):
            continue
    return rows


def _tpex_month(code: str, month: dt.date):
    params = urllib.parse.urlencode({
        "code": code,
        "date": month.strftime("%Y/%m/01"),
        "id": "",
        "response": "json",
    })
    url = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock?" + params
    payload = _request_json(url)
    tables = payload.get("tables") or []
    if not tables:
        return []
    rows = []
    for item in tables[0].get("data") or []:
        if len(item) < 8:
            continue
        try:
            day = _roc_date(item[0])
            lots = _number(item[1])
            open_price, high, low, close = (_number(x) for x in item[3:7])
            change = _number(item[7])
            if close is None or lots is None:
                continue
            rows.append({
                "date": day, "open": open_price, "high": high, "low": low,
                "raw_close": close, "change": change, "volume": lots * 1000.0,
            })
        except (ValueError, TypeError):
            continue
    return rows


def _twse_exright_month(month: dt.date):
    params = urllib.parse.urlencode({
        "startDate": month.strftime("%Y%m01"),
        "endDate": (dt.date(month.year, month.month, calendar.monthrange(month.year, month.month)[1])).strftime("%Y%m%d"),
        "response": "json",
    })
    url = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U?" + params
    payload = _request_json(url)
    fields = payload.get("fields") or []
    rows = payload.get("data") or []
    if not fields:
        return {}
    positions = {name: index for index, name in enumerate(fields)}
    required = ("資料日期", "股票代號", "除權息參考價", "權/息")
    if any(name not in positions for name in required):
        raise MarketDataError("TWSE 除權息資料欄位格式不符")
    result = {}
    for item in rows:
        try:
            day_text = str(item[positions["資料日期"]]).replace("年", "/").replace("月", "/").replace("日", "")
            day = _roc_date(day_text)
            symbol = str(item[positions["股票代號"]]).strip()
            kind = str(item[positions["權/息"]]).strip()
            reference = _number(item[positions["除權息參考價"]])
            # Only cash-dividend-only rows are used as a fallback. Rights,
            # stock dividends and capital-increase cases require their own
            # verified share/price adjustment factors.
            if symbol and kind in ("息", "除息") and reference and reference > 0:
                result[(symbol, day)] = reference
        except (ValueError, TypeError, IndexError):
            continue
    return result


def _tpex_exright_month(month: dt.date):
    last_day = calendar.monthrange(month.year, month.month)[1]
    params = urllib.parse.urlencode({
        "startDate": month.strftime("%Y/%m/01"),
        "endDate": month.strftime(f"%Y/%m/{last_day:02d}"),
        "response": "json",
    })
    url = "https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ?" + params
    payload = _request_json(url)
    result = {}
    for table in payload.get("tables") or []:
        fields = table.get("fields") or []
        rows = table.get("data") or []
        positions = {name: index for index, name in enumerate(fields)}
        required = ("除權息日期", "代號", "除權息參考價", "權/息")
        if any(name not in positions for name in required):
            continue
        for item in rows:
            try:
                day = _roc_date(item[positions["除權息日期"]])
                symbol = str(item[positions["代號"]]).strip()
                kind = str(item[positions["權/息"]]).strip()
                reference = _number(item[positions["除權息參考價"]])
                if symbol and kind in ("息", "除息") and reference and reference > 0:
                    result[(symbol, day)] = reference
            except (ValueError, TypeError, IndexError):
                continue
    return result


def _official_cash_dividend_references(code: str, source: str, rows):
    missing = sorted({
        row["date"].replace(day=1)
        for row in rows
        if row.get("change") is None
    })
    if not missing:
        return {}
    fetcher = _twse_exright_month if source == "TWSE" else _tpex_exright_month
    references = {}
    for month in missing:
        references.update(fetcher(month))
    return {
        day: price
        for (symbol, day), price in references.items()
        if symbol == str(code).strip().upper()
    }


def _fetch_months(fetcher, code: str, months, max_workers: int = 2):
    collected = []
    errors = []
    # Two concurrent requests balance throughput with public exchange rate limits.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetcher, code, month): month for month in months}
        for future in as_completed(futures):
            try:
                collected.extend(future.result())
            except Exception as exc:
                errors.append(str(exc))
    if errors:
        raise MarketDataError(
            "官方歷史行情有月份下載失敗，為避免不完整資料進入模型，已停止分析：" + errors[0]
        )
    collected.sort(key=lambda row: row["date"])
    deduped = {}
    for row in collected:
        deduped[row["date"]] = row
    return [deduped[key] for key in sorted(deduped)]


def _identify_exchange(code: str, hint: str | None, months):
    candidates = [("TWSE", _twse_month), ("TPEX", _tpex_month)]
    if hint == "TPEX":
        candidates.reverse()
    probe_months = months[-3:]
    for name, fetcher in candidates:
        for month in reversed(probe_months):
            try:
                if fetcher(code, month):
                    return name, fetcher
            except Exception:
                continue
    raise MarketDataError(f"官方交易所查無 {code} 的近期歷史行情")


def _to_adjusted_frame(rows, source: str, code: str | None = None, corporate_action_references=None):
    if not rows:
        raise MarketDataError("官方交易所沒有回傳可用的日線資料")
    rows = sorted(rows, key=lambda row: row["date"])
    adjusted = []
    previous_adjusted = None
    corporate_action_references = corporate_action_references or {}
    for row in rows:
        raw_close = float(row["raw_close"])
        if raw_close <= 0:
            raise MarketDataError(f"{row['date']} 的原始收盤價不合理")
        change = row["change"]
        if previous_adjusted is None:
            adjusted_close = raw_close
        else:
            if change is None:
                reference_close = corporate_action_references.get(row["date"])
                if reference_close is None:
                    raise MarketDataError(
                        f"{row['date']} 的官方漲跌價差無法解析，且沒有可核對的現金除息參考價，停止使用這批資料"
                    )
            else:
                reference_close = raw_close - float(change)
            if reference_close <= 0:
                raise MarketDataError(f"{row['date']} 的調整參考價不合理")
            adjusted_close = previous_adjusted * raw_close / reference_close
        # Keep OHLC on the same chained-adjusted price basis as Close.
        adjustment_factor = adjusted_close / raw_close
        adjusted.append({
            "Date": pd.Timestamp(row["date"]),
            "Close": adjusted_close,
            "RawClose": raw_close,
            "Volume": float(row["volume"]),
            "Open": float(row["open"]) * adjustment_factor if row["open"] is not None else None,
            "High": float(row["high"]) * adjustment_factor if row["high"] is not None else None,
            "Low": float(row["low"]) * adjustment_factor if row["low"] is not None else None,
        })
        previous_adjusted = adjusted_close
    frame = pd.DataFrame(adjusted).set_index("Date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.attrs["source"] = source
    frame.attrs["price_basis"] = "official_reference_price_chained_adjusted"
    return frame


def load_stock_history(code: str, exchange_hint: str | None = None, years: int = 10):
    code = str(code).strip().upper()
    cache_key = (code, exchange_hint, years)
    now = time.time()
    with _CACHE_LOCK:
        cached = _STOCK_CACHE.get(cache_key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return cached[1].copy()

    months = _month_starts(years)
    source, fetcher = _identify_exchange(code, exchange_hint, months)
    # TWSE applies stricter request throttling than TPEx; keep its monthly
    # history requests serial while retaining limited TPEx concurrency.
    rows = _fetch_months(fetcher, code, months, max_workers=1 if source == "TWSE" else 2)
    references = _official_cash_dividend_references(code, source, rows)
    frame = _to_adjusted_frame(rows, source, code, references)
    if len(frame) < 60:
        raise MarketDataError(f"{code} 的官方歷史資料不足 60 個交易日")
    with _CACHE_LOCK:
        _STOCK_CACHE[cache_key] = (now, frame.copy())
    return frame


def _taiex_month(month: dt.date):
    params = urllib.parse.urlencode({
        "date": month.strftime("%Y%m01"),
        "response": "json",
    })
    url = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?" + params
    payload = _request_json(url)
    rows = []
    for item in payload.get("data") or []:
        if len(item) < 5:
            continue
        try:
            day = _roc_date(item[0])
            close = _number(item[4])
            if close is not None:
                rows.append((day, close))
        except (ValueError, TypeError):
            continue
    return rows


def load_taiex_history(years: int = 10):
    global _INDEX_CACHE
    now = time.time()
    with _CACHE_LOCK:
        cached = _INDEX_CACHE.get(years)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return cached[1].copy()

    months = _month_starts(years)
    records = []
    # Keep TWSE benchmark requests serial to avoid exchange-side throttling
    # on long historical ranges.
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(_taiex_month, month) for month in months]
        errors = []
        for future in as_completed(futures):
            try:
                records.extend(future.result())
            except Exception as exc:
                errors.append(str(exc))
    if errors:
        raise MarketDataError(
            "官方加權指數歷史資料有月份下載失敗，為避免基準資料不完整，已停止分析：" + errors[0]
        )
    if not records:
        raise MarketDataError("無法取得官方加權指數歷史資料")
    by_date = {}
    for day, close in records:
        by_date[day] = close
    series = pd.Series(
        [by_date[key] for key in sorted(by_date)],
        index=pd.DatetimeIndex(sorted(by_date)),
        name="TAIEX",
        dtype=float,
    )
    if len(series) < 252:
        raise MarketDataError("官方加權指數歷史資料不足 252 個交易日")
    with _CACHE_LOCK:
        _INDEX_CACHE[years] = (now, series.copy())
    return series

