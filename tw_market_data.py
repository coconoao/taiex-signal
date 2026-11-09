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
import math
import random
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from curl_cffi import requests as curl_requests


class MarketDataError(RuntimeError):
    """Market-data failure with a non-sensitive, machine-readable category."""

    def __init__(self, message, diagnostic_code=None):
        super().__init__(message)
        self.diagnostic_code = diagnostic_code


_HEADERS = {
    # Match a normal browser request to the TWSE historical-data page.
    # Some edge nodes treat generic client headers differently.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.twse.com.tw/zh/trading/historical/stock-day.html",
}
_CACHE_LOCK = threading.Lock()
_STOCK_CACHE = {}
_INDEX_CACHE = {}
_EXRIGHT_CACHE = {}
_EXRIGHT_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_TTL_SECONDS = 4 * 60 * 60


def _request_json(url: str) -> dict:
    original_host = urllib.parse.urlparse(url).hostname
    request_url = url
    last_error = None
    last_status = None
    redirect_count = 0

    for attempt in range(3):
        try:
            response = curl_requests.get(
                request_url,
                headers=_HEADERS,
                timeout=15,
                impersonate="chrome",
                allow_redirects=False,
            )
            status = response.status_code
            if status == 308:
                location = response.headers.get("Location")
                target = urllib.parse.urljoin(request_url, location or "")
                target_parts = urllib.parse.urlparse(target)
                if (
                    location
                    and redirect_count < 3
                    and target_parts.scheme == "https"
                    and (target_parts.hostname or "").lower() == (original_host or "").lower()
                ):
                    request_url = target
                    redirect_count += 1
                    continue
                last_status = status
                break

            if status in (401, 403):
                last_status = status
                break
            if status == 408 or status == 429 or status >= 500:
                last_status = status
                if attempt < 2:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = min(4.0, max(0.25, float(retry_after)))
                    except (TypeError, ValueError):
                        delay = 0.5 * (2 ** attempt) + random.uniform(0.0, 0.25)
                    time.sleep(delay)
                    continue
                break
            if status >= 400:
                last_status = status
                break

            try:
                parsed = json.loads(response.content.decode("utf-8-sig"))
            except json.JSONDecodeError as exc:
                last_error = exc
                last_status = None
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt) + random.uniform(0.0, 0.25))
                    continue
                break

            if not isinstance(parsed, dict):
                raise MarketDataError(
                    "交易所回傳格式不是 JSON 物件",
                    diagnostic_code="provider_invalid_json_shape",
                )
            return parsed
        except MarketDataError:
            raise
        except Exception as exc:
            last_error = exc
            last_status = None
            error_name = type(exc).__name__.lower()
            retryable = (
                isinstance(exc, (OSError, TimeoutError, ConnectionError))
                or "timeout" in error_name
                or "connection" in error_name
            )
            if retryable and attempt < 2:
                time.sleep(0.5 * (2 ** attempt) + random.uniform(0.0, 0.25))
                continue
            break

    host = urllib.parse.urlparse(url).hostname or "unknown-host"
    if host in ("www.twse.com.tw", "twse.com.tw"):
        source = "twse"
    elif host in ("www.tpex.org.tw", "tpex.org.tw"):
        source = "tpex"
    else:
        source = "unknown"

    if last_status is not None:
        diagnostic = f"{source}_http_{int(last_status)}"
        detail = f"HTTPError(http={int(last_status)})"
    elif isinstance(last_error, json.JSONDecodeError):
        diagnostic = f"{source}_invalid_json"
        detail = "JSONDecodeError"
    elif getattr(last_error, "errno", None) is not None:
        errno = int(last_error.errno)
        diagnostic = f"{source}_network_errno_{errno}"
        detail = f"{type(last_error).__name__}(errno={errno})"
    elif isinstance(last_error, TimeoutError) or "timeout" in type(last_error).__name__.lower():
        diagnostic = f"{source}_timeout"
        detail = f"{type(last_error).__name__}/timeout"
    else:
        diagnostic = f"{source}_request_error"
        detail = type(last_error).__name__ if last_error is not None else "HTTPError"

    raise MarketDataError(
        f"官方行情來源連線失敗：{host} {detail}",
        diagnostic_code=diagnostic,
    )


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
        number = float(raw)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
        raise MarketDataError(
            f"TWSE STOCK_DAY 回傳非成功狀態：{status or 'missing stat'}",
            diagnostic_code="twse_provider_status",
        )
    rows = []
    data = payload.get("data") or []
    if not isinstance(data, list):
        raise MarketDataError(
            "TWSE 歷史行情資料格式不符",
            diagnostic_code="provider_invalid_json_shape",
        )
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 8:
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
    if not isinstance(tables, list):
        raise MarketDataError(
            "TPEx 歷史行情回傳格式不符",
            diagnostic_code="provider_invalid_json_shape",
        )

    rows = []
    # The response may contain multiple tables; do not assume the first one
    # is always the trading-data table.
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_rows = table.get("data") or []
        if not isinstance(table_rows, list):
            continue
        for item in table_rows:
            if not isinstance(item, (list, tuple)) or len(item) < 8:
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
        raise MarketDataError("TWSE 除權息資料欄位格式不符", diagnostic_code="provider_invalid_json_shape")
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


def _cached_exright_month(source: str, month: dt.date):
    key = (source, month)
    now = time.time()
    with _CACHE_LOCK:
        cached = _EXRIGHT_CACHE.get(key)
        if cached and now - cached[0] < _EXRIGHT_CACHE_TTL_SECONDS:
            return dict(cached[1])

    fetcher = _twse_exright_month if source == "TWSE" else _tpex_exright_month
    references = fetcher(month)
    with _CACHE_LOCK:
        _EXRIGHT_CACHE[key] = (now, dict(references))
    return dict(references)


def _official_cash_dividend_references(code: str, source: str, rows):
    missing = sorted({
        row["date"].replace(day=1)
        for row in rows
        if row.get("change") is None
    })
    if not missing:
        return {}

    references = {}
    for month in missing:
        references.update(_cached_exright_month(source, month))
    return {
        day: price
        for (symbol, day), price in references.items()
        if symbol == str(code).strip().upper()
    }


def _fetch_months(fetcher, code: str, months, max_workers: int = 2):
    collected = []
    errors = []
    # Keep bounded concurrency; callers can lower it for stricter providers.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetcher, code, month): month for month in months}
        for future in as_completed(futures):
            try:
                collected.extend(future.result())
            except Exception as exc:
                errors.append(exc)
    if errors:
        diagnostic_codes = []
        for exc in errors:
            diagnostic_code = getattr(exc, "diagnostic_code", None)
            diagnostic_codes.append(
                diagnostic_code
                if isinstance(diagnostic_code, str)
                and diagnostic_code.replace("_", "").isalnum()
                else "provider_error"
            )
        safe_codes = ",".join(dict.fromkeys(diagnostic_codes))[:160]
        raise MarketDataError(
            "官方歷史行情有月份下載失敗，為避免不完整資料進入模型，已停止分析",
            diagnostic_code="history_fetch_failed:" + safe_codes,
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

    # Probe a small recent window so newly listed or temporarily inactive
    # securities are not rejected solely because the latest month is empty.
    probe_months = months[-6:]
    probe_failures = []
    for name, fetcher in candidates:
        for month in reversed(probe_months):
            try:
                if fetcher(code, month):
                    return name, fetcher
            except Exception as exc:
                category = getattr(exc, "diagnostic_code", None)
                if not isinstance(category, str) or not category.replace("_", "").isalnum():
                    category = "provider_error"
                probe_failures.append(
                    category if category.startswith(name.lower() + "_")
                    else f"{name.lower()}_{category}"
                )
                # A transport, HTTP, or invalid-JSON failure is provider-wide
                # for this attempt; try the other official exchange instead.
                if (
                    category.endswith("_timeout")
                    or "_network_errno_" in category
                    or "_http_" in category
                    or category.endswith("_invalid_json")
                    or category == "provider_invalid_json_shape"
                ):
                    break

    if probe_failures:
        diagnostic = "exchange_probe_failed:" + ",".join(dict.fromkeys(probe_failures))[:160]
    else:
        diagnostic = "no_recent_history"
    raise MarketDataError(
        f"官方交易所查無 {code} 的近期歷史行情",
        diagnostic_code=diagnostic,
    )


def _to_adjusted_frame(rows, source: str, code: str | None = None, corporate_action_references=None):
    if not rows:
        raise MarketDataError(
            "官方交易所沒有回傳可用的日線資料",
            diagnostic_code="no_recent_history",
        )
    rows = sorted(rows, key=lambda row: row["date"])
    adjusted = []
    previous_adjusted = None
    corporate_action_references = corporate_action_references or {}
    for row in rows:
        raw_close = float(row["raw_close"])
        if not math.isfinite(raw_close) or raw_close <= 0:
            raise MarketDataError(
                f"{row['date']} 的原始收盤價不合理",
                diagnostic_code="raw_close_invalid",
            )
        change = row["change"]
        if previous_adjusted is None:
            adjusted_close = raw_close
        else:
            if change is None:
                reference_close = corporate_action_references.get(row["date"])
                if reference_close is None:
                    raise MarketDataError(
                        f"{row['date']} 的官方漲跌價差無法解析，且沒有可核對的除權息參考價，停止使用這批資料",
                        diagnostic_code="adjustment_reference_missing",
                    )
            else:
                reference_close = raw_close - float(change)
            if not math.isfinite(reference_close) or reference_close <= 0:
                raise MarketDataError(
                    f"{row['date']} 的調整參考價不合理",
                    diagnostic_code="adjustment_reference_invalid",
                )
            adjusted_close = previous_adjusted * raw_close / reference_close
        if not math.isfinite(adjusted_close) or adjusted_close <= 0:
            raise MarketDataError(
                f"{row['date']} 的調整後收盤價不合理",
                diagnostic_code="adjustment_reference_invalid",
            )
        adjusted.append({
            "Date": pd.Timestamp(row["date"]),
            "Close": adjusted_close,
            "RawClose": raw_close,
            "Volume": float(row["volume"]),
            "Open": row["open"],
            "High": row["high"],
            "Low": row["low"],
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

    stage_started = time.perf_counter()
    try:
        source, fetcher = _identify_exchange(code, exchange_hint, months)
    except Exception as exc:
        print(
            "[stock-analysis] stage=exchange_probe status=failed "
            f"elapsed_ms={round((time.perf_counter() - stage_started) * 1000)} "
            f"error_class={type(exc).__name__}",
            flush=True,
        )
        raise
    print(
        "[stock-analysis] stage=exchange_probe status=ok "
        f"elapsed_ms={round((time.perf_counter() - stage_started) * 1000)}",
        flush=True,
    )

    stage_started = time.perf_counter()
    try:
        rows = _fetch_months(
            fetcher, code, months, max_workers=1 if source == "TWSE" else 2
        )
    except Exception as exc:
        print(
            "[stock-analysis] stage=stock_history_download status=failed "
            f"elapsed_ms={round((time.perf_counter() - stage_started) * 1000)} "
            f"error_class={type(exc).__name__}",
            flush=True,
        )
        raise
    print(
        "[stock-analysis] stage=stock_history_download status=ok "
        f"elapsed_ms={round((time.perf_counter() - stage_started) * 1000)}",
        flush=True,
    )

    stage_started = time.perf_counter()
    try:
        references = _official_cash_dividend_references(code, source, rows)
    except Exception as exc:
        diagnostic = getattr(exc, "diagnostic_code", None) or "provider_error"
        print(
            "[stock-analysis] stage=corporate_action_reference status=failed "
            f"elapsed_ms={round((time.perf_counter() - stage_started) * 1000)} "
            f"error_class={type(exc).__name__} diagnostic_code={diagnostic}",
            flush=True,
        )
        raise
    print(
        "[stock-analysis] stage=corporate_action_reference status=ok "
        f"elapsed_ms={round((time.perf_counter() - stage_started) * 1000)}",
        flush=True,
    )

    stage_started = time.perf_counter()
    try:
        frame = _to_adjusted_frame(rows, source, code, references)
    except Exception as exc:
        diagnostic = getattr(exc, "diagnostic_code", None) or "provider_or_data_validation_error"
        print(
            "[stock-analysis] stage=price_adjustment status=failed "
            f"elapsed_ms={round((time.perf_counter() - stage_started) * 1000)} "
            f"error_class={type(exc).__name__} diagnostic_code={diagnostic}",
            flush=True,
        )
        raise

    # Return shorter histories as data; downstream model horizons and Beta
    # are independently marked insufficient instead of rejecting the symbol.
    if frame.empty:
        raise MarketDataError(
            f"{code} 的官方歷史資料為空",
            diagnostic_code="no_recent_history",
        )
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
    # The benchmark endpoint is also TWSE; serialize requests to reduce
    # transient anti-bot and rate-limit responses during full backfills.
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(_taiex_month, month) for month in months]
        errors = []
        for future in as_completed(futures):
            try:
                records.extend(future.result())
            except Exception as exc:
                errors.append(exc)
    if errors:
        diagnostic_codes = []
        for exc in errors:
            diagnostic_code = getattr(exc, "diagnostic_code", None)
            diagnostic_codes.append(
                diagnostic_code
                if isinstance(diagnostic_code, str)
                and diagnostic_code.replace("_", "").isalnum()
                else "provider_error"
            )
        safe_codes = ",".join(dict.fromkeys(diagnostic_codes))[:160]
        raise MarketDataError(
            "官方加權指數歷史資料有月份下載失敗，為避免基準資料不完整，已停止分析：" + str(errors[0]),
            diagnostic_code="history_fetch_failed:" + safe_codes,
        )
    if not records:
        raise MarketDataError(
            "無法取得官方加權指數歷史資料",
            diagnostic_code="twse_provider_error",
        )
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
        raise MarketDataError(
            "官方加權指數歷史資料不足 252 個交易日",
            diagnostic_code="twse_provider_error",
        )
    with _CACHE_LOCK:
        _INDEX_CACHE[years] = (now, series.copy())
    return series

