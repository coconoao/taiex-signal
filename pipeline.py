"""台股加權指數訊號 MVP（國際 / 區域 / 本地 三層）
用法:  pip install pandas numpy scikit-learn yfinance lxml
       python pipeline.py --demo      # 合成資料，先看流程
       python pipeline.py             # 抓真實資料（需連網）；首次回補三大法人、融資各約 10 多分鐘，之後增量更新
       python pipeline.py --no-chips --no-margin --no-futures  # 略過籌碼／信用／期貨，只跑其餘層（旗標可單獨使用）
執行時間：台股交易日早上 8:30 前後（美股已收盤、台股未開盤、前一日籌碼已公布）。
輸出 signals.json，供 dashboard.html 讀取。僅供研究，非投資建議。
"""
import os, sys, json, time, io, datetime as dt, urllib.request, urllib.parse, urllib.error
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TICKERS = {"TAIEX": "^TWII", "SOX": "^SOX", "TSM": "TSM", "VIX": "^VIX", "US10Y": "^TNX",
           "N225": "^N225", "KOSPI": "^KS11", "USDTWD": "TWD=X"}
GLOBAL = ["SOX", "TSM", "VIX", "US10Y"]      # 國際層
REGION = ["N225", "KOSPI", "USDTWD"]         # 區域層（亞洲鄰近市場、匯率）
BUY, SELL, COST = 0.55, 0.45, 0.002          # 買進/賣出門檻、單次換倉成本


def _parse_market_number(value):
    text = str(value or "").strip().replace(",", "").replace(" ", "")
    if text in ("", "--", "---", "X", "N/A"):
        return None
    try:
        number = float(text)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _fetch_twse_taiex_day(day):
    """Fetch one completed TAIEX close from the official TWSE daily index endpoint."""
    date_text = pd.Timestamp(day).strftime("%Y%m%d")
    url = (
        "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?"
        + urllib.parse.urlencode({"date": date_text, "type": "ALL", "response": "json"})
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; TAIEXDataQuality/1.0)",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.twse.com.tw/"
    })
    raw = _read_with_backoff(req, timeout=20)
    payload = json.loads(raw.decode("utf-8-sig"))
    status = str(payload.get("stat", ""))
    if "沒有符合" in status or "查無" in status:
        return None

    tables = payload.get("tables") or []
    if isinstance(payload.get("data"), list):
        tables = list(tables) + [{"fields": payload.get("fields") or [], "data": payload["data"]}]
    for table in tables:
        fields = table.get("fields") or []
        for row in table.get("data") or []:
            if isinstance(row, dict):
                record = row
            elif fields and len(row) == len(fields):
                record = dict(zip(fields, row))
            else:
                continue
            index_name = str(
                record.get("指數") or record.get("指數名稱")
                or record.get("Index") or record.get("名稱") or ""
            )
            if "發行量加權股價指數" not in index_name:
                continue
            close = (
                record.get("收盤指數") or record.get("收盤")
                or record.get("Close") or record.get("close")
            )
            value = _parse_market_number(close)
            if value is None or not (1000 <= value <= 100000):
                raise ValueError(f"TWSE 加權指數收盤值異常：{close!r}")
            return value
    return None


def _fetch_twse_taiex_recent(now, calendar_days=7):
    """Use TWSE as the authoritative fallback for recent completed daily closes."""
    local_now = pd.Timestamp(now).tz_convert("Asia/Taipei")
    end_day = local_now.normalize().tz_localize(None)
    if local_now.hour < 14:
        end_day -= pd.Timedelta(days=1)
    start_day = end_day - pd.Timedelta(days=calendar_days)
    days = pd.bdate_range(start_day, end_day)
    rows = {}
    for day in days:
        try:
            value = _fetch_twse_taiex_day(day)
            if value is not None:
                rows[pd.Timestamp(day).normalize()] = value
        except Exception as exc:
            print(
                f"TWSE 加權指數備援暫不可用（{day.date()}）："
                f"{type(exc).__name__}: {exc}", flush=True
            )
            # Stop on upstream/network failure; do not hammer the official endpoint.
            if isinstance(exc, urllib.error.HTTPError) and (exc.code == 429 or exc.code >= 500):
                break
            if isinstance(exc, (TimeoutError, OSError, urllib.error.URLError)):
                break
    return rows


def _fetch_fred_reference(series_id, now, calendar_days=120):
    """Fetch a clearly-labelled FRED reference series; never silently substitute its definition."""
    end_day = pd.Timestamp(now).normalize().tz_localize(None)
    start_day = end_day - pd.Timedelta(days=calendar_days)
    params = {
        "id": series_id,
        "cosd": str(start_day.date()),
        "coed": str(end_day.date())
    }
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "TAIEXDataQuality/1.0",
        "Accept": "text/csv,*/*"
    })
    raw = _read_with_backoff(req, timeout=25)
    frame = pd.read_csv(io.BytesIO(raw))
    frame.columns = [str(x).lstrip("\ufeff").strip() for x in frame.columns]
    date_col = next((x for x in ("observation_date", "DATE", "date") if x in frame.columns), None)
    value_col = series_id if series_id in frame.columns else (
        "VALUE" if "VALUE" in frame.columns else None
    )
    if date_col is None or value_col is None:
        raise ValueError(f"FRED {series_id} CSV 欄位不符合預期")
    dates = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    values = pd.to_numeric(frame[value_col], errors="coerce")
    mask = dates.notna() & values.notna()
    series = pd.Series(values[mask].to_numpy(), index=pd.DatetimeIndex(dates[mask]))
    return series[~series.index.duplicated(keep="last")].sort_index()


def _fetch_finmind_tsm_reference(now, calendar_days=120):
    """Fetch adjusted TSM daily close from FinMind for source cross-check only."""
    end_day = pd.Timestamp(now).normalize().tz_localize(None)
    start_day = end_day - pd.Timedelta(days=calendar_days)
    frame = _fetch_finmind_data(
        "USStockPrice", start_day, end_day, data_id="TSM"
    )
    if frame.empty or "date" not in frame or "Adj_Close" not in frame:
        raise ValueError("FinMind TSM 缺少 date／Adj_Close 欄位")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    values = pd.to_numeric(frame["Adj_Close"], errors="coerce")
    mask = dates.notna() & values.notna() & (values > 0)
    series = pd.Series(values[mask].to_numpy(), index=pd.DatetimeIndex(dates[mask]))
    return series[~series.index.duplicated(keep="last")].sort_index()


def _fetch_market_reference_data(now):
    """Independent daily references for cross-checking; these do not enter model features."""
    references = {}
    specs = (
        ("VIX", "VIXCLS", "點", "FRED／CBOE VIX 日收盤；參考資料"),
        ("US10Y", "DGS10", "%", "FRED 美國 10 年期固定期限殖利率；定義與 Yahoo ^TNX 略有差異"),
        ("USDTWD", "DEXTAUS", "TWD/USD", "FRED 紐約中午匯率；不是 Yahoo TWD=X 即時報價")
    )
    for key, series_id, unit, note in specs:
        try:
            series = _fetch_fred_reference(series_id, now)
            if len(series):
                day = pd.Timestamp(series.index[-1]).date()
                references[key] = {
                    "source": f"FRED {series_id}",
                    "date": str(day),
                    "value": round(float(series.iloc[-1]), 4),
                    "unit": unit,
                    "note": note
                }
        except Exception as exc:
            print(f"FRED {series_id} 參考資料暫不可用：{type(exc).__name__}: {exc}", flush=True)
    try:
        series = _fetch_finmind_tsm_reference(now)
        if len(series):
            references["TSM"] = {
                "source": "FinMind USStockPrice Adj_Close",
                "date": str(pd.Timestamp(series.index[-1]).date()),
                "value": round(float(series.iloc[-1]), 4),
                "unit": "USD",
                "note": "還原收盤價；僅供交叉參考，不直接替代模型序列"
            }
    except Exception as exc:
        print(f"FinMind TSM 參考資料暫不可用：{type(exc).__name__}: {exc}", flush=True)
    return references


def load_live(years=10, cache_path="market_history.csv"):
    """Load multi-year market history, persist it, then refresh only the recent window."""
    import yfinance as yf

    cached = None
    if os.path.isfile(cache_path) and os.path.getsize(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached.index = pd.to_datetime(cached.index).tz_localize(None)
            cached = cached[~cached.index.duplicated(keep="last")].sort_index()
            if "TAIEX" not in cached or cached["TAIEX"].dropna().empty:
                cached = None
        except Exception as exc:
            print(f"既有行情快取無法讀取，改抓完整歷史資料：{type(exc).__name__}: {exc}", flush=True)
            cached = None

    last_error = None
    fresh = None
    target_start = (pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
                    - pd.DateOffset(years=years))
    cache_needs_extension = (
        cached is not None and
        (cached["TAIEX"].dropna().index.min() > target_start + pd.Timedelta(days=90))
    )
    period = f"{years}y" if cached is None or cache_needs_extension else "3mo"
    if cache_needs_extension:
        print(
            f"歷史行情快取尚未涵蓋目標 {years} 年，嘗試擴充完整歷史；"
            "若來源限流，會保留既有快取並稍後重試。",
            flush=True
        )
    for attempt in range(3):
        try:
            raw = yf.download(list(TICKERS.values()), period=period,
                              auto_adjust=True, progress=False, threads=False, timeout=20)
            if raw is None or raw.empty:
                raise ValueError("Yahoo Finance 沒有回傳行情資料")
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    close = raw.xs("Close", axis=1, level=0)
                elif "Close" in raw.columns.get_level_values(1):
                    close = raw.xs("Close", axis=1, level=1)
                else:
                    raise ValueError("Yahoo Finance 回傳欄位缺少 Close")
            else:
                close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame()
            if isinstance(close.columns, pd.MultiIndex):
                close.columns = close.columns.get_level_values(-1)
            fresh = close.rename(columns={v: k for k, v in TICKERS.items()})
            if fresh.empty or not fresh.notna().any().any():
                raise ValueError("Yahoo Finance 回傳資料沒有可用行情欄位")
            if "TAIEX" not in fresh or fresh["TAIEX"].dropna().empty:
                print("Yahoo Finance 本次缺少加權指數欄位，將由 TWSE 官方資料／快取補足。", flush=True)
            break
        except Exception as exc:
            last_error = exc
            print(f"行情下載失敗（第 {attempt + 1}/3 次）：{type(exc).__name__}: {exc}", flush=True)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            fresh = None

    yahoo_available = fresh is not None
    if fresh is not None:
        fresh.index = pd.to_datetime(fresh.index).tz_localize(None)
        fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
        df = fresh.combine_first(cached) if cached is not None else fresh
    elif cached is not None:
        df = cached.copy()
        print(f"Yahoo Finance 暫不可用，改用既有行情快取：{len(df)} 筆；最後錯誤：{last_error}", flush=True)
    else:
        df = pd.DataFrame()

    # Yahoo may throttle cloud-hosted requests. Refresh the Taiwan index close
    # directly from TWSE, while leaving unavailable overseas series on cache.
    now = pd.Timestamp.now(tz="Asia/Taipei")
    official_taiex = _fetch_twse_taiex_recent(now)
    if official_taiex:
        if "TAIEX" not in df.columns:
            df["TAIEX"] = np.nan
        else:
            # pandas 3.x rejects assigning fractional official closes into an
            # integer-typed cached series; normalize before merging TWSE data.
            df["TAIEX"] = pd.to_numeric(df["TAIEX"], errors="coerce").astype(float)
        for day, value in official_taiex.items():
            df.loc[day, "TAIEX"] = value
        print(
            f"TWSE 官方加權指數備援補入 {len(official_taiex)} 個交易日；"
            f"最新日期 {max(official_taiex).date()}",
            flush=True
        )
    if df.empty or "TAIEX" not in df or df["TAIEX"].dropna().empty:
        raise RuntimeError(
            f"無法取得加權指數歷史資料，且官方備援與既有快取皆不可用。最後 Yahoo 錯誤：{last_error}"
        )

    df = df[~df.index.duplicated(keep="last")].sort_index()
    if fresh is not None or official_taiex:
        tmp = cache_path + ".tmp"
        df.to_csv(tmp, index_label="date")
        os.replace(tmp, cache_path)
        taiex_dates = df["TAIEX"].dropna().index
        print(
            f"行情歷史快取已更新：{len(taiex_dates)} 筆；"
            f"涵蓋 {taiex_dates.min().date()} 至 {taiex_dates.max().date()}",
            flush=True
        )
        try:
            import storage
            if storage.enabled():
                saved = storage.save_data()
                print(f"行情快取已持久化至 Supabase（檔案數 {saved}）", flush=True)
        except Exception as exc:
            print(f"行情快取 Supabase 保存失敗：{type(exc).__name__}: {exc}", flush=True)

    # Capture provider coverage before limited forward-fill so stale inputs are visible.
    yahoo_columns = {
        column for column in TICKERS
        if yahoo_available and column in fresh and fresh[column].notna().any()
    }
    yahoo_status = (
        "ok" if len(yahoo_columns) == len(TICKERS)
        else "partial" if yahoo_columns else "limited_or_unavailable"
    )
    reference_data = (
        _fetch_market_reference_data(now)
        if yahoo_status != "ok" else {}
    )
    if reference_data:
        summary = "、".join(f"{key}:{value['date']}" for key, value in reference_data.items())
        print(f"獨立備援參考資料已取得：{summary}", flush=True)
    elif yahoo_status != "ok":
        print("獨立備援參考資料本次未取得；不以空值冒充最新資料。", flush=True)
    source_frame = df.copy()
    df = df.dropna(subset=["TAIEX"]).ffill(limit=3)
    data_status = {
        "updated_at": now.isoformat(),
        "yahoo_status": yahoo_status,
        "yahoo_columns": sorted(yahoo_columns),
        "taiwan_index_source": "TWSE 官方日收盤資料" if official_taiex else (
            "Yahoo Finance" if yahoo_available and "TAIEX" in yahoo_columns else "既有快取"
        ),
        "taiwan_index_official_rows": len(official_taiex),
        "reference_data": reference_data,
        "reference_data_note": "備援參考值只供交叉核對，不直接替代模型特徵；各資料有自己的定義與發布時間。",
        "series": {}
    }
    for column in TICKERS:
        if column not in source_frame or source_frame[column].dropna().empty:
            data_status["series"][column] = {
                "source": "無可用資料", "latest_date": None, "age_calendar_days": None
            }
            continue
        latest = pd.Timestamp(source_frame[column].last_valid_index()).tz_localize(None)
        if column == "TAIEX" and official_taiex:
            source = "TWSE 官方日收盤資料"
        elif yahoo_available and column in fresh and fresh[column].notna().any():
            source = "Yahoo Finance"
        else:
            source = "快取（Yahoo Finance 未更新）"
        data_status["series"][column] = {
            "source": source,
            "latest_date": str(latest.date()),
            "age_calendar_days": max(0, int((now.normalize().tz_localize(None) - latest.normalize()).days))
        }
    df.attrs["data_status"] = data_status
    if now.hour < 14:
        df = df[df.index < now.tz_localize(None).normalize()]
    if len(df) < 500:
        raise ValueError(f"有效大盤歷史資料不足：只有 {len(df)} 個交易日")
    return df

def load_demo(n=1800, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    r = pd.DataFrame(rng.normal(0, 0.01, (n, len(TICKERS))), index=idx, columns=list(TICKERS))
    r["TAIEX"] = 0.0003 + 0.3 * r["SOX"].shift(1).fillna(0) + 0.15 * r["KOSPI"].shift(1).fillna(0) + rng.normal(0, 0.008, n)
    return 100 * (1 + r).cumprod()

# Keep each scheduled run bounded. Recent dates are always prioritized while
# older missing dates are backfilled gradually across subsequent runs.
# Each scheduled run backfills a larger bounded batch while preserving checkpoints.
BACKFILL_DATES_PER_RUN = 200
RECENT_DATES_PER_RUN = 5
CHECKPOINT_EVERY = 5


def _pending_dates(old, today, days_back, trading_days=None, required_columns=None):
    start = today - pd.Timedelta(days=days_back)
    if trading_days is None:
        dates = pd.bdate_range(start, today)
    else:
        dates = pd.DatetimeIndex(trading_days).tz_localize(None).normalize().unique().sort_values()
        dates = dates[(dates >= start) & (dates <= today)]
    if len(old):
        # An index entry is not proof that a source row was downloaded.
        # Retry dates whose required values are absent, avoiding permanently
        # skipped dates left by earlier partial or empty checkpoints.
        if required_columns:
            missing_columns = [c for c in required_columns if c not in old.columns]
            if missing_columns:
                complete = pd.Series(False, index=old.index)
            else:
                complete = old[list(required_columns)].notna().all(axis=1)
            existing = pd.DatetimeIndex(old.index[complete]).tz_localize(None).normalize().unique()
        else:
            existing = pd.DatetimeIndex(old.index).tz_localize(None).normalize().unique()
        dates = dates[~dates.isin(existing)]
    if not len(dates):
        return dates
    # Keep the newest data fresh first; only then spend the remaining request
    # budget on older gaps. Failed dates remain absent and are retried later.
    recent = dates[-RECENT_DATES_PER_RUN:]
    older = dates[:-RECENT_DATES_PER_RUN][:BACKFILL_DATES_PER_RUN]
    return recent.append(older).unique()

def _open_with_backoff(req, timeout):
    """Retry transient throttling, server errors, and network timeouts."""
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            if attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = float(retry_after) if retry_after else (5, 15, 30)[attempt]
            except (TypeError, ValueError):
                delay = (5, 15, 30)[attempt]
            delay = max(1, min(60, delay))
            print(f"資料來源 HTTP {exc.code}，{delay:.0f} 秒後重試（{attempt + 1}/3）", flush=True)
            time.sleep(delay)
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            if attempt == 3:
                raise
            delay = (5, 15, 30)[attempt]
            print(f"資料來源連線逾時／網路錯誤（{type(exc).__name__}），{delay} 秒後重試（{attempt + 1}/3）", flush=True)
            time.sleep(delay)


def _read_with_backoff(req, timeout):
    """Retry opening and reading as one request, avoiding nested retry loops."""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            if attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = float(retry_after) if retry_after else (5, 15, 30)[attempt]
            except (TypeError, ValueError):
                delay = (5, 15, 30)[attempt]
            delay = max(1, min(60, delay))
            print(f"資料來源 HTTP {exc.code}，{delay:.0f} 秒後重試（{attempt + 1}/3）", flush=True)
            time.sleep(delay)
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            if attempt == 3:
                raise
            delay = (5, 15, 30)[attempt]
            print(f"資料讀取逾時／網路錯誤（{type(exc).__name__}），{delay} 秒後重試（{attempt + 1}/3）", flush=True)
            time.sleep(delay)


def _merge_and_save(old, rows, path):
    out = pd.concat([old, pd.DataFrame.from_dict(rows, orient="index")]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if len(out):
        tmp = path + ".tmp"
        out.to_csv(tmp, index_label="date")
        os.replace(tmp, path)
        try:
            import storage
            if storage.enabled():
                saved = storage.save_data()
                print(f"已保存檢查點：{path}（Supabase 檔案數 {saved}）", flush=True)
        except Exception as exc:
            raise RuntimeError(f"資料檢查點寫入 Supabase 失敗：{path}") from exc
    return out


def _checkpoint_if_needed(old, rows, path):
    if rows and len(rows) % CHECKPOINT_EVERY == 0:
        _merge_and_save(old, rows, path)

def _is_transient_fetch_error(exc):
    """Return True for throttling, upstream server failures, and network timeouts."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (TimeoutError, OSError, urllib.error.URLError))


def _fetch_finmind_data(dataset, start_date, end_date, data_id=None):
    """Fetch a bounded date range from FinMind; token is optional for its public quota."""
    params = {"dataset": dataset, "start_date": str(pd.Timestamp(start_date).date()),
              "end_date": str(pd.Timestamp(end_date).date())}
    if data_id:
        params["data_id"] = data_id
    url = "https://api.finmindtrade.com/api/v4/data?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "TaiexSignal/1.0"}
    token = os.environ.get("FINMIND_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    payload = json.loads(_read_with_backoff(req, timeout=25).decode("utf-8"))
    if payload.get("status") != 200 or not isinstance(payload.get("data"), list):
        raise ValueError(f"FinMind {dataset} 回傳狀態或資料格式不正確")
    return pd.DataFrame(payload["data"])




def _fetch_finmind_data_for_dates(dataset, dates, data_id=None, batch_size=25):
    """Fetch sparse historical gaps in small date batches so one timeout cannot lose the whole backfill."""
    dates = pd.DatetimeIndex(dates).tz_localize(None).normalize().unique().sort_values()
    frames = []
    for offset in range(0, len(dates), batch_size):
        batch = dates[offset:offset + batch_size]
        try:
            frame = _fetch_finmind_data(dataset, batch.min(), batch.max(), data_id=data_id)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            print(
                f"FinMind {dataset} 區間 {batch.min().date()} 至 {batch.max().date()} "
                f"暫不可用：{type(exc).__name__}；其餘區間繼續回補",
                flush=True
            )
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if "date" in result:
        # Preserve multiple investor/product rows for the same date.
        keys = ["date"] + [key for key in
                           ("name", "institutional_investors", "futures_id", "contract_date", "call_put")
                           if key in result.columns]
        result = result.drop_duplicates(subset=keys, keep="last")
    return result


def fetch_chips(days_back=3650, path="chips.csv", trading_days=None):
    """證交所三大法人買賣超（元），增量寫入 chips.csv（date, foreign_net, trust_net）。"""
    old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    dates = _pending_dates(old, today, days_back, trading_days, required_columns=("foreign_net", "trust_net"))
    rows = {}

    # FinMind first for older historical gaps; TWSE remains primary for recent data.
    recent_cutoff = today - pd.DateOffset(years=3)
    historical_dates = dates[dates < recent_cutoff]
    try:
        if len(historical_dates):
            alt = _fetch_finmind_data_for_dates("TaiwanStockTotalInstitutionalInvestors", historical_dates)
            if not alt.empty and {"date", "name", "buy", "sell"}.issubset(alt.columns):
                alt["date"] = pd.to_datetime(alt["date"], errors="coerce").dt.normalize()
                alt["net"] = pd.to_numeric(alt["buy"], errors="coerce") - pd.to_numeric(alt["sell"], errors="coerce")
                for date, group in alt.dropna(subset=["date", "net"]).groupby("date"):
                    if date not in historical_dates:
                        continue
                    by_name = group.groupby("name")["net"].sum()
                    foreign = float(by_name.get("Foreign_Investor", 0)) + float(by_name.get("Foreign_Dealer_Self", 0))
                    trust = by_name.get("Investment_Trust")
                    if trust is not None:
                        rows[date] = {"foreign_net": foreign, "trust_net": float(trust)}
                print(f"FinMind 三大法人歷史資料先行補入 {len(rows)} 個日期", flush=True)
    except Exception as exc:
        print(f"FinMind 三大法人主備援暫不可用：{type(exc).__name__}: {exc}", flush=True)

    for d in dates:
        if d in rows or d < recent_cutoff:
            continue
        url = f"https://www.twse.com.tw/rwd/zh/fund/BFI82U?type=day&dayDate={d:%Y%m%d}&response=json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.loads(_read_with_backoff(req, timeout=15).decode("utf-8"))
        except urllib.error.HTTPError as e:
            if _is_transient_fetch_error(e):
                print(f"三大法人 {d.date()} HTTP {e.code}：上游暫時不可用，停止本次回補；保留已取得資料供下次接續", flush=True)
                break
            print(f"三大法人 {d.date()} HTTP {e.code}：略過此日期", flush=True)
            time.sleep(1); continue
        except Exception as e:
            if _is_transient_fetch_error(e):
                print(f"三大法人 {d.date()} 網路持續失敗（{type(e).__name__}）：停止本次回補；保留已取得資料供下次接續", flush=True)
                break
            print(f"三大法人 {d.date()} 下載／解析失敗：{type(e).__name__}: {e}", flush=True)
            time.sleep(1); continue
        time.sleep(0.7)
        if j.get("stat") != "OK": continue  # 休市日或尚未公布
        diff = {r[0]: float(str(r[3]).replace(",", "")) for r in j["data"]}
        fgn = sorted([k for k in diff if "外資及陸資" in k], key=lambda k: "不含" in k)  # 優先取含外資自營商的整體
        trust = [k for k in diff if "投信" in k]
        if fgn and trust:
            rows[d] = {"foreign_net": diff[fgn[0]], "trust_net": diff[trust[0]]}
            _checkpoint_if_needed(old, rows, path)
        else:
            print("欄位名稱與預期不同，請檢查", d.date(), list(diff))
    # TWSE is primary. Fill only missing dates from FinMind market-wide data.
    try:
        needed = [d for d in dates if d not in rows and d >= recent_cutoff]
        if needed:
            alt = _fetch_finmind_data_for_dates("TaiwanStockTotalInstitutionalInvestors", needed)
            if not alt.empty and {"date", "name", "buy", "sell"}.issubset(alt.columns):
                alt["date"] = pd.to_datetime(alt["date"], errors="coerce").dt.normalize()
                alt["net"] = pd.to_numeric(alt["buy"], errors="coerce") - pd.to_numeric(alt["sell"], errors="coerce")
                for date, group in alt.dropna(subset=["date", "net"]).groupby("date"):
                    if date not in needed:
                        continue
                    by_name = group.groupby("name")["net"].sum()
                    foreign = float(by_name.get("Foreign_Investor", 0)) + float(by_name.get("Foreign_Dealer_Self", 0))
                    trust = by_name.get("Investment_Trust")
                    if trust is not None:
                        rows[date] = {"foreign_net": foreign, "trust_net": float(trust)}
                print(f"FinMind 籌碼備援補入 {sum(d in rows for d in needed)} 個日期", flush=True)
    except Exception as exc:
        print(f"FinMind 籌碼備援暫不可用：{type(exc).__name__}", flush=True)
    return _merge_and_save(old, rows, path)

def demo_chips(df, seed=3):  # 合成籌碼，僅供示範
    rng = np.random.default_rng(seed)
    f = df["TAIEX"].pct_change().shift(-1).fillna(0) * 4e11 + rng.normal(0, 8e9, len(df))
    return pd.DataFrame({"foreign_net": f, "trust_net": 0.2 * f + rng.normal(0, 3e9, len(df))}, index=df.index)

def _num(x):
    try: return float(str(x).replace(",", ""))
    except ValueError: return None

def _tx_foreign_oi_net(tabs):
    """從期交所頁面表格取「臺股期貨／外資／未平倉多空淨額口數」（12 個數字中的第 11 個）。"""
    for tb in tabs:
        cells = tb.astype(str).values.tolist()
        hit = [i for i, r in enumerate(cells) if "臺股期貨" in r]
        for r in (cells[hit[0]:hit[0] + 3] if hit else []):  # 自營商／投信／外資三列
            k = next((i for i, c in enumerate(r) if "外資" in c), None)
            if k is not None:
                nums = [v for v in (_num(c) for c in r[k + 1:]) if v is not None]
                return nums[10] if len(nums) >= 12 else None
    return None

def _fetch_taifex_openapi_latest():
    """Fetch the latest official TAIFEX institutional futures snapshot."""
    url = ("https://openapi.taifex.com.tw/v1/"
           "MarketDataOfMajorInstitutionalTradersDetailsOfFuturesContractsBytheDate")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    payload = json.loads(_read_with_backoff(req, timeout=15).decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("TAIFEX OpenAPI 回傳格式不是資料列清單")
    for row in payload:
        if row.get("ContractCode") != "臺股期貨" or row.get("Item") != "外資及陸資":
            continue
        date = pd.to_datetime(str(row.get("Date", "")), format="%Y%m%d", errors="coerce")
        net = _num(row.get("OpenInterest(Net)"))
        if pd.isna(date) or net is None:
            raise ValueError("TAIFEX OpenAPI 最新資料缺少有效日期或未平倉淨額")
        return date.normalize(), net
    raise ValueError("TAIFEX OpenAPI 找不到臺股期貨／外資及陸資資料列")


def fetch_futures(days_back=3650, path="futures.csv", trading_days=None):
    """臺指期外資未平倉淨口數；FinMind 自 2018-06-05 起優先回補，TAIFEX 官方資料作近期備援。"""
    from io import StringIO
    old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    source_start = pd.Timestamp("2018-06-05")
    if trading_days is None:
        eligible_days = pd.bdate_range(today - pd.Timedelta(days=days_back), today)
    else:
        eligible_days = pd.DatetimeIndex(trading_days).tz_localize(None).normalize().unique().sort_values()
    eligible_days = eligible_days[eligible_days >= source_start]
    dates = _pending_dates(old, today, days_back, eligible_days,
                           required_columns=("foreign_oi_net",))
    rows = {}

    # FinMind has a documented TX institutional futures history from 2018-06-05.
    # Query it first so dates outside TAIFEX's ordinary three-year web window
    # are not repeatedly sent to the exchange's rate-limited historical form.
    try:
        if len(dates):
            alt = _fetch_finmind_data_for_dates(
                "TaiwanFuturesInstitutionalInvestors", dates, data_id="TX"
            )
            required = {"date", "institutional_investors",
                        "long_open_interest_balance_volume",
                        "short_open_interest_balance_volume"}
            if not alt.empty and required.issubset(alt.columns):
                alt["date"] = pd.to_datetime(alt["date"], errors="coerce").dt.normalize()
                for col in ("long_open_interest_balance_volume",
                            "short_open_interest_balance_volume"):
                    alt[col] = pd.to_numeric(alt[col], errors="coerce")
                foreign = alt[
                    alt["institutional_investors"].astype(str).isin(
                        ["Foreign_Investor", "外資", "外資及陸資"]
                    )
                ].dropna(subset=[
                    "date", "long_open_interest_balance_volume",
                    "short_open_interest_balance_volume"
                ])
                foreign = foreign.assign(
                    net=foreign["long_open_interest_balance_volume"] -
                        foreign["short_open_interest_balance_volume"]
                )
                for date, group in foreign.groupby("date"):
                    if date in dates:
                        rows[date] = {"foreign_oi_net": float(group["net"].sum())}
                print(f"FinMind 臺指期歷史資料先行補入 {len(rows)} 個日期", flush=True)
    except Exception as exc:
        print(f"FinMind 臺指期歷史資料暫不可用：{type(exc).__name__}: {exc}", flush=True)

    # TAIFEX's public historical query is used only for recent dates.
    recent_cutoff = today - pd.DateOffset(years=3)
    try:
        latest_date, latest_net = _fetch_taifex_openapi_latest()
        if latest_date in dates and latest_date not in rows:
            rows[latest_date] = {"foreign_oi_net": latest_net}
            print(f"期交所 OpenAPI 已取得最新交易日：{latest_date.date()}", flush=True)
    except Exception as exc:
        print(f"期交所 OpenAPI 備援暫不可用：{type(exc).__name__}: {exc}", flush=True)

    for d in dates:
        if d in rows or d < recent_cutoff:
            continue
        time.sleep(0.7)
        body = urllib.parse.urlencode({
            "queryType": "1", "goDay": "", "doQuery": "1",
            "dateaddcnt": "", "queryDate": f"{d:%Y/%m/%d}",
            "commodityId": ""
        }).encode()
        try:
            req = urllib.request.Request(
                "https://www.taifex.com.tw/cht/3/futContractsDate",
                body, {"User-Agent": "Mozilla/5.0"}
            )
            html = _read_with_backoff(req, timeout=20).decode("utf-8", "ignore")
            time.sleep(0.7)
            if "臺股期貨" not in html:
                continue
            net = _tx_foreign_oi_net(pd.read_html(StringIO(html)))
        except urllib.error.HTTPError as exc:
            print(f"期交所 {d.date()} HTTP {exc.code}：停止本次剩餘回補，保留已取得資料", flush=True)
            if exc.code == 429 or exc.code >= 500:
                break
            time.sleep(1)
            continue
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            print(f"期交所 {d.date()} 連線失敗：{type(exc).__name__}；停止本次回補", flush=True)
            break
        except Exception as exc:
            print(f"期交所 {d.date()} 解析失敗：{type(exc).__name__}: {exc}", flush=True)
            time.sleep(1)
            continue
        if net is not None:
            rows[d] = {"foreign_oi_net": net}

    if not rows and not len(old):
        print("期貨：本次沒有取得資料，將在下次執行重試", flush=True)
    return _merge_and_save(old, rows, path)

def demo_futures(df, seed=9):  # 合成期貨淨未平倉，僅供示範
    rng = np.random.default_rng(seed)
    r = df["TAIEX"].pct_change().fillna(0)
    return pd.DataFrame({"foreign_oi_net": np.cumsum(r * 2e5 + rng.normal(0, 1500, len(df)))}, index=df.index)

def fetch_margin(days_back=3650, path="margin.csv", trading_days=None):
    """證交所市場融資餘額（元），增量寫入 margin.csv（date, margin_bal）。資料約晚上 9 點後公布。"""
    old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    dates = _pending_dates(old, today, days_back, trading_days, required_columns=("margin_bal", "margin_units", "short_units"))
    rows = {}

    # FinMind first for older historical gaps; TWSE remains primary for recent data.
    recent_cutoff = today - pd.DateOffset(years=3)
    historical_dates = dates[dates < recent_cutoff]
    try:
        if len(historical_dates):
            alt = _fetch_finmind_data_for_dates("TaiwanStockTotalMarginPurchaseShortSale", historical_dates)
            if not alt.empty and {"date", "name", "TodayBalance"}.issubset(alt.columns):
                alt["date"] = pd.to_datetime(alt["date"], errors="coerce").dt.normalize()
                alt["TodayBalance"] = pd.to_numeric(alt["TodayBalance"], errors="coerce")
                for date, group in alt.dropna(subset=["date", "TodayBalance"]).groupby("date"):
                    if date not in historical_dates:
                        continue
                    balances = group.groupby("name")["TodayBalance"].last()
                    money, margin_units, short_units = (balances.get("MarginPurchaseMoney"),
                                                        balances.get("MarginPurchase"),
                                                        balances.get("ShortSale"))
                    if all(v is not None and pd.notna(v) for v in (money, margin_units, short_units)):
                        rows[date] = {"margin_bal": float(money), "margin_units": float(margin_units),
                                      "short_units": float(short_units)}
                print(f"FinMind 融資融券歷史資料先行補入 {len(rows)} 個日期", flush=True)
    except Exception as exc:
        print(f"FinMind 融資融券主備援暫不可用：{type(exc).__name__}: {exc}", flush=True)

    for d in dates:
        if d in rows or d < recent_cutoff:
            continue
        url = f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d:%Y%m%d}&selectType=MS&response=json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.loads(_read_with_backoff(req, timeout=15).decode("utf-8"))
        except urllib.error.HTTPError as e:
            if _is_transient_fetch_error(e):
                print(f"融資融券 {d.date()} HTTP {e.code}：上游暫時不可用，停止本次回補；保留已取得資料供下次接續", flush=True)
                break
            print(f"融資融券 {d.date()} HTTP {e.code}：略過此日期", flush=True)
            time.sleep(1); continue
        except Exception as e:
            if _is_transient_fetch_error(e):
                print(f"融資融券 {d.date()} 網路持續失敗（{type(e).__name__}）：停止本次回補；保留已取得資料供下次接續", flush=True)
                break
            print(f"融資融券 {d.date()} 下載／解析失敗：{type(e).__name__}: {e}", flush=True)
            time.sleep(1); continue
        time.sleep(0.7)
        if j.get("stat") != "OK": continue
        tabs = j.get("tables") or [{"data": j.get("creditList", [])}]
        last = {str(r[0]): _num(r[-1]) for t in tabs for r in t.get("data", [])}  # 最後一欄＝今日餘額
        pick = lambda pre: next((v for k, v in last.items() if k.startswith(pre) and v is not None), None)
        amt, lot, sht = pick("融資金額"), pick("融資(交易單位)"), pick("融券(交易單位)")
        if None not in (amt, lot, sht):
            rows[d] = {"margin_bal": amt * 1000, "margin_units": lot, "short_units": sht}  # 仟元→元
            _checkpoint_if_needed(old, rows, path)
        else:
            print("找不到融資／融券欄位，請檢查", d.date(), list(last))
    # TWSE is primary. FinMind separates TWD amounts from share counts.
    try:
        needed = [d for d in dates if d not in rows and d >= recent_cutoff]
        if needed:
            alt = _fetch_finmind_data_for_dates("TaiwanStockTotalMarginPurchaseShortSale", needed)
            if not alt.empty and {"date", "name", "TodayBalance"}.issubset(alt.columns):
                alt["date"] = pd.to_datetime(alt["date"], errors="coerce").dt.normalize()
                alt["TodayBalance"] = pd.to_numeric(alt["TodayBalance"], errors="coerce")
                for date, group in alt.dropna(subset=["date", "TodayBalance"]).groupby("date"):
                    if date not in needed:
                        continue
                    balances = group.groupby("name")["TodayBalance"].last()
                    money, margin_units, short_units = (balances.get("MarginPurchaseMoney"),
                                                        balances.get("MarginPurchase"),
                                                        balances.get("ShortSale"))
                    if all(v is not None and pd.notna(v) for v in (money, margin_units, short_units)):
                        rows[date] = {"margin_bal": float(money), "margin_units": float(margin_units),
                                      "short_units": float(short_units)}
                print(f"FinMind 信用資料備援補入 {sum(d in rows for d in needed)} 個日期", flush=True)
    except Exception as exc:
        print(f"FinMind 信用資料備援暫不可用：{type(exc).__name__}", flush=True)
    return _merge_and_save(old, rows, path)

def demo_margin(df, seed=5):  # 合成融資餘額，僅供示範
    rng = np.random.default_rng(seed)
    r = df["TAIEX"].pct_change().fillna(0)
    bal = 2e11 * np.exp(np.cumsum(0.5 * r + rng.normal(0, 0.004, len(df))))
    short = bal / 1e6 * np.exp(np.cumsum(rng.normal(0, 0.01, len(df))))
    return pd.DataFrame({"margin_bal": bal, "margin_units": bal / 1e5, "short_units": short}, index=df.index)

def build(df, chips=None, margin=None, h=1, futures=None):
    r = df.pct_change()
    X = pd.DataFrame(index=df.index)
    for k in GLOBAL: X[f"國際:{k}"] = r[k]
    for k in REGION: X[f"區域:{k}"] = r[k]
    t = df["TAIEX"]
    X["本地:當日漲跌"] = r["TAIEX"]
    X["本地:動能5日"] = t.pct_change(5)
    X["本地:偏離20日均線"] = t / t.rolling(20).mean() - 1
    X["本地:波動10日"] = r["TAIEX"].rolling(10).std()
    first = df.index[0]
    # Optional data layers are admitted only when they overlap enough market
    # sessions to support their rolling features. Row count alone is unsafe:
    # a cache with many off-calendar or sparse rows can leave only a handful
    # of complete training samples and abort the whole scheduled refresh.
    chips_aligned = chips.reindex(X.index) if chips is not None else None
    chips_valid = (chips_aligned[["foreign_net", "trust_net"]].notna().all(axis=1)
                   if chips_aligned is not None and {"foreign_net", "trust_net"}.issubset(chips_aligned.columns)
                   else pd.Series(False, index=X.index))
    chips_count = int(chips_valid.sum())
    if chips is not None and chips_count >= 140:
        for k, name in [("foreign_net", "外資"), ("trust_net", "投信")]:
            c = chips_aligned[k]
            X[f"籌碼:{name}買賣超"] = c / c.abs().rolling(60, min_periods=20).mean()
            s = np.sign(c)
            X[f"籌碼:{name}連買連賣天數"] = s.groupby((s != s.shift()).cumsum()).cumsum().clip(-10, 10)
        first = max(first, chips_aligned.index[chips_valid].min())
    elif chips is not None:
        print(f"籌碼有效資料不足（完整有效交易日 {chips_count} 日，至少 140 日），本次略過此層")
    margin_aligned = margin.reindex(X.index) if margin is not None else None
    margin_valid = (margin_aligned["margin_bal"].notna()
                    if margin_aligned is not None and "margin_bal" in margin_aligned.columns
                    else pd.Series(False, index=X.index))
    margin_count = int(margin_valid.sum())
    if margin is not None and margin_count >= 140:
        b = margin_aligned["margin_bal"]
        X["信用:融資餘額5日變化"] = b.pct_change(5)
        X["信用:融資餘額偏離60日均"] = b / b.rolling(60, min_periods=20).mean() - 1
        X["信用:融資與指數背離"] = b.pct_change(5) - t.pct_change(5)
        short_valid = (margin_aligned[["short_units", "margin_units"]].notna().all(axis=1)
                       if {"short_units", "margin_units"}.issubset(margin_aligned.columns)
                       else pd.Series(False, index=X.index))
        if int(short_valid.sum()) >= 140:
            sh = margin_aligned["short_units"]
            ratio = sh / margin_aligned["margin_units"]
            X["信用:融券餘額5日變化"] = sh.pct_change(5)
            X["信用:券資比偏離60日均"] = ratio / ratio.rolling(60, min_periods=20).mean() - 1
        first = max(first, margin_aligned.index[margin_valid].min())
    elif margin is not None:
        print(f"信用有效資料不足（融資餘額有效交易日 {margin_count} 日，至少 140 日），本次略過此層")
    futures_aligned = futures.reindex(X.index) if futures is not None else None
    futures_valid = (futures_aligned["foreign_oi_net"].notna()
                     if futures_aligned is not None and "foreign_oi_net" in futures_aligned.columns
                     else pd.Series(False, index=X.index))
    futures_count = int(futures_valid.sum())
    if futures is not None and futures_count >= 120:
        o = futures_aligned["foreign_oi_net"]
        sd = o.rolling(120, min_periods=40).std()
        X["期貨:外資淨未平倉(相對近120日)"] = (o - o.rolling(120, min_periods=40).mean()) / sd
        X["期貨:外資淨未平倉5日變化"] = o.diff(5) / sd
        first = max(first, futures_aligned.index[futures_valid].min())
    elif futures is not None:
        print(f"期貨有效資料不足（外資未平倉有效交易日 {futures_count} 日，至少 120 日），本次略過此層")
    # 時序：美股 D 日收盤早於台股 D+1 開盤，故 D 列特徵預測 D+1 報酬，無資料洩漏
    fwd = t.shift(-h) / t - 1  # 未來 h 個交易日報酬（h=1 即隔日）
    y = (fwd > 0).astype(float).where(fwd.notna())
    return X.loc[first:], y.loc[first:], fwd.loc[first:]

def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))

def walk_forward(X, y, start=500, step=20, h=1):
    valid = X.notna().all(axis=1)
    idx = X.index[valid & y.notna()]
    minimum = max(80, h + 20)
    if len(idx) < minimum:
        # Diagnostic only: never relax the sample requirement or fill missing
        # observations. Report the exact feature coverage causing rejection.
        coverage = X.notna().sum().sort_values()
        missing = (1 - X.notna().mean()).sort_values(ascending=False)
        detail = []
        for col in coverage.index:
            s = X[col].dropna()
            detail.append({
                "feature": str(col),
                "valid_rows": int(coverage[col]),
                "missing_pct": round(float(missing[col]) * 100, 1),
                "first_date": str(s.index.min().date()) if len(s) else None,
                "last_date": str(s.index.max().date()) if len(s) else None,
            })
        print("資料品質診斷（僅列各特徵覆蓋率；不補值、不降低門檻）：" +
              json.dumps(detail, ensure_ascii=False), flush=True)
        raise ValueError(f"有效資料不足：完整樣本 {len(idx)} 筆，最低要求 {minimum} 筆；請依上方特徵覆蓋診斷修復來源或日期對齊。")
    prob = pd.Series(np.nan, index=X.index)
    first_test = min(max(start, h + 30), max(h + 30, len(idx) // 2))
    for i in range(first_test, len(idx), step):
        tr = idx[:i - h + 1]  # purge：排除標籤區間尚未結束的樣本
        if len(tr) < 30 or y.loc[tr].nunique() < 2:
            continue
        m = model().fit(X.loc[tr], y.loc[tr])
        te = idx[i:i + step]
        prob.loc[te] = m.predict_proba(X.loc[te])[:, 1]
    if y.loc[idx].nunique() < 2:
        raise ValueError("訓練資料只有單一類別，無法估計上漲機率。")
    m = model().fit(X.loc[idx], y.loc[idx])
    latest = X.index[valid][-1]
    raw_latest = float(m.predict_proba(X.loc[[latest]])[0, 1])

    # 只用歷史 walk-forward 樣本外預測及其已實現標籤校準最新機率。
    # 歷史回測值維持當時產生的原始預測，避免把未來校準資訊倒灌進回測。
    cal = prob.loc[idx].dropna()
    cal = cal[y.loc[cal.index].notna()]
    calibrated = raw_latest
    if len(cal) >= 100 and y.loc[cal.index].nunique() == 2:
        clipped = np.clip(cal.to_numpy(), 1e-5, 1 - 1e-5)
        z = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, max_iter=300)
        calibrator.fit(z, y.loc[cal.index].astype(int))
        z_latest = np.log(np.clip(raw_latest, 1e-5, 1 - 1e-5) /
                          (1 - np.clip(raw_latest, 1e-5, 1 - 1e-5)))
        calibrated = float(calibrator.predict_proba([[z_latest]])[0, 1])
    prob.loc[latest] = calibrated
    contrib = m[-1].coef_[0] * m[0].transform(X.loc[[latest]])[0]
    return prob, latest, pd.Series(contrib, index=X.columns)

def backtest(prob, fwd):
    p = prob.dropna()
    p = p[fwd.reindex(p.index).notna()]
    f = fwd.loc[p.index]
    actual = (f > 0).astype(float)
    # 擴張式基準只使用當時以前已實現的方向比例，避免把未來標籤洩漏進比較基準。
    baseline_prob = actual.expanding(min_periods=1).mean().shift(1).fillna(0.5)
    baseline_brier = float(((baseline_prob - actual) ** 2).mean())
    model_brier = float(((p - actual) ** 2).mean())
    baseline_direction_acc = float(((baseline_prob > 0.5) == (actual > 0.5)).mean())
    brier_skill = 1 - model_brier / baseline_brier if baseline_brier > 0 else None
    pos = pd.Series(np.where(p > BUY, 1.0, np.where(p < SELL, 0.0, np.nan)), index=p.index).ffill().fillna(0)
    turnover = pos.diff().abs()
    if len(turnover): turnover.iloc[0] = abs(float(pos.iloc[0]))  # 首筆進場也計入交易成本
    strat = pos * f - COST * turnover.fillna(0)
    def stat(s):
        eq = (1 + s).cumprod()
        return round((eq.iloc[-1] ** (252 / len(s)) - 1) * 100, 1), round((eq / eq.cummax() - 1).min() * 100, 1)
    a, d = stat(strat); ba, bd = stat(f)
    held = f[pos == 1]
    eq_s, eq_b = (1 + strat).cumprod(), (1 + f).cumprod()
    k = slice(None, None, 5)
    return {
        "days": int(len(p)), "trades": int(turnover.fillna(0).sum()),
        "direction_acc": round(float(((p > 0.5) == (f > 0)).mean()) * 100, 1),
        "baseline_direction_acc": round(baseline_direction_acc * 100, 1),
        "brier": round(model_brier, 4),
        "baseline_brier": round(baseline_brier, 4),
        "brier_skill": round(brier_skill, 4) if brier_skill is not None else None,
        "log_loss": round(float(-((f > 0) * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (f <= 0) * np.log(1 - np.clip(p, 1e-6, 1 - 1e-6))).mean()), 4),
        "hold_win_rate": round(float((held > 0).mean()) * 100, 1) if len(held) else None,
        "strategy": {"annual": a, "max_dd": d}, "buyhold": {"annual": ba, "max_dd": bd},
        "curve": {"dates": [str(x.date()) for x in p.index[k]],
                  "strategy": [round(float(v), 4) for v in eq_s.iloc[k]],
                  "buyhold": [round(float(v), 4) for v in eq_b.iloc[k]]},
    }

def weekly_report(prob, fwd, latest, contrib):
    """未來 5 個交易日展望：目前訊號，加上歷史上同類訊號之後 5 日的實際分布。"""
    p = prob.dropna()
    p = p[fwd.reindex(p.index).notna()]
    p = p.iloc[::5]  # 五日報酬區間重疊；每五個交易日取一筆，避免樣本數膨脹
    f = fwd.loc[p.index]
    name = lambda v: "偏多" if v > BUY else "偏空" if v < SELL else "中性"
    q = lambda s, a: round(float(s.quantile(a)) * 100, 2)
    buckets = {k: {"n": int(len(v)), "win": round(float((v > 0).mean()) * 100, 1), "avg": round(float(v.mean()) * 100, 2),
                   "p10": q(v, .1), "p90": q(v, .9)} for k, v in f.groupby(p.map(name))}
    pu = float(prob.loc[latest])
    layers = contrib.groupby(lambda c: c.split(":")[0]).sum()
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:8]
    return {"asof": str(latest.date()), "prob_up": round(pu, 3), "bucket": name(pu), "expected": buckets.get(name(pu)),
            "buckets": buckets, "direction_acc": round(float(((p > 0.5) == (f > 0)).mean()) * 100, 1), "days": int(len(p)),
            "layers": {k: round(float(v), 3) for k, v in layers.items()},
            "features": [{"name": k, "contrib": round(float(v), 3)} for k, v in top.items()]}

def historical_horizons(series):
    """非預測：以不重疊交易日區間計算歷史報酬分布，避免把重疊視窗誤當獨立樣本。"""
    out = {}
    for key, label, days in (("1m", "約 1 個月", 21), ("3m", "約 3 個月", 63), ("6m", "約 6 個月", 126)):
        forward = series.shift(-days) / series - 1
        # A single fixed non-overlap anchor can make the result sensitive to
        # where the sample starts. Evaluate every possible anchor and report
        # the range as a robustness diagnostic, not as independent samples.
        phase_rates = []
        phase_counts = []
        for offset in range(days):
            phase = forward.iloc[offset::days].dropna()
            if len(phase):
                phase_rates.append(float((phase > 0).mean()) * 100)
                phase_counts.append(int(len(phase)))
        values = forward.iloc[::days].dropna()
        overlapping = forward.dropna()
        if values.empty:
            out[key] = {"label": label, "days": days, "n": 0,
                        "overlap_n": int(len(overlapping))}
            continue
        n = int(len(values))
        wins = int((values > 0).sum())
        # Wilson 95% interval for the historical positive-return frequency.
        # This quantifies sampling uncertainty; it is not a future forecast interval.
        z = 1.96
        phat = wins / n
        denom = 1 + z * z / n
        center = (phat + z * z / (2 * n)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
        out[key] = {
            "label": label, "days": days, "n": n,
            "overlap_n": int(len(overlapping)),
            "overlap_positive_rate": round(float((overlapping > 0).mean()) * 100, 1),
            "positive_rate": round(phat * 100, 1),
            "positive_rate_ci95": [round(max(0.0, center - half) * 100, 1),
                                   round(min(1.0, center + half) * 100, 1)],
            "sample_note": "樣本偏少，估計不穩定" if n < 30 else "歷史樣本",
            "phase_positive_rate_range": [round(min(phase_rates), 1), round(max(phase_rates), 1)] if phase_rates else None,
            "phase_sample_count_range": [min(phase_counts), max(phase_counts)] if phase_counts else None,
            "sample_start": str(values.index.min().date()),
            "sample_end": str(values.index.max().date()),
            "mean": round(float(values.mean()) * 100, 2),
            "median": round(float(values.median()) * 100, 2),
            "p10": round(float(values.quantile(.10)) * 100, 2),
            "p90": round(float(values.quantile(.90)) * 100, 2),
            "min": round(float(values.min()) * 100, 2),
            "max": round(float(values.max()) * 100, 2),
        }
    return out


def write_taiwan_data_status(df, chips=None, margin=None, futures=None):
    """Persist Taiwan-only source coverage before any international/model validation."""
    now = pd.Timestamp.now(tz="Asia/Taipei")
    expected = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce")).dropna()
    expected = expected.tz_localize(None).normalize().unique().sort_values() if len(expected) else pd.DatetimeIndex([])

    def coverage(frame, columns, source, source_window_years=None):
        recent_start = now.normalize().tz_localize(None) - pd.Timedelta(days=365)
        recent_expected = expected[expected >= recent_start]
        empty = {
            "source": "無可用資料", "latest_date": None, "start_date": None,
            "valid_rows": 0, "expected_rows": int(len(expected)),
            "coverage_pct": 0.0 if len(expected) else None,
            "full_history_coverage_pct": 0.0 if len(expected) else None,
            "recent_1y_expected_rows": int(len(recent_expected)),
            "recent_1y_valid_rows": 0,
            "recent_1y_coverage_pct": 0.0 if len(recent_expected) else None,
            "observed_period_expected_rows": 0,
            "observed_period_coverage_pct": None,
            "source_window_years": source_window_years,
            "source_window_expected_rows": None,
            "source_window_valid_rows": None,
            "source_window_coverage_pct": None,
            "missing_rows": int(len(expected)), "missing_dates": [str(x.date()) for x in expected[:20]],
            "required_columns": list(columns), "age_calendar_days": None
        }
        if frame is None or not isinstance(frame, pd.DataFrame) or not len(frame):
            return empty
        if not set(columns).issubset(frame.columns):
            empty["source"] = source + "（缺少必要欄位）"
            return empty
        valid = frame[list(columns)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        valid_index = pd.DatetimeIndex(pd.to_datetime(frame.index[valid], errors="coerce")).dropna()
        valid_index = valid_index.tz_localize(None).normalize().unique().sort_values()
        latest = valid_index.max() if len(valid_index) else None
        missing = expected.difference(valid_index)
        observed_expected = (expected[(expected >= valid_index.min()) & (expected <= valid_index.max())]
                             if len(valid_index) else pd.DatetimeIndex([]))
        recent_valid = recent_expected.intersection(valid_index)
        source_expected = source_valid = None
        source_pct = None
        if source_window_years and len(expected):
            source_start = now.normalize().tz_localize(None) - pd.DateOffset(years=source_window_years)
            source_expected_dates = expected[expected >= source_start]
            source_valid_dates = source_expected_dates.intersection(valid_index)
            source_expected, source_valid = len(source_expected_dates), len(source_valid_dates)
            source_pct = round(source_valid / source_expected * 100, 1) if source_expected else None
        return {
            "source": source if latest is not None else "尚無完整有效資料",
            "latest_date": str(latest.date()) if latest is not None else None,
            "start_date": str(valid_index.min().date()) if len(valid_index) else None,
            "valid_rows": int(valid.sum()),
            "expected_rows": int(len(expected)),
            "coverage_pct": round(len(expected.intersection(valid_index)) / len(expected) * 100, 1) if len(expected) else None,
            "full_history_coverage_pct": round(len(expected.intersection(valid_index)) / len(expected) * 100, 1) if len(expected) else None,
            "recent_1y_expected_rows": int(len(recent_expected)),
            "recent_1y_valid_rows": int(len(recent_valid)),
            "recent_1y_coverage_pct": round(len(recent_valid) / len(recent_expected) * 100, 1) if len(recent_expected) else None,
            "observed_period_expected_rows": int(len(observed_expected)),
            "observed_period_coverage_pct": round(len(observed_expected.intersection(valid_index)) / len(observed_expected) * 100, 1) if len(observed_expected) else None,
            "source_window_years": source_window_years,
            "source_window_expected_rows": source_expected,
            "source_window_valid_rows": source_valid,
            "source_window_coverage_pct": source_pct,
            "missing_rows": int(len(missing)),
            "missing_dates": [str(x.date()) for x in missing[:20]],
            "required_columns": list(columns),
            "age_calendar_days": max(0, int((now.normalize().tz_localize(None) -
                                             pd.Timestamp(latest).normalize()).days))
                                  if latest is not None else None
        }

    market_status = df.attrs.get("data_status", {}) if isinstance(df, pd.DataFrame) else {}
    taiex = df["TAIEX"].dropna() if isinstance(df, pd.DataFrame) and "TAIEX" in df else pd.Series(dtype=float)
    latest_taiex = pd.Timestamp(taiex.index[-1]) if len(taiex) else None
    taiex_index = pd.DatetimeIndex(pd.to_datetime(taiex.index, errors="coerce")).dropna()
    taiex_index = taiex_index.tz_localize(None).normalize().unique().sort_values()
    taiex_missing = expected.difference(taiex_index)
    layers = {
        "TAIEX": {
            "source": market_status.get("taiwan_index_source", "來源未確認"),
            "latest_date": str(latest_taiex.date()) if latest_taiex is not None else None,
            "valid_rows": int(len(taiex)),
            "start_date": str(taiex_index.min().date()) if len(taiex_index) else None,
            "expected_rows": int(len(expected)),
            "coverage_pct": round(len(taiex_index.intersection(expected)) / len(expected) * 100, 1) if len(expected) else None,
            "full_history_coverage_pct": round(len(taiex_index.intersection(expected)) / len(expected) * 100, 1) if len(expected) else None,
            "recent_1y_expected_rows": int(len(expected[expected >= now.normalize().tz_localize(None) - pd.Timedelta(days=365)])),
            "recent_1y_valid_rows": int(len(taiex_index.intersection(expected[expected >= now.normalize().tz_localize(None) - pd.Timedelta(days=365)]))),
            "recent_1y_coverage_pct": round(
                len(taiex_index.intersection(expected[expected >= now.normalize().tz_localize(None) - pd.Timedelta(days=365)])) /
                len(expected[expected >= now.normalize().tz_localize(None) - pd.Timedelta(days=365)]) * 100, 1
            ) if len(expected[expected >= now.normalize().tz_localize(None) - pd.Timedelta(days=365)]) else None,
            "missing_rows": int(len(taiex_missing)),
            "missing_dates": [str(x.date()) for x in taiex_missing[:20]],
            "age_calendar_days": max(0, int((now.normalize().tz_localize(None) -
                                             latest_taiex.normalize()).days))
                                  if latest_taiex is not None else None
        },
        "institutional": coverage(chips, ("foreign_net", "trust_net"),
                                  "TWSE 官方資料（缺漏日期由 FinMind 補入）"),
        "margin": coverage(margin, ("margin_bal", "margin_units", "short_units"),
                           "TWSE 官方資料（缺漏日期由 FinMind 補入）"),
        "futures": coverage(futures, ("foreign_oi_net",),
                            "TAIFEX 官方資料 + FinMind（FinMind 臺指期自 2018-06-05 起）", source_window_years=8)
    }
    payload = {
        "updated_at": now.isoformat(),
        "demo": False,
        "note": "完整期間覆蓋率以指數歷史日期為分母；另列近一年覆蓋率。期貨另列近三年來源期間覆蓋率；實際資料起訖不等於來源保證可用期間。此報表不代表即時報價。",
        "layers": layers
    }
    tmp = "taiwan_data_status.json.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, "taiwan_data_status.json")
    try:
        import storage
        if storage.enabled():
            saved = storage.save_data()
            print(f"台股資料品質狀態已保存至 Supabase（檔案數 {saved}）", flush=True)
    except Exception as exc:
        print(f"台股資料品質狀態 Supabase 保存失敗：{type(exc).__name__}", flush=True)
    summary = "、".join(
        f"{key}:{value['latest_date'] or '無資料'}（{value['valid_rows']} 筆）"
        for key, value in layers.items()
    )
    print(f"台股資料覆蓋狀態：{summary}", flush=True)



def update_news_research_outcomes(series):
    """只在下一個交易日收盤資料出現後，為已保存新聞快照補上實際結果。"""
    path = "news_research_history.jsonl"
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        dates = {str(index.date()): i for i, index in enumerate(series.index)}
        changed = False
        for row in rows:
            if row.get("outcome") is not None:
                continue
            i = dates.get(str(row.get("asof", "")))
            if i is None or i + 1 >= len(series):
                continue
            start, end = float(series.iloc[i]), float(series.iloc[i + 1])
            if not np.isfinite(start) or not np.isfinite(end) or start == 0:
                continue
            row["outcome"] = {
                "next_session_date": str(series.index[i + 1].date()),
                "next_session_return_pct": round((end / start - 1) * 100, 4),
            }
            changed = True
        if changed:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        latest = rows[-1] if rows else None
        context = {"status": "insufficient", "sample_count": 0,
                   "note": "相似新聞環境的歷史樣本不足 20 筆，暫不顯示統計比例。"}
        current_score = (latest or {}).get("features", {}).get("news_environment_score")
        if current_score is not None:
            peers = []
            for row in rows[:-1]:
                features = row.get("features") or {}
                outcome = row.get("outcome") or {}
                score = features.get("news_environment_score")
                value = outcome.get("next_session_return_pct")
                if score is None or value is None:
                    continue
                try:
                    if abs(float(score) - float(current_score)) <= 0.20:
                        peers.append(float(value))
                except (TypeError, ValueError):
                    continue
            context["sample_count"] = len(peers)
            if len(peers) >= 20:
                arr = np.asarray(peers, dtype=float)
                context = {
                    "status": "available", "sample_count": len(peers),
                    "positive_return_pct": round(float(np.mean(arr > 0) * 100), 1),
                    "negative_return_pct": round(float(np.mean(arr < 0) * 100), 1),
                    "average_return_pct": round(float(np.mean(arr)), 3),
                    "median_return_pct": round(float(np.median(arr)), 3),
                    "p10_return_pct": round(float(np.percentile(arr, 10)), 3),
                    "p90_return_pct": round(float(np.percentile(arr, 90)), 3),
                    "score_tolerance": 0.20,
                    "note": "僅為相似新聞環境下的歷史描述統計，不是未來機率或買賣訊號。",
                }
        summary = {
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "latest": latest, "history_count": len(rows),
            "labeled_count": sum(isinstance(row.get("outcome"), dict) for row in rows),
            "historical_context": context,
            "note": "新聞特徵僅作資料研究；歷史結果只在後續交易日收盤資料到齊後標記，不回填未來資訊。",
        }
        with open("news_research.json.tmp", "w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False)
        os.replace("news_research.json.tmp", "news_research.json")
        print(f"新聞研究資料：累積 {len(rows)} 個日期快照，已取得後續收盤結果 {summary['labeled_count']} 筆", flush=True)
    except Exception as exc:
        print(f"新聞研究資料標記暫不可用：{type(exc).__name__}", flush=True)


if __name__ == "__main__":
    demo = "--demo" in sys.argv
    df = load_demo() if demo else load_live()
    if not demo:
        update_news_research_outcomes(df["TAIEX"])
    skip_backfill = "--skip-backfill" in sys.argv
    def read_existing(name):
        if not os.path.exists(name) or os.path.getsize(name) == 0:
            return None
        try:
            frame = pd.read_csv(name, index_col=0, parse_dates=True)
            return frame if len(frame) else None
        except Exception as exc:
            print(f"{name} 無法讀取，該層暫時略過：{type(exc).__name__}: {exc}", flush=True)
            return None

    if skip_backfill and not demo:
        # Fast boot avoids external backfill requests, but can still assess the
        # restored local caches without pretending missing layers are complete.
        chips = read_existing("chips.csv")
        margin = read_existing("margin.csv")
        futures = read_existing("futures.csv")
        print("快速啟動模式：使用已還原的籌碼／信用／期貨快取；不呼叫外部回補來源。", flush=True)
    else:
        chips = demo_chips(df) if demo else (
            read_existing("chips.csv") if "--no-chips" in sys.argv
            else fetch_chips(trading_days=df.index)
        )
        if os.path.exists("margin.csv") and "short_units" not in pd.read_csv("margin.csv", nrows=1):
            print("舊版融資資料缺少融券欄位：保留既有歷史融資資料，融券欄位從本次開始累積。")
        margin = demo_margin(df) if demo else (
            read_existing("margin.csv") if "--no-margin" in sys.argv
            else fetch_margin(trading_days=df.index)
        )
        futures = demo_futures(df) if demo else (
            read_existing("futures.csv") if "--no-futures" in sys.argv
            else fetch_futures(trading_days=df.index)
        )
    if not demo:
        # Generate a quality report on both full and fast runs. Fast runs use
        # restored caches only; absent or incomplete layers remain explicitly missing.
        write_taiwan_data_status(df, chips, margin, futures)
    X, y, fwd = build(df, chips, margin, futures=futures)
    prob, latest, contrib = walk_forward(X, y)
    pu = float(prob.loc[latest])
    layers = contrib.groupby(lambda c: c.split(":")[0]).sum()
    out = {
        "asof": str(latest.date()), "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_basis": "latest_available_taiwan_trading_day", "demo": demo, "prob_up": round(pu, 3),
        "signal": "買進" if pu > BUY else "賣出／減碼" if pu < SELL else "觀望",
        "thresholds": {"buy": BUY, "sell": SELL},
        "missing_layers": [k for k in ("籌碼", "信用", "期貨") if k not in layers.index],
        "data_status": df.attrs.get("data_status", {}),
        "layers": {k: round(float(v), 3) for k, v in layers.items()},
        "features": [{"name": k, "contrib": round(float(v), 3)}
                     for k, v in contrib.reindex(contrib.abs().sort_values(ascending=False).index).items()],
        "backtest": backtest(prob, fwd),
    }
    tmp = "signals.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, "signals.json")
    print(f"{out['asof']}  隔日上漲機率 {pu:.1%}  訊號：{out['signal']}")
    X5, y5, f5 = build(df, chips, margin, h=5, futures=futures)
    p5, l5, c5 = walk_forward(X5, y5, h=5)
    tmp = "weekly_prob.csv.tmp"
    p5.to_csv(tmp, header=["prob"], index_label="date")  # 供個股估計使用
    os.replace(tmp, "weekly_prob.csv")
    wk = weekly_report(p5, f5, l5, c5)
    wk["horizon_history"] = historical_horizons(df["TAIEX"])
    wk["data_status"] = df.attrs.get("data_status", {})
    tmp = "weekly.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(wk, f, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, "weekly.json")
    print(f"{wk['asof']}  一週上漲機率 {wk['prob_up']:.1%}  傾向：{wk['bucket']}")
