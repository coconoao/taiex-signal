"""台股加權指數訊號 MVP（國際 / 區域 / 本地 三層）
用法:  pip install pandas numpy scikit-learn yfinance lxml
       python pipeline.py --demo      # 合成資料，先看流程
       python pipeline.py             # 抓真實資料（需連網）；首次回補三大法人、融資各約 10 多分鐘，之後增量更新
       python pipeline.py --no-chips --no-margin --no-futures  # 略過籌碼／信用／期貨，只跑其餘層（旗標可單獨使用）
執行時間：台股交易日早上 8:30 前後（美股已收盤、台股未開盤、前一日籌碼已公布）。
輸出 signals.json，供 dashboard.html 讀取。僅供研究，非投資建議。
"""
import os, sys, json, time, datetime as dt, urllib.request, urllib.parse, urllib.error
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TICKERS = {"TAIEX": "^TWII", "SOX": "^SOX", "TSM": "TSM", "VIX": "^VIX", "US10Y": "^TNX",
           "N225": "^N225", "KOSPI": "^KS11", "USDTWD": "TWD=X"}
GLOBAL = ["SOX", "TSM", "VIX", "US10Y"]      # 國際層
REGION = ["N225", "KOSPI", "USDTWD"]         # 區域層（亞洲鄰近市場、匯率）
BUY, SELL, COST = 0.55, 0.45, 0.002          # 買進/賣出門檻、單次換倉成本

def load_live(years=6, cache_path="market_history.csv"):
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
    period = "3mo" if cached is not None else f"{years}y"
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
            if "TAIEX" not in fresh or fresh["TAIEX"].dropna().empty:
                raise ValueError("Yahoo Finance 回傳資料沒有台灣加權指數")
            break
        except Exception as exc:
            last_error = exc
            print(f"行情下載失敗（第 {attempt + 1}/3 次）：{type(exc).__name__}: {exc}", flush=True)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            fresh = None

    if fresh is None and cached is None:
        raise RuntimeError(f"無法取得即時歷史行情，且沒有可用快取。最後錯誤：{last_error}")
    if fresh is not None:
        fresh.index = pd.to_datetime(fresh.index).tz_localize(None)
        fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
        df = fresh.combine_first(cached) if cached is not None else fresh
        df = df[~df.index.duplicated(keep="last")].sort_index()
        tmp = cache_path + ".tmp"
        df.to_csv(tmp, index_label="date")
        os.replace(tmp, cache_path)
        print(f"行情歷史快取已更新：{len(df)} 筆（本次下載 {period}）", flush=True)
        try:
            import storage
            if storage.enabled():
                saved = storage.save_data()
                print(f"行情快取已持久化至 Supabase（檔案數 {saved}）", flush=True)
        except Exception as exc:
            print(f"行情快取 Supabase 保存失敗：{type(exc).__name__}: {exc}", flush=True)
    else:
        df = cached
        print(f"Yahoo Finance 暫不可用，改用既有行情快取：{len(df)} 筆；最後錯誤：{last_error}", flush=True)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["TAIEX"]).ffill(limit=3)
    now = pd.Timestamp.now(tz="Asia/Taipei")
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
BACKFILL_DATES_PER_RUN = 30
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


def fetch_chips(days_back=1460, path="chips.csv", trading_days=None):
    """證交所三大法人買賣超（元），增量寫入 chips.csv（date, foreign_net, trust_net）。"""
    old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    dates = _pending_dates(old, today, days_back, trading_days, required_columns=("foreign_net", "trust_net"))
    rows = {}

    for d in dates:
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
        needed = [d for d in dates if d not in old.index and d not in rows]
        if needed:
            alt = _fetch_finmind_data("TaiwanStockTotalInstitutionalInvestors", min(needed), max(needed))
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


def fetch_futures(days_back=1095, path="futures.csv", trading_days=None):
    """期交所三大法人「臺股期貨」外資未平倉淨口數，增量寫入 futures.csv（date, foreign_oi_net）。官方僅提供近三年。"""
    from io import StringIO
    old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    dates = _pending_dates(old, today, days_back, trading_days, required_columns=("margin_bal", "margin_units", "short_units"))
    rows = {}

    # The official OpenAPI serves the latest published snapshot. Use it only
    # when its own Date exactly matches a pending trading day; never backfill
    # historical dates with a current snapshot.
    try:
        latest_date, latest_net = _fetch_taifex_openapi_latest()
        if latest_date in dates:
            old = _merge_and_save(old, {latest_date: {"foreign_oi_net": latest_net}}, path)
            dates = dates[dates != latest_date]
            print(f"期交所 OpenAPI 已保存最新交易日：{latest_date.date()}", flush=True)
    except Exception as exc:
        print(f"期交所 OpenAPI 備援暫不可用：{type(exc).__name__}: {exc}", flush=True)
    for d in dates:
        time.sleep(0.7)
        body = urllib.parse.urlencode({"queryType": "1", "goDay": "", "doQuery": "1", "dateaddcnt": "",
                                       "queryDate": f"{d:%Y/%m/%d}", "commodityId": ""}).encode()
        try:
            req = urllib.request.Request("https://www.taifex.com.tw/cht/3/futContractsDate", body, {"User-Agent": "Mozilla/5.0"})
            html = _read_with_backoff(req, timeout=20).decode("utf-8", "ignore")
            time.sleep(0.7)  # Respect TAIFEX request rate limits.
            if "臺股期貨" not in html: continue  # 休市日或尚未公布
            net = _tx_foreign_oi_net(pd.read_html(StringIO(html)))
        except urllib.error.HTTPError as e:
            print(f"期交所 {d.date()} HTTP {e.code}：略過本次剩餘回補，保留下次執行", flush=True)
            if e.code == 429 or e.code >= 500:
                break
            time.sleep(1); continue
        except (TimeoutError, OSError, urllib.error.URLError) as e:
            print(f"期交所 {d.date()} 連線持續失敗：{type(e).__name__}；停止本次回補，保留已取得資料", flush=True)
            break
        except Exception as e:
            print(f"期交所 {d.date()} 下載／解析失敗：{type(e).__name__}: {e}", flush=True)
            time.sleep(1); continue
        if net is None:
            print("解析不到臺股期貨外資列，請檢查", d.date())
        else:
            rows[d] = {"foreign_oi_net": net}
            _checkpoint_if_needed(old, rows, path)
    # TAIFEX is primary. Query FinMind once for the missing date range.
    try:
        needed = [d for d in dates if d not in old.index and d not in rows]
        if needed:
            alt = _fetch_finmind_data("TaiwanFuturesInstitutionalInvestors", min(needed), max(needed), data_id="TX")
            required = {"date", "institutional_investors", "long_open_interest_balance_volume",
                        "short_open_interest_balance_volume"}
            if not alt.empty and required.issubset(alt.columns):
                alt["date"] = pd.to_datetime(alt["date"], errors="coerce").dt.normalize()
                for col in ("long_open_interest_balance_volume", "short_open_interest_balance_volume"):
                    alt[col] = pd.to_numeric(alt[col], errors="coerce")
                foreign = alt[alt["institutional_investors"].astype(str).isin(["Foreign_Investor", "外資", "外資及陸資"])]
                foreign = foreign.dropna(subset=["date", "long_open_interest_balance_volume",
                                                 "short_open_interest_balance_volume"])
                foreign = foreign.assign(net=foreign["long_open_interest_balance_volume"] -
                                         foreign["short_open_interest_balance_volume"])
                for date, group in foreign.groupby("date"):
                    if date in needed:
                        rows[date] = {"foreign_oi_net": float(group["net"].sum())}
                print(f"FinMind 期貨備援補入 {sum(d in rows for d in needed)} 個日期", flush=True)
    except Exception as exc:
        print(f"FinMind 期貨備援暫不可用：{type(exc).__name__}", flush=True)
    if not rows and not len(old):
        print("期貨：本次沒有取得資料，將在下次執行重試")
    return _merge_and_save(old, rows, path)

def demo_futures(df, seed=9):  # 合成期貨淨未平倉，僅供示範
    rng = np.random.default_rng(seed)
    r = df["TAIEX"].pct_change().fillna(0)
    return pd.DataFrame({"foreign_oi_net": np.cumsum(r * 2e5 + rng.normal(0, 1500, len(df)))}, index=df.index)

def fetch_margin(days_back=1460, path="margin.csv", trading_days=None):
    """證交所市場融資餘額（元），增量寫入 margin.csv（date, margin_bal）。資料約晚上 9 點後公布。"""
    old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    dates = _pending_dates(old, today, days_back, trading_days, required_columns=("foreign_oi_net",))
    rows = {}
    for d in dates:
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
        needed = [d for d in dates if d not in old.index and d not in rows]
        if needed:
            alt = _fetch_finmind_data("TaiwanStockTotalMarginPurchaseShortSale", min(needed), max(needed))
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
    chips_overlap = len(X.index.intersection(chips.index)) if chips is not None else 0
    if chips is not None and chips_overlap >= 140:
        for k, name in [("foreign_net", "外資"), ("trust_net", "投信")]:
            c = chips[k].reindex(X.index)
            X[f"籌碼:{name}買賣超"] = c / c.abs().rolling(60, min_periods=20).mean()
            s = np.sign(c)
            X[f"籌碼:{name}連買連賣天數"] = s.groupby((s != s.shift()).cumsum()).cumsum().clip(-10, 10)
        first = max(first, chips.index.min())
    elif chips is not None:
        print(f"籌碼資料覆蓋不足（與行情重疊 {chips_overlap} 日，至少 140 日），本次略過此層")
    margin_overlap = len(X.index.intersection(margin.index)) if margin is not None else 0
    if margin is not None and margin_overlap >= 140:
        b = margin["margin_bal"].reindex(X.index)
        X["信用:融資餘額5日變化"] = b.pct_change(5)
        X["信用:融資餘額偏離60日均"] = b / b.rolling(60, min_periods=20).mean() - 1
        X["信用:融資與指數背離"] = b.pct_change(5) - t.pct_change(5)
        if {"short_units", "margin_units"}.issubset(margin.columns):
            sh = margin["short_units"].reindex(X.index)
            ratio = sh / margin["margin_units"].reindex(X.index)
            X["信用:融券餘額5日變化"] = sh.pct_change(5)
            X["信用:券資比偏離60日均"] = ratio / ratio.rolling(60, min_periods=20).mean() - 1
        first = max(first, margin.index.min())
    elif margin is not None:
        print(f"信用資料覆蓋不足（與行情重疊 {margin_overlap} 日，至少 140 日），本次略過此層")
    futures_overlap = len(X.index.intersection(futures.index)) if futures is not None else 0
    if futures is not None and futures_overlap >= 120:
        o = futures["foreign_oi_net"].reindex(X.index)
        sd = o.rolling(120, min_periods=40).std()
        X["期貨:外資淨未平倉(相對近120日)"] = (o - o.rolling(120, min_periods=40).mean()) / sd
        X["期貨:外資淨未平倉5日變化"] = o.diff(5) / sd
        first = max(first, futures.index.min())
    elif futures is not None:
        print(f"期貨資料覆蓋不足（與行情重疊 {futures_overlap} 日，至少 120 日），本次略過此層")
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
        values = (series.shift(-days) / series - 1).iloc[::days].dropna()
        if values.empty:
            out[key] = {"label": label, "days": days, "n": 0}
            continue
        out[key] = {
            "label": label, "days": days, "n": int(len(values)),
            "positive_rate": round(float((values > 0).mean()) * 100, 1),
            "mean": round(float(values.mean()) * 100, 2),
            "median": round(float(values.median()) * 100, 2),
            "p10": round(float(values.quantile(.10)) * 100, 2),
            "p90": round(float(values.quantile(.90)) * 100, 2),
            "min": round(float(values.min()) * 100, 2),
            "max": round(float(values.max()) * 100, 2),
        }
    return out


if __name__ == "__main__":
    demo = "--demo" in sys.argv
    df = load_demo() if demo else load_live()
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

    chips = demo_chips(df) if demo else (
        read_existing("chips.csv") if skip_backfill or "--no-chips" in sys.argv
        else fetch_chips(trading_days=df.index)
    )
    if os.path.exists("margin.csv") and "short_units" not in pd.read_csv("margin.csv", nrows=1):
        print("舊版融資資料缺少融券欄位：保留既有歷史融資資料，融券欄位從本次開始累積。")
    margin = demo_margin(df) if demo else (
        read_existing("margin.csv") if skip_backfill or "--no-margin" in sys.argv
        else fetch_margin(trading_days=df.index)
    )
    futures = demo_futures(df) if demo else (
        read_existing("futures.csv") if skip_backfill or "--no-futures" in sys.argv
        else fetch_futures(trading_days=df.index)
    )
    if skip_backfill:
        print("快速啟動模式：使用已保存的真實資料計算；缺少資料層會明確標示，歷史回補留待排程執行。", flush=True)
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
    tmp = "weekly.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(wk, f, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, "weekly.json")
    print(f"{wk['asof']}  一週上漲機率 {wk['prob_up']:.1%}  傾向：{wk['bucket']}")
