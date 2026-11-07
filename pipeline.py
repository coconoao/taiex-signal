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

def load_live(years=6):
    import yfinance as yf
    last_error = None
    df = None
    for attempt in range(3):
        try:
            raw = yf.download(list(TICKERS.values()), period=f"{years}y",
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
            df = close.rename(columns={v: k for k, v in TICKERS.items()})
            if "TAIEX" not in df or df["TAIEX"].dropna().empty:
                raise ValueError("Yahoo Finance 回傳資料沒有台灣加權指數")
            break
        except Exception as e:
            last_error = e
            print(f"行情下載失敗（第 {attempt + 1}/3 次）：{type(e).__name__}: {e}", flush=True)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            df = None
    if df is None:
        raise RuntimeError(f"無法取得即時歷史行情；可能是 Yahoo Finance 限流（HTTP 429）。最後錯誤：{last_error}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["TAIEX"]).ffill(limit=3)
    now = pd.Timestamp.now(tz="Asia/Taipei")
    if now.hour < 14:  # 盤中或開盤前：排除今日尚未收盤的資料列
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
BACKFILL_DATES_PER_RUN = 10
RECENT_DATES_PER_RUN = 5
CHECKPOINT_EVERY = 5


def _pending_dates(old, today, days_back, trading_days=None):
    start = today - pd.Timedelta(days=days_back)
    if trading_days is None:
        dates = pd.bdate_range(start, today)
    else:
        dates = pd.DatetimeIndex(trading_days).tz_localize(None).normalize().unique().sort_values()
        dates = dates[(dates >= start) & (dates <= today)]
    if len(old):
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

def fetch_chips(days_back=1460, path="chips.csv", trading_days=None):
    """證交所三大法人買賣超（元），增量寫入 chips.csv（date, foreign_net, trust_net）。"""
    old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    dates = _pending_dates(old, today, days_back, trading_days)
    rows = {}

    for d in dates:
        url = f"https://www.twse.com.tw/rwd/zh/fund/BFI82U?type=day&dayDate={d:%Y%m%d}&response=json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.loads(_read_with_backoff(req, timeout=15).decode("utf-8"))
        except urllib.error.HTTPError as e:
            print(f"三大法人 {d.date()} HTTP {e.code}：略過本次剩餘回補，保留下次執行", flush=True)
            if e.code == 429: break
            time.sleep(1); continue
        except Exception as e:
            print(f"三大法人 {d.date()} 下載失敗：{type(e).__name__}: {e}", flush=True)
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
    dates = _pending_dates(old, today, days_back, trading_days)
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
            if e.code == 429: break
            time.sleep(1); continue
        except Exception as e:
            print(f"期交所 {d.date()} 下載／解析失敗：{type(e).__name__}: {e}", flush=True)
            time.sleep(1); continue
        if net is None:
            print("解析不到臺股期貨外資列，請檢查", d.date())
        else:
            rows[d] = {"foreign_oi_net": net}
            _checkpoint_if_needed(old, rows, path)
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
    dates = _pending_dates(old, today, days_back, trading_days)
    rows = {}
    for d in dates:
        url = f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d:%Y%m%d}&selectType=MS&response=json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.loads(_read_with_backoff(req, timeout=15).decode("utf-8"))
        except urllib.error.HTTPError as e:
            print(f"融資融券 {d.date()} HTTP {e.code}：略過本次剩餘回補，保留下次執行", flush=True)
            if e.code == 429: break
            time.sleep(1); continue
        except Exception as e:
            print(f"融資融券 {d.date()} 下載失敗：{type(e).__name__}: {e}", flush=True)
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
    if chips is not None and len(chips) > 60:
        for k, name in [("foreign_net", "外資"), ("trust_net", "投信")]:
            c = chips[k].reindex(X.index)
            X[f"籌碼:{name}買賣超"] = c / c.abs().rolling(60, min_periods=20).mean()  # 以近 60 日平均規模標準化
            s = np.sign(c)
            X[f"籌碼:{name}連買連賣天數"] = s.groupby((s != s.shift()).cumsum()).cumsum().clip(-10, 10)
        first = chips.index.min()  # 只用有籌碼資料的期間
    elif chips is not None:
        print("籌碼資料不足，本次略過籌碼層")
    if margin is not None and len(margin) > 60:
        b = margin["margin_bal"].reindex(X.index)
        X["信用:融資餘額5日變化"] = b.pct_change(5)
        X["信用:融資餘額偏離60日均"] = b / b.rolling(60, min_periods=20).mean() - 1
        X["信用:融資與指數背離"] = b.pct_change(5) - t.pct_change(5)  # 指數跌、融資卻增＝散戶逆勢加碼
        if "short_units" in margin:
            sh = margin["short_units"].reindex(X.index)
            ratio = sh / margin["margin_units"].reindex(X.index)  # 券資比
            X["信用:融券餘額5日變化"] = sh.pct_change(5)
            X["信用:券資比偏離60日均"] = ratio / ratio.rolling(60, min_periods=20).mean() - 1
        first = max(first, margin.index.min())
    elif margin is not None:
        print("融資資料不足，本次略過信用層")
    if futures is not None and len(futures) > 60:
        o = futures["foreign_oi_net"].reindex(X.index)
        sd = o.rolling(120, min_periods=40).std()
        X["期貨:外資淨未平倉(相對近120日)"] = (o - o.rolling(120, min_periods=40).mean()) / sd
        X["期貨:外資淨未平倉5日變化"] = o.diff(5) / sd
        first = max(first, futures.index.min())
    elif futures is not None:
        print("期貨資料不足，本次略過期貨層")
    # 時序：美股 D 日收盤早於台股 D+1 開盤，故 D 列特徵預測 D+1 報酬，無資料洩漏
    fwd = t.shift(-h) / t - 1  # 未來 h 個交易日報酬（h=1 即隔日）
    y = (fwd > 0).astype(float).where(fwd.notna())
    return X.loc[first:], y.loc[first:], fwd.loc[first:]

def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))

def walk_forward(X, y, start=500, step=20, h=1):
    valid = X.notna().all(axis=1)
    idx = X.index[valid & y.notna()]
    if len(idx) < max(80, h + 20):
        raise ValueError(f"有效訓練樣本不足：只有 {len(idx)} 筆；至少需要 {max(80, h + 20)} 筆。")
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
        "brier": round(float(((p - (f > 0).astype(float)) ** 2).mean()), 4),
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

if __name__ == "__main__":
    demo = "--demo" in sys.argv
    df = load_demo() if demo else load_live()
    chips = demo_chips(df) if demo else None if "--no-chips" in sys.argv else fetch_chips(trading_days=df.index)
    if os.path.exists("margin.csv") and "short_units" not in pd.read_csv("margin.csv", nrows=1):
        print("舊版融資資料缺少融券欄位：保留既有歷史融資資料，融券欄位從本次開始累積。")
    margin = demo_margin(df) if demo else None if "--no-margin" in sys.argv else fetch_margin(trading_days=df.index)
    futures = demo_futures(df) if demo else None if "--no-futures" in sys.argv else fetch_futures(trading_days=df.index)
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
    tmp = "weekly.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(wk, f, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, "weekly.json")
    print(f"{wk['asof']}  一週上漲機率 {wk['prob_up']:.1%}  傾向：{wk['bucket']}")
