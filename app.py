"""啟動：python app.py [--demo] [--port 8000] [--host 127.0.0.1] [--no-auto]
自動更新：台股交易日（台北時間）早上 8:30 自動執行 pipeline.py、news.py；啟動時若資料是示範或超過 18 小時，也會補跑一次。
API 金鑰：使用者各自填在瀏覽器；分析時透過登入後的 /api/ai 即時轉送，不寫入伺服器資料檔或應用程式日誌。
  伺服器排程若也要跑 AI 分析，可用環境變數提供營運者自己的金鑰（見 news.py）。
對外部署（--host 0.0.0.0 或 READONLY=1）：自動唯讀，關閉手動更新與匯出到本機；排程照常執行。
"""
import datetime as dt, json, os, random, subprocess, sys, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import base64, hmac, secrets, urllib.request, urllib.error
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo

CODE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TAIEX_DATA") or CODE  # 以 .app 執行時，資料存在使用者資料夾
os.makedirs(DATA, exist_ok=True)
os.chdir(DATA)
if DATA != CODE:
    import shutil
    for f in ("dashboard.html", "manifest.webmanifest", "icon-192.png", "icon-512.png", "apple-touch-icon.png"):
        shutil.copy(os.path.join(CODE, f), DATA)
    for f in ("signals.json", "weekly.json", "news.json"):  # 首次啟動放入示範資料
        if not os.path.exists(f): shutil.copy(os.path.join(CODE, f), DATA)

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
HOST = arg("--host", os.environ.get("HOST", "127.0.0.1"))
PORT = int(arg("--port", os.environ.get("PORT", "8000")))
DEMO = "--demo" in sys.argv
READONLY = os.environ.get("READONLY", "1" if HOST not in ("127.0.0.1", "localhost") else "0").lower() in ("1", "true", "yes")
# Cloud Run 會水平擴充；排程應交給 Cloud Scheduler / Cloud Run Jobs，不可每個服務執行個體各跑一份。
AUTO = "--no-auto" not in sys.argv and not DEMO and not os.environ.get("K_SERVICE")
TZ = ZoneInfo("Asia/Taipei")
JOB, LOCK, QUOTE = {"running": None, "log": ""}, threading.Lock(), {"t": 0, "v": None}
US_MARKET_CACHE = {"t": 0, "v": None}
AUTH_USER = os.environ.get("TAIEX_USER", "")
AUTH_PASSWORD = os.environ.get("TAIEX_PASSWORD", "")

# Google OAuth：設定 GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET 後啟用。
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get(
    "GOOGLE_REDIRECT_URI",
    "https://taiex-signal.onrender.com/auth/google/callback"
)
GOOGLE_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)
SESSION_SIGNING_KEY = (
    os.environ.get("SESSION_SECRET") or GOOGLE_CLIENT_SECRET
).encode("utf-8")
SESSION_COOKIE = "taiex_session"
STATE_COOKIE = "taiex_oauth_state"
SESSION_TTL = 7 * 24 * 3600

def sign_session(email, expires):
    payload = base64.urlsafe_b64encode(
        f"{email}|{expires}".encode("utf-8")
    ).decode("ascii").rstrip("=")
    signature = hmac.new(SESSION_SIGNING_KEY, payload.encode("ascii"), "sha256").hexdigest()
    return payload + "." + signature

def session_email(token):
    try:
        payload, signature = token.rsplit(".", 1)
        expected = hmac.new(SESSION_SIGNING_KEY, payload.encode("ascii"), "sha256").hexdigest()
        if not hmac.compare_digest(signature, expected):
            return ""
        raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode("utf-8")
        email, expires = raw.rsplit("|", 1)
        email = email.strip().lower()
        return email if email and int(expires) > int(time.time()) else ""
    except Exception:
        return ""

def valid_session(token):
    return bool(session_email(token))

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "").strip().lower()
STOCK_WORKER_URL = os.environ.get("STOCK_WORKER_URL", "").rstrip("/")
STOCK_WORKER_TOKEN = os.environ.get("STOCK_WORKER_TOKEN", "")

class StockServiceUnavailable(RuntimeError):
    """The production stock-analysis service is not configured or reachable."""


class StockWorkerGatewayError(RuntimeError):
    """The configured stock worker failed or returned an invalid response."""


def warm_stock_worker(timeout=90):
    """Wake the free-tier Worker before analysis; /healthz is intentionally public."""
    if not STOCK_WORKER_URL:
        raise StockServiceUnavailable("個股分析服務尚未設定")
    deadline = time.monotonic() + timeout
    last_error = "尚未收到健康檢查回應"
    while time.monotonic() < deadline:
        req = urllib.request.Request(
            STOCK_WORKER_URL + "/healthz",
            headers={"Cache-Control": "no-cache"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=min(15, max(1, deadline - time.monotonic()))) as response:
                health = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and isinstance(health, dict) and health.get("ok") is True:
                return
            last_error = "子服務健康檢查回傳格式錯誤"
        except urllib.error.HTTPError as exc:
            last_error = "HTTP %s" % exc.code
            if exc.code not in (502, 503, 504):
                break
        except Exception as exc:
            last_error = type(exc).__name__
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(3, remaining))
    print("[stock-worker] health check did not become ready: %s" % last_error, flush=True)
    raise StockWorkerGatewayError("個股分析子服務喚醒逾時（%s），請稍後重試" % last_error)


def request_stock_worker(code):
    if not STOCK_WORKER_URL or not STOCK_WORKER_TOKEN:
        raise StockServiceUnavailable("個股分析服務尚未設定")
    # Render free services sleep when idle. Wake the isolated Worker first so
    # the analysis request does not become the cold-start request.
    warm_stock_worker()
    code = str(code or "").strip().upper()
    payload = json.dumps({"code": code}).encode("utf-8")
    req = urllib.request.Request(
        STOCK_WORKER_URL + "/analyze",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + STOCK_WORKER_TOKEN,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print("[stock-worker] analyze endpoint returned HTTP %s" % exc.code, flush=True)
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            message = detail.get("error") or "個股分析服務回傳錯誤"
        except Exception:
            message = "個股分析服務回傳 HTTP %s" % exc.code
        if exc.code in (400, 404, 422):
            raise ValueError(str(message)[:1000])
        raise StockWorkerGatewayError(str(message)[:1000])
    except Exception as exc:
        print("[stock-worker] analyze transport failed: %s" % type(exc).__name__, flush=True)
        raise StockWorkerGatewayError("無法連接個股分析服務：%s" % type(exc).__name__)
    if not isinstance(result, dict):
        raise StockWorkerGatewayError("個股分析服務回傳格式錯誤")
    return result


def analyze_stock(code):
    # Public/read-only deployments must use the isolated official-data Worker.
    # Never silently fall back to the legacy local/Yahoo implementation in production.
    if STOCK_WORKER_URL and STOCK_WORKER_TOKEN:
        return request_stock_worker(code)
    if READONLY:
        raise StockServiceUnavailable("個股分析服務尚未設定")
    import stock
    return stock.analyze(code)


def production_oauth_ready():
    """Public deployments must fail closed unless OAuth and its permission store are configured."""
    return bool(
        GOOGLE_ENABLED and SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and ADMIN_EMAIL
    )

def supabase_request(method, path, payload=None, prefer=None):
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase 尚未設定完整")
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Accept": "application/json"
    }
    # 新版 sb_secret_* 不是 JWT，只能放在 apikey；舊版 service_role JWT 才使用 Bearer。
    if not SUPABASE_SERVICE_ROLE_KEY.startswith("sb_secret_"):
        headers["Authorization"] = "Bearer " + SUPABASE_SERVICE_ROLE_KEY
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(
        SUPABASE_URL + "/rest/v1/" + path,
        data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else []
    except urllib.error.HTTPError as e:
        # 不回傳 request headers，避免任何金鑰意外進入錯誤訊息。
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("Supabase 資料庫請求失敗（HTTP %s）：%s" % (e.code, detail))

def site_user_by_email(email):
    q = urllib.parse.urlencode({
        "email": "eq." + email,
        "select": "email,role,status,created_at,updated_at,approved_by"
    })
    rows = supabase_request("GET", "site_users?" + q)
    return rows[0] if rows else None

def ensure_site_user(email):
    email = email.strip().lower()
    if not email:
        raise RuntimeError("Google 帳號沒有有效 Email")
    if not ADMIN_EMAIL:
        raise RuntimeError("尚未設定 ADMIN_EMAIL")
    if email == ADMIN_EMAIL:
        row = site_user_by_email(email)
        if row:
            if row.get("role") != "admin" or row.get("status") != "approved":
                q = urllib.parse.urlencode({"email": "eq." + email})
                supabase_request("PATCH", "site_users?" + q, {
                    "role": "admin", "status": "approved", "approved_by": email
                }, "return=minimal")
        else:
            supabase_request("POST", "site_users", {
                "email": email, "role": "admin", "status": "approved",
                "approved_by": email
            }, "return=minimal")
        return site_user_by_email(email)
    row = site_user_by_email(email)
    if not row:
        # 忽略併發登入造成的唯一鍵衝突；隨後重新讀取資料列。
        supabase_request("POST", "site_users", {"email": email},
                         "resolution=ignore-duplicates,return=minimal")
    return site_user_by_email(email)

def list_site_users():
    return supabase_request(
        "GET",
        "site_users?select=email,role,status,created_at,updated_at,approved_by&order=created_at.desc"
    )

def set_site_user_status(email, status, actor):
    email = email.strip().lower()
    if not email or email == ADMIN_EMAIL:
        raise ValueError("不能變更設定中的管理員帳號")
    if status not in ("approved", "rejected", "revoked"):
        raise ValueError("不支援的權限狀態")
    q = urllib.parse.urlencode({"email": "eq." + email, "role": "eq.user"})
    rows = supabase_request("PATCH", "site_users?" + q, {
        "status": status, "approved_by": actor
    }, "return=representation")
    if not rows:
        raise ValueError("找不到可管理的使用者")
    return rows[0]

def access_message(status):
    messages = {
        "pending": ("等待管理員核准", "你的 Google 帳號已完成登入，請等待網站管理員核准使用權限。"),
        "rejected": ("目前未獲核准", "此 Google 帳號目前未獲網站使用權限。如有疑問，請聯絡網站管理員。"),
        "revoked": ("使用權限已撤銷", "此 Google 帳號的網站使用權限已被撤銷。")
    }
    title, message = messages.get(status, ("無法使用", "目前無法授權此帳號。"))
    return ("<!doctype html><html lang='zh-Hant'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>%s</title><body style='font:16px sans-serif;max-width:560px;margin:12vh auto;padding:24px;line-height:1.7'>"
            "<h2>%s</h2><p>%s</p><p><a href='/auth/logout'>登出 Google 帳號</a></p></body></html>"
            % (title, title, message))


def run_jobs(*whats):
    if not LOCK.acquire(blocking=False): return False
    log, exit_code = "", 0
    try:
        for w in whats:
            JOB["running"] = w
            script = "pipeline" if w == "pipeline-fast" else w
            args = (["--demo"] if DEMO and script == "pipeline" else
                    ["--skip-backfill"] if w == "pipeline-fast" else [])
            cmd = [sys.executable, "-u", os.path.join(CODE, script + ".py")] + args
            try:
                p = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                def stream_output():
                    nonlocal log
                    for line in p.stdout:
                        entry = f"[{w}] {line}"
                        log = (log + entry)[-3000:]
                        JOB["log"] = log
                        print(entry, end="", flush=True)
                reader = threading.Thread(target=stream_output, daemon=True)
                reader.start()
                try:
                    code = p.wait(timeout=6 * 3600)
                except subprocess.TimeoutExpired:
                    p.kill()
                    code = p.wait()
                    log = (log + f"[{w}] timed out after 6 hours\\n")[-3000:]
                reader.join()
                log = (log + f"[{w}] exit={code}\\n")[-3000:]
                JOB["log"] = log
                print(f"[{w}] exit={code}", flush=True)
            except Exception as e:
                code = 1
                log = (log + f"[{w}] {type(e).__name__}: {e}\\n")[-3000:]
                JOB["log"] = log
                print(log[-500:], flush=True)
            if code == 0 and os.environ.get("TAIEX_STORAGE", "").strip().lower() == "supabase":
                try:
                    import storage
                    # Fast startup reads restored caches but must never write
                    # them back: GitHub Actions is the authoritative market-data writer.
                    save_names = {
                        "pipeline-fast": (),
                        "news": (
                            "news.json", "news_history.jsonl", "policy_events.jsonl",
                            "news_research.json", "news_research_history.jsonl",
                        ),
                        "news_enhanced_model": (
                            "news_enhanced_model.json", "news_research_history.jsonl",
                        ),
                    }.get(w)
                    saved = (
                        storage.save_data(DATA, names=save_names)
                        if save_names is not None else storage.save_data(DATA)
                    )
                    if saved:
                        log += f"[storage] saved {saved} selected data files to Supabase\\n"
                except Exception as e:
                    code = 1
                    log += f"[storage] persistence failed: {type(e).__name__}: {e}\\n"
            if code:
                exit_code = code
                break
    finally:
        JOB.update(running=None, log=log[-3000:], exit_code=exit_code, finished=int(time.time()))
        if log: print(log[-3000:], flush=True)  # 讓 Render 執行紀錄可見 pipeline 的成功／失敗原因
        LOCK.release()
    return exit_code == 0

def stale():
    # 啟動時必須同時確認模型輸出與市場資料存在；Render 檔案系統是暫存的，
    # 不能因為 repo 內附帶的 signals/weekly 範例檔很新，就跳過資料重建。
    required = (
        "signals.json", "weekly.json", "weekly_prob.csv", "news.json",
        "chips.csv", "margin.csv", "futures.csv",
    )
    try:
        if any(not os.path.isfile(name) or os.path.getsize(name) == 0 for name in required):
            return True
        with open("signals.json", encoding="utf-8") as stream:
            signals = json.load(stream)
        if signals.get("demo"):
            return True
        # 新版月級距統計與台灣／政策新聞尚未產生時，啟動後自動補跑一次。
        with open("weekly.json", encoding="utf-8") as stream:
            weekly = json.load(stream)
        if not isinstance(weekly.get("horizon_history"), dict):
            return True
        with open("news.json", encoding="utf-8") as stream:
            news = json.load(stream)
        kinds = {h.get("kind") for h in news.get("headlines", []) if isinstance(h, dict)}
        if not {"taiwan", "policy"}.issubset(kinds):
            return True
        if not isinstance(news.get("policy_events"), list):
            return True
        return any(time.time() - os.path.getmtime(name) > 18 * 3600 for name in required)
    except Exception:
        return True

def scheduler():
    last = None
    # Supabase/GitHub Actions is the canonical writer for market signals and backtests.
    # Do not recompute pipeline-fast on Render startup: its partial cache can produce
    # a different, lower-coverage backtest and overwrite the restored canonical file.
    # Refresh only independent news artifacts; the scheduled GitHub workflow owns
    # the full market pipeline and persists signals.json / weekly.json.
    run_jobs("news", "news_enhanced_model")
    while True:
        now = dt.datetime.now(TZ)
        if now.weekday() < 5 and (now.hour, now.minute) == (8, 30) and last != now.date():
            last = now.date(); run_jobs("pipeline", "news", "news_enhanced_model")
        time.sleep(20)

def _spot_session(now):
    return now.weekday() < 5 and (now.hour, now.minute) >= (8, 45) and (now.hour, now.minute) < (13, 45)

def _afterhours_session(now):
    # 期貨夜盤：交易日 15:00 至次日 05:00（週五夜盤延續至週六清晨）。
    return ((now.weekday() < 5 and (now.hour, now.minute) >= (15, 0)) or
            (0 < now.weekday() < 6 and (now.hour, now.minute) < (5, 0)))

def _futures_quote():
    """Yahoo 台指期近月 WTX& 最新可取得分鐘行情；非期交所官方即時串流，可能延遲。"""
    import pandas as pd
    import yfinance as yf
    raw = yf.download("WTX&", period="5d", interval="1m", prepost=True,
                      auto_adjust=False, progress=False, threads=False, timeout=12)
    if raw is None or raw.empty:
        raise RuntimeError("目前無法取得台指期近月行情")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise RuntimeError("台指期行情沒有有效成交價")
    stamp = close.index[-1]
    if getattr(stamp, "tzinfo", None) is None:
        stamp = stamp.tz_localize(TZ)
    else:
        stamp = stamp.tz_convert(TZ)
    now = dt.datetime.now(TZ)
    age = (now - stamp.to_pydatetime()).total_seconds()
    if age < -60 or age > 20 * 60:
        raise RuntimeError("台指期最新資料已超過 20 分鐘，暫不顯示為即時行情")
    daily = yf.download("WTX&", period="10d", interval="1d",
                        auto_adjust=False, progress=False, threads=False, timeout=12)
    prev = None
    if daily is not None and not daily.empty:
        dc = daily["Close"]
        if isinstance(dc, pd.DataFrame):
            dc = dc.iloc[:, 0]
        dc = dc.dropna()
        if len(dc) >= 2:
            prev = float(dc.iloc[-2])
    last = float(close.iloc[-1])
    if not prev or prev <= 0:
        raise RuntimeError("目前無法取得台指期前一交易日收盤價")
    return {"last": round(last, 2), "change": round(last - prev, 2),
            "pct": round((last / prev - 1) * 100, 2),
            "time": stamp.strftime("%H:%M:%S"), "market": "futures",
            "label": "台指期近月（夜盤）", "source": "Yahoo Finance，可能延遲",
            "delay_note": "非期交所官方即時串流，請以資料時間為準"}

def _twse_daily_taiex_close(day):
    """Fetch one official completed TAIEX close; this is daily data, not an intraday quote."""
    params = urllib.parse.urlencode({
        "date": day.strftime("%Y%m%d"), "type": "ALL", "response": "json"
    })
    url = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?" + params
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; TAIEXDataQuality/1.0)",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.twse.com.tw/"
    })
    with urllib.request.urlopen(req, timeout=8) as response:
        payload = json.loads(response.read().decode("utf-8-sig"))
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
            name = str(record.get("指數") or record.get("指數名稱") or record.get("Index") or "")
            if "發行量加權股價指數" not in name:
                continue
            raw = record.get("收盤指數") or record.get("收盤") or record.get("Close")
            try:
                value = float(str(raw).replace(",", "").strip())
            except (TypeError, ValueError):
                return None
            return value if 1000 <= value <= 100000 else None
    return None


def _twse_last_close_quote():
    """Return the two latest official daily closes, explicitly labelled as non-real-time."""
    now = dt.datetime.now(TZ)
    first_offset = 1 if now.hour < 14 else 0
    closes = []
    for offset in range(first_offset, first_offset + 10):
        day = now.date() - dt.timedelta(days=offset)
        try:
            value = _twse_daily_taiex_close(day)
            if value is not None:
                closes.append((day, value))
                if len(closes) == 2:
                    break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                print(f"TWSE 官方收盤備援暫不可用：HTTP {exc.code}", flush=True)
                break
        except Exception as exc:
            print(f"TWSE 官方收盤備援 {day} 暫不可用：{type(exc).__name__}", flush=True)
            if isinstance(exc, (TimeoutError, OSError, urllib.error.URLError)):
                break
    if not closes:
        return None
    latest_day, latest = closes[0]
    previous = closes[1][1] if len(closes) > 1 else None
    change = latest - previous if previous and previous > 0 else 0.0
    pct = change / previous * 100 if previous and previous > 0 else 0.0
    return {
        "last": round(latest, 2), "change": round(change, 2),
        "pct": round(pct, 2), "time": latest_day.isoformat(),
        "market": "spot_close", "label": "加權指數最近官方收盤",
        "source": "TWSE 官方日收盤資料（非即時）",
        "delay_note": "Yahoo 即時來源不可用或目前非交易時段；此數值為最近已完成交易日收盤"
    }


def _yahoo_last_close_quote():
    """Return recent Yahoo TAIEX daily closes as a labelled, non-real-time fallback."""
    import pandas as pd
    import yfinance as yf

    raw = yf.download("^TWII", period="10d", interval="1d",
                      auto_adjust=False, progress=False, threads=False, timeout=12)
    if raw is None or raw.empty:
        return None
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 2:
        return None
    latest_stamp, previous_stamp = close.index[-1], close.index[-2]
    latest_day = latest_stamp.date()
    now = dt.datetime.now(TZ)
    if latest_day > now.date() or (now.date() - latest_day).days > 10:
        return None
    latest, previous = float(close.iloc[-1]), float(close.iloc[-2])
    if not (1000 <= latest <= 100000 and 1000 <= previous <= 100000):
        return None
    change = latest - previous
    return {
        "last": round(latest, 2), "change": round(change, 2),
        "pct": round(change / previous * 100, 2),
        "time": latest_day.isoformat(), "market": "spot_close",
        "label": "加權指數最近收盤（Yahoo）",
        "source": "Yahoo Finance 日線資料（非即時）",
        "delay_note": "官方 TWSE 收盤來源暫不可用；此數值為最近已完成交易日收盤"
    }


def quote():
    # Intraday Yahoo quotes are attempted first; official TWSE daily close is the
    # safe fallback and is never presented as a live/intraday value.
    if QUOTE["v"] and time.time() - QUOTE["t"] < 20:
        return QUOTE["v"]
    now = dt.datetime.now(TZ)
    if _spot_session(now):
        try:
            import yfinance as yf
            fi = yf.Ticker("^TWII").fast_info
            last, prev = float(fi["last_price"]), float(fi["previous_close"])
            v = {"last": round(last, 2), "change": round(last - prev, 2),
                 "pct": round((last / prev - 1) * 100, 2),
                 "time": now.strftime("%H:%M:%S"), "market": "spot",
                 "label": "加權指數現貨", "source": "Yahoo Finance，可能延遲"}
            QUOTE.update(t=time.time(), v=v)
            return v
        except Exception as exc:
            print(f"Yahoo 加權指數即時報價不可用，改查 TWSE 官方日收盤：{type(exc).__name__}", flush=True)
    if _afterhours_session(now):
        try:
            v = _futures_quote()
            QUOTE.update(t=time.time(), v=v)
            return v
        except Exception as exc:
            print(f"台指期行情不可用，改查 TWSE 官方日收盤：{type(exc).__name__}", flush=True)
    fallback = _twse_last_close_quote()
    if fallback:
        QUOTE.update(t=time.time(), v=fallback)
        return fallback
    try:
        fallback = _yahoo_last_close_quote()
    except Exception as exc:
        print(f"Yahoo 加權指數日收盤備援暫不可用：{type(exc).__name__}", flush=True)
        fallback = None
    if fallback:
        QUOTE.update(t=time.time(), v=fallback)
        return fallback
    raise RuntimeError("Yahoo 即時行情、TWSE 官方日收盤與 Yahoo 日收盤備援目前都無法取得資料")

def us_market_quote():
    """美股最新可取得的 5 分鐘行情；Yahoo 非官方資料，可能延遲或缺漏，快取 60 秒。"""
    if US_MARKET_CACHE["v"] and time.time() - US_MARKET_CACHE["t"] < 60:
        return US_MARKET_CACHE["v"]
    import pandas as pd
    import yfinance as yf

    tickers = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ Composite",
        "^DJI": "Dow Jones",
        "^SOX": "PHLX Semiconductor",
        "TSM": "TSMC ADR",
        "^VIX": "VIX"
    }
    raw = yf.download(
        list(tickers), period="5d", interval="5m", auto_adjust=True,
        progress=False, threads=4, timeout=15, prepost=False
    )
    if raw is None or raw.empty:
        raise RuntimeError("目前無法取得美股盤中資料")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw.xs("Close", axis=1, level=0)
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1)
        else:
            raise RuntimeError("美股行情缺少 Close 欄位")
    else:
        close = raw[["Close"]].rename(columns={"Close": "^GSPC"})
    daily = yf.download(
        list(tickers), period="5d", interval="1d", auto_adjust=True,
        progress=False, threads=4, timeout=15
    )
    if isinstance(daily.columns, pd.MultiIndex):
        if "Close" in daily.columns.get_level_values(0):
            daily_close = daily.xs("Close", axis=1, level=0)
        elif "Close" in daily.columns.get_level_values(1):
            daily_close = daily.xs("Close", axis=1, level=1)
        else:
            daily_close = pd.DataFrame()
    else:
        daily_close = daily[["Close"]].rename(columns={"Close": "^GSPC"}) if not daily.empty else pd.DataFrame()

    rows = []
    chart = None
    for symbol, name in tickers.items():
        if symbol not in close.columns:
            continue
        series = close[symbol].dropna()
        if series.empty:
            continue
        last_time, last = series.index[-1], float(series.iloc[-1])
        # 五日回補只用來找到最近交易時段；圖表只呈現該時段，避免休市時混入多日。
        session = series[series.index.date == pd.Timestamp(last_time).date()]
        previous = None
        if symbol in daily_close.columns:
            d = daily_close[symbol].dropna()
            if len(d):
                intraday_day = pd.Timestamp(last_time).date()
                daily_day = pd.Timestamp(d.index[-1]).date()
                previous = float(d.iloc[-2] if daily_day == intraday_day and len(d) > 1 else d.iloc[-1])
        change = last - previous if previous else None
        pct = (change / previous * 100) if previous else None
        row = {
            "symbol": symbol, "name": name, "last": round(last, 4),
            "previous_close": round(previous, 4) if previous else None,
            "change": round(change, 4) if change is not None else None,
            "pct": round(pct, 3) if pct is not None else None,
            "time": pd.Timestamp(last_time).isoformat(),
            "points": [{"time": pd.Timestamp(i).isoformat(), "value": round(float(v), 4)}
                       for i, v in session.tail(78).items()]
        }
        rows.append(row)
        if symbol == "^GSPC":
            chart = {"name": name, "symbol": symbol, "points": row["points"]}
    result = {
        "source": "Yahoo Finance（yfinance）",
        "interval": "5m",
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note": "顯示目前可取得的最新 5 分鐘資料；可能延遲、休市時會停留在最近一個交易時段，並非交易所保證即時報價。",
        "items": rows,
        "chart": chart
    }
    if not rows:
        raise RuntimeError("美股資料暫時沒有可用報價")
    US_MARKET_CACHE.update(t=time.time(), v=result)
    return result


def trigger_cloud_run_refresh():
    """透過 Cloud Run service identity 啟動持久化資料更新 Job。"""
    job = os.environ.get("CLOUD_RUN_REFRESH_JOB", "taiex-refresh")
    region = os.environ.get("CLOUD_RUN_REGION", "asia-east1")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    headers = {"Metadata-Flavor": "Google"}
    if not project:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            project = response.read().decode("utf-8").strip()
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers=headers
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        access_token = json.loads(response.read().decode("utf-8"))["access_token"]
    status_path = os.path.join(DATA, "refresh_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump({"running": True, "current": "queued", "results": [],
                   "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}, f, ensure_ascii=False)
    url = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/jobs/{job}:run"
    req = urllib.request.Request(
        url, data=b"{}", headers={"Authorization": "Bearer " + access_token, "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            operation = json.loads(response.read().decode("utf-8"))
        return operation.get("name", "started")
    except Exception as e:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"running": False, "current": None, "exit_code": 1,
                       "results": [{"step": "trigger", "exit_code": 1, "log": str(e)}],
                       "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}, f, ensure_ascii=False)
        raise


ADMIN_PAGE = """<!doctype html>
<html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>使用者管理｜TAIEX Signal</title>
<style>
body{font:16px/1.6 system-ui,-apple-system,sans-serif;max-width:900px;margin:32px auto;padding:0 16px;color:#14202b}
a{color:#345b86}.muted{color:#657383}.user{border:1px solid #d9e0e7;border-radius:8px;padding:12px;margin:10px 0;display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
button{padding:6px 12px;margin:3px;border:1px solid #345b86;border-radius:5px;background:#345b86;color:white;cursor:pointer}
button.secondary{background:white;color:#345b86}.actions{display:flex;gap:4px;flex-wrap:wrap}
</style>
<h1>網站使用者管理</h1><p><a href="/">返回網站</a>　<a href="/auth/logout">Google 登出</a></p>
<p class="muted">只有管理員能操作。核准後可使用網站；拒絕或撤銷後會立即失去存取權。</p>
<div id="msg" class="muted">載入中…</div><div id="users"></div>
<script>
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
// Encode apostrophes before inserting email values into single-quoted attributes.
const escAttr=s=>esc(s).replaceAll(String.fromCharCode(39),"&#39;");
async function load(){
 const r=await fetch("/api/admin/users",{cache:"no-store"}),d=await r.json();
 if(!r.ok) throw Error(d.error||"載入失敗");
 const rows=d.users||[];document.getElementById("msg").textContent="使用者數量："+rows.length;
 document.getElementById("users").innerHTML=rows.map(u=>{
  const status={pending:"等待核准",approved:"已核准",rejected:"已拒絕",revoked:"已撤銷"}[u.status]||u.status;
  const isAdmin=u.role==="admin";
  return '<div class="user"><div><b>'+escAttr(u.email)+'</b><div class="muted">'+esc(u.role)+' · '+esc(status)+'</div></div>'+
   (isAdmin?'<span>管理員</span>':'<div class="actions">'+
   '<button data-email="'+escAttr(u.email)+'" data-status="approved">核准</button>'+
   '<button class="secondary" data-email="'+escAttr(u.email)+'" data-status="rejected">拒絕</button>'+
   '<button class="secondary" data-email="'+escAttr(u.email)+'" data-status="revoked">撤銷</button></div>')+'</div>';
 }).join("")||"<p>目前沒有使用者。</p>";
 document.querySelectorAll("button[data-email]").forEach(b=>b.onclick=()=>change(b.dataset.email,b.dataset.status));
}
async function change(email,status){
 if(!confirm("確定將 "+email+" 設為 "+status+"？"))return;
 const r=await fetch("/api/admin/users",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"dash"},body:JSON.stringify({email,status})});
 const d=await r.json();if(!r.ok){alert(d.error||"更新失敗");return;}await load();
}
load().catch(e=>document.getElementById("msg").textContent=e.message);
</script></html>"""


class H(SimpleHTTPRequestHandler):
    def authorized(self):
        self.current_user = None
        if READONLY and not production_oauth_ready():
            self.send_error(503, "正式網站尚未完成 Google OAuth 與 Supabase 權限設定")
            return False
        if GOOGLE_ENABLED:
            cookies = SimpleCookie()
            try:
                cookies.load(self.headers.get("Cookie", ""))
                token = cookies[SESSION_COOKIE].value
            except Exception:
                token = ""
            email = session_email(token)
            if not email:
                self.send_response(302)
                self.send_header("Location", "/auth/google")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False
            try:
                row = ensure_site_user(email)
            except Exception:
                self.send_error(503, "使用者權限資料庫暫時無法使用，請稍後再試")
                return False
            if not row:
                self.send_error(503, "無法讀取使用者權限")
                return False
            self.current_user = row
            # 管理員身分必須同時符合伺服器設定的 ADMIN_EMAIL。
            # 其他帳號只有 role=user 且 status=approved 才能使用網站。
            if email == ADMIN_EMAIL and row.get("role") == "admin" and row.get("status") == "approved":
                return True
            if email != ADMIN_EMAIL and row.get("role") == "user" and row.get("status") == "approved":
                return True
            page = access_message(row.get("status", "pending")).encode("utf-8")
            self.send_response(200 if row.get("status") == "pending" else 403)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return False

        if not AUTH_USER or not AUTH_PASSWORD:
            self.send_error(503, "Authentication not configured")
            return False

        user, password = "", ""
        try:
            header = self.headers.get("Authorization", "")
            scheme, token = header.split(" ", 1)
            if scheme.lower() != "basic":
                raise ValueError()
            decoded = base64.b64decode(token, validate=True).decode("utf-8")
            user, password = decoded.split(":", 1)
        except Exception:
            pass

        valid = (
            hmac.compare_digest(user, AUTH_USER)
            and hmac.compare_digest(password, AUTH_PASSWORD)
        )
        if not valid:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="TAIEX Signal"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False
        return True

    def google_login(self):
        state = secrets.token_urlsafe(32)
        params = urllib.parse.urlencode({
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "prompt": "select_account"
        })
        self.send_response(302)
        self.send_header("Location", "https://accounts.google.com/o/oauth2/v2/auth?" + params)
        self.send_header("Set-Cookie", f"{STATE_COOKIE}={state}; Path=/; Max-Age=600; HttpOnly; Secure; SameSite=Lax")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def google_callback(self, query):
        values = urllib.parse.parse_qs(query)
        state = values.get("state", [""])[0]
        code = values.get("code", [""])[0]
        cookies = SimpleCookie()
        try:
            cookies.load(self.headers.get("Cookie", ""))
            saved_state = cookies[STATE_COOKIE].value
        except Exception:
            saved_state = ""
        if not state or not saved_state or not hmac.compare_digest(state, saved_state) or not code:
            return self.send_error(400, "Google sign-in state validation failed")

        try:
            token_data = urllib.parse.urlencode({
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code"
            }).encode("utf-8")
            request = urllib.request.Request(
                "https://oauth2.googleapis.com/token",
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                tokens = json.loads(response.read().decode("utf-8"))
            access_token = tokens.get("access_token")
            if not access_token:
                raise ValueError("Missing access token")
            request = urllib.request.Request(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": "Bearer " + access_token}
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                profile = json.loads(response.read().decode("utf-8"))
            email = profile.get("email", "")
            if not email or profile.get("email_verified") is not True:
                raise ValueError("Google email is not verified")
            expires = int(time.time()) + SESSION_TTL
            session = sign_session(email, expires)
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie", f"{SESSION_COOKIE}={session}; Path=/; Max-Age={SESSION_TTL}; HttpOnly; Secure; SameSite=Lax")
            self.send_header("Set-Cookie", f"{STATE_COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()
        except Exception:
            self.send_error(502, "Google sign-in failed. Check OAuth configuration and redirect URI.")


    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        try:
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            # 使用者切頁或重新整理時，瀏覽器可能先關閉連線；不應讓正常取消請求產生錯誤堆疊。
            pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/healthz":
            return self.send_json({"ok": True})
        if GOOGLE_ENABLED and u.path == "/auth/google":
            return self.google_login()
        if GOOGLE_ENABLED and u.path == "/auth/google/callback":
            return self.google_callback(u.query)
        if u.path == "/auth/logout":
            self.send_response(302)
            self.send_header("Location", "/auth/google" if GOOGLE_ENABLED else "/")
            self.send_header("Set-Cookie", f"{SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not self.authorized():
            return

        if GOOGLE_ENABLED and u.path == "/api/me":
            return self.send_json({
                "email": self.current_user.get("email"),
                "role": self.current_user.get("role"),
                "status": self.current_user.get("status")
            })
        if GOOGLE_ENABLED and u.path == "/admin":
            if self.current_user.get("role") != "admin":
                return self.send_error(403, "管理員限定")
            page = ADMIN_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            return self.wfile.write(page)
        if GOOGLE_ENABLED and u.path == "/api/admin/users":
            if self.current_user.get("role") != "admin":
                return self.send_json({"error": "管理員限定"}, 403)
            try:
                return self.send_json({"users": list_site_users()})
            except Exception:
                return self.send_json({"error": "無法讀取使用者清單"}, 503)
        if u.path == "/api/taiwan-data-status":
            try:
                status_path = os.path.join(DATA, "taiwan_data_status.json")
                with open(status_path, encoding="utf-8") as stream:
                    status = json.load(stream)
                if status.get("demo"):
                    return self.send_json({"error": "示範資料不列入正式台股資料品質檢查"}, 409)
                return self.send_json(status)
            except FileNotFoundError:
                return self.send_json({"error": "尚未完成第一次台股完整資料品質檢查"}, 404)
            except Exception:
                return self.send_json({"error": "目前無法讀取台股資料品質狀態"}, 503)
        if u.path == "/api/status":
            state = {
                "running": JOB.get("running"),
                "exit_code": JOB.get("exit_code"),
                "finished": JOB.get("finished"),
                "readonly": READONLY,
                "auto": AUTO,
                "cloud_run": bool(os.environ.get("K_SERVICE"))
            }
            if os.environ.get("K_SERVICE"):
                try:
                    saved = json.load(open(os.path.join(DATA, "refresh_status.json"), encoding="utf-8"))
                    state["running"] = saved.get("current") if saved.get("running") else None
                    state["exit_code"] = saved.get("exit_code")
                    state["updated_at"] = saved.get("updated_at")
                    if not saved.get("running"):
                        JOB["running"] = None
                except Exception:
                    pass
            return self.send_json(state)
        if u.path == "/api/us-market":
            try:
                return self.send_json(us_market_quote())
            except Exception as e:
                # Yahoo 暫時失敗時保留最後成功資料，但明確標記為舊快取。
                if US_MARKET_CACHE["v"]:
                    stale = dict(US_MARKET_CACHE["v"])
                    stale["stale"] = True
                    stale["note"] = "即時來源暫時無法更新，以下為伺服器最後一次成功取得的行情；請留意報價時間。"
                    return self.send_json(stale)
                return self.send_json({"error": str(e)}, 503)
        if u.path in ("/api/quote", "/api/stock"):
            if u.path == "/api/quote":
                try:
                    return self.send_json(quote())
                except Exception as e:
                    return self.send_json({"error": str(e)}, 503)
            code = urllib.parse.parse_qs(u.query).get("code", [""])[0]
            try:
                return self.send_json(analyze_stock(code))
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            except StockServiceUnavailable as e:
                return self.send_json({"error": str(e)}, 503)
            except StockWorkerGatewayError as e:
                return self.send_json({"error": str(e)}, 502)
            except Exception:
                return self.send_json({"error": "個股分析暫時無法完成，請稍後重試"}, 500)
        if u.path == "/export":
            import export_html
            b = export_html.build().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{export_html.default_name()}"')
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        if u.path in ("/", "/dashboard.html"):
            # 首頁不可使用條件式快取；避免 iOS/Safari 在重新部署後繼續拿舊 HTML。
            try:
                with open(os.path.join(DATA, "dashboard.html"), "rb") as stream:
                    page = stream.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                try:
                    self.wfile.write(page)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            except FileNotFoundError:
                return self.send_error(404, "dashboard.html not found")
        name = os.path.basename(u.path)
        if name.endswith(".py") or name == "settings.json":
            return self.send_error(404)
        if u.path == "/":
            self.path = "/dashboard.html"
        return super().do_GET()

    def do_POST(self):
        if not self.authorized():
            return
        cloud_job_request = bool(os.environ.get("K_SERVICE")) and self.path == "/api/run"
        admin_request = GOOGLE_ENABLED and self.path == "/api/admin/users"
        # 對外部署仍保護未登入/非 OAuth 模式；已通過 Supabase 核准的 Google 使用者可操作既有功能。
        oauth_approved = bool(
            GOOGLE_ENABLED and self.current_user and
            (self.current_user.get("role") == "admin" or self.current_user.get("status") == "approved")
        )
        if READONLY and not oauth_approved and not cloud_job_request and not admin_request:
            return self.send_json({"error": "唯讀模式，已停用此功能"}, 403)
        if self.headers.get("X-Requested-With") != "dash":
            return self.send_json({"error": "拒絕"}, 403)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 40000:
                return self.send_json({"error": "請求內容過大"}, 413)
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self.send_json({"error": "請求格式錯誤"}, 400)
        if self.path == "/api/admin/users":
            if not GOOGLE_ENABLED or not self.current_user or self.current_user.get("role") != "admin":
                return self.send_json({"error": "管理員限定"}, 403)
            try:
                user = set_site_user_status(
                    body.get("email", ""), body.get("status", ""), self.current_user.get("email", "")
                )
                return self.send_json({"ok": True, "user": user})
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            except Exception:
                return self.send_json({"error": "更新使用者權限失敗"}, 503)

        if self.path == "/api/run" and body.get("what") in ("pipeline", "news", "all"):
            if os.environ.get("K_SERVICE"):
                if JOB.get("running"):
                    return self.send_json({"error": "已有資料更新工作正在執行"}, 409)
                try:
                    saved = json.load(open(os.path.join(DATA, "refresh_status.json"), encoding="utf-8"))
                    if saved.get("running"):
                        return self.send_json({"error": "已有資料更新工作正在執行"}, 409)
                except Exception:
                    pass
                JOB["running"] = "queued"
                try:
                    operation = trigger_cloud_run_refresh()
                    return self.send_json({"ok": True, "operation": operation})
                except Exception as e:
                    JOB["running"] = None
                    return self.send_json({"error": f"無法啟動 Cloud Run 更新工作：{e}"}, 502)
            if JOB["running"]:
                return self.send_json({"error": "已有工作進行中"}, 409)
            whats = ("pipeline", "news") if body["what"] == "all" else (body["what"],)
            threading.Thread(target=run_jobs, args=whats, daemon=True).start()
            return self.send_json({"ok": True})
        if self.path == "/api/export":
            import export_html
            d = os.path.expanduser("~/Downloads")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, export_html.default_name())
            open(path, "w", encoding="utf-8").write(export_html.build())
            return self.send_json({"path": path})
        return self.send_json({"error": "未知請求"}, 404)
if __name__ == "__main__":
    if os.environ.get("TAIEX_STORAGE", "").strip().lower() == "supabase":
        import storage
        restored = storage.restore_data(DATA)
        print(f"Supabase 已還原 {restored} 個資料檔", flush=True)
    if AUTO: threading.Thread(target=scheduler, daemon=True).start()
    print(f"http://{HOST}:{PORT}  {'（唯讀）' if READONLY else ''}{'（自動更新已啟用）' if AUTO else ''}")
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()