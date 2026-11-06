"""啟動：python app.py [--demo] [--port 8000] [--host 127.0.0.1] [--no-auto]
自動更新：台股交易日（台北時間）早上 8:30 自動執行 pipeline.py、news.py；啟動時若資料是示範或超過 18 小時，也會補跑一次。
API 金鑰：使用者各自填在瀏覽器；分析時透過登入後的 /api/ai 即時轉送，不寫入伺服器資料檔或應用程式日誌。
  伺服器排程若也要跑 AI 分析，可用環境變數提供營運者自己的金鑰（見 news.py）。
對外部署（--host 0.0.0.0 或 READONLY=1）：自動唯讀，關閉手動更新與匯出到本機；排程照常執行。
"""
import datetime as dt, json, os, subprocess, sys, threading, time, urllib.parse
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
            cmd = [sys.executable, os.path.join(CODE, w + ".py")] + (["--demo"] if DEMO and w == "pipeline" else [])
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
                code = p.returncode
                log += f"[{w}] exit={code} " + (p.stdout + p.stderr)[-1200:] + "\n"
            except Exception as e:
                code = 1
                log += f"[{w}] {type(e).__name__}: {e}\n"
            if code:
                exit_code = code
                break
    finally:
        JOB.update(running=None, log=log[-3000:], exit_code=exit_code, finished=int(time.time()))
        if log: print(log[-3000:], flush=True)  # 讓 Render 執行紀錄可見 pipeline 的成功／失敗原因
        LOCK.release()
    return exit_code == 0

def stale():
    # 個股分析依賴 weekly.json 與 weekly_prob.csv；不能只看 signals.json。
    required = ("signals.json", "weekly.json", "weekly_prob.csv")
    try:
        if any(not os.path.exists(name) or os.path.getsize(name) == 0 for name in required):
            return True
        signals = json.load(open("signals.json", encoding="utf-8"))
        if signals.get("demo"):
            return True
        return any(time.time() - os.path.getmtime(name) > 18 * 3600 for name in required)
    except Exception:
        return True

def scheduler():
    last = None
    if stale(): run_jobs("pipeline", "news")
    while True:
        now = dt.datetime.now(TZ)
        if now.weekday() < 5 and (now.hour, now.minute) == (8, 30) and last != now.date():
            last = now.date(); run_jobs("pipeline", "news")
        time.sleep(20)

def quote():  # 加權指數即時報價（Yahoo，約有數分鐘延遲），快取 20 秒
    if QUOTE["v"] and time.time() - QUOTE["t"] < 20: return QUOTE["v"]
    import yfinance as yf
    fi = yf.Ticker("^TWII").fast_info
    last, prev = float(fi["last_price"]), float(fi["previous_close"])
    v = {"last": round(last, 2), "change": round(last - prev, 2), "pct": round((last / prev - 1) * 100, 2),
         "time": dt.datetime.now(TZ).strftime("%H:%M:%S")}
    QUOTE.update(t=time.time(), v=v)
    return v

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
        list(tickers), period="1d", interval="5m", auto_adjust=True,
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
                       for i, v in series.tail(78).items()]
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


def call_ai_provider(provider, api_key, model, prompt):
    """將瀏覽器提供的金鑰即時轉送至供應商；不寫入檔案、環境變數或日誌。"""
    if not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 4096:
        raise ValueError("API 金鑰格式不正確")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 30000:
        raise ValueError("分析內容為空或超過長度限制")
    if provider == "Claude":
        body = {"model": model or "claude-sonnet-4-5", "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        url = "https://api.anthropic.com/v1/messages"
    elif provider == "ChatGPT":
        body = {"model": model or "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}}
        headers = {"Authorization": "Bearer " + api_key}
        url = "https://api.openai.com/v1/chat/completions"
    elif provider == "Gemini":
        model = model or "gemini-2.5-flash"
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"}}
        headers = {"x-goog-api-key": api_key}
        url = "https://generativelanguage.googleapis.com/v1beta/models/" + urllib.parse.quote(model, safe="") + ":generateContent"
    else:
        raise ValueError("不支援的 AI 服務")
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:1200]
        try:
            detail = json.loads(detail).get("error", {}).get("message", detail)
        except Exception:
            pass
        raise RuntimeError(f"{provider} API 回應 {e.code}: {detail}")
    if provider == "Claude":
        return "".join(x.get("text", "") for x in result.get("content", []) if x.get("type") == "text")
    if provider == "ChatGPT":
        choices = result.get("choices") or []
        return choices[0].get("message", {}).get("content", "") if choices else ""
    candidates = result.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini 沒有回傳可用內容；請檢查模型名稱、金鑰與配額")
    return "".join(x.get("text", "") for x in candidates[0].get("content", {}).get("parts", []))



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
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

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
        if u.path == "/api/status":
            state = {**JOB, "readonly": READONLY, "auto": AUTO, "cloud_run": bool(os.environ.get("K_SERVICE"))}
            if os.environ.get("K_SERVICE"):
                try:
                    saved = json.load(open(os.path.join(DATA, "refresh_status.json"), encoding="utf-8"))
                    state["running"] = saved.get("current") if saved.get("running") else None
                    if not saved.get("running"):
                        JOB["running"] = None
                    state["log"] = "\\n".join(
                        f'{x.get("step")}: exit={x.get("exit_code")} {x.get("log", "")}'
                        for x in saved.get("results", [])
                    )[-3000:]
                    state["exit_code"] = saved.get("exit_code")
                except Exception:
                    pass
            return self.send_json(state)
        if u.path == "/api/us-market":
            try:
                return self.send_json(us_market_quote())
            except Exception as e:
                return self.send_json({"error": str(e)}, 503)
        if u.path in ("/api/quote", "/api/stock"):
            try:
                if u.path == "/api/quote":
                    return self.send_json(quote())
                import stock
                return self.send_json(stock.analyze(
                    urllib.parse.parse_qs(u.query).get("code", [""])[0]
                ))
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)
        if u.path == "/export":
            import export_html
            b = export_html.build().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{export_html.default_name()}"')
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
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
        if READONLY and not oauth_approved and self.path != "/api/ai" and not cloud_job_request and not admin_request:
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
        if self.path == "/api/ai":
            try:
                result = call_ai_provider(
                    body.get("provider"), body.get("key"), body.get("model", ""), body.get("prompt", "")
                )
                return self.send_json({"text": result})
            except Exception as e:
                return self.send_json({"error": str(e)}, 502)
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
    if AUTO: threading.Thread(target=scheduler, daemon=True).start()
    print(f"http://{HOST}:{PORT}  {'（唯讀）' if READONLY else ''}{'（自動更新已啟用）' if AUTO else ''}")
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
