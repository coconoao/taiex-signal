"""啟動：python app.py [--demo] [--port 8000] [--host 127.0.0.1] [--no-auto]
自動更新：台股交易日（台北時間）早上 8:30 自動執行 pipeline.py、news.py；啟動時若資料是示範或超過 18 小時，也會補跑一次。
API 金鑰：使用者各自填在自己的瀏覽器（儀表板「設定」頁），不經過、也不儲存在本伺服器。
  伺服器排程若也要跑 AI 分析，可用環境變數提供營運者自己的金鑰（見 news.py）。
對外部署（--host 0.0.0.0 或 READONLY=1）：自動唯讀，關閉手動更新與匯出到本機；排程照常執行。
"""
import datetime as dt, json, os, subprocess, sys, threading, time, urllib.parse
import base64, hmac
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
HOST, PORT, DEMO = arg("--host", "127.0.0.1"), int(arg("--port", 8000)), "--demo" in sys.argv
READONLY = bool(os.environ.get("READONLY")) or HOST not in ("127.0.0.1", "localhost")
AUTO = "--no-auto" not in sys.argv and not DEMO
TZ = ZoneInfo("Asia/Taipei")
JOB, LOCK, QUOTE = {"running": None, "log": ""}, threading.Lock(), {"t": 0, "v": None}
AUTH_USER = os.environ.get("TAIEX_USER", "")
AUTH_PASSWORD = os.environ.get("TAIEX_PASSWORD", "")

def run_jobs(*whats):
    if not LOCK.acquire(blocking=False): return False
    log = ""
    try:
        for w in whats:
            JOB["running"] = w
            cmd = [sys.executable, os.path.join(CODE, w + ".py")] + (["--demo"] if DEMO and w == "pipeline" else [])
            p = subprocess.run(cmd, capture_output=True, text=True)
            log += f"[{w}] " + (p.stdout + p.stderr)[-600:] + "\n"
    finally:
        JOB.update(running=None, log=log[-1500:]); LOCK.release()
    return True

def stale():
    try:
        return json.load(open("signals.json", encoding="utf-8")).get("demo") or time.time() - os.path.getmtime("signals.json") > 18 * 3600
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

class H(SimpleHTTPRequestHandler):
      def authorized(self):
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
    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
              if not self.authorized():
            return
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/status": return self.send_json({**JOB, "readonly": READONLY, "auto": AUTO})
        if u.path in ("/api/quote", "/api/stock"):
            try:
                if u.path == "/api/quote": return self.send_json(quote())
                import stock
                return self.send_json(stock.analyze(urllib.parse.parse_qs(u.query).get("code", [""])[0]))
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)
        if u.path == "/export":
            import export_html
            b = export_html.build().encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{export_html.default_name()}"')
            self.send_header("Content-Length", str(len(b))); self.end_headers(); return self.wfile.write(b)
        name = os.path.basename(u.path)
        if name.endswith(".py") or name == "settings.json": return self.send_error(404)
        if u.path == "/": self.path = "/dashboard.html"
        return super().do_GET()

    def do_POST(self):
              if not self.authorized():
            return
        if READONLY: return self.send_json({"error": "唯讀模式，已停用此功能"}, 403)
        if self.headers.get("X-Requested-With") != "dash": return self.send_json({"error": "拒絕"}, 403)  # 擋跨站請求
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/run" and body.get("what") in ("pipeline", "news"):
            if JOB["running"]: return self.send_json({"error": "已有工作進行中"}, 409)
            threading.Thread(target=run_jobs, args=(body["what"],), daemon=True).start()
            return self.send_json({"ok": True})
        if self.path == "/api/export":
            import export_html
            d = os.path.expanduser("~/Downloads"); os.makedirs(d, exist_ok=True)
            path = os.path.join(d, export_html.default_name())
            open(path, "w", encoding="utf-8").write(export_html.build())
            return self.send_json({"path": path})
        return self.send_json({"error": "未知請求"}, 404)

if __name__ == "__main__":
    if AUTO: threading.Thread(target=scheduler, daemon=True).start()
    print(f"http://{HOST}:{PORT}  {'（唯讀）' if READONLY else ''}{'（自動更新已啟用）' if AUTO else ''}")
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
