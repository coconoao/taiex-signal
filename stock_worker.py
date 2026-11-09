"""Dedicated stock-analysis worker service.

Public health endpoint; analysis endpoint requires a shared bearer token.
The worker runs stock research independently from the main dashboard service.
"""
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tw_market_data import MarketDataError

TOKEN = os.environ.get("STOCK_WORKER_TOKEN", "")
MAX_BODY = 4096
DIAGNOSTICS_UI_ENABLED = os.environ.get("STAGING_DIAGNOSTICS_UI", "").lower() == "true"
_SAFE_DIAGNOSTIC = re.compile(
    r"(?:no_recent_history|provider_invalid_json_shape|provider_error|"
    r"(?:twse|tpex)_(?:http_\d{3}|network_errno_-?\d{1,4}|timeout|invalid_json|request_error|provider_status|provider_invalid_json_shape|provider_error)|"
    r"(?:exchange_probe_failed|history_fetch_failed):"
    r"(?:(?:twse|tpex)_(?:http_\d{3}|network_errno_-?\d{1,4}|timeout|invalid_json|request_error|provider_status|provider_invalid_json_shape|provider_error)|provider_invalid_json_shape|provider_error)"
    r"(?:,(?:(?:twse|tpex)_(?:http_\d{3}|network_errno_-?\d{1,4}|timeout|invalid_json|request_error|provider_status|provider_invalid_json_shape|provider_error)|provider_invalid_json_shape|provider_error)){0,11})"
)


def analyze_code(code):
    """Return a safe payload/status pair; upstream failures are not client errors."""
    try:
        import stock
        return stock.analyze(code), 200
    except MarketDataError as exc:
        # Prefer structured, allowlisted diagnostic categories. Fall back to
        # parsing legacy messages only when no structured category is present.
        diagnostic = getattr(exc, "diagnostic_code", None)
        if isinstance(diagnostic, str) and _SAFE_DIAGNOSTIC.fullmatch(diagnostic):
            detail = diagnostic
        else:
            match = re.search(
                r"(?:www\.)?(?:twse\.com\.tw|tpex\.org\.tw) "
                r"[A-Za-z]+(?:/[A-Za-z]+)?(?:\(errno=\d+\))?(?:\(http=\d+\))?"
                r"(?:\(redirect=(?:www\.)?(?:twse\.com\.tw|tpex\.org\.tw)/[A-Za-z0-9_./-]*\))?",
                str(exc),
            )
            detail = match.group(0) if match else "provider_or_data_validation_error"
        print("[stock-worker] official_market_data_unavailable: " + detail, flush=True)
        return {
            "error": "官方行情來源暫時無法連線或資料不足，請稍後重試"
        }, 503
    except ValueError as exc:
        return {"error": str(exc)[:1000]}, 422
    except Exception as exc:
        # Log only the exception class; never log tokens, headers, or stock payloads.
        print(
            "[stock-worker] analysis_failed: " + type(exc).__name__,
            flush=True,
        )
        return {"error": "個股分析暫時無法完成，請稍後重試"}, 500


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload, status=200):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/diagnostics" and DIAGNOSTICS_UI_ENABLED:
            page = """<!doctype html>
<html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>Worker staging test</title>
<style>body{font:16px system-ui;max-width:640px;margin:32px auto;padding:0 16px}input,button{box-sizing:border-box;width:100%;padding:12px;margin:8px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:12px}</style>
<h2>Staging 個股資料連線測試</h2>
<p>Token 僅在本頁記憶體中使用，不會儲存或顯示在網址。</p>
<label>Staging Bearer Token<input id="token" type="password" autocomplete="off" autocapitalize="off"></label>
<label>股票代號<input id="code" value="2330" inputmode="numeric"></label>
<button id="run">開始測試</button><pre id="result">尚未測試</pre>
<script>
document.getElementById("run").addEventListener("click",async()=>{
 const token=document.getElementById("token").value;
 const code=document.getElementById("code").value.trim();
 const out=document.getElementById("result");
 if(!token){out.textContent="請輸入 staging token";return}
 out.textContent="測試中，請稍候…";
 try{
  const response=await fetch("/analyze",{method:"POST",headers:{"Content-Type":"application/json","Authorization":"Bearer "+token},body:JSON.stringify({code})});
  const body=await response.json();
  out.textContent="HTTP "+response.status+"\\n"+JSON.stringify(body,null,2);
 }catch(e){out.textContent="連線失敗："+e.name}
});
</script></html>"""
            raw = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path == "/healthz":
            return self._json({"ok": True, "service": "stock-worker"})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/analyze":
            return self._json({"error": "not found"}, 404)
        supplied = self.headers.get("Authorization", "")
        if not TOKEN or supplied != "Bearer " + TOKEN:
            return self._json({"error": "unauthorized"}, 401)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                return self._json({"error": "invalid request size"}, 413)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            code = str(body.get("code", "")).strip().upper()
            if not code or len(code) > 12 or not code[0].isdigit() or not code.isalnum():
                return self._json({"error": "請輸入有效的台股代號"}, 400)
        except Exception:
            return self._json({"error": "請求格式錯誤"}, 400)

        os.chdir("/tmp")
        result, status = analyze_code(code)
        return self._json(result, status)

    def log_message(self, fmt, *args):
        # Avoid logging request payloads and user-supplied symbols.
        return


if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("STOCK_WORKER_TOKEN is required")
    port = int(os.environ.get("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
