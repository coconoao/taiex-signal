"""Dedicated stock-analysis worker service.

Public health endpoint; analysis endpoint requires a shared bearer token.
The worker runs stock research independently from the main dashboard service.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = os.environ.get("STOCK_WORKER_TOKEN", "")
MAX_BODY = 4096


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

        try:
            import stock
            result = stock.analyze(code)
            return self._json(result)
        except Exception as exc:
            # Do not log secrets or request headers.
            return self._json({"error": str(exc)[:1000]}, 400)

    def log_message(self, fmt, *args):
        # Avoid logging request payloads and user-supplied symbols.
        return


if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("STOCK_WORKER_TOKEN is required")
    port = int(os.environ.get("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
