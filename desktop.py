"""Mac 桌面視窗版：python desktop.py（需 pip install pywebview；沒裝時改用瀏覽器開啟）。"""
import os, socket, threading
import app

def pick(port=8765):  # 固定埠號，瀏覽器儲存的金鑰才不會每次啟動都消失
    s = socket.socket()
    try: s.bind(("127.0.0.1", port))
    except OSError: s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close(); return p

port = pick()
threading.Thread(target=app.ThreadingHTTPServer(("127.0.0.1", port), app.H).serve_forever, daemon=True).start()
if app.AUTO: threading.Thread(target=app.scheduler, daemon=True).start()
url = f"http://127.0.0.1:{port}"
try:
    import webview
except ImportError:
    import webbrowser; webbrowser.open(url); threading.Event().wait()
else:
    webview.create_window("台股訊號", url, width=460, height=860, min_size=(360, 560))
    webview.start(private_mode=False, storage_path=os.path.join(app.DATA, "webview"))
