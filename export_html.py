"""輸出單一檔案 HTML：內嵌目前資料，可直接雙擊開啟或上傳到任何靜態空間。
用法：python export_html.py [輸出檔名]      （在資料所在資料夾執行；不含任何金鑰）"""
import json, os, sys, datetime as dt

def default_name():
    return f"taiex_signal_{dt.date.today():%Y%m%d}.html"

def build():
    html = open("dashboard.html", encoding="utf-8").read()
    snap = {f: json.load(open(f, encoding="utf-8")) for f in ("signals.json", "weekly.json", "news.json") if os.path.exists(f)}
    blob = json.dumps(snap, ensure_ascii=False).replace("</", "<\\/")  # 避免標題內的 </script> 截斷
    return html.replace("</head>", f"<script>window.__SNAPSHOT__={blob};</script>\n</head>", 1)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else default_name()
    open(out, "w", encoding="utf-8").write(build())
    print("已輸出", out)
