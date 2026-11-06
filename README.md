# 台股訊號儀表板（研究用途，非投資建議）

預測台灣加權指數「隔日」與「未來一週」上漲機率，附個股估計、新聞與機構觀點參考。手機瀏覽器、Mac App、單一 HTML 檔皆可使用。

## 兩種部署方式（手機都能用）
**A. 免伺服器：GitHub Pages + Actions（免費，手機直接開網址）**
1. 在 GitHub 建立一個**公開**倉庫（私人倉庫的 Pages 需付費方案）。
2. 在本資料夾開終端機：
   ```
   git init -b main && git add -A && git commit -m "init"
   git remote add origin https://github.com/<帳號>/<倉庫>.git
   git push -u origin main
   ```
   （`.github` 是隱藏資料夾，網頁拖曳上傳常會漏掉，建議用 git 指令。）
3. 倉庫 Settings → Pages → Build and deployment → Source 選 **GitHub Actions**。
4. 到 Actions 頁，點「update-and-deploy」→ Run workflow。推送後會先發佈示範版網站；手動執行才會抓真實資料。首次回補要數十分鐘到數小時，完成後自動重新發佈。
5. 之後每個台股交易日台北 8:30 左右自動更新（GitHub 排程可能延遲數分鐘到一小時）。
6. 手機 Safari 開 `https://<帳號>.github.io/<倉庫>/`，點「分享」→「加入主畫面」。
- 營運者金鑰（可選）：Settings → Secrets and variables → Actions，加入 `ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GEMINI_API_KEY`（Secrets）與 `OPENAI_MODEL`、`GEMINI_MODEL`（Variables），排程就會順便跑 AI，結果所有人共用。使用者自己的金鑰仍在網頁「設定」頁輸入，只存在自己的瀏覽器。
- 限制：沒有個股與即時報價（需要伺服器）。證交所、期交所有時會擋海外 IP；若抓不到，頁面會顯示「本次沒有取得這些資料來源」，預測仍可用但少幾層訊號，這時請改用 B。公開倉庫 60 天沒有活動時 GitHub 可能停用排程，到 Actions 頁重新啟用即可。

**B. 自己的伺服器（Docker 或 `python app.py --host 0.0.0.0`）：功能完整**
含自動排程、加權指數即時報價、個股分頁。對外時自動唯讀；請用 Tailscale 或反向代理加上密碼，不要直接開放到公網（個股與報價 API 會被濫用）。

## API 金鑰：每位使用者各自輸入
在「設定」分頁填入自己的 Claude／OpenAI／Gemini 金鑰：只存在該裝置的瀏覽器，由瀏覽器直接呼叫各家 API，伺服器不接收也不儲存。預設只存本次分頁，勾選「記住」才存到本機。「內建規則」不需金鑰，永遠可用。
- OpenAI、Gemini 需填模型名稱（我沒有預設，避免猜錯）。
- 營運者也可用環境變數 `ANTHROPIC_API_KEY`、`OPENAI_API_KEY`+`OPENAI_MODEL`、`GEMINI_API_KEY`+`GEMINI_MODEL` 讓排程跑 AI，結果所有人共用。

## 本機與 Mac App
```
pip install -r requirements.txt
python app.py --demo     # 先看示範資料
python app.py            # 真實資料：啟動後自動補抓，之後每個交易日 8:30 自動更新
bash build_mac.sh        # 產生「台股訊號.app」（需 Python 3.10+），資料存 ~/Library/Application Support/TaiexSignal
```
## 輸出成 HTML
`python export_html.py [檔名]`，或「設定」頁的「匯出單一 HTML 檔」。內嵌目前資料與新聞，不含金鑰，可雙擊開啟或上傳到任何靜態空間；快照不會自動更新。

## 資料來源與限制
- Yahoo Finance（行情、即時報價約有數分鐘延遲、財報）、證交所（籌碼、融資券）、期交所（期貨，官方僅近三年）、Google News RSS（標題）。欄位格式若改版，程式會印出警告。
- 新聞與 AI 分數尚未納入機率與回測，每日紀錄於 `news_history.jsonl`。
- 預測是機率，不是保證；回測不代表未來。
