# Google Cloud Run 部署（台灣區域）

本專案可部署到 Cloud Run。建議使用 Cloud Run 服務處理網站請求，Cloud Storage 保存資料檔，Cloud Run Job 執行耗時的行情／新聞更新。

## 架構與重要設定

- Cloud Run region：`asia-east1`（台灣）。
- Cloud Run service：網站與 Google OAuth；預設 `READONLY=1`，避免 HTTP 請求結束後背景執行緒被暫停。
- Cloud Storage bucket：持久保存 `signals.json`、`weekly.json`、`weekly_prob.csv`、`news.json`、CSV 歷史資料。
- Cloud Run Job：執行 `python /app/refresh.py`，依序更新行情與新聞。
- OAuth：Cloud Run URL 必須加入 Google OAuth 用戶端的「已授權 JavaScript 來源」及「已授權重新導向 URI」。

Cloud Run 免費額度依每月用量計算；Cloud Storage、網路流量、建置與超出免費額度的 Cloud Run 用量可能收費。部署前請先設定預算通知，並確認 Google Cloud Billing 設定。不要設定最小執行個體數為 1。

## 1. 建立資料 bucket

在 Google Cloud Console 的 Cloud Storage 建立 bucket，位置選 `asia-east1`。Bucket 名稱必須全域唯一。記下名稱，以下以 `YOUR_BUCKET` 表示。

把 Cloud Run service 與 Cloud Run Job 使用的服務帳戶授予該 bucket 的 **Storage Object User** 權限。另將 Cloud Run service 的服務帳戶授予 Secret Manager Secret 的 **Secret Manager Secret Accessor** 權限，以及更新工作 Job 的 **Cloud Run Invoker（roles/run.invoker）** 權限。不要把 bucket 設為公開。

## 2. 建立 Google OAuth Secret

在 Secret Manager 建立名為 `google-client-secret` 的 Secret，將 Google OAuth Client Secret 放入 Secret Manager。不要把 Secret 寫入 GitHub、程式碼或聊天。

Google OAuth Client ID 可作為 Cloud Run 環境變數；Client Secret 使用 Secret Manager 參照。

## 3. 部署 Cloud Run service

在專案根目錄執行（需先安裝並登入 Google Cloud CLI，選好正確的 Google Cloud project）：

```bash
gcloud run deploy taiex-signal \
  --source . \
  --region asia-east1 \
  --allow-unauthenticated \
  --max 1 \
  --min 0 \
  --memory 1Gi \
  --cpu 1 \
  --port 8080 \
  --timeout 3600 \
  --set-env-vars TAIEX_DATA=/mnt/data,READONLY=1,CLOUD_RUN_REFRESH_JOB=taiex-refresh,CLOUD_RUN_REGION=asia-east1 \
  --set-secrets GOOGLE_CLIENT_SECRET=google-client-secret:latest \
  --add-volume mount-path=/mnt/data,type=cloud-storage,bucket=YOUR_BUCKET
```

`--allow-unauthenticated` 只代表 Cloud Run 網路入口可到達網站；應用程式本身仍會要求 Google OAuth 登入。不要移除 `GOOGLE_CLIENT_SECRET` 的 Secret Manager 參照，也不要把 `READONLY` 改成 0。

部署完成後，複製 Cloud Run 顯示的服務網址，例如 `https://taiex-signal-xxxxx-uc.a.run.app`。

在 Cloud Run 的 Variables & Secrets 加入 `GOOGLE_CLIENT_ID`，並設定：

```
GOOGLE_REDIRECT_URI=https://YOUR_CLOUD_RUN_URL/auth/google/callback
```

接著到 Google Cloud Console 的 OAuth 用戶端，新增：
- Authorized JavaScript origin：`https://YOUR_CLOUD_RUN_URL`
- Authorized redirect URI：`https://YOUR_CLOUD_RUN_URL/auth/google/callback`

保留原本 Render 的來源與 Redirect URI，直到 Cloud Run 登入、資料更新與個股分析都測試成功。

## 4. 建立資料更新 Job

先取得 Cloud Run service 使用的映像檔：

```bash
IMAGE=$(gcloud run services describe taiex-signal --region asia-east1 --format='value(spec.template.spec.containers[0].image)')
```

建立更新工作：

```bash
gcloud run jobs create taiex-refresh \
  --image "$IMAGE" \
  --region asia-east1 \
  --command python \
  --args /app/refresh.py \
  --task-timeout 6h \
  --max-retries 0 \
  --set-env-vars TAIEX_DATA=/mnt/data \
  --add-volume mount-path=/mnt/data,type=cloud-storage,bucket=YOUR_BUCKET
```

確認 Job 使用的服務帳戶也有該 bucket 的 Storage Object User 權限。並在該 Job 的 IAM 權限中，將 Cloud Run service 的服務帳戶加入 **Cloud Run Invoker（roles/run.invoker）**，讓網站按鈕能啟動更新工作。首次更新可能需要較長時間，因為程式會建立歷史資料。

也可從命令列手動執行一次：

```bash
gcloud run jobs execute taiex-refresh --region asia-east1 --wait
```

Job 完成後，網站會從同一個 bucket 讀取新資料。確認首頁不再顯示示範資料，再測試個股分析與新聞頁面。

## 5. 測試清單

- 開啟 Cloud Run 網址，確認會導向 Google 登入。
- 登入後確認首頁可讀取 `signals.json`、`weekly.json`、`news.json`。
- 確認隔日資料不再標示示範資料。
- 確認 `weekly_prob.csv` 已產生後，測試個股代號查詢。
- 在設定頁輸入 AI API 金鑰並執行分析。金鑰只會在瀏覽器儲存，分析時透過已登入的同源 API 即時轉送，不會寫入資料 bucket 或伺服器日誌。
- 只有全部測試通過後，才考慮停止 Render 服務。

## 注意

目前可由網站的「立即更新行情與新聞」按鈕手動啟動 Job；確認穩定後，再另外設定 Cloud Scheduler 的每日排程。不要讓 Cloud Run service 的每個執行個體各自啟動排程。
