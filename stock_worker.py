"""Dedicated stock-analysis worker service.

Public health endpoint; analysis endpoint requires a shared bearer token.
The worker runs stock research independently from the main dashboard service.
"""
import concurrent.futures
import json
import os
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tw_market_data import MarketDataError
from taiex_realtime import IndexQuoteError, load_taiex_realtime

TOKEN = os.environ.get("STOCK_WORKER_TOKEN", "")
MAX_BODY = 4096
JOB_TTL_SECONDS = 1800
MAX_STORED_JOBS = 20
_JOB_LOCK = threading.Lock()
_JOBS = {}
_JOB_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="stock-analysis")
DIAGNOSTICS_UI_ENABLED = os.environ.get("STAGING_DIAGNOSTICS_UI", "").lower() == "true"
_SAFE_DIAGNOSTIC = re.compile(
    r"(?:no_recent_history|provider_invalid_json_shape|provider_error|adjustment_reference_missing|adjustment_reference_invalid|raw_close_invalid|"
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


def _prune_jobs_locked(now=None):
    now = time.time() if now is None else now
    expired = [
        job_id for job_id, job in _JOBS.items()
        if job.get("finished_at") and now - job["finished_at"] > JOB_TTL_SECONDS
    ]
    for job_id in expired:
        _JOBS.pop(job_id, None)
    if len(_JOBS) > MAX_STORED_JOBS:
        completed = sorted(
            ((job.get("finished_at") or now, job_id) for job_id, job in _JOBS.items()
             if job.get("finished_at")),
        )
        for _, job_id in completed[:max(0, len(_JOBS) - MAX_STORED_JOBS)]:
            _JOBS.pop(job_id, None)


def _run_analysis_job(job_id, code):
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job["state"] = "running"
    result, status = analyze_code(code)
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update({
                "state": "completed",
                "result": result,
                "http_status": status,
                "finished_at": time.time(),
            })


def _create_analysis_job(code):
    job_id = secrets.token_urlsafe(24)
    with _JOB_LOCK:
        _prune_jobs_locked()
        active = sum(1 for job in _JOBS.values() if job["state"] in ("queued", "running"))
        if active >= 2:
            return None
        _JOBS[job_id] = {
            "state": "queued",
            "created_at": time.time(),
            "finished_at": None,
        }
    try:
        _JOB_POOL.submit(_run_analysis_job, job_id, code)
    except RuntimeError:
        with _JOB_LOCK:
            _JOBS.pop(job_id, None)
        return None
    return job_id


def _get_analysis_job(job_id):
    with _JOB_LOCK:
        _prune_jobs_locked()
        job = _JOBS.get(job_id)
        if job is None:
            return None
        return dict(job)


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
<button id="run">開始測試</button><pre id="result">尚未測試</pre><h3>分析摘要</h3><pre id="summary">分析完成後會顯示易讀摘要</pre><details><summary>查看原始 JSON（除錯用）</summary><pre id="raw">尚無資料</pre></details>
<script>
document.getElementById("run").addEventListener("click",async()=>{
 const token=document.getElementById("token").value;
 const code=document.getElementById("code").value.trim();
 const out=document.getElementById("result");
 const button=document.getElementById("run");
 if(!token){out.textContent="請輸入 staging token";return}
 if(!code){out.textContent="請輸入股票代號";return}
 button.disabled=true;
 const started=performance.now();
 const elapsed=()=>((performance.now()-started)/1000).toFixed(1)+" 秒";
 const request=async(url,options={},limit=20000)=>{
  const controller=new AbortController();
  let timedOut=false;
  const timer=setTimeout(()=>{timedOut=true;controller.abort()},limit);
  try{return await fetch(url,{...options,signal:controller.signal,cache:"no-store"})}
  catch(e){if(timedOut)e.message="client_timeout";throw e}
  finally{clearTimeout(timer)}
 };
 try{
  out.textContent="步驟 1/2：檢查 Worker 連線…";
  let health;
  const healthStart=performance.now();
  try{health=await request("/healthz",{},20000)}
  catch(e){out.textContent="健康檢查失敗（"+e.name+"），等待 "+elapsed()+"。尚未送出分析請求。";return}
  const healthTime=((performance.now()-healthStart)/1000).toFixed(1);
  if(!health.ok){out.textContent="健康檢查 HTTP "+health.status+"（"+healthTime+" 秒）。尚未送出分析請求。";return}
  out.textContent="健康檢查通過（HTTP "+health.status+"，"+healthTime+" 秒）。正在建立背景分析工作…";
  try{
   const created=await request("/analysis-jobs",{method:"POST",headers:{"Content-Type":"application/json","Authorization":"Bearer "+token},body:JSON.stringify({code})},20000);
   const createdRaw=await created.text();
   let createdBody;
   try{createdBody=JSON.parse(createdRaw)}catch(_){createdBody={error:"Worker 回應不是有效 JSON"}}
   if(!created.ok||!createdBody.job_id){out.textContent="建立分析工作失敗：HTTP "+created.status+"\\n"+JSON.stringify(createdBody,null,2);return}
   const jobId=createdBody.job_id;
   out.textContent="背景分析已建立（HTTP "+created.status+"）。Worker 正在處理；此頁每 3 秒查詢一次狀態。";
   for(let attempt=0;attempt<180;attempt++){
    await new Promise(resolve=>setTimeout(resolve,3000));
    const poll=await request("/analysis-jobs/"+encodeURIComponent(jobId),{headers:{"Authorization":"Bearer "+token}},20000);
    const pollRaw=await poll.text();
    let body;
    try{body=JSON.parse(pollRaw)}catch(_){body={error:"狀態回應不是有效 JSON"}}
    if(poll.status===202){
     out.textContent="背景分析進行中（"+elapsed()+"）。狀態："+(body.state||"處理中")+"；已查詢 "+(attempt+1)+" 次。";
     continue;
    }
    if(poll.ok&&body.state==="completed"){
     if(Number(body.analysis_http_status)>=400){
      const failure=body.result&&body.result.error?body.result.error:"分析服務回報失敗";
      out.textContent="分析未成功（HTTP "+body.analysis_http_status+"）\\n"+failure+"\\n總等待："+elapsed();
      document.getElementById("summary").textContent="本次沒有產生有效分析資料。請依上方錯誤排除後再測試。";
      document.getElementById("raw").textContent=JSON.stringify(body.result||{},null,2);
      return;
     }
     out.textContent="健康檢查：HTTP "+health.status+"（"+healthTime+" 秒）\\n分析工作：完成\\n分析 HTTP 狀態："+body.analysis_http_status+"\\n總等待："+elapsed();
     const data=body.result||{};
     const pct=(v,d=1)=>typeof v==="number"?v.toFixed(d)+"%":"資料不足";
     const line=(label,value)=>label+"："+(value===undefined||value===null||value===""?"—":value);
     const rows=[
      "【標的基本資料】",
      line("股票代號",data.code),
      line("類型",data.asset_type),
      line("資料日期",data.asof),
      line("最近收盤價",typeof data.last==="number"?data.last.toFixed(2):"—"),
      line("資料來源",data.market_data_source||data.quote_source),
      line("歷史資料區間",data.history_start+" 至 "+data.history_asof),
      line("歷史交易日數",data.history_sessions),
      line("歷史資料年限",data.history_years_requested+" 年"),
      "",
      "【風險與歷史指標】",
      line("Beta",data.beta),
      line("20 日動能",pct(data.mom20)),
      line("60 日年化波動度",pct(data.vol60)),
      "",
      "【歷史報酬統計（非預測）】"
     ];
     const hist=data.historical_horizons||{};
     for(const key of ["1m","3m","6m"]){
      const h=hist[key];
      rows.push(key+"："+(h?("歷史正報酬比例 "+pct(h.positive_pct)+
       "；平均報酬 "+pct(h.average_return_pct,2)+"；樣本 "+h.n):"資料不足"));
     }
     rows.push("","【研究模型估計（非買賣訊號）】");
     const horizons=(data.long_term_model||{}).horizons||{};
     for(const key of ["1m","3m","6m"]){
      const h=horizons[key]||{};
      if(h.status==="available"&&typeof h.estimated_positive_return_probability_pct==="number"){
       rows.push(key+"：正報酬機率估計 "+pct(h.estimated_positive_return_probability_pct,2));
       rows.push("  分類："+(h.research_category||"—")+"；訓練樣本："+(h.training_samples??"—"));
      }else{
       rows.push(key+"：尚未達到校準條件（"+(h.note||"暫不顯示機率")+")");
      }
     }
     rows.push("",(data.long_term_model||{}).disclaimer||"研究用途；不構成投資建議或交易指示。");
     document.getElementById("summary").textContent=rows.join("\\n");
     document.getElementById("raw").textContent=JSON.stringify(data,null,2);
     return;
    }
    out.textContent="查詢分析工作失敗：HTTP "+poll.status+"\\n"+JSON.stringify(body,null,2);
    return;
   }
   out.textContent="背景分析仍未完成，已等待 "+elapsed()+"。工作可能仍在 Worker 執行；請稍後重新開啟頁面並檢查紀錄。";
  }catch(e){
   const reason=e.name==="AbortError"?"瀏覽器中止狀態查詢":e.message==="client_timeout"?"單次狀態查詢超過 20 秒":e.name;
   out.textContent="背景分析連線失敗："+reason+"\\n總等待："+elapsed()+"。請查看 Worker 是否有安全的分析錯誤紀錄。";
  }
 }finally{button.disabled=false}
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
        if self.path == "/api/taiex":
            supplied = self.headers.get("Authorization", "")
            if not TOKEN or supplied != "Bearer " + TOKEN:
                return self._json({"error": "unauthorized"}, 401)
            try:
                return self._json(load_taiex_realtime())
            except IndexQuoteError:
                return self._json({
                    "available": False, "realtime": False,
                    "status": "UPSTREAM_ERROR", "symbol": "IX0001",
                    "name": "發行量加權股價指數", "source": "Fugle WebSocket",
                }, 503)
        if self.path.startswith("/analysis-jobs/"):
            supplied = self.headers.get("Authorization", "")
            if not TOKEN or supplied != "Bearer " + TOKEN:
                return self._json({"error": "unauthorized"}, 401)
            job_id = self.path[len("/analysis-jobs/"):]
            if not re.fullmatch(r"[A-Za-z0-9_-]{20,64}", job_id):
                return self._json({"error": "not found"}, 404)
            job = _get_analysis_job(job_id)
            if job is None:
                return self._json({"error": "工作不存在或已過期"}, 404)
            payload = {"job_id": job_id, "state": job["state"]}
            if job["state"] == "completed":
                payload["result"] = job["result"]
                payload["analysis_http_status"] = job["http_status"]
            return self._json(payload, 200 if job["state"] == "completed" else 202)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path not in ("/analyze", "/analysis-jobs"):
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

        if self.path == "/analysis-jobs":
            os.chdir("/tmp")
            job_id = _create_analysis_job(code)
            if job_id is None:
                return self._json({"error": "目前已有分析工作執行或排隊，請稍後再試"}, 429)
            return self._json({"job_id": job_id, "state": "queued"}, 202)

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
