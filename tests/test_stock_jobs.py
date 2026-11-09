import io
import json
import unittest
from unittest.mock import patch

import app as web_app


class SiteStockJobTests(unittest.TestCase):
    def setUp(self):
        with web_app._SITE_STOCK_JOB_LOCK:
            web_app._SITE_STOCK_JOBS.clear()

    def tearDown(self):
        with web_app._SITE_STOCK_JOB_LOCK:
            web_app._SITE_STOCK_JOBS.clear()

    def test_create_job_validates_symbol_and_queues_background_task(self):
        with patch.multiple(
            web_app,
            STOCK_WORKER_URL="https://worker.example",
            STOCK_WORKER_TOKEN="test-token",
        ), patch.object(web_app._SITE_STOCK_JOB_POOL, "submit") as submit:
            job_id = web_app.create_site_stock_job(" 2330 ")
            self.assertIsInstance(job_id, str)
            self.assertEqual(web_app._SITE_STOCK_JOBS[job_id]["state"], "starting")
            submit.assert_called_once_with(web_app._start_site_stock_job, job_id, "2330")

    def test_create_job_rejects_invalid_symbol(self):
        with patch.multiple(
            web_app,
            STOCK_WORKER_URL="https://worker.example",
            STOCK_WORKER_TOKEN="test-token",
        ):
            with self.assertRaises(ValueError):
                web_app.create_site_stock_job("../etc/passwd")

    def test_expired_job_is_removed_and_returns_gone(self):
        with web_app._SITE_STOCK_JOB_LOCK:
            web_app._SITE_STOCK_JOBS["A" * 24] = {
                "state": "running",
                "remote_id": "B" * 24,
                "created_at": 1,
            }
        with patch.object(web_app.time, "time", return_value=web_app._SITE_STOCK_JOB_TTL + 2):
            payload, status = web_app.get_site_stock_job("A" * 24)
        self.assertEqual(status, 410)
        self.assertEqual(payload["state"], "failed")
        self.assertNotIn("A" * 24, web_app._SITE_STOCK_JOBS)

    def test_completed_remote_job_returns_analysis_result(self):
        site_id = "A" * 24
        with web_app._SITE_STOCK_JOB_LOCK:
            web_app._SITE_STOCK_JOBS[site_id] = {
                "state": "running",
                "remote_id": "B" * 24,
                "created_at": 100,
            }
        response = io.BytesIO(json.dumps({
            "state": "completed",
            "analysis_http_status": 200,
            "result": {"code": "2330", "ok": True},
        }).encode("utf-8"))
        with patch.multiple(
            web_app,
            STOCK_WORKER_URL="https://worker.example",
            STOCK_WORKER_TOKEN="test-token",
        ), patch.object(web_app.urllib.request, "urlopen", return_value=response) as urlopen:
            payload, status = web_app.get_site_stock_job(site_id)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"state": "completed", "result": {"code": "2330", "ok": True}})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertIn("/analysis-jobs/" + "B" * 24, request.full_url)

    def test_worker_start_failure_marks_site_job_failed(self):
        site_id = "A" * 24
        with web_app._SITE_STOCK_JOB_LOCK:
            web_app._SITE_STOCK_JOBS[site_id] = {"state": "starting", "created_at": 100}
        with patch.object(web_app, "warm_stock_worker", side_effect=web_app.StockWorkerGatewayError("worker unavailable")):
            web_app._start_site_stock_job(site_id, "2330")
        self.assertEqual(web_app._SITE_STOCK_JOBS[site_id]["state"], "failed")
        self.assertEqual(web_app._SITE_STOCK_JOBS[site_id]["error"], "worker unavailable")


if __name__ == "__main__":
    unittest.main()
