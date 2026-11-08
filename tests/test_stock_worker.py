import sys
import types
import unittest
from unittest.mock import Mock, patch

import stock_worker
from tw_market_data import MarketDataError


class AnalyzeCodeTests(unittest.TestCase):
    def _run(self, outcome):
        fake_stock = types.ModuleType("stock")
        if isinstance(outcome, BaseException):
            fake_stock.analyze = Mock(side_effect=outcome)
        else:
            fake_stock.analyze = Mock(return_value=outcome)
        with patch.dict(sys.modules, {"stock": fake_stock}):
            return stock_worker.analyze_code("2330")

    def test_success_returns_200(self):
        payload = {"code": "2330", "name": "測試"}
        result, status = self._run(payload)
        self.assertEqual(status, 200)
        self.assertEqual(result, payload)

    def test_official_market_data_error_returns_503(self):
        result, status = self._run(MarketDataError("private provider detail"))
        self.assertEqual(status, 503)
        self.assertIn("稍後重試", result["error"])
        self.assertNotIn("private provider detail", result["error"])

    def test_structured_diagnostic_logs_category_without_exception_message(self):
        error = MarketDataError(
            "private detail with symbol 2330 and query token=secret",
            diagnostic_code="exchange_probe_failed:twse_http_403,tpex_network_errno_101",
        )
        with patch("builtins.print") as log:
            result, status = self._run(error)
        self.assertEqual(status, 503)
        logged = " ".join(str(call) for call in log.call_args_list)
        self.assertIn("twse_http_403", logged)
        self.assertIn("tpex_network_errno_101", logged)
        self.assertNotIn("2330", logged)
        self.assertNotIn("secret", logged)
        self.assertNotIn("private detail", logged)

    def test_structured_history_diagnostic_accepts_provider_shape_category(self):
        error = MarketDataError(
            "private detail",
            diagnostic_code="history_fetch_failed:provider_invalid_json_shape",
        )
        with patch("builtins.print") as log:
            result, status = self._run(error)
        self.assertEqual(status, 503)
        logged = " ".join(str(call) for call in log.call_args_list)
        self.assertIn("history_fetch_failed:provider_invalid_json_shape", logged)
        self.assertNotIn("private detail", logged)

    def test_structured_diagnostic_rejects_untrusted_text(self):
        error = MarketDataError(
            "private detail",
            diagnostic_code="bad category; symbol=2330",
        )
        with patch("builtins.print") as log:
            self._run(error)
        logged = " ".join(str(call) for call in log.call_args_list)
        self.assertNotIn("symbol=2330", logged)
        self.assertIn("provider_or_data_validation_error", logged)

    def test_redirect_diagnostic_logs_only_allowlisted_host_and_path(self):
        error = MarketDataError(
            "官方行情來源連線失敗：www.twse.com.tw "
            "HTTPError(http=307)(redirect=www.twse.com.tw/new/path)"
        )
        with patch("builtins.print") as log:
            result, status = self._run(error)
        self.assertEqual(status, 503)
        logged = " ".join(str(call) for call in log.call_args_list)
        self.assertIn("redirect=www.twse.com.tw/new/path", logged)
        self.assertNotIn("?", logged)

    def test_redirect_diagnostic_rejects_untrusted_redirect_host(self):
        error = MarketDataError(
            "官方行情來源連線失敗：www.twse.com.tw "
            "HTTPError(http=307)(redirect=evil.example/private)"
        )
        with patch("builtins.print") as log:
            self._run(error)
        logged = " ".join(str(call) for call in log.call_args_list)
        self.assertNotIn("evil.example", logged)

    def test_analysis_value_error_returns_422(self):
        result, status = self._run(ValueError("歷史資料不足"))
        self.assertEqual(status, 422)
        self.assertEqual(result["error"], "歷史資料不足")

    def test_unexpected_error_returns_500_without_raw_detail(self):
        result, status = self._run(RuntimeError("private internal detail"))
        self.assertEqual(status, 500)
        self.assertNotIn("private internal detail", result["error"])


if __name__ == "__main__":
    unittest.main()
