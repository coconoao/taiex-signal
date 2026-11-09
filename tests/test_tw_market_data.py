import datetime as dt
import unittest
from unittest.mock import patch

import tw_market_data


class TestTaiwanMarketData(unittest.TestCase):
    def test_request_json_logs_safe_network_diagnostic(self):
        error = __import__("urllib.error").error.URLError(
            OSError(101, "Network is unreachable")
        )
        with patch.object(tw_market_data.urllib.request, "urlopen", side_effect=error):
            with self.assertRaisesRegex(
                tw_market_data.MarketDataError,
                r"twse\.com\.tw URLError/OSError\(errno=101\)",
            ):
                tw_market_data._request_json(
                    "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?stockNo=2330"
                )

    def test_request_json_preserves_http_status_in_safe_diagnostic(self):
        from urllib.error import HTTPError
        error = HTTPError(
            "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?stockNo=2330",
            403, "Forbidden", {}, None,
        )
        with patch.object(tw_market_data.urllib.request, "urlopen", side_effect=error):
            with self.assertRaisesRegex(
                tw_market_data.MarketDataError,
                r"twse\.com\.tw HTTPError\(http=403\)",
            ):
                tw_market_data._request_json(
                    "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?stockNo=2330"
                )

    def test_request_json_follows_same_host_https_308_redirect(self):
        from io import BytesIO
        from urllib.error import HTTPError

        redirect = HTTPError(
            "https://www.twse.com.tw/old-path",
            308, "Permanent Redirect", {"Location": "/new-path"}, None,
        )
        response = BytesIO(b'{"stat":"OK","data":[]}')
        response.__enter__ = lambda: response
        response.__exit__ = lambda *args: None
        with patch.object(
            tw_market_data.urllib.request, "urlopen",
            side_effect=[redirect, response],
        ) as open_url:
            payload = tw_market_data._request_json(
                "https://www.twse.com.tw/old-path"
            )
        self.assertEqual(payload, {"stat": "OK", "data": []})
        self.assertEqual(open_url.call_count, 2)
        self.assertEqual(
            open_url.call_args_list[1].args[0].full_url,
            "https://www.twse.com.tw/new-path",
        )

    def test_request_json_rejects_cross_host_308_redirect(self):
        from urllib.error import HTTPError

        redirect = HTTPError(
            "https://www.twse.com.tw/old-path",
            308, "Permanent Redirect",
            {"Location": "https://attacker.example/new-path"}, None,
        )
        with patch.object(
            tw_market_data.urllib.request, "urlopen", side_effect=redirect
        ) as open_url:
            with self.assertRaises(tw_market_data.MarketDataError) as caught:
                tw_market_data._request_json(
                    "https://www.twse.com.tw/old-path"
                )
        self.assertEqual(caught.exception.diagnostic_code, "twse_http_308")
        self.assertEqual(open_url.call_count, 1)

    def test_request_json_attaches_safe_http_diagnostic_code(self):
        from urllib.error import HTTPError
        error = HTTPError(
            "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?stockNo=SECRET",
            403, "Forbidden", {}, None,
        )
        with patch.object(tw_market_data.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(tw_market_data.MarketDataError) as caught:
                tw_market_data._request_json(
                    "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?stockNo=SECRET"
                )
        self.assertEqual(caught.exception.diagnostic_code, "twse_http_403")
        self.assertNotIn("SECRET", caught.exception.diagnostic_code)

    def test_fetch_months_preserves_provider_diagnostic_category(self):
        def fail(*args):
            raise tw_market_data.MarketDataError(
                "private detail", diagnostic_code="provider_invalid_json_shape"
            )

        with self.assertRaises(tw_market_data.MarketDataError) as caught:
            tw_market_data._fetch_months(fail, "2330", [dt.date(2026, 11, 1)])
        self.assertEqual(
            caught.exception.diagnostic_code,
            "history_fetch_failed:provider_invalid_json_shape",
        )
        self.assertNotIn("2330", caught.exception.diagnostic_code)

    def test_exchange_probe_preserves_safe_provider_failure_categories(self):
        def twse_failure(*args):
            raise tw_market_data.MarketDataError(
                "private detail 2330", diagnostic_code="twse_http_403"
            )

        def tpex_failure(*args):
            raise tw_market_data.MarketDataError(
                "private detail 2330", diagnostic_code="tpex_network_errno_101"
            )

        months = [dt.date(2026, 9, 1), dt.date(2026, 10, 1), dt.date(2026, 11, 1)]
        with (
            patch.object(tw_market_data, "_twse_month", side_effect=twse_failure) as twse_mock,
            patch.object(tw_market_data, "_tpex_month", side_effect=tpex_failure) as tpex_mock,
        ):
            with self.assertRaises(tw_market_data.MarketDataError) as caught:
                tw_market_data._identify_exchange("2330", None, months)
        self.assertEqual(twse_mock.call_count, 1)
        self.assertEqual(tpex_mock.call_count, 1)
        self.assertIn("twse_http_403", caught.exception.diagnostic_code)
        self.assertIn("tpex_network_errno_101", caught.exception.diagnostic_code)
        self.assertNotIn("2330", caught.exception.diagnostic_code)
        self.assertNotIn("private detail", caught.exception.diagnostic_code)

    def test_exchange_probe_without_provider_errors_is_no_recent_history(self):
        months = [dt.date(2026, 11, 1)]
        with (
            patch.object(tw_market_data, "_twse_month", return_value=[]),
            patch.object(tw_market_data, "_tpex_month", return_value=[]),
        ):
            with self.assertRaises(tw_market_data.MarketDataError) as caught:
                tw_market_data._identify_exchange("2330", None, months)
        self.assertEqual(caught.exception.diagnostic_code, "no_recent_history")

    def test_roc_date_converts_taiwan_calendar(self):
        self.assertEqual(
            tw_market_data._roc_date("115/11/06"), dt.date(2026, 11, 6)
        )

    def test_number_handles_exchange_formats(self):
        self.assertEqual(tw_market_data._number("+1,234.50"), 1234.5)
        self.assertEqual(tw_market_data._number("-12.00"), -12.0)
        self.assertIsNone(tw_market_data._number("X0.00"))
        self.assertIsNone(tw_market_data._number("--"))

    def test_adjusted_close_uses_exchange_reference_price_change(self):
        rows = [
            {"date": dt.date(2026, 1, 2), "raw_close": 100.0, "change": 0.0,
             "volume": 1000, "open": 100, "high": 100, "low": 100},
            # Ordinary unchanged-price session: reference price is also 95.
            {"date": dt.date(2026, 1, 5), "raw_close": 95.0, "change": 0.0,
             "volume": 2000, "open": 95, "high": 95, "low": 95},
            {"date": dt.date(2026, 1, 6), "raw_close": 96.0, "change": 1.0,
             "volume": 3000, "open": 95, "high": 97, "low": 95},
        ]
        frame = tw_market_data._to_adjusted_frame(rows, "TWSE")
        self.assertEqual(frame["Close"].iloc[0], 100.0)
        self.assertEqual(frame["Close"].iloc[1], 100.0)
        self.assertAlmostEqual(frame["Close"].iloc[2], 100 * 96 / 95, places=4)
        self.assertEqual(frame["RawClose"].tolist(), [100.0, 95.0, 96.0])
        self.assertEqual(frame.attrs["source"], "TWSE")

    def test_adjusted_series_fails_closed_on_unparsed_change(self):
        rows = [
            {"date": dt.date(2026, 1, 2), "raw_close": 100.0, "change": 0.0,
             "volume": 1000, "open": 100, "high": 100, "low": 100},
            {"date": dt.date(2026, 1, 5), "raw_close": 95.0, "change": None,
             "volume": 2000, "open": 95, "high": 95, "low": 95},
        ]
        with self.assertRaisesRegex(tw_market_data.MarketDataError, "無法解析"):
            tw_market_data._to_adjusted_frame(rows, "TPEx")

    def test_twse_rejects_unexpected_status_even_if_rows_exist(self):
        row = ["115/11/06", "1,000", "100,000", "100", "100", "100", "100", "0"]
        with patch.object(
            tw_market_data, "_request_json",
            return_value={"stat": "invalid ticker", "data": [row]},
        ):
            with self.assertRaisesRegex(tw_market_data.MarketDataError, "非成功狀態"):
                tw_market_data._twse_month("2330", dt.date(2026, 11, 1))

    def test_twse_no_data_status_returns_empty_rows(self):
        with patch.object(
            tw_market_data, "_request_json",
            return_value={"stat": "很抱歉，沒有符合條件的資料!", "data": []},
        ):
            self.assertEqual(
                tw_market_data._twse_month("2330", dt.date(2026, 11, 1)), []
            )

    def test_taiex_history_preserves_safe_provider_diagnostic(self):
        with (
            patch.object(tw_market_data, "_month_starts", return_value=[dt.date(2026, 11, 1)]),
            patch.object(
                tw_market_data,
                "_taiex_month",
                side_effect=tw_market_data.MarketDataError(
                    "private provider detail", diagnostic_code="twse_http_403"
                ),
            ),
            patch.dict(tw_market_data._INDEX_CACHE, {}, clear=True),
        ):
            with self.assertRaises(tw_market_data.MarketDataError) as caught:
                tw_market_data.load_taiex_history(years=4)
        self.assertEqual(
            caught.exception.diagnostic_code,
            "history_fetch_failed:twse_http_403",
        )
        self.assertNotIn("private provider detail", caught.exception.diagnostic_code)

    def test_taiex_uses_official_indices_report_path(self):
        seen = []

        def fake_request(url):
            seen.append(url)
            return {"data": [["115/11/06", "100", "101", "99", "100.5"]]}

        with patch.object(tw_market_data, "_request_json", side_effect=fake_request):
            rows = tw_market_data._taiex_month(dt.date(2026, 11, 1))
        self.assertEqual(rows, [(dt.date(2026, 11, 6), 100.5)])
        self.assertTrue(seen[0].startswith(
            "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?"
        ))

    def test_tpex_historical_volume_is_converted_from_lots_to_shares(self):
        payload = {
            "tables": [{
                "data": [["115/11/06", "12", "100", "10", "11", "9", "10.5", "0"]]
            }]
        }
        with patch.object(tw_market_data, "_request_json", return_value=payload):
            rows = tw_market_data._tpex_month("6488", dt.date(2026, 11, 1))
        self.assertEqual(rows[0]["volume"], 12000.0)


    def test_x_prefixed_change_fails_closed_in_adjusted_series(self):
        rows = [
            {"date": dt.date(2026, 1, 2), "raw_close": 100.0, "change": 0.0,
             "volume": 1000, "open": 100, "high": 100, "low": 100},
            {"date": dt.date(2026, 1, 5), "raw_close": 95.0, "change": None,
             "volume": 2000, "open": 95, "high": 95, "low": 95},
        ]
        with self.assertRaisesRegex(tw_market_data.MarketDataError, "無法解析"):
            tw_market_data._to_adjusted_frame(rows, "TWSE")


    def test_taiex_cache_is_scoped_by_requested_years(self):
        dates = [
            (dt.date(2025, 1, 1) + dt.timedelta(days=i), 10000.0 + i)
            for i in range(252)
        ]
        with (
            patch.object(tw_market_data, "_INDEX_CACHE", {}),
            patch.object(tw_market_data, "_month_starts", return_value=[dt.date(2026, 11, 1)]),
            patch.object(tw_market_data, "_taiex_month", return_value=dates) as fetch,
            patch.object(tw_market_data.time, "time", return_value=1000.0),
        ):
            tw_market_data.load_taiex_history(years=2)
            tw_market_data.load_taiex_history(years=10)
            self.assertEqual(fetch.call_count, 2)

    def test_roc_date_parses_twse_chinese_date(self):
        self.assertEqual(
            tw_market_data._roc_date("114年12月11日"), dt.date(2025, 12, 11)
        )

    def test_twse_exright_parser_uses_cash_dividend_reference_only(self):
        payload = {
            "fields": ["資料日期", "股票代號", "除權息參考價", "權/息"],
            "data": [
                ["114年12月11日", "2330", "1,499.99", "息"],
                ["114年12月12日", "2330", "100.00", "權"],
            ],
        }
        with patch.object(tw_market_data, "_request_json", return_value=payload):
            refs = tw_market_data._twse_exright_month(dt.date(2025, 12, 1))
        self.assertEqual(refs[("2330", dt.date(2025, 12, 11))], 1499.99)
        self.assertNotIn(("2330", dt.date(2025, 12, 12)), refs)

    def test_tpex_exright_parser_reads_table_schema(self):
        payload = {
            "tables": [{
                "fields": ["除權息日期", "代號", "除權息參考價", "權/息"],
                "data": [
                    ["114/12/31", "6488", "411.00", "除息"],
                    ["114/12/30", "6488", "400.00", "除權"],
                ],
            }]
        }
        with patch.object(tw_market_data, "_request_json", return_value=payload):
            refs = tw_market_data._tpex_exright_month(dt.date(2025, 12, 1))
        self.assertEqual(refs[("6488", dt.date(2025, 12, 31))], 411.0)
        self.assertNotIn(("6488", dt.date(2025, 12, 30)), refs)

    def test_adjusted_close_uses_official_cash_dividend_reference(self):
        rows = [
            {"date": dt.date(2025, 12, 10), "raw_close": 1505.0, "change": 0.0,
             "volume": 1000, "open": 1500, "high": 1505, "low": 1490},
            {"date": dt.date(2025, 12, 11), "raw_close": 1500.0, "change": None,
             "volume": 2000, "open": 1500, "high": 1510, "low": 1490},
        ]
        frame = tw_market_data._to_adjusted_frame(
            rows, "TWSE", "2330", {dt.date(2025, 12, 11): 1499.99}
        )
        self.assertAlmostEqual(frame["Close"].iloc[1], 1505.0 * 1500.0 / 1499.99, places=5)

    def test_adjustment_still_fails_closed_without_official_action_factor(self):
        rows = [
            {"date": dt.date(2025, 12, 10), "raw_close": 413.0, "change": 0.0,
             "volume": 1000, "open": 413, "high": 413, "low": 413},
            {"date": dt.date(2025, 12, 31), "raw_close": 410.0, "change": None,
             "volume": 2000, "open": 411, "high": 412, "low": 409},
        ]
        with self.assertRaisesRegex(tw_market_data.MarketDataError, "沒有可核對"):
            tw_market_data._to_adjusted_frame(rows, "TPEx", "6488", {})



if __name__ == "__main__":
    unittest.main()
