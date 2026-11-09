import datetime as dt
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import tw_market_data


class TestTaiwanMarketData(unittest.TestCase):
    def test_request_json_logs_safe_network_diagnostic(self):
        error = OSError(101, "Network is unreachable")
        with patch.object(tw_market_data.curl_requests, "get", side_effect=error):
            with self.assertRaisesRegex(
                tw_market_data.MarketDataError,
                r"twse\.com\.tw OSError\(errno=101\)",
            ):
                tw_market_data._request_json(
                    "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?stockNo=2330"
                )

    def test_request_json_preserves_http_status_in_safe_diagnostic(self):
        from types import SimpleNamespace
        response = SimpleNamespace(status_code=403, headers={}, content=b"")
        with patch.object(tw_market_data.curl_requests, "get", return_value=response):
            with self.assertRaisesRegex(
                tw_market_data.MarketDataError,
                r"twse\.com\.tw HTTPError\(http=403\)",
            ):
                tw_market_data._request_json(
                    "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?stockNo=2330"
                )

    def test_request_json_follows_same_host_https_308_redirect(self):
        from types import SimpleNamespace
        redirect = SimpleNamespace(status_code=308, headers={"Location": "/new-path"}, content=b"")
        response = SimpleNamespace(status_code=200, headers={}, content=b'{"stat":"OK","data":[]}')
        with patch.object(
            tw_market_data.curl_requests, "get",
            side_effect=[redirect, response],
        ) as get:
            payload = tw_market_data._request_json("https://www.twse.com.tw/old-path")
        self.assertEqual(payload, {"stat": "OK", "data": []})
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[1].args[0], "https://www.twse.com.tw/new-path")
        self.assertEqual(get.call_args_list[1].kwargs["impersonate"], "chrome")

    def test_request_json_rejects_cross_host_308_redirect(self):
        from types import SimpleNamespace
        redirect = SimpleNamespace(
            status_code=308,
            headers={"Location": "https://attacker.example/new-path"},
            content=b"",
        )
        with patch.object(
            tw_market_data.curl_requests, "get", return_value=redirect
        ) as get:
            with self.assertRaises(tw_market_data.MarketDataError) as caught:
                tw_market_data._request_json("https://www.twse.com.tw/old-path")
        self.assertEqual(caught.exception.diagnostic_code, "twse_http_308")
        self.assertEqual(get.call_count, 1)

    def test_request_json_attaches_safe_http_diagnostic_code(self):
        from types import SimpleNamespace
        response = SimpleNamespace(status_code=403, headers={}, content=b"")
        with patch.object(tw_market_data.curl_requests, "get", return_value=response):
            with self.assertRaises(tw_market_data.MarketDataError) as caught:
                tw_market_data._request_json(
                    "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?stockNo=SECRET"
                )
        self.assertEqual(caught.exception.diagnostic_code, "twse_http_403")
        self.assertNotIn("SECRET", caught.exception.diagnostic_code)

    def test_request_json_uses_bounded_socket_timeout_and_chrome_impersonation(self):
        response = type("Response", (), {"status_code": 200, "headers": {}, "content": b'{"ok":true}'})()
        with patch.object(tw_market_data.curl_requests, "get", return_value=response) as get:
            self.assertEqual(
                tw_market_data._request_json("https://www.twse.com.tw/test"),
                {"ok": True},
            )
        self.assertEqual(get.call_args.kwargs["timeout"], 15)
        self.assertEqual(get.call_args.kwargs["impersonate"], "chrome")
        self.assertFalse(get.call_args.kwargs["allow_redirects"])

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

    def test_exchange_probe_checks_only_latest_month_per_provider(self):
        months = [
            dt.date(2026, 9, 1),
            dt.date(2026, 10, 1),
            dt.date(2026, 11, 1),
        ]
        timeout = tw_market_data.MarketDataError(
            "timeout", diagnostic_code="twse_timeout"
        )
        tpex_timeout = tw_market_data.MarketDataError(
            "timeout", diagnostic_code="tpex_timeout"
        )
        with (
            patch.object(
                tw_market_data, "_twse_month", side_effect=timeout
            ) as twse_mock,
            patch.object(
                tw_market_data, "_tpex_month", side_effect=tpex_timeout
            ) as tpex_mock,
        ):
            with self.assertRaises(tw_market_data.MarketDataError):
                tw_market_data._identify_exchange("2330", None, months)
        self.assertEqual(twse_mock.call_count, 1)
        self.assertEqual(tpex_mock.call_count, 1)
        self.assertEqual(twse_mock.call_args.args[1], dt.date(2026, 11, 1))
        self.assertEqual(tpex_mock.call_args.args[1], dt.date(2026, 11, 1))

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

    def test_number_rejects_non_finite_values(self):
        self.assertIsNone(tw_market_data._number("nan"))
        self.assertIsNone(tw_market_data._number("inf"))
        self.assertIsNone(tw_market_data._number("-inf"))

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

    def test_twse_exright_parser_includes_rights_and_dividend_reference_prices(self):
        payload = {
            "fields": ["資料日期", "股票代號", "除權息參考價", "權/息"],
            "data": [
                ["114年12月11日", "2330", "1,499.99", "息"],
                ["114年12月12日", "2330", "100.00", "權"],
                ["114年12月15日", "2330", "98.50", "權息"],
            ],
        }
        with patch.object(tw_market_data, "_request_json", return_value=payload):
            refs = tw_market_data._twse_exright_month(dt.date(2025, 12, 1))
        self.assertEqual(refs[("2330", dt.date(2025, 12, 11))], 1499.99)
        self.assertEqual(refs[("2330", dt.date(2025, 12, 12))], 100.0)
        self.assertEqual(refs[("2330", dt.date(2025, 12, 15))], 98.5)

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
        self.assertEqual(refs[("6488", dt.date(2025, 12, 30))], 400.0)

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



    def test_request_json_retries_invalid_json_before_success(self):
        from types import SimpleNamespace
        invalid = SimpleNamespace(status_code=200, headers={}, content=b"<html>temporary edge response</html>")
        valid = SimpleNamespace(status_code=200, headers={}, content=b'{"ok":true}')
        with (
            patch.object(tw_market_data.curl_requests, "get", side_effect=[invalid, valid]) as get,
            patch.object(tw_market_data.time, "sleep"),
        ):
            self.assertEqual(
                tw_market_data._request_json("https://www.twse.com.tw/test"),
                {"ok": True},
            )
        self.assertEqual(get.call_count, 2)

    def test_request_json_stops_when_retry_after_exceeds_safe_window(self):
        from types import SimpleNamespace
        response = SimpleNamespace(
            status_code=429, headers={"Retry-After": "30"}, content=b""
        )
        with (
            patch.object(tw_market_data.curl_requests, "get", return_value=response) as get,
            patch.object(tw_market_data.time, "sleep") as sleep,
        ):
            with self.assertRaises(tw_market_data.MarketDataError) as caught:
                tw_market_data._request_json("https://www.twse.com.tw/test")
        self.assertEqual(caught.exception.diagnostic_code, "twse_http_429")
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_request_json_retries_429_using_retry_after(self):
        from types import SimpleNamespace
        limited = SimpleNamespace(
            status_code=429, headers={"Retry-After": "2"}, content=b""
        )
        valid = SimpleNamespace(status_code=200, headers={}, content=b'{"ok":true}')
        with (
            patch.object(tw_market_data.curl_requests, "get", side_effect=[limited, valid]) as get,
            patch.object(tw_market_data.time, "sleep") as sleep,
        ):
            self.assertEqual(
                tw_market_data._request_json("https://www.twse.com.tw/test"),
                {"ok": True},
            )
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(2.0)

    def test_request_json_does_not_retry_client_errors(self):
        from types import SimpleNamespace
        for status in (400, 401, 403):
            response = SimpleNamespace(status_code=status, headers={}, content=b"")
            with (
                patch.object(tw_market_data.curl_requests, "get", return_value=response) as get,
                patch.object(tw_market_data.time, "sleep") as sleep,
            ):
                with self.assertRaises(tw_market_data.MarketDataError) as caught:
                    tw_market_data._request_json("https://www.twse.com.tw/test")
            self.assertEqual(caught.exception.diagnostic_code, f"twse_http_{status}")
            self.assertEqual(get.call_count, 1)
            sleep.assert_not_called()

    def test_request_json_reports_invalid_json_after_retry_budget(self):
        from types import SimpleNamespace
        invalid = SimpleNamespace(status_code=200, headers={}, content=b"<html>blocked</html>")
        with (
            patch.object(tw_market_data.curl_requests, "get", side_effect=[invalid, invalid, invalid]) as get,
            patch.object(tw_market_data.time, "sleep"),
        ):
            with self.assertRaises(tw_market_data.MarketDataError) as caught:
                tw_market_data._request_json("https://www.twse.com.tw/test")
        self.assertEqual(caught.exception.diagnostic_code, "twse_invalid_json")
        self.assertEqual(get.call_count, 3)

    def test_request_json_reports_timeout_after_retry_budget(self):
        with (
            patch.object(
                tw_market_data.curl_requests, "get", side_effect=TimeoutError("timed out")
            ) as get,
            patch.object(tw_market_data.time, "sleep"),
        ):
            with self.assertRaises(tw_market_data.MarketDataError) as caught:
                tw_market_data._request_json("https://www.twse.com.tw/test")
        self.assertEqual(caught.exception.diagnostic_code, "twse_timeout")
        self.assertEqual(get.call_count, 3)

    def test_request_json_retries_transient_network_error(self):
        from types import SimpleNamespace
        error = OSError(101, "Network is unreachable")
        valid = SimpleNamespace(status_code=200, headers={}, content=b'{"ok":true}')
        with (
            patch.object(tw_market_data.curl_requests, "get", side_effect=[error, valid]) as get,
            patch.object(tw_market_data.time, "sleep"),
        ):
            self.assertEqual(
                tw_market_data._request_json("https://www.twse.com.tw/test"),
                {"ok": True},
            )
        self.assertEqual(get.call_count, 2)

    def test_exchange_probe_uses_older_month_when_latest_month_is_empty(self):
        months = [
            dt.date(2026, 9, 1),
            dt.date(2026, 10, 1),
            dt.date(2026, 11, 1),
        ]
        older_rows = [{
            "date": dt.date(2026, 10, 5), "raw_close": 100.0, "change": 0.0,
            "volume": 1000, "open": 100, "high": 100, "low": 100,
        }]
        with (
            patch.object(tw_market_data, "_twse_month", side_effect=[[], older_rows]) as twse,
            patch.object(tw_market_data, "_tpex_month") as tpex,
        ):
            source, fetcher = tw_market_data._identify_exchange("1234", None, months)
            self.assertIs(fetcher, twse)
            self.assertEqual(twse.call_count, 2)
            self.assertEqual(twse.call_args_list[1].args[1], dt.date(2026, 10, 1))
            tpex.assert_not_called()
        self.assertEqual(source, "TWSE")

    def test_tpex_month_reads_data_from_later_table(self):
        payload = {
            "tables": [
                {"data": []},
                {"data": [[
                    "115/11/06", "12", "100", "10", "11", "9", "10.5", "0"
                ]]},
            ]
        }
        with patch.object(tw_market_data, "_request_json", return_value=payload):
            rows = tw_market_data._tpex_month("6488", dt.date(2026, 11, 1))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["volume"], 12000.0)

    def test_exright_month_cache_avoids_repeated_provider_requests(self):
        month = dt.date(2025, 12, 1)
        refs = {("2330", dt.date(2025, 12, 11)): 1499.99}
        with (
            patch.object(tw_market_data, "_EXRIGHT_CACHE", {}),
            patch.object(tw_market_data, "_twse_exright_month", return_value=refs) as fetch,
            patch.object(tw_market_data.time, "time", return_value=1000.0),
        ):
            first = tw_market_data._cached_exright_month("TWSE", month)
            second = tw_market_data._cached_exright_month("TWSE", month)
        self.assertEqual(first, refs)
        self.assertEqual(second, refs)
        self.assertEqual(fetch.call_count, 1)

    def test_load_stock_history_accepts_short_listing_history(self):
        start = dt.date(2026, 1, 2)
        rows = []
        previous = 100.0
        for i in range(22):
            close = 100.0 + i
            rows.append({
                "date": start + dt.timedelta(days=i),
                "raw_close": close,
                "change": 0.0 if i == 0 else close - previous,
                "volume": 1000.0,
                "open": close,
                "high": close,
                "low": close,
            })
            previous = close
        with (
            patch.object(tw_market_data, "_STOCK_CACHE", {}),
            patch.object(tw_market_data, "_month_starts", return_value=[dt.date(2026, 1, 1)]),
            patch.object(tw_market_data, "_identify_exchange", return_value=("TWSE", tw_market_data._twse_month)),
            patch.object(tw_market_data, "_fetch_months", return_value=rows),
            patch.object(tw_market_data, "_official_corporate_action_references", return_value={}),
        ):
            frame = tw_market_data.load_stock_history("1234", years=1)
        self.assertEqual(len(frame), 22)
        self.assertEqual(frame.attrs["source"], "TWSE")

    def test_roc_date_accepts_compact_tpex_and_gregorian_dates(self):
        self.assertEqual(tw_market_data._roc_date("1151109"), dt.date(2026, 11, 9))
        self.assertEqual(tw_market_data._roc_date("20261109"), dt.date(2026, 11, 9))

    def test_twse_capital_reduction_reference_parser(self):
        payload = {
            "fields": ["恢復買賣日期", "股票代號", "恢復買賣參考價"],
            "data": [["115/11/09", "4207", "19.74"]],
        }
        with patch.object(tw_market_data, "_request_json", return_value=payload):
            refs = tw_market_data._twse_reduction_month(dt.date(2026, 11, 1))
        self.assertEqual(refs[("4207", dt.date(2026, 11, 9))], 19.74)

    def test_tpex_capital_reduction_reference_parser(self):
        payload = {
            "tables": [{
                "fields": ["恢復買賣日期", "股票代號", "減資恢復買賣開始日參考價格"],
                "data": [["1151109", "4207", "19.74"]],
            }]
        }
        with patch.object(tw_market_data, "_request_json", return_value=payload):
            refs = tw_market_data._tpex_reduction_month(dt.date(2026, 11, 1))
        self.assertEqual(refs[("4207", dt.date(2026, 11, 9))], 19.74)

    def test_twse_par_value_reference_parser(self):
        payload = {
            "fields": ["恢復買賣日期", "股票代號", "恢復買賣參考價"],
            "data": [["115/09/07", "6949", "74.50"]],
        }
        with patch.object(tw_market_data, "_request_json", return_value=payload):
            refs = tw_market_data._twse_par_value_month(dt.date(2026, 9, 1))
        self.assertEqual(refs[("6949", dt.date(2026, 9, 7))], 74.5)

    def test_twse_etf_split_reference_parser(self):
        payload = {
            "fields": ["恢復買賣日期", "ETF代號", "恢復買賣參考價"],
            "data": [["114/06/18", "0050", "47.16"]],
        }
        with patch.object(tw_market_data, "_request_json", return_value=payload):
            refs = tw_market_data._twse_etf_split_month(dt.date(2025, 6, 1))
        self.assertEqual(refs[("0050", dt.date(2025, 6, 18))], 47.16)

    def test_tpex_etf_split_parser_merges_split_and_reverse_split(self):
        split = {("0050", dt.date(2025, 6, 18)): 47.16}
        reverse = {("00632R", dt.date(2024, 12, 11)): 22.96}
        with patch.object(
            tw_market_data, "_tpex_reference_month", side_effect=[split, reverse]
        ):
            refs = tw_market_data._tpex_etf_split_month(dt.date(2025, 6, 1))
        self.assertEqual(refs, {**split, **reverse})

    def test_corporate_action_resolver_falls_back_to_capital_reduction(self):
        day = dt.date(2026, 11, 9)
        rows = [{
            "date": day, "raw_close": 19.80, "change": None,
            "volume": 1000, "open": 19.8, "high": 19.8, "low": 19.8,
        }]
        def fetch(source, action, month):
            if action == "reduction":
                return {("4207", day): 19.74}
            return {}
        with patch.object(tw_market_data, "_cached_action_month", side_effect=fetch) as cached:
            refs = tw_market_data._official_corporate_action_references("4207", "TWSE", rows)
        self.assertEqual(refs, {day: 19.74})
        self.assertEqual([call.args[1] for call in cached.call_args_list], ["exright", "reduction"])

    def test_action_cache_deduplicates_concurrent_provider_fetches(self):
        month = dt.date(2025, 12, 1)
        refs = {("2330", dt.date(2025, 12, 11)): 1499.99}
        calls = []

        def slow_fetch(requested_month):
            calls.append(requested_month)
            import time
            time.sleep(0.05)
            return refs

        with (
            patch.object(tw_market_data, "_EXRIGHT_CACHE", {}),
            patch.object(tw_market_data, "_ACTION_FETCH_LOCKS", {}),
            patch.object(tw_market_data, "_ACTION_ERROR_CACHE", {}),
            patch.object(tw_market_data, "_twse_exright_month", side_effect=slow_fetch),
        ):
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(
                    lambda _: tw_market_data._cached_exright_month("TWSE", month),
                    range(6),
                ))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(result == refs for result in results))

    def test_empty_action_cache_expires_quickly(self):
        month = dt.date(2025, 12, 1)
        clock = iter([
            1000.0, 1000.0, 1000.0,
            1001.0,
            1301.0, 1301.0, 1301.0,
        ])
        with (
            patch.object(tw_market_data, "_EXRIGHT_CACHE", {}),
            patch.object(tw_market_data, "_ACTION_FETCH_LOCKS", {}),
            patch.object(tw_market_data, "_ACTION_ERROR_CACHE", {}),
            patch.object(tw_market_data, "_twse_exright_month", return_value={}) as fetch,
            patch.object(tw_market_data.time, "time", side_effect=lambda: next(clock)),
        ):
            tw_market_data._cached_exright_month("TWSE", month)
            tw_market_data._cached_exright_month("TWSE", month)
            tw_market_data._cached_exright_month("TWSE", month)
        self.assertEqual(fetch.call_count, 2)

    def test_adjustment_failure_has_machine_readable_category(self):
        rows = [
            {"date": dt.date(2026, 1, 2), "raw_close": 100.0, "change": 0.0,
             "volume": 1000, "open": 100, "high": 100, "low": 100},
            {"date": dt.date(2026, 1, 5), "raw_close": 95.0, "change": None,
             "volume": 2000, "open": 95, "high": 95, "low": 95},
        ]
        with self.assertRaises(tw_market_data.MarketDataError) as caught:
            tw_market_data._to_adjusted_frame(rows, "TWSE")
        self.assertEqual(caught.exception.diagnostic_code, "adjustment_reference_missing")


if __name__ == "__main__":
    unittest.main()
