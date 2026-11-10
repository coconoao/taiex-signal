import sys
import datetime as dt
import tempfile
import types
import unittest
import urllib.error
from email.message import Message
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

from pipeline import (
    BACKFILL_DATES_PER_RUN,
    backtest,
    build,
    RECENT_DATES_PER_RUN,
    _checkpoint_if_needed,
    _merge_and_save,
    _open_with_backoff,
    _read_with_backoff,
    _pending_dates,
    fetch_chips,
    fetch_futures,
    _fetch_taifex_openapi_latest,
    _fetch_twse_taiex_recent,
    load_live,
    TICKERS,
    write_taiwan_data_status,
)


class LoadLiveTests(unittest.TestCase):
    def test_yahoo_download_disables_parallel_threads(self):
        import numpy as np

        index = pd.bdate_range(
            end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
            periods=600,
        )
        columns = pd.MultiIndex.from_product([["Close"], list(TICKERS.values())])
        values = np.tile(np.arange(100, 100 + len(index)), (len(TICKERS), 1)).T
        raw = pd.DataFrame(values, index=index, columns=columns)
        fake_yfinance = types.SimpleNamespace(download=Mock(return_value=raw))

        with tempfile.TemporaryDirectory() as directory:
            cache_path = directory + "/market_history.csv"
            with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
                result = load_live(cache_path=cache_path)
            saved = pd.read_csv(cache_path, index_col=0, parse_dates=True)

        self.assertGreaterEqual(len(saved), 500)

        self.assertIn("TAIEX", result.columns)
        self.assertGreaterEqual(len(result), 500)
        self.assertIs(fake_yfinance.download.call_args.kwargs["threads"], False)



    def test_uses_saved_history_when_yahoo_is_unavailable(self):
        import numpy as np

        index = pd.bdate_range(
            end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
            periods=600,
        )
        cached = pd.DataFrame(
            {name: np.arange(100, 100 + len(index), dtype=float)
             for name in TICKERS},
            index=index,
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = directory + "/market_history.csv"
            cached.to_csv(cache_path, index_label="date")
            fake_yfinance = types.SimpleNamespace(
                download=Mock(side_effect=TimeoutError("Yahoo unavailable"))
            )
            with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
                with patch("pipeline.time.sleep"):
                    with patch("pipeline._fetch_twse_taiex_recent", return_value={}):
                        result = load_live(cache_path=cache_path)
            self.assertGreaterEqual(len(result), 500)
            self.assertEqual(len(fake_yfinance.download.call_args_list), 3)
            self.assertEqual(result["TAIEX"].dropna().iloc[-1],
                             cached["TAIEX"].iloc[-1])


class TwseRecentCloseTests(unittest.TestCase):
    def test_checks_newest_trading_day_before_older_fallback_dates(self):
        now = pd.Timestamp("2026-11-09 16:30", tz="Asia/Taipei")
        calls = []

        def fetch(day):
            day = pd.Timestamp(day).normalize()
            calls.append(day)
            if day == pd.Timestamp("2026-11-09"):
                return None  # Today's official close has not been published yet.
            if day == pd.Timestamp("2026-11-06"):
                return 27000.0
            return None

        with patch("pipeline._fetch_twse_taiex_day", side_effect=fetch):
            result = _fetch_twse_taiex_recent(now, calendar_days=7)

        self.assertEqual(calls[0], pd.Timestamp("2026-11-09"))
        self.assertEqual(calls[1], pd.Timestamp("2026-11-06"))
        self.assertEqual(result, {pd.Timestamp("2026-11-06"): 27000.0})


class TaiwanDataStatusTests(unittest.TestCase):
    def test_coverage_uses_taiwan_sessions_not_cross_market_dates(self):
        import json
        import os
        import numpy as np

        dates = pd.bdate_range(
            end=pd.Timestamp.now(tz="Asia/Taipei").normalize().tz_localize(None),
            periods=3,
        )
        market = pd.DataFrame({"TAIEX": [100.0, 101.0, np.nan]}, index=dates)
        chips = pd.DataFrame({"foreign_net": [1.0] * 3, "trust_net": [2.0] * 3}, index=dates)
        margin = pd.DataFrame({
            "margin_bal": [100.0] * 3, "margin_units": [10.0] * 3,
            "short_units": [1.0] * 3,
        }, index=dates)
        futures = pd.DataFrame({"foreign_oi_net": [5.0] * 3}, index=dates)
        fake_storage = types.SimpleNamespace(enabled=lambda: False)

        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                with patch.dict(sys.modules, {"storage": fake_storage}):
                    write_taiwan_data_status(market, chips, margin, futures)
                with open("taiwan_data_status.json", encoding="utf-8") as stream:
                    status = json.load(stream)
            finally:
                os.chdir(previous)

        for layer in ("institutional", "margin", "futures"):
            self.assertEqual(status["layers"][layer]["expected_rows"], 2)
            self.assertEqual(status["layers"][layer]["coverage_pct"], 100.0)
            self.assertEqual(status["layers"][layer]["missing_rows"], 0)
        self.assertEqual(status["layers"]["futures"]["source_window_start"], "2018-06-05")
        self.assertEqual(status["layers"]["futures"]["source_window_expected_rows"], 2)


class PendingDatesTests(unittest.TestCase):
    def test_limits_backfill_and_keeps_latest_missing_dates(self):
        today = pd.Timestamp("2026-11-06")
        trading_days = pd.bdate_range(end=today, periods=250)
        result = _pending_dates(pd.DataFrame(), today, 500, trading_days)
        self.assertEqual(len(result), BACKFILL_DATES_PER_RUN + RECENT_DATES_PER_RUN)
        self.assertEqual(result[:RECENT_DATES_PER_RUN].tolist(), trading_days[-RECENT_DATES_PER_RUN:].tolist())
        expected_backfill = trading_days[-(BACKFILL_DATES_PER_RUN + RECENT_DATES_PER_RUN):-RECENT_DATES_PER_RUN]
        self.assertEqual(result[RECENT_DATES_PER_RUN:].tolist(), expected_backfill.tolist())

    def test_excludes_dates_already_saved_and_fills_oldest_gaps_next(self):
        today = pd.Timestamp("2026-11-06")
        trading_days = pd.bdate_range(end=today, periods=100)
        saved = pd.DataFrame(index=trading_days[:40])
        result = _pending_dates(saved, today, 365, trading_days)
        self.assertEqual(len(result), 60)  # only 60 unsaved dates remain
        self.assertEqual(result[:RECENT_DATES_PER_RUN].tolist(), trading_days[-RECENT_DATES_PER_RUN:].tolist())
        self.assertEqual(result[RECENT_DATES_PER_RUN], trading_days[40])
        self.assertFalse(result.isin(saved.index).any())

    def test_returns_all_dates_when_only_five_or_fewer_are_missing(self):
        today = pd.Timestamp("2026-11-06")
        trading_days = pd.bdate_range(end=today, periods=5)
        result = _pending_dates(pd.DataFrame(), today, 365, trading_days)
        self.assertEqual(result.tolist(), trading_days.tolist())

    def test_respects_days_back_window(self):
        today = pd.Timestamp("2026-11-06")
        trading_days = pd.bdate_range(end=today, periods=100)
        result = _pending_dates(pd.DataFrame(), today, 10, trading_days)
        self.assertTrue((result >= today - pd.Timedelta(days=10)).all())
        self.assertTrue((result <= today).all())


class RetryTests(unittest.TestCase):
    @staticmethod
    def rate_limit_error():
        headers = Message()
        headers["Retry-After"] = "2"
        return urllib.error.HTTPError(
            "https://example.test/data", 429, "Too Many Requests", headers, None
        )

    def test_retries_rate_limit_and_honors_retry_after(self):
        response = Mock()
        with patch("pipeline.urllib.request.urlopen",
                   side_effect=[self.rate_limit_error(), response]) as urlopen:
            with patch("pipeline.time.sleep") as sleep:
                result = _open_with_backoff(Mock(), timeout=10)
        self.assertIs(result, response)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2.0)

    def test_stops_after_four_rate_limit_responses(self):
        error = self.rate_limit_error()
        with patch("pipeline.urllib.request.urlopen", side_effect=[error] * 4):
            with patch("pipeline.time.sleep") as sleep:
                with self.assertRaises(urllib.error.HTTPError):
                    _open_with_backoff(Mock(), timeout=10)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2.0, 2.0, 2.0])

    def test_retries_timeout_while_reading_response_body(self):
        import io

        class FirstReadTimesOut(io.BytesIO):
            def read(self, *args, **kwargs):
                raise TimeoutError("The read operation timed out")

        first = FirstReadTimesOut(b"")
        second = io.BytesIO(b'{"ok": true}')
        response1 = MagicMock()
        response1.__enter__.return_value = first
        response2 = MagicMock()
        response2.__enter__.return_value = second
        with patch("pipeline.urllib.request.urlopen",
                   side_effect=[response1, response2]) as open_url:
            with patch("pipeline.time.sleep") as sleep:
                result = _read_with_backoff(Mock(), timeout=10)
        self.assertEqual(result, b'{"ok": true}')
        self.assertEqual(open_url.call_count, 2)
        sleep.assert_called_once_with(5)


class TaifexOpenApiTests(unittest.TestCase):
    def test_extracts_only_taiwan_futures_foreign_open_interest_net(self):
        import json

        payload = [
            {
                "Date": "20261106",
                "ContractCode": "臺股期貨",
                "Item": "外資及陸資",
                "OpenInterest(Net)": "-12345",
            },
            {
                "Date": "20261106",
                "ContractCode": "電子期貨",
                "Item": "外資及陸資",
                "OpenInterest(Net)": "999",
            },
        ]
        with patch("pipeline._read_with_backoff", return_value=json.dumps(payload).encode()):
            date, net = _fetch_taifex_openapi_latest()

        self.assertEqual(date, pd.Timestamp("2026-11-06"))
        self.assertEqual(net, -12345.0)

    def test_rejects_missing_taiwan_futures_foreign_row(self):
        import json

        payload = [{"Date": "20261106", "ContractCode": "電子期貨",
                    "Item": "外資及陸資", "OpenInterest(Net)": "10"}]
        with patch("pipeline._read_with_backoff", return_value=json.dumps(payload).encode()):
            with self.assertRaisesRegex(ValueError, "找不到"):
                _fetch_taifex_openapi_latest()


class FuturesNetworkFailureTests(unittest.TestCase):
    def test_stops_batch_after_persistent_network_timeout(self):
        import os

        dates = pd.bdate_range("2026-11-02", periods=3)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "futures.csv")
            with patch("pipeline._pending_dates", return_value=dates):
                with patch("pipeline._fetch_taifex_openapi_latest",
                           side_effect=ValueError("OpenAPI unavailable")):
                    with patch("pipeline._read_with_backoff",
                               side_effect=TimeoutError("source timeout")) as read:
                        with patch("pipeline._fetch_finmind_data",
                                   side_effect=TimeoutError("backup unavailable")):
                            with patch("pipeline.time.sleep"):
                                result = fetch_futures(path=path)
        self.assertEqual(read.call_count, 1)
        self.assertTrue(result.empty)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_merges_and_persists_after_five_rows(self):
        old = pd.DataFrame({"value": [1]}, index=[pd.Timestamp("2026-01-01")])
        rows = {
            pd.Timestamp("2026-01-02") + pd.Timedelta(days=i): {"value": i + 2}
            for i in range(5)
        }
        fake_storage = types.SimpleNamespace(
            enabled=lambda: True,
            save_data=Mock(return_value=4),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/chips.csv"
            with patch.dict(sys.modules, {"storage": fake_storage}):
                _checkpoint_if_needed(old, rows, path)
            saved = pd.read_csv(path, index_col=0, parse_dates=True)
        self.assertEqual(len(saved), 6)
        self.assertTrue(saved.index.equals(saved.index.sort_values()))
        fake_storage.save_data.assert_called_once_with()
        self.assertFalse(__import__("os").path.exists(path + ".tmp"))

    def test_merge_saves_local_data_without_supabase(self):
        rows = {pd.Timestamp("2026-01-02"): {"value": 2}}
        fake_storage = types.SimpleNamespace(enabled=lambda: False)
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/data.csv"
            with patch.dict(sys.modules, {"storage": fake_storage}):
                result = _merge_and_save(pd.DataFrame(), rows, path)
            self.assertTrue(__import__("os").path.exists(path))
            self.assertFalse(__import__("os").path.exists(path + ".tmp"))
        self.assertEqual(len(result), 1)


class FetchCheckpointIntegrationTests(unittest.TestCase):
    def test_fetch_chips_persists_checkpoint_during_real_fetch_loop(self):
        import io
        import json
        import os

        dates = pd.bdate_range("2026-10-26", periods=6)
        payload = {
            "stat": "OK",
            "data": [
                ["外資及陸資買賣超", "", "", "100"],
                ["投信買賣超", "", "", "50"],
            ],
        }
        responses = [io.BytesIO(json.dumps(payload).encode("utf-8")) for _ in dates]
        observed_rows_at_save = []
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "chips.csv")

            def record_save():
                count = len(pd.read_csv(path, index_col=0)) if os.path.exists(path) else 0
                observed_rows_at_save.append(count)
                return 8

            fake_storage = types.SimpleNamespace(
                enabled=lambda: True,
                save_data=Mock(side_effect=record_save),
            )
            with patch("pipeline._pending_dates", return_value=dates):
                with patch("pipeline.urllib.request.urlopen", side_effect=responses):
                    with patch("pipeline.time.sleep"):
                        with patch.dict(sys.modules, {"storage": fake_storage}):
                            result = fetch_chips(path=path)
            saved = pd.read_csv(path, index_col=0, parse_dates=True)

        self.assertEqual(len(result), 6)
        self.assertEqual(len(saved), 6)
        self.assertEqual(observed_rows_at_save, [5, 6])


class BacktestCalendarTests(unittest.TestCase):
    def test_sparse_predictions_use_full_daily_calendar_for_performance(self):
        import numpy as np

        dates = pd.bdate_range("2026-01-05", periods=20)
        forward_returns = pd.Series(np.full(len(dates), 0.001), index=dates)
        # Only two model predictions span a 20-session backtest interval.
        probabilities = pd.Series([0.70, 0.70], index=[dates[0], dates[-1]])

        result = backtest(probabilities, forward_returns)

        self.assertEqual(result["days"], 2)
        self.assertEqual(result["prediction_days"], 2)
        self.assertEqual(result["period_sessions"], 20)
        self.assertEqual(result["period_start"], str(dates[0].date()))
        self.assertEqual(result["period_end"], str(dates[-1].date()))
        self.assertEqual(len(result["curve"]["dates"]), 4)
        self.assertEqual(result["curve"]["dates"][0], str(dates[0].date()))
        self.assertEqual(result["curve"]["dates"][-1], str(dates[15].date()))


class DelayedMarginPublicationTests(unittest.TestCase):
    def test_latest_session_remains_predictable_when_margin_release_is_delayed(self):
        import numpy as np

        index = pd.bdate_range("2025-01-01", periods=260)
        market = pd.DataFrame(index=index)
        for offset, name in enumerate(["SOX", "TSM", "VIX", "US10Y", "N225", "KOSPI", "USDTWD", "TAIEX"]):
            market[name] = 100 + np.arange(len(index)) * (0.08 + offset * 0.003)
        chips = pd.DataFrame({
            "foreign_net": np.linspace(-100, 150, len(index)),
            "trust_net": np.linspace(80, -40, len(index)),
        }, index=index)
        margin = pd.DataFrame({
            "margin_bal": np.linspace(1000, 1200, len(index) - 1),
            "margin_units": np.linspace(500, 550, len(index) - 1),
            "short_units": np.linspace(100, 130, len(index) - 1),
        }, index=index[:-1])

        X, y, _ = build(market, chips=chips, margin=margin)

        self.assertEqual(X.index[-1], index[-1])
        self.assertTrue(X.loc[index[-1]].notna().all())
        self.assertTrue(pd.isna(y.loc[index[-1]]))



class ForecastTradingCalendarTests(unittest.TestCase):
    def test_parses_iso_and_roc_dates(self):
        from pipeline import _parse_twse_calendar_date
        self.assertEqual(_parse_twse_calendar_date("2026-11-11"), dt.date(2026, 11, 11))
        self.assertEqual(_parse_twse_calendar_date("1151109"), dt.date(2026, 11, 9))

    def test_uses_official_closures_when_assigning_forecast_dates(self):
        from pipeline import _future_taiwan_trading_dates
        with patch("pipeline._fetch_twse_closed_dates", side_effect=[
            frozenset({dt.date(2026, 11, 11)}),
        ]):
            dates, source = _future_taiwan_trading_dates("2026-11-10", 3)
        self.assertEqual(dates, ["2026-11-12", "2026-11-13", "2026-11-16"])
        self.assertEqual(source, "twse_official")

    def test_marks_weekday_only_fallback_when_official_calendar_fails(self):
        from pipeline import _future_taiwan_trading_dates
        with patch("pipeline._fetch_twse_closed_dates", return_value=None):
            dates, source = _future_taiwan_trading_dates("2026-11-10", 3)
        self.assertEqual(dates, ["2026-11-11", "2026-11-12", "2026-11-13"])
        self.assertEqual(source, "weekdays_only_fallback")


if __name__ == "__main__":
    unittest.main()
