import sys
import tempfile
import types
import unittest
import urllib.error
from email.message import Message
from unittest.mock import Mock, patch

import pandas as pd

from pipeline import (
    BACKFILL_DATES_PER_RUN,
    RECENT_DATES_PER_RUN,
    _checkpoint_if_needed,
    _merge_and_save,
    _open_with_backoff,
    _read_with_backoff,
    _pending_dates,
    fetch_chips,
)


class PendingDatesTests(unittest.TestCase):
    def test_limits_backfill_and_keeps_latest_missing_dates(self):
        today = pd.Timestamp("2026-11-06")
        trading_days = pd.bdate_range(end=today, periods=120)
        result = _pending_dates(pd.DataFrame(), today, 365, trading_days)
        self.assertEqual(len(result), BACKFILL_DATES_PER_RUN + RECENT_DATES_PER_RUN)
        self.assertEqual(result[:BACKFILL_DATES_PER_RUN].tolist(), trading_days[:BACKFILL_DATES_PER_RUN].tolist())
        self.assertEqual(result[-RECENT_DATES_PER_RUN:].tolist(), trading_days[-RECENT_DATES_PER_RUN:].tolist())

    def test_excludes_dates_already_saved_and_fills_oldest_gaps_next(self):
        today = pd.Timestamp("2026-11-06")
        trading_days = pd.bdate_range(end=today, periods=100)
        saved = pd.DataFrame(index=trading_days[:40])
        result = _pending_dates(saved, today, 365, trading_days)
        self.assertEqual(len(result), BACKFILL_DATES_PER_RUN + RECENT_DATES_PER_RUN)
        self.assertEqual(result[0], trading_days[40])
        self.assertEqual(result[-RECENT_DATES_PER_RUN:].tolist(), trading_days[-RECENT_DATES_PER_RUN:].tolist())
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
        with patch("pipeline._open_with_backoff", side_effect=[first, second]) as open_url:
            with patch("pipeline.time.sleep") as sleep:
                result = _read_with_backoff(Mock(), timeout=10)
        self.assertEqual(result, b'{"ok": true}')
        self.assertEqual(open_url.call_count, 2)
        sleep.assert_called_once_with(5)


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
                with patch("pipeline._open_with_backoff", side_effect=responses):
                    with patch("pipeline.time.sleep"):
                        with patch.dict(sys.modules, {"storage": fake_storage}):
                            result = fetch_chips(path=path)
            saved = pd.read_csv(path, index_col=0, parse_dates=True)

        self.assertEqual(len(result), 6)
        self.assertEqual(len(saved), 6)
        self.assertEqual(observed_rows_at_save, [5, 6])


if __name__ == "__main__":
    unittest.main()
