import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import research_regime_backtest as backtest


class OfficialCalendarGateTests(unittest.TestCase):
    def test_official_calendar_passes_only_on_exact_match(self):
        expected = pd.date_range("2026-01-05", periods=3, freq="B")
        with patch.object(backtest.pipeline, "_expected_twse_trading_days",
                          return_value=(expected, "twse_official_holiday_schedule")):
            result = backtest._validate_taiwan_calendar(expected)
        self.assertTrue(result["passed"])
        self.assertEqual(result["missing_count"], 0)
        self.assertEqual(result["unexpected_count"], 0)

    def test_missing_official_session_fails(self):
        expected = pd.date_range("2026-01-05", periods=3, freq="B")
        with patch.object(backtest.pipeline, "_expected_twse_trading_days",
                          return_value=(expected, "twse_official_holiday_schedule")):
            result = backtest._validate_taiwan_calendar(expected.delete(1))
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_count"], 1)

    def test_unavailable_official_calendar_fails_closed(self):
        observed = pd.date_range("2026-01-05", periods=3, freq="B")
        with patch.object(backtest.pipeline, "_expected_twse_trading_days",
                          return_value=(pd.DatetimeIndex([]), "unavailable")):
            result = backtest._validate_taiwan_calendar(observed)
        self.assertFalse(result["passed"])
        self.assertEqual(result["source"], "unavailable")
        self.assertEqual(result["expected_sessions"], 0)

    def test_official_openapi_bare_list_with_roc_dates_is_parsed(self):
        payload = [
            {"Name": "中華民國開國紀念日", "Date": "1150101",
             "Weekday": "四", "Description": "依規定放假1日。"},
            {"Name": "國曆新年開始交易日", "Date": "1150102",
             "Weekday": "五", "Description": "國曆新年開始交易。"},
            {"Name": "農曆春節", "Date": "1150218",
             "Weekday": "三", "Description": "依規定放假1日。"},
            {"Name": "農曆春節後開始交易日", "Date": "1150223",
             "Weekday": "一", "Description": "農曆春節後開始交易。"},
            {"Name": "和平紀念日", "Date": "1150227",
             "Weekday": "五", "Description": "依規定放假1日。"},
            {"Name": "兒童節及民族掃墓節", "Date": "1150403",
             "Weekday": "五", "Description": "依規定放假1日。"},
            {"Name": "勞動節", "Date": "1150501",
             "Weekday": "五", "Description": "依規定放假1日。"},
            {"Name": "端午節", "Date": "1150619",
             "Weekday": "五", "Description": "依規定放假1日。"},
            {"Name": "中秋節", "Date": "1150925",
             "Weekday": "五", "Description": "依規定放假1日。"},
            {"Name": "國慶日", "Date": "1151009",
             "Weekday": "五", "Description": "依規定放假1日。"},
            {"Name": "臺灣光復紀念日", "Date": "1151025",
             "Weekday": "日", "Description": "依規定放假1日。"},
            {"Name": "行憲紀念日", "Date": "1151225",
             "Weekday": "五", "Description": "依規定放假1日。"},
        ]
        with patch.object(backtest.pipeline, "_read_with_backoff",
                          return_value=__import__("json").dumps(payload).encode("utf-8")):
            with patch.object(backtest.pipeline.pd.Timestamp, "now",
                              return_value=pd.Timestamp("2026-11-11", tz="Asia/Taipei")):
                backtest.pipeline._TWSE_CLOSED_DATES_CACHE.clear()
                result = backtest.pipeline._fetch_twse_closed_dates(2026)
        self.assertEqual(result, frozenset({
            pd.Timestamp("2026-01-01").date(),
            pd.Timestamp("2026-02-18").date(),
            pd.Timestamp("2026-02-27").date(),
            pd.Timestamp("2026-04-03").date(),
            pd.Timestamp("2026-05-01").date(),
            pd.Timestamp("2026-06-19").date(),
            pd.Timestamp("2026-09-25").date(),
            pd.Timestamp("2026-10-09").date(),
            pd.Timestamp("2026-12-25").date(),
        }))

    def test_rwd_uses_full_date_parameter_and_returns_historical_year(self):
        payload = {
            "stat": "ok",
            "date": "20250101",
            "title": "114 年市場開休市日期",
            "fields": ["日期", "名稱", "說明"],
            "data": [
                ["2025-01-01", "中華民國開國紀念日", "依規定放假1日。"],
                ["2025-01-02", "國曆新年開始交易日", "國曆新年開始交易。"],
                ["2025-02-03", "農曆春節後開始交易日", "農曆春節後開始交易。"],
                ["2025-02-28", "和平紀念日", "依規定放假1日。"],
                ["2025-04-04", "兒童節及民族掃墓節", "依規定放假1日。"],
                ["2025-05-01", "勞動節", "依規定放假1日。"],
                ["2025-05-30", "端午節", "依規定放假1日。"],
                ["2025-06-03", "端午節後開始交易日", "端午節後開始交易。"],
                ["2025-09-29", "中秋節", "依規定放假1日。"],
                ["2025-10-10", "國慶日", "依規定放假1日。"],
                ["2025-12-25", "行憲紀念日", "依規定放假1日。"],
                ["2025-12-26", "年末交易日", "市場交易。"],
            ],
            "queryYear": 2025,
            "total": 12,
        }
        requested = []
        def fake_read(req, timeout=15):
            requested.append(req.full_url)
            return __import__("json").dumps(payload, ensure_ascii=False).encode("utf-8")
        with patch.object(backtest.pipeline, "_read_with_backoff", side_effect=fake_read):
            backtest.pipeline._TWSE_CLOSED_DATES_CACHE.clear()
            result = backtest.pipeline._fetch_twse_closed_dates(2025)
        self.assertIn("date=20250101", requested[0])
        self.assertEqual(result, frozenset({
            pd.Timestamp("2025-01-01").date(),
            pd.Timestamp("2025-02-28").date(),
            pd.Timestamp("2025-04-04").date(),
            pd.Timestamp("2025-05-01").date(),
            pd.Timestamp("2025-05-30").date(),
            pd.Timestamp("2025-09-29").date(),
            pd.Timestamp("2025-10-10").date(),
            pd.Timestamp("2025-12-25").date(),
        }))



    def test_valid_annual_schedule_may_have_no_q3_closure(self):
        payload = {
            "stat": "ok",
            "queryYear": 2025,
            "total": 12,
            "fields": ["日期", "名稱", "說明"],
            "data": [
                ["2025-01-01", "中華民國開國紀念日", "依規定放假1日。"],
                ["2025-01-02", "國曆新年開始交易日", "國曆新年開始交易。"],
                ["2025-02-03", "農曆春節後開始交易日", "農曆春節後開始交易。"],
                ["2025-02-27", "和平紀念日", "依規定放假1日。"],
                ["2025-02-28", "和平紀念日", "依規定放假1日。"],
                ["2025-04-04", "兒童節", "依規定放假1日。"],
                ["2025-05-01", "勞動節", "依規定放假1日。"],
                ["2025-05-30", "端午節", "依規定放假1日。"],
                ["2025-06-03", "端午節後開始交易日", "端午節後開始交易。"],
                ["2025-10-10", "國慶日", "依規定放假1日。"],
                ["2025-10-13", "國慶日後開始交易日", "國慶日後開始交易。"],
                ["2025-12-25", "行憲紀念日", "依規定放假1日。"],
            ],
        }
        with patch.object(backtest.pipeline, "_read_with_backoff",
                          return_value=__import__("json").dumps(payload).encode("utf-8")):
            backtest.pipeline._TWSE_CLOSED_DATES_CACHE.clear()
            result = backtest.pipeline._fetch_twse_closed_dates(2025)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 8)

    def test_incomplete_official_schedule_is_rejected(self):
        payload = {
            "stat": "ok",
            "queryYear": 2025,
            "total": 2,
            "fields": ["日期", "名稱", "說明"],
            "data": [
                ["2025-01-01", "中華民國開國紀念日", "依規定放假1日。"],
                ["2025-01-02", "國曆新年開始交易日", "國曆新年開始交易。"],
            ],
        }
        with patch.object(backtest.pipeline, "_read_with_backoff",
                          return_value=__import__("json").dumps(payload).encode("utf-8")):
            backtest.pipeline._TWSE_CLOSED_DATES_CACHE.clear()
            self.assertIsNone(backtest.pipeline._fetch_twse_closed_dates(2025))

    def test_declared_total_mismatch_is_rejected(self):
        payload = {
            "stat": "ok",
            "queryYear": 2025,
            "total": 99,
            "fields": ["日期", "名稱", "說明"],
            "data": [["2025-01-01", "中華民國開國紀念日", "依規定放假1日。"]],
        }
        with patch.object(backtest.pipeline, "_read_with_backoff",
                          return_value=__import__("json").dumps(payload).encode("utf-8")):
            backtest.pipeline._TWSE_CLOSED_DATES_CACHE.clear()
            self.assertIsNone(backtest.pipeline._fetch_twse_closed_dates(2025))

    def test_latest_year_response_is_not_accepted_for_historical_year(self):
        payload = [
            {"Name": "中華民國開國紀念日", "Date": "1150101",
             "Weekday": "四", "Description": "依規定放假1日。"},
        ]
        with patch.object(backtest.pipeline, "_read_with_backoff",
                          return_value=__import__("json").dumps(payload).encode("utf-8")):
            backtest.pipeline._TWSE_CLOSED_DATES_CACHE.clear()
            result = backtest.pipeline._fetch_twse_closed_dates(2025)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
