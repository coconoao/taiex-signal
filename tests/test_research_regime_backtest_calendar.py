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


if __name__ == "__main__":
    unittest.main()
