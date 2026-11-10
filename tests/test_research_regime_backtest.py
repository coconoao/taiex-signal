import unittest
from unittest.mock import patch

import pandas as pd

from research_regime_backtest import _validate_taiwan_calendar


class RealMarketCalendarValidationTests(unittest.TestCase):
    def setUp(self):
        self.expected = pd.DatetimeIndex([
            "2026-11-09", "2026-11-10", "2026-11-12", "2026-11-13"
        ])

    def test_passes_when_observed_sessions_match_official_calendar(self):
        with patch(
            "research_regime_backtest.pipeline._expected_twse_trading_days",
            return_value=(self.expected, "twse_official_holiday_schedule"),
        ):
            result = _validate_taiwan_calendar(
                self.expected, self.expected[0], self.expected[-1]
            )
        self.assertEqual(result["status"], "available")
        self.assertTrue(result["passed"])
        self.assertEqual(result["missing_count"], 0)
        self.assertEqual(result["unexpected_count"], 0)

    def test_fails_and_reports_missing_official_session(self):
        observed = self.expected.delete(2)
        with patch(
            "research_regime_backtest.pipeline._expected_twse_trading_days",
            return_value=(self.expected, "twse_official_holiday_schedule"),
        ):
            result = _validate_taiwan_calendar(
                observed, self.expected[0], self.expected[-1]
            )
        self.assertEqual(result["status"], "available")
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_dates"], ["2026-11-12"])

    def test_unavailable_official_calendar_never_passes(self):
        with patch(
            "research_regime_backtest.pipeline._expected_twse_trading_days",
            return_value=(pd.DatetimeIndex([]), "unavailable"),
        ):
            result = _validate_taiwan_calendar(
                self.expected, self.expected[0], self.expected[-1]
            )
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["passed"])
        self.assertIsNone(result["missing_count"])


if __name__ == "__main__":
    unittest.main()
