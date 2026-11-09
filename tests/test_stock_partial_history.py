import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import stock


class PartialListingHistoryTests(unittest.TestCase):
    def _run_analysis(self, sessions):
        index = pd.bdate_range("2026-10-01", periods=sessions)
        close = pd.Series(np.linspace(50.0, 60.0, sessions), index=index)
        frame = pd.DataFrame({
            "Close": close,
            "RawClose": close,
            "Volume": pd.Series(np.full(sessions, 1000.0), index=index),
        }, index=index)
        frame.attrs["source"] = "TWSE"
        market = pd.Series(np.linspace(20000.0, 20500.0, sessions), index=index)
        with (
            patch("tw_market_data.load_stock_history", return_value=frame),
            patch("tw_market_data.load_taiex_history", return_value=market),
            patch("stock_model_research.analyze", return_value={
                "horizons": {}, "recent_signals": {}
            }),
            patch("os.path.exists", return_value=False),
        ):
            return stock.analyze("1234")

    def test_short_history_returns_data_without_failing_beta(self):
        result = self._run_analysis(25)
        self.assertEqual(result["history_sessions"], 25)
        self.assertIsNone(result["beta"])
        self.assertIsNotNone(result["mom20"])
        self.assertIsNotNone(result["vol60"])

    def test_very_short_history_marks_longer_metrics_unavailable(self):
        result = self._run_analysis(10)
        self.assertEqual(result["history_sessions"], 10)
        self.assertIsNone(result["beta"])
        self.assertIsNone(result["mom20"])
        self.assertIsNotNone(result["vol60"])


if __name__ == "__main__":
    unittest.main()
