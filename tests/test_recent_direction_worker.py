import unittest

import numpy as np
import pandas as pd

import stock_model_research as research


class RecentDirectionWorkerTests(unittest.TestCase):
    def test_uses_prelaunch_history_as_valid_sessions(self):
        # Historical prices intentionally predate the website launch.
        index = pd.bdate_range("2020-01-02", periods=150)
        close = pd.Series(100 * np.exp(np.linspace(0, 0.2, len(index))), index=index)
        market = pd.Series(15000 * np.exp(np.linspace(0, 0.1, len(index))), index=index)

        result = research._recent_direction_signal(close, market, 21)

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["sessions_used"], 150)
        self.assertEqual(result["asof"], str(index[-1].date()))
        self.assertEqual(result["minimum_sessions"], 22)
        self.assertIsNotNone(result["signal"])

    def test_longer_horizon_reports_real_history_count(self):
        index = pd.bdate_range("2021-01-04", periods=80)
        close = pd.Series(np.linspace(50, 80, len(index)), index=index)
        market = pd.Series(np.linspace(14000, 16000, len(index)), index=index)

        result = research._recent_direction_signal(close, market, 63)

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["sessions_used"], 80)
        self.assertEqual(result["minimum_sessions"], 64)


if __name__ == "__main__":
    unittest.main()
