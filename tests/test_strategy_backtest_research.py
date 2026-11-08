import unittest
import numpy as np
import pandas as pd

from strategy_backtest_research import simulate_profit_target, walk_forward_probabilities


class StrategyBacktestResearchTests(unittest.TestCase):
    def test_profit_target_enters_next_close_and_exits_at_threshold(self):
        dates = pd.bdate_range("2024-01-01", periods=41)
        close = pd.Series(100.0, index=dates)
        close.iloc[12] = 102.0
        prob = pd.Series(0.7, index=dates[10:40])
        result = simulate_profit_target(prob, close, 0.02)
        self.assertEqual(result["status"], "available")
        self.assertGreater(result["trade_count"], 0)
        self.assertEqual(result["trades"][0]["entry_date"], str(dates[11].date()))
        self.assertEqual(result["trades"][0]["exit_date"], str(dates[12].date()))
        self.assertEqual(result["trades"][0]["exit_reason"], "達到報酬門檻")

    def test_profit_target_requires_next_session_price(self):
        dates = pd.bdate_range("2024-01-01", periods=36)
        close = pd.Series(np.linspace(100, 110, 35), index=dates)
        prob = pd.Series(0.7, index=dates[5:35])
        result = simulate_profit_target(prob, close, 0.01)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["period_start"], str(dates[6].date()))

if __name__ == "__main__":
    unittest.main()
