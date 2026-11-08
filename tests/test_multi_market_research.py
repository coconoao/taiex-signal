import unittest

import numpy as np
import pandas as pd

from multi_market_research import MARKETS, make_dataset, run_research


class MultiMarketResearchTests(unittest.TestCase):
    def test_cross_market_features_are_lagged(self):
        dates = pd.bdate_range("2020-01-01", periods=40)
        prices = pd.DataFrame({
            key: 100 * np.cumprod(1 + np.linspace(-0.002, 0.003, len(dates)))
            for key in MARKETS
        }, index=dates)
        X, y, forward = make_dataset(prices, "SP500")
        expected = prices["NIKKEI225"].pct_change().shift(1).reindex(X.index)
        pd.testing.assert_series_equal(
            X["NIKKEI225_return_lag1"], expected.loc[X.index],
            check_names=False,
        )
        self.assertTrue(y.notna().all())
        self.assertTrue(forward.notna().all())

    def test_insufficient_history_is_not_scored(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        prices = pd.DataFrame({
            key: np.arange(100, 100 + len(dates), dtype=float)
            for key in MARKETS
        }, index=dates)
        result = run_research(prices)
        self.assertEqual(result["markets"]["SP500"]["status"], "insufficient")
        self.assertNotIn("direction_accuracy_pct", result["markets"]["SP500"])


if __name__ == "__main__":
    unittest.main()
