import unittest

import numpy as np
import pandas as pd

from stock_model_research import analyze


class FourYearResearchFeasibilityTests(unittest.TestCase):
    def test_four_year_history_supports_all_three_research_horizons(self):
        # A conservative four-year approximation of Taiwan trading sessions, without any live provider
        # calls. This exercises the actual walk-forward and calibration pipeline.
        dates = pd.bdate_range("2022-01-03", periods=980)
        rng = np.random.default_rng(20261108)
        stock_returns = rng.normal(0.00025, 0.012, len(dates))
        market_returns = rng.normal(0.00018, 0.009, len(dates))
        stock_close = pd.Series(
            100 * np.cumprod(1 + stock_returns), index=dates, dtype=float
        )
        market_close = pd.Series(
            15000 * np.cumprod(1 + market_returns), index=dates, dtype=float
        )
        volume = pd.Series(
            rng.lognormal(mean=12.0, sigma=0.25, size=len(dates)),
            index=dates,
            dtype=float,
        )

        result = analyze(stock_close, market_close, volume)
        self.assertEqual(result["history_sessions"], 980)
        self.assertEqual(set(result["horizons"]), {"1m", "3m", "6m"})

        for horizon in ("1m", "3m", "6m"):
            with self.subTest(horizon=horizon):
                estimate = result["horizons"][horizon]
                self.assertIn(estimate["status"], ("available", "uncalibrated"))
                self.assertGreaterEqual(estimate["training_samples"], 500)
                holdout = estimate["untouched_holdout"]
                self.assertIsNotNone(holdout)
                self.assertGreaterEqual(holdout["sample_count"], 20)

        # Six months is the limiting horizon: it must still produce enough
        # matured out-of-sample predictions for the existing calibrator.
        self.assertEqual(result["horizons"]["6m"]["status"], "available")


if __name__ == "__main__":
    unittest.main()
