import unittest

import numpy as np
import pandas as pd

import stock_model_research as research


def synthetic_data(n=900):
    index = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(7)
    stock = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))), index=index)
    market = pd.Series(15000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))), index=index)
    volume = pd.Series(rng.integers(100_000, 1_000_000, n), index=index)
    return stock, market, volume


class StockModelResearchTests(unittest.TestCase):
    def test_medium_long_horizons_are_defined_in_trading_sessions(self):
        self.assertEqual(research.HORIZONS, {"1m": 21, "3m": 63, "6m": 126})

    def test_features_are_point_in_time_and_do_not_use_future_prices(self):
        stock, market, volume = synthetic_data()
        cutoff = stock.index[650]
        original = research.build_features(stock, market, volume)

        changed_stock = stock.copy()
        changed_market = market.copy()
        changed_volume = volume.copy()
        changed_stock.loc[changed_stock.index > cutoff] *= 1.8
        changed_market.loc[changed_market.index > cutoff] *= 0.7
        changed_volume.loc[changed_volume.index > cutoff] *= 2.0
        changed = research.build_features(changed_stock, changed_market, changed_volume)

        pd.testing.assert_series_equal(
            original.loc[cutoff], changed.loc[cutoff], check_names=False
        )


    def test_block_bootstrap_accuracy_interval_is_reproducible(self):
        rng = np.random.default_rng(12)
        actual = rng.integers(0, 2, 240)
        probability = np.where(actual == 1, 0.6, 0.4)
        first = research._block_bootstrap_accuracy_ci(probability, actual, 21)
        second = research._block_bootstrap_accuracy_ci(probability, actual, 21)
        self.assertEqual(first, second)
        self.assertLessEqual(first["lower_95_pct"], first["upper_95_pct"])
        self.assertEqual(first["method"], "moving-block bootstrap")

    def test_block_bootstrap_with_small_sample_returns_no_interval(self):
        actual = np.array([0, 1] * 20)
        probability = np.full(40, 0.5)
        self.assertIsNone(
            research._block_bootstrap_accuracy_ci(probability, actual, 21)
        )

    def test_feature_builder_includes_price_volume_and_market_relative_inputs(self):
        stock, market, volume = synthetic_data()
        features = research.build_features(stock, market, volume)
        self.assertIn("volume_ratio_20", features)
        self.assertIn("relative_return_126", features)
        self.assertIn("stock_volatility_126", features)
        self.assertGreater(int(features.notna().any(axis=1).sum()), 500)


if __name__ == "__main__":
    unittest.main()
