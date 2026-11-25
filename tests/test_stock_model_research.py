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

    def test_recent_signal_is_available_with_short_history_before_model_training_threshold(self):
        index = pd.bdate_range("2026-01-05", periods=50)
        stock = pd.Series(100 * np.exp(np.arange(50) * 0.003), index=index)
        market = pd.Series(15000 * np.exp(np.arange(50) * 0.0002), index=index)

        result = research._recent_direction_signal(stock, market, 21)

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["asof"], str(index[-1].date()))
        self.assertEqual(result["signal"], "偏多")
        self.assertIsNone(result.get("probability"))
        self.assertGreaterEqual(result["sessions_used"], result["minimum_sessions"])

    def test_recent_signal_respects_horizon_specific_minimum_history(self):
        index = pd.bdate_range("2026-01-05", periods=50)
        stock = pd.Series(np.linspace(100, 110, 50), index=index)
        market = pd.Series(np.linspace(15000, 15100, 50), index=index)

        one_month = research._recent_direction_signal(stock, market, 21)
        three_months = research._recent_direction_signal(stock, market, 63)
        six_months = research._recent_direction_signal(stock, market, 126)

        self.assertEqual(one_month["status"], "available")
        self.assertEqual(three_months["status"], "insufficient")
        self.assertEqual(six_months["status"], "insufficient")
        self.assertIsNone(three_months["signal"])

    def test_recent_signal_uses_only_observations_through_asof_date(self):
        stock, market, _ = synthetic_data(180)
        cutoff = stock.index[120]
        baseline = research._recent_direction_signal(stock.loc[:cutoff], market.loc[:cutoff], 21)
        changed_stock = stock.copy()
        changed_market = market.copy()
        changed_stock.loc[changed_stock.index > cutoff] *= 3
        changed_market.loc[changed_market.index > cutoff] *= 0.5
        changed = research._recent_direction_signal(
            changed_stock.loc[:cutoff], changed_market.loc[:cutoff], 21
        )

        self.assertEqual(baseline, changed)

    def test_feature_builder_includes_price_volume_and_market_relative_inputs(self):
        stock, market, volume = synthetic_data()
        features = research.build_features(stock, market, volume)
        self.assertIn("volume_ratio_20", features)
        self.assertIn("relative_return_126", features)
        self.assertIn("rolling_alpha_20", features)
        self.assertIn("rolling_alpha_60", features)
        self.assertIn("rolling_beta_20", features)
        self.assertIn("rolling_beta_60", features)
        self.assertIn("stock_volatility_126", features)
        self.assertGreater(int(features.notna().any(axis=1).sum()), 500)


    def test_rolling_alpha_beta_features_are_trailing_only(self):
        stock, market, volume = synthetic_data()
        cutoff = stock.index[650]
        original = research.build_features(stock, market, volume)
        self.assertIn("rolling_beta_20", original)
        self.assertIn("rolling_beta_60", original)
        self.assertIn("rolling_alpha_20", original)
        self.assertIn("rolling_alpha_60", original)

        changed_stock = stock.copy()
        changed_market = market.copy()
        changed_stock.loc[changed_stock.index > cutoff] *= 1.8
        changed_market.loc[changed_market.index > cutoff] *= 0.7
        changed = research.build_features(changed_stock, changed_market, volume)
        pd.testing.assert_series_equal(
            original.loc[cutoff], changed.loc[cutoff], check_names=False
        )

    def test_missing_market_observation_does_not_compress_beta_calendar(self):
        stock, market, volume = synthetic_data(180)
        missing_date = stock.index[90]
        market = market.drop(index=missing_date)
        features = research.build_features(stock, market, volume)
        self.assertTrue(features.index.equals(stock.index))
        self.assertTrue(pd.isna(features.loc[missing_date, "market_return_1"]))
        self.assertEqual(len(features), len(stock))

    def test_matured_training_dates_purge_unfinished_forward_labels(self):
        index = pd.bdate_range("2024-01-01", periods=20)
        valid = np.ones(len(index), dtype=bool)
        latest = index[-1]
        selected = research._matured_training_dates(index, valid, latest, horizon=5)
        self.assertEqual(selected[-1], index[-6])
        self.assertNotIn(index[-5], selected)
        self.assertNotIn(index[-1], selected)

    def test_holdout_rows_are_never_used_for_calibration(self):
        index = pd.bdate_range("2024-01-01", periods=8)
        wf = {
            "raw": pd.Series([0.4, 0.5, 0.6, 0.7, 0.8], index=index[:5]),
            "development_index": index[:3],
            "holdout_index": index[3:5],
        }
        positions = pd.Series(np.arange(len(index)), index=index)
        eligible = research._development_calibration_dates(
            wf, positions, latest_position=7, horizon=2
        )
        self.assertEqual(eligible, list(index[:3]))
        eligible = research._development_calibration_dates(
            wf, positions, latest_position=3, horizon=2
        )
        self.assertEqual(eligible, list(index[:2]))


if __name__ == "__main__":
    unittest.main()
