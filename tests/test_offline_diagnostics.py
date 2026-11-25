import unittest

import numpy as np
import pandas as pd

from scripts.research_experiments import (
    _matured_development_dates,
    annual_regime_experiments,
    probability_calibration_experiment,
    rolling_quantile_thresholds,
)


class OfflineResearchExperimentTests(unittest.TestCase):
    def test_rolling_quantile_thresholds_are_trailing_only(self):
        index = pd.bdate_range("2020-01-01", periods=700)
        rng = np.random.default_rng(20261125)
        returns = rng.normal(0.0002, 0.012, len(index))
        close = pd.Series(100 * np.cumprod(1 + returns), index=index)
        cutoff = index[520]

        baseline = rolling_quantile_thresholds(close)
        changed = close.copy()
        changed.loc[changed.index > cutoff] *= 1.7
        altered = rolling_quantile_thresholds(changed)

        pd.testing.assert_frame_equal(
            baseline.loc[:cutoff], altered.loc[:cutoff], check_freq=False
        )
        self.assertTrue(baseline.loc[cutoff].notna().all())

    def test_annual_regime_experiments_include_fixed_and_adaptive_methods(self):
        index = pd.bdate_range("2018-01-01", periods=900)
        rng = np.random.default_rng(17)
        returns = rng.normal(0.0003, 0.012, len(index))
        close = pd.Series(100 * np.cumprod(1 + returns), index=index)
        high = close * (1 + np.abs(rng.normal(0.006, 0.002, len(index))))
        low = close * (1 - np.abs(rng.normal(0.006, 0.002, len(index))))

        rows = annual_regime_experiments(close, high=high, low=low)
        methods = {row["threshold_set"] for row in rows}
        self.assertTrue({
            "fixed_current",
            "fixed_candidate",
            "rolling_quantile_252",
            "realized_vol_20d_standardized",
            "atr_20d_standardized",
        }.issubset(methods))
        for row in rows:
            counts = sum(row[f"{name}_n"] for name in ("low", "normal", "high", "extreme"))
            self.assertEqual(counts, row["samples"])

    def test_calibration_training_dates_exclude_unmatured_and_holdout_rows(self):
        index = pd.bdate_range("2024-01-01", periods=10)
        wf = {
            "development_index": index[:5],
            "holdout_index": index[5:],
        }
        eligible = _matured_development_dates(wf, index, horizon=2)
        self.assertEqual(list(eligible), list(index[:4]))
        self.assertTrue(all(day not in wf["holdout_index"] for day in eligible))

    def test_probability_experiment_reports_raw_platt_and_isotonic_holdout_metrics(self):
        index = pd.bdate_range("2019-01-01", periods=950)
        rng = np.random.default_rng(29)
        market_return = rng.normal(0.0002, 0.01, len(index))
        stock_return = 0.0002 + 0.35 * market_return + rng.normal(0, 0.012, len(index))
        close = pd.Series(100 * np.cumprod(1 + stock_return), index=index)
        features = pd.DataFrame({
            "momentum": pd.Series(stock_return, index=index).rolling(5).sum().fillna(0),
            "market": market_return,
            "volatility": pd.Series(stock_return, index=index).rolling(20).std().fillna(0),
            "noise_a": rng.normal(size=len(index)),
            "noise_b": rng.normal(size=len(index)),
        }, index=index)

        result = probability_calibration_experiment(features, close, horizon=21)
        self.assertEqual(result["status"], "available")
        self.assertEqual(len(result["experiments"]), 8)
        available = [row for row in result["experiments"] if row["status"] == "available"]
        self.assertTrue(available)
        for config in available:
            methods = config["methods"]
            self.assertIn("raw", methods)
            for name in ("raw", "platt_scaling", "isotonic_regression"):
                if methods[name]["status"] == "available":
                    metrics = methods[name]["holdout_metrics"]
                    self.assertIn("brier_score", metrics)
                    self.assertIn("brier_skill", metrics)
                    self.assertIn("log_loss", metrics)
                    self.assertIn("direction_accuracy_pct", metrics)
                    self.assertGreater(metrics["samples"], 0)
            for name in ("platt_scaling", "isotonic_regression"):
                if methods[name]["status"] == "available":
                    self.assertLessEqual(
                        pd.Timestamp(methods[name]["calibration_fit_end"]),
                        pd.Timestamp(config["holdout_start"]),
                    )


if __name__ == "__main__":
    unittest.main()
