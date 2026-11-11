import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from regime_interval_research import expanding_quantile_regimes, interval_metrics, rolling_conformal_intervals


class DynamicRegimeIntervalTests(unittest.TestCase):
    def test_expanding_quantiles_use_only_strictly_prior_outcomes(self):
        values = np.arange(1.0, 301.0) / 1000.0
        baseline = expanding_quantile_regimes(values, min_history=20)
        changed = values.copy()
        changed[250:] = 9.0
        revised = expanding_quantile_regimes(changed, min_history=20)
        np.testing.assert_allclose(baseline["thresholds"][250], revised["thresholds"][250])
        self.assertEqual(baseline["labels"][250], 3)
        self.assertEqual(baseline["threshold_policy"], "expanding_prior_matured_outcomes_only")

    def test_expanding_quantiles_warmup_and_invalid_values_are_unlabelled(self):
        result = expanding_quantile_regimes([0.1, 0.2, np.nan, 0.3, 0.4], min_history=3)
        self.assertEqual(result["labels"].tolist()[:4], [-1, -1, -1, -1])
        self.assertGreaterEqual(result["labels"][4], 0)
        self.assertTrue(np.isnan(result["thresholds"][2]).all())

    def test_latest_matured_history_thresholds_exist_before_future_outcome(self):
        values = np.concatenate([np.linspace(0.001, 0.3, 300), [np.nan]])
        result = expanding_quantile_regimes(
            values, min_history=252, origin_positions=np.arange(len(values)), horizon=1
        )
        self.assertTrue(np.isfinite(result["thresholds"][-1]).all())
        self.assertEqual(result["labels"][-1], -1)
        self.assertEqual(result["history_counts"][-1], 300)

    def test_conformal_excludes_unmatured_prior_forecasts(self):
        result = rolling_conformal_intervals(np.zeros(6), [1, 2, 3, 4, 5, 6],
                                             horizon=2, min_calibration_samples=1,
                                             coverages=(0.5, 0.8))
        row = result["rows"][3]
        self.assertEqual(row["calibration_samples"], 2)
        self.assertEqual(row["intervals"]["50"]["upper"], 2.0)

    def test_regime_calibration_falls_back_to_global_when_sparse(self):
        result = rolling_conformal_intervals(np.zeros(10), np.arange(1, 11, dtype=float),
                                             horizon=1, min_calibration_samples=2,
                                             min_regime_samples=6, regime_labels=np.zeros(10, dtype=int))
        row = result["rows"][5]
        self.assertEqual(row["calibration_source"], "same_horizon_global_fallback_insufficient_regime_samples")
        self.assertEqual(row["calibration_samples"], 5)

    def test_metrics_report_coverage_width_and_interval_score(self):
        result = rolling_conformal_intervals([0, 0, 0, 0], [1, 1, 1, 1],
                                             horizon=1, min_calibration_samples=1,
                                             coverages=(0.5,))
        metrics = interval_metrics(result, [1, 1, 1, 1])
        self.assertEqual(metrics["50"]["samples"], 3)
        self.assertEqual(metrics["50"]["empirical_coverage"], 1.0)
        self.assertEqual(metrics["50"]["mean_width"], 2.0)


    def test_signed_residual_intervals_are_asymmetric_when_errors_are_skewed(self):
        predictions = np.zeros(6)
        actuals = [1, 1, 1, 1, 10, np.nan]
        result = rolling_conformal_intervals(
            predictions, actuals, horizon=1, min_calibration_samples=1,
            coverages=(0.5,), method="signed_residual",
        )
        interval = result["rows"][5]["intervals"]["50"]
        self.assertEqual(result["method"], "signed_residual")
        self.assertGreater(interval["lower"], 0.0)
        self.assertEqual(interval["upper"], 10.0)
        self.assertNotIn("radius", interval)

    def test_invalid_conformal_method_fails_closed(self):
        with self.assertRaises(ValueError):
            rolling_conformal_intervals([0, 0], [1, 1], horizon=1, method="guess")

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            rolling_conformal_intervals([1, 2], [1], horizon=1)
        with self.assertRaises(ValueError):
            expanding_quantile_regimes([1, 2], min_history=0)

    def test_dynamic_thresholds_exclude_unmatured_horizon_outcomes(self):
        values = np.arange(1.0, 21.0)
        positions = np.arange(20)
        baseline = expanding_quantile_regimes(values, min_history=5,
                                              origin_positions=positions, horizon=3)
        changed = values.copy()
        changed[8] = 1000.0
        revised = expanding_quantile_regimes(changed, min_history=5,
                                             origin_positions=positions, horizon=3)
        np.testing.assert_allclose(baseline["thresholds"][10], revised["thresholds"][10])


if __name__ == "__main__":
    unittest.main()
