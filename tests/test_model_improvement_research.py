import unittest
import numpy as np
import pandas as pd

from model_improvement_research import evaluate_predictions, walk_forward_candidate, current_research_judgment, prepare_market_frame


class ModelImprovementResearchTests(unittest.TestCase):
    def test_paired_block_bootstrap_reports_directional_deltas(self):
        from model_improvement_research import paired_block_bootstrap
        y = np.array([0, 1] * 50)
        incumbent = np.full(100, 0.5)
        candidate = np.where(y == 1, 0.8, 0.2)
        result = paired_block_bootstrap(candidate, incumbent, y, block_size=10, repeats=100)
        self.assertEqual(result["status"], "available")
        self.assertLess(result["brier_delta_candidate_minus_incumbent"], 0)
        self.assertGreater(result["accuracy_delta_percentage_points"], 0)
        self.assertEqual(len(result["brier_delta_95pct_ci"]), 2)

    def test_paired_comparison_uses_identical_calendar(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        X = pd.DataFrame({"market": np.arange(10, dtype=float),
                          "chip": np.arange(10, dtype=float)}, index=dates)
        X.loc[dates[4], "chip"] = np.nan
        y = pd.Series(np.arange(10) % 2, index=dates, dtype=float)
        valid = X.notna().all(axis=1) & y.notna()
        paired_index = X.index[valid]
        market_only = X.loc[paired_index, ["market"]]
        all_features = X.loc[paired_index]
        self.assertTrue(market_only.index.equals(all_features.index))
        self.assertNotIn(dates[4], market_only.index)

    def test_metrics_compare_against_baseline(self):
        result = evaluate_predictions(
            [0.8, 0.2, 0.7, 0.3],
            [1, 0, 1, 0],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.assertEqual(result["direction_accuracy_pct"], 100.0)
        self.assertLess(result["brier"], result["baseline_brier"])
        self.assertGreater(result["brier_skill"], 0)

    def test_sparse_unused_features_do_not_reduce_market_only_samples(self):
        dates = pd.bdate_range("2020-01-01", periods=180)
        market = pd.DataFrame({"market": np.sin(np.arange(180) / 7)}, index=dates)
        full = market.copy()
        full["sparse_chip"] = np.nan
        full.loc[dates[:120], "sparse_chip"] = np.arange(120, dtype=float)
        y = pd.Series((np.arange(180) % 3 == 0).astype(float), index=dates)

        class FixedClassifier:
            def fit(self, x, target):
                self.p = float(target.mean())
                return self
            def predict_proba(self, x):
                return np.column_stack([np.full(len(x), 1-self.p), np.full(len(x), self.p)])

        _, _, market_result = walk_forward_candidate(market, y, FixedClassifier, step=5)
        _, _, full_result = walk_forward_candidate(full, y, FixedClassifier, step=5)
        self.assertEqual(market_result["status"], "available")
        self.assertEqual(full_result["status"], "insufficient")
        self.assertGreater(len(market.dropna()), len(full.dropna()))


    def test_market_preprocessing_matches_production_trading_calendar(self):
        dates = pd.bdate_range("2026-10-01", periods=8)
        frame = pd.DataFrame({
            "TAIEX": [30000.0, 30100.0, np.nan, 30200.0, 30300.0, 30400.0, 30500.0, 30600.0],
            "SOX": [100.0, np.nan, 102.0, 103.0, np.nan, np.nan, np.nan, 107.0],
        }, index=dates)
        prepared = prepare_market_frame(frame)
        self.assertNotIn(dates[2], prepared.index)
        self.assertEqual(prepared.loc[dates[1], "SOX"], 100.0)
        self.assertEqual(prepared.loc[dates[6], "SOX"], 103.0)
        self.assertEqual(prepared.loc[dates[7], "SOX"], 107.0)

    def test_current_judgment_uses_latest_complete_feature_row(self):
        dates = pd.bdate_range("2020-01-01", periods=150)
        X = pd.DataFrame({"x": np.sin(np.arange(150) / 7)}, index=dates)
        X.loc[dates[-5:], "x"] = np.nan
        y = pd.Series((np.arange(150) % 2).astype(float), index=dates)

        class FixedClassifier:
            def fit(self, x, target):
                self.p = float(target.mean())
                return self
            def predict_proba(self, x):
                return np.column_stack([np.full(len(x), 1-self.p), np.full(len(x), self.p)])

        result = current_research_judgment(X, y, FixedClassifier)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["asof"], str(dates[-6].date()))
        self.assertEqual(result["training_samples"], 144)
        self.assertIn("category", result)

    def test_walk_forward_predictions_are_chronological_and_oos(self):
        dates = pd.bdate_range("2020-01-01", periods=160)
        X = pd.DataFrame({"x": np.sin(np.arange(160) / 7)}, index=dates)
        y = pd.Series((np.arange(160) % 3 == 0).astype(float), index=dates)
        class FixedClassifier:
            def fit(self, x, y):
                self.p = float(y.mean())
                return self
            def predict_proba(self, x):
                return np.column_stack([np.full(len(x), 1-self.p), np.full(len(x), self.p)])
        pred, baseline, result = walk_forward_candidate(X, y, FixedClassifier, step=5)
        self.assertEqual(result["status"], "available")
        self.assertTrue((pred.index > dates[0]).all())
        self.assertEqual(pred.index.min(), dates[100])
        self.assertTrue(pred.index.equals(baseline.index))


if __name__ == "__main__":
    unittest.main()
