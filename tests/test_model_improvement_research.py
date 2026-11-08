import unittest
import numpy as np
import pandas as pd

from model_improvement_research import evaluate_predictions, walk_forward_candidate


class ModelImprovementResearchTests(unittest.TestCase):
    def test_metrics_compare_against_baseline(self):
        result = evaluate_predictions(
            [0.8, 0.2, 0.7, 0.3],
            [1, 0, 1, 0],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.assertEqual(result["direction_accuracy_pct"], 100.0)
        self.assertLess(result["brier"], result["baseline_brier"])
        self.assertGreater(result["brier_skill"], 0)

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
        self.assertEqual(pred.index.min(), dates[80])
        self.assertTrue(pred.index.equals(baseline.index))


if __name__ == "__main__":
    unittest.main()
