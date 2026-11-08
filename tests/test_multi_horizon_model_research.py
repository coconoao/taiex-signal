import unittest
import numpy as np
import pandas as pd

from multi_horizon_model_research import _metrics, _fit_calibrator, _apply_calibrator


class MultiHorizonModelResearchTests(unittest.TestCase):
    def test_metrics_reports_brier_skill(self):
        y = np.array([0, 1, 0, 1] * 30)
        p = np.array([0.2, 0.8, 0.3, 0.7] * 30)
        result = _metrics(p, y)
        self.assertEqual(result["sample_count"], 120)
        self.assertGreater(result["brier_skill_vs_prevalence"], 0)

    def test_calibrator_requires_minimum_and_both_classes(self):
        self.assertIsNone(_fit_calibrator(np.linspace(0.1, 0.9, 50), np.array([0, 1] * 25)))
        self.assertIsNone(_fit_calibrator(np.linspace(0.1, 0.9, 100), np.ones(100)))
        cal = _fit_calibrator(np.linspace(0.1, 0.9, 100), np.array([0, 1] * 50))
        self.assertIsNotNone(cal)
        calibrated = _apply_calibrator(cal, [0.2, 0.8])
        self.assertEqual(len(calibrated), 2)
        self.assertTrue(np.all((calibrated > 0) & (calibrated < 1)))

if __name__ == "__main__":
    unittest.main()
