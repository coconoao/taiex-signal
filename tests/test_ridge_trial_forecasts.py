import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import pipeline


class RidgeTrialForecastTests(unittest.TestCase):
    def _frame(self, rows):
        index = pd.bdate_range("2024-01-02", periods=rows)
        rng = np.random.default_rng(17)
        returns = rng.normal(0.0003, 0.009, rows)
        close = 18000 * np.cumprod(1 + returns)
        return pd.DataFrame({"TAIEX": close}, index=index)

    def _stub_build(self, frame, chips, margin, h=1, futures=None):
        close = frame["TAIEX"]
        features = pd.DataFrame({
            "momentum": close.pct_change().fillna(0),
            "trend": np.linspace(-1, 1, len(close)),
        }, index=frame.index)
        forward = close.shift(-h) / close - 1
        return features, None, forward

    def test_returns_two_separate_three_horizon_candidates(self):
        frame = self._frame(340)
        with patch("pipeline.build", side_effect=self._stub_build):
            candidates = pipeline.build_ridge_trial_forecasts(
                frame, None, None, alphas=(10.0, 100.0)
            )

        self.assertEqual(
            [item["model_version"] for item in candidates],
            ["ridge-direct-alpha-10-trial-v1", "ridge-direct-alpha-100-trial-v1"],
        )
        for candidate in candidates:
            self.assertEqual(candidate["status"], "research")
            self.assertEqual(candidate["asof"], str(frame.index[-1].date()))
            self.assertEqual(len(candidate["sessions"]), 3)
            for horizon, session in enumerate(candidate["sessions"], start=1):
                self.assertEqual(session["horizon_sessions"], horizon)
                self.assertEqual(session["status"], "research")
                self.assertIsInstance(session["forecast_close"], float)
                self.assertGreaterEqual(session["training_samples"], 120)
                self.assertIn("oos_samples", session["validation"])
                self.assertIn("mae_points", session["validation"])
                self.assertIn("flat_baseline_mae_points", session["validation"])

    def test_marks_candidate_insufficient_when_training_history_is_short(self):
        frame = self._frame(100)
        with patch("pipeline.build", side_effect=self._stub_build):
            candidates = pipeline.build_ridge_trial_forecasts(
                frame, None, None, alphas=(10.0,)
            )
        self.assertEqual(candidates[0]["status"], "insufficient")
        self.assertTrue(all(x["status"] == "insufficient" for x in candidates[0]["sessions"]))


if __name__ == "__main__":
    unittest.main()
