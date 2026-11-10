import inspect
import unittest

import pipeline


class RidgeForecastTrialTests(unittest.TestCase):
    def test_trials_include_aggressive_incumbent_and_conservative_settings(self):
        defaults = inspect.signature(pipeline.build_ridge_trial_forecasts).parameters["alphas"].default
        self.assertEqual(tuple(defaults), (1.0, 5.0, 10.0, 25.0, 100.0))
        self.assertIn(25.0, defaults)  # production incumbent
        self.assertLess(min(defaults), 25.0)  # less regularized challengers


if __name__ == "__main__":
    unittest.main()
