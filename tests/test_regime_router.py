import unittest

import numpy as np
import pandas as pd

from regime_router import (
    REGIME_NAMES,
    build_regime_router_forecasts,
    classify_regimes,
    _select_expert,
)


class RegimeRouterTests(unittest.TestCase):
    def test_regime_boundaries_are_mutually_exclusive(self):
        values = np.array([0.0, 0.005, -0.007, 0.01, -0.02, 0.0749, -0.075, 0.10])
        self.assertEqual(
            classify_regimes(values).tolist(),
            [0, 0, 1, 1, 2, 2, 3, 3],
        )

    def test_extreme_expert_falls_back_to_high_when_sparse(self):
        class Dummy:
            pass

        model, name, samples, reason = _select_expert(
            3, Dummy(), {2: Dummy()}, {0: 100, 1: 100, 2: 52, 3: 4}
        )
        self.assertEqual(name, "high")
        self.assertEqual(samples, 52)
        self.assertIn("極高波動樣本不足", reason)

    def test_router_runs_independently_for_each_horizon(self):
        rng = np.random.default_rng(20261110)
        n = 720
        dates = pd.bdate_range("2018-01-01", periods=n)
        pattern = np.resize(np.array([0.002, -0.007, 0.018, -0.082, 0.003, 0.012]), n)
        returns = pd.Series(pattern, index=dates, dtype=float)
        returns.iloc[-1] = np.nan
        close_values = np.empty(n, dtype=float)
        close_values[0] = 10000.0
        for i in range(1, n):
            close_values[i] = close_values[i - 1] * (1.0 + (pattern[i - 1] if np.isfinite(pattern[i - 1]) else 0.0))
        close = pd.Series(close_values, index=dates)
        X = pd.DataFrame(
            rng.normal(size=(n, 5)),
            index=dates,
            columns=["global", "local", "vol", "momentum", "flow"],
        )
        result = build_regime_router_forecasts(
            {1: (X, returns)},
            close,
            target_meta={1: {"target_date": "2021-01-01", "trading_calendar_source": "test"}},
        )
        self.assertEqual(result["routing_priority"], "router_first")
        self.assertEqual(len(result["sessions"]), 3)
        d1 = result["sessions"][0]
        self.assertEqual(d1["horizon_sessions"], 1)
        self.assertIn(d1["status"], ("research", "insufficient"))
        if d1["status"] == "research":
            self.assertEqual(set(d1["regime_probabilities"]), set(REGIME_NAMES))
            self.assertIn(d1["expert_used"], (*REGIME_NAMES, "global_fallback"))
            self.assertIn("router_validation", d1)
            self.assertIn("routed_forecast_validation", d1)
            self.assertFalse(d1["probabilities_calibrated"])


if __name__ == "__main__":
    unittest.main()
