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
        features_by_horizon = {}
        for horizon in (1, 2, 3):
            features_by_horizon[horizon] = (X, close.shift(-horizon) / close - 1.0)
        market_data = pd.DataFrame({
            "TAIEX": close,
            "VIX": np.linspace(12.0, 28.0, n),
            "SOX": 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.012, n)),
            "TSM": 100.0 * np.cumprod(1.0 + rng.normal(0.0006, 0.01, n)),
            "N225": 20000.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.008, n)),
            "KOSPI": 2500.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.009, n)),
            "USDTWD": 30.0 * np.cumprod(1.0 + rng.normal(0.0, 0.002, n)),
            "US10Y": np.linspace(1.5, 4.5, n),
        }, index=dates)
        result = build_regime_router_forecasts(
            features_by_horizon,
            close,
            target_meta={h: {"target_date": "2021-01-01", "trading_calendar_source": "test"} for h in (1, 2, 3)},
            market_data=market_data,
        )
        self.assertEqual(result["routing_priority"], "router_first")
        self.assertEqual(len(result["sessions"]), 3)
        for horizon, item in enumerate(result["sessions"], start=1):
            self.assertEqual(item["horizon_sessions"], horizon)
            self.assertEqual(item["status"], "research")
            self.assertEqual(set(item["regime_probabilities"]), set(REGIME_NAMES))
            self.assertAlmostEqual(sum(item["regime_probabilities"].values()), 1.0, places=4)
            self.assertIn(item["expert_used"], (*REGIME_NAMES, "global_fallback"))
            self.assertIn("router_validation", item)
            self.assertIn("routed_forecast_validation", item)
            self.assertFalse(item["probabilities_calibrated"])


if __name__ == "__main__":
    unittest.main()
