import unittest

import numpy as np
import pandas as pd

from regime_router import (
    REGIME_NAMES,
    build_regime_router_forecasts,
    classify_regimes,
    _select_expert,
    _eligible_calibration_records,
    _historical_magnitude_profile,
    _future_max_daily_move,
    _route_regime,
    _select_risk_thresholds,
)


class RegimeRouterTests(unittest.TestCase):
    def test_router_target_uses_max_daily_move_not_cumulative_return(self):
        prices = pd.Series([100.0, 102.0, 100.0, 101.0], index=pd.bdate_range("2024-01-01", periods=4))
        target = _future_max_daily_move(prices, 2)
        self.assertAlmostEqual(float(target.iloc[0]), 0.02, places=8)
        self.assertAlmostEqual(float(prices.iloc[2] / prices.iloc[0] - 1.0), 0.0, places=8)

    def test_regime_boundaries_are_mutually_exclusive(self):
        values = np.array([0.0, 0.005, -0.007, 0.01, -0.02, 0.0749, -0.075, 0.10])
        self.assertEqual(
            classify_regimes(values).tolist(),
            [0, 0, 1, 1, 2, 2, 3, 3],
        )

    def test_calibration_only_uses_outcomes_known_at_forecast_origin(self):
        records = [
            (4, [0.7, 0.2, 0.08, 0.02], 0),
            (6, [0.6, 0.25, 0.1, 0.05], 1),
            (7, [0.5, 0.3, 0.15, 0.05], 2),
        ]
        eligible = _eligible_calibration_records(records, current_position=10, horizon=3)
        self.assertEqual([record[0] for record in eligible], [4, 6, 7])
        eligible = _eligible_calibration_records(records, current_position=9, horizon=3)
        self.assertEqual([record[0] for record in eligible], [4, 6])

    def test_historical_scenario_profile_uses_return_fractions_for_points(self):
        returns = np.resize(np.array([-0.05, -0.02, 0.01, 0.03, 0.08]), 40)
        profile = _historical_magnitude_profile(returns, np.abs(returns), 10000.0)
        self.assertEqual(profile["status"], "available")
        self.assertEqual(profile["samples"], 40)
        self.assertLess(profile["scenario_p10_close"], 10000.0)
        self.assertGreater(profile["scenario_p90_close"], 10000.0)
        self.assertEqual(set(profile["historical_abs_move_exceedance_pct"]), {"1.0", "2.0", "3.0", "5.0", "7.5"})

    def test_risk_first_router_escalates_to_high_even_when_low_is_top_score(self):
        probabilities = [0.52, 0.20, 0.23, 0.05]
        thresholds = {"high_extreme": 0.25, "extreme": 0.15}
        self.assertEqual(_route_regime(probabilities, thresholds), 2)

    def test_extreme_probability_escalates_directly_to_extreme(self):
        probabilities = [0.40, 0.20, 0.10, 0.30]
        thresholds = {"high_extreme": 0.25, "extreme": 0.15}
        self.assertEqual(_route_regime(probabilities, thresholds), 3)

    def test_risk_thresholds_learn_from_prior_oos_scores(self):
        probabilities = [[0.7, 0.2, 0.08, 0.02] for _ in range(170)]
        actual = [0] * 150 + [2] * 20
        for i in range(150, 170):
            probabilities[i] = [0.1, 0.1, 0.7, 0.1]
        thresholds = _select_risk_thresholds(probabilities, actual)
        self.assertEqual(thresholds["source"], "prior_calibrated_oos_f2")
        self.assertLessEqual(thresholds["high_extreme"], 0.8)
        self.assertEqual(thresholds["training_oos_samples"], 170)

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
            self.assertGreaterEqual(item["predicted_max_daily_move_pct"], 0.0)
            self.assertEqual(set(item["regime_probabilities"]), set(REGIME_NAMES))
            self.assertIn("high_extreme", item["router_decision_thresholds"])
            self.assertIn("extreme", item["router_decision_thresholds"])
            self.assertAlmostEqual(sum(item["regime_probabilities"].values()), 1.0, places=4)
            self.assertIn(item["expert_used"], (*REGIME_NAMES, "global_fallback"))
            self.assertIn("router_validation", item)
            self.assertIn("routed_forecast_validation", item)
            self.assertIsInstance(item["probabilities_calibrated"], bool)
            if item["probabilities_calibrated"]:
                self.assertEqual(item["probability_calibration"]["method"], "prequential_multinomial_logistic")


if __name__ == "__main__":
    unittest.main()
