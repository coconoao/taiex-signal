import unittest

import numpy as np
import pandas as pd

from regime_router import (
    REGIME_NAMES,
    build_regime_router_forecasts,
    classify_regimes,
    _select_expert,
    _eligible_calibration_records,
    _purged_training_indices,
    _augment_router_features,
    _lag_base_cross_market_features,
    _lag_post_close_source_frame,
    _lag_post_close_features,
    _forecast_metrics,
    _expert_performance_blend_weight,
    _gate_policy_sensitivity,
    _paired_block_bootstrap_delta,
    _historical_magnitude_profile,
    _future_max_daily_move,
    _route_regime,
    _routing_reason,
    _select_risk_thresholds,
)


class RegimeRouterTests(unittest.TestCase):
    def test_expert_performance_gate_uses_only_matured_prior_oos_records(self):
        records = []
        for position in range(100):
            records.append({
                "forecast_position": position,
                "requested_expert": "high",
                "raw_expert_predicted_close": 101.0 if position < 78 else 110.0,
                "global_predicted_close": 102.0,
                "actual_close": 100.0,
            })
        # At position 80 with a 3-session horizon, only origins 0..77
        # have matured. Their expert MAE is half the global Ridge MAE.
        gate = _expert_performance_blend_weight(
            records, "high", 80, 3,
            "raw_expert_predicted_close", "global_predicted_close", "actual_close",
        )
        self.assertEqual(gate["samples"], 78)
        self.assertEqual(gate["weight"], 1.0)
        # At position 110, all 100 outcomes are mature; the later expert
        # errors make it worse than Ridge, so its contribution must be zero.
        later = _expert_performance_blend_weight(
            records, "high", 110, 3,
            "raw_expert_predicted_close", "global_predicted_close", "actual_close",
        )
        self.assertEqual(later["samples"], 100)
        self.assertEqual(later["weight"], 0.0)

    def test_gate_policy_sensitivity_uses_earlier_blocks_and_reports_holdout(self):
        rows = []
        for block, start in enumerate((100, 160, 220, 280)):
            for offset in range(10):
                position = start + offset
                rows.append({
                    "forecast_position": position,
                    "oos_block_start_position": start,
                    "expert_used": "high",
                    "raw_expert_predicted_close": 102.0,
                    "global_predicted_close": 101.0,
                    "actual_close": 100.0,
                    "base_close": 100.0,
                    "raw_expert_predicted_max_daily_move": 0.02,
                    "global_predicted_max_daily_move": 0.015,
                    "actual_max_daily_move": 0.02,
                })
        result = _gate_policy_sensitivity(rows, horizon=1)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["oos_samples"], 40)
        self.assertIn(result["best_policy_on_early_window"], result["policies"])
        self.assertEqual(result["selected_policy_late_holdout"]["oos_samples"], 12)
        self.assertEqual(result["policies"]["global_only"]["full_oos"]["point_mae"], 1.0)

    def test_paired_block_bootstrap_detects_consistent_expert_improvement(self):
        rows = [
            {"expert_prediction": 101.0, "global_prediction": 102.0, "actual": 100.0}
            for _ in range(80)
        ]
        result = _paired_block_bootstrap_delta(
            rows, "expert_prediction", "global_prediction", "actual",
            seed=17, block_size=10, replicates=100,
        )
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["observed_mae_delta_vs_global"], -1.0)
        self.assertTrue(result["significant_improvement_at_95pct"])

    def test_forecast_metrics_reports_exact_oos_date_window(self):
        rows = [
            {
                "forecast_asof": "2024-01-02",
                "predicted_close": 101.0, "actual_close": 100.0,
                "actual_return": 0.0, "predicted_return": 0.01,
                "actual_max_daily_move": 0.004, "predicted_max_daily_move": 0.003,
                "global_predicted_close": 100.5, "global_predicted_max_daily_move": 0.002,
                "actual_regime": 0, "expert_used": "low",
            },
            {
                "forecast_asof": "2024-01-03",
                "predicted_close": 102.0, "actual_close": 100.0,
                "actual_return": 0.0, "predicted_return": 0.02,
                "actual_max_daily_move": 0.02, "predicted_max_daily_move": 0.015,
                "global_predicted_close": 101.0, "global_predicted_max_daily_move": 0.01,
                "actual_regime": 2, "expert_used": "high",
            },
        ]
        metrics = _forecast_metrics(rows)
        self.assertEqual(metrics["oos_start_date"], "2024-01-02")
        self.assertEqual(metrics["oos_end_date"], "2024-01-03")
        self.assertEqual(metrics["oos_samples"], 2)

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

    def test_existing_pipeline_cross_market_features_are_lagged(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        features = pd.DataFrame({
            "國際:NASDAQ": [0.01, 0.02, -0.03, 0.04, 0.05, 0.06],
            "區域:N225": [0.11, 0.12, -0.13, 0.14, 0.15, 0.16],
            "本地:當日漲跌": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
            "籌碼:外資買賣超": [1, 2, 3, 4, 5, 6],
        }, index=dates)
        aligned = _lag_base_cross_market_features(features)
        self.assertTrue(pd.isna(aligned.loc[dates[0], "國際:NASDAQ"]))
        self.assertEqual(aligned.loc[dates[3], "國際:NASDAQ"], -0.03)
        self.assertEqual(aligned.loc[dates[3], "區域:N225"], -0.13)
        self.assertEqual(aligned.loc[dates[3], "本地:當日漲跌"], 0.004)
        self.assertEqual(aligned.loc[dates[3], "籌碼:外資買賣超"], 4)

    def test_post_close_source_frames_are_lagged_on_full_taiwan_calendar(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        source = pd.DataFrame({
            "foreign_net": [10.0, 20.0, 30.0, 40.0, 50.0],
        }, index=dates[[0, 1, 3, 4, 5]])
        aligned = _lag_post_close_source_frame(source, dates)
        self.assertTrue(pd.isna(aligned.loc[dates[0], "foreign_net"]))
        self.assertEqual(aligned.loc[dates[1], "foreign_net"], 10.0)
        # Missing source data on session 2 stays missing on session 3 after lag;
        # the value from session 1 must not be shifted across a missing session.
        self.assertEqual(aligned.loc[dates[2], "foreign_net"], 20.0)
        self.assertTrue(pd.isna(aligned.loc[dates[3], "foreign_net"]))
        self.assertEqual(aligned.loc[dates[4], "foreign_net"], 30.0)

    def test_post_close_engineered_features_are_lagged_by_router(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        features = pd.DataFrame({
            "籌碼:外資買賣超": [10.0, 20.0, 30.0, 40.0],
            "信用:融資餘額5日變化": [1.0, 2.0, 3.0, 4.0],
            "期貨:外資淨未平倉5日變化": [100.0, 200.0, 300.0, 400.0],
            "本地:當日漲跌": [0.1, 0.2, 0.3, 0.4],
        }, index=dates)
        aligned = _lag_post_close_features(features)
        self.assertTrue(pd.isna(aligned.loc[dates[0], "籌碼:外資買賣超"]))
        self.assertEqual(aligned.loc[dates[2], "籌碼:外資買賣超"], 20.0)
        self.assertEqual(aligned.loc[dates[2], "信用:融資餘額5日變化"], 2.0)
        self.assertEqual(aligned.loc[dates[2], "期貨:外資淨未平倉5日變化"], 200.0)
        self.assertEqual(aligned.loc[dates[2], "本地:當日漲跌"], 0.3)

    def test_cross_market_features_are_lagged_to_taiwan_cash_close(self):
        dates = pd.bdate_range("2024-01-01", periods=70)
        market = pd.DataFrame({
            "SOX": np.linspace(100.0, 170.0, 70),
            "TSM": np.linspace(50.0, 85.0, 70),
            "VIX": np.linspace(10.0, 25.0, 70),
            "US10Y": np.linspace(3.0, 4.0, 70),
            "N225": np.linspace(30000.0, 37000.0, 70),
            "KOSPI": np.linspace(2500.0, 3200.0, 70),
            "USDTWD": np.linspace(31.0, 32.0, 70),
            "TAIEX": np.linspace(17000.0, 20500.0, 70),
        }, index=dates)
        features = _augment_router_features(pd.DataFrame(index=dates), market)
        expected_sox_daily = market["SOX"].pct_change().shift(1)
        expected_sox_momentum = market["SOX"].pct_change(5).shift(1)
        expected_vix_level = (
            market["VIX"] / market["VIX"].rolling(60, min_periods=20).mean() - 1.0
        ).shift(1)
        expected_yield_change = market["US10Y"].diff(5).shift(1)
        point = dates[30]
        self.assertAlmostEqual(
            features.loc[point, "國際:SOX"], expected_sox_daily.loc[point]
        )
        self.assertNotAlmostEqual(
            features.loc[point, "國際:SOX"],
            market["SOX"].pct_change().loc[point],
        )
        self.assertAlmostEqual(
            features.loc[point, "router:SOX_momentum_5d"],
            expected_sox_momentum.loc[point],
        )
        self.assertAlmostEqual(
            features.loc[point, "router:VIX_level_vs_60d_mean"],
            expected_vix_level.loc[point],
        )
        self.assertAlmostEqual(
            features.loc[point, "router:US10Y_change_5d"],
            expected_yield_change.loc[point],
        )

    def test_walk_forward_purge_uses_full_trading_calendar_with_missing_features(self):
        dates = pd.bdate_range("2024-01-01", periods=12)
        # Valid feature rows omit two sessions before the first test origin.
        candidates = dates[[0, 1, 3, 5, 6, 8]]
        first_test = dates[9]
        fit = _purged_training_indices(dates, candidates, first_test, horizon=3)
        self.assertEqual(list(fit), list(dates[[0, 1, 3, 5, 6]]))
        # The row at position 8 is only one session before the test origin
        # and must not leak its still-open three-session target into training.

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

    def test_router_explains_risk_threshold_decision(self):
        reason = _routing_reason(
            [0.52, 0.20, 0.23, 0.05],
            {"high_extreme": 0.25, "extreme": 0.15},
            2,
        )
        self.assertIn("達到升級門檻", reason)

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
        self.assertEqual(thresholds["source"], "prior_calibrated_oos_f2_with_prevalence_guard")
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
        # Simulate the latest complete feature row lagging the latest close:
        # the last three feature rows are unavailable, while their close labels
        # may already exist beyond the actual forecast origin.
        X.iloc[-3:, 0] = np.nan
        features_by_horizon = {
            horizon: (X, close.shift(-horizon) / close - 1.0)
            for horizon in (1, 2, 3)
        }
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
            self.assertEqual(item["unmatured_training_samples_excluded"], horizon - 1)
            self.assertEqual(
                item["candidate_training_samples"] - item["training_samples"],
                horizon - 1,
            )
            self.assertGreaterEqual(item["predicted_max_daily_move_pct"], 0.0)
            self.assertEqual(set(item["regime_probabilities"]), set(REGIME_NAMES))
            self.assertTrue(item["routing_reason"])
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
