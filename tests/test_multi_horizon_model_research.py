import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import multi_horizon_model_research as mhr

from multi_horizon_model_research import (
    _metrics, _fit_calibrator, _apply_calibrator, build_macro_proxy_features, latest_estimate,
    _promotion_quality_check,
)


class MultiHorizonModelResearchTests(unittest.TestCase):
    def test_promotion_gate_accounts_for_overlapping_horizon_labels(self):
        good_years = {
            "2024": {"brier_skill_vs_prevalence": 0.03},
            "2025": {"brier_skill_vs_prevalence": 0.02},
            "2026": {"brier_skill_vs_prevalence": 0.01},
        }
        metrics = {"brier_skill_vs_prevalence": 0.04}
        too_few = _promotion_quality_check(metrics, good_years, 126 * 2, 126)
        self.assertFalse(too_few["eligible"])
        self.assertEqual(too_few["effective_nonoverlap_windows"], 2)
        enough = _promotion_quality_check(metrics, good_years, 126 * 20, 126)
        self.assertTrue(enough["eligible"])
        bad_years = dict(good_years)
        bad_years["2026"] = {"brier_skill_vs_prevalence": -0.01}
        unstable = _promotion_quality_check(metrics, bad_years, 126 * 20, 126)
        self.assertFalse(unstable["eligible"])

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


    def test_macro_proxy_features_are_point_in_time(self):
        rng = np.random.default_rng(7)
        index = pd.bdate_range("2020-01-01", periods=260)
        columns = ["TAIEX", "SOX", "TSM", "VIX", "US10Y", "N225", "KOSPI", "USDTWD"]
        data = {}
        for col in columns:
            steps = rng.normal(0.0002, 0.01, len(index))
            data[col] = 100 * np.cumprod(1 + steps)
        market = pd.DataFrame(data, index=index)
        features = build_macro_proxy_features(market)
        self.assertIn("宏觀代理:SOX_報酬21日", features)
        self.assertIn("宏觀代理:VIX滾動Z分數", features)
        self.assertIn("宏觀代理:美國10年債殖利率變化21日", features)
        self.assertIn("宏觀代理:USD_TWD變化21日", features)

        changed = market.copy()
        changed.loc[index[180]:, columns] *= 1.7
        changed_features = build_macro_proxy_features(changed)
        pd.testing.assert_frame_equal(
            features.loc[:index[179]],
            changed_features.loc[:index[179]],
        )

    def test_macro_proxy_features_do_not_fill_missing_source_data(self):
        index = pd.bdate_range("2021-01-01", periods=100)
        market = pd.DataFrame({
            "TAIEX": np.arange(100, dtype=float) + 100,
            "SOX": np.arange(100, dtype=float) + 200,
            "VIX": np.arange(100, dtype=float) + 20,
            "US10Y": np.arange(100, dtype=float) / 100 + 2,
            "USDTWD": np.arange(100, dtype=float) / 1000 + 28,
        }, index=index)
        market.loc[index[70], "SOX"] = np.nan
        features = build_macro_proxy_features(market)
        self.assertTrue(np.isnan(features.loc[index[70], "宏觀代理:SOX_報酬5日"]))


    def test_latest_estimate_uses_latest_feature_row_even_when_target_is_unmatured(self):
        rng = np.random.default_rng(11)
        index = pd.bdate_range("2020-01-01", periods=720)
        x = rng.normal(size=720)
        X = pd.DataFrame({"x1": x, "x2": rng.normal(size=720)}, index=index)
        y = pd.Series((x + rng.normal(scale=0.5, size=720) > 0).astype(float), index=index)
        y.iloc[-5:] = np.nan
        oos_index = index[100:600:3]
        oos_probability = pd.Series(np.linspace(0.2, 0.8, len(oos_index)), index=oos_index)
        oos_actual = pd.Series((np.arange(len(oos_index)) % 2).astype(int), index=oos_index)
        result = latest_estimate(
            X, y, 5,
            lambda: LogisticRegression(max_iter=300),
            oos_probability, oos_actual,
        )
        self.assertEqual(result["asof"], str(index[-1].date()))


    def test_run_passes_raw_oos_predictions_to_latest_calibration(self):
        index = pd.bdate_range("2020-01-01", periods=260)
        rng = np.random.default_rng(9)
        columns = ["TAIEX", "SOX", "TSM", "VIX", "US10Y", "N225", "KOSPI", "USDTWD"]
        market = pd.DataFrame({
            name: 100 + np.cumsum(rng.normal(0, 1, len(index)))
            for name in columns
        }, index=index)
        X = market.copy()
        y = pd.Series((np.arange(len(index)) % 2).astype(float), index=index)
        fwd = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
        raw_oos = pd.Series([0.12, 0.28, 0.71], index=index[-3:])
        calibrated_oos = pd.Series([0.39, 0.48, 0.63], index=index[-3:])
        wf = {
            "raw_probability": raw_oos,
            "probability": calibrated_oos,
            "actual": pd.Series([0, 1, 1], index=index[-3:]),
        }
        captured = []

        def fake_evaluate(X_candidate, y_candidate, fwd_candidate, horizon, name, factory):
            return {
                "status": "evaluated",
                "development": {"brier": 0.1 if "market_only" in name else 0.2},
                "feature_set": "market_only_common_calendar",
                "promotion_quality_check": {},
            }

        def fake_latest(X_candidate, y_candidate, horizon, factory, oos_raw_probability, oos_actual):
            captured.append(oos_raw_probability.copy())
            return {"status": "available"}

        with (
            patch.object(mhr.mir, "load_inputs", return_value=(market, None, None, None)),
            patch.object(mhr.pipeline, "build", return_value=(X, y, fwd)),
            patch.object(mhr, "evaluate_candidate", side_effect=fake_evaluate),
            patch.object(mhr, "walk_forward", return_value=wf),
            patch.object(mhr, "latest_estimate", side_effect=fake_latest),
        ):
            mhr.run()

        self.assertEqual(len(captured), len(mhr.HORIZONS))
        for passed in captured:
            pd.testing.assert_series_equal(passed, raw_oos)
            self.assertFalse(passed.equals(calibrated_oos))

if __name__ == "__main__":
    unittest.main()
