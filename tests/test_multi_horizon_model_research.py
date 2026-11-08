import unittest
import numpy as np
import pandas as pd

from multi_horizon_model_research import (
    _metrics, _fit_calibrator, _apply_calibrator, build_macro_proxy_features,
)


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

if __name__ == "__main__":
    unittest.main()
