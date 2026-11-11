import numpy as np
import pytest

from regime_interval_research import expanding_quantile_regimes, interval_metrics, rolling_conformal_intervals


def test_expanding_quantiles_use_only_strictly_prior_outcomes():
    values = np.arange(1.0, 301.0) / 1000.0
    baseline = expanding_quantile_regimes(values, min_history=20)
    changed = values.copy()
    changed[250:] = 9.0
    revised = expanding_quantile_regimes(changed, min_history=20)
    np.testing.assert_allclose(baseline["thresholds"][250], revised["thresholds"][250])
    assert baseline["labels"][250] == 3
    assert baseline["threshold_policy"] == "expanding_prior_matured_outcomes_only"


def test_expanding_quantiles_warmup_and_invalid_values_are_unlabelled():
    result = expanding_quantile_regimes([0.1, 0.2, np.nan, 0.3, 0.4], min_history=3)
    assert result["labels"].tolist()[:4] == [-1, -1, -1, -1]
    assert result["labels"][4] >= 0
    assert np.isnan(result["thresholds"][2]).all()


def test_conformal_excludes_unmatured_prior_forecasts():
    result = rolling_conformal_intervals(np.zeros(6), [1, 2, 3, 4, 5, 6],
                                         horizon=2, min_calibration_samples=1,
                                         coverages=(0.5, 0.8))
    row = result["rows"][3]
    assert row["calibration_samples"] == 2
    assert row["intervals"]["50"]["upper"] == 2.0


def test_regime_calibration_falls_back_to_global_when_sparse():
    result = rolling_conformal_intervals(np.zeros(10), np.arange(1, 11, dtype=float),
                                         horizon=1, min_calibration_samples=2,
                                         min_regime_samples=5, regime_labels=np.zeros(10, dtype=int))
    row = result["rows"][5]
    assert row["calibration_source"] == "same_horizon_global_fallback_insufficient_regime_samples"
    assert row["calibration_samples"] == 5


def test_metrics_report_coverage_width_and_interval_score():
    result = rolling_conformal_intervals([0, 0, 0, 0], [1, 1, 1, 1],
                                         horizon=1, min_calibration_samples=1,
                                         coverages=(0.5,))
    metrics = interval_metrics(result, [1, 1, 1, 1])
    assert metrics["50"]["samples"] == 3
    assert metrics["50"]["empirical_coverage"] == 1.0
    assert metrics["50"]["mean_width"] == 2.0


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        rolling_conformal_intervals([1, 2], [1], horizon=1)
    with pytest.raises(ValueError):
        expanding_quantile_regimes([1, 2], min_history=0)



def test_dynamic_thresholds_exclude_unmatured_horizon_outcomes():
    values = np.arange(1.0, 21.0)
    positions = np.arange(20)
    baseline = expanding_quantile_regimes(values, min_history=5,
                                          origin_positions=positions, horizon=3)
    changed = values.copy()
    changed[8] = 1000.0
    revised = expanding_quantile_regimes(changed, min_history=5,
                                          origin_positions=positions, horizon=3)
    # At origin 10, row 8 has not matured yet (8 + 3 > 10).
    np.testing.assert_allclose(baseline["thresholds"][10], revised["thresholds"][10])
