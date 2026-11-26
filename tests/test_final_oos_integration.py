"""End-to-end wiring tests for paired ATR routing and prequential Platt."""
import numpy as np
import pandas as pd

from regime_router import build_regime_router_forecasts
from scripts.final_oos_methods import atr_percent
from stock_model_research import prequential_platt_candidates


def test_router_fixed_and_atr_modes_share_oos_origins():
    rng = np.random.default_rng(19)
    index = pd.bdate_range("2022-01-03", periods=360)
    returns = rng.normal(0.0002, 0.012, len(index))
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index)
    features = pd.DataFrame(
        {"feature_a": rng.normal(size=len(index)), "feature_b": rng.normal(size=len(index))},
        index=index,
    )
    forward = close.shift(-1) / close - 1.0
    high = close * (1.0 + rng.uniform(0.002, 0.015, len(index)))
    low = close * (1.0 - rng.uniform(0.002, 0.015, len(index)))
    atr = atr_percent(high, low, close, window=20)
    inputs = {1: (features, forward)}

    fixed = build_regime_router_forecasts(
        inputs, close, post_close_features_lagged=True,
        include_gate_sensitivity=False, regime_threshold_mode="fixed",
    )
    adaptive = build_regime_router_forecasts(
        inputs, close, post_close_features_lagged=True,
        include_gate_sensitivity=False, regime_threshold_mode="atr",
        atr_pct=atr, evaluation_start_date="2022-01-03",
    )
    f = fixed["sessions"][0]["routed_forecast_validation"]
    a = adaptive["sessions"][0]["routed_forecast_validation"]
    assert f["oos_samples"] > 0
    assert a["oos_samples"] == f["oos_samples"]
    assert a["oos_start_date"] == f["oos_start_date"]
    assert a["oos_end_date"] == f["oos_end_date"]
    assert adaptive["sessions"][0]["regime_threshold_mode"] == "atr"
    assert "annual_oos_metrics" in adaptive["sessions"][0]


def test_prequential_platt_runs_both_unscaled_base_candidates_with_full_calendar():
    rng = np.random.default_rng(23)
    index = pd.bdate_range("2017-01-02", periods=2200)
    returns = rng.normal(0.0003, 0.012, len(index))
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index)
    features = pd.DataFrame(
        {f"x{i}": rng.normal(size=len(index)) for i in range(4)}, index=index
    )
    result = prequential_platt_candidates(
        features, close, horizon=21, oos_start="2024-01-01"
    )
    assert result["status"] == "available"
    assert result["target_horizon_sessions"] == 21
    assert set(result["candidates"]) == {"0.1", "0.01"}
    for candidate in result["candidates"].values():
        assert candidate["status"] == "available"
        assert candidate["base_model"] == "LogisticRegression_no_scaling"
        assert candidate["purge_sessions"] == 21
        assert candidate["overall"]["samples"] >= 300
        assert candidate["overall"]["end_date"] >= "2024-01-01"
