"""Leakage and warmup tests for final OOS research primitives."""
import numpy as np
import pandas as pd

from scripts.final_oos_methods import (
    atr_percent,
    classify_fixed_and_atr,
    fit_expanding_platt,
    purged_prior_dates,
)


def test_atr_warmup_falls_back_to_fixed_labels_then_uses_trailing_atr():
    idx = pd.bdate_range("2024-01-02", periods=25)
    close = pd.Series(100.0 + np.arange(25), index=idx)
    high = close + 1.0
    low = close - 1.0
    atr = atr_percent(high, low, close, window=20)
    future = pd.Series(0.008, index=idx)
    result = classify_fixed_and_atr(future, atr)
    assert result.loc[idx[:19], "atr_warmup_fallback"].all()
    assert (result.loc[idx[:19], "atr"] == result.loc[idx[:19], "fixed"]).all()
    assert not result.loc[idx[19:], "atr_warmup_fallback"].any()


def test_atr_is_trailing_only_and_does_not_change_when_future_rows_change():
    idx = pd.bdate_range("2024-01-02", periods=30)
    close = pd.Series(100.0 + np.arange(30), index=idx)
    high, low = close + 1.0, close - 1.0
    first = atr_percent(high, low, close, 20)
    high2, low2 = high.copy(), low.copy()
    high2.iloc[25:] += 500
    low2.iloc[25:] -= 500
    second = atr_percent(high2, low2, close, 20)
    pd.testing.assert_series_equal(first.iloc[:25], second.iloc[:25])


def test_purge_uses_full_calendar_positions_not_filtered_rows():
    calendar = pd.bdate_range("2024-01-02", periods=40)
    candidates = calendar[[0, 5, 10, 15, 20]]
    kept = purged_prior_dates(calendar, candidates, calendar[25], 21)
    assert kept.tolist() == [calendar[0]]


def test_expanding_platt_never_uses_unmatured_or_same_block_labels():
    calendar = pd.bdate_range("2024-01-02", periods=500)
    dates = calendar[::2]
    raw = pd.Series(np.linspace(0.1, 0.9, len(dates)), index=dates)
    actual = pd.Series((np.arange(len(dates)) % 2), index=dates)
    result = fit_expanding_platt(
        raw, actual, calendar, horizon_sessions=21, block_size=10,
        min_samples=20, min_per_class=5, c_value=0.1,
    )
    for date, row in result.iterrows():
        fit_end = row["calibration_fit_end"]
        if fit_end is not None:
            assert pd.Timestamp(fit_end) < date
            fit_pos = calendar.get_loc(pd.Timestamp(fit_end))
            test_pos = calendar.get_loc(date)
            assert fit_pos + 21 <= test_pos
    first_block = result.iloc[:10]
    assert (first_block["calibration_status"] == "raw_insufficient_matured_history").all()
