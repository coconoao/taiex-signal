import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import research_regime_backtest as backtest


def test_official_calendar_passes_only_on_exact_match(monkeypatch):
    expected = pd.date_range("2026-01-05", periods=3, freq="B")
    monkeypatch.setattr(
        backtest.pipeline,
        "_expected_twse_trading_days",
        lambda start, end: (expected, "twse_official_holiday_schedule"),
    )
    result = backtest._validate_taiwan_calendar(expected)
    assert result["passed"] is True
    assert result["missing_count"] == 0
    assert result["unexpected_count"] == 0


def test_missing_official_session_fails(monkeypatch):
    expected = pd.date_range("2026-01-05", periods=3, freq="B")
    monkeypatch.setattr(
        backtest.pipeline,
        "_expected_twse_trading_days",
        lambda start, end: (expected, "twse_official_holiday_schedule"),
    )
    result = backtest._validate_taiwan_calendar(expected.delete(1))
    assert result["passed"] is False
    assert result["missing_count"] == 1


def test_unavailable_official_calendar_fails_closed(monkeypatch):
    observed = pd.date_range("2026-01-05", periods=3, freq="B")
    monkeypatch.setattr(
        backtest.pipeline,
        "_expected_twse_trading_days",
        lambda start, end: (pd.DatetimeIndex([]), "unavailable"),
    )
    result = backtest._validate_taiwan_calendar(observed)
    assert result["passed"] is False
    assert result["source"] == "unavailable"
    assert result["expected_sessions"] == 0
