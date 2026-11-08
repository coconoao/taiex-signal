import datetime as dt

import pandas as pd

import tw_market_data


def test_roc_date_converts_taiwan_calendar():
    assert tw_market_data._roc_date("115/11/06") == dt.date(2026, 11, 6)


def test_number_handles_exchange_formats():
    assert tw_market_data._number("+1,234.50") == 1234.5
    assert tw_market_data._number("-12.00") == -12.0
    assert tw_market_data._number("X0.00") == 0.0
    assert tw_market_data._number("--") is None


def test_adjusted_close_uses_exchange_reference_price_change():
    rows = [
        {"date": dt.date(2026, 1, 2), "raw_close": 100.0, "change": 0.0,
         "volume": 1000, "open": 100, "high": 100, "low": 100},
        # Ex-dividend reference price is 95; a 95 close is a flat adjusted return.
        {"date": dt.date(2026, 1, 5), "raw_close": 95.0, "change": 0.0,
         "volume": 2000, "open": 95, "high": 95, "low": 95},
        {"date": dt.date(2026, 1, 6), "raw_close": 96.0, "change": 1.0,
         "volume": 3000, "open": 95, "high": 97, "low": 95},
    ]
    frame = tw_market_data._to_adjusted_frame(rows, "TWSE")
    assert frame["Close"].iloc[0] == 100.0
    assert frame["Close"].iloc[1] == 100.0
    assert round(frame["Close"].iloc[2], 4) == round(100 * 96 / 95, 4)
    assert frame["RawClose"].tolist() == [100.0, 95.0, 96.0]
    assert frame.attrs["source"] == "TWSE"


def test_adjusted_series_fails_closed_on_unparsed_change():
    rows = [
        {"date": dt.date(2026, 1, 2), "raw_close": 100.0, "change": 0.0,
         "volume": 1000, "open": 100, "high": 100, "low": 100},
        {"date": dt.date(2026, 1, 5), "raw_close": 95.0, "change": None,
         "volume": 2000, "open": 95, "high": 95, "low": 95},
    ]
    try:
        tw_market_data._to_adjusted_frame(rows, "TPEx")
    except tw_market_data.MarketDataError as exc:
        assert "無法解析" in str(exc)
    else:
        raise AssertionError("Unparsed adjustment must not silently enter model data")
