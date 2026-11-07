import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from pipeline import fetch_chips, fetch_futures, fetch_margin


DAY = pd.Timestamp("2026-11-06")


class FinMindFallbackTests(unittest.TestCase):
    def test_chips_fallback_maps_foreign_and_trust_net(self):
        alt = pd.DataFrame([
            {"date": "2026-11-06", "name": "Foreign_Investor", "buy": 100, "sell": 40},
            {"date": "2026-11-06", "name": "Foreign_Dealer_Self", "buy": 10, "sell": 2},
            {"date": "2026-11-06", "name": "Investment_Trust", "buy": 30, "sell": 10},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/chips.csv"
            with patch("pipeline._read_with_backoff", side_effect=TimeoutError), \
                 patch("pipeline._fetch_finmind_data", return_value=alt), \
                 patch("pipeline.time.sleep"):
                result = fetch_chips(path=path, trading_days=pd.DatetimeIndex([DAY]))
        self.assertEqual(result.loc[DAY, "foreign_net"], 68)
        self.assertEqual(result.loc[DAY, "trust_net"], 20)

    def test_margin_fallback_preserves_currency_units(self):
        alt = pd.DataFrame([
            {"date": "2026-11-06", "name": "MarginPurchaseMoney", "TodayBalance": 1234567},
            {"date": "2026-11-06", "name": "MarginPurchase", "TodayBalance": 1000},
            {"date": "2026-11-06", "name": "ShortSale", "TodayBalance": 200},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/margin.csv"
            with patch("pipeline._read_with_backoff", side_effect=TimeoutError), \
                 patch("pipeline._fetch_finmind_data", return_value=alt), \
                 patch("pipeline.time.sleep"):
                result = fetch_margin(path=path, trading_days=pd.DatetimeIndex([DAY]))
        self.assertEqual(result.loc[DAY, "margin_bal"], 1234567)
        self.assertEqual(result.loc[DAY, "margin_units"], 1000)
        self.assertEqual(result.loc[DAY, "short_units"], 200)

    def test_futures_fallback_calculates_foreign_open_interest_net(self):
        alt = pd.DataFrame([{
            "date": "2026-11-06",
            "institutional_investors": "外資",
            "long_open_interest_balance_volume": 1000,
            "short_open_interest_balance_volume": 700,
        }])
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/futures.csv"
            with patch("pipeline._fetch_taifex_openapi_latest", side_effect=TimeoutError), \
                 patch("pipeline._read_with_backoff", side_effect=TimeoutError), \
                 patch("pipeline._fetch_finmind_data", return_value=alt), \
                 patch("pipeline.time.sleep"):
                result = fetch_futures(path=path, trading_days=pd.DatetimeIndex([DAY]))
        self.assertEqual(result.loc[DAY, "foreign_oi_net"], 300)


if __name__ == "__main__":
    unittest.main()
