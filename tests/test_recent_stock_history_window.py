import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import stock
import tw_market_data


class TestRecentStockHistoryWindow(unittest.TestCase):
    def test_stock_analysis_requests_four_years_for_stock_and_market(self):
        dates = pd.bdate_range("2025-01-01", periods=300)
        rng = np.random.default_rng(7)
        stock_close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates))), index=dates)
        market_close = pd.Series(20000 * np.cumprod(1 + rng.normal(0.0002, 0.008, len(dates))), index=dates)
        frame = pd.DataFrame({
            "Close": stock_close,
            "RawClose": stock_close,
            "Volume": 100000.0,
        }, index=dates)
        frame.attrs["source"] = "TWSE"

        with (
            patch.object(stock.tw_market_data, "load_stock_history", return_value=frame) as stock_loader,
            patch.object(stock.tw_market_data, "load_taiex_history", return_value=market_close) as market_loader,
            patch("os.path.exists", return_value=False),
            patch("stock_model_research.analyze", return_value={"status": "insufficient"}),
        ):
            result = stock.analyze("2330")

        stock_loader.assert_called_once_with("2330", None, years=4)
        market_loader.assert_called_once_with(years=4)
        self.assertEqual(result["history_years_requested"], 4)
        self.assertEqual(result["long_term_model"]["status"], "insufficient")

    def test_four_year_window_is_about_49_monthly_requests(self):
        months = tw_market_data._month_starts(years=4)
        self.assertEqual(len(months), 49)


if __name__ == "__main__":
    unittest.main()
