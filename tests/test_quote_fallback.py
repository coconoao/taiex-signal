import datetime as dt
import unittest
from unittest.mock import patch

import app


class QuoteFallbackTests(unittest.TestCase):
    def setUp(self):
        app.QUOTE.update(t=0, v=None)

    def tearDown(self):
        app.QUOTE.update(t=0, v=None)

    def test_quote_uses_yahoo_daily_close_when_twse_fallback_is_unavailable(self):
        expected = {
            "last": 23000.0,
            "change": 100.0,
            "pct": 0.44,
            "time": "2026-11-06",
            "market": "spot_close",
            "label": "加權指數最近收盤（Yahoo）",
            "source": "Yahoo Finance 日線資料（非即時）",
            "delay_note": "官方 TWSE 收盤來源暫不可用；此數值為最近已完成交易日收盤",
        }
        with patch.object(app, "_spot_session", return_value=False), \
             patch.object(app, "_afterhours_session", return_value=False), \
             patch.object(app, "_twse_last_close_quote", return_value=None), \
             patch.object(app, "_yahoo_last_close_quote", return_value=expected):
            self.assertEqual(app.quote(), expected)

    def test_quote_keeps_official_twse_close_ahead_of_yahoo_daily_fallback(self):
        official = {
            "last": 22900.0,
            "change": 50.0,
            "pct": 0.22,
            "time": "2026-11-06",
            "market": "spot_close",
            "label": "加權指數最近官方收盤",
            "source": "TWSE 官方日收盤資料（非即時）",
            "delay_note": "Yahoo 即時來源不可用或目前非交易時段；此數值為最近已完成交易日收盤",
        }
        with patch.object(app, "_spot_session", return_value=False), \
             patch.object(app, "_afterhours_session", return_value=False), \
             patch.object(app, "_twse_last_close_quote", return_value=official), \
             patch.object(app, "_yahoo_last_close_quote") as yahoo:
            self.assertEqual(app.quote(), official)
            yahoo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
