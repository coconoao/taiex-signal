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


class USMarketCacheConcurrencyTests(unittest.TestCase):
    def setUp(self):
        app.US_MARKET_CACHE.update(t=0, v=None)
        app.US_MARKET_FAILURE.update(t=0, error=None)

    def tearDown(self):
        app.US_MARKET_CACHE.update(t=0, v=None)
        app.US_MARKET_FAILURE.update(t=0, error=None)

    def test_concurrent_cache_misses_share_one_refresh(self):
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor

        expected = {"items": [{"symbol": "^GSPC"}]}
        calls = []
        barrier = threading.Barrier(5)

        def fake_fetch():
            calls.append(1)
            time.sleep(0.05)
            app.US_MARKET_CACHE.update(t=app.time.time(), v=expected)
            return expected

        def request():
            barrier.wait()
            return app.us_market_quote()

        with patch.object(app, "_fetch_us_market_quote", side_effect=fake_fetch):
            with ThreadPoolExecutor(max_workers=5) as pool:
                results = list(pool.map(lambda _: request(), range(5)))

        self.assertEqual(len(calls), 1)
        self.assertTrue(all(result is expected for result in results))

    def test_failed_refresh_enters_short_cooldown(self):
        with patch.object(app, "_fetch_us_market_quote", side_effect=RuntimeError("upstream unavailable")) as fetch:
            with self.assertRaisesRegex(RuntimeError, "upstream unavailable"):
                app.us_market_quote()
            with self.assertRaisesRegex(RuntimeError, "短暫冷卻中"):
                app.us_market_quote()
        fetch.assert_called_once()
