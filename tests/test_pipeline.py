import unittest

import pandas as pd

from pipeline import (
    BACKFILL_DATES_PER_RUN,
    RECENT_DATES_PER_RUN,
    _pending_dates,
)


class PendingDatesTests(unittest.TestCase):
    def test_limits_backfill_and_keeps_latest_missing_dates(self):
        today = pd.Timestamp("2026-11-06")
        trading_days = pd.bdate_range(end=today, periods=120)

        result = _pending_dates(pd.DataFrame(), today, 365, trading_days)

        self.assertEqual(len(result), BACKFILL_DATES_PER_RUN + RECENT_DATES_PER_RUN)
        self.assertEqual(result[:BACKFILL_DATES_PER_RUN].tolist(), trading_days[:BACKFILL_DATES_PER_RUN].tolist())
        self.assertEqual(result[-RECENT_DATES_PER_RUN:].tolist(), trading_days[-RECENT_DATES_PER_RUN:].tolist())

    def test_excludes_dates_already_saved_and_fills_oldest_gaps_next(self):
        today = pd.Timestamp("2026-11-06")
        trading_days = pd.bdate_range(end=today, periods=100)
        saved = pd.DataFrame(index=trading_days[:40])

        result = _pending_dates(saved, today, 365, trading_days)

        self.assertEqual(len(result), BACKFILL_DATES_PER_RUN + RECENT_DATES_PER_RUN)
        self.assertEqual(result[0], trading_days[40])
        self.assertEqual(result[-RECENT_DATES_PER_RUN:].tolist(), trading_days[-RECENT_DATES_PER_RUN:].tolist())
        self.assertFalse(result.isin(saved.index).any())

    def test_returns_all_dates_when_only_five_or_fewer_are_missing(self):
        today = pd.Timestamp("2026-11-06")
        trading_days = pd.bdate_range(end=today, periods=5)

        result = _pending_dates(pd.DataFrame(), today, 365, trading_days)

        self.assertEqual(result.tolist(), trading_days.tolist())

    def test_respects_days_back_window(self):
        today = pd.Timestamp("2026-11-06")
        trading_days = pd.bdate_range(end=today, periods=100)

        result = _pending_dates(pd.DataFrame(), today, 10, trading_days)

        self.assertTrue((result >= today - pd.Timedelta(days=10)).all())
        self.assertTrue((result <= today).all())


if __name__ == "__main__":
    unittest.main()
