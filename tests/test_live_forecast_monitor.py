import datetime as dt
import json
import os
import tempfile
import unittest
from zoneinfo import ZoneInfo

from forecast_monitor import (
    append_daily_live_snapshot,
    build_live_snapshot_record,
    read_live_snapshot_history,
    summarize_live_forecast_quality,
)

TZ = ZoneInfo("Asia/Taipei")


def snapshot(asof="2026-11-20"):
    return {
        "schema_version": 2,
        "record_type": "index_forecast_snapshot",
        "demo": False,
        "asof": asof,
        "forecast": {
            "schema_version": 2,
            "record_type": "index_forecast_snapshot",
            "asof": asof,
            "status": "research",
            "base_close": 10000,
            "method": "direct_horizon_ridge",
            "model_version": "ridge-direct-3session-v2",
            "sessions": [
                {
                    "horizon_sessions": horizon,
                    "target_date": f"2026-11-{20 + horizon:02d}",
                    "status": "research",
                    "forecast_close": 10000 + horizon * 100,
                    "predicted_return_pct": float(horizon),
                }
                for horizon in (1, 2, 3)
            ],
        },
    }


class LiveForecastMonitorTests(unittest.TestCase):
    def test_build_requires_complete_non_demo_production_snapshot(self):
        record = build_live_snapshot_record(snapshot())
        self.assertEqual(record["record_type"], "index_forecast_live_snapshot")
        self.assertEqual(record["evaluation"], "prospective_live_snapshot")
        self.assertEqual([row["horizon_sessions"] for row in record["sessions"]], [1, 2, 3])

        demo = snapshot()
        demo["demo"] = True
        self.assertIsNone(build_live_snapshot_record(demo))

        incomplete = snapshot()
        incomplete["forecast"]["sessions"].pop()
        self.assertIsNone(build_live_snapshot_record(incomplete))

    def test_append_waits_until_after_close_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "index_forecast_live_snapshots.jsonl")
            before_close = dt.datetime(2026, 11, 23, 15, 59, tzinfo=TZ)
            after_close = dt.datetime(2026, 11, 23, 16, 5, tzinfo=TZ)

            early = append_daily_live_snapshot(path, snapshot(), now=before_close)
            self.assertFalse(early["appended"])
            self.assertEqual(early["reason"], "before_post_close_cutoff")
            self.assertFalse(os.path.exists(path))

            first = append_daily_live_snapshot(path, snapshot(), now=after_close)
            second = append_daily_live_snapshot(path, snapshot(), now=after_close)
            self.assertTrue(first["appended"])
            self.assertFalse(second["appended"])
            self.assertEqual(second["reason"], "already_recorded")
            self.assertEqual(len(read_live_snapshot_history(path)), 1)

    def test_matured_live_snapshots_are_compared_with_flat_baseline(self):
        record = build_live_snapshot_record(snapshot())
        closes = {
            "2026-11-21": 10100,
            "2026-11-22": 9900,
            "2026-11-23": 10300,
        }
        summary = summarize_live_forecast_quality([record], closes, window_sessions=20, minimum_samples=20)
        rows = summary["horizons"]
        self.assertEqual([row["horizon_sessions"] for row in rows], [1, 2, 3])
        self.assertTrue(all(row["sample_count"] == 1 for row in rows))
        self.assertTrue(all(row["status"] == "provisional" for row in rows))
        self.assertEqual(rows[0]["model_mae_points"], 0.0)
        self.assertEqual(rows[0]["flat_baseline_mae_points"], 100.0)
        self.assertEqual(rows[1]["model_mae_points"], 300.0)
        self.assertEqual(rows[1]["flat_baseline_mae_points"], 100.0)

    def test_unmatured_targets_are_not_scored(self):
        record = build_live_snapshot_record(snapshot())
        summary = summarize_live_forecast_quality([record], {"2026-11-21": 10100})
        self.assertEqual(len(summary["horizons"]), 1)
        self.assertEqual(summary["horizons"][0]["horizon_sessions"], 1)


if __name__ == "__main__":
    unittest.main()
