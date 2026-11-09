import json
import os
import tempfile
import unittest

import storage


class ForecastDataV2MigrationTests(unittest.TestCase):
    def test_market_sessions_v2_merges_existing_csv_history(self):
        with tempfile.TemporaryDirectory() as directory:
            fixtures = {
                "market_history.csv": (
                    "date,TSM,USDTWD,KOSPI,N225,SOX,US10Y,TAIEX,VIX\n"
                    "2026-11-06,310,31.2,4100,50000,7000,4.1,24000,18\n"
                    "2026-11-09,312,31.1,4110,50100,7100,4.2,24100,17\n"
                ),
                "chips.csv": (
                    "date,foreign_net,trust_net\n"
                    "2026-11-06,100,-20\n"
                    "2026-11-09,200,30\n"
                ),
                "margin.csv": (
                    "date,margin_bal,margin_units,short_units\n"
                    "2026-11-06,1000,80,10\n"
                    "2026-11-09,1100,82,11\n"
                ),
                "futures.csv": (
                    "date,foreign_oi_net\n"
                    "2026-11-06,500\n"
                    "2026-11-09,600\n"
                ),
            }
            for name, value in fixtures.items():
                with open(os.path.join(directory, name), "w", encoding="utf-8") as stream:
                    stream.write(value)

            self.assertEqual(storage.write_market_sessions_v2(directory), 2)
            with open(os.path.join(directory, "market_sessions_v2.jsonl"), encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["schema_version"], 2)
            self.assertEqual(rows[1]["record_type"], "market_session")
            self.assertEqual(rows[1]["trade_date"], "2026-11-09")
            self.assertEqual(rows[1]["instrument"]["close"], 24100.0)
            self.assertEqual(rows[1]["institutional"]["foreign_net"], 200.0)
            self.assertEqual(rows[1]["margin"]["short_units"], 11.0)
            self.assertEqual(rows[1]["futures"]["foreign_open_interest_net"], 600.0)

    def test_legacy_direction_snapshot_is_archived_and_migrated_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = {"asof": "2026-11-09", "prob_up": 0.317, "signal": "舊版方向分類", "demo": False}
            path = os.path.join(directory, "signals.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(legacy, stream, ensure_ascii=False)

            self.assertTrue(storage.migrate_legacy_signals_v2(directory))
            with open(os.path.join(directory, "signals_legacy_v1.json"), encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), legacy)
            with open(path, encoding="utf-8") as stream:
                migrated = json.load(stream)
            self.assertEqual(migrated["schema_version"], 2)
            self.assertEqual(migrated["forecast"]["status"], "pending_backfill")
            self.assertEqual(migrated["legacy_archive"]["key"], "signals_legacy_v1.json")
            self.assertFalse(storage.migrate_legacy_signals_v2(directory))


if __name__ == "__main__":
    unittest.main()
