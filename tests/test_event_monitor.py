import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import event_monitor

UTC = dt.timezone.utc


class EventMonitorTests(unittest.TestCase):
    def setUp(self):
        self.watchlist = {
            "entities": [
                {"entity_id": "person:donald-trump", "display_name": "Donald Trump",
                 "entity_type": "person", "aliases": ["Donald Trump", "President Trump", "Trump"]},
                {"entity_id": "person:jensen-huang", "display_name": "Jensen Huang",
                 "entity_type": "person", "aliases": ["Jensen Huang", "黃仁勳"]},
                {"entity_id": "role:fed-chair", "display_name": "Federal Reserve Chair",
                 "entity_type": "role", "aliases": ["Federal Reserve Chair", "Fed Chair"]},
            ]
        }
        self.now = dt.datetime(2026, 11, 10, 0, 30, tzinfo=UTC)

    def item(self, **overrides):
        value = {
            "id": 1, "kind": "policy", "tag": "US policy",
            "title": "President Trump announces new semiconductor export policy",
            "description": "A policy update affecting Taiwan chip supply chains.",
            "link": "https://www.whitehouse.gov/briefing-room/example",
            "published": "Tue, 10 Nov 2026 00:00:00 GMT",
            "source": "White House",
        }
        value.update(overrides)
        return value

    def test_parses_timezone_aware_publication_time(self):
        value, zone, quality = event_monitor.parse_published("Tue, 10 Nov 2026 00:00:00 GMT")
        self.assertEqual(value, "2026-11-10T00:00:00Z")
        self.assertIsNotNone(zone)
        self.assertEqual(quality, "source_timestamp_verified")

    def test_does_not_guess_timezone_for_naive_timestamp(self):
        value, zone, quality = event_monitor.parse_published("2026-11-10 08:00:00")
        self.assertIsNone(value)
        self.assertIsNone(zone)
        self.assertEqual(quality, "source_timestamp_unzoned")

    def test_matches_named_person_and_role(self):
        item = self.item(title="President Trump met the Federal Reserve Chair")
        matches = event_monitor.match_entities(item, self.watchlist)
        self.assertEqual({row["entity_id"] for row in matches},
                         {"person:donald-trump", "role:fed-chair"})
        self.assertEqual(next(row for row in matches if row["entity_id"] == "role:fed-chair")["entity_type"], "role")

    def test_does_not_match_ambiguous_trump_without_context(self):
        item = self.item(title="Trump card game tournament announced", description="")
        self.assertEqual(event_monitor.match_entities(item, self.watchlist), [])

    def test_event_is_candidate_and_not_directional_forecast(self):
        event = event_monitor.build_event(self.item(), self.watchlist, self.now)
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "policy_trade")
        self.assertEqual(event["taiwan_impact"]["direction_scenario"], "unknown")
        self.assertEqual(event["taiwan_impact"]["confidence"], 0.25)
        self.assertEqual(event["extraction"]["method"], "rules")
        self.assertEqual(event["evidence"]["status"], "official_confirmed")
        self.assertEqual(event["timestamps"]["forecast_eligible_at_utc"], "2026-11-10T00:30:00Z")

    def test_allegation_is_flagged_for_review_not_confirmed(self):
        event = event_monitor.build_event(
            self.item(title="Report says alleged intelligence operation targeted Jensen Huang"),
            self.watchlist, self.now
        )
        self.assertEqual(event["evidence"]["status"], "allegation")
        self.assertTrue(event["evidence"]["review_required"])
        self.assertEqual(event["evidence"]["review_status"], "needs_followup")

    def test_rejects_missing_or_non_http_source_url(self):
        self.assertIsNone(event_monitor.build_event(self.item(link=""), self.watchlist, self.now))
        self.assertIsNone(event_monitor.build_event(self.item(link="javascript:alert(1)"), self.watchlist, self.now))

    def test_first_seen_is_immutable_on_repeat_observation(self):
        first = event_monitor.build_event(self.item(), self.watchlist, self.now)
        first["timestamps"]["first_seen_at_utc"] = "2026-11-09T23:00:00Z"
        later = event_monitor.build_event(
            self.item(), self.watchlist, self.now + dt.timedelta(hours=3)
        )
        merged = event_monitor.merge_history([first], [later])
        self.assertEqual(merged[0]["timestamps"]["first_seen_at_utc"], "2026-11-09T23:00:00Z")
        self.assertEqual(merged[0]["timestamps"]["forecast_eligible_at_utc"], "2026-11-09T23:00:00Z")
        self.assertEqual(merged[0]["timestamps"]["last_seen_at_utc"], "2026-11-10T03:30:00Z")

    def test_material_source_update_preserves_prior_version(self):
        first = event_monitor.build_event(self.item(), self.watchlist, self.now)
        revised_item = self.item(description="Updated White House release with additional policy detail.")
        revised = event_monitor.build_event(
            revised_item, self.watchlist, self.now + dt.timedelta(hours=2)
        )
        merged = event_monitor.merge_history([first], [revised])
        versions = sorted(row["version"] for row in merged)
        self.assertEqual(versions, [1, 2])
        latest = next(row for row in merged if row["version"] == 2)
        self.assertTrue(latest["classification"]["is_material_update"])
        self.assertEqual(latest["classification"]["supersedes_event_version"], 1)
        self.assertEqual(latest["timestamps"]["first_seen_at_utc"], "2026-11-10T00:30:00Z")

    def test_run_writes_local_research_jsonl_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            watch_path = root / "watchlist.json"
            output = root / "events.jsonl"
            watch_path.write_text(json.dumps(self.watchlist), encoding="utf-8")
            with patch("event_monitor.news.headlines", return_value=[self.item()]):
                result = event_monitor.run(output, watch_path, now=self.now)
            self.assertEqual(result["status"], "research_only")
            self.assertEqual(result["events_written"], 1)
            rows = event_monitor.load_jsonl(output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], event_monitor.build_event(self.item(), self.watchlist, self.now)["event_id"])


if __name__ == "__main__":
    unittest.main()
