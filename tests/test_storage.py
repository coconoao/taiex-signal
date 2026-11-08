import os
import tempfile
import unittest
from unittest.mock import patch

import storage


class StorageTests(unittest.TestCase):
    def test_storage_is_disabled_without_explicit_opt_in(self):
        with patch.dict(os.environ, {"TAIEX_STORAGE": "", "SUPABASE_URL": "https://example.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "dummy"}):
            self.assertFalse(storage.enabled())

    def test_restore_ignores_unrecognized_keys_and_writes_known_files(self):
        rows = [
            {"key": "signals.json", "content": "{\"ok\": true}"},
            {"key": "../secrets.txt", "content": "must not write"},
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(storage, "enabled", return_value=True), patch.object(storage, "_request", return_value=rows):
            self.assertEqual(storage.restore_data(directory), 1)
            with open(os.path.join(directory, "signals.json"), encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "{\"ok\": true}")
            self.assertFalse(os.path.exists(os.path.join(directory, "..", "secrets.txt")))

    def test_dedicated_research_files_are_excluded_from_routine_saves(self):
        self.assertNotIn("news_gdelt_history.jsonl", storage.AUTO_SAVE_FILES)
        self.assertNotIn("news_gdelt_research.json", storage.AUTO_SAVE_FILES)
        self.assertNotIn("news_policy_archive.json", storage.AUTO_SAVE_FILES)
        self.assertNotIn("news_policy_research.json", storage.AUTO_SAVE_FILES)

    def test_save_only_known_existing_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(storage, "enabled", return_value=True), patch.object(storage, "_request") as request:
            with open(os.path.join(directory, "news.json"), "w", encoding="utf-8") as stream:
                stream.write("{}")
            self.assertEqual(storage.save_data(directory), 1)
            self.assertEqual(request.call_args.args[0], "POST")
            self.assertEqual(request.call_args.args[1], "site_data?on_conflict=key")
            self.assertEqual(request.call_args.args[2][0]["key"], "news.json")


if __name__ == "__main__":
    unittest.main()
