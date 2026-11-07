import unittest
from unittest.mock import MagicMock, patch
import urllib.error

import app


class GeminiNetworkRetryTests(unittest.TestCase):
    def test_retries_one_timeout_then_returns_response(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"candidates":[{"content":{"parts":[{"text":"{}"}]}}]}'
        with patch.object(app.urllib.request, "urlopen", side_effect=[TimeoutError("timed out"), response]) as urlopen, \
             patch.object(app.time, "sleep") as sleep:
            result = app.call_ai_provider("Gemini", "test-key", "gemini-2.5-flash", "test prompt")
        self.assertEqual(result, "{}")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()

    def test_returns_safe_error_after_second_network_failure(self):
        with patch.object(app.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")) as urlopen, \
             patch.object(app.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "Gemini 連線逾時或網路中斷"):
                app.call_ai_provider("Gemini", "test-key", "gemini-2.5-flash", "test prompt")
        self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
