import io
import os
import unittest
from unittest.mock import patch

import news


class GeminiNewsLatencyTests(unittest.TestCase):
    def test_post_retries_one_read_timeout(self):
        response = io.BytesIO(b'{"ok": true}')
        with (
            patch.object(news.urllib.request, "urlopen",
                         side_effect=[TimeoutError("read timed out"), response]) as urlopen,
            patch.object(news.time, "sleep") as sleep,
        ):
            result = news.post("https://example.invalid", {}, {"test": True})

        self.assertEqual(result, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_post_does_not_retry_non_timeout_url_error(self):
        error = news.urllib.error.URLError("connection refused")
        with patch.object(news.urllib.request, "urlopen", side_effect=error) as urlopen:
            with self.assertRaises(news.urllib.error.URLError):
                news.post("https://example.invalid", {}, {"test": True})
        self.assertEqual(urlopen.call_count, 1)

    def test_gemini_uses_low_thinking_level_for_news_classification(self):
        response = {
            "candidates": [{
                "content": {"parts": [{"text": '{"next_day":0,"next_week":0,"confidence":0.5,"drivers":[],"summary":"中性","analysts":[]}'}]}
            }]
        }
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(news, "post", return_value=response) as post,
        ):
            text = news.ask_gemini("測試新聞")

        self.assertIn('"next_day":0', text)
        payload = post.call_args.args[2]
        self.assertEqual(
            payload["generationConfig"]["thinkingConfig"]["thinkingLevel"], "low"
        )


if __name__ == "__main__":
    unittest.main()
