import unittest
from unittest.mock import patch

import news


class NewsModelStatusTests(unittest.TestCase):
    def test_gemini_is_the_only_configured_news_ai_provider(self):
        self.assertEqual(set(news.MODELS), {"Gemini"})

    def test_gemini_is_reported_as_not_configured_without_server_key(self):
        with patch.dict(news.MODELS, {"Gemini": (news.ask_gemini, ["GEMINI_API_KEY"])}, clear=True):
            status = news.model_statuses({}, {})
        self.assertEqual(status["Gemini"]["status"], "not_configured")
        self.assertNotIn("GEMINI_API_KEY", str(status["Gemini"]))

    def test_gemini_is_reported_as_ready_after_successful_analysis(self):
        with patch.dict(news.MODELS, {"Gemini": (news.ask_gemini, ["GEMINI_API_KEY"])}, clear=True):
            status = news.model_statuses({"Gemini": news.ask_gemini}, {"Gemini": {"next_day": 0.2}})
        self.assertEqual(status["Gemini"], {"status": "ok"})

    def test_gemini_api_failure_is_reported_without_losing_provider_name(self):
        with patch.dict(news.MODELS, {"Gemini": (news.ask_gemini, ["GEMINI_API_KEY"])}, clear=True):
            status = news.model_statuses({"Gemini": news.ask_gemini}, {"Gemini": {"error": "HTTPError: 403"}})
        self.assertEqual(status["Gemini"]["status"], "error")
        self.assertIn("403", status["Gemini"]["message"])


if __name__ == "__main__":
    unittest.main()
