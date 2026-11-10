import json
import os
import tempfile
import unittest
from unittest.mock import patch

import news


SAMPLE_HEADLINES = [
    {"id": 0, "kind": "market", "tag": "美股", "title": "Nasdaq stocks rally after strong gains",
     "description": "", "link": "https://example.com/1", "published": "today", "source": "Example"},
    {"id": 1, "kind": "market", "tag": "半導體", "title": "Semiconductor shares rise on strong demand",
     "description": "", "link": "https://example.com/2", "published": "today", "source": "Example"},
    {"id": 2, "kind": "taiwan", "tag": "台灣產經", "title": "台灣出口成長，半導體需求強勁",
     "description": "", "link": "https://example.com/3", "published": "today", "source": "CNA"},
    {"id": 3, "kind": "policy", "tag": "台灣政府政策", "title": "政府公布產業投資補助政策",
     "description": "", "link": "https://example.com/4", "published": "today", "source": "政府"},
]


class NewsRulesOnlyTests(unittest.TestCase):
    def test_news_module_has_no_generative_ai_provider(self):
        self.assertFalse(hasattr(news, "ask_gemini"))
        self.assertFalse(hasattr(news, "MODELS"))

    def test_lexicon_returns_explicit_non_ai_result(self):
        result = news.lexicon(SAMPLE_HEADLINES)
        self.assertIn("next_day", result)
        self.assertIn("next_week", result)
        self.assertEqual(result["confidence"], 0.3)
        self.assertIn("非語意模型", result["summary"])

    def test_main_writes_news_without_any_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(news, "headlines", return_value=SAMPLE_HEADLINES),
                patch.object(news, "save_research_snapshot"),
                patch.object(news, "save_policy_event_history", return_value=[]),
                patch.object(news, "policy_events", return_value=[]),
            ):
                old = os.getcwd()
                try:
                    os.chdir(directory)
                    news.main()
                    with open("news.json", encoding="utf-8") as stream:
                        payload = json.load(stream)
                    self.assertEqual(payload["analysis_mode"], "rules_only")
                    self.assertEqual(payload["headlines"], SAMPLE_HEADLINES)
                    self.assertIn("標題規則整理", payload["models"])
                    self.assertEqual(payload["model_status"], {})
                    with open("news_history.jsonl", encoding="utf-8") as stream:
                        history = json.loads(stream.readline())
                    self.assertEqual(history["analysis_mode"], "rules_only")
                finally:
                    os.chdir(old)


if __name__ == "__main__":
    unittest.main()
