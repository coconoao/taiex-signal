import json
import os
import tempfile
import unittest
from unittest.mock import patch

import news
import news_enhanced_model


class TaiwanNewsRelevanceTests(unittest.TestCase):
    def test_feature_version_counts_only_market_relevant_news(self):
        headlines = [
            {"id": 1, "kind": "market", "title": "Nasdaq stocks rally after strong earnings"},
            {"id": 2, "kind": "taiwan", "title": "台積電營收成長，半導體需求回升"},
            {"id": 3, "kind": "policy", "title": "半導體出口下滑，供應鏈面臨壓力"},
            {"id": 4, "kind": "taiwan", "title": "地方選舉造勢活動週末登場"},
            {"id": 5, "kind": "analyst", "tag": "高盛",
             "title": "Goldman flags fresh buying in China stocks"},
        ]
        features = news.research_features(headlines)
        self.assertEqual(features["feature_version"], 2)
        self.assertEqual(features["headline_count"], 5)
        self.assertEqual(features["relevant_headline_count"], 3)
        self.assertEqual(features["relevant_category_counts"], {
            "market": 1, "taiwan": 1, "policy": 1
        })
        self.assertEqual(features["positive_headline_count"], 2)
        self.assertEqual(features["negative_headline_count"], 1)

    def test_rule_baseline_uses_taiwan_and_policy_news(self):
        headlines = [
            {"id": 1, "kind": "taiwan", "title": "台股上漲，台積電營收成長"},
            {"id": 2, "kind": "policy", "title": "半導體出口下滑，供應鏈需求疲弱"},
            {"id": 3, "kind": "taiwan", "title": "地方活動週末登場"},
        ]
        result = news.lexicon(headlines)
        self.assertGreater(result["next_day"], 0)
        self.assertEqual(result["next_day"], result["next_week"])
        self.assertEqual(len(result["drivers"]), 2)
        self.assertTrue(all("非語意判讀" in item["why"] for item in result["drivers"]))

    def test_analyst_view_requires_institution_taiwan_and_explicit_stance(self):
        irrelevant = {
            "id": 1, "kind": "analyst", "tag": "高盛",
            "title": "Goldman flags fresh state buying in China stocks",
        }
        not_attributed = {
            "id": 2, "kind": "analyst", "tag": "花旗",
            "title": "Taiwan shares close lower but recover from lows - Focus Taiwan",
        }
        explicit = {
            "id": 3, "kind": "analyst", "tag": "高盛",
            "title": "Goldman Sachs says Taiwan stocks are overweight",
        }
        negated = {
            "id": 4, "kind": "analyst", "tag": "高盛",
            "title": "Goldman Sachs says Taiwan stocks are not bearish",
        }
        quoted_not_a_view = {
            "id": 5, "kind": "analyst", "tag": "野村",
            "title": "Taiwan Stocks Plunge 1,612 Points; Guo Cites Nomura Report: "
                     "Negative Cash Flow Misread, Not a Bearish Signal",
        }
        self.assertIsNone(news.explicit_analyst_view(irrelevant))
        self.assertIsNone(news.explicit_analyst_view(not_attributed))
        self.assertEqual(news.explicit_analyst_view(explicit), "看多")
        self.assertIsNone(news.explicit_analyst_view(negated))
        self.assertIsNone(news.explicit_analyst_view(quoted_not_a_view))

    def test_ai_analyst_calls_are_checked_against_source_title(self):
        headline = {
            "id": 7, "kind": "analyst", "tag": "高盛",
            "title": "Goldman Sachs sees Taiwan stocks as overweight",
        }
        self.assertEqual(
            news.validate_analyst_calls(
                [{"id": 7, "name": "高盛", "stance": "看多"}], [headline]
            ),
            [{"id": 7, "name": "高盛", "stance": "看多"}],
        )
        self.assertEqual(
            news.validate_analyst_calls(
                [{"id": 7, "name": "高盛", "stance": "看空"}], [headline]
            ),
            [],
        )

    def test_news_model_rejects_old_feature_schema(self):
        self.assertIsNone(news_enhanced_model.vector(
            {"feature_version": 1, "headline_count": 100},
            {"prob_up": 0.6, "missing_layer_count": 0},
        ))

    def test_news_model_accepts_new_feature_schema(self):
        vector = news_enhanced_model.vector(
            {
                "feature_version": 2,
                "relevant_headline_count": 20,
                "relevant_category_counts": {
                    "market": 8, "taiwan": 7, "policy": 5
                },
                "news_environment_score": 0.1,
                "positive_headline_count": 8,
                "negative_headline_count": 6,
                "published_time_coverage_pct": 90,
            },
            {"prob_up": 0.6, "missing_layer_count": 1},
        )
        self.assertEqual(vector.shape, (10,))


    def test_existing_same_day_snapshot_backfills_later_market_context(self):
        articles = [{
            "kind": "market",
            "title": "Nasdaq stocks rally after strong earnings",
            "source": "Example",
            "published": "2026-11-09T12:00:00Z",
            "link": "https://example.com/story",
        }]
        features = news.research_features(articles)
        original = {
            "asof": "2026-11-09",
            "captured_at": "2026-11-09T12:30:00+00:00",
            "market_context": {},
            "features": features,
            "articles": articles,
            "outcome": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                with open("signals.json", "w", encoding="utf-8") as stream:
                    json.dump({
                        "asof": "2026-11-09",
                        "prob_up": 0.61,
                        "missing_layers": ["chips"],
                    }, stream)
                with open("news_research_history.jsonl", "w", encoding="utf-8") as stream:
                    stream.write(json.dumps(original, ensure_ascii=False) + "\n")
                with patch.object(news, "latest_market_asof", return_value="2026-11-09"):
                    news.save_research_snapshot([{"kind": "market", "title": "new title"}])
                with open("news_research_history.jsonl", encoding="utf-8") as stream:
                    repaired = json.loads(stream.readline())
            finally:
                os.chdir(previous)

        self.assertEqual(repaired["market_context"], {
            "prob_up": 0.61,
            "missing_layer_count": 1,
        })
        self.assertEqual(repaired["articles"], articles)
        self.assertEqual(repaired["features"], features)
        self.assertEqual(repaired["captured_at"], original["captured_at"])

if __name__ == "__main__":
    unittest.main()
