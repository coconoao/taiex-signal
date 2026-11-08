import os
import unittest
from unittest.mock import patch

import news


class CloudflareWorkersAITests(unittest.TestCase):
    def test_cloudflare_provider_uses_account_token_and_json_chat_request(self):
        response_text = '{"next_day": 0.1, "next_week": -0.1, "confidence": 0.5, "drivers": [], "summary": "測試", "analysts": []}'
        with patch.dict(os.environ, {
            "CLOUDFLARE_ACCOUNT_ID": "a" * 32,
            "CLOUDFLARE_API_TOKEN": "test-token",
        }, clear=False), patch.object(news, "post", return_value={
            "result": {"response": response_text}
        }) as mocked_post:
            actual = news.ask_cloudflare("請輸出 JSON")

        self.assertEqual(actual, response_text)
        args, kwargs = mocked_post.call_args
        self.assertIn("/accounts/" + "a" * 32 + "/ai/run/@cf/zai-org/glm-4.7-flash", args[0])
        self.assertEqual(args[1], {"Authorization": "Bearer test-token"})
        self.assertEqual(args[2]["messages"], [{"role": "user", "content": "請輸出 JSON"}])
        self.assertEqual(args[2]["response_format"], {"type": "json_object"})
        self.assertEqual(args[2]["max_tokens"], 2000)

    def test_cloudflare_provider_rejects_invalid_account_id_before_request(self):
        with patch.dict(os.environ, {
            "CLOUDFLARE_ACCOUNT_ID": "not-an-account-id",
            "CLOUDFLARE_API_TOKEN": "test-token",
        }, clear=False), patch.object(news, "post") as mocked_post:
            with self.assertRaisesRegex(ValueError, "ACCOUNT_ID"):
                news.ask_cloudflare("prompt")
        mocked_post.assert_not_called()

    def test_cloudflare_provider_requires_text_response(self):
        with patch.dict(os.environ, {
            "CLOUDFLARE_ACCOUNT_ID": "b" * 32,
            "CLOUDFLARE_API_TOKEN": "test-token",
        }, clear=False), patch.object(news, "post", return_value={"result": {}}):
            with self.assertRaisesRegex(ValueError, "未回傳文字結果"):
                news.ask_cloudflare("prompt")

    def test_cloudflare_provider_is_registered_with_both_runtime_secrets(self):
        self.assertEqual(
            news.MODELS["Cloudflare"][1],
            ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"],
        )


if __name__ == "__main__":
    unittest.main()
