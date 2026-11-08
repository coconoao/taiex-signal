import unittest

from cloudflare_ai_response import extract_text


class CloudflareAIResponseTests(unittest.TestCase):
    def test_documented_workers_ai_envelope(self):
        self.assertEqual(
            extract_text({"success": True, "result": {"response": "diagnosis"}}),
            "diagnosis",
        )

    def test_openai_compatible_choices_envelope(self):
        self.assertEqual(
            extract_text({"choices": [{"message": {"content": "diagnosis"}}]}),
            "diagnosis",
        )

    def test_text_content_parts(self):
        self.assertEqual(
            extract_text({"result": {"response": [{"type": "text", "text": "part one"}, {"text": "part two"}]}}),
            "part one\npart two",
        )

    def test_http_200_with_cloudflare_errors_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Cloudflare error 1001"):
            extract_text({
                "success": False,
                "errors": [{"code": 1001, "message": "model unavailable"}],
            })

    def test_missing_text_reports_structure_without_dumping_body(self):
        with self.assertRaisesRegex(ValueError, "result_keys"):
            extract_text({"success": True, "result": {"usage": {"total_tokens": 3}}})


if __name__ == "__main__":
    unittest.main()
