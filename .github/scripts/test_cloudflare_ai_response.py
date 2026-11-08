import unittest

from cloudflare_ai_response import extract_text, _redact_sensitive_text


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

    def test_output_redacts_quoted_and_unquoted_credentials(self):
        value = 'API_KEY="quoted-secret-value"\naccess_token=unquoted_secret_value_123'
        redacted = _redact_sensitive_text(value)
        self.assertNotIn("quoted-secret-value", redacted)
        self.assertNotIn("unquoted_secret_value_123", redacted)
        self.assertEqual(redacted.count("[REDACTED]"), 2)
        prefixed = 'CLOUDFLARE_API_TOKEN="cloudflare-token-value"\\nSUPABASE_SERVICE_ROLE_KEY=supabase_secret_value_123'
        prefixed_redacted = _redact_sensitive_text(prefixed)
        self.assertNotIn("cloudflare-token-value", prefixed_redacted)
        self.assertNotIn("supabase_secret_value_123", prefixed_redacted)
        self.assertEqual(prefixed_redacted.count("[REDACTED]"), 2)

    def test_output_redacts_bearer_and_jwt_tokens(self):
        bearer = "Bearer " + "A" * 32
        jwt = ".".join(["eyJ" + "a" * 20, "eyJ" + "b" * 20, "eyJ" + "c" * 20])
        redacted = _redact_sensitive_text(bearer + " " + jwt)
        self.assertNotIn("A" * 32, redacted)
        self.assertNotIn(jwt, redacted)

    def test_missing_text_reports_structure_without_dumping_body(self):
        with self.assertRaisesRegex(ValueError, "result_keys"):
            extract_text({"success": True, "result": {"usage": {"total_tokens": 3}}})


if __name__ == "__main__":
    unittest.main()
