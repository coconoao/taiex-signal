import time
import unittest
from unittest.mock import patch

import app as web_app


class ProductionAuthConfigurationTests(unittest.TestCase):
    def test_public_deployment_requires_oauth_and_supabase(self):
        with patch.object(web_app, "READONLY", True), \\
             patch.object(web_app, "GOOGLE_ENABLED", False), \\
             patch.object(web_app, "SUPABASE_URL", "https://example.supabase.co"), \\
             patch.object(web_app, "SUPABASE_SERVICE_ROLE_KEY", "secret"), \\
             patch.object(web_app, "ADMIN_EMAIL", "admin@example.com"):
            self.assertFalse(web_app.production_oauth_ready())

    def test_public_deployment_accepts_complete_oauth_configuration(self):
        with patch.object(web_app, "READONLY", True), \\
             patch.object(web_app, "GOOGLE_ENABLED", True), \\
             patch.object(web_app, "SUPABASE_URL", "https://example.supabase.co"), \\
             patch.object(web_app, "SUPABASE_SERVICE_ROLE_KEY", "secret"), \\
             patch.object(web_app, "ADMIN_EMAIL", "admin@example.com"):
            self.assertTrue(web_app.production_oauth_ready())


class SessionSecurityTests(unittest.TestCase):
    def test_signed_session_round_trip_normalizes_email(self):
        token = web_app.sign_session("User@Example.com", int(time.time()) + 60)
        self.assertEqual(web_app.session_email(token), "user@example.com")

    def test_expired_session_is_rejected(self):
        token = web_app.sign_session("user@example.com", int(time.time()) - 1)
        self.assertEqual(web_app.session_email(token), "")

    def test_tampered_session_is_rejected(self):
        token = web_app.sign_session("user@example.com", int(time.time()) + 60)
        altered = token[:-1] + ("0" if token[-1] != "0" else "1")
        self.assertEqual(web_app.session_email(altered), "")


class UserAdministrationTests(unittest.TestCase):
    def setUp(self):
        self.admin_email = web_app.ADMIN_EMAIL
        web_app.ADMIN_EMAIL = "admin@example.com"

    def tearDown(self):
        web_app.ADMIN_EMAIL = self.admin_email

    def test_cannot_change_configured_admin(self):
        with self.assertRaises(ValueError):
            web_app.set_site_user_status("ADMIN@example.com", "revoked", "admin@example.com")

    def test_cannot_assign_admin_role_through_status_api(self):
        with self.assertRaises(ValueError):
            web_app.set_site_user_status("user@example.com", "admin", "admin@example.com")

    @patch.object(web_app, "supabase_request", return_value=[{
        "email": "user@example.com", "role": "user", "status": "approved"
    }])
    def test_status_update_is_limited_to_user_role(self, request):
        web_app.set_site_user_status(" User@Example.com ", "approved", "admin@example.com")
        args, kwargs = request.call_args
        self.assertEqual(args[0], "PATCH")
        self.assertIn("email=eq.user%40example.com", args[1])
        self.assertIn("role=eq.user", args[1])
        self.assertEqual(args[2]["status"], "approved")
        self.assertEqual(args[2]["approved_by"], "admin@example.com")


if __name__ == "__main__":
    unittest.main()
