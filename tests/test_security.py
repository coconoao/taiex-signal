import time
import unittest
from unittest.mock import patch

import app as web_app


class ProductionAuthConfigurationTests(unittest.TestCase):
    def test_public_deployment_requires_oauth_and_supabase(self):
        with patch.multiple(
            web_app,
            GOOGLE_ENABLED=False,
            SUPABASE_URL="https://example.supabase.co",
            SUPABASE_SERVICE_ROLE_KEY="secret",
            ADMIN_EMAIL="admin@example.com",
        ):
            self.assertFalse(web_app.production_oauth_ready())

    def test_public_deployment_accepts_complete_oauth_configuration(self):
        with patch.multiple(
            web_app,
            GOOGLE_ENABLED=True,
            SUPABASE_URL="https://example.supabase.co",
            SUPABASE_SERVICE_ROLE_KEY="secret",
            ADMIN_EMAIL="admin@example.com",
        ):
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



class RequestAuthTests(unittest.TestCase):
    def make_handler(self, cookie="session-token"):
        handler = object.__new__(web_app.H)
        handler.headers = {"Cookie": f"{web_app.SESSION_COOKIE}={cookie}"}
        handler.responses = []
        handler.headers_sent = []
        handler.wfile = __import__("io").BytesIO()
        handler.send_response = lambda code: handler.responses.append(("response", code))
        handler.send_error = lambda code, message=None: handler.responses.append(("error", code, message))
        handler.send_header = lambda key, value: handler.headers_sent.append((key, value))
        handler.end_headers = lambda: None
        return handler

    def test_public_request_fails_closed_without_oauth_configuration(self):
        handler = self.make_handler()
        with patch.object(web_app, "READONLY", True), \\
             patch.object(web_app, "production_oauth_ready", return_value=False):
            self.assertFalse(handler.authorized())
        self.assertEqual(handler.responses[0][0:2], ("error", 503))

    def test_approved_user_is_allowed_and_pending_user_is_denied(self):
        cases = [
            ({"email": "user@example.com", "role": "user", "status": "approved"}, True),
            ({"email": "user@example.com", "role": "user", "status": "pending"}, False),
            ({"email": "user@example.com", "role": "user", "status": "rejected"}, False),
            ({"email": "user@example.com", "role": "user", "status": "revoked"}, False),
            ({"email": "user@example.com", "role": "admin", "status": "approved"}, False),
        ]
        for row, expected in cases:
            with self.subTest(row=row):
                handler = self.make_handler()
                with patch.object(web_app, "READONLY", True), \\
                     patch.object(web_app, "production_oauth_ready", return_value=True), \\
                     patch.object(web_app, "GOOGLE_ENABLED", True), \\
                     patch.object(web_app, "ADMIN_EMAIL", "admin@example.com"), \\
                     patch.object(web_app, "session_email", return_value="user@example.com"), \\
                     patch.object(web_app, "ensure_site_user", return_value=row):
                    self.assertEqual(handler.authorized(), expected)

    def test_configured_admin_requires_admin_role_and_approved_status(self):
        for row, expected in [
            ({"email": "admin@example.com", "role": "admin", "status": "approved"}, True),
            ({"email": "admin@example.com", "role": "user", "status": "approved"}, False),
            ({"email": "admin@example.com", "role": "admin", "status": "revoked"}, False),
        ]:
            with self.subTest(row=row):
                handler = self.make_handler()
                with patch.object(web_app, "READONLY", True), \\
                     patch.object(web_app, "production_oauth_ready", return_value=True), \\
                     patch.object(web_app, "GOOGLE_ENABLED", True), \\
                     patch.object(web_app, "ADMIN_EMAIL", "admin@example.com"), \\
                     patch.object(web_app, "session_email", return_value="admin@example.com"), \\
                     patch.object(web_app, "ensure_site_user", return_value=row):
                    self.assertEqual(handler.authorized(), expected)

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
