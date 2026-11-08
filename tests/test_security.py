import io
import time
import unittest
from unittest.mock import patch

import app as web_app


class StockWorkerRoutingTests(unittest.TestCase):
    def test_readonly_site_fails_closed_without_worker_configuration(self):
        with patch.multiple(
            web_app,
            READONLY=True,
            STOCK_WORKER_URL="",
            STOCK_WORKER_TOKEN="",
        ):
            with self.assertRaises(web_app.StockServiceUnavailable):
                web_app.analyze_stock("2330")

    def test_readonly_site_routes_to_configured_worker(self):
        with patch.multiple(
            web_app,
            READONLY=True,
            STOCK_WORKER_URL="https://worker.example",
            STOCK_WORKER_TOKEN="test-token",
        ), patch.object(web_app, "request_stock_worker", return_value={"code": "2330"}) as worker:
            self.assertEqual(web_app.analyze_stock("2330"), {"code": "2330"})
            worker.assert_called_once_with("2330")

    def test_local_development_keeps_legacy_local_analysis_fallback(self):
        import types
        fake_stock = types.SimpleNamespace(analyze=lambda code: {"code": code, "local": True})
        with patch.multiple(
            web_app,
            READONLY=False,
            STOCK_WORKER_URL="",
            STOCK_WORKER_TOKEN="",
        ), patch.dict("sys.modules", {"stock": fake_stock}):
            self.assertEqual(web_app.analyze_stock("2330"), {"code": "2330", "local": True})


    def _worker_http_error(self, status):
        return web_app.urllib.error.HTTPError(
            "https://worker.example/analyze",
            status,
            "worker error",
            None,
            io.BytesIO(b'{"error":"worker failed"}'),
        )

    def test_worker_upstream_503_is_gateway_failure(self):
        with patch.multiple(
            web_app,
            STOCK_WORKER_URL="https://worker.example",
            STOCK_WORKER_TOKEN="test-token",
        ), patch.object(
            web_app.urllib.request,
            "urlopen",
            side_effect=self._worker_http_error(503),
        ):
            with self.assertRaises(web_app.StockWorkerGatewayError):
                web_app.request_stock_worker("2330")

    def test_worker_validation_422_is_preserved_as_value_error(self):
        with patch.multiple(
            web_app,
            STOCK_WORKER_URL="https://worker.example",
            STOCK_WORKER_TOKEN="test-token",
        ), patch.object(
            web_app.urllib.request,
            "urlopen",
            side_effect=self._worker_http_error(422),
        ):
            with self.assertRaisesRegex(ValueError, "worker failed"):
                web_app.request_stock_worker("2330")

    def test_worker_network_timeout_is_gateway_failure(self):
        with patch.multiple(
            web_app,
            STOCK_WORKER_URL="https://worker.example",
            STOCK_WORKER_TOKEN="test-token",
        ), patch.object(
            web_app.urllib.request,
            "urlopen",
            side_effect=TimeoutError("private timeout detail"),
        ):
            with self.assertRaises(web_app.StockWorkerGatewayError):
                web_app.request_stock_worker("2330")


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
    def make_handler(self):
        handler = object.__new__(web_app.H)
        handler.headers = {"Cookie": f"{web_app.SESSION_COOKIE}=session-token"}
        handler.responses = []
        handler.wfile = io.BytesIO()
        handler.send_response = lambda code: handler.responses.append(("response", code))
        handler.send_error = lambda code, message=None: handler.responses.append(("error", code, message))
        handler.send_header = lambda key, value: None
        handler.end_headers = lambda: None
        return handler

    def test_oauth_callback_rejects_missing_or_mismatched_state(self):
        for query, cookie in [
            ("code=authorization-code", "expected-state"),
            ("state=wrong-state&code=authorization-code", "expected-state"),
            ("state=expected-state", "expected-state"),
        ]:
            with self.subTest(query=query):
                handler = self.make_handler()
                handler.headers = {"Cookie": f"{web_app.STATE_COOKIE}={cookie}"}
                with patch.object(web_app, "GOOGLE_ENABLED", True):
                    handler.google_callback(query)
                self.assertEqual(handler.responses[0][0:2], ("error", 400))

    def test_public_request_fails_closed_without_oauth_configuration(self):
        handler = self.make_handler()
        with patch.multiple(
            web_app,
            READONLY=True,
            production_oauth_ready=lambda: False,
        ):
            self.assertFalse(handler.authorized())
        self.assertEqual(handler.responses[0][0:2], ("error", 503))

    def test_approved_user_is_allowed_and_other_statuses_are_denied(self):
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
                with patch.multiple(
                    web_app,
                    READONLY=True,
                    production_oauth_ready=lambda: True,
                    GOOGLE_ENABLED=True,
                    ADMIN_EMAIL="admin@example.com",
                    session_email=lambda token: "user@example.com",
                    ensure_site_user=lambda email: row,
                ):
                    self.assertEqual(handler.authorized(), expected)

    def test_configured_admin_requires_admin_role_and_approved_status(self):
        cases = [
            ({"email": "admin@example.com", "role": "admin", "status": "approved"}, True),
            ({"email": "admin@example.com", "role": "user", "status": "approved"}, False),
            ({"email": "admin@example.com", "role": "admin", "status": "revoked"}, False),
        ]
        for row, expected in cases:
            with self.subTest(row=row):
                handler = self.make_handler()
                with patch.multiple(
                    web_app,
                    READONLY=True,
                    production_oauth_ready=lambda: True,
                    GOOGLE_ENABLED=True,
                    ADMIN_EMAIL="admin@example.com",
                    session_email=lambda token: "admin@example.com",
                    ensure_site_user=lambda email: row,
                ):
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
