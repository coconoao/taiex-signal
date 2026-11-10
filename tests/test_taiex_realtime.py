import json
import os
import unittest
from unittest.mock import Mock, patch

import taiex_realtime


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.closed = False

    def send(self, value):
        self.sent.append(json.loads(value))

    def settimeout(self, value):
        pass

    def recv(self):
        if self.messages:
            return json.dumps(self.messages.pop(0))
        raise TimeoutError("no more messages")

    def close(self):
        self.closed = True


class TaiexRealtimeTests(unittest.TestCase):
    def test_missing_api_key_returns_unavailable_without_connecting(self):
        with patch.dict(os.environ, {"FUGLE_API_KEY": ""}):
            with patch("websocket.create_connection") as connect:
                result = taiex_realtime.load_taiex_realtime()
        self.assertEqual(result["status"], "NO_API_KEY")
        self.assertFalse(result["realtime"])
        connect.assert_not_called()

    def test_authenticates_and_subscribes_to_price_index_only(self):
        import datetime as dt
        now_us = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1_000_000)
        ws = FakeWebSocket([
            {"event": "authenticated"},
            {"event": "snapshot", "data": {
                "symbol": "IX0001", "type": "INDEX",
                "index": 28000.5, "time": now_us,
            }},
        ])
        with patch.dict(os.environ, {"FUGLE_API_KEY": "private-test-key"}):
            with patch("websocket.create_connection", return_value=ws):
                result = taiex_realtime.load_taiex_realtime()
        self.assertEqual(result["symbol"], "IX0001")
        self.assertEqual(result["index"], 28000.5)
        self.assertEqual(result["status"], "REALTIME")
        self.assertEqual(ws.sent[0]["event"], "auth")
        self.assertEqual(ws.sent[1]["event"], "subscribe")
        self.assertEqual(ws.sent[1]["data"], {
            "channel": "indices", "symbol": "IX0001",
        })
        self.assertNotIn("private-test-key", str(result))
        self.assertTrue(ws.closed)

    def test_rejects_other_symbol_even_if_valid_index_message(self):
        ws = FakeWebSocket([
            {"event": "authenticated"},
            {"event": "snapshot", "data": {
                "symbol": "IR0001", "type": "INDEX",
                "index": 28000.5, "time": 1790000000000000,
            }},
        ])
        with patch.dict(os.environ, {"FUGLE_API_KEY": "private-test-key"}):
            with patch("websocket.create_connection", return_value=ws):
                with patch.object(taiex_realtime.time, "monotonic", side_effect=[1, 1, 1, 20]):
                    result = taiex_realtime.load_taiex_realtime(timeout=2)
        self.assertEqual(result["status"], "NO_SNAPSHOT")
        self.assertFalse(result["available"])

    def test_invalid_authentication_is_reported_without_secret(self):
        ws = FakeWebSocket([
            {"event": "error", "code": 1000, "data": {"message": "Invalid authentication credentials"}},
        ])
        with patch.dict(os.environ, {"FUGLE_API_KEY": "private-test-key"}):
            with patch("websocket.create_connection", return_value=ws):
                result = taiex_realtime.load_taiex_realtime()
        self.assertEqual(result["status"], "AUTH_FAILED")
        self.assertNotIn("private-test-key", str(result))


if __name__ == "__main__":
    unittest.main()
