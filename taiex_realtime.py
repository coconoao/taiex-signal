"""On-demand Fugle WebSocket snapshot for the TAIEX price index."""
import datetime as dt
import json
import os
import time
from zoneinfo import ZoneInfo

INDEX_SYMBOL = "IX0001"
WS_URL = "wss://api.fugle.tw/marketdata/v1.0/stock/streaming"
_TAIPEI = ZoneInfo("Asia/Taipei")


class IndexQuoteError(RuntimeError):
    pass


def load_taiex_realtime(timeout=8):
    """Open one short-lived connection and return a verified IX0001 snapshot.

    This avoids keeping a WebSocket alive on a free Render instance that may sleep.
    The caller must treat missing/old timestamps as unavailable, never as live.
    """
    api_key = os.environ.get("FUGLE_API_KEY", "").strip()
    if not api_key:
        return {
            "available": False, "realtime": False, "status": "NO_API_KEY",
            "symbol": INDEX_SYMBOL, "name": "發行量加權股價指數",
            "source": "Fugle",
        }

    try:
        import websocket
    except ImportError as exc:
        raise IndexQuoteError("WebSocket client dependency is unavailable") from exc

    ws = None
    deadline = time.monotonic() + max(2, min(int(timeout), 12))
    try:
        ws = websocket.create_connection(
            WS_URL, timeout=min(5, timeout), enable_multithread=False,
            origin="https://developer.fugle.tw",
        )
        ws.send(json.dumps({"event": "auth", "data": {"apikey": api_key}}))
        authenticated = False
        subscribed = False
        latest = None
        while time.monotonic() < deadline:
            ws.settimeout(max(0.5, min(2.0, deadline - time.monotonic())))
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                continue
            event = message.get("event")
            if event == "error":
                code = message.get("code")
                if code == 1000:
                    return {
                        "available": False, "realtime": False,
                        "status": "AUTH_FAILED", "symbol": INDEX_SYMBOL,
                        "name": "發行量加權股價指數", "source": "Fugle",
                    }
                raise IndexQuoteError("Fugle WebSocket returned an error")
            if event == "authenticated":
                authenticated = True
                ws.send(json.dumps({
                    "event": "subscribe",
                    "data": {"channel": "indices", "symbol": INDEX_SYMBOL},
                }))
                subscribed = True
                continue
            if event not in ("snapshot", "data") or not authenticated or not subscribed:
                continue
            data = message.get("data") or {}
            if data.get("symbol") != INDEX_SYMBOL or data.get("type") != "INDEX":
                continue
            try:
                value = float(data["index"])
                timestamp_us = int(data["time"])
            except (KeyError, TypeError, ValueError):
                continue
            if value <= 0 or timestamp_us <= 0:
                continue
            quote_time = dt.datetime.fromtimestamp(timestamp_us / 1_000_000, tz=dt.timezone.utc).astimezone(_TAIPEI)
            now = dt.datetime.now(_TAIPEI)
            age = max(0.0, (now - quote_time).total_seconds())
            latest = {
                "available": True,
                "realtime": age <= 90 and quote_time.date() == now.date(),
                "status": "REALTIME" if age <= 90 and quote_time.date() == now.date() else "STALE",
                "symbol": INDEX_SYMBOL,
                "name": "發行量加權股價指數",
                "index": value,
                "timestamp": quote_time.isoformat(timespec="seconds"),
                "age_seconds": round(age, 1),
                "source": "Fugle WebSocket",
            }
            if latest["realtime"]:
                return latest
        if latest:
            return latest
        return {
            "available": False, "realtime": False, "status": "NO_SNAPSHOT",
            "symbol": INDEX_SYMBOL, "name": "發行量加權股價指數",
            "source": "Fugle WebSocket",
        }
    except IndexQuoteError:
        raise
    except Exception as exc:
        raise IndexQuoteError("Fugle index connection failed: " + type(exc).__name__) from exc
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
