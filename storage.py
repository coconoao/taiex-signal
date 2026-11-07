"""Supabase-backed persistence for generated market/news data.

The Render free filesystem is ephemeral. This module mirrors only public market
and news artifacts; authentication secrets and user credentials are never stored.
"""
import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request

FILES = (
    "signals.json", "weekly.json", "weekly_prob.csv", "market_history.csv",
    "chips.csv", "margin.csv", "futures.csv",
    "news.json", "news_history.jsonl", "policy_events.jsonl",
    "news_research.json", "news_research_history.jsonl",
    "news_gdelt_history.jsonl", "news_gdelt_research.json",
    "news_policy_archive.json", "news_policy_research.json",
)
# The official historical archive has its own importer. Exclude it from routine
# Render pipeline saves so an older in-memory copy cannot overwrite a fresh backfill.
AUTO_SAVE_FILES = tuple(name for name in FILES if name != "news_policy_archive.json")
POLICY_ARCHIVE_FILE = "news_policy_archive.json"
TABLE = "site_data"


def enabled():
    return (
        os.environ.get("TAIEX_STORAGE", "").strip().lower() == "supabase"
        and bool(os.environ.get("SUPABASE_URL"))
        and bool(os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
    )


def _request(method, path, payload=None, prefer=None):
    base = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not base or not key:
        raise RuntimeError("Supabase storage credentials are not configured")
    headers = {"apikey": key, "Accept": "application/json"}
    if not key.startswith("sb_secret_"):
        headers["Authorization"] = "Bearer " + key
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    else:
        body = None
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(
        base + "/rest/v1/" + path, data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else []
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            "Supabase data storage request failed (HTTP %s)" % exc.code
        ) from None


def restore_data(directory="."):
    """Restore known data files from Supabase; return number of files restored."""
    if not enabled():
        return 0
    rows = _request("GET", TABLE + "?select=key,content")
    allowed = set(FILES)
    restored = 0
    os.makedirs(directory, exist_ok=True)
    for row in rows:
        name = row.get("key")
        content = row.get("content")
        if name not in allowed or not isinstance(content, str):
            continue
        target = os.path.join(directory, name)
        tmp = target + ".restore.tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
        restored += 1
    return restored


def save_data(directory=".", names=None):
    """Upsert selected data files; routine saves omit the separately managed archive."""
    if not enabled():
        return 0
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = []
    for name in (AUTO_SAVE_FILES if names is None else tuple(names)):
        if name not in FILES:
            raise ValueError("Unknown data file: " + str(name))
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as stream:
                rows.append({"key": name, "content": stream.read(), "updated_at": now})
    if rows:
        _request(
            "POST",
            TABLE + "?on_conflict=key",
            rows,
            "resolution=merge-duplicates,return=minimal",
        )
    return len(rows)


if __name__ == "__main__":
    import sys
    if not enabled():
        raise SystemExit("Set TAIEX_STORAGE=supabase, SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "restore":
        print("Restored data files:", restore_data())
    elif action == "save":
        print("Saved data files:", save_data())
    elif action == "save-policy-archive":
        print(
            "Saved policy archive files:",
            save_data(names=(POLICY_ARCHIVE_FILE,)),
        )
    else:
        raise SystemExit("Usage: python storage.py restore|save|save-policy-archive")
