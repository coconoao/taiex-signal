"""Immutable post-close forecast snapshots and prospective quality diagnostics.

This module does not train, tune, or alter any forecasting model. It records
one completed-session snapshot per model version and evaluates only those
snapshots whose target closes are subsequently present in observed data.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from collections import defaultdict
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")
POST_CLOSE_CUTOFF = dt.time(16, 0)
EXPECTED_HORIZONS = (1, 2, 3)


def _positive_finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def build_live_snapshot_record(snapshot, captured_at=None):
    """Convert a production schema-v2 snapshot to a compact immutable record."""
    if not isinstance(snapshot, dict) or snapshot.get("demo") is not False:
        return None
    if snapshot.get("schema_version") != 2 or snapshot.get("record_type") != "index_forecast_snapshot":
        return None

    forecast = snapshot.get("forecast")
    if not isinstance(forecast, dict) or forecast.get("schema_version") != 2:
        return None
    if forecast.get("status") != "research":
        return None

    asof = str(forecast.get("asof") or snapshot.get("asof") or "")
    try:
        asof_day = dt.date.fromisoformat(asof)
    except ValueError:
        return None

    base_close = _positive_finite(forecast.get("base_close"))
    model_version = str(forecast.get("model_version") or "").strip()
    if base_close is None or not model_version:
        return None

    source_sessions = forecast.get("sessions")
    if not isinstance(source_sessions, list):
        return None

    sessions = []
    seen = set()
    for item in source_sessions:
        if not isinstance(item, dict):
            continue
        try:
            horizon = int(item.get("horizon_sessions"))
        except (TypeError, ValueError):
            continue
        if horizon not in EXPECTED_HORIZONS or horizon in seen or item.get("status") != "research":
            continue
        target_date = str(item.get("target_date") or "")
        try:
            target_day = dt.date.fromisoformat(target_date)
        except ValueError:
            continue
        if target_day <= asof_day:
            continue
        predicted_close = _positive_finite(item.get("forecast_close"))
        if predicted_close is None:
            continue
        predicted_return_pct = item.get("predicted_return_pct")
        try:
            predicted_return_pct = float(predicted_return_pct)
        except (TypeError, ValueError):
            predicted_return_pct = (predicted_close / base_close - 1.0) * 100
        if not math.isfinite(predicted_return_pct):
            continue
        sessions.append({
            "horizon_sessions": horizon,
            "target_date": target_date,
            "predicted_close": round(predicted_close, 2),
            "predicted_return_pct": round(predicted_return_pct, 6),
        })
        seen.add(horizon)

    if seen != set(EXPECTED_HORIZONS):
        return None

    timestamp = captured_at or dt.datetime.now(dt.timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    timestamp = timestamp.astimezone(dt.timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "record_type": "index_forecast_live_snapshot",
        "instrument": "TAIEX",
        "forecast_asof": asof,
        "forecast_computed_at": str(snapshot.get("computed_at") or ""),
        "captured_at": timestamp,
        "base_close": round(base_close, 2),
        "model": str(forecast.get("method") or "unknown"),
        "model_version": model_version,
        "evaluation": "prospective_live_snapshot",
        "sessions": sorted(sessions, key=lambda row: row["horizon_sessions"]),
    }


def append_daily_live_snapshot(path, snapshot, now=None, cutoff=POST_CLOSE_CUTOFF):
    """Append one complete production snapshot after close; idempotent by date/version."""
    local_now = now or dt.datetime.now(TAIPEI)
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=TAIPEI)
    local_now = local_now.astimezone(TAIPEI)
    if local_now.time().replace(tzinfo=None) < cutoff:
        return {"appended": False, "reason": "before_post_close_cutoff"}

    record = build_live_snapshot_record(snapshot, captured_at=local_now)
    if record is None:
        return {"appended": False, "reason": "snapshot_incomplete_or_not_production"}
    if record["forecast_asof"] != local_now.date().isoformat():
        return {"appended": False, "reason": "snapshot_asof_is_not_current_taiwan_session"}
    try:
        computed_at = dt.datetime.fromisoformat(
            record["forecast_computed_at"].replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return {"appended": False, "reason": "snapshot_computed_at_missing_or_invalid"}
    if computed_at.tzinfo is None:
        computed_at = computed_at.replace(tzinfo=dt.timezone.utc)
    computed_local = computed_at.astimezone(TAIPEI)
    if computed_local.date() != local_now.date() or computed_local.time().replace(tzinfo=None) < cutoff:
        return {"appended": False, "reason": "snapshot_was_not_computed_after_post_close_cutoff"}

    existing_keys = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                previous = json.loads(line)
                if previous.get("record_type") == "index_forecast_live_snapshot":
                    existing_keys.add((previous.get("forecast_asof"), previous.get("model_version")))
    key = (record["forecast_asof"], record["model_version"])
    if key in existing_keys:
        return {"appended": False, "reason": "already_recorded"}

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"appended": True, "reason": "recorded", "record": record}


def read_live_snapshot_history(path):
    """Read valid JSONL records; malformed non-empty lines fail loudly."""
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if record.get("record_type") == "index_forecast_live_snapshot":
                    records.append(record)
    return records


def summarize_live_forecast_quality(records, actual_closes, window_sessions=20, minimum_samples=20, as_of=None):
    """Evaluate matured live snapshots against observed closes and a flat baseline.

    actual_closes may be a date->close mapping or a pandas Series indexed by date.
    No threshold is used to label a model as good/bad; small samples remain
    explicitly provisional.
    """
    current = as_of or dt.datetime.now(TAIPEI)
    if current.tzinfo is None:
        current = current.replace(tzinfo=TAIPEI)
    current = current.astimezone(TAIPEI)

    close_map = {}
    source = actual_closes.items() if hasattr(actual_closes, "items") else []
    for day, value in source:
        try:
            key = str(day)[:10]
        except Exception:
            continue
        number = _positive_finite(value)
        if number is not None:
            close_map[key] = number

    candidates = defaultdict(dict)
    for record in records or []:
        if not isinstance(record, dict) or record.get("evaluation") != "prospective_live_snapshot":
            continue
        base = _positive_finite(record.get("base_close"))
        model_version = str(record.get("model_version") or "unknown")
        try:
            captured = str(record.get("captured_at") or "")
            captured_dt = dt.datetime.fromisoformat(captured.replace("Z", "+00:00"))
        except ValueError:
            captured_dt = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        for item in record.get("sessions", []):
            try:
                horizon = int(item.get("horizon_sessions"))
                target = dt.date.fromisoformat(str(item.get("target_date")))
            except (TypeError, ValueError):
                continue
            session_complete = (
                target < current.date()
                or (target == current.date() and current.time().replace(tzinfo=None) >= POST_CLOSE_CUTOFF)
            )
            actual = close_map.get(target.isoformat()) if session_complete else None
            predicted = _positive_finite(item.get("predicted_close"))
            if horizon not in EXPECTED_HORIZONS or base is None or actual is None or predicted is None:
                continue
            key = (model_version, horizon, target.isoformat())
            previous = candidates[(model_version, horizon)].get(target.isoformat())
            if previous is None or captured_dt > previous["captured_dt"]:
                candidates[(model_version, horizon)][target.isoformat()] = {
                    "captured_dt": captured_dt,
                    "base": base,
                    "predicted": predicted,
                    "actual": actual,
                }

    summaries = []
    for (model_version, horizon), by_target in sorted(candidates.items()):
        ordered = sorted(by_target.items())
        selected = ordered[-max(1, int(window_sessions)):]
        rows = [row for _, row in selected]
        model_errors = [abs(row["predicted"] - row["actual"]) for row in rows]
        flat_errors = [abs(row["base"] - row["actual"]) for row in rows]
        actual_returns = [(row["actual"] / row["base"] - 1.0) * 100 for row in rows]
        predicted_returns = [(row["predicted"] / row["base"] - 1.0) * 100 for row in rows]
        direction_hits = [
            (pred > 0 and actual > 0) or (pred < 0 and actual < 0) or (pred == 0 and actual == 0)
            for pred, actual in zip(predicted_returns, actual_returns)
        ]
        n = len(rows)
        summaries.append({
            "horizon_sessions": horizon,
            "model_version": model_version,
            "sample_count": n,
            "window_start": selected[0][0],
            "window_end": selected[-1][0],
            "status": "provisional" if n < int(minimum_samples) else "monitoring",
            "minimum_samples": int(minimum_samples),
            "model_mae_points": round(sum(model_errors) / n, 2),
            "flat_baseline_mae_points": round(sum(flat_errors) / n, 2),
            "mae_delta_vs_flat_points": round((sum(model_errors) - sum(flat_errors)) / n, 2),
            "mean_absolute_error_pct": round(
                sum(error / row["actual"] * 100 for error, row in zip(model_errors, rows)) / n, 3
            ),
            "mean_return_bias_pp": round(
                sum(pred - actual for pred, actual in zip(predicted_returns, actual_returns)) / n, 3
            ),
            "direction_accuracy_pct": round(sum(direction_hits) / n * 100, 1),
        })
    return {
        "schema_version": 1,
        "record_type": "live_forecast_quality_monitor",
        "evaluation": "prospective_live_snapshot",
        "window_sessions": int(window_sessions),
        "horizons": summaries,
        "note": "僅評估已封存的正式盤後預測；樣本不足時標示 provisional，不用此監控直接調整模型或產生交易訊號。",
    }
