#!/usr/bin/env python3
"""Safely extract text from a Cloudflare Workers AI response envelope."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def _text_from_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, list):
        parts = [
            part.get("text", "")
            for part in value
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        joined = "\n".join(part for part in parts if part.strip())
        return joined or None
    return None


def _safe_error_message(value: Any) -> str:
    message = str(value or "Cloudflare API error")[:500]
    message = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)((?:api[_-]?token|access[_-]?token|secret|password)(?:\s*[:=]\s*)?)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    return message


def _redact_sensitive_text(value: str) -> str:
    """Redact common credential formats before analysis is written to artifacts."""
    patterns = [
        (re.compile(r"(?i)(\\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|private[_-]?key|credential|client[_-]?secret|service[_-]?role)\\b\\s*[:=]\\s*)(['\\"])([^'\\"]{8,})\\2"), r"\\1\\2[REDACTED]\\2"),
        (re.compile(r"(?i)(\\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|private[_-]?key|credential|client[_-]?secret|service[_-]?role)\\b\\s*[:=]\\s*)([A-Za-z0-9_./+=-]{12,})"), r"\\1[REDACTED]"),
        (re.compile(r"(?i)(bearer\\s+)[A-Za-z0-9._~+/-]{16,}"), r"\\1[REDACTED]"),
        (re.compile(r"(?i)\\b(?:sk-[A-Za-z0-9_-]{16,}|sb_secret_[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{20,})\\b"), "[REDACTED]"),
        (re.compile(r"\\beyJ[A-Za-z0-9_-]{10,}\\.[A-Za-z0-9_-]{10,}\\.[A-Za-z0-9_-]{10,}\\b"), "[REDACTED]"),
        (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\\s\\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "[REDACTED PRIVATE KEY]"),
    ]
    redacted = value
    for pattern, replacement in patterns:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def extract_text(data: Any) -> str:
    if not isinstance(data, dict):
        raise ValueError("Cloudflare returned an unexpected JSON envelope.")

    errors = data.get("errors")
    if isinstance(errors, list) and errors:
        details = []
        for item in errors[:3]:
            if isinstance(item, dict):
                code = item.get("code", "unknown")
                details.append(f"Cloudflare error {code}: {_safe_error_message(item.get('message'))}")
        raise ValueError("\n".join(details) or "Cloudflare returned an API error.")

    if data.get("success") is False:
        raise ValueError("Cloudflare reported success=false without a readable errors array.")

    result = data.get("result")
    candidates: list[Any] = []
    if isinstance(result, dict):
        candidates.extend([result.get("response"), result.get("output_text"), result.get("text")])
        choices = result.get("choices")
    else:
        choices = None

    if not isinstance(choices, list):
        choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        candidates.append(message.get("content") if isinstance(message, dict) else None)
        candidates.append(first.get("text") if isinstance(first, dict) else None)

    for candidate in candidates:
        text = _text_from_content(candidate)
        if text:
            return _redact_sensitive_text(text)

    result_keys = sorted(result.keys()) if isinstance(result, dict) else []
    result_type = type(result).__name__
    choice_count = len(data.get("choices", [])) if isinstance(data.get("choices"), list) else 0
    message_count = len(data.get("messages", [])) if isinstance(data.get("messages"), list) else 0
    raise ValueError(
        "Cloudflare returned HTTP success but no readable text. "
        f"success={data.get('success')!r}; result_type={result_type}; "
        f"result_keys={result_keys}; choices={choice_count}; messages={message_count}. "
        "Response content and credentials were not printed."
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: cloudflare_ai_response.py RESPONSE_JSON OUTPUT_MARKDOWN", file=sys.stderr)
        return 2
    source, destination = map(Path, sys.argv[1:])
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        text = extract_text(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    destination.write_text(text, encoding="utf-8")
    print("Cloudflare AI analysis completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
