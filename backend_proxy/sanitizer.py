"""
Response sanitization helpers for the backend proxy.

The proxy is responsible for hiding sensitive internal fields before data is
returned to browser clients.
"""

from __future__ import annotations

from typing import Any

_SENSITIVE_FIELDS = frozenset(
    {
        "debug",
        "internal_notes",
        "notes",
        "raw_context",
        "raw_prompt",
    }
)


def sanitize_payload(payload: Any) -> Any:
    """Recursively remove sensitive keys from arbitrary payloads.

    Args:
        payload: Any JSON-serialisable value.

    Returns:
        A sanitised copy of the payload.
    """
    if isinstance(payload, dict):
        cleaned = {}
        for key, value in payload.items():
            if key.lower() in _SENSITIVE_FIELDS:
                continue
            cleaned[key] = sanitize_payload(value)
        return cleaned

    if isinstance(payload, list):
        return [sanitize_payload(item) for item in payload]

    if isinstance(payload, tuple):
        return [sanitize_payload(item) for item in payload]

    return payload
