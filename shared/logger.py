"""
Structured logging utility for the Central AI Module.

Provides a pre-configured, JSON-formatted logger that reads the desired log
level from the LOG_LEVEL environment variable (default: info).  Every module
should import ``get_logger(__name__)`` to obtain a contextual logger instance.

Features:
- Structured JSON output (via python-json-logger).
- Automatic injection of ``correlation_id``, ``project_name``, and ``reg_no``
  from :mod:`shared.request_context` context variables.
- PII masking: email addresses, phone numbers, and Aadhaar-like IDs are
  redacted in log message text and ``extra`` fields before emission.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Optional

# Third-party structured JSON handler (falls back to StreamHandler if
# python-json-logger is not installed).
try:
    from pythonjsonlogger import jsonlogger  # type: ignore[import-untyped]

    _HAS_JSON_LOGGER = True
except ImportError:
    _HAS_JSON_LOGGER = False


_LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
_CONFIGURED: bool = False

# ── PII masking patterns ───────────────────────────────────────────────────

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email addresses → ***@***.***
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "***@***.***"),
    # Phone numbers (10+ digits, optional intl prefix)
    (re.compile(r"(\+?\d[\d\-\s]{8,}\d)"), "***-***-****"),
    # Aadhaar-style 12-digit IDs (space / dash separated groups of 4)
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "****-****-****"),
]


def mask_pii(text: str) -> str:
    """Replace known PII patterns in *text* with redacted placeholders."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _mask_value(value: Any) -> Any:
    """Recursively mask PII in strings, dicts, and lists."""
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {k: _mask_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mask_value(v) for v in value]
    return value


# ── Context-injecting filter ───────────────────────────────────────────────

class _ContextFilter(logging.Filter):
    """Injects correlation_id, project_name, reg_no from contextvars and
    applies PII masking to the log message and extra fields."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Lazy import to avoid circular dependency at module load time
        try:
            from shared.request_context import (
                correlation_id_var,
                project_name_var,
                reg_no_var,
            )
            record.correlation_id = correlation_id_var.get("")  # type: ignore[arg-type]
            record.project_name = project_name_var.get(None)
            record.reg_no = reg_no_var.get(None)
        except ImportError:
            record.correlation_id = ""   # type: ignore[assignment]
            record.project_name = None   # type: ignore[assignment]
            record.reg_no = None         # type: ignore[assignment]

        # ── PII masking on the message itself ──────────────────────────
        if isinstance(record.msg, str):
            record.msg = mask_pii(record.msg)

        # ── PII masking on extra fields attached via `extra={}` ────────
        # Standard LogRecord fields that should NOT be masked
        _STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)
        for key in set(record.__dict__) - _STANDARD:
            val = getattr(record, key, None)
            if val is not None:
                setattr(record, key, _mask_value(val))

        return True  # always emit


# ── Logger configuration ───────────────────────────────────────────────────

def _configure_root_logger() -> None:
    """Configure the root logger once with structured JSON output."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)

    if _HAS_JSON_LOGGER:
        formatter = jsonlogger.JsonFormatter(
            fmt=(
                "%(asctime)s %(name)s %(levelname)s %(message)s "
                "%(correlation_id)s %(project_name)s %(reg_no)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)-8s | %(name)s | "
                "cid=%(correlation_id)s | proj=%(project_name)s | "
                "reg=%(reg_no)s | %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handler.setFormatter(formatter)
    handler.addFilter(_ContextFilter())
    root.addHandler(handler)
    try:
        from shared.config import settings as _s
        if _s.slack_webhook_url:
            from shared.alerting import init_slack_alerting
            root.addHandler(init_slack_alerting(_s.slack_webhook_url, _s.alert_rate_limit_seconds))
    except Exception:
        pass  # alerting init failure must never break logging

    # Silence client-library request logging. httpx/httpcore log the full
    # request URL at INFO, and the SMS gateway call carries the gateway
    # credentials and the plaintext OTP in the query string — logging them
    # would ship secrets/OTP codes to stdout and Loki. The PII/masking filter
    # cannot reach them (they live in the LogRecord ``args`` URL), so silence
    # these loggers outright. urllib3/requests are muted for the same reason.
    for noisy in ("httpx", "httpcore", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger, ensuring root config is applied once.

    Args:
        name: Logger namespace, typically ``__name__``.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    _configure_root_logger()
    return logging.getLogger(name or "central_ai")
