"""
Shared async request context for correlation and identity metadata.

Both the MCP server and the backend proxy populate these context variables so
structured logs can include request-scoped identifiers without threading
values through every function call.
"""

from __future__ import annotations

import contextvars
from typing import Optional

# ── Request-scoped context variables ─────────────────────────────────────────

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id",
    default="",
)
project_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "project_name",
    default=None,
)
reg_no_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "reg_no",
    default=None,
)
