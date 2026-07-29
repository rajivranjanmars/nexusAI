"""
Shared middleware for correlation-ID tracking across services.

Provides request tracing by assigning/forwarding correlation-IDs and storing
them in shared request context for downstream code to access.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from shared.request_context import correlation_id_var, project_name_var, reg_no_var


class CorrelationHandler(BaseHTTPMiddleware):
    """Populate correlation-ID context for request tracing.

    Every request gets a unique ``X-Correlation-ID`` header (either forwarded
    from the client or generated server-side). The ID is stored in shared request
    context so downstream code can access it for structured logging and tracing.

    Args:
        app: The wrapped ASGI application.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        """Attach correlation-ID to incoming request and response.

        Args:
            request: Incoming Starlette request.
            call_next: Middleware continuation callback.

        Returns:
            Downstream response with correlation-ID header attached.
        """
        correlation_id = request.headers.get("x-correlation-id") or str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
        project_name_var.set(None)
        reg_no_var.set(None)
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
