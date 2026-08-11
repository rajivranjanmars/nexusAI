"""
Shared FastAPI dependencies for the backend proxy.

Extracted from main.py so that rag_routes.py (and other routers) can import
them without creating a circular dependency with main.py.
"""
from fastapi import Request
from backend_proxy.admin_auth import AdminAuthError, AdminUserContext, decode_admin_token
from backend_proxy.auth import AuthenticatedUser, ProxyAuthError, decode_token
from shared.request_context import project_name_var, reg_no_var


# ── App-token (chatbot) dependencies ─────────────────────────────────────────


async def get_current_user(request: Request) -> AuthenticatedUser:
    """Resolve the current authenticated user from the bearer token.

    Raises:
        ProxyAuthError: If the token is missing or invalid.
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise ProxyAuthError("Missing bearer token", code="MISSING_TOKEN")

    token = auth_header[7:].strip()
    user = decode_token(token, expected_type="access")
    project_name_var.set(user.project_name)
    reg_no_var.set(user.actor_id)
    return user


# ── Admin dashboard (email+password) dependencies ────────────────────────────


def _extract_bearer(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise AdminAuthError("Missing bearer token", code="MISSING_TOKEN")
    return auth_header[7:].strip()


async def get_admin_user(request: Request) -> AdminUserContext:
    """Resolve the current admin dashboard user from the bearer token."""
    token = _extract_bearer(request)
    return decode_admin_token(token, expected_type="admin_access")


async def require_super_admin(request: Request) -> AdminUserContext:
    """Require super_admin role.  Returns 403 for app_admin."""
    ctx = await get_admin_user(request)
    if ctx.admin_role != "super_admin":
        raise AdminAuthError(
            "Requires super_admin privileges",
            code="FORBIDDEN_ROLE",
            status_code=403,
        )
    return ctx


async def require_app_admin(request: Request) -> AdminUserContext:
    """Require at least app_admin role.  Sets the scoped app_id on the context."""
    ctx = await get_admin_user(request)
    if ctx.admin_role not in ("super_admin", "app_admin"):
        raise AdminAuthError(
            "Requires admin privileges",
            code="FORBIDDEN_ROLE",
            status_code=403,
        )
    return ctx


async def get_admin_scope(request: Request) -> str | None:
    """Return the scoped app_id for the current admin, or None for super_admin."""
    ctx = await get_admin_user(request)
    if ctx.admin_role == "super_admin":
        return None
    return ctx.app_id or None
