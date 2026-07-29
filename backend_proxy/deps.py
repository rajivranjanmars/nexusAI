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


async def require_admin_user(request: Request) -> AuthenticatedUser:
    """Resolve the current user and require admin role.

    Backward-compatible: accepts the original app-token admin role
    (``actor_type == "admin"``).  For the new admin dashboard auth path,
    use ``require_super_admin`` or ``require_app_admin`` instead.
    """
    user = await get_current_user(request)
    if user.role != "admin":
        raise ProxyAuthError("Requires admin privileges", code="FORBIDDEN_ROLE", status_code=403)
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


# ── Hybrid auth (accepts old app-token OR new admin dashboard token) ──────────


async def require_admin_hybrid(request: Request) -> AdminUserContext | AuthenticatedUser:
    """Accept either new admin dashboard token OR old app-token admin token.

    Returns the two sign-in paths' own real types instead of collapsing one
    into the other: AdminUserContext for dashboard (email+password) login,
    AuthenticatedUser for the legacy app-token ``role: "admin"`` claim. Use
    ``admin_actor_label()`` (below) to get a display identity without caring
    which path authenticated the request; use ``isinstance`` if the caller
    needs to branch on which path it was (see ``get_admin_scope``).
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise AdminAuthError("Missing bearer token", code="MISSING_TOKEN")

    token = auth_header[7:].strip()

    # Try admin dashboard token first
    admin_error = None
    try:
        ctx = decode_admin_token(token, expected_type="admin_access")
        return ctx
    except AdminAuthError as e:
        admin_error = e

    # Fall back to old app-token admin
    try:
        user = decode_token(token, expected_type="access")
        if user.role != "admin":
            raise AdminAuthError(
                "Requires admin privileges",
                code="FORBIDDEN_ROLE",
                status_code=403,
            )
        return user
    except (AdminAuthError, ProxyAuthError):
        # Both auth attempts failed, raise the original admin token error
        raise admin_error


def admin_actor_label(principal: AdminUserContext | AuthenticatedUser) -> str:
    """Return a display identity for an admin principal, regardless of which
    sign-in path produced it (dashboard login vs. legacy app-token role).
    """
    if isinstance(principal, AdminUserContext):
        return principal.email or principal.admin_user_id
    return principal.actor_id or "unknown-admin"


async def get_admin_scope(request: Request) -> str | None:
    """Return the scoped app_id for the current admin, or None for super_admin.

    Tries the new admin dashboard JWT first.  Falls back to the original
    app-token admin role.  Returns ``None`` for super_admin / old admin
    (no scoping — see all apps).  Returns the app_id string for app_admin.
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise AdminAuthError("Missing bearer token", code="MISSING_TOKEN")

    token = auth_header[7:].strip()

    # Try admin dashboard token first
    try:
        ctx = decode_admin_token(token, expected_type="admin_access")
        if ctx.admin_role == "super_admin":
            return None
        return ctx.app_id or None
    except AdminAuthError:
        pass

    # Fall back to old app-token admin
    user = decode_token(token, expected_type="access")
    if user.role != "admin":
        raise AdminAuthError(
            "Requires admin privileges",
            code="FORBIDDEN_ROLE",
            status_code=403,
        )
    return None  # old admin sees everything
