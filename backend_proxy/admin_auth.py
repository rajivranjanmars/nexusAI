"""
Admin dashboard authentication helpers.

Provides a separate auth path for admin dashboard users (email + password)
that is independent of the app-signed JWT flow used by chatbot clients.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Tuple

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from sqlalchemy import select, update

from db.models.admin_user import AdminUser
from db.postgres import get_session
from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)


# ── Dataclass ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AdminUserContext:
    """Runtime principal for an authenticated admin dashboard user."""

    admin_user_id: str
    email: str
    display_name: str
    admin_role: str          # "super_admin" | "app_admin"
    app_id: str | None       # None for super_admin
    token_id: str


# ── Auth errors ──────────────────────────────────────────────────────────────


class AdminAuthError(Exception):
    """Raised when admin authentication fails."""

    def __init__(self, message: str, code: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


# ── Token helpers ────────────────────────────────────────────────────────────


def _build_admin_payload(
    user: AdminUserContext,
    token_type: str,
    ttl: timedelta,
) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "sub": user.admin_user_id,
        "admin_user_id": user.admin_user_id,
        "email": user.email,
        "display_name": user.display_name,
        "admin_role": user.admin_role,
        "app_id": user.app_id,
        "type": token_type,
        "iat": now,
        "exp": now + ttl,
        "jti": str(uuid.uuid4()),
        "iss": settings.proxy_token_issuer,
    }


def issue_admin_access_token(user: AdminUserContext) -> Tuple[str, int]:
    """Issue a short-lived access token for an admin user."""
    ttl_s = settings.proxy_access_token_ttl_minutes * 60
    payload = _build_admin_payload(user, token_type="admin_access", ttl=timedelta(seconds=ttl_s))
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.proxy_jwt_algorithm)
    return token, ttl_s


def issue_admin_refresh_token(user: AdminUserContext) -> Tuple[str, int]:
    """Issue a long-lived refresh token for an admin user."""
    ttl_s = settings.proxy_refresh_token_ttl_hours * 3600
    payload = _build_admin_payload(user, token_type="admin_refresh", ttl=timedelta(seconds=ttl_s))
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.proxy_jwt_algorithm)
    return token, ttl_s


def decode_admin_token(token: str, expected_type: str) -> AdminUserContext:
    """Decode and validate an admin JWT."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.proxy_jwt_algorithm],
            issuer=settings.proxy_token_issuer,
            options={"require": ["exp", "iat", "iss", "jti", "sub", "type"]},
        )
    except ExpiredSignatureError as exc:
        raise AdminAuthError("Token expired", code="TOKEN_EXPIRED") from exc
    except InvalidTokenError as exc:
        raise AdminAuthError("Invalid token", code="INVALID_TOKEN") from exc

    if payload.get("type") != expected_type:
        raise AdminAuthError(
            f"Invalid token type: expected {expected_type}",
            code="INVALID_TOKEN_TYPE",
            status_code=400,
        )

    return AdminUserContext(
        admin_user_id=str(payload.get("admin_user_id", "")),
        email=str(payload.get("email", "")),
        display_name=str(payload.get("display_name", "")),
        admin_role=str(payload.get("admin_role", "")),
        app_id=str(payload.get("app_id")) if payload.get("app_id") else None,
        token_id=str(payload["jti"]),
    )


# ── Login / refresh ──────────────────────────────────────────────────────────


async def authenticate_admin(email: str, password: str) -> AdminUser:
    """Return the AdminUser ORM record if credentials are valid, else raise."""
    async with get_session() as session:
        result = await session.execute(
            select(AdminUser).where(AdminUser.email == email)
        )
        admin = result.scalar_one_or_none()

    if admin is None:
        raise AdminAuthError("Invalid email or password", code="INVALID_CREDENTIALS")

    if not admin.is_active:
        raise AdminAuthError("Account is deactivated", code="ACCOUNT_INACTIVE", status_code=403)

    if not admin.verify_password(password):
        raise AdminAuthError("Invalid email or password", code="INVALID_CREDENTIALS")

    # Update last_login_at
    async with get_session() as session:
        await session.execute(
            update(AdminUser)
            .where(AdminUser.id == admin.id)
            .values(last_login_at=datetime.now(timezone.utc))
        )
        await session.commit()

    return admin


def _admin_to_context(admin: AdminUser) -> AdminUserContext:
    return AdminUserContext(
        admin_user_id=str(admin.id),
        email=admin.email,
        display_name=admin.display_name,
        admin_role=admin.role,
        app_id=str(admin.app_id) if admin.app_id else None,
        token_id="",
    )


async def admin_login(email: str, password: str) -> dict:
    """Authenticate admin and return tokens + profile."""
    admin = await authenticate_admin(email, password)
    ctx = _admin_to_context(admin)

    access_token, access_ttl = issue_admin_access_token(ctx)
    refresh_token, refresh_ttl = issue_admin_refresh_token(ctx)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": access_ttl,
        "refresh_expires_in": refresh_ttl,
        "token_type": "Bearer",
        "admin_user": {
            "id": str(admin.id),
            "email": admin.email,
            "display_name": admin.display_name,
            "admin_role": admin.role,
            "app_id": str(admin.app_id) if admin.app_id else None,
        },
    }


async def admin_refresh(refresh_token: str) -> dict:
    """Validate refresh token and issue a new access token."""
    ctx = decode_admin_token(refresh_token, expected_type="admin_refresh")

    # Verify admin still exists and is active
    async with get_session() as session:
        result = await session.execute(
            select(AdminUser).where(AdminUser.id == ctx.admin_user_id)
        )
        admin = result.scalar_one_or_none()

    if admin is None or not admin.is_active:
        raise AdminAuthError("Account deactivated or deleted", code="ACCOUNT_INACTIVE", status_code=403)

    fresh_ctx = _admin_to_context(admin)
    access_token, access_ttl = issue_admin_access_token(fresh_ctx)

    return {
        "access_token": access_token,
        "expires_in": access_ttl,
        "token_type": "Bearer",
    }


# ── Audit log helper ────────────────────────────────────────────────────────


async def log_admin_action(
    *,
    admin_user_id: str,
    action: str,
    target_type: str,
    target_id: str | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
) -> None:
    """Record an admin action in the audit log. Fire-and-forget."""
    from db.models.audit_log import AdminAuditLog
    from db.postgres import get_session

    try:
        async with get_session() as session:
            entry = AdminAuditLog(
                admin_user_id=uuid.UUID(admin_user_id) if admin_user_id else None,
                action=action,
                target_type=target_type,
                target_id=target_id,
                details=details,
                ip_address=ip_address,
            )
            session.add(entry)
            await session.commit()
    except Exception:
        logger.exception("Failed to write audit log entry")