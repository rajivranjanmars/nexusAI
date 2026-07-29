"""
JWT authentication helpers for the backend proxy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from db.app_registry import AppContext, resolve_by_app_id
from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AuthenticatedUser:
    """Authenticated proxy user derived from a validated token."""

    actor_id: str
    project_name: str
    role: str
    token_id: str
    app_id: str = ""
    actor_type: str = "student"
    allowed_workflows: tuple = ()
    allowed_tools: tuple = ()
    force_workflow: str = ""
    phone: str = ""
    phone_verified: bool = False


def mark_phone_verified(user: "AuthenticatedUser", phone: str) -> "AuthenticatedUser":
    """Return a copy of ``user`` marked as having verified ``phone`` via OTP."""
    return replace(user, phone=phone, phone_verified=True)


def phone_session_id(user: "AuthenticatedUser") -> str:
    """Return a stable conversation key for a phone-verified guest.

    Anonymous guests have an empty ``actor_id`` and would otherwise receive a
    fresh, throwaway session per request. Keying their conversation off the
    verified phone lets a returning user resume the same conversation memory.
    Returns an empty string when the user is not a phone-verified guest.
    """
    if user.phone and user.phone_verified:
        return f"phone:{user.phone}"
    return ""


class ProxyAuthError(Exception):
    """Raised when proxy authentication or authorisation fails."""

    def __init__(self, message: str, code: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _normalize_origin(value: str) -> str:
    """Normalize Origin/Referer/domain strings to scheme://host[:port]."""
    raw = (value or "").strip()
    if not raw:
        return ""

    if "://" not in raw:
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw.rstrip("/")

    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def _allowed_origins(domain_field: str) -> list[str]:
    """Parse one or more allowed origins from the stored app domain field.

    Supports comma-separated, semicolon-separated, or newline-separated values.
    """
    raw = (domain_field or "").strip()
    if not raw:
        return []

    parts = [
        item.strip()
        for chunk in raw.splitlines()
        for item in chunk.replace(";", ",").split(",")
    ]
    return [_normalize_origin(item) for item in parts if item.strip()]


async def authenticate_app_jwt(signed_token: str, request_domain: str) -> tuple[AuthenticatedUser, AppContext]:
    """Authenticate an application via its signed JWT.

    Decodes the JWT to find the app ID, loads the public key from the App Registry,
    and mathematically verifies the signature and domain.
    """
    try:
        unverified = jwt.decode(
            signed_token,
            algorithms=["ES256", "RS256"],
            options={"verify_signature": False},
        )
        app_id_str = unverified.get("iss")
        if not app_id_str:
            raise ProxyAuthError("Missing issuer (app_id) in signed token", code="INVALID_JWT")
    except Exception as exc:
        raise ProxyAuthError("Malformed signed token", code="INVALID_JWT") from exc

    app_ctx = resolve_by_app_id(app_id_str)
    if app_ctx is None:
        logger.warning("App authentication failed: invalid app_id")
        raise ProxyAuthError("Invalid App ID", code="INVALID_APP_ID")

    if not app_ctx.is_active:
        raise ProxyAuthError("Application is deactivated", code="APP_INACTIVE", status_code=403)

    normalized_request_origin = _normalize_origin(request_domain)
    allowed_origins = _allowed_origins(app_ctx.domain)
    if normalized_request_origin and allowed_origins and normalized_request_origin not in allowed_origins:
        logger.warning(
            "Domain mismatch: %s not in %s",
            normalized_request_origin,
            allowed_origins,
        )
        raise ProxyAuthError("Domain verification failed", code="DOMAIN_MISMATCH", status_code=403)

    try:
        payload = jwt.decode(
            signed_token,
            key=app_ctx.public_key,
            algorithms=["ES256", "RS256"],
            options={"require": ["iss", "exp"]},
        )
    except ExpiredSignatureError as exc:
        raise ProxyAuthError("Signed token expired", code="TOKEN_EXPIRED") from exc
    except InvalidTokenError as exc:
        raise ProxyAuthError("Invalid signature", code="INVALID_SIGNATURE", status_code=401) from exc

    actor_id = str(payload.get("actor_id", ""))
    actor_type = str(payload.get("actor_type", "student"))
    derived_role = "admin" if actor_type == "admin" else "app"

    user = AuthenticatedUser(
        actor_id=actor_id,
        project_name=app_ctx.app_name,
        role=derived_role,
        token_id="",
        app_id=app_ctx.app_id,
        actor_type=actor_type,
        allowed_workflows=tuple(app_ctx.allowed_workflows),
        allowed_tools=tuple(app_ctx.allowed_tools),
    )
    return user, app_ctx


def _build_token_payload(user: AuthenticatedUser, token_type: str, ttl: timedelta) -> Dict[str, Any]:
    """Build a signed JWT payload for a proxy user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.actor_id or user.app_id,
        "actor_id": user.actor_id,
        "project_name": user.project_name,
        "role": user.role,
        "type": token_type,
        "iat": now,
        "exp": now + ttl,
        "jti": str(uuid.uuid4()),
        "iss": settings.proxy_token_issuer,
        "app_id": user.app_id,
        "actor_type": user.actor_type,
        "allowed_workflows": list(user.allowed_workflows),
        "allowed_tools": list(user.allowed_tools),
        "force_workflow": user.force_workflow,
        "phone": user.phone,
        "phone_verified": user.phone_verified,
    }
    return payload


def issue_access_token(user: AuthenticatedUser) -> Tuple[str, int]:
    """Issue a short-lived access token for a proxy user."""
    ttl_s = settings.proxy_access_token_ttl_minutes * 60
    payload = _build_token_payload(
        user=user,
        token_type="access",
        ttl=timedelta(seconds=ttl_s),
    )
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.proxy_jwt_algorithm)
    return token, ttl_s


def issue_refresh_token(user: AuthenticatedUser) -> Tuple[str, int]:
    """Issue a refresh token for a proxy user."""
    ttl_s = settings.proxy_refresh_token_ttl_hours * 3600
    payload = _build_token_payload(
        user=user,
        token_type="refresh",
        ttl=timedelta(seconds=ttl_s),
    )
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.proxy_jwt_algorithm)
    return token, ttl_s


def decode_token(token: str, expected_type: str) -> AuthenticatedUser:
    """Decode and validate a proxy JWT."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.proxy_jwt_algorithm],
            issuer=settings.proxy_token_issuer,
            options={"require": ["exp", "iat", "iss", "jti", "sub", "type"]},
        )
    except ExpiredSignatureError as exc:
        raise ProxyAuthError("Token expired", code="TOKEN_EXPIRED") from exc
    except InvalidTokenError as exc:
        raise ProxyAuthError("Invalid token", code="INVALID_TOKEN") from exc

    token_type = payload.get("type")
    if token_type != expected_type:
        raise ProxyAuthError(
            f"Invalid token type: expected {expected_type}",
            code="INVALID_TOKEN_TYPE",
            status_code=400,
        )

    return AuthenticatedUser(
        actor_id=str(payload.get("actor_id", "")),
        project_name=str(payload.get("project_name", "unknown")),
        role=str(payload.get("role", "student")),
        token_id=str(payload["jti"]),
        app_id=str(payload.get("app_id", "")),
        actor_type=str(payload.get("actor_type", "student")),
        allowed_workflows=tuple(payload.get("allowed_workflows", ())),
        allowed_tools=tuple(payload.get("allowed_tools", ())),
        force_workflow=str(payload.get("force_workflow", "")),
        phone=str(payload.get("phone", "")),
        phone_verified=bool(payload.get("phone_verified", False)),
    )
