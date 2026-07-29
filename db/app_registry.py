"""
App Registry resolver using SQLAlchemy ORM.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from sqlalchemy import select
from db.models.app import App
from db.postgres import get_sync_session
from shared.logger import get_logger

logger = get_logger(__name__)

_CACHE_TTL_SECONDS: int = 300
_cache_by_app_id: Dict[str, _CachedApp] = {}

@dataclass(frozen=True)
class AppContext:
    app_id: str
    app_name: str
    public_key: str
    domain: str
    allowed_workflows: List[str]
    allowed_tools: List[str]
    llm_model_override: Optional[str]
    rate_limit_rpm: int
    token_quota_monthly: Optional[int]
    cache_ttl_seconds: int
    otp_enabled: bool
    caching_enabled: bool
    otp_length: int
    otp_ttl_seconds: int
    otp_max_attempts: int
    otp_resend_cooldown_seconds: int
    otp_max_requests_per_hour: int
    sms_daily_cap: int
    is_active: bool
    app_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    # NOTE: the encrypted SMS credential blob is deliberately NOT part of
    # AppContext. Keeping it out of the 5-minute in-process cache (and out of
    # anything handed to orchestration/prompt code) means the ciphertext is only
    # ever read fresh from the DB at OTP-send time — see
    # backend_proxy.main._fetch_sms_credentials.

@dataclass
class _CachedApp:
    context: AppContext
    expires_at: float



def _app_to_context(app: App) -> AppContext:
    return AppContext(
        app_id=str(app.app_id),
        app_name=app.app_name,
        public_key=app.public_key,
        domain=app.domain,
        allowed_workflows=app.allowed_workflows,
        allowed_tools=app.allowed_tools,
        llm_model_override=app.llm_model_override,
        rate_limit_rpm=app.rate_limit_rpm,
        token_quota_monthly=app.token_quota_monthly,
        cache_ttl_seconds=app.cache_ttl_seconds,
        otp_enabled=app.otp_enabled,
        caching_enabled=app.caching_enabled,
        otp_length=app.otp_length,
        otp_ttl_seconds=app.otp_ttl_seconds,
        otp_max_attempts=app.otp_max_attempts,
        otp_resend_cooldown_seconds=app.otp_resend_cooldown_seconds,
        otp_max_requests_per_hour=app.otp_max_requests_per_hour,
        sms_daily_cap=app.sms_daily_cap,
        is_active=app.is_active,
        app_config=app.app_config,
        metadata=app.metadata_,
    )



def resolve_by_app_id(app_id: str) -> Optional[AppContext]:
    cached = _cache_by_app_id.get(app_id)
    if cached and cached.expires_at > time.time():
        return cached.context

    with get_sync_session() as session:
        app = session.execute(select(App).where(App.app_id == app_id, App.is_active == True)).scalar_one_or_none()
        if not app:
            return None
        
        context = _app_to_context(app)
        _cache_app(context)
        return context

def _cache_app(context: AppContext) -> None:
    entry = _CachedApp(context=context, expires_at=time.time() + _CACHE_TTL_SECONDS)
    _cache_by_app_id[context.app_id] = entry

def invalidate_cache() -> None:
    _cache_by_app_id.clear()
