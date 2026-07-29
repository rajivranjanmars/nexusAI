"""
SQLAlchemy ORM model for the App Registry table.

Maps to the ``apps`` table in PostgreSQL and holds per-app configuration
including allowed workflows, allowed tools, rate limits, token quotas,
and visual configuration (theme, colors, icons).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text, Uuid, func, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


from db.models.base import Base


class App(Base):
    """
    Represents a registered application in the App Registry.
    
    The app_config field supports hierarchical configuration:
    {
        "system_prompt": {
            "base": "You are LPU AI Buddy, an admission counselor assistant...",
            "persona": {
                "name": "LPU AI Buddy",
                "tone": "Friendly, informative, and encouraging",
                "rules": ["Always provide accurate admission information"],
                "example_response": "Example response text"
            }
        },
        "workflow_config": {
            "admission_inquiry": {
                "system_suffix": "Provide comprehensive admission guidance...",
                "response_style": {
                    "include_sources": true,
                    "max_rag_chars": 6000,
                    "format_instructions": "Use detailed formatting..."
                }
            },
            "lead_capture": {
                "system_suffix": "Keep responses brief for lead capture...",
                "response_style": {
                    "include_sources": false,
                    "max_rag_chars": 2000,
                    "format_instructions": "Be concise and direct..."
                }
            }
        },
        "response_styles": {
            "custom_detailed": {
                "include_sources": true,
                "max_rag_chars": 8000,
                "format_instructions": "Custom formatting rules..."
            }
        }
    }
    """

    __tablename__ = "apps"

    app_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    app_name: Mapped[str] = mapped_column(Text, nullable=False)
    public_key: Mapped[str] = mapped_column(
        Text, nullable=False, comment="PEM formatted ES256 public key",
    )
    domain: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    allowed_workflows: Mapped[List[str]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    allowed_tools: Mapped[List[str]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    llm_model_override: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    rate_limit_rpm: Mapped[int] = mapped_column(
        Integer, nullable=False, default=60,
    )
    token_quota_monthly: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    cache_ttl_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3600,
    )
    otp_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        comment="Per-app toggle for guest phone/OTP verification.",
    )
    caching_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
        comment="Per-app toggle for semantic response caching.",
    )
    # ── Per-app OTP behavior knobs (non-secret; safe to cache in AppContext) ──
    otp_length: Mapped[int] = mapped_column(
        Integer, nullable=False, default=6,
        comment="OTP digit count. Server-clamped to a safe range at use time.",
    )
    otp_ttl_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, default=300,
        comment="OTP validity window in seconds. Server-clamped at use time.",
    )
    otp_max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5,
        comment="Wrong-guess cap per OTP. Server-clamped at use time.",
    )
    otp_resend_cooldown_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, default=30,
        comment="Seconds a mobile must wait between OTP requests.",
    )
    otp_max_requests_per_hour: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5,
        comment="Rolling-hour per-mobile OTP request cap (anti SMS-bombing).",
    )
    sms_daily_cap: Mapped[int] = mapped_column(
        Integer, nullable=False, default=500,
        comment="Max OTP SMS this app may send per day (cost/abuse budget).",
    )
    # ── Per-app SMS gateway credentials (SECRET; Fernet-encrypted blob) ───────
    # NEVER serialized to a client, never added to AppContext / to_dict /
    # admin-detail / the audit diff. Decrypted only at OTP-send time.
    sms_credentials_encrypted: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Fernet-encrypted JSON of per-app SMS gateway credentials; never serialized to clients.",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )
    app_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "app_config", JSON, nullable=True,
        comment="Visual/UI config: theme, colors, icons, logo_url, etc.",
    )
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSON, nullable=True,
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
    )

    def to_dict(self) -> dict:
        """Serialize the model to a plain dictionary."""
        return {
            "app_id": str(self.app_id),
            "app_name": self.app_name,
            "public_key": self.public_key,
            "domain": self.domain,
            "allowed_workflows": self.allowed_workflows,
            "allowed_tools": self.allowed_tools,
            "llm_model_override": self.llm_model_override,
            "rate_limit_rpm": self.rate_limit_rpm,
            "token_quota_monthly": self.token_quota_monthly,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "otp_enabled": self.otp_enabled,
            "caching_enabled": self.caching_enabled,
            "otp_length": self.otp_length,
            "otp_ttl_seconds": self.otp_ttl_seconds,
            "otp_max_attempts": self.otp_max_attempts,
            "otp_resend_cooldown_seconds": self.otp_resend_cooldown_seconds,
            "otp_max_requests_per_hour": self.otp_max_requests_per_hour,
            "sms_daily_cap": self.sms_daily_cap,
            "is_active": self.is_active,
            "app_config": self.app_config,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
