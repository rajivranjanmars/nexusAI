"""
SQLAlchemy ORM model for the Token Usage tracking table.

Maps to the ``token_usage`` table in PostgreSQL and records every LLM call
with app, actor, model, token counts, and cost for billing and analytics.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, Numeric, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from db.models.base import Base


class TokenUsage(Base):
    """Records a single LLM invocation with usage and cost metadata."""

    __tablename__ = "token_usage"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    app_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid, nullable=True, index=True,
    )
    actor_id: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, index=True,
    )
    actor_type: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True,
    )
    action: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    model: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    model_tier: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True,
    )
    prompt_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    completion_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    estimated_cost_usd: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 6), nullable=True,
    )
    cache_hit: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, index=True,
    )

    def to_dict(self) -> dict:
        """Serialize the model to a plain dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "app_id": str(self.app_id) if self.app_id else None,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "action": self.action,
            "model": self.model,
            "model_tier": self.model_tier,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": float(self.estimated_cost_usd) if self.estimated_cost_usd else None,
            "cache_hit": self.cache_hit,
            "latency_ms": self.latency_ms,
            "correlation_id": self.correlation_id,
        }
