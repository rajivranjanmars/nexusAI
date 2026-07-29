"""
SQLAlchemy ORM model for the Semantic Response Cache table.

Maps to the ``response_cache`` table in PostgreSQL (pgvector) and stores
full workflow responses keyed by question embedding similarity.  Enables
near-zero-cost repeat answers for semantically equivalent queries.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text, Uuid, func, JSON
from sqlalchemy.orm import Mapped, mapped_column

from db.models.base import Base


class ResponseCache(Base):
    """A cached workflow response with its question embedding."""

    __tablename__ = "response_cache"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    app_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid, nullable=True, index=True,
    )
    workflow: Mapped[str] = mapped_column(
        String(256), nullable=False, index=True,
    )
    question_text: Mapped[str] = mapped_column(
        Text, nullable=False,
    )
    # question_embedding is managed via raw SQL (pgvector column)
    # �?" not representable in standard SQLAlchemy without the pgvector extension type.
    response_text: Mapped[str] = mapped_column(
        Text, nullable=False,
    )
    tool_results: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True,
    )
    hit_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    expires_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    invalidated_by: Mapped[Optional[List[str]]] = mapped_column(
        JSON, nullable=True,
    )

    def to_dict(self) -> dict:
        """Serialize the model to a plain dictionary."""
        return {
            "id": str(self.id),
            "app_id": str(self.app_id) if self.app_id else None,
            "workflow": self.workflow,
            "question_text": self.question_text,
            "response_text": self.response_text,
            "tool_results": self.tool_results,
            "hit_count": self.hit_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "invalidated_by": self.invalidated_by,
        }