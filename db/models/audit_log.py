"""
SQLAlchemy ORM models for audit logs.

AppAuditLog: Per-app configuration audit events.
  Records who changed what on a registered app (and when), so toggles like
  ``otp_enabled`` / ``caching_enabled`` have a traceable change history.

AdminAuditLog: Admin dashboard audit log.
  Records every admin mutation (create, update, delete, activate, deactivate)
  with the acting admin user, target, and optional before/after details.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict

from sqlalchemy import BigInteger, DateTime, ForeignKey, String, Uuid, func, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db.models.base import Base


class AppAuditLog(Base):
    """One audited change made to a registered application."""

    __tablename__ = "app_audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    app_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, index=True)
    changed_by: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    changes: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    def to_dict(self) -> dict:
        """Serialize the model to a plain dictionary."""
        return {
            "id": str(self.id),
            "app_id": str(self.app_id),
            "changed_by": self.changed_by,
            "action": self.action,
            "changes": self.changes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AdminAuditLog(Base):
    """Immutable record of an admin action for compliance and debugging."""

    __tablename__ = "admin_audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    admin_user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("admin_users.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    action: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True,
        comment="e.g. app.update, user.create, rag.crawl, cache.clear",
    )
    target_type: Mapped[str] = mapped_column(
        String(64), nullable=False,
        comment="e.g. app, admin_user, rag, cache",
    )
    target_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    details: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="Before/after snapshots, payload, or summary",
    )
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
