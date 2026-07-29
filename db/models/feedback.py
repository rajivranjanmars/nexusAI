"""
SQLAlchemy ORM model for chat feedback records.
"""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import Boolean, DateTime, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from db.models.base import Base


class ChatFeedback(Base):
    """Stores one feedback submission tied to a chat session and response."""

    __tablename__ = "chat_feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    app_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, index=True)
    actor_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    feedback_flag: Mapped[bool] = mapped_column(Boolean, nullable=False)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    def to_dict(self) -> dict[str, str | bool | None]:
        """Serialize the model to a plain dictionary."""

        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "app_id": str(self.app_id),
            "actor_id": self.actor_id,
            "feedback_flag": self.feedback_flag,
            "feedback_text": self.feedback_text,
            "user_message": self.user_message,
            "response": self.response,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
