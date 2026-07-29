"""create chat_feedback table

Revision ID: 007_create_chat_feedback_table
Revises: 006_create_token_usage_table
Create Date: 2026-05-29
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "007_create_chat_feedback_table"
down_revision: Union[str, None] = "006_create_token_usage_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if _has_table("chat_feedback"):
        return

    op.create_table(
        "chat_feedback",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("app_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("actor_id", sa.String(length=255), nullable=False),
        sa.Column("feedback_flag", sa.Boolean(), nullable=False),
        sa.Column("feedback_text", sa.Text(), nullable=True),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_chat_feedback_session_id", "chat_feedback", ["session_id"])
    op.create_index("ix_chat_feedback_app_id", "chat_feedback", ["app_id"])
    op.create_index("ix_chat_feedback_actor_id", "chat_feedback", ["actor_id"])
    op.create_index("ix_chat_feedback_created_at", "chat_feedback", ["created_at"])


def downgrade() -> None:
    op.drop_table("chat_feedback")
