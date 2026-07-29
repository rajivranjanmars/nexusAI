"""create admin_audit_log table

Revision ID: 009_create_admin_audit_log_table
Revises: 008_create_admin_users_table
Create Date: 2026-07-13
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision: str = "009_create_admin_audit_log_table"
down_revision: Union[str, None] = "008_create_admin_users_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if _has_table("admin_audit_log"):
        return

    op.create_table(
        "admin_audit_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "admin_user_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("admin_users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "action",
            sa.String(length=128),
            nullable=False,
            comment="e.g. app.update, user.create, rag.crawl, cache.clear",
        ),
        sa.Column(
            "target_type",
            sa.String(length=64),
            nullable=False,
            comment="e.g. app, admin_user, rag, cache",
        ),
        sa.Column("target_id", sa.String(length=255), nullable=True),
        sa.Column(
            "details",
            postgresql.JSONB,
            nullable=True,
            comment="Before/after snapshots, payload, or summary",
        ),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_audit_log_admin_user", "admin_audit_log", ["admin_user_id"])
    op.create_index("ix_audit_log_action", "admin_audit_log", ["action"])
    op.create_index("ix_audit_log_created_at", "admin_audit_log", ["created_at"])


def downgrade() -> None:
    op.drop_table("admin_audit_log")