"""create admin_users table

Revision ID: 008_create_admin_users_table
Revises: 007_create_chat_feedback_table
Create Date: 2026-07-13
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "008_create_admin_users_table"
down_revision: Union[str, None] = "007_create_chat_feedback_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if _has_table("admin_users"):
        return

    op.create_table(
        "admin_users",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column(
            "role",
            sa.String(length=32),
            nullable=False,
            server_default="app_admin",
            comment="super_admin | app_admin",
        ),
        sa.Column(
            "app_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("apps.app_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )

    op.create_index("ix_admin_users_email", "admin_users", ["email"])
    op.create_index("ix_admin_users_role", "admin_users", ["role"])
    op.create_index("ix_admin_users_app_id", "admin_users", ["app_id"])

    # ponytail: inline CHECKs via raw SQL since Alembic doesn't emit them
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'chk_admin_role'
            ) THEN
                ALTER TABLE admin_users ADD CONSTRAINT chk_admin_role
                    CHECK (role IN ('super_admin', 'app_admin'));
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'chk_app_admin_has_app'
            ) THEN
                ALTER TABLE admin_users ADD CONSTRAINT chk_app_admin_has_app
                    CHECK (role = 'super_admin' OR app_id IS NOT NULL);
            END IF;
        END;
        $$;
    """)


def downgrade() -> None:
    op.drop_table("admin_users")