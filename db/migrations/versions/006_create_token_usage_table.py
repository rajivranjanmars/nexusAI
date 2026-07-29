"""create token_usage table

Revision ID: 006_create_token_usage_table
Revises: 005_drop_api_key_hash
Create Date: 2026-04-29
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "006_create_token_usage_table"
down_revision: Union[str, None] = "005_drop_api_key_hash"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if _has_table("token_usage"):
        return

    op.create_table(
        "token_usage",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("app_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("actor_id", sa.String(length=128), nullable=True),
        sa.Column("actor_type", sa.String(length=32), nullable=True),
        sa.Column("action", sa.Text(), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("model_tier", sa.String(length=16), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("estimated_cost_usd", sa.Numeric(10, 6), nullable=True),
        sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("correlation_id", sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_token_usage_app_id", "token_usage", ["app_id"])
    op.create_index("ix_token_usage_actor_id", "token_usage", ["actor_id"])
    op.create_index("ix_token_usage_correlation_id", "token_usage", ["correlation_id"])
    op.create_index("ix_token_usage_timestamp", "token_usage", ["timestamp"])


def downgrade() -> None:
    op.drop_table("token_usage")
