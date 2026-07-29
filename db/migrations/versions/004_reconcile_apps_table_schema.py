"""reconcile apps table schema

Revision ID: 004_reconcile_apps_table
Revises: 003_crawl_job_audit
Create Date: 2026-04-29
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision: str = "004_reconcile_apps_table"
down_revision: Union[str, None] = "003_crawl_job_audit"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    return {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    if not _has_table("apps"):
        op.create_table(
            "apps",
            sa.Column("app_id", sa.Uuid(as_uuid=True), nullable=False),
            sa.Column("app_name", sa.Text(), nullable=False),
            sa.Column("public_key", sa.Text(), nullable=False),
            sa.Column("domain", sa.String(length=255), nullable=False),
            sa.Column("allowed_workflows", sa.JSON(), nullable=False),
            sa.Column("allowed_tools", sa.JSON(), nullable=False),
            sa.Column("llm_model_override", sa.Text(), nullable=True),
            sa.Column("rate_limit_rpm", sa.Integer(), nullable=False, server_default="60"),
            sa.Column("token_quota_monthly", sa.Integer(), nullable=True),
            sa.Column("cache_ttl_seconds", sa.Integer(), nullable=False, server_default="3600"),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("app_config", sa.JSON(), nullable=True),
            sa.Column("metadata", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint("app_id"),
        )
        return

    existing_columns = _column_names("apps")

    def add_column_if_missing(name: str, column: sa.Column) -> None:
        if name not in existing_columns:
            op.add_column("apps", column)

    add_column_if_missing(
        "public_key",
        sa.Column("public_key", sa.Text(), nullable=False, server_default=""),
    )
    add_column_if_missing(
        "domain",
        sa.Column("domain", sa.String(length=255), nullable=False, server_default="http://localhost:3000"),
    )
    add_column_if_missing(
        "allowed_workflows",
        sa.Column("allowed_workflows", sa.JSON(), nullable=False, server_default="[]"),
    )
    add_column_if_missing(
        "allowed_tools",
        sa.Column("allowed_tools", sa.JSON(), nullable=False, server_default="[]"),
    )
    add_column_if_missing(
        "llm_model_override",
        sa.Column("llm_model_override", sa.Text(), nullable=True),
    )
    add_column_if_missing(
        "rate_limit_rpm",
        sa.Column("rate_limit_rpm", sa.Integer(), nullable=False, server_default="60"),
    )
    add_column_if_missing(
        "token_quota_monthly",
        sa.Column("token_quota_monthly", sa.Integer(), nullable=True),
    )
    add_column_if_missing(
        "cache_ttl_seconds",
        sa.Column("cache_ttl_seconds", sa.Integer(), nullable=False, server_default="3600"),
    )
    add_column_if_missing(
        "is_active",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    add_column_if_missing(
        "app_config",
        sa.Column("app_config", sa.JSON(), nullable=True),
    )
    add_column_if_missing(
        "metadata",
        sa.Column("metadata", sa.JSON(), nullable=True),
    )
    add_column_if_missing(
        "created_at",
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
    )
    add_column_if_missing(
        "updated_at",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
    )

    pass


def downgrade() -> None:
    # Intentionally non-destructive: this migration is a production schema
    # reconciliation step and should not blindly drop columns on downgrade.
    pass
