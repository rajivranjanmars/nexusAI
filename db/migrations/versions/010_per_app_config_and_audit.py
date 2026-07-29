"""per-app feature toggles, OTP/SMS config, and app audit log

Adds the per-app OTP/caching toggles, the per-app OTP behavior knobs, the
encrypted SMS-credentials column, and the app_audit_log table — all in one
migration since they ship together as a single feature.

Revision ID: 008_per_app_config_and_audit
Revises: 007_create_chat_feedback_table
Create Date: 2026-07-08
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "008_per_app_config_and_audit"
down_revision: Union[str, None] = "007_create_chat_feedback_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    """Return whether the current database contains ``table_name``."""
    return table_name in inspect(op.get_bind()).get_table_names()


def _column_names(table_name: str) -> set[str]:
    """Return the column names currently present on ``table_name``."""
    return {c["name"] for c in inspect(op.get_bind()).get_columns(table_name)}


# apps columns: (name, type, server_default) — nullable=False except the blob.
_APP_COLUMNS = [
    ("otp_enabled", sa.Boolean(), sa.false()),
    ("caching_enabled", sa.Boolean(), sa.true()),
    ("otp_length", sa.Integer(), "6"),
    ("otp_ttl_seconds", sa.Integer(), "300"),
    ("otp_max_attempts", sa.Integer(), "5"),
    ("otp_resend_cooldown_seconds", sa.Integer(), "30"),
    ("otp_max_requests_per_hour", sa.Integer(), "5"),
    ("sms_daily_cap", sa.Integer(), "500"),
]


def upgrade() -> None:
    """Add per-app feature configuration and its audit trail."""
    existing = _column_names("apps")
    for name, col_type, default in _APP_COLUMNS:
        if name not in existing:
            op.add_column(
                "apps",
                sa.Column(name, col_type, nullable=False, server_default=default),
            )
    if "sms_credentials_encrypted" not in existing:
        # Fernet-encrypted JSON of per-app SMS gateway credentials; never
        # serialized to clients. Nullable — apps without OTP have none.
        op.add_column("apps", sa.Column("sms_credentials_encrypted", sa.Text(), nullable=True))

    if not _has_table("app_audit_log"):
        op.create_table(
            "app_audit_log",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("app_id", sa.Uuid(), nullable=False),
            sa.Column("changed_by", sa.String(length=255), nullable=False),
            sa.Column("action", sa.String(length=50), nullable=False),
            sa.Column("changes", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_app_audit_log_app_id", "app_audit_log", ["app_id"])
        op.create_index("ix_app_audit_log_created_at", "app_audit_log", ["created_at"])


def downgrade() -> None:
    """Remove per-app feature configuration and its audit trail."""
    if _has_table("app_audit_log"):
        op.drop_table("app_audit_log")
    existing = _column_names("apps")
    if "sms_credentials_encrypted" in existing:
        op.drop_column("apps", "sms_credentials_encrypted")
    for name, _type, _default in reversed(_APP_COLUMNS):
        if name in existing:
            op.drop_column("apps", name)
