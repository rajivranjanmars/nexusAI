"""Merge the admin and per-app configuration migration branches.

Both branches were released from ``007_create_chat_feedback_table``. The
per-app revision identifier is intentionally retained because deployed
databases already reference it in ``alembic_version``.

Revision ID: 011_merge_migration_heads
Revises: ('008_per_app_config_and_audit', '009_create_admin_audit_log_table')
Create Date: 2026-07-18
"""

revision: str = "011_merge_migration_heads"
down_revision: tuple[str, str] = (
    "008_per_app_config_and_audit",
    "009_create_admin_audit_log_table",
)
branch_labels: None = None
depends_on: None = None


def upgrade() -> None:
    """Record convergence of the admin and per-app migration branches."""


def downgrade() -> None:
    """Return Alembic bookkeeping to the two pre-merge branch heads."""
