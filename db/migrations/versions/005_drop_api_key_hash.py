"""drop api_key_hash from apps table

Revision ID: 005_drop_api_key_hash
Revises: 004_reconcile_apps_table
Create Date: 2026-04-29
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "005_drop_api_key_hash"
down_revision: Union[str, None] = "004_reconcile_apps_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    return {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    existing_columns = _column_names("apps")
    if "api_key_hash" in existing_columns:
        op.drop_index("ix_apps_api_key_hash", table_name="apps")
        op.drop_column("apps", "api_key_hash")


def downgrade() -> None:
    # Intentionally non-destructive.
    pass
