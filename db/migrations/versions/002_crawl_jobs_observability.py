"""add_crawl_jobs_observability_columns

Revision ID: 002_crawl_jobs_observability
Revises: 001_add_crawl_jobs_table
Create Date: 2026-04-24

Adds URL-level counters, last_activity_at, and pages_per_minute to crawl_jobs
for live progress tracking and stuck-job detection.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '002_crawl_jobs_observability'
down_revision: Union[str, None] = '001_add_crawl_jobs_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('crawl_jobs', sa.Column('urls_queued',            sa.Integer(), nullable=False, server_default='0'))
    op.add_column('crawl_jobs', sa.Column('urls_processed',         sa.Integer(), nullable=False, server_default='0'))
    op.add_column('crawl_jobs', sa.Column('urls_skipped_quality',   sa.Integer(), nullable=False, server_default='0'))
    op.add_column('crawl_jobs', sa.Column('urls_skipped_duplicate', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('crawl_jobs', sa.Column('urls_failed',            sa.Integer(), nullable=False, server_default='0'))
    op.add_column('crawl_jobs', sa.Column('last_activity_at',       sa.DateTime(), nullable=True))
    op.add_column('crawl_jobs', sa.Column('pages_per_minute',       sa.Float(),    nullable=True))


def downgrade() -> None:
    op.drop_column('crawl_jobs', 'pages_per_minute')
    op.drop_column('crawl_jobs', 'last_activity_at')
    op.drop_column('crawl_jobs', 'urls_failed')
    op.drop_column('crawl_jobs', 'urls_skipped_duplicate')
    op.drop_column('crawl_jobs', 'urls_skipped_quality')
    op.drop_column('crawl_jobs', 'urls_processed')
    op.drop_column('crawl_jobs', 'urls_queued')
