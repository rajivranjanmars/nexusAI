"""add_crawl_jobs_table

Revision ID: 001_add_crawl_jobs_table
Revises: 
Create Date: 2026-04-24 10:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_add_crawl_jobs_table'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'crawl_jobs',
        sa.Column('job_id', sa.Uuid(as_uuid=True), nullable=False),
        sa.Column('app_id', sa.String(length=255), nullable=False),
        sa.Column('seed_url', sa.String(length=2048), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('pages_found', sa.Integer(), nullable=False),
        sa.Column('pages_skipped', sa.Integer(), nullable=False),
        sa.Column('pages_failed', sa.Integer(), nullable=False),
        sa.Column('chunks_upserted', sa.Integer(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('job_id')
    )
    op.create_index(op.f('ix_crawl_jobs_app_id'), 'crawl_jobs', ['app_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_crawl_jobs_app_id'), table_name='crawl_jobs')
    op.drop_table('crawl_jobs')
