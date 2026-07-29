"""add crawl job page audit and coverage fields

Revision ID: 003_crawl_job_audit
Revises: 002_crawl_jobs_observability
Create Date: 2026-04-28
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "003_crawl_job_audit"
down_revision: Union[str, None] = "002_crawl_jobs_observability"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("crawl_jobs", sa.Column("expected_urls", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("crawl_jobs", sa.Column("expected_urls_from_sitemap", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("crawl_jobs", sa.Column("coverage_pct", sa.Float(), nullable=True))

    op.create_table(
        "crawl_job_pages",
        sa.Column("page_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False),
        sa.Column("section", sa.String(length=255), nullable=True),
        sa.Column("discovered_via", sa.String(length=50), nullable=False, server_default="crawl"),
        sa.Column("depth", sa.Integer(), nullable=True),
        sa.Column("crawl_status", sa.String(length=50), nullable=False, server_default="discovered"),
        sa.Column("ingestion_status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("fetch_ms", sa.Integer(), nullable=True),
        sa.Column("content_chars", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("crawled_at", sa.DateTime(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["crawl_jobs.job_id"]),
        sa.UniqueConstraint("job_id", "url", name="uq_crawl_job_pages_job_id_url"),
    )
    op.create_index(op.f("ix_crawl_job_pages_job_id"), "crawl_job_pages", ["job_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_crawl_job_pages_job_id"), table_name="crawl_job_pages")
    op.drop_table("crawl_job_pages")
    op.drop_column("crawl_jobs", "coverage_pct")
    op.drop_column("crawl_jobs", "expected_urls_from_sitemap")
    op.drop_column("crawl_jobs", "expected_urls")
