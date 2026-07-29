"""
SQLAlchemy models for RAG knowledge ingestion.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)

from db.models.base import Base


class CrawlJob(Base):
    """Tracks the status and live progress of an async crawl + ingestion job."""

    __tablename__ = "crawl_jobs"

    job_id = Column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    app_id          = Column(String(255), nullable=False, index=True)
    seed_url        = Column(String(2048), nullable=False)
    status          = Column(String(50), nullable=False, default="queued")

    # ── URL-level counters (from CrawlStats) ─────────────────────────────────
    urls_queued     = Column(Integer, nullable=False, default=0)
    urls_processed  = Column(Integer, nullable=False, default=0)
    urls_skipped_quality   = Column(Integer, nullable=False, default=0)
    urls_skipped_duplicate = Column(Integer, nullable=False, default=0)
    urls_failed     = Column(Integer, nullable=False, default=0)

    # ── Ingestion counters ────────────────────────────────────────────────────
    pages_found     = Column(Integer, nullable=False, default=0)
    pages_skipped   = Column(Integer, nullable=False, default=0)
    pages_failed    = Column(Integer, nullable=False, default=0)
    chunks_upserted = Column(Integer, nullable=False, default=0)

    # ── Timing ───────────────────────────────────────────────────────────────
    started_at      = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at     = Column(DateTime, nullable=True)
    last_activity_at = Column(DateTime, nullable=True)   # updated during crawl; used for stuck detection

    # ── Diagnostics ──────────────────────────────────────────────────────────
    pages_per_minute = Column(Float, nullable=True)
    expected_urls = Column(Integer, nullable=False, default=0)
    expected_urls_from_sitemap = Column(Integer, nullable=False, default=0)
    coverage_pct = Column(Float, nullable=True)
    error           = Column(Text, nullable=True)


class CrawlJobPage(Base):
    """Per-URL audit trail for a crawl job."""

    __tablename__ = "crawl_job_pages"
    __table_args__ = (
        UniqueConstraint("job_id", "url", name="uq_crawl_job_pages_job_id_url"),
    )

    page_id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("crawl_jobs.job_id"),
        nullable=False,
        index=True,
    )
    url = Column(String(2048), nullable=False)
    section = Column(String(255), nullable=True)
    discovered_via = Column(String(50), nullable=False, default="crawl")
    depth = Column(Integer, nullable=True)
    crawl_status = Column(String(50), nullable=False, default="discovered")
    ingestion_status = Column(String(50), nullable=False, default="pending")
    http_status = Column(Integer, nullable=True)
    fetch_ms = Column(Integer, nullable=True)
    content_chars = Column(Integer, nullable=True)
    title = Column(String(512), nullable=True)
    content_hash = Column(String(64), nullable=True)
    chunk_count = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)
    crawled_at = Column(DateTime, nullable=True)
    ingested_at = Column(DateTime, nullable=True)
