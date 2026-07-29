"""
RAG Administration API Endpoints.

Handles triggering background crawls and manual knowledge ingestion.
Protected by admin role requirement.
"""
import asyncio
import base64
import hashlib
import json
import uuid
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from sqlalchemy import case, func, select

from backend_proxy.admin_auth import AdminUserContext
from backend_proxy.deps import get_admin_scope, require_admin_hybrid, require_app_admin
from db.crawler import AsyncCrawler
from db.ingestion import IngestPipeline
from db.models.knowledge import CrawlJob, CrawlJobPage
from db.postgres import get_session
from shared.chroma_client import get_collection, get_parent_collection
from shared.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


class CrawlRequest(BaseModel):
    app_id: str
    seed_url: str
    max_depth: int = 2
    allowed_paths: Optional[list[str]] = None
    auto_discover_sitemap: bool = True


class IngestRequest(BaseModel):
    app_id: str
    source_type: Literal["file", "url", "structured"]
    content: str
    source_ref: str


def _build_section_summary(rows: list[tuple[str, int, int, int, int]]) -> list[dict[str, int | str]]:
    return [
        {
            "section": section or "/",
            "total_urls": total_urls,
            "extracted_urls": extracted_urls,
            "ingested_urls": ingested_urls,
            "total_chunks": total_chunks,
        }
        for section, total_urls, extracted_urls, ingested_urls, total_chunks in rows
    ]


async def _get_section_summary(session, job_uuid: uuid.UUID) -> list[dict[str, int | str]]:
    result = await session.execute(
        select(
            CrawlJobPage.section,
            func.count(CrawlJobPage.page_id).label("total_urls"),
            func.sum(case((CrawlJobPage.crawl_status == "extracted", 1), else_=0)).label("extracted_urls"),
            func.sum(case((CrawlJobPage.ingestion_status == "ingested", 1), else_=0)).label("ingested_urls"),
            func.sum(CrawlJobPage.chunk_count).label("total_chunks"),
        )
        .where(CrawlJobPage.job_id == job_uuid)
        .group_by(CrawlJobPage.section)
        .order_by(func.count(CrawlJobPage.page_id).desc())
    )
    return _build_section_summary(result.all())


async def _get_page_status_counts(session, job_uuid: uuid.UUID) -> dict[str, dict[str, int]]:
    crawl_rows = await session.execute(
        select(CrawlJobPage.crawl_status, func.count(CrawlJobPage.page_id))
        .where(CrawlJobPage.job_id == job_uuid)
        .group_by(CrawlJobPage.crawl_status)
    )
    ingestion_rows = await session.execute(
        select(CrawlJobPage.ingestion_status, func.count(CrawlJobPage.page_id))
        .where(CrawlJobPage.job_id == job_uuid)
        .group_by(CrawlJobPage.ingestion_status)
    )
    return {
        "crawl": {status or "unknown": count for status, count in crawl_rows.all()},
        "ingestion": {status or "unknown": count for status, count in ingestion_rows.all()},
    }


async def _serialize_job_status(session, job: CrawlJob) -> dict:
    _STUCK_THRESHOLD_S = 120
    now = datetime.utcnow()
    job_uuid = job.job_id

    if job.started_at:
        end_time = job.finished_at or now
        elapsed_s = int((end_time - job.started_at).total_seconds())
    else:
        elapsed_s = 0

    is_stuck = False
    if job.status == "processing" and job.last_activity_at:
        idle_s = (now - job.last_activity_at).total_seconds()
        is_stuck = idle_s > _STUCK_THRESHOLD_S

    progress_denominator = job.expected_urls or job.urls_queued
    progress_pct = None
    if progress_denominator and progress_denominator > 0:
        progress_pct = round(
            min(job.urls_processed / progress_denominator * 100, 100), 1
        )

    return {
        "job_id": str(job.job_id),
        "app_id": job.app_id,
        "seed_url": job.seed_url,
        "status": job.status,
        "progress_pct": progress_pct,
        "elapsed_seconds": elapsed_s,
        "pages_per_minute": job.pages_per_minute,
        "is_stuck": is_stuck,
        "coverage_pct": job.coverage_pct,
        "urls_queued": job.urls_queued,
        "urls_processed": job.urls_processed,
        "urls_skipped_quality": job.urls_skipped_quality,
        "urls_skipped_duplicate": job.urls_skipped_duplicate,
        "urls_failed": job.urls_failed,
        "expected_urls": job.expected_urls,
        "expected_urls_from_sitemap": job.expected_urls_from_sitemap,
        "pages_found": job.pages_found,
        "pages_skipped": job.pages_skipped,
        "pages_failed": job.pages_failed,
        "chunks_upserted": job.chunks_upserted,
        "page_status_counts": await _get_page_status_counts(session, job_uuid),
        "section_summary": await _get_section_summary(session, job_uuid),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "last_activity_at": job.last_activity_at.isoformat() if job.last_activity_at else None,
        "error": job.error,
    }

async def _run_crawl_background(
    app_id: str,
    job_id: str,
    seed_url: str,
    max_depth: int,
    allowed_paths: Optional[list[str]],
    auto_discover_sitemap: bool,
):
    """Background task to run crawler and ingestion."""
    pipeline = IngestPipeline(app_id=app_id, crawl_job_id=job_id)
    try:
        await pipeline.update_job_status("processing")

        crawler = AsyncCrawler(
            seed_url=seed_url,
            max_depth=max_depth,
            allowed_paths=allowed_paths,
            auto_discover_sitemap=auto_discover_sitemap,
            correlation_id=job_id,
        )

        pages = await crawler.run()
        url_stats = crawler.stats.to_dict()
        await pipeline.sync_page_audits([page.to_dict() for page in crawler.page_audits.values()])

        chunks_upserted = 0
        pages_skipped = 0
        pages_failed = 0

        for page in pages:
            try:
                upserted = await pipeline.ingest_page(
                    source_type="crawl",
                    source_ref=page.url,
                    source_url=page.url,
                    content=page.content,
                    content_hash=page.content_hash
                )
                chunks_upserted += upserted
                if upserted == 0:
                    pages_skipped += 1
                    await pipeline.update_page_ingestion_status(
                        source_ref=page.url,
                        status="deduplicated",
                        chunk_count=0,
                    )
                else:
                    await pipeline.update_page_ingestion_status(
                        source_ref=page.url,
                        status="ingested",
                        chunk_count=upserted,
                    )
                # Touch last_activity_at so stuck detection sees progress
                await pipeline.heartbeat()
            except Exception as exc:
                logger.error("Failed to ingest page %s: %s", page.url, exc)
                pages_failed += 1
                await pipeline.update_page_ingestion_status(
                    source_ref=page.url,
                    status="failed",
                    chunk_count=0,
                    error=str(exc),
                )

        await pipeline.update_job_status(
            status="done",
            pages_found=len(pages),
            pages_skipped=pages_skipped,
            pages_failed=pages_failed,
            chunks_upserted=chunks_upserted,
            url_stats=url_stats,
        )
        if chunks_upserted > 0:
            await pipeline.invalidate_shared_response_cache()

    except Exception as exc:
        logger.exception("Crawl job %s failed", job_id)
        await pipeline.update_job_status(status="failed", error=str(exc))


@router.post("/crawl")
async def crawl_website_endpoint(
    payload: CrawlRequest,
    background_tasks: BackgroundTasks,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
):
    """Trigger a background crawl of a website."""
    # ponytail: app_admin auto-scoped to their app
    if scope:
        payload.app_id = scope

    job_id = str(uuid.uuid4())

    async with get_session() as session:
        job = CrawlJob(
            job_id=uuid.UUID(job_id),
            app_id=payload.app_id,
            seed_url=payload.seed_url,
            status="queued"
        )
        session.add(job)
        await session.commit()

    background_tasks.add_task(
        _run_crawl_background,
        app_id=payload.app_id,
        job_id=job_id,
        seed_url=payload.seed_url,
        max_depth=payload.max_depth,
        allowed_paths=payload.allowed_paths,
        auto_discover_sitemap=payload.auto_discover_sitemap,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "auto_discover_sitemap": payload.auto_discover_sitemap,
    }


@router.post("/ingest")
async def ingest_knowledge_endpoint(
    payload: IngestRequest,
    request: Request,
):
    """Ingest a single piece of knowledge. Accepts app-level or admin dashboard tokens."""
    user = await require_admin_hybrid(request)
    if isinstance(user, AdminUserContext):
        if user.admin_role == "app_admin":
            payload.app_id = user.app_id
    else:
        if user.app_id:
            payload.app_id = user.app_id
    pipeline = IngestPipeline(app_id=payload.app_id)
    
    extracted_text = ""
    structured_source_url = ""

    if payload.source_type == "url":
        crawler = AsyncCrawler(seed_url=payload.content, max_depth=0, auto_discover_sitemap=False)
        pages = await crawler.run()
        if pages:
            extracted_text = pages[0].content
    elif payload.source_type == "file":
        import io
        import pypdf
        import docx
        
        file_bytes = base64.b64decode(payload.content)
        if payload.source_ref.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            extracted_text = "\n".join(page.extract_text() for page in reader.pages)
        elif payload.source_ref.lower().endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_bytes))
            extracted_text = "\n".join(p.text for p in doc.paragraphs)
        else:
            extracted_text = file_bytes.decode("utf-8", errors="replace")
    elif payload.source_type == "structured":
        try:
            records = json.loads(payload.content)
            if isinstance(records, list):
                extracted_text = "\n\n".join(
                    " | ".join(f"{k}: {v}" for k, v in rec.items())
                    for rec in records
                )
                if len(records) == 1 and isinstance(records[0], dict):
                    structured_source_url = str(records[0].get("source_url", "") or "")
            elif isinstance(records, dict):
                extracted_text = " | ".join(f"{k}: {v}" for k, v in records.items())
                structured_source_url = str(records.get("source_url", "") or "")
        except json.JSONDecodeError:
            extracted_text = payload.content

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted")

    content_hash = hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()

    upserted = await pipeline.ingest_page(
        source_type=payload.source_type,
        source_ref=payload.source_ref,
        source_url=payload.source_ref if payload.source_type == "url" else structured_source_url,
        content=extracted_text,
        content_hash=content_hash
    )
    cache_invalidated = 0
    if upserted > 0:
        cache_invalidated = await pipeline.invalidate_shared_response_cache()
    
    return {
        "status": "success",
        "chunks_upserted": upserted,
        "shared_cache_entries_invalidated": cache_invalidated,
    }


@router.delete("/app/{app_id}")
async def delete_app_knowledge(
    app_id: str,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
):
    """Delete all stored RAG chunks for a single app (child and parent collections)."""
    if scope and scope != app_id:
        raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")
    collection = get_collection()
    parent_collection = get_parent_collection()

    existing = collection.get(where={"app_id": app_id})
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])

    existing_parents = parent_collection.get(where={"app_id": app_id})
    if existing_parents and existing_parents["ids"]:
        parent_collection.delete(ids=existing_parents["ids"])

    pipeline = IngestPipeline(app_id=app_id)
    cache_invalidated = await pipeline.invalidate_shared_response_cache()

    return {
        "status": "deleted",
        "app_id": app_id,
        "chunks_deleted": len(existing["ids"]) if existing else 0,
        "parent_chunks_deleted": len(existing_parents["ids"]) if existing_parents else 0,
        "shared_cache_entries_invalidated": cache_invalidated,
    }


@router.get("/status/{job_id}")
async def get_crawl_status(
    job_id: str,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
):
    """Check the status and live progress of a crawl job."""
    async with get_session() as session:
        job_uuid = uuid.UUID(job_id)
        job = await session.get(CrawlJob, job_uuid)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        # ponytail: app_admin can only view their own app's jobs
        if scope and job.app_id != scope:
            raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")
        return await _serialize_job_status(session, job)


@router.get("/status/app/{app_id}/latest")
async def get_latest_crawl_status_for_app(
    app_id: str,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
):
    """Fetch the most recent crawl job status for an app."""
    if scope and scope != app_id:
        raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")
    async with get_session() as session:
        result = await session.execute(
            select(CrawlJob)
            .where(CrawlJob.app_id == app_id)
            .order_by(CrawlJob.started_at.desc())
            .limit(1)
        )
        job = result.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="No crawl jobs found for this app")
        return await _serialize_job_status(session, job)


@router.get("/status/{job_id}/pages")
async def get_crawl_status_pages(
    job_id: str,
    limit: int = 100,
    offset: int = 0,
    crawl_status: Optional[str] = None,
    ingestion_status: Optional[str] = None,
    section: Optional[str] = None,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
):
    """Inspect page-level crawl and ingestion results for a crawl job."""
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    job_uuid = uuid.UUID(job_id)

    async with get_session() as session:
        job = await session.get(CrawlJob, job_uuid)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if scope and job.app_id != scope:
            raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")

        stmt = select(CrawlJobPage).where(CrawlJobPage.job_id == job_uuid)
        if crawl_status:
            stmt = stmt.where(CrawlJobPage.crawl_status == crawl_status)
        if ingestion_status:
            stmt = stmt.where(CrawlJobPage.ingestion_status == ingestion_status)
        if section:
            stmt = stmt.where(CrawlJobPage.section == section)

        total = await session.scalar(select(func.count()).select_from(stmt.subquery()))
        rows = await session.execute(
            stmt.order_by(CrawlJobPage.section.asc(), CrawlJobPage.url.asc())
            .offset(offset)
            .limit(limit)
        )

        items = []
        for page in rows.scalars().all():
            items.append(
                {
                    "url": page.url,
                    "section": page.section,
                    "discovered_via": page.discovered_via,
                    "depth": page.depth,
                    "crawl_status": page.crawl_status,
                    "ingestion_status": page.ingestion_status,
                    "http_status": page.http_status,
                    "fetch_ms": page.fetch_ms,
                    "content_chars": page.content_chars,
                    "title": page.title,
                    "chunk_count": page.chunk_count,
                    "error": page.error,
                    "crawled_at": page.crawled_at.isoformat() if page.crawled_at else None,
                    "ingested_at": page.ingested_at.isoformat() if page.ingested_at else None,
                }
            )

        return {
            "job_id": str(job.job_id),
            "app_id": job.app_id,
            "total": total or 0,
            "limit": limit,
            "offset": offset,
            "items": items,
        }
