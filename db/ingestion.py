"""
Ingestion Pipeline for RAG chunks.

Handles chunking via tiktoken, embedding using the central `cache_module.vector_store`,
deduplication via ChromaDB metadata, and persisting chunks to ChromaDB.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import Any, List, Literal, Optional
from urllib.parse import urlparse

import tiktoken
from sqlalchemy import select

from cache_module.semantic_cache import invalidate_cache
from cache_module.vector_store import _embed
from db.models.knowledge import CrawlJob, CrawlJobPage
from db.postgres import get_session
from shared.bm25_index import invalidate_index
from shared.chroma_client import get_collection, get_parent_collection
from shared.logger import get_logger

logger = get_logger(__name__)

PARENT_MAX_TOKENS = 512
PARENT_OVERLAP = 64
CHILD_MAX_TOKENS = 128
CHILD_OVERLAP = 16

_HEADING_RE = re.compile(
    r"(?:^|\n)(?:#{1,3}\s+(.+)|<h[1-3][^>]*>([^<]+)</h[1-3]>)",
    re.IGNORECASE,
)
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_ALPHANUMERIC_CODE_RE = re.compile(r"\b([A-Z]{2,}\d+|[A-Z]+-\d+|\d+[A-Z]+)\b")


def _extract_section_heading(text: str, preceding_text: str) -> Optional[str]:
    """Infer a section heading near a chunk boundary."""

    if not text.strip():
        return None

    matches = list(_HEADING_RE.finditer(preceding_text[-600:]))
    if not matches:
        return None

    last_match = matches[-1]
    heading = (last_match.group(1) or last_match.group(2) or "").strip()
    return heading[:120] or None


def _extract_entity_tags(text: str) -> list[str]:
    """Extract lightweight tags for better downstream retrieval diagnostics."""

    tags: list[str] = []
    seen: set[str] = set()
    for match in _PROPER_NOUN_RE.finditer(text):
        value = match.group(1).strip()
        if value and value not in seen:
            seen.add(value)
            tags.append(value)
        if len(tags) >= 10:
            return tags

    for match in _ALPHANUMERIC_CODE_RE.finditer(text):
        value = match.group(1).strip()
        if value and value not in seen:
            seen.add(value)
            tags.append(value)
        if len(tags) >= 10:
            break

    return tags


class IngestPipeline:
    def __init__(self, app_id: str, crawl_job_id: Optional[str] = None):
        self.app_id = app_id
        self.crawl_job_id = crawl_job_id
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = CHILD_MAX_TOKENS
        self.overlap = CHILD_OVERLAP
        self.batch_size = 32

    async def invalidate_shared_response_cache(self) -> int:
        """Expire shared response-cache rows for this app after RAG changes.

        This is called once per ingest operation or completed crawl job,
        rather than once per page, to avoid unnecessary cache churn under load.
        """
        try:
            invalidated = await invalidate_cache(
                app_id=self.app_id,
                mutation_type="rag_reingest",
            )
            invalidate_index(self.app_id)
            logger.info(
                "Invalidated %d shared response-cache rows and BM25 cache for app %s after RAG ingest",
                invalidated,
                self.app_id,
            )
            return invalidated
        except Exception as exc:
            logger.warning(
                "Shared response-cache invalidation failed for app %s: %s",
                self.app_id,
                exc,
            )
            return 0

    def _token_windows(self, tokens: list[int], max_tokens: int, overlap: int) -> list[tuple[int, int]]:
        """Split token IDs into overlapping windows and return start/end offsets."""

        if not tokens:
            return []

        windows: list[tuple[int, int]] = []
        start = 0
        step = max(max_tokens - overlap, 1)
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            windows.append((start, end))
            if end >= len(tokens):
                break
            start += step
        return windows

    def _build_chunk_payload(
        self,
        *,
        chunk_id: str,
        content: str,
        source_type: str,
        source_ref: str,
        source_url: str,
        chunk_type: str,
        chunk_index: int,
        content_hash: str,
        ingested_at: str,
        section_heading: Optional[str],
        entity_tags: list[str],
        parent_chunk_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build a Chroma-compatible payload for one chunk."""

        return {
            "id": chunk_id,
            "content": content,
            "metadata": {
                "app_id": self.app_id,
                "source_type": source_type,
                "source_ref": source_ref,
                "source_url": source_url,
                "chunk_index": chunk_index,
                "chunk_type": chunk_type,
                "parent_chunk_id": parent_chunk_id or "",
                "section_heading": section_heading or "",
                "entity_tags": json.dumps(entity_tags),
                "content_hash": content_hash,
                "ingested_at": ingested_at,
                "crawl_job_id": self.crawl_job_id or "",
            },
        }

    def chunk_text(self, text: str) -> List[str]:
        """Return retrieval-compatible child chunks for legacy callers."""

        _, child_chunks = self._chunk_hierarchical(
            content=text,
            source_ref="legacy",
            source_type="structured",
            source_url="",
            content_hash="",
        )
        return [chunk["content"] for chunk in child_chunks]

    def _chunk_hierarchical(
        self,
        *,
        content: str,
        source_ref: str,
        source_type: str,
        source_url: str,
        content_hash: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build parent and child chunk payloads for one source document."""

        tokens = self.tokenizer.encode(content)
        if not tokens:
            return [], []

        parent_windows = self._token_windows(tokens, PARENT_MAX_TOKENS, PARENT_OVERLAP)
        child_windows = self._token_windows(tokens, CHILD_MAX_TOKENS, CHILD_OVERLAP)

        parent_chunks: list[dict[str, Any]] = []
        child_chunks: list[dict[str, Any]] = []
        ingested_at = datetime.utcnow().isoformat()

        parent_ranges: list[tuple[int, int, str]] = []
        for index, (start, end) in enumerate(parent_windows):
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()
            if not chunk_text:
                continue

            chunk_id = str(uuid.uuid4())
            parent_ranges.append((start, end, chunk_id))
            char_start = len(self.tokenizer.decode(tokens[:start]))
            section_heading = _extract_section_heading(chunk_text, content[:char_start])
            entity_tags = _extract_entity_tags(chunk_text)
            parent_chunks.append(
                self._build_chunk_payload(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    source_type=source_type,
                    source_ref=source_ref,
                    source_url=source_url,
                    chunk_type="parent",
                    chunk_index=index,
                    content_hash=content_hash,
                    ingested_at=ingested_at,
                    section_heading=section_heading,
                    entity_tags=entity_tags,
                )
            )

        for index, (start, end) in enumerate(child_windows):
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()
            if not chunk_text:
                continue

            midpoint = start + ((end - start) // 2)
            parent_chunk_id = ""
            for parent_start, parent_end, candidate_id in parent_ranges:
                if parent_start <= midpoint < parent_end:
                    parent_chunk_id = candidate_id
                    break
            if not parent_chunk_id and parent_ranges:
                parent_chunk_id = parent_ranges[-1][2]

            char_start = len(self.tokenizer.decode(tokens[:start]))
            section_heading = _extract_section_heading(chunk_text, content[:char_start])
            entity_tags = _extract_entity_tags(chunk_text)
            child_chunks.append(
                self._build_chunk_payload(
                    chunk_id=str(uuid.uuid4()),
                    content=chunk_text,
                    source_type=source_type,
                    source_ref=source_ref,
                    source_url=source_url,
                    chunk_type="child",
                    chunk_index=index,
                    content_hash=content_hash,
                    ingested_at=ingested_at,
                    section_heading=section_heading,
                    entity_tags=entity_tags,
                    parent_chunk_id=parent_chunk_id,
                )
            )

        return parent_chunks, child_chunks

    async def ingest_page(
        self,
        source_type: Literal["crawl", "file", "structured", "url"],
        source_ref: str,
        source_url: str,
        content: str,
        content_hash: str,
    ) -> int:
        """
        Process a single document: deduplicate, chunk, embed, and store in ChromaDB.
        Returns the number of chunks upserted.
        """
        collection = await asyncio.to_thread(get_collection)
        parent_collection = await asyncio.to_thread(get_parent_collection)
        
        # Deduplication check
        # We query ChromaDB for existing chunks with this source_ref and app_id
        existing_result = await asyncio.to_thread(
            collection.get,
            where={"$and": [{"app_id": self.app_id}, {"source_ref": source_ref}]},
        )

        existing_parent_result = await asyncio.to_thread(
            parent_collection.get,
            where={"$and": [{"app_id": self.app_id}, {"source_ref": source_ref}]},
        )

        if existing_result and existing_result["ids"]:
            # Check if content_hash matches
            # The hash is identical for all chunks of the same source
            first_meta = existing_result["metadatas"][0]
            if first_meta.get("content_hash") == content_hash:
                logger.debug("Skipping ingestion for %s (hash match)", source_ref)
                return 0  # No change

            # Content differs, so delete old chunks
            await asyncio.to_thread(
                collection.delete,
                ids=existing_result["ids"],
            )
            logger.info("Deleted %d old chunks for %s", len(existing_result["ids"]), source_ref)

        if existing_parent_result and existing_parent_result["ids"]:
            await asyncio.to_thread(
                parent_collection.delete,
                ids=existing_parent_result["ids"],
            )
            logger.info(
                "Deleted %d old parent chunks for %s",
                len(existing_parent_result["ids"]),
                source_ref,
            )

        parent_chunks, child_chunks = self._chunk_hierarchical(
            content=content,
            source_ref=source_ref,
            source_type=source_type,
            source_url=source_url,
            content_hash=content_hash,
        )
        if not child_chunks:
            return 0

        async def _embed_payloads(chunks: list[dict[str, Any]]) -> list[list[float]]:
            embeddings: list[list[float]] = []
            for index in range(0, len(chunks), self.batch_size):
                batch = chunks[index : index + self.batch_size]
                batch_embeddings = await asyncio.gather(*(_embed(item["content"]) for item in batch))
                embeddings.extend(batch_embeddings)
            return embeddings

        parent_embeddings = await _embed_payloads(parent_chunks)
        child_embeddings = await _embed_payloads(child_chunks)

        if parent_chunks:
            await asyncio.to_thread(
                parent_collection.add,
                documents=[chunk["content"] for chunk in parent_chunks],
                embeddings=parent_embeddings,
                metadatas=[chunk["metadata"] for chunk in parent_chunks],
                ids=[chunk["id"] for chunk in parent_chunks],
            )

        await asyncio.to_thread(
            collection.add,
            documents=[chunk["content"] for chunk in child_chunks],
            embeddings=child_embeddings,
            metadatas=[chunk["metadata"] for chunk in child_chunks],
            ids=[chunk["id"] for chunk in child_chunks],
        )
        invalidate_index(self.app_id)

        return len(child_chunks)

    async def sync_page_audits(self, audits: List[dict[str, Any]]) -> None:
        """Persist per-URL crawl audit rows for the current job."""
        if not self.crawl_job_id or not audits:
            return

        job_uuid = uuid.UUID(self.crawl_job_id)
        async with get_session() as session:
            result = await session.execute(
                select(CrawlJobPage).where(CrawlJobPage.job_id == job_uuid)
            )
            existing_rows = result.scalars().all()
            by_url = {row.url: row for row in existing_rows}

            for audit in audits:
                url = audit["url"]
                row = by_url.get(url)
                if row is None:
                    row = CrawlJobPage(job_id=job_uuid, url=url)
                    session.add(row)
                    by_url[url] = row

                row.section = audit.get("section")
                row.discovered_via = audit.get("discovered_via") or row.discovered_via
                row.depth = audit.get("depth")
                row.crawl_status = audit.get("crawl_status") or row.crawl_status
                row.ingestion_status = audit.get("ingestion_status") or row.ingestion_status
                row.http_status = audit.get("http_status")
                row.fetch_ms = audit.get("fetch_ms")
                row.content_chars = audit.get("content_chars")
                row.title = audit.get("title")
                row.content_hash = audit.get("content_hash")
                row.chunk_count = audit.get("chunk_count", row.chunk_count or 0)
                row.error = audit.get("error")
                crawled_at = audit.get("crawled_at")
                ingested_at = audit.get("ingested_at")
                row.crawled_at = datetime.fromisoformat(crawled_at) if crawled_at else row.crawled_at
                row.ingested_at = datetime.fromisoformat(ingested_at) if ingested_at else row.ingested_at

    async def update_page_ingestion_status(
        self,
        *,
        source_ref: str,
        status: str,
        chunk_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Update ingestion outcome for a single audited URL."""
        if not self.crawl_job_id:
            return

        job_uuid = uuid.UUID(self.crawl_job_id)
        async with get_session() as session:
            result = await session.execute(
                select(CrawlJobPage).where(
                    CrawlJobPage.job_id == job_uuid,
                    CrawlJobPage.url == source_ref,
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                parsed = urlparse(source_ref)
                segments = [segment for segment in parsed.path.split("/") if segment]
                row = CrawlJobPage(
                    job_id=job_uuid,
                    url=source_ref,
                    section=segments[0] if segments else "/",
                )
                session.add(row)
            row.ingestion_status = status
            row.chunk_count = chunk_count
            row.error = error
            row.ingested_at = datetime.utcnow()

    async def update_job_status(
        self,
        status: str,
        pages_found: int = 0,
        pages_skipped: int = 0,
        pages_failed: int = 0,
        chunks_upserted: int = 0,
        error: Optional[str] = None,
        url_stats: Optional[dict] = None,
    ):
        """Update the tracking row in PostgreSQL.

        Args:
            status: New job status.
            pages_found: Number of pages found this update.
            pages_skipped: Number of unchanged pages (hash match).
            pages_failed: Number of pages that failed ingestion.
            chunks_upserted: Number of new chunks stored.
            error: Optional error message.
            url_stats: Optional CrawlStats.to_dict() snapshot.
        """
        if not self.crawl_job_id:
            return

        now = datetime.utcnow()
        async with get_session() as session:
            job = await session.get(CrawlJob, uuid.UUID(self.crawl_job_id))
            if job:
                job.status = status
                job.last_activity_at = now
                if status in ["done", "failed"]:
                    job.finished_at = now
                job.pages_found += pages_found
                job.pages_skipped += pages_skipped
                job.pages_failed += pages_failed
                job.chunks_upserted += chunks_upserted
                if error:
                    job.error = error
                # URL-level counters from the crawler
                if url_stats:
                    job.urls_queued = url_stats.get("urls_queued", job.urls_queued)
                    job.urls_processed = url_stats.get("urls_processed", job.urls_processed)
                    job.urls_skipped_quality = url_stats.get("urls_skipped_quality", job.urls_skipped_quality)
                    job.urls_skipped_duplicate = url_stats.get("urls_skipped_duplicate", job.urls_skipped_duplicate)
                    job.urls_failed = url_stats.get("urls_failed", job.urls_failed)
                    job.pages_per_minute = url_stats.get("pages_per_minute")
                    job.expected_urls = url_stats.get("expected_urls", job.expected_urls)
                    job.expected_urls_from_sitemap = url_stats.get(
                        "expected_urls_from_sitemap", job.expected_urls_from_sitemap
                    )
                    job.coverage_pct = url_stats.get("coverage_pct", job.coverage_pct)
                session.add(job)
                await session.commit()

    async def heartbeat(self) -> None:
        """Touch last_activity_at without changing any counters.

        Called periodically during long ingestion runs so the stuck-detection
        logic in the status endpoint can see the job is still alive.
        """
        if not self.crawl_job_id:
            return
        async with get_session() as session:
            job = await session.get(CrawlJob, uuid.UUID(self.crawl_job_id))
            if job:
                job.last_activity_at = datetime.utcnow()
                session.add(job)
                await session.commit()
