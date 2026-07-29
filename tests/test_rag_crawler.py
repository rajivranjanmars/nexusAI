"""Integration tests for the current RAG crawl orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from backend_proxy import rag_routes


@dataclass(frozen=True)
class _Page:
    """Minimal crawled page consumed by the ingestion pipeline."""

    url: str
    content: str
    content_hash: str


@dataclass(frozen=True)
class _AuditPage:
    """Minimal per-page audit snapshot emitted by the crawler."""

    url: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the audit snapshot for the pipeline."""
        return {"url": self.url, "crawl_status": "extracted"}


class _Stats:
    """Minimal crawler statistics snapshot."""

    def to_dict(self) -> dict[str, int]:
        """Serialize the counters expected by the job-status update."""
        return {"urls_processed": 2, "urls_failed": 0}


class _FakeCrawler:
    """Deterministic replacement for the network crawler."""

    def __init__(self, **kwargs: Any) -> None:
        """Capture crawler configuration and provide deterministic results."""
        self.kwargs = kwargs
        self.stats = _Stats()
        self.page_audits = {
            "first": _AuditPage("https://example.test/first"),
            "second": _AuditPage("https://example.test/second"),
        }

    async def run(self) -> list[_Page]:
        """Return one new page and one duplicate page."""
        return [
            _Page("https://example.test/first", "new content", "hash-1"),
            _Page("https://example.test/second", "duplicate content", "hash-2"),
        ]


class _FakePipeline:
    """Record crawl orchestration calls without external databases or Chroma."""

    instances: list["_FakePipeline"] = []

    def __init__(self, app_id: str, crawl_job_id: str) -> None:
        """Initialize call recording for one crawl job."""
        self.app_id = app_id
        self.crawl_job_id = crawl_job_id
        self.status_updates: list[tuple[str, dict[str, Any]]] = []
        self.page_updates: list[dict[str, Any]] = []
        self.synced_audits: list[dict[str, Any]] = []
        self.heartbeats = 0
        self.cache_invalidations = 0
        self._ingest_results = iter((1, 0))
        self.instances.append(self)

    async def update_job_status(self, status: str, **kwargs: Any) -> None:
        """Record a job-status transition."""
        self.status_updates.append((status, kwargs))

    async def sync_page_audits(self, audits: list[dict[str, Any]]) -> None:
        """Record crawler audit snapshots."""
        self.synced_audits = audits

    async def ingest_page(self, **kwargs: Any) -> int:
        """Return deterministic inserted and deduplicated chunk counts."""
        return next(self._ingest_results)

    async def update_page_ingestion_status(self, **kwargs: Any) -> None:
        """Record a page ingestion status change."""
        self.page_updates.append(kwargs)

    async def heartbeat(self) -> None:
        """Record crawler progress heartbeats."""
        self.heartbeats += 1

    async def invalidate_shared_response_cache(self) -> None:
        """Record response-cache invalidation after new content is ingested."""
        self.cache_invalidations += 1


@pytest.mark.asyncio
async def test_rag_crawl_orchestration_ingests_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process pages, record audit state, and invalidate cache after ingestion."""
    _FakePipeline.instances.clear()
    monkeypatch.setattr(rag_routes, "AsyncCrawler", _FakeCrawler)
    monkeypatch.setattr(rag_routes, "IngestPipeline", _FakePipeline)

    await rag_routes._run_crawl_background(
        app_id="app-a",
        job_id="job-a",
        seed_url="https://example.test",
        max_depth=1,
        allowed_paths=["/"],
        auto_discover_sitemap=False,
    )

    pipeline = _FakePipeline.instances[0]
    final_status, final_payload = pipeline.status_updates[-1]
    assert pipeline.status_updates[0][0] == "processing"
    assert final_status == "done"
    assert final_payload["pages_found"] == 2
    assert final_payload["pages_skipped"] == 1
    assert final_payload["pages_failed"] == 0
    assert final_payload["chunks_upserted"] == 1
    assert [update["status"] for update in pipeline.page_updates] == [
        "ingested",
        "deduplicated",
    ]
    assert len(pipeline.synced_audits) == 2
    assert pipeline.heartbeats == 2
    assert pipeline.cache_invalidations == 1
