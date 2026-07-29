"""
Admin Observability API Endpoints.

Surfaces token usage analytics, RAG knowledge store stats, LLM/cache
performance metrics, and cost configuration.  Accepts both old app-token
admin and new admin dashboard JWT via ``get_admin_scope``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query

from backend_proxy.auth import AuthenticatedUser
from backend_proxy.admin_auth import AdminUserContext
from backend_proxy.deps import get_admin_scope, require_app_admin, require_super_admin
from backend_proxy.schemas import (
    CacheStatsResponse,
    CostMapResponse,
    LlmHealthResponse,
    RagAppStatsItem,
    RagStatsResponse,
    TokenUsageByAppResponse,
    TokenUsageRecentResponse,
    TokenUsageSummaryResponse,
)
from shared.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/admin/observability", tags=["observability"])


# ── Token Usage Endpoints ───────────────────────────────────────────────────


@router.get("/tokens/summary", response_model=TokenUsageSummaryResponse)
async def tokens_summary(
    period: str = Query("30d", regex="^(today|7d|30d|all)$"),
    app_id: Optional[str] = None,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
) -> TokenUsageSummaryResponse:
    """Aggregated token usage: total tokens, cost, by model, by action."""
    from db.token_tracker import get_usage_summary

    effective_app_id = scope or app_id  # ponytail: app_admin auto-scoped
    data = await get_usage_summary(period=period, app_id=effective_app_id)
    return TokenUsageSummaryResponse(**data)


@router.get("/tokens/by-app/{app_id}", response_model=TokenUsageByAppResponse)
async def tokens_by_app(
    app_id: str,
    period: str = Query("30d", regex="^(today|7d|30d|all)$"),
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
) -> TokenUsageByAppResponse:
    """Daily token usage breakdown for a specific app."""
    from db.token_tracker import get_usage_by_app

    # ponytail: app_admin can only query their own app
    if scope and scope != app_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")
    data = await get_usage_by_app(app_id=app_id, period=period)
    return TokenUsageByAppResponse(**data)


@router.get("/tokens/recent", response_model=TokenUsageRecentResponse)
async def tokens_recent(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    app_id: Optional[str] = None,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
) -> TokenUsageRecentResponse:
    """Paginated list of recent individual LLM invocations."""
    from db.token_tracker import get_recent_usage

    effective_app_id = scope or app_id
    data = await get_recent_usage(limit=limit, offset=offset, app_id=effective_app_id)
    return TokenUsageRecentResponse(**data)


# ── RAG Knowledge Store Endpoints ───────────────────────────────────────────


@router.get("/rag/stats", response_model=RagStatsResponse)
async def rag_stats(
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
) -> RagStatsResponse:
    """Global RAG stats: total chunks, chunks per app, collection metadata."""
    from db.ingestion import get_collection

    collection = await asyncio.to_thread(get_collection)
    total_chunks = await asyncio.to_thread(collection.count)

    all_data = await asyncio.to_thread(
        collection.get,
        include=["metadatas"],
    )

    app_stats: dict[str, dict] = {}
    for meta in all_data.get("metadatas") or []:
        aid = meta.get("app_id", "unknown")
        # ponytail: filter by app_id for app_admin
        if scope and aid != scope:
            continue
        if aid not in app_stats:
            app_stats[aid] = {"chunk_count": 0, "source_types": {}, "sources": set()}
        app_stats[aid]["chunk_count"] += 1
        source_type = meta.get("source_type", "unknown")
        app_stats[aid]["source_types"][source_type] = (
            app_stats[aid]["source_types"].get(source_type, 0) + 1
        )
        source_ref = meta.get("source_ref")
        if source_ref:
            app_stats[aid]["sources"].add(source_ref)

    apps = [
        RagAppStatsItem(
            app_id=aid,
            chunk_count=stats["chunk_count"],
            source_types=stats["source_types"],
            source_count=len(stats["sources"]),
        )
        for aid, stats in sorted(app_stats.items(), key=lambda x: x[1]["chunk_count"], reverse=True)
    ]

    return RagStatsResponse(
        total_chunks=total_chunks,
        total_apps=len(app_stats),
        apps=apps,
    )


@router.get("/rag/stats/{app_id}", response_model=RagStatsResponse)
async def rag_stats_by_app(
    app_id: str,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
) -> RagStatsResponse:
    """Per-app RAG stats: chunk count, source breakdown."""
    from fastapi import HTTPException
    from db.ingestion import get_collection

    if scope and scope != app_id:
        raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")

    collection = await asyncio.to_thread(get_collection)

    app_data = await asyncio.to_thread(
        collection.get,
        where={"app_id": app_id},
        include=["metadatas"],
    )

    metadatas = app_data.get("metadatas") or []
    chunk_count = len(metadatas)

    source_types: dict[str, int] = {}
    sources: set[str] = set()
    for meta in metadatas:
        st = meta.get("source_type", "unknown")
        source_types[st] = source_types.get(st, 0) + 1
        source_ref = meta.get("source_ref")
        if source_ref:
            sources.add(source_ref)

    return RagStatsResponse(
        total_chunks=chunk_count,
        total_apps=1 if chunk_count > 0 else 0,
        apps=[
            RagAppStatsItem(
                app_id=app_id,
                chunk_count=chunk_count,
                source_types=source_types,
                source_count=len(sources),
            )
        ] if chunk_count > 0 else [],
    )


# ── LLM & Cache Endpoints ──────────────────────────────────────────────────


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(
    app_id: Optional[str] = None,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
) -> CacheStatsResponse:
    """Semantic response cache statistics (global or per-app)."""
    from cache_module.semantic_cache import get_cache_stats

    effective_app_id = scope or app_id
    data = await get_cache_stats(app_id=effective_app_id)
    return CacheStatsResponse(**data)


@router.get("/cache/stats/{app_id}", response_model=CacheStatsResponse)
async def cache_stats_by_app(
    app_id: str,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
) -> CacheStatsResponse:
    """Per-app semantic response cache statistics."""
    from fastapi import HTTPException
    from cache_module.semantic_cache import get_cache_stats

    if scope and scope != app_id:
        raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")
    data = await get_cache_stats(app_id=app_id)
    return CacheStatsResponse(**data)


@router.post("/cache/clear")
async def cache_clear(
    app_id: Optional[str] = None,
    _admin: AdminUserContext = Depends(require_app_admin),
    scope: str | None = Depends(get_admin_scope),
) -> dict:
    """Manually clear the semantic response cache (global or per-app)."""
    from cache_module.semantic_cache import clear_cache

    if scope and app_id and scope != app_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")
    effective_app_id = scope or app_id
    count = await clear_cache(app_id=effective_app_id)
    return {"status": "ok", "cleared_count": count}


@router.get("/llm/health", response_model=LlmHealthResponse)
async def llm_health(
    _admin: AdminUserContext = Depends(require_app_admin),
) -> LlmHealthResponse:
    """LLM provider health check: send a trivial completion and report latency."""
    from shared.config import settings

    try:
        import openai

        client = openai.AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=15,
        )

        t0 = time.monotonic()
        response = await client.chat.completions.create(
            model=settings.llm_fast_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        return LlmHealthResponse(
            status="ok",
            model=settings.llm_fast_model,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return LlmHealthResponse(
            status="error",
            model=settings.llm_fast_model,
            latency_ms=0,
            error=str(exc),
        )


@router.get("/llm/cost-map", response_model=CostMapResponse)
async def llm_cost_map(
    _admin: AdminUserContext = Depends(require_app_admin),
) -> CostMapResponse:
    """Return the active LLM cost configuration (from env or defaults)."""
    from llm.cost_estimator import get_cost_map

    return CostMapResponse(cost_map=get_cost_map())
