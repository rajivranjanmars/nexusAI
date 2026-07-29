"""
Semantic Response Cache — Redis + ChromaDB backend.

Stores cached responses in Redis (fast key-value lookup with TTL) and uses a
dedicated ChromaDB collection for semantic similarity matching on question
embeddings.

Two cache scopes are supported:

- ``shared``: reusable across actors inside the same app/workflow. Suitable
  for public website-grounded RAG responses.
- ``actor``: reusable only for the same actor. Suitable for answers derived
  from actor-specific data such as grades or enrollment state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from cache_module.vector_store import _embed
from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_SIMILARITY_THRESHOLD: float = 0.92
_DEFAULT_CACHE_TTL_SECONDS: int = 3600
CacheScope = Literal["shared", "actor"]

# ── ChromaDB collection for cache embeddings ─────────────────────────────────

_cache_collection = None


def _get_cache_collection():
    """Return (or lazily create) the ChromaDB collection used for cache lookups."""
    global _cache_collection
    if _cache_collection is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.PersistentClient(
            path=settings.chroma_db_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _cache_collection = client.get_or_create_collection(
            name="response_cache",
            metadata={"hnsw:space": "cosine"},
        )
    return _cache_collection


# ── Redis helpers ────────────────────────────────────────────────────────────

def _cache_redis_key(cache_id: str) -> str:
    return f"rcache:{cache_id}"


async def _redis_get(key: str) -> Optional[Dict[str, Any]]:
    from db import redis_client
    return await redis_client.get(key)


async def _redis_set(key: str, value: Any, ttl: int) -> None:
    from db import redis_client
    await redis_client.set(key, value, ttl=ttl)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _normalise_scope(scope: Optional[str]) -> CacheScope:
    return "actor" if scope == "actor" else "shared"


def _fingerprint_response(response_text: str) -> str:
    return hashlib.sha256(response_text.encode("utf-8")).hexdigest()


def _build_where_filter(
    app_id: Optional[str],
    workflow: str,
    cache_scope: CacheScope,
    actor_id: Optional[str],
) -> Dict[str, Any]:
    """Build a ChromaDB where-filter for the cache lookup."""
    conditions: List[Dict[str, str]] = [
        {"workflow": workflow},
        {"cache_scope": cache_scope},
    ]

    if app_id:
        conditions.append({"app_id": app_id})
    else:
        conditions.append({"app_id": "__none__"})

    if cache_scope == "actor" and actor_id:
        conditions.append({"actor_id": actor_id})
    else:
        conditions.append({"actor_id": "__none__"})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Public API ───────────────────────────────────────────────────────────────


async def check_cache(
    app_id: Optional[str],
    workflow: str,
    question: str,
    *,
    actor_id: Optional[str] = None,
    cache_scope: CacheScope = "shared",
    threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    question_embedding: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    """Look up a semantically similar cached response.

    Uses ChromaDB for similarity search, Redis for response text storage.

    Args:
        app_id: Application UUID string (filters cache by app).
        workflow: Workflow name to scope the search.
        question: The user's natural-language question.
        actor_id: Actor identifier for scoped cache.
        cache_scope: Cache scope ('shared' or 'actor').
        threshold: Cosine similarity threshold (0.0–1.0).
        question_embedding: Pre-computed embedding to avoid re-embedding.

    Returns:
        Cache hit metadata including the cached response text, or ``None``
        if no valid match is found.
    """
    embedding = question_embedding or await _embed(question)
    resolved_scope = _normalise_scope(cache_scope)

    def _lookup() -> Optional[Dict[str, Any]]:
        collection = _get_cache_collection()

        where_filter = _build_where_filter(app_id, workflow, resolved_scope, actor_id)

        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=1,
                where=where_filter,
            )
        except Exception as exc:
            logger.debug("ChromaDB cache query failed: %s", exc)
            return None

        if not results or not results["ids"] or not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - distance

        if similarity < threshold:
            logger.debug(
                "Cache miss: best similarity %.3f < threshold %.3f",
                similarity, threshold,
            )
            return None

        cache_id = results["ids"][0][0]
        return {"cache_id": cache_id, "similarity": similarity}

    hit_meta = await asyncio.to_thread(_lookup)
    if hit_meta is None:
        return None

    # Fetch the full response from Redis
    redis_key = _cache_redis_key(hit_meta["cache_id"])
    cached_data = await _redis_get(redis_key)
    if cached_data is None:
        # Embedding exists in ChromaDB but Redis entry expired — stale entry
        logger.debug("Cache embedding found but Redis data expired for %s", hit_meta["cache_id"])
        return None

    logger.info(
        "Semantic cache HIT (similarity=%.3f) for workflow '%s'",
        hit_meta["similarity"], workflow,
    )
    return {
        "response_text": cached_data.get("response_text", ""),
        "similarity": hit_meta["similarity"],
        "cache_scope": resolved_scope,
        "fingerprint": cached_data.get("response_fingerprint"),
        "tool_results": cached_data.get("tool_results", {}),
    }


async def store_response(
    app_id: Optional[str],
    workflow: str,
    question: str,
    response_text: str,
    *,
    actor_id: Optional[str] = None,
    cache_scope: CacheScope = "shared",
    tool_results: Optional[Dict[str, Any]] = None,
    ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
    invalidated_by: Optional[List[str]] = None,
    question_embedding: Optional[List[float]] = None,
) -> str:
    """Cache a workflow response with its question embedding.

    Stores the embedding in ChromaDB for similarity lookup and the full
    response payload in Redis with a TTL for automatic expiry.

    Args:
        app_id: Application UUID string.
        workflow: Workflow name.
        question: The original question text.
        response_text: The full response to cache.
        actor_id: Actor identifier for scoped caches.
        cache_scope: 'shared' or 'actor'.
        tool_results: Optional snapshot of tool data used in the response.
        ttl_seconds: Time-to-live in seconds.
        invalidated_by: List of mutation types that should invalidate this
            cache entry (e.g. ``["student_update"]``).
        question_embedding: Pre-computed embedding to avoid re-embedding.

    Returns:
        The UUID of the stored cache entry.
    """
    embedding = question_embedding or await _embed(question)
    cache_id = str(uuid.uuid4())
    resolved_scope = _normalise_scope(cache_scope)
    scoped_actor_id = actor_id if resolved_scope == "actor" else "__none__"
    response_fingerprint = _fingerprint_response(response_text)

    # Store embedding in ChromaDB
    def _store_embedding() -> None:
        collection = _get_cache_collection()
        collection.add(
            ids=[cache_id],
            embeddings=[embedding],
            documents=[question],
            metadatas=[{
                "app_id": app_id or "__none__",
                "workflow": workflow,
                "cache_scope": resolved_scope,
                "actor_id": scoped_actor_id or "__none__",
                "response_fingerprint": response_fingerprint,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }],
        )

    await asyncio.to_thread(_store_embedding)

    # Store response payload in Redis (with TTL for auto-expiry)
    redis_payload = {
        "response_text": response_text,
        "response_fingerprint": response_fingerprint,
        "workflow": workflow,
        "cache_scope": resolved_scope,
        "tool_results": tool_results,
        "invalidated_by": invalidated_by,
        "question": question,
    }
    await _redis_set(_cache_redis_key(cache_id), redis_payload, ttl=ttl_seconds)

    logger.info(
        "Stored cache entry %s for workflow '%s' (ttl=%ds)",
        cache_id, workflow, ttl_seconds,
    )
    return cache_id


async def invalidate_cache(
    app_id: Optional[str] = None,
    workflow: Optional[str] = None,
    mutation_type: Optional[str] = None,
    actor_id: Optional[str] = None,
) -> int:
    """Expire matching cache entries.

    Called after data-mutation tools (e.g. ``update_student``) to ensure
    stale cached responses are not served. Deletes matching embeddings from
    ChromaDB; Redis entries will expire naturally via TTL.

    Args:
        app_id: Restrict invalidation to a specific app.
        workflow: Restrict to a specific workflow.
        mutation_type: Currently unused (kept for API compatibility).
        actor_id: Restrict invalidation to a specific actor.

    Returns:
        Number of cache entries expired.
    """
    def _invalidate() -> int:
        collection = _get_cache_collection()

        conditions: List[Dict[str, str]] = []
        if app_id:
            conditions.append({"app_id": app_id})
        if workflow:
            conditions.append({"workflow": workflow})
        if actor_id:
            conditions.append({"actor_id": actor_id})

        if not conditions:
            return 0

        where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        try:
            results = collection.get(where=where_filter)
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                return len(results["ids"])
        except Exception as exc:
            logger.warning("Cache invalidation failed: %s", exc)

        return 0

    count = await asyncio.to_thread(_invalidate)
    logger.info(
        "Invalidated %d cache entries (app=%s, workflow=%s, mutation=%s)",
        count, app_id, workflow, mutation_type,
    )
    return count


async def get_cache_stats(app_id: Optional[str] = None) -> Dict[str, Any]:
    """Return cache statistics for admin dashboards.

    Args:
        app_id: Restrict stats to a specific app.

    Returns:
        Dict with ``total_entries`` and basic metadata.
    """
    def _stats() -> Dict[str, Any]:
        collection = _get_cache_collection()

        try:
            if app_id:
                results = collection.get(where={"app_id": app_id})
            else:
                results = collection.get()

            total = len(results["ids"]) if results and results["ids"] else 0
        except Exception:
            total = 0

        return {
            "total_entries": total,
            "active_entries": total,  # Redis TTL handles expiry
            "total_hits": 0,  # Hit tracking not implemented in this backend
            "top_cached": [],
        }

    return await asyncio.to_thread(_stats)


async def clear_cache(app_id: Optional[str] = None) -> int:
    """Completely clear the semantic cache for an app or globally.

    Deletes embeddings from ChromaDB and keys from Redis.

    Args:
        app_id: Optional app ID to restrict clearing.

    Returns:
        Number of entries cleared.
    """
    def _get_ids() -> List[str]:
        collection = _get_cache_collection()
        # ChromaDB doesn't support empty where-filter for "all" in some versions,
        # so we fetch without a where if app_id is None.
        if app_id:
            results = collection.get(where={"app_id": app_id})
        else:
            results = collection.get()

        ids = results["ids"] if results and results["ids"] else []
        if ids:
            collection.delete(ids=ids)
        return ids

    cache_ids = await asyncio.to_thread(_get_ids)

    from db import redis_client
    count = 0
    for cid in cache_ids:
        await redis_client.delete(_cache_redis_key(cid))
        count += 1

    logger.info("Manually cleared %d cache entries (app_id=%s)", count, app_id)
    return count
