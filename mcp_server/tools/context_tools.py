"""
MCP Tools — Context operations.

Provides ``get_context`` (fetch session + semantic search) and
``build_context`` (embed & store) as MCP tools, delegating all heavy lifting
to the ``db`` layer.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from shared.redis_client import get, set
from shared.logger import get_logger

logger = get_logger(__name__)


async def get_context(student_id: str, query: str) -> Dict[str, Any]:
    """Fetch merged context from Redis session cache and pgvector.

    Args:
        student_id: The student whose context to retrieve.
        query: Natural-language query for semantic search.

    Returns:
        A dict with ``session`` (cached data) and ``semantic_results``
        (vector-search hits).
    """
    logger.info("get_context called", extra={"student_id": student_id})

    # 1. Check Redis for cached session context
    cache_key = f"session:{student_id}"
    session_data = await get(cache_key)

    # 2. Semantic search in pgvector
    # Note: This is currently a placeholder - implement proper vector search
    # when the vector store is properly integrated
    semantic_results = []  # Placeholder

    context: Dict[str, Any] = {
        "student_id": student_id,
        "session": session_data or {},
        "semantic_results": semantic_results,
    }

    logger.debug(
        "Context assembled: session=%s, vectors=%d",
        bool(session_data),
        len(semantic_results),
    )
    return context


async def build_context(student_id: str, data: Dict[str, Any]) -> bool:
    """Embed and store student context into pgvector + Redis.

    Args:
        student_id: The student identifier.
        data: Arbitrary context dict to persist (profile info, conversation, etc.).

    Returns:
        ``True`` on success.
    """
    logger.info("build_context called", extra={"student_id": student_id})

    # 1. Cache in Redis (1-hour TTL)
    cache_key = f"session:{student_id}"
    await set(cache_key, data, ttl=3600)

    # 2. Embed key textual content and store in pgvector
    # Note: This is currently a placeholder - implement proper vector storage
    # when the vector store is properly integrated
    text_blob = json.dumps(data, default=str)
    # Placeholder for vector storage implementation
    
    logger.info("Context built and stored for student %s", student_id)
    return True
