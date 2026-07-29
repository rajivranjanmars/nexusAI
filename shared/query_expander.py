"""
Query expansion helpers for recall-oriented RAG retrieval.
"""

from __future__ import annotations

import hashlib
import json

from llm.llm_client import call_fast
from shared.logger import get_logger
from shared.redis_client import get as redis_get
from shared.redis_client import set as redis_set

logger = get_logger(__name__)

_EXPANSION_CACHE_TTL_SECONDS = 24 * 60 * 60
_EXPANSION_SYSTEM_PROMPT = (
    "You rewrite search queries for retrieval recall. Given one user question, return exactly "
    "3 alternative phrasings that preserve meaning but vary wording. Output only a JSON array "
    "of strings with no markdown or commentary."
)


async def expand_query(query: str, app_id: str) -> list[str]:
    """Return the original query plus up to 3 LLM-generated variants."""

    normalized_query = (query or "").strip()
    if not normalized_query:
        return []

    cache_key = f"rag:qexp:{app_id}:{hashlib.md5(normalized_query.encode('utf-8')).hexdigest()}"
    try:
        cached = await redis_get(cache_key)
        if isinstance(cached, list):
            cached_variants = [
                str(item).strip()
                for item in cached
                if str(item).strip() and str(item).strip() != normalized_query
            ]
            return [normalized_query, *cached_variants[:3]]
    except Exception as exc:
        logger.debug("Query expansion cache read failed for app %s: %s", app_id, exc)

    try:
        response, _usage = await call_fast(
            normalized_query,
            system=_EXPANSION_SYSTEM_PROMPT,
            max_tokens=200,
            app_id=app_id,
            action="query_expansion",
        )
        payload = json.loads(response)
        if not isinstance(payload, list):
            raise ValueError("Expected JSON list from query expansion")

        variants: list[str] = []
        seen = {normalized_query.casefold()}
        for item in payload:
            variant = str(item).strip()
            if not variant or variant.casefold() in seen:
                continue
            seen.add(variant.casefold())
            variants.append(variant)
            if len(variants) >= 3:
                break

        try:
            await redis_set(cache_key, variants, ttl=_EXPANSION_CACHE_TTL_SECONDS)
        except Exception as exc:
            logger.debug("Query expansion cache write failed for app %s: %s", app_id, exc)

        return [normalized_query, *variants]
    except Exception as exc:
        logger.warning("Query expansion failed for app %s: %s", app_id, exc)
        return [normalized_query]
