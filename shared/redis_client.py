"""
Shared Redis client for the entire application.

Provides a singleton Redis client with common operations like get, set, delete,
and specialized operations like rate limiting. Used by both the backend proxy
and database layers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import redis.asyncio as aioredis

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

# ── Client singleton ────────────────────────────────────────────────────────

_client: Optional[aioredis.Redis] = None


async def get_client() -> aioredis.Redis:
    """Return (or lazily create) the shared async Redis client."""
    global _client
    if _client is None:
        _client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
        logger.info("Shared Redis client initialised", extra={"url": settings.redis_url})
    return _client


# ── Basic operations ────────────────────────────────────────────────────────


async def get(key: str) -> Optional[Any]:
    """Get a value from Redis, deserialising JSON automatically.

    Args:
        key: The cache key.

    Returns:
        The deserialised value, or ``None`` if the key does not exist.
    """
    client = await get_client()
    raw = await client.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


async def set(key: str, value: Any, ttl: int = 3600) -> bool:
    """Store a value in Redis, serialising to JSON.

    Args:
        key: The cache key.
        value: Any JSON-serialisable value.
        ttl: Expiry in seconds (default 1 hour).

    Returns:
        ``True`` if the key was set successfully.
    """
    client = await get_client()
    serialised = json.dumps(value) if not isinstance(value, str) else value
    result = await client.set(key, serialised, ex=ttl)
    logger.debug("Redis SET %s (ttl=%ds)", key, ttl)
    return bool(result)


async def delete(key: str) -> bool:
    """Delete a key from Redis.

    Args:
        key: The cache key to remove.

    Returns:
        ``True`` if the key existed and was deleted.
    """
    client = await get_client()
    result = await client.delete(key)
    logger.debug("Redis DEL %s (deleted=%s)", key, bool(result))
    return bool(result)


# ── Rate limiting operations ────────────────────────────────────────────────


@dataclass(frozen=True)
class RateLimitState:
    """Metadata describing the current rate limit window."""

    limit: int
    remaining: int
    retry_after: int


class RateLimitExceeded(Exception):
    """Raised when a caller exceeds the configured rate limit."""

    def __init__(self, message: str, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after


async def enforce_rate_limit(
    subject: str,
    scope: str,
    rate_limit_override: int | None = None,
) -> RateLimitState:
    """Apply a fixed-window rate limit for a subject and scope pair.

    Args:
        subject: Logical caller identifier, typically student ID or IP address.
        scope: Endpoint or action scope.
        rate_limit_override: Optional per-app rate limit from the App Registry.
            Falls back to the global setting when ``None``.

    Returns:
        Current window metadata for response headers.

    Raises:
        RateLimitExceeded: If the request exceeds the configured threshold.
    """
    client = await get_client()
    limit = rate_limit_override or settings.proxy_rate_limit_requests
    window_s = settings.proxy_rate_limit_window_seconds
    now = int(time.time())
    bucket = now // window_s
    retry_after = window_s - (now % window_s)
    redis_key = f"proxy-rate-limit:{scope}:{subject}:{bucket}"

    current = await client.incr(redis_key)
    if current == 1:
        await client.expire(redis_key, window_s)

    remaining = max(limit - current, 0)
    if current > limit:
        logger.warning(
            "Rate limit exceeded",
            extra={
                "scope": scope,
                "subject": subject,
                "retry_after": retry_after,
            },
        )
        raise RateLimitExceeded("Rate limit exceeded", retry_after=retry_after)

    return RateLimitState(
        limit=limit,
        remaining=remaining,
        retry_after=retry_after,
    )


# ── Legacy compatibility ────────────────────────────────────────────────────


class _RedisClient:
    """Legacy compatibility wrapper for existing imports."""
    
    async def get(self, key: str) -> Optional[Any]:
        return await get(key)

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        return await set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        return await delete(key)


# Singleton used by legacy imports
redis_client = _RedisClient()


async def close() -> None:
    """Close the Redis connection pool."""
    global _client
    if _client is not None:
        await _client.close()  # type: ignore[union-attr]
        _client = None
        logger.info("Shared Redis connection pool closed")
