"""One-shot lead-capture response flags."""

from __future__ import annotations

from typing import Any

from shared.redis_client import get_client


async def maybe_flag_completion_celebration(
    session_id: str,
    lead_progress: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return lead progress with a one-time celebration hint on completion."""

    progress = dict(lead_progress or {})
    if progress.get("status") != "completed":
        progress.pop("do_celebrations", None)
        return progress

    redis_key = f"lead-capture:celebration:{session_id or 'anonymous'}"
    client = await get_client()
    claimed = await client.set(redis_key, "1", ex=7 * 24 * 3600, nx=True)
    if claimed:
        progress["do_celebrations"] = True
    else:
        progress.pop("do_celebrations", None)
    return progress
