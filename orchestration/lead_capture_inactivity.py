"""Redis-backed inactivity tracking for lead-capture sessions."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from shared.logger import get_logger
from shared.redis_client import get_client

logger = get_logger(__name__)

LEAD_SNAPSHOT_TTL_SECONDS = 7 * 24 * 3600
PARTIAL_LEAD_INACTIVITY_SECONDS = 150
PARTIAL_LEAD_SCAN_INTERVAL_SECONDS = 60

_PARTIAL_LEAD_DUE_SET = "lead-capture:partial-flush-due"

CandidatePredicate = Callable[[dict[str, str]], bool]
FlushSession = Callable[[str], Awaitable[dict[str, Any]]]


def lead_snapshot_key(session_id: str) -> str:
    """Return the Redis key for a lead-capture session snapshot."""

    return f"lead-capture:snapshot:{session_id}"


async def load_lead_snapshot(session_id: str) -> dict[str, str]:
    """Load the accumulated lead snapshot for a session."""

    if not session_id:
        return {}

    client = await get_client()
    raw_snapshot = await client.get(lead_snapshot_key(session_id))
    if not raw_snapshot:
        return {}

    try:
        parsed = json.loads(raw_snapshot) if isinstance(raw_snapshot, str) else raw_snapshot
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid lead snapshot for session %s", session_id)
        return {}

    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(value) for key, value in parsed.items() if value is not None}


async def remember_lead_snapshot(session_id: str, payload: dict[str, str]) -> dict[str, str]:
    """Merge the latest lead fields into the session snapshot and return it."""

    if not session_id:
        return {}
    if not payload:
        return await load_lead_snapshot(session_id)

    snapshot = await load_lead_snapshot(session_id)
    snapshot.update({key: str(value) for key, value in payload.items() if value is not None})
    client = await get_client()
    await client.set(
        lead_snapshot_key(session_id),
        json.dumps(snapshot),
        ex=LEAD_SNAPSHOT_TTL_SECONDS,
    )
    return snapshot


async def schedule_partial_flush(
    session_id: str,
    snapshot: dict[str, str],
    is_candidate: CandidatePredicate,
) -> None:
    """Schedule or clear an inactivity flush for a partial lead snapshot."""

    if not session_id:
        return

    client = await get_client()
    if is_candidate(snapshot):
        due_at = int(time.time()) + PARTIAL_LEAD_INACTIVITY_SECONDS
        await client.zadd(_PARTIAL_LEAD_DUE_SET, {session_id: due_at})
        return

    await client.zrem(_PARTIAL_LEAD_DUE_SET, session_id)


async def clear_partial_flush(session_id: str) -> None:
    """Remove a session from the partial lead inactivity schedule."""

    if not session_id:
        return

    client = await get_client()
    await client.zrem(_PARTIAL_LEAD_DUE_SET, session_id)


async def touch_lead_session_activity(
    session_id: str,
    is_candidate: CandidatePredicate,
) -> None:
    """Extend the partial flush deadline for an active lead-capture session."""

    if not session_id:
        return

    snapshot = await load_lead_snapshot(session_id)
    if snapshot:
        await schedule_partial_flush(session_id, snapshot, is_candidate)


async def flush_inactive_partial_leads(
    flush_session: FlushSession,
    limit: int = 50,
) -> dict[str, Any]:
    """Flush partial lead sessions whose inactivity deadline has passed."""

    client = await get_client()
    due_session_ids = await client.zrangebyscore(
        _PARTIAL_LEAD_DUE_SET,
        min=0,
        max=int(time.time()),
        start=0,
        num=limit,
    )

    dispatched = 0
    skipped = 0
    for session_id in due_session_ids:
        session_key = str(session_id)
        await client.zrem(_PARTIAL_LEAD_DUE_SET, session_key)
        result = await flush_session(session_key)
        if result.get("dispatched"):
            dispatched += 1
        else:
            skipped += 1

    return {
        "success": True,
        "scanned": len(due_session_ids),
        "dispatched": dispatched,
        "skipped": skipped,
    }


async def partial_lead_inactivity_loop(flush_session: FlushSession) -> None:
    """Continuously flush inactive partial lead sessions."""

    while True:
        try:
            await flush_inactive_partial_leads(flush_session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Inactive partial lead flush loop failed: %s", exc)
        await asyncio.sleep(PARTIAL_LEAD_SCAN_INTERVAL_SECONDS)
