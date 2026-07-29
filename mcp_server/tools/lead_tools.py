"""Lead capture persistence helpers."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Literal

import httpx

from orchestration.ameyo_service import trigger_ameyo_call
from orchestration.lead_capture_inactivity import (
    clear_partial_flush,
    load_lead_snapshot,
    remember_lead_snapshot,
    schedule_partial_flush,
    touch_lead_session_activity as touch_tracked_lead_activity,
)
from orchestration.lpude_lead_service import submit_lpude_lead
from shared.logger import get_logger
from shared.redis_client import get_client

logger = get_logger(__name__)

# To communicate with the lead-chatbot API from the MCP container
LEAD_API_BASE = "http://host.docker.internal:8090/api/internal/lead"

_COMPLETED_LEAD_FIELDS = (
    "name",
    "mobile",
    "email",
    "program_level",
    "program_names",
    "state",
    "city",
    "address",
)
_DISPATCH_TTL_SECONDS = 7 * 24 * 3600


def _normalize_mobile(value: str) -> str:
    """Normalize Indian mobile input to a plain 10-digit number."""

    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    return digits


def _is_valid_mobile(value: str) -> bool:
    """Return whether a normalized mobile number is valid for downstream use."""

    return len(value) == 10


def _is_completed_payload(payload: Dict[str, str]) -> bool:
    """Return whether the payload represents a completed lead."""

    if payload.get("status") != "completed":
        return False
    if not all(payload.get(field_name) for field_name in _COMPLETED_LEAD_FIELDS):
        return False
    return _is_valid_mobile(payload["mobile"])


def _dispatch_key_base(session_id: str, mode: Literal["partial", "completed"]) -> str:
    """Return the Redis key namespace for a lead dispatch mode."""

    return f"lead-capture:{mode}-dispatch:{session_id}"


def _ameyo_key_base(session_id: str) -> str:
    """Return the Redis key namespace for one-time Ameyo dispatch."""

    return f"lead-capture:ameyo-dispatch:{session_id}"


def _split_program_names(value: str | None) -> list[str]:
    """Split a comma-separated program name string into clean names."""

    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _has_lpude_minimum_fields(payload: Dict[str, str]) -> bool:
    """Return whether the lead has the minimum fields LPUDE accepts."""

    return all(payload.get(field) for field in ("name", "email", "mobile")) and _is_valid_mobile(
        payload["mobile"]
    )


def _is_terminal_status(payload: Dict[str, str]) -> bool:
    """Return whether the lead snapshot is already terminal."""

    return payload.get("status") in {"completed", "existing_student"}


def _is_partial_flush_candidate(payload: Dict[str, str]) -> bool:
    """Return whether a partial lead should be flushed after inactivity."""

    if _is_terminal_status(payload):
        return False
    candidate = dict(payload)
    candidate["mobile"] = _normalize_mobile(candidate.get("mobile", ""))
    return _has_lpude_minimum_fields(candidate)


async def _wait_for_ameyo_completion(client: Any, ameyo_key_base: str) -> bool:
    """Wait briefly for an in-flight Ameyo dispatch to finish."""

    for _ in range(20):
        await asyncio.sleep(0.25)
        if await client.get(f"{ameyo_key_base}:completed"):
            return True
    return False


async def touch_lead_session_activity(session_id: str) -> None:
    """Extend the partial flush deadline for an active lead-capture session."""

    await touch_tracked_lead_activity(session_id, _is_partial_flush_candidate)


async def _dispatch_lead(
    session_id: str,
    payload: Dict[str, str],
    mode: Literal["partial", "completed"],
) -> None:
    """Submit lead data to downstream integrations for the requested mode."""

    mobile = payload["mobile"]
    redis_key_base = _dispatch_key_base(session_id or mobile, mode)
    partial_key_base = _dispatch_key_base(session_id or mobile, "partial")
    ameyo_key_base = _ameyo_key_base(session_id or mobile)
    client = await get_client()
    program_names = _split_program_names(payload.get("program_names"))
    lpude_required = mode == "completed" or _has_lpude_minimum_fields(payload)
    try:
        if mode == "completed":
            await clear_partial_flush(session_id)

        partial_lpude_done = await client.get(f"{partial_key_base}:lpude")
        should_send_lpude = (
            lpude_required
            and not await client.get(f"{redis_key_base}:lpude")
            and not (mode == "completed" and partial_lpude_done)
        )
        if should_send_lpude:
            try:
                await submit_lpude_lead(
                    {
                        "name": payload.get("name", ""),
                        "email": payload.get("email", ""),
                        "mobile": mobile,
                        "submission_id": session_id or mobile,
                        "program": payload.get("program_names", ""),
                        "state": payload.get("state", ""),
                        "city": payload.get("city", ""),
                        "address": payload.get("address", ""),
                    }
                )
                await client.set(f"{redis_key_base}:lpude", "1", ex=_DISPATCH_TTL_SECONDS)
            except Exception as exc:
                logger.warning("%s lead dispatch to LPUDE failed for session %s: %s", mode, session_id, exc)

        partial_ameyo_done = await client.get(f"{partial_key_base}:ameyo")
        ameyo_done_once = await client.get(f"{ameyo_key_base}:completed")
        should_send_ameyo = (
            not ameyo_done_once
            and not partial_ameyo_done
            and not await client.get(f"{redis_key_base}:ameyo")
        )
        if should_send_ameyo:
            ameyo_claimed = await client.set(
                f"{ameyo_key_base}:processing",
                "1",
                ex=10 * 60,
                nx=True,
            )
            if ameyo_claimed:
                try:
                    ameyo_payload = {
                        "phone": mobile,
                        "lead_id": session_id or mobile,
                        "program_names": program_names,
                        "request_id": session_id or mobile,
                    }
                    if payload.get("name"):
                        ameyo_payload["customer_name"] = payload["name"]
                    await trigger_ameyo_call(ameyo_payload)
                    await client.set(f"{redis_key_base}:ameyo", "1", ex=_DISPATCH_TTL_SECONDS)
                    await client.set(f"{ameyo_key_base}:completed", "1", ex=_DISPATCH_TTL_SECONDS)
                except Exception as exc:
                    logger.warning("%s lead dispatch to Ameyo failed for session %s: %s", mode, session_id, exc)
                finally:
                    await client.delete(f"{ameyo_key_base}:processing")
            elif mode == "completed":
                await _wait_for_ameyo_completion(client, ameyo_key_base)

        lpude_done = True if not lpude_required else (
            await client.get(f"{redis_key_base}:lpude") or partial_lpude_done
        )
        ameyo_done = await client.get(f"{ameyo_key_base}:completed") or await client.get(
            f"{redis_key_base}:ameyo"
        )
        if mode == "completed":
            ameyo_done = ameyo_done or partial_ameyo_done
        if lpude_done and ameyo_done:
            await client.set(f"{redis_key_base}:completed", "1", ex=_DISPATCH_TTL_SECONDS)
        elif mode == "partial":
            await schedule_partial_flush(session_id or mobile, payload, _is_partial_flush_candidate)
    finally:
        await client.delete(f"{redis_key_base}:processing")


async def _claim_dispatch(
    session_id: str,
    mode: Literal["partial", "completed"],
) -> bool:
    """Claim a dispatch lock for the session and mode."""

    redis_key_base = _dispatch_key_base(session_id, mode)
    client = await get_client()
    if await client.get(f"{redis_key_base}:completed"):
        logger.info("Skipping duplicate %s lead dispatch for session %s", mode, session_id)
        return False
    claimed = await client.set(
        f"{redis_key_base}:processing",
        "1",
        ex=10 * 60,
        nx=True,
    )
    if not claimed:
        logger.info("Skipping in-flight %s lead dispatch for session %s", mode, session_id)
        return False
    return True


async def _should_dispatch_completed_lead_async(session_id: str, payload: Dict[str, str]) -> bool:
    """Return whether this completed payload should trigger downstream dispatch."""

    if not _is_completed_payload(payload):
        return False

    await clear_partial_flush(session_id)
    return await _claim_dispatch(session_id or payload["mobile"], "completed")


async def flush_partial_lead_after_inactivity(session_id: str) -> Dict[str, Any]:
    """Trigger best-effort downstream dispatch for an inactive partial session."""

    snapshot = await load_lead_snapshot(session_id)
    if not snapshot:
        return {"success": True, "dispatched": False, "reason": "no_lead_snapshot"}
    if _is_terminal_status(snapshot):
        return {"success": True, "dispatched": False, "reason": f"status_{snapshot['status']}"}

    mobile = _normalize_mobile(snapshot.get("mobile", ""))
    snapshot["mobile"] = mobile
    if not _has_lpude_minimum_fields(snapshot):
        return {"success": True, "dispatched": False, "reason": "missing_partial_minimum"}

    client = await get_client()
    if await client.get(f"{_dispatch_key_base(session_id or mobile, 'completed')}:completed"):
        return {"success": True, "dispatched": False, "reason": "already_dispatched"}

    if not await _claim_dispatch(session_id or mobile, "partial"):
        return {"success": True, "dispatched": False, "reason": "dispatch_in_flight"}

    asyncio.create_task(_dispatch_lead(session_id, snapshot, "partial"))
    return {"success": True, "dispatched": True, "reason": "partial_dispatch_queued"}


async def upsert_lead(
    session_id: str,
    name: str | None = None,
    mobile: str | None = None,
    email: str | None = None,
    program_level: str | None = None,
    program_names: str | None = None,
    state: str | None = None,
    city: str | None = None,
    address: str | None = None,
    status: str | None = None,
) -> Dict[str, Any]:
    """Upsert lead information to the external lead storage.

    Any provided field will update the lead's profile.
    """
    payload: Dict[str, str] = {}
    if name is not None:
        payload["name"] = name
    if mobile is not None:
        payload["mobile"] = mobile
    if email is not None:
        payload["email"] = email
    if program_level is not None:
        payload["program_level"] = program_level
    if program_names is not None:
        payload["program_names"] = program_names
    if state is not None:
        payload["state"] = state
    if city is not None:
        payload["city"] = city
    if address is not None:
        payload["address"] = address
    if status is not None:
        payload["status"] = status

    if not payload:
        return {"success": True, "message": "No data provided to update."}

    if "mobile" in payload:
        payload["mobile"] = _normalize_mobile(payload["mobile"])

    logger.info("Upserting lead data for session %s: %s", session_id, list(payload.keys()))

    try:
        url = f"{LEAD_API_BASE}/{session_id}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.patch(url, json=payload)
            resp.raise_for_status()
            response_details: Any
            if not resp.content:
                response_details = None
            else:
                try:
                    response_details = resp.json()
                except ValueError:
                    response_details = resp.text

            snapshot = await remember_lead_snapshot(session_id, payload)
            await schedule_partial_flush(session_id, snapshot, _is_partial_flush_candidate)
            if await _should_dispatch_completed_lead_async(session_id, payload):
                asyncio.create_task(_dispatch_lead(session_id, payload, "completed"))
            return {"success": True, "details": response_details}
    except Exception as exc:
        logger.error("Failed to upsert lead: %s", exc)
        return {"success": False, "error": str(exc)}
