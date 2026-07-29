"""Ameyo click-to-call integration."""

from __future__ import annotations

import json
import os
from typing import Dict, TypedDict

import httpx


class AmeyoCallPayload(TypedDict, total=False):
    """Click-to-call payload for completed lead submissions."""

    phone: str
    user_id: str
    customer_name: str
    lead_id: str
    program_codes: list[str]
    program_names: list[str]
    source: str
    request_id: str


class AmeyoCallResponse(TypedDict):
    """Normalized Ameyo response payload."""

    ameyo_call_ref: str


def _required_config(name: str) -> str:
    """Return a required Ameyo config value or raise a clear error."""

    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is not configured")
    return value


def _optional_config(name: str) -> str:
    """Return an optional Ameyo config value."""

    return os.getenv(name, "").strip()


def _parse_additional_params(raw_value: str) -> Dict[str, str]:
    """Parse configured Ameyo additional parameters."""

    if not raw_value:
        return {}

    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("AMEYO_ADDITIONAL_PARAMS_JSON must be a JSON object")

    normalized: Dict[str, str] = {}
    for key, value in parsed.items():
        if value is not None:
            normalized[str(key)] = str(value)
    return normalized


def _pick_call_ref(response_body: dict | None, fallback_request_id: str) -> str:
    """Pick the most useful call reference from an Ameyo response."""

    if not response_body:
        return fallback_request_id

    ref_candidate = (
        response_body.get("ameyoCallRef")
        or response_body.get("callRef")
        or response_body.get("requestId")
        or response_body.get("callId")
        or response_body.get("call_id")
    )
    if isinstance(ref_candidate, str) and ref_candidate.strip():
        return ref_candidate.strip()
    return fallback_request_id


async def trigger_ameyo_call(payload: AmeyoCallPayload) -> AmeyoCallResponse:
    """Trigger a best-effort Ameyo click-to-call request."""

    base_url = _required_config("AMEYO_BASE_URL")
    campaign_id = _required_config("AMEYO_CAMPAIGN_ID")
    ameyo_user_id = payload.get("user_id") or _required_config("AMEYO_USER_ID")
    password = _required_config("AMEYO_PASSWORD")
    terminal = _required_config("AMEYO_TERMINAL")
    request_id = (
        payload.get("request_id")
        or payload.get("lead_id")
        or f"nexus-{ameyo_user_id}"
    )

    additional_params = _parse_additional_params(
        _optional_config("AMEYO_ADDITIONAL_PARAMS_JSON")
    )
    if payload.get("customer_name"):
        additional_params["customerName"] = payload["customer_name"]
    if payload.get("lead_id"):
        additional_params["leadId"] = payload["lead_id"]
    program_codes = payload.get("program_codes", [])
    if program_codes:
        additional_params["programCodes"] = ",".join(program_codes)
    if payload.get("program_names"):
        additional_params["programNames"] = ",".join(payload["program_names"])
    additional_params["source"] = payload.get("source") or "whatsapp-bot"

    endpoint = f"{base_url.rstrip('/')}/ameyowebaccess/command/"
    request_data: dict[str, object] = {
        "userId": ameyo_user_id,
        "password": password,
        "terminal": terminal,
        "campaignId": campaign_id,
        "phone": payload["phone"],
        "requestId": request_id,
    }
    should_add_customer = _optional_config("AMEYO_SHOULD_ADD_CUSTOMER")
    if should_add_customer:
        request_data["shouldAddCustomer"] = should_add_customer
    if additional_params:
        request_data["additionalParams"] = additional_params

    params = {"command": "clickToCall", "data": json.dumps(request_data)}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            endpoint,
            params=params,
            headers={"accept": "application/json, text/plain, */*"},
        )

    body_text = response.text
    if response.status_code >= 400:
        raise RuntimeError(
            f"Ameyo call API failed with status {response.status_code}: {body_text}"
        )

    response_body: dict | None = None
    if body_text.strip():
        try:
            parsed = json.loads(body_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            response_body = parsed

    if response_body:
        status_value = str(response_body.get("status", "")).lower()
        result_value = str(response_body.get("result", "")).lower()
        if status_value in {"error", "failed", "failure"}:
            raise RuntimeError(f"Ameyo call API returned failure status: {body_text}")
        if result_value in {"error", "failed", "failure"}:
            raise RuntimeError(f"Ameyo call API returned failure result: {body_text}")

    return {"ameyo_call_ref": _pick_call_ref(response_body, request_id)}
