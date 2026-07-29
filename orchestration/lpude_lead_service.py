"""LPUDE lead submission integration."""

from __future__ import annotations

import httpx
import os
from typing import TypedDict


class LpudeLeadPayload(TypedDict, total=False):
    """Completed lead payload sent to the live LPUDE enquiry endpoint."""

    name: str
    email: str
    mobile: str
    submission_id: str
    program: str
    state: str
    city: str
    address: str


class LpudeLeadResponse(TypedDict):
    """Normalized LPUDE lead API response."""

    lead_id: str


def _lpude_lead_api_url() -> str:
    """Return the configured LPUDE lead endpoint URL."""

    return os.getenv("LPUDE_LEAD_API_URL", "").strip()


async def submit_lpude_lead(payload: LpudeLeadPayload) -> LpudeLeadResponse:
    """Submit a completed lead to the LPUDE enquiry endpoint."""

    api_url = _lpude_lead_api_url()
    if not api_url:
        raise ValueError("LPUDE_LEAD_API_URL is not configured")

    request_body = {
        "name": payload["name"],
        "email": payload["email"],
        "mobile": payload["mobile"],
    }
    for optional_field in ("program", "state", "city", "address"):
        value = payload.get(optional_field)
        if value:
            request_body[optional_field] = value

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(api_url, json=request_body)

    if response.status_code >= 400:
        raise RuntimeError(
            f"LPUDE lead API failed with status {response.status_code}: {response.text}"
        )

    response_body = response.json()
    if not isinstance(response_body, dict) or response_body.get("success") is not True:
        raise RuntimeError(f"LPUDE lead API returned unexpected response: {response.text}")

    lead_id = response_body.get("lead_id")
    if isinstance(lead_id, str) and lead_id.strip():
        return {"lead_id": lead_id.strip()}

    fallback_id = payload.get("submission_id") or payload["mobile"]
    return {"lead_id": fallback_id}
