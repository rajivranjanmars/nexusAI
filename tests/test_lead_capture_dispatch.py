"""Regression tests for lead-capture downstream dispatch semantics."""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest


async def _unconfigured_get_client() -> Any:
    """Fail clearly if a test forgets to patch Redis."""

    raise RuntimeError("Redis client was not patched for this test")


@pytest.fixture()
def lead_modules(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, Any]]:
    """Import lead modules with a scoped Redis stub."""

    redis_client_stub = types.ModuleType("shared.redis_client")
    redis_client_stub.get_client = _unconfigured_get_client
    monkeypatch.setitem(sys.modules, "shared.redis_client", redis_client_stub)
    sys.modules.pop("mcp_server.tools.lead_tools", None)
    sys.modules.pop("orchestration.lead_capture_inactivity", None)

    lead_tools = importlib.import_module("mcp_server.tools.lead_tools")
    lead_capture_inactivity = importlib.import_module("orchestration.lead_capture_inactivity")
    yield lead_tools, lead_capture_inactivity

    sys.modules.pop("mcp_server.tools.lead_tools", None)
    sys.modules.pop("orchestration.lead_capture_inactivity", None)
    tools_package = sys.modules.get("mcp_server.tools")
    if tools_package and hasattr(tools_package, "lead_tools"):
        delattr(tools_package, "lead_tools")
    orchestration_package = sys.modules.get("orchestration")
    if orchestration_package and hasattr(orchestration_package, "lead_capture_inactivity"):
        delattr(orchestration_package, "lead_capture_inactivity")


class FakeRedis:
    """Small async Redis double for the lead-capture state machine tests."""

    def __init__(self) -> None:
        """Initialize in-memory key-value and sorted-set storage."""

        self.values: dict[str, str] = {}
        self.zsets: dict[str, dict[str, int]] = {}

    async def get(self, key: str) -> str | None:
        """Return a stored string value."""

        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        """Store a value, honoring Redis-like NX semantics."""

        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def delete(self, key: str) -> int:
        """Delete a key and return the number of removed keys."""

        existed = key in self.values
        self.values.pop(key, None)
        return 1 if existed else 0

    async def zadd(self, key: str, mapping: dict[str, int]) -> int:
        """Add or update sorted-set members."""

        self.zsets.setdefault(key, {}).update(mapping)
        return len(mapping)

    async def zrem(self, key: str, member: str) -> int:
        """Remove a sorted-set member."""

        existed = member in self.zsets.get(key, {})
        self.zsets.get(key, {}).pop(member, None)
        return 1 if existed else 0

    async def zrangebyscore(
        self,
        key: str,
        min: int,
        max: int,
        start: int = 0,
        num: int | None = None,
    ) -> list[str]:
        """Return members whose scores are within the requested range."""

        members = [
            member
            for member, score in sorted(self.zsets.get(key, {}).items(), key=lambda item: item[1])
            if min <= score <= max
        ]
        if num is None:
            return members[start:]
        return members[start : start + num]


@pytest.fixture()
def fake_redis(
    monkeypatch: pytest.MonkeyPatch,
    lead_modules: tuple[Any, Any],
) -> FakeRedis:
    """Patch lead-capture modules to use a shared fake Redis client."""

    lead_tools, lead_capture_inactivity = lead_modules
    redis = FakeRedis()

    async def get_fake_client() -> FakeRedis:
        """Return the fake Redis client."""

        return redis

    monkeypatch.setattr(lead_tools, "get_client", get_fake_client)
    monkeypatch.setattr(lead_capture_inactivity, "get_client", get_fake_client)
    return redis


@pytest.fixture()
def capture_dispatch_calls(
    monkeypatch: pytest.MonkeyPatch,
    lead_modules: tuple[Any, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Patch downstream integrations and capture their payloads."""

    lead_tools, _lead_capture_inactivity = lead_modules
    calls: dict[str, list[dict[str, Any]]] = {"ameyo": [], "lpude": []}

    async def trigger_ameyo_call(payload: dict[str, Any]) -> dict[str, str]:
        """Capture Ameyo call payloads."""

        calls["ameyo"].append(payload)
        return {"ameyo_call_ref": str(payload.get("request_id", "fake-call"))}

    async def submit_lpude_lead(payload: dict[str, Any]) -> dict[str, str]:
        """Capture LPUDE lead payloads."""

        calls["lpude"].append(payload)
        return {"lead_id": str(payload.get("submission_id", "fake-lead"))}

    monkeypatch.setattr(lead_tools, "trigger_ameyo_call", trigger_ameyo_call)
    monkeypatch.setattr(lead_tools, "submit_lpude_lead", submit_lpude_lead)
    return calls


async def _dispatch_if_claimed(
    lead_tools: Any,
    session_id: str,
    payload: dict[str, str],
    mode: str,
) -> None:
    """Claim and run a dispatch synchronously for deterministic tests."""

    dispatch_mode = "completed" if mode == "completed" else "partial"
    assert await lead_tools._claim_dispatch(session_id, dispatch_mode) is True
    await lead_tools._dispatch_lead(session_id, payload, dispatch_mode)


def test_partial_ameyo_is_not_repeated_by_completed_dispatch(
    lead_modules: tuple[Any, Any],
    fake_redis: FakeRedis,
    capture_dispatch_calls: dict[str, list[dict[str, Any]]],
) -> None:
    """A later completed dispatch must not call Ameyo again after partial Ameyo."""

    async def run_test() -> None:
        """Execute the async dispatch assertions."""

        lead_tools, _lead_capture_inactivity = lead_modules
        await _dispatch_if_claimed(
            lead_tools,
            "session-1",
            {
                "name": "Rajiv Ranjan",
                "email": "rajiv@example.com",
                "mobile": "8540029641",
            },
            "partial",
        )

        completed_payload = {
            "name": "Rajiv Ranjan",
            "mobile": "8540029641",
            "email": "rajiv@example.com",
            "program_level": "PG",
            "program_names": "Master of Business Administration",
            "state": "Bihar",
            "city": "Patna",
            "address": "Test address",
            "status": "completed",
        }
        assert await lead_tools._should_dispatch_completed_lead_async("session-1", completed_payload) is True
        await lead_tools._dispatch_lead("session-1", completed_payload, "completed")

        assert len(capture_dispatch_calls["ameyo"]) == 1
        assert len(capture_dispatch_calls["lpude"]) == 1
        assert capture_dispatch_calls["lpude"][-1]["email"] == "rajiv@example.com"
        assert await fake_redis.get("lead-capture:completed-dispatch:session-1:completed") == "1"

    asyncio.run(run_test())


def test_partial_dispatch_requires_name_email_and_mobile(
    lead_modules: tuple[Any, Any],
    fake_redis: FakeRedis,
    capture_dispatch_calls: dict[str, list[dict[str, Any]]],
) -> None:
    """A partial lead should not dispatch until both services have minimum data."""

    async def run_test() -> None:
        """Execute the async minimum-field assertions."""

        lead_tools, lead_capture_inactivity = lead_modules
        await lead_capture_inactivity.remember_lead_snapshot(
            "session-2",
            {"name": "Rajiv Ranjan", "mobile": "8540029641"},
        )
        result = await lead_tools.flush_partial_lead_after_inactivity("session-2")

        assert result == {
            "success": True,
            "dispatched": False,
            "reason": "missing_partial_minimum",
        }
        assert capture_dispatch_calls["ameyo"] == []
        assert capture_dispatch_calls["lpude"] == []

    asyncio.run(run_test())


def test_partial_dispatch_calls_both_services_with_minimum_fields(
    lead_modules: tuple[Any, Any],
    fake_redis: FakeRedis,
    capture_dispatch_calls: dict[str, list[dict[str, Any]]],
) -> None:
    """A name+email+mobile partial lead should dispatch to both services."""

    async def run_test() -> None:
        """Execute the async partial-dispatch assertions."""

        lead_tools, _lead_capture_inactivity = lead_modules
        await _dispatch_if_claimed(
            lead_tools,
            "session-2b",
            {
                "name": "Rajiv Ranjan",
                "email": "rajiv@example.com",
                "mobile": "8540029641",
            },
            "partial",
        )

        assert len(capture_dispatch_calls["ameyo"]) == 1
        assert len(capture_dispatch_calls["lpude"]) == 1
        assert await fake_redis.get("lead-capture:partial-dispatch:session-2b:completed") == "1"

    asyncio.run(run_test())


def test_partial_dispatch_reschedules_after_downstream_failure(
    lead_modules: tuple[Any, Any],
    fake_redis: FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed partial dispatch should be scheduled for retry."""

    calls: dict[str, list[dict[str, Any]]] = {"ameyo": [], "lpude": []}

    async def run_test() -> None:
        """Execute the async retry-scheduling assertions."""

        lead_tools, lead_capture_inactivity = lead_modules

        async def trigger_ameyo_call(payload: dict[str, Any]) -> dict[str, str]:
            """Capture Ameyo payloads."""

            calls["ameyo"].append(payload)
            return {"ameyo_call_ref": str(payload.get("request_id", "fake-call"))}

        async def submit_lpude_lead(payload: dict[str, Any]) -> dict[str, str]:
            """Fail LPUDE to simulate a transient downstream outage."""

            calls["lpude"].append(payload)
            raise RuntimeError("temporary LPUDE outage")

        monkeypatch.setattr(lead_tools, "trigger_ameyo_call", trigger_ameyo_call)
        monkeypatch.setattr(lead_tools, "submit_lpude_lead", submit_lpude_lead)

        await _dispatch_if_claimed(
            lead_tools,
            "session-retry",
            {
                "name": "Rajiv Ranjan",
                "email": "rajiv@example.com",
                "mobile": "8540029641",
            },
            "partial",
        )

        due_set = lead_capture_inactivity._PARTIAL_LEAD_DUE_SET
        assert "session-retry" in fake_redis.zsets[due_set]
        assert await fake_redis.get("lead-capture:partial-dispatch:session-retry:completed") is None
        assert len(calls["ameyo"]) == 1
        assert len(calls["lpude"]) == 1

    asyncio.run(run_test())


def test_completed_dispatch_does_not_duplicate_in_flight_partial_ameyo(
    lead_modules: tuple[Any, Any],
    fake_redis: FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed lead racing with partial Ameyo should not place a second call."""

    calls: dict[str, list[dict[str, Any]]] = {"ameyo": [], "lpude": []}

    async def run_test() -> None:
        """Execute the concurrent dispatch race."""

        lead_tools, _lead_capture_inactivity = lead_modules
        ameyo_started = asyncio.Event()
        release_ameyo = asyncio.Event()

        async def trigger_ameyo_call(payload: dict[str, Any]) -> dict[str, str]:
            """Capture and hold Ameyo until completed dispatch overlaps."""

            calls["ameyo"].append(payload)
            ameyo_started.set()
            await release_ameyo.wait()
            return {"ameyo_call_ref": str(payload.get("request_id", "fake-call"))}

        async def submit_lpude_lead(payload: dict[str, Any]) -> dict[str, str]:
            """Capture LPUDE payloads."""

            calls["lpude"].append(payload)
            return {"lead_id": str(payload.get("submission_id", "fake-lead"))}

        monkeypatch.setattr(lead_tools, "trigger_ameyo_call", trigger_ameyo_call)
        monkeypatch.setattr(lead_tools, "submit_lpude_lead", submit_lpude_lead)

        partial_payload = {
            "name": "Rajiv Ranjan",
            "email": "rajiv@example.com",
            "mobile": "8540029641",
        }
        assert await lead_tools._claim_dispatch("session-4", "partial") is True
        partial_task = asyncio.create_task(
            lead_tools._dispatch_lead("session-4", partial_payload, "partial")
        )
        await ameyo_started.wait()

        completed_payload = {
            "name": "Rajiv Ranjan",
            "mobile": "8540029641",
            "email": "rajiv@example.com",
            "program_level": "PG",
            "program_names": "Master of Business Administration",
            "state": "Bihar",
            "city": "Patna",
            "address": "Test address",
            "status": "completed",
        }
        assert await lead_tools._should_dispatch_completed_lead_async("session-4", completed_payload) is True
        completed_task = asyncio.create_task(
            lead_tools._dispatch_lead("session-4", completed_payload, "completed")
        )
        await asyncio.sleep(0)
        release_ameyo.set()
        await asyncio.gather(partial_task, completed_task)

        assert len(calls["ameyo"]) == 1
        assert len(calls["lpude"]) == 1
        assert await fake_redis.get("lead-capture:completed-dispatch:session-4:completed") == "1"

    asyncio.run(run_test())


def test_touch_lead_activity_extends_inactivity_deadline(
    lead_modules: tuple[Any, Any],
    fake_redis: FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any chat activity for a partial session should push the flush deadline."""

    async def run_test() -> None:
        """Execute the async activity-touch assertions."""

        lead_tools, lead_capture_inactivity = lead_modules
        monkeypatch.setattr(lead_capture_inactivity.time, "time", lambda: 1_000)
        snapshot = await lead_capture_inactivity.remember_lead_snapshot(
            "session-3",
            {
                "name": "Rajiv Ranjan",
                "email": "rajiv@example.com",
                "mobile": "8540029641",
            },
        )
        await lead_capture_inactivity.schedule_partial_flush(
            "session-3",
            snapshot,
            lead_tools._is_partial_flush_candidate,
        )

        due_set = lead_capture_inactivity._PARTIAL_LEAD_DUE_SET
        assert fake_redis.zsets[due_set]["session-3"] == 1_150

        monkeypatch.setattr(lead_capture_inactivity.time, "time", lambda: 1_100)
        await lead_tools.touch_lead_session_activity("session-3")

        assert fake_redis.zsets[due_set]["session-3"] == 1_250

    asyncio.run(run_test())
