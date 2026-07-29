"""Regression tests for lead_capture sticky-lock behavior in detect_workflow_node.

Stubs out the heavy DB-backed imports (llm.classifier, orchestration.workflow_policy,
orchestration.conversation_memory) the same way test_lead_capture_dispatch.py stubs
shared.redis_client, so this can run without a live Postgres/Redis connection.
"""

from __future__ import annotations

import importlib
import sys
import time
import types
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture()
def node(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    classify_calls: list[str] = []
    clear_calls: list[tuple[str, str]] = []

    classifier_stub = types.ModuleType("llm.classifier")

    async def _classify_intent(*args, **kwargs):
        classify_calls.append(args[0] if args else kwargs.get("user_input", ""))
        return "general", {}

    classifier_stub.classify_intent = _classify_intent

    policy_stub = types.ModuleType("orchestration.workflow_policy")
    policy_stub.resolve_effective_workflow = lambda *a, **k: (a[0] if a else "general", "")

    memory_stub = types.ModuleType("orchestration.conversation_memory")

    async def _clear_lead_progress(app_id: str, session_id: str) -> None:
        clear_calls.append((app_id, session_id))

    async def _get_lead_progress(app_id: str, session_id: str) -> dict:
        return {}

    memory_stub.clear_lead_progress = _clear_lead_progress
    memory_stub.get_lead_progress = _get_lead_progress

    monkeypatch.setitem(sys.modules, "llm.classifier", classifier_stub)
    monkeypatch.setitem(sys.modules, "orchestration.workflow_policy", policy_stub)
    monkeypatch.setitem(sys.modules, "orchestration.conversation_memory", memory_stub)
    sys.modules.pop("orchestration.nodes.detect_workflow_node", None)

    module = importlib.import_module("orchestration.nodes.detect_workflow_node")
    module._classify_calls = classify_calls
    module._clear_calls = clear_calls
    yield module

    sys.modules.pop("orchestration.nodes.detect_workflow_node", None)


def _state(lead_progress: dict) -> dict:
    return {
        "app_id": "app1",
        "session_id": "sess1",
        "actor_id": "actor1",
        "user_input": "what programs do you offer",
        "lead_progress": lead_progress,
    }


@pytest.mark.asyncio
async def test_fresh_in_progress_session_stays_sticky(node):
    state = _state({"status": "in_progress", "updated_at": time.time()})
    result = await node.detect_workflow_node(state)
    assert result["detected_workflow"] == "lead_capture"
    assert node._clear_calls == []
    assert node._classify_calls == []


@pytest.mark.asyncio
async def test_stale_in_progress_session_releases_lock(node):
    stale_ts = time.time() - node.LEAD_CAPTURE_ABANDON_SECONDS - 1
    state = _state({"status": "in_progress", "updated_at": stale_ts})
    result = await node.detect_workflow_node(state)
    assert result["detected_workflow"] == "general"
    assert node._clear_calls == [("app1", "sess1")]
    assert node._classify_calls == ["what programs do you offer"]


@pytest.mark.asyncio
async def test_missing_timestamp_stays_sticky_for_back_compat(node):
    state = _state({"status": "in_progress"})
    result = await node.detect_workflow_node(state)
    assert result["detected_workflow"] == "lead_capture"
    assert node._clear_calls == []
