"""
Tests for the disambiguation node.

Stubs conversation_memory and workflow_config functions to avoid requiring
a live Redis or database connection.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass(frozen=True)
class _WorkflowResponseConfig:
    """Mock WorkflowResponseConfig."""
    strategy: str = "answer"


@dataclass(frozen=True)
class _DisambiguationConfig:
    """Mock DisambiguationConfig."""
    min_gain: float = 0.35
    axes: tuple = ()


@dataclass(frozen=True)
class _WorkflowConfig:
    """Mock WorkflowConfig."""
    workflow_response_config: dict
    disambiguation: _DisambiguationConfig


def _make_workflow_config(
    strategy: str = "answer",
    axes: tuple = (),
) -> _WorkflowConfig:
    """Build a mock WorkflowConfig."""
    return _WorkflowConfig(
        workflow_response_config={
            "general": _WorkflowResponseConfig(strategy=strategy),
        },
        disambiguation=_DisambiguationConfig(min_gain=0.35, axes=axes),
    )


@pytest.fixture()
def node(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Fixture that stubs conversation_memory and workflow_config."""
    get_calls: list[tuple[str, str]] = []
    set_calls: list[tuple[str, str, dict]] = []
    clear_calls: list[tuple[str, str]] = []

    # Memory holder for test-configurable return values
    memory_holder: dict[str, Any] = {
        "get_clarification_return": {},
        "get_clarification_raises": None,
        "set_clarification_raises": None,
    }

    memory_stub = types.ModuleType("orchestration.conversation_memory")

    async def _get_clarification(app_id: str, session_key: str) -> dict:
        get_calls.append((app_id, session_key))
        if memory_holder["get_clarification_raises"]:
            raise memory_holder["get_clarification_raises"]
        return memory_holder["get_clarification_return"]

    async def _set_clarification(app_id: str, session_key: str, clarification: dict) -> None:
        set_calls.append((app_id, session_key, clarification))
        if memory_holder["set_clarification_raises"]:
            raise memory_holder["set_clarification_raises"]

    async def _clear_clarification(app_id: str, session_key: str) -> None:
        clear_calls.append((app_id, session_key))

    memory_stub.get_clarification = _get_clarification
    memory_stub.set_clarification = _set_clarification
    memory_stub.clear_clarification = _clear_clarification

    # Stub workflow_config with a mutable config holder
    config_holder: dict[str, _WorkflowConfig] = {"current": _make_workflow_config()}

    def _get_workflow_config_stub(app_id: str) -> _WorkflowConfig:
        return config_holder["current"]

    config_stub = types.ModuleType("orchestration.workflow_config")
    config_stub.get_workflow_config = _get_workflow_config_stub

    monkeypatch.setitem(sys.modules, "orchestration.conversation_memory", memory_stub)
    monkeypatch.setitem(sys.modules, "orchestration.workflow_config", config_stub)
    sys.modules.pop("orchestration.nodes.disambiguate_node", None)

    module = importlib.import_module("orchestration.nodes.disambiguate_node")
    module._get_calls = get_calls
    module._set_calls = set_calls
    module._clear_calls = clear_calls

    # Store references for test manipulation
    module._memory_holder = memory_holder
    module._config_holder = config_holder

    yield module

    sys.modules.pop("orchestration.nodes.disambiguate_node", None)


def _state(
    app_id: str = "app1",
    session_id: str | None = None,
    actor_id: str | None = "actor1",
    user_input: str = "tell me about programmes",
    detected_workflow: str = "general",
    rag_context: list | None = None,
) -> dict:
    """Build a minimal WorkflowState for testing."""
    state = {
        "app_id": app_id,
        "user_input": user_input,
        "detected_workflow": detected_workflow,
    }
    if session_id is not None:
        state["session_id"] = session_id
    if actor_id is not None:
        state["actor_id"] = actor_id
    if rag_context is not None:
        state["rag_context"] = rag_context
    return state


def _chunk(source_url: str, score: float = 0.8, content: str | None = None) -> dict:
    """Build a single RAG chunk. Default content varies by URL to pass differ_gate."""
    if content is None:
        # Use URL as content to ensure differ_gate sees different content
        content = source_url
    return {
        "source_url": source_url,
        "score": score,
        "content": content,
    }


@pytest.mark.asyncio
async def test_missing_app_id_returns_unchanged(node):
    """Node returns state unchanged when app_id is missing."""
    state = _state(app_id="")
    result = await node.disambiguate_node(state)
    assert result == state
    assert node._get_calls == []
    assert node._set_calls == []


@pytest.mark.asyncio
async def test_missing_session_key_returns_unchanged(node):
    """Node returns state unchanged when both session_id and actor_id are missing."""
    state = _state(actor_id="", session_id=None)
    result = await node.disambiguate_node(state)
    assert result == state
    assert node._get_calls == []
    assert node._set_calls == []


@pytest.mark.asyncio
async def test_elicit_workflow_returns_unchanged(node):
    """Node returns state unchanged when workflow strategy is 'elicit'."""
    # Override config stub to return elicit strategy
    node._config_holder["current"] = _WorkflowConfig(
        workflow_response_config={
            "lead_capture": _WorkflowResponseConfig(strategy="elicit"),
            "general": _WorkflowResponseConfig(strategy="answer"),
        },
        disambiguation=_DisambiguationConfig(min_gain=0.35, axes=()),
    )

    state = _state(detected_workflow="lead_capture")
    result = await node.disambiguate_node(state)
    assert result == state


@pytest.mark.asyncio
async def test_empty_rag_context_returns_unchanged(node):
    """Node returns state unchanged when rag_context is empty."""
    state = _state(rag_context=[])
    result = await node.disambiguate_node(state)
    assert result == state
    # Node still calls get_clarification to check budget
    assert node._get_calls == [("app1", "actor1")]
    assert node._set_calls == []


@pytest.mark.asyncio
async def test_single_programme_returns_unchanged(node):
    """Node returns state unchanged when all chunks are from the same programme."""
    # Configure the node with programme axis
    node._config_holder["current"] = _WorkflowConfig(
        workflow_response_config={
            "general": _WorkflowResponseConfig(strategy="answer"),
        },
        disambiguation=_DisambiguationConfig(min_gain=0.0, axes=(("programme", "/programmes/"),)),
    )

    state = _state(
        rag_context=[
            _chunk("https://example.com/programmes/mba/overview", score=0.9),
            _chunk("https://example.com/programmes/mba/fees", score=0.8),
        ]
    )
    result = await node.disambiguate_node(state)
    assert result == state
    assert node._get_calls == [("app1", "actor1")]
    # No question was asked, so clarification was not set
    assert node._set_calls == []


@pytest.mark.asyncio
async def test_ambiguous_programmes_asks_question(node):
    """Node sets llm_response and pending clarification when programmes are ambiguous."""
    # Configure the node with programme axis
    node._config_holder["current"] = _WorkflowConfig(
        workflow_response_config={
            "general": _WorkflowResponseConfig(strategy="answer"),
        },
        disambiguation=_DisambiguationConfig(min_gain=0.0, axes=(("programme", "/programmes/"),)),
    )

    state = _state(
        rag_context=[
            _chunk("https://example.com/programmes/mba/overview", score=0.9),
            _chunk("https://example.com/programmes/mba/fees", score=0.8),
            _chunk("https://example.com/programmes/bca/overview", score=0.85),
        ]
    )
    result = await node.disambiguate_node(state)

    # Check that llm_response was set with a question
    assert result.get("llm_response") is not None
    assert "which programme" in result["llm_response"].lower()
    assert "MBA" in result["llm_response"]
    assert "BCA" in result["llm_response"]

    # Check that metadata was set
    assert result["metadata"]["prechecked_response"] is True

    # Check that clarification was set in state
    clarification = result["clarification"]
    assert clarification is not None
    assert clarification["pending"] is True
    assert clarification["axis"] == "programme"
    assert len(clarification["options"]) >= 2

    # Check that set_clarification was called
    assert node._set_calls == [("app1", "actor1", clarification)]


@pytest.mark.asyncio
async def test_pending_clarification_clears_and_proceeds(node):
    """Node clears pending clarification and proceeds to answer."""
    # Override get_clarification to return pending state
    node._memory_holder["get_clarification_return"] = {
        "pending": True,
        "axis": "programme",
        "options": ["mba", "bca"],
        "asked_for": "tell me about programmes",
    }

    # Configure with programme axis
    node._config_holder["current"] = _WorkflowConfig(
        workflow_response_config={
            "general": _WorkflowResponseConfig(strategy="answer"),
        },
        disambiguation=_DisambiguationConfig(min_gain=0.0, axes=(("programme", "/programmes/"),)),
    )

    state = _state(
        rag_context=[
            _chunk("https://example.com/programmes/mba/overview", score=0.9),
            _chunk("https://example.com/programmes/bca/overview", score=0.85),
        ]
    )
    result = await node.disambiguate_node(state)

    # Node should return without setting llm_response
    assert result.get("llm_response") is None
    assert result.get("clarification") is None

    # clear_clarification should have been called
    assert node._clear_calls == [("app1", "actor1")]


@pytest.mark.asyncio
async def test_max_four_options_shown(node):
    """Node shows at most 4 options in the question."""
    # Configure with programme axis
    node._config_holder["current"] = _WorkflowConfig(
        workflow_response_config={
            "general": _WorkflowResponseConfig(strategy="answer"),
        },
        disambiguation=_DisambiguationConfig(min_gain=0.0, axes=(("programme", "/programmes/"),)),
    )

    state = _state(
        rag_context=[
            _chunk("https://example.com/programmes/mba/overview", score=0.9),
            _chunk("https://example.com/programmes/bca/overview", score=0.85),
            _chunk("https://example.com/programmes/mtech/overview", score=0.8),
            _chunk("https://example.com/programmes/phd/overview", score=0.75),
            _chunk("https://example.com/programmes/diploma/overview", score=0.7),
        ]
    )
    result = await node.disambiguate_node(state)

    question = result.get("llm_response", "")
    # Count options shown: look for uppercase options
    shown_options = [opt for opt in ["MBA", "BCA", "MTECH", "PHD", "DIPLOMA"] if opt in question]
    assert len(shown_options) == 4  # max of 4 shown

    # But clarification should have all options
    clarification = result["clarification"]
    assert len(clarification["options"]) >= 5


@pytest.mark.asyncio
async def test_get_clarification_exception_treated_as_no_pending(node):
    """Node treats get_clarification exceptions as 'no pending clarification'."""
    # Make get_clarification raise an exception
    node._memory_holder["get_clarification_raises"] = RuntimeError("Redis connection failed")

    # Configure with programme axis
    node._config_holder["current"] = _WorkflowConfig(
        workflow_response_config={
            "general": _WorkflowResponseConfig(strategy="answer"),
        },
        disambiguation=_DisambiguationConfig(min_gain=0.0, axes=(("programme", "/programmes/"),)),
    )

    state = _state(
        rag_context=[
            _chunk("https://example.com/programmes/mba/overview", score=0.9),
            _chunk("https://example.com/programmes/bca/overview", score=0.85),
        ]
    )
    result = await node.disambiguate_node(state)

    # Node should still ask the question
    assert result.get("llm_response") is not None
    assert "which programme" in result["llm_response"].lower()
    assert result["metadata"]["prechecked_response"] is True


@pytest.mark.asyncio
async def test_set_clarification_exception_still_returns_question(node):
    """Node still returns the question even if set_clarification fails."""
    # Make set_clarification raise an exception
    node._memory_holder["set_clarification_raises"] = RuntimeError("Redis connection failed")

    # Configure with programme axis
    node._config_holder["current"] = _WorkflowConfig(
        workflow_response_config={
            "general": _WorkflowResponseConfig(strategy="answer"),
        },
        disambiguation=_DisambiguationConfig(min_gain=0.0, axes=(("programme", "/programmes/"),)),
    )

    state = _state(
        rag_context=[
            _chunk("https://example.com/programmes/mba/overview", score=0.9),
            _chunk("https://example.com/programmes/bca/overview", score=0.85),
        ]
    )
    result = await node.disambiguate_node(state)

    # Node should still have set llm_response in state
    assert result.get("llm_response") is not None
    assert "which programme" in result["llm_response"].lower()
    assert result["metadata"]["prechecked_response"] is True
    assert result["clarification"] is not None
