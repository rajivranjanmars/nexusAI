"""
Tests for narrowed answer labels and "Other" retry logic.

Tests the prompt builder's ambiguity labeling feature and the disambiguate
node's retry-once logic for unmatched user input.
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
class _ResponseStyleConfig:
    """Mock ResponseStyleConfig."""
    format_instructions: str = "Format with clear sections and bullet points.\n"


@dataclass(frozen=True)
class _WorkflowResponseConfig:
    """Mock WorkflowResponseConfig."""
    strategy: str = "answer"
    default_style: str = "balanced"
    include_sources: bool = True
    max_rag_chars: int = 4000
    history_limit: int = 6
    system_suffix: str | None = None


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
    response_styles: dict


def _make_workflow_config(
    strategy: str = "answer",
    axes: tuple = (),
    response_styles: dict | None = None,
) -> _WorkflowConfig:
    """Build a mock WorkflowConfig."""
    if response_styles is None:
        response_styles = {
            "balanced": _ResponseStyleConfig(),
        }
    return _WorkflowConfig(
        workflow_response_config={
            "general": _WorkflowResponseConfig(strategy=strategy),
        },
        disambiguation=_DisambiguationConfig(min_gain=0.35, axes=axes),
        response_styles=response_styles,
    )


@pytest.fixture()
def prompt_builder_fixture(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Fixture that stubs workflow_config and prompt_loader for PromptBuilder tests."""
    # Stub workflow_config
    config_holder: dict[str, _WorkflowConfig] = {"current": _make_workflow_config()}

    def _get_workflow_config_stub(app_id: str) -> _WorkflowConfig:
        return config_holder["current"]

    config_stub = types.ModuleType("orchestration.workflow_config")
    config_stub.get_workflow_config = _get_workflow_config_stub
    config_stub.WorkflowResponseConfig = _WorkflowResponseConfig

    # Stub prompt_loader to avoid loading files
    def _render_prompt(prompt_id: str, **kwargs: Any) -> tuple[str, str]:
        return "System prompt", "User prompt"

    prompt_loader_stub = types.ModuleType("shared.prompt_loader")
    prompt_loader_stub.render_prompt = _render_prompt

    # Stub app_registry to return None (no app context)
    def _resolve_by_app_id(app_id: str) -> None:
        return None

    app_registry_stub = types.ModuleType("db.app_registry")
    app_registry_stub.resolve_by_app_id = _resolve_by_app_id

    monkeypatch.setitem(sys.modules, "orchestration.workflow_config", config_stub)
    monkeypatch.setitem(sys.modules, "shared.prompt_loader", prompt_loader_stub)
    monkeypatch.setitem(sys.modules, "db.app_registry", app_registry_stub)
    sys.modules.pop("llm.prompt_builder", None)

    module = importlib.import_module("llm.prompt_builder")
    module._config_holder = config_holder

    yield module

    sys.modules.pop("llm.prompt_builder", None)


@pytest.fixture()
def disambiguate_fixture(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Fixture that stubs conversation_memory and workflow_config for disambiguate_node tests."""
    get_calls: list[tuple[str, str]] = []
    set_calls: list[tuple[str, str, dict]] = []
    clear_calls: list[tuple[str, str]] = []

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
    module._memory_holder = memory_holder
    module._config_holder = config_holder

    yield module

    sys.modules.pop("orchestration.nodes.disambiguate_node", None)


def _chunk(
    source_url: str,
    score: float = 0.8,
    content: str | None = None,
    section_heading: str = "",
) -> dict:
    """Build a single RAG chunk."""
    if content is None:
        content = source_url
    return {
        "source_url": source_url,
        "score": score,
        "content": content,
        "section_heading": section_heading,
    }


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


# ============================================================================
# Tests for build_regular_prompt (CHANGE 1)
# ============================================================================


def test_build_regular_prompt_multiple_programmes_adds_ambiguity_label(
    prompt_builder_fixture,
):
    """Prompt contains AMBIGUITY LABELLING when rag_context spans multiple programmes."""
    prompt_builder_fixture._config_holder["current"] = _make_workflow_config(
        axes=(("programme", "/programmes/"),),
    )

    from llm.prompt_builder import PromptBuilder

    builder = PromptBuilder(app_id="", workflow="general")

    # RAG context with content differing by programme
    rag_context = [
        _chunk(
            "https://example.com/programmes/mba/overview",
            score=0.9,
            content="MBA programme overview",
        ),
        _chunk(
            "https://example.com/programmes/mba/fees",
            score=0.8,
            content="MBA fees structure",
        ),
        _chunk(
            "https://example.com/programmes/bca/overview",
            score=0.85,
            content="BCA programme overview",
        ),
    ]

    from orchestration.workflow_config import WorkflowResponseConfig

    workflow_config = WorkflowResponseConfig(
        default_style="balanced",
        include_sources=True,
        max_rag_chars=4000,
        history_limit=6,
    )

    system_prompt, user_prompt = builder.build_regular_prompt(
        user_query="Tell me about programmes",
        student_data="",
        retrieved_context="",
        history_text="",
        rag_context=rag_context,
        workflow_config=workflow_config,
        answer_confidence=0.7,
    )

    # System prompt should contain the AMBIGUITY LABELLING instruction
    assert "AMBIGUITY LABELLING" in system_prompt
    assert "programme" in system_prompt.lower()


def test_build_regular_prompt_single_programme_no_ambiguity_label(
    prompt_builder_fixture,
):
    """Prompt does NOT contain AMBIGUITY LABELLING when rag_context is from a single programme."""
    prompt_builder_fixture._config_holder["current"] = _make_workflow_config(
        axes=(("programme", "/programmes/"),),
    )

    from llm.prompt_builder import PromptBuilder

    builder = PromptBuilder(app_id="", workflow="general")

    # RAG context with content from only MBA
    rag_context = [
        _chunk(
            "https://example.com/programmes/mba/overview",
            score=0.9,
            content="MBA programme overview",
        ),
        _chunk(
            "https://example.com/programmes/mba/fees",
            score=0.8,
            content="MBA fees structure",
        ),
    ]

    from orchestration.workflow_config import WorkflowResponseConfig

    workflow_config = WorkflowResponseConfig(
        default_style="balanced",
        include_sources=True,
        max_rag_chars=4000,
        history_limit=6,
    )

    system_prompt, user_prompt = builder.build_regular_prompt(
        user_query="Tell me about programmes",
        student_data="",
        retrieved_context="",
        history_text="",
        rag_context=rag_context,
        workflow_config=workflow_config,
        answer_confidence=0.7,
    )

    # System prompt should NOT contain the AMBIGUITY LABELLING instruction
    assert "AMBIGUITY LABELLING" not in system_prompt


def test_build_regular_prompt_empty_rag_context_no_ambiguity_label(
    prompt_builder_fixture,
):
    """Prompt does NOT contain AMBIGUITY LABELLING when rag_context is empty."""
    from llm.prompt_builder import PromptBuilder

    builder = PromptBuilder(app_id="", workflow="general")

    from orchestration.workflow_config import WorkflowResponseConfig

    workflow_config = WorkflowResponseConfig(
        default_style="balanced",
        include_sources=True,
        max_rag_chars=4000,
        history_limit=6,
    )

    system_prompt, user_prompt = builder.build_regular_prompt(
        user_query="Tell me about programmes",
        student_data="",
        retrieved_context="",
        history_text="",
        rag_context=[],
        workflow_config=workflow_config,
        answer_confidence=0.7,
    )

    # System prompt should NOT contain the AMBIGUITY LABELLING instruction
    assert "AMBIGUITY LABELLING" not in system_prompt


# ============================================================================
# Tests for disambiguate_node retry logic (CHANGE 2)
# ============================================================================


@pytest.mark.asyncio
async def test_pending_clarification_matched_reply_clears(disambiguate_fixture):
    """When user reply matches an option, clarification is cleared and node returns unchanged."""
    # Override get_clarification to return pending state with options
    disambiguate_fixture._memory_holder["get_clarification_return"] = {
        "pending": True,
        "axis": "programme",
        "options": ["mba", "bca"],
        "asked_for": "which programme",
    }

    # Configure node with programme axis
    disambiguate_fixture._config_holder["current"] = _make_workflow_config(
        axes=(("programme", "/programmes/"),),
    )

    # User reply matches one of the options
    state = _state(
        user_input="mba",
        rag_context=[
            _chunk("https://example.com/programmes/mba/overview"),
            _chunk("https://example.com/programmes/bca/overview"),
        ],
    )

    result = await disambiguate_fixture.disambiguate_node(state)

    # Node should NOT set llm_response (no re-ask)
    assert result.get("llm_response") is None
    assert result.get("clarification") is None

    # clear_clarification should have been called
    assert disambiguate_fixture._clear_calls == [("app1", "actor1")]


@pytest.mark.asyncio
async def test_pending_clarification_unmatched_reply_retries_once(disambiguate_fixture):
    """When user reply doesn't match any option, node re-asks the same facet once."""
    # Override get_clarification to return pending state
    disambiguate_fixture._memory_holder["get_clarification_return"] = {
        "pending": True,
        "axis": "programme",
        "options": ["mba", "bca"],
        "asked_for": "which programme",
    }

    # Configure node with programme axis
    disambiguate_fixture._config_holder["current"] = _make_workflow_config(
        axes=(("programme", "/programmes/"),),
    )

    # User reply does NOT match any option
    state = _state(
        user_input="m.tech",
        rag_context=[
            _chunk("https://example.com/programmes/mba/overview"),
            _chunk("https://example.com/programmes/bca/overview"),
        ],
    )

    result = await disambiguate_fixture.disambiguate_node(state)

    # Node should set llm_response with retry question
    assert result.get("llm_response") is not None
    assert "didn't catch" in result["llm_response"].lower()
    assert "programme" in result["llm_response"].lower()

    # metadata should mark as prechecked
    assert result["metadata"]["prechecked_response"] is True

    # Clarification should be updated with retried: True
    clarification = result.get("clarification")
    assert clarification is not None
    assert clarification.get("retried") is True
    assert clarification.get("pending") is True

    # set_clarification should have been called with retried flag
    assert len(disambiguate_fixture._set_calls) == 1
    call_clarification = disambiguate_fixture._set_calls[0][2]
    assert call_clarification.get("retried") is True

    # clear_clarification should NOT have been called
    assert disambiguate_fixture._clear_calls == []


@pytest.mark.asyncio
async def test_pending_clarification_already_retried_clears(disambiguate_fixture):
    """When retried flag is already True, node clears and does NOT ask again."""
    # Override get_clarification to return pending state WITH retried flag
    disambiguate_fixture._memory_holder["get_clarification_return"] = {
        "pending": True,
        "retried": True,
        "axis": "programme",
        "options": ["mba", "bca"],
        "asked_for": "which programme",
    }

    # Configure node with programme axis
    disambiguate_fixture._config_holder["current"] = _make_workflow_config(
        axes=(("programme", "/programmes/"),),
    )

    # User reply still doesn't match
    state = _state(
        user_input="m.tech",
        rag_context=[
            _chunk("https://example.com/programmes/mba/overview"),
            _chunk("https://example.com/programmes/bca/overview"),
        ],
    )

    result = await disambiguate_fixture.disambiguate_node(state)

    # Node should NOT set llm_response (don't re-ask again)
    assert result.get("llm_response") is None
    assert result.get("clarification") is None

    # clear_clarification should have been called
    assert disambiguate_fixture._clear_calls == [("app1", "actor1")]

    # set_clarification should NOT have been called (no retry)
    assert disambiguate_fixture._set_calls == []


@pytest.mark.asyncio
async def test_pending_clarification_substring_match_counts(disambiguate_fixture):
    """User reply matching as substring of an option counts as a match."""
    # Override get_clarification to return pending state
    disambiguate_fixture._memory_holder["get_clarification_return"] = {
        "pending": True,
        "axis": "programme",
        "options": ["mba", "bca"],
        "asked_for": "which programme",
    }

    # Configure node with programme axis
    disambiguate_fixture._config_holder["current"] = _make_workflow_config(
        axes=(("programme", "/programmes/"),),
    )

    # User reply contains "bca" as substring
    state = _state(
        user_input="I want to do BCA",
        rag_context=[
            _chunk("https://example.com/programmes/mba/overview"),
            _chunk("https://example.com/programmes/bca/overview"),
        ],
    )

    result = await disambiguate_fixture.disambiguate_node(state)

    # Node should NOT set llm_response (matches "bca")
    assert result.get("llm_response") is None
    assert result.get("clarification") is None

    # clear_clarification should have been called
    assert disambiguate_fixture._clear_calls == [("app1", "actor1")]
