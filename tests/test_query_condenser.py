"""Tests for conversation-aware query condensation."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_FAST_MODEL", "test-fast")
os.environ.setdefault("LLM_SMART_MODEL", "test-smart")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("PGVECTOR_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("JWT_SECRET", "test-secret-with-at-least-32-bytes")


def test_condense_query_with_empty_history_returns_query_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query condensation should return the original query when history is empty."""
    from shared.query_expander import condense_query_with_history

    # Mock call_fast to ensure it's NOT called
    call_fast_mock = AsyncMock()
    monkeypatch.setattr("shared.query_expander.call_fast", call_fast_mock)

    async def run_test():
        result = await condense_query_with_history("what are the fees?", [], "test-app")
        assert result == "what are the fees?"
        call_fast_mock.assert_not_awaited()

    asyncio.run(run_test())


def test_condense_query_with_history_calls_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query condensation should call LLM when history is non-empty."""
    from shared.query_expander import condense_query_with_history

    call_fast_mock = AsyncMock(return_value=("MBA programme fees", {}))
    monkeypatch.setattr("shared.query_expander.call_fast", call_fast_mock)

    async def run_test():
        history = [
            {"role": "user", "content": "Tell me about MBA"},
            {"role": "assistant", "content": "The MBA program..."},
        ]
        result = await condense_query_with_history("what about fees?", history, "test-app")
        assert result == "MBA programme fees"
        call_fast_mock.assert_awaited_once()

    asyncio.run(run_test())


def test_condense_query_returns_original_on_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query condensation should return original query if LLM call fails."""
    from shared.query_expander import condense_query_with_history

    call_fast_mock = AsyncMock(side_effect=Exception("LLM error"))
    monkeypatch.setattr("shared.query_expander.call_fast", call_fast_mock)

    async def run_test():
        history = [{"role": "user", "content": "Tell me about MBA"}]
        result = await condense_query_with_history("what about fees?", history, "test-app")
        assert result == "what about fees?"

    asyncio.run(run_test())


def test_condense_query_limits_history_to_recent_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query condensation should only use the most recent 6 history messages."""
    from shared.query_expander import condense_query_with_history

    call_fast_mock = AsyncMock(return_value=("condensed query", {}))
    monkeypatch.setattr("shared.query_expander.call_fast", call_fast_mock)

    async def run_test():
        history = [
            {"role": "user", "content": f"Message {i}"} for i in range(10)
        ]
        await condense_query_with_history("current query", history, "test-app")
        call_fast_mock.assert_awaited_once()
        call_args = call_fast_mock.call_args
        prompt = call_args[0][0] if call_args[0] else ""
        assert "Message 9" in prompt
        assert "Message 0" not in prompt

    asyncio.run(run_test())


def test_rag_retrieve_node_uses_condensed_query_with_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG retrieval should use condensed query when session and history exist."""
    from orchestration.nodes import rag_retrieve_node

    history = [
        {"role": "user", "content": "Tell me about MBA"},
        {"role": "assistant", "content": "The MBA program..."},
    ]
    get_history_mock = AsyncMock(return_value=history)
    condense_mock = AsyncMock(return_value="MBA programme fees")
    run_retrieval_mock = AsyncMock(
        return_value=type("Result", (), {
            "chunks": [],
            "answer_confidence": 0.5,
            "retrieval_debug": {"t_total": "0.1"},
        })()
    )
    get_retrieval_config_mock = lambda _: type("Config", (), {})()

    monkeypatch.setattr("orchestration.nodes.rag_retrieve_node.get_history", get_history_mock)
    monkeypatch.setattr("orchestration.nodes.rag_retrieve_node.condense_query_with_history", condense_mock)
    monkeypatch.setattr("orchestration.nodes.rag_retrieve_node.run_retrieval", run_retrieval_mock)
    monkeypatch.setattr("orchestration.nodes.rag_retrieve_node._get_retrieval_config", get_retrieval_config_mock)

    async def run_test():
        state = {
            "app_id": "test-app",
            "user_input": "what about fees?",
            "actor_id": "user-123",
            "session_id": "session-456",
        }
        await rag_retrieve_node.rag_retrieve_node(state)
        get_history_mock.assert_awaited_once_with("test-app", "session-456")
        condense_mock.assert_awaited_once_with("what about fees?", history, "test-app")
        run_retrieval_mock.assert_awaited_once_with("MBA programme fees", "test-app")

    asyncio.run(run_test())


def test_rag_retrieve_node_skips_condensation_for_guest_with_no_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG retrieval should skip condensation and use raw query for guests (no session)."""
    from orchestration.nodes import rag_retrieve_node

    condense_mock = AsyncMock()
    run_retrieval_mock = AsyncMock(
        return_value=type("Result", (), {
            "chunks": [],
            "answer_confidence": 0.5,
            "retrieval_debug": {"t_total": "0.1"},
        })()
    )
    get_retrieval_config_mock = lambda _: type("Config", (), {})()

    monkeypatch.setattr("orchestration.nodes.rag_retrieve_node.condense_query_with_history", condense_mock)
    monkeypatch.setattr("orchestration.nodes.rag_retrieve_node.run_retrieval", run_retrieval_mock)
    monkeypatch.setattr("orchestration.nodes.rag_retrieve_node._get_retrieval_config", get_retrieval_config_mock)

    async def run_test():
        state = {
            "app_id": "test-app",
            "user_input": "what about fees?",
        }
        await rag_retrieve_node.rag_retrieve_node(state)
        condense_mock.assert_not_awaited()
        run_retrieval_mock.assert_awaited_once_with("what about fees?", "test-app")

    asyncio.run(run_test())
