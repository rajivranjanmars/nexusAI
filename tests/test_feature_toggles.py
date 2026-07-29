"""Runtime tests for the per-app OTP and caching feature toggles."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_FAST_MODEL", "test-fast")
os.environ.setdefault("LLM_SMART_MODEL", "test-smart")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("PGVECTOR_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("JWT_SECRET", "test-secret-with-at-least-32-bytes")


def test_proxy_cache_toggle_reflects_registered_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reflect each app's cache toggle while preserving the documented default."""
    from backend_proxy import main as proxy_main

    monkeypatch.setattr(
        proxy_main,
        "resolve_by_app_id",
        lambda app_id: SimpleNamespace(caching_enabled=app_id == "enabled-app"),
    )

    assert proxy_main._caching_enabled_for_app("") is True
    assert proxy_main._caching_enabled_for_app("enabled-app") is True
    assert proxy_main._caching_enabled_for_app("disabled-app") is False


@pytest.mark.asyncio
async def test_cache_nodes_bypass_reads_and_writes_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid cache lookup, embedding, and storage for a disabled app."""
    from orchestration.nodes import cache_node

    embed = AsyncMock(return_value=[1.0, 0.0])
    check_cache = AsyncMock(return_value=None)
    store_response = AsyncMock()
    monkeypatch.setattr(
        cache_node,
        "resolve_by_app_id",
        lambda _app_id: SimpleNamespace(caching_enabled=False),
    )
    monkeypatch.setattr(cache_node, "should_use_response_cache", lambda *args, **kwargs: True)
    monkeypatch.setattr(cache_node, "_embed", embed)
    monkeypatch.setattr(cache_node, "check_cache", check_cache)
    monkeypatch.setattr(cache_node, "store_response", store_response)

    state = {
        "app_id": "disabled-app",
        "detected_workflow": "general",
        "user_input": "hello",
        "llm_response": "response",
    }
    await cache_node.cache_check_node(state)
    await cache_node.cache_store_node(state)

    assert state["cache_hit"] is False
    embed.assert_not_awaited()
    check_cache.assert_not_awaited()
    store_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_nodes_read_and_write_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Perform cache lookup and storage for an enabled app."""
    from orchestration.nodes import cache_node

    embed = AsyncMock(return_value=[1.0, 0.0])
    check_cache = AsyncMock(return_value=None)
    store_response = AsyncMock()
    monkeypatch.setattr(
        cache_node,
        "resolve_by_app_id",
        lambda _app_id: SimpleNamespace(caching_enabled=True),
    )
    monkeypatch.setattr(cache_node, "should_use_response_cache", lambda *args, **kwargs: True)
    monkeypatch.setattr(cache_node, "_cache_scope_for_state", lambda _state: "shared")
    monkeypatch.setattr(cache_node, "_resolve_cache_config", lambda _state: (3600, 0.8))
    monkeypatch.setattr(cache_node, "_embed", embed)
    monkeypatch.setattr(cache_node, "check_cache", check_cache)
    monkeypatch.setattr(cache_node, "store_response", store_response)

    state = {
        "app_id": "enabled-app",
        "detected_workflow": "general",
        "user_input": "hello",
        "llm_response": "response",
    }
    await cache_node.cache_check_node(state)
    await cache_node.cache_store_node(state)

    assert state["cache_hit"] is False
    embed.assert_awaited_once_with("hello")
    check_cache.assert_awaited_once()
    store_response.assert_awaited_once()
