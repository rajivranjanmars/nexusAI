"""Tests for chat feedback request and admin feedback routes."""

from __future__ import annotations

import importlib
import os
import sys
import types
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

os.environ.setdefault("LLM_BASE_URL", "https://example.com")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_FAST_MODEL", "fast-model")
os.environ.setdefault("LLM_SMART_MODEL", "smart-model")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("PGVECTOR_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("JWT_SECRET", "test-secret")

from backend_proxy.schemas import ChatFeedbackRequest, ChatFeedbackResponse
from backend_proxy.admin_auth import AdminUserContext, AdminAuthError

_MODULES_TO_RESET: tuple[str, ...] = (
    "backend_proxy.main",
    "backend_proxy.rag_routes",
    "backend_proxy.resolvers",
    "db.feedback_tracker",
    "db.postgres",
)


@asynccontextmanager
async def _fake_async_session() -> Iterator[None]:
    """Yield a placeholder async DB session for route imports."""

    yield None


@contextmanager
def _fake_sync_session() -> Iterator[None]:
    """Yield a placeholder sync DB session for tracker imports."""

    yield None


def _cleanup_test_modules() -> None:
    """Remove modules imported under the fake SQL layer so tests do not leak state."""

    for module_name in _MODULES_TO_RESET:
        sys.modules.pop(module_name, None)
    db_package = sys.modules.get("db")
    if db_package is not None and hasattr(db_package, "postgres"):
        delattr(db_package, "postgres")


def _build_admin_user() -> AdminUserContext:
    """Return a representative admin user for dependency overrides."""

    return AdminUserContext(
        admin_user_id="admin-1",
        email="admin@example.com",
        display_name="Admin",
        admin_role="super_admin",
        app_id=None,
        token_id="token-1",
    )


def _build_app_user() -> Any:
    """Return a representative authenticated application user."""

    return types.SimpleNamespace(
        actor_id="actor-42",
        project_name="feedback-app",
        role="app",
        token_id="token-2",
        app_id="00000000-0000-0000-0000-000000000001",
        actor_type="student",
    )


def _make_feedback_record(feedback_id: str, created_at: str, actor_id: str = "actor-42") -> dict[str, Any]:
    """Build one serialized feedback record for route tests."""

    return {
        "id": feedback_id,
        "session_id": "session-1",
        "app_id": "00000000-0000-0000-0000-000000000001",
        "actor_id": actor_id,
        "feedback_flag": True,
        "feedback_text": "helpful",
        "user_message": "hello",
        "response": "hi there",
        "created_at": created_at,
    }


def _build_rate_limit_state() -> types.SimpleNamespace:
    """Return a minimal rate-limit state object accepted by the route helper."""

    return types.SimpleNamespace(limit=25, remaining=24, retry_after=60)


@pytest.fixture
def proxy_main_module() -> Iterator[Any]:
    """Import ``backend_proxy.main`` behind a fake SQL module and clean it up afterward."""

    _cleanup_test_modules()

    fake_postgres = types.ModuleType("db.postgres")
    fake_postgres.get_session = _fake_async_session
    fake_postgres.get_sync_session = _fake_sync_session
    sys.modules["db.postgres"] = fake_postgres

    importlib.invalidate_caches()
    proxy_main = importlib.import_module("backend_proxy.main")
    try:
        yield proxy_main
    finally:
        _cleanup_test_modules()


@pytest.fixture
def client(proxy_main_module: Any) -> Iterator[TestClient]:
    """Create a fresh FastAPI test client for each feedback route test."""

    app = proxy_main_module.create_app()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_chat_feedback_request_normalizes_text_fields() -> None:
    """Blank optional text should collapse to None and required strings should be trimmed."""

    payload = ChatFeedbackRequest(
        session_id="  session-1  ",
        feedback_flag=True,
        feedback_text="   ",
        user_message="  hello  ",
        response="  hi there  ",
    )

    assert payload.session_id == "session-1"
    assert payload.feedback_flag is True
    assert payload.feedback_text is None
    assert payload.user_message == "hello"
    assert payload.response == "hi there"


def test_chat_feedback_request_rejects_blank_required_fields() -> None:
    """Blank required strings should fail validation."""

    with pytest.raises(ValidationError):
        ChatFeedbackRequest(
            session_id="   ",
            feedback_flag=False,
            feedback_text="needs work",
            user_message="hello",
            response="hi there",
        )


def test_chat_feedback_response_defaults_success() -> None:
    """Feedback responses should default the success flag to true."""

    payload = ChatFeedbackResponse(id="feedback-id-123")

    assert payload.model_dump() == {
        "success": True,
        "id": "feedback-id-123",
    }


def test_chat_feedback_persists_token_actor_id(
    client: TestClient,
    proxy_main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The feedback write route should persist the actor ID from the authenticated token."""

    async def _user_override() -> Any:
        return _build_app_user()

    async def _record_feedback_stub(**kwargs: Any) -> str:
        assert kwargs == {
            "app_id": "00000000-0000-0000-0000-000000000001",
            "actor_id": "actor-42",
            "session_id": "session-1",
            "feedback_flag": True,
            "feedback_text": "helpful",
            "user_message": "hello",
            "response": "hi there",
        }
        return "feedback-123"

    async def _enforce_rate_limit_stub(**kwargs: Any) -> types.SimpleNamespace:
        assert kwargs["subject"] == "actor-42"
        assert kwargs["scope"] == "chat-feedback"
        return _build_rate_limit_state()

    client.app.dependency_overrides[proxy_main_module._get_current_user] = _user_override
    monkeypatch.setattr(proxy_main_module, "record_feedback", _record_feedback_stub)
    monkeypatch.setattr(proxy_main_module, "enforce_rate_limit", _enforce_rate_limit_stub)

    response = client.post(
        "/api/chat/feedback",
        json={
            "session_id": "session-1",
            "feedback_flag": True,
            "feedback_text": "helpful",
            "user_message": "hello",
            "response": "hi there",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "id": "feedback-123"}


def test_feedback_list_returns_paginated_results(
    client: TestClient,
    proxy_main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The list route should preserve pagination semantics and include actor IDs."""

    async def _admin_override() -> Any:
        return _build_admin_user()

    async def _list_feedback_stub(**kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "page": 2,
            "page_size": 3,
            "last": None,
            "actor_id": None,
            "start_date": datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc),
            "end_date": datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc),
        }
        return {
            "items": [
                _make_feedback_record("feedback-2", "2026-01-02T00:00:00+00:00"),
                _make_feedback_record("feedback-1", "2026-01-01T00:00:00+00:00"),
            ],
            "page": 2,
            "page_size": 3,
            "total": 8,
        }

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _admin_override
    monkeypatch.setattr(proxy_main_module, "list_feedback", _list_feedback_stub)

    response = client.get(
        "/api/feedback/all",
        params={
            "page": 2,
            "page_size": 3,
            "start_date": "2025-09-01T00:00:00Z",
            "end_date": "2026-09-01T00:00:00Z",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            _make_feedback_record("feedback-2", "2026-01-02T00:00:00+00:00"),
            _make_feedback_record("feedback-1", "2026-01-01T00:00:00+00:00"),
        ],
        "page": 2,
        "page_size": 3,
        "total": 8,
    }


def test_feedback_list_returns_last_mode_results(
    client: TestClient,
    proxy_main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The list route should support recent-N mode without pagination parameters."""

    async def _admin_override() -> Any:
        return _build_admin_user()

    async def _list_feedback_stub(**kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "page": 1,
            "page_size": 50,
            "last": 2,
            "actor_id": None,
            "start_date": datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc),
            "end_date": datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc),
        }
        return {
            "items": [
                _make_feedback_record("feedback-9", "2026-03-01T00:00:00+00:00"),
                _make_feedback_record("feedback-8", "2026-02-01T00:00:00+00:00"),
            ],
            "page": 1,
            "page_size": 2,
            "total": 2,
        }

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _admin_override
    monkeypatch.setattr(proxy_main_module, "list_feedback", _list_feedback_stub)

    response = client.get(
        "/api/feedback/all",
        params={
            "last": 2,
            "start_date": "2025-09-01T00:00:00Z",
            "end_date": "2026-09-01T00:00:00Z",
        },
    )

    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert [item["id"] for item in response.json()["items"]] == ["feedback-9", "feedback-8"]


def test_feedback_list_filters_by_actor_id(
    client: TestClient,
    proxy_main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The list route should forward the optional actor ID filter."""

    async def _admin_override() -> Any:
        return _build_admin_user()

    async def _list_feedback_stub(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["actor_id"] == "actor-99"
        return {
            "items": [_make_feedback_record("feedback-99", "2026-01-05T00:00:00+00:00", actor_id="actor-99")],
            "page": 1,
            "page_size": 50,
            "total": 1,
        }

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _admin_override
    monkeypatch.setattr(proxy_main_module, "list_feedback", _list_feedback_stub)

    response = client.get("/api/feedback/all", params={"actor_id": " actor-99 "})

    assert response.status_code == 200
    assert response.json()["items"][0]["actor_id"] == "actor-99"


def test_feedback_list_treats_naive_dates_as_utc(
    client: TestClient,
    proxy_main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naive list date filters should be treated as UTC."""

    async def _admin_override() -> Any:
        return _build_admin_user()

    async def _list_feedback_stub(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["start_date"] == datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert kwargs["end_date"] == datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
        return {"items": [], "page": 1, "page_size": 50, "total": 0}

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _admin_override
    monkeypatch.setattr(proxy_main_module, "list_feedback", _list_feedback_stub)

    response = client.get(
        "/api/feedback/all",
        params={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-02T00:00:00",
        },
    )

    assert response.status_code == 200


def test_feedback_list_normalizes_non_utc_dates(
    client: TestClient,
    proxy_main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aware list date filters should be converted to UTC before querying."""

    async def _admin_override() -> Any:
        return _build_admin_user()

    async def _list_feedback_stub(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["start_date"] == datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert kwargs["end_date"] == datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        return {"items": [], "page": 1, "page_size": 50, "total": 0}

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _admin_override
    monkeypatch.setattr(proxy_main_module, "list_feedback", _list_feedback_stub)

    response = client.get(
        "/api/feedback/all",
        params={
            "start_date": "2026-01-01T02:00:00+02:00",
            "end_date": "2026-01-01T06:30:00+05:30",
        },
    )

    assert response.status_code == 200


def test_feedback_list_rejects_last_with_pagination(
    client: TestClient,
    proxy_main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recent-N mode should reject explicit pagination parameters."""

    async def _admin_override() -> Any:
        return _build_admin_user()

    async def _unused_list_feedback_stub(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("list_feedback should not be called for invalid query combos")

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _admin_override
    monkeypatch.setattr(proxy_main_module, "list_feedback", _unused_list_feedback_stub)

    response = client.get("/api/feedback/all", params={"last": 5, "page": 1})

    assert response.status_code == 400
    assert response.json() == {"detail": "last cannot be combined with pagination parameters"}


def test_feedback_list_rejects_invalid_date_range_after_normalization(
    client: TestClient,
    proxy_main_module: Any,
) -> None:
    """Date ordering should be validated after both values are normalized to UTC."""

    async def _admin_override() -> Any:
        return _build_admin_user()

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _admin_override

    response = client.get(
        "/api/feedback/all",
        params={
            "start_date": "2026-01-01T02:00:00+02:00",
            "end_date": "2025-12-31T23:30:00+00:00",
        },
    )

    assert response.status_code == 422


def test_feedback_list_rejects_non_admin(client: TestClient, proxy_main_module: Any) -> None:
    """The list route should remain admin-only."""

    async def _forbidden_override() -> Any:
        raise AdminAuthError(
            "Requires admin privileges",
            code="FORBIDDEN_ROLE",
            status_code=403,
        )

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _forbidden_override

    response = client.get("/api/feedback/all")

    assert response.status_code == 403
    assert response.json() == {"error": "Requires admin privileges", "code": "FORBIDDEN_ROLE"}


def test_feedback_detail_returns_record(
    client: TestClient,
    proxy_main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detail route should return stored feedback records including actor ID."""

    async def _admin_override() -> Any:
        return _build_admin_user()

    async def _get_feedback_stub(feedback_id: str) -> dict[str, Any] | None:
        assert feedback_id == "feedback-123"
        return _make_feedback_record("feedback-123", "2026-01-03T00:00:00+00:00")

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _admin_override
    monkeypatch.setattr(proxy_main_module, "get_feedback_by_id", _get_feedback_stub)

    response = client.get("/api/feedback/feedback-123")

    assert response.status_code == 200
    assert response.json()["id"] == "feedback-123"
    assert response.json()["response"] == "hi there"
    assert response.json()["actor_id"] == "actor-42"


def test_feedback_detail_returns_404_when_missing(
    client: TestClient,
    proxy_main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detail route should return 404 for unknown feedback IDs."""

    async def _admin_override() -> Any:
        return _build_admin_user()

    async def _missing_feedback_stub(feedback_id: str) -> dict[str, Any] | None:
        assert feedback_id == "missing-id"
        return None

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _admin_override
    monkeypatch.setattr(proxy_main_module, "get_feedback_by_id", _missing_feedback_stub)

    response = client.get("/api/feedback/missing-id")

    assert response.status_code == 404
    assert response.json() == {"detail": "Feedback not found"}


def test_feedback_detail_rejects_non_admin(client: TestClient, proxy_main_module: Any) -> None:
    """The detail route should remain admin-only."""

    async def _forbidden_override() -> Any:
        raise AdminAuthError(
            "Requires admin privileges",
            code="FORBIDDEN_ROLE",
            status_code=403,
        )

    client.app.dependency_overrides[proxy_main_module.require_app_admin] = _forbidden_override

    response = client.get("/api/feedback/feedback-123")

    assert response.status_code == 403
    assert response.json() == {"error": "Requires admin privileges", "code": "FORBIDDEN_ROLE"}
