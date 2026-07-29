"""HTTP-level tests for the per-app OTP + SMS auth endpoints.

Each test boots the real FastAPI app behind an in-memory fake Redis + fake DB
session and exercises the OTP request/verify/gate/admin flows end to end,
covering both the happy path and the security invariants (gate enforcement,
fail-closed behavior, credential redaction, SSRF/template rejection, and the
per-app abuse caps).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LLM_BASE_URL", "https://example.com")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_FAST_MODEL", "fast-model")
os.environ.setdefault("LLM_SMART_MODEL", "smart-model")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("PGVECTOR_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("JWT_SECRET", "test-secret")

from cryptography.fernet import Fernet

APP_ID = "00000000-0000-0000-0000-0000000000aa"
_FERNET_KEY = Fernet.generate_key().decode()


# ── In-memory fake Redis (async) ─────────────────────────────────────────────


class FakeRedis:
    """Minimal async Redis emulation covering the OTP + rate-limit ops.

    Emulates the single OTP-verify Lua script in Python so the atomic
    compare/attempt logic is exercised end to end.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.expiry: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        value = int(self.store.get(key, "0")) + 1
        self.store[key] = str(value)
        return value

    async def decr(self, key: str) -> int:
        value = int(self.store.get(key, "0")) - 1
        self.store[key] = str(value)
        return value

    async def expire(self, key: str, seconds: int) -> bool:
        if key in self.store:
            self.expiry[key] = seconds
            return True
        return False

    async def ttl(self, key: str) -> int:
        if key not in self.store:
            return -2
        return self.expiry.get(key, -1)

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.store[key] = value if isinstance(value, str) else str(value)
        if ex is not None:
            self.expiry[key] = ex
        return True

    async def exists(self, key: str) -> int:
        return 1 if key in self.store else 0

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.store:
                del self.store[key]
                self.expiry.pop(key, None)
                removed += 1
        return removed

    async def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        import json as _json

        key = args[0]

        # Emulates backend_proxy.otp_service._DECR_FLOOR_LUA (decrement, floor 0).
        if "DECR" in script:
            value = int(self.store.get(key, "0"))
            if value > 0:
                value -= 1
                self.store[key] = str(value)
            return value

        # Otherwise emulates _VERIFY_LUA.
        supplied_hash = args[1]
        max_attempts = int(args[2])
        raw = self.store.get(key)
        if raw is None:
            return ["OTP_EXPIRED", "0"]
        try:
            data = _json.loads(raw)
        except ValueError:
            await self.delete(key)
            return ["OTP_EXPIRED", "0"]
        attempts = int(data.get("attempts", 0))
        if attempts >= max_attempts:
            await self.delete(key)
            return ["OTP_TOO_MANY_ATTEMPTS", "0"]
        if data.get("hash") == supplied_hash:
            await self.delete(key)
            return ["OK", "0"]
        attempts += 1
        data["attempts"] = attempts
        self.store[key] = _json.dumps(data)
        return ["OTP_INVALID", str(max_attempts - attempts)]


# ── Fake DB session (async) — only what the OTP/admin paths touch ────────────


class FakeAppRecord:
    """Stand-in for the App ORM row used by the async session paths."""

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    @property
    def sms_credentials_encrypted(self) -> str | None:
        return self._state.get("sms_credentials_encrypted")

    @sms_credentials_encrypted.setter
    def sms_credentials_encrypted(self, value: str | None) -> None:
        self._state["sms_credentials_encrypted"] = value


class FakeAsyncSession:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    async def get(self, _model: Any, _pk: Any) -> Any:
        if self._state.get("app_missing"):
            return None
        return FakeAppRecord(self._state)

    def add(self, _obj: Any) -> None:
        pass

    async def flush(self) -> None:
        pass

    async def refresh(self, _obj: Any) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def execute(self, _stmt: Any) -> Any:
        return types.SimpleNamespace(
            scalar_one=lambda: 0,
            scalars=lambda: types.SimpleNamespace(all=lambda: []),
        )


# Shared mutable state the tests mutate to drive the app.
_APP_STATE: dict[str, Any] = {}


@asynccontextmanager
async def _fake_get_session() -> Iterator[Any]:
    yield FakeAsyncSession(_APP_STATE)


@contextmanager
def _fake_get_sync_session() -> Iterator[Any]:
    yield None


_MODULES_TO_RESET = (
    "backend_proxy.main",
    "backend_proxy.rag_routes",
    "backend_proxy.observability_routes",
    "backend_proxy.resolvers",
    "db.postgres",
)


def _cleanup() -> None:
    for name in _MODULES_TO_RESET:
        sys.modules.pop(name, None)
    db_pkg = sys.modules.get("db")
    if db_pkg is not None and hasattr(db_pkg, "postgres"):
        delattr(db_pkg, "postgres")


@pytest.fixture
def harness(monkeypatch) -> Iterator[Any]:
    _cleanup()

    fake_postgres = types.ModuleType("db.postgres")
    fake_postgres.get_session = _fake_get_session
    fake_postgres.get_sync_session = _fake_get_sync_session
    sys.modules["db.postgres"] = fake_postgres

    importlib.invalidate_caches()
    proxy_main = importlib.import_module("backend_proxy.main")

    # Fake Redis singleton.
    import shared.redis_client as redis_client

    fake_redis = FakeRedis()
    monkeypatch.setattr(redis_client, "_client", fake_redis)

    # Encryption key on the live settings object (read at call time).
    from shared.config import settings

    monkeypatch.setattr(settings, "sms_secret_key", _FERNET_KEY, raising=False)

    from db.app_registry import AppContext
    from backend_proxy.auth import AuthenticatedUser

    # Default per-app state.
    _APP_STATE.clear()
    _APP_STATE.update(
        {
            "otp_enabled": True,
            "otp_length": 6,
            "otp_ttl_seconds": 300,
            "otp_max_attempts": 5,
            "otp_resend_cooldown_seconds": 30,
            "otp_max_requests_per_hour": 5,
            "sms_daily_cap": 500,
            "sms_credentials_encrypted": None,
            "app_missing": False,
        }
    )

    def _fake_resolve_by_app_id(app_id: str):
        if not app_id or _APP_STATE.get("app_missing"):
            return None
        return AppContext(
            app_id=app_id,
            app_name="Test App",
            public_key="",
            domain="https://example.com",
            allowed_workflows=["general"],
            allowed_tools=[],
            llm_model_override=None,
            rate_limit_rpm=60,
            token_quota_monthly=None,
            cache_ttl_seconds=3600,
            otp_enabled=_APP_STATE["otp_enabled"],
            caching_enabled=True,
            otp_length=_APP_STATE["otp_length"],
            otp_ttl_seconds=_APP_STATE["otp_ttl_seconds"],
            otp_max_attempts=_APP_STATE["otp_max_attempts"],
            otp_resend_cooldown_seconds=_APP_STATE["otp_resend_cooldown_seconds"],
            otp_max_requests_per_hour=_APP_STATE["otp_max_requests_per_hour"],
            sms_daily_cap=_APP_STATE["sms_daily_cap"],
            is_active=True,
            app_config=None,
            metadata=None,
        )

    monkeypatch.setattr(proxy_main, "resolve_by_app_id", _fake_resolve_by_app_id)

    # Capture the OTP that would be SMS'd, so verify tests can submit it.
    sent: dict[str, Any] = {}

    async def _fake_send(mobile, otp, **kwargs):
        if _APP_STATE.get("send_should_fail"):
            raise RuntimeError("gateway down")
        sent["mobile"] = mobile
        sent["otp"] = otp
        sent["kwargs"] = kwargs

    monkeypatch.setattr(proxy_main, "send_otp_sms", _fake_send)

    # Stub the downstream chat call for the happy-path chat test.
    async def _fake_chat_complete(**kwargs):
        return {"response": "hi", "workflow": "general", "cache_hit": False, "token_usage": {}}

    monkeypatch.setattr(proxy_main, "_call_chat_complete", _fake_chat_complete)

    # Audit persistence uses the sync DB session (faked to None here); stub it.
    recorded: list[dict] = []

    async def _fake_record_app_change(**kwargs):
        recorded.append(kwargs)
        return "audit-id"

    monkeypatch.setattr(proxy_main, "record_app_change", _fake_record_app_change)

    app = proxy_main.create_app()

    # Auth: a mutable current-user holder + dependency overrides.
    holder: dict[str, Any] = {}

    def _guest() -> AuthenticatedUser:
        return AuthenticatedUser(
            actor_id="", project_name="Test App", role="app", token_id="t",
            app_id=APP_ID, actor_type="student", phone="", phone_verified=False,
        )

    def _verified_guest() -> AuthenticatedUser:
        return AuthenticatedUser(
            actor_id="", project_name="Test App", role="app", token_id="t",
            app_id=APP_ID, actor_type="student", phone="9998887777", phone_verified=True,
        )

    def _admin() -> AuthenticatedUser:
        return AuthenticatedUser(
            actor_id="admin-1", project_name="admin", role="admin", token_id="t",
            app_id=APP_ID, actor_type="admin",
        )

    holder["user"] = _guest()
    app.dependency_overrides[proxy_main._get_current_user] = lambda: holder["user"]
    app.dependency_overrides[proxy_main.require_admin_user] = lambda: _admin()
    app.dependency_overrides[proxy_main.require_admin_hybrid] = lambda: _admin()
    app.dependency_overrides[proxy_main.get_admin_scope] = lambda: None

    with TestClient(app, raise_server_exceptions=False) as client:
        yield types.SimpleNamespace(
            client=client,
            state=_APP_STATE,
            sent=sent,
            holder=holder,
            guest=_guest,
            verified_guest=_verified_guest,
            admin=_admin,
            fernet_key=_FERNET_KEY,
        )

    _cleanup()


def _store_creds(user_id="lpude", password="p@ss", api_url=None, template=None) -> str:
    from backend_proxy.sms_secrets import encrypt_sms_credentials

    creds = {"user_id": user_id, "password": password}
    if api_url:
        creds["api_url"] = api_url
    if template:
        creds["message_template"] = template
    return encrypt_sms_credentials(creds)


# ── OTP gate blocks unverified guests ────────────────────────────────────


def test_unverified_guest_cannot_chat(harness):
    harness.holder["user"] = harness.guest()
    resp = harness.client.post("/api/chat/message", json={"message": "hello"})
    assert resp.status_code == 403
    assert resp.json().get("code") == "PHONE_VERIFICATION_REQUIRED"


# ── reserved phone: session_id is rejected ───────────────────────────────


def test_client_cannot_squat_phone_session(harness):
    harness.holder["user"] = harness.verified_guest()
    resp = harness.client.post(
        "/api/chat/message",
        json={"message": "hi", "session_id": "phone:9111111111"},
    )
    assert resp.status_code == 400


def test_own_phone_session_round_trips(harness):
    # A verified guest gets session_id="phone:<their-num>" back and echoes it on
    # the next turn — their OWN key must be accepted (not a foreign IDOR attempt).
    harness.holder["user"] = harness.verified_guest()  # phone 9998887777
    first = harness.client.post("/api/chat/message", json={"message": "hi"})
    assert first.status_code == 200
    returned = first.json()["session_id"]
    assert returned == "phone:9998887777"
    second = harness.client.post(
        "/api/chat/message", json={"message": "again", "session_id": returned}
    )
    assert second.status_code == 200


# ── OTP endpoints refuse when the app has OTP disabled ────────────────────


def test_otp_request_refused_when_disabled(harness):
    harness.state["otp_enabled"] = False
    resp = harness.client.post("/api/auth/otp/request", json={"phone": "9876543210"})
    assert resp.status_code == 403


# ── no shared "default" namespace when app_id is empty ───────────────────


def test_empty_app_id_refused(harness):
    from backend_proxy.auth import AuthenticatedUser

    harness.holder["user"] = AuthenticatedUser(
        actor_id="", project_name="x", role="app", token_id="t",
        app_id="", actor_type="student",
    )
    resp = harness.client.post("/api/auth/otp/request", json={"phone": "9876543210"})
    assert resp.status_code == 403


# ── admin write-boundary validation ─────────────────────────────────


def test_template_injection_rejected(harness):
    resp = harness.client.put(
        f"/api/admin/apps/{APP_ID}/sms-credentials",
        json={
            "user_id": "u",
            "password": "p",
            "message_template": "{otp.__class__.__init__.__globals__}",
        },
    )
    # message_template is validated in the request schema alongside api_url (422).
    assert resp.status_code == 422


def test_out_of_range_otp_knobs_rejected(harness):
    for field, bad in [
        ("otp_length", 1),
        ("otp_ttl_seconds", 10 ** 9),
        ("otp_max_attempts", 9999),
        ("otp_resend_cooldown_seconds", 0),
    ]:
        resp = harness.client.patch(f"/api/admin/apps/{APP_ID}", json={field: bad})
        assert resp.status_code == 422, f"{field}={bad} should be rejected"


# ── fail-closed when misconfigured ──────────────────────────────────


def test_fail_closed_without_encryption_key(harness, monkeypatch):
    from shared.config import settings

    # Store a real ciphertext (under the harness key), THEN drop the key. The
    # blob is non-None, so decrypt reaches _fernet() and must fail closed on the
    # missing key — exercising the no-key path (distinct from R8's no-creds).
    harness.state["sms_credentials_encrypted"] = _store_creds()
    monkeypatch.setattr(settings, "sms_secret_key", None, raising=False)
    resp = harness.client.post("/api/auth/otp/request", json={"phone": "9876543210"})
    assert resp.status_code == 502
    # Never sent, never bypassed.
    assert "otp" not in harness.sent


def test_fail_closed_without_stored_credentials(harness):
    harness.state["sms_credentials_encrypted"] = None
    resp = harness.client.post("/api/auth/otp/request", json={"phone": "9876543210"})
    assert resp.status_code == 502
    assert "otp" not in harness.sent


# ── SMS credentials are stored encrypted, redacted, never echoed ─────────


def test_sms_credentials_write_is_encrypted_and_redacted(harness):
    resp = harness.client.put(
        f"/api/admin/apps/{APP_ID}/sms-credentials",
        json={"user_id": "lpude-secret-acct", "password": "sup3r-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["configured"] is True
    assert "sup3r-secret" not in str(body)
    assert body["user_id_hint"].endswith("acct")
    stored = harness.state["sms_credentials_encrypted"]
    assert stored and "sup3r-secret" not in stored  # ciphertext, not plaintext


# ── per-app SMS daily budget cap + rollback on failure ──────────────────


def test_sms_budget_cap_blocks_abuse(harness):
    harness.state["sms_credentials_encrypted"] = _store_creds()
    harness.state["sms_daily_cap"] = 2
    # 2 distinct numbers succeed, the 3rd is blocked by the daily budget.
    assert harness.client.post("/api/auth/otp/request", json={"phone": "9000000001"}).status_code == 200
    assert harness.client.post("/api/auth/otp/request", json={"phone": "9000000002"}).status_code == 200
    third = harness.client.post("/api/auth/otp/request", json={"phone": "9000000003"})
    assert third.status_code == 429


def test_budget_refunded_on_send_failure(harness):
    harness.state["sms_credentials_encrypted"] = _store_creds()
    harness.state["sms_daily_cap"] = 1
    harness.state["send_should_fail"] = True
    # Send fails -> 502 and the budget unit is refunded, so a later good send works.
    assert harness.client.post("/api/auth/otp/request", json={"phone": "9000000001"}).status_code == 502
    harness.state["send_should_fail"] = False
    assert harness.client.post("/api/auth/otp/request", json={"phone": "9000000002"}).status_code == 200


# ── sustained verify lockout across OTP regenerations ───────────────────


def test_sustained_verify_lockout(harness):
    harness.state["sms_credentials_encrypted"] = _store_creds()
    # threshold = max(max_attempts * max_requests_per_hour, 10) = max(5*2, 10) = 10,
    # and 2 requests * 5 live-invalid guesses = 10 failures makes it reachable.
    harness.state["otp_max_attempts"] = 5
    harness.state["otp_max_requests_per_hour"] = 2

    import shared.redis_client as _rc

    # Real cross-regeneration brute force: request a fresh OTP, burn its 5
    # attempts guessing wrong, repeat. Each wrong guess is against a LIVE OTP.
    for _round in range(2):
        _rc._client.store.pop(f"otp:cooldown:{APP_ID}:9123456780", None)
        harness.client.post("/api/auth/otp/request", json={"phone": "9123456780"})
        for _guess in range(5):
            harness.client.post(
                "/api/auth/otp/verify", json={"phone": "9123456780", "otp": "000000"}
            )

    # 10 sustained failures accrued → the next verify is locked out (429).
    locked = False
    for _ in range(3):
        r = harness.client.post(
            "/api/auth/otp/verify", json={"phone": "9123456780", "otp": "000000"}
        )
        if r.status_code == 429:
            locked = True
            break
    assert locked, "cross-regeneration wrong guesses against LIVE OTPs should lock verification"


def test_verify_without_request_cannot_lock_victim(harness):
    # Verifying a number WITHOUT ever requesting an OTP must NOT accrue lockout
    # failures (OTP_EXPIRED is not counted), so a number cannot be locked out
    # with no SMS ever sent.
    harness.state["sms_credentials_encrypted"] = _store_creds()
    harness.state["otp_max_requests_per_hour"] = 1  # threshold floor 10
    for _ in range(20):
        r = harness.client.post(
            "/api/auth/otp/verify",
            json={"phone": "9445556666", "otp": "000000"},
        )
        # Always a plain 400 (expired/invalid), never a 429 lockout.
        assert r.status_code == 400
    # Victim's counter must still be clear: a fresh request + correct OTP works.
    assert harness.client.post("/api/auth/otp/request", json={"phone": "9445556666"}).status_code == 200
    real = harness.sent["otp"]
    assert harness.client.post(
        "/api/auth/otp/verify", json={"phone": "9445556666", "otp": real}
    ).status_code == 200


# ── happy path + app-scoped keys ────────────────────────────────────────


def test_happy_path_request_verify_issues_tokens(harness):
    harness.state["sms_credentials_encrypted"] = _store_creds(template="{otp} is your code")
    req = harness.client.post("/api/auth/otp/request", json={"phone": "9123456789"})
    assert req.status_code == 200
    assert req.json()["expires_in"] == 300
    otp = harness.sent["otp"]
    assert len(otp) == 6

    verify = harness.client.post(
        "/api/auth/otp/verify",
        json={"phone": "9123456789", "otp": otp},
    )
    assert verify.status_code == 200
    body = verify.json()
    assert body["phone_verified"] is True
    assert body["access_token"] and body["refresh_token"]


# ── api_url SSRF / exfiltration sink is rejected at write time ───────────


def test_ssrf_api_url_rejected(harness):
    # Non-https and non-allowlisted (internal/metadata) hosts must be refused
    # before the credentials are ever stored.
    for bad_url in [
        "http://enterprise.smsgupshup.com/rest",   # not https
        "https://169.254.169.254/latest/meta-data",  # metadata host, not allowlisted
        "https://evil.example/collect",              # arbitrary host, not allowlisted
    ]:
        resp = harness.client.put(
            f"/api/admin/apps/{APP_ID}/sms-credentials",
            json={"user_id": "u", "password": "p", "api_url": bad_url},
        )
        assert resp.status_code == 422, f"{bad_url} should be rejected"
    # The default allowlisted host is accepted (no api_url → platform default).
    ok = harness.client.put(
        f"/api/admin/apps/{APP_ID}/sms-credentials",
        json={"user_id": "u", "password": "p"},
    )
    assert ok.status_code == 200


# ── deactivated / unresolvable app fails CLOSED (gate not dropped) ───────


def test_deactivated_app_fails_closed(harness):
    # An unverified guest holding a live token for an app that can no longer be
    # resolved (deactivated/deleted) must be DENIED chat, not let through.
    harness.holder["user"] = harness.guest()
    harness.state["app_missing"] = True
    resp = harness.client.post("/api/chat/message", json={"message": "hi"})
    assert resp.status_code == 403
    assert resp.json().get("code") in ("APP_UNAVAILABLE", "PHONE_VERIFICATION_REQUIRED")


# ── forced send failure keeps the per-mobile cooldown (no throttle bypass)


def test_forced_failure_keeps_cooldown(harness):
    harness.state["sms_credentials_encrypted"] = _store_creds()
    harness.state["send_should_fail"] = True
    first = harness.client.post("/api/auth/otp/request", json={"phone": "9333222111"})
    assert first.status_code == 502
    # Cooldown must still be in effect — a gateway failure does not refund it.
    second = harness.client.post("/api/auth/otp/request", json={"phone": "9333222111"})
    assert second.status_code == 429


def test_wrong_otp_is_rejected(harness):
    harness.state["sms_credentials_encrypted"] = _store_creds()
    harness.client.post("/api/auth/otp/request", json={"phone": "9123456700"})
    real = harness.sent["otp"]
    wrong = "111111" if real != "111111" else "222222"
    verify = harness.client.post(
        "/api/auth/otp/verify", json={"phone": "9123456700", "otp": wrong}
    )
    assert verify.status_code == 400
