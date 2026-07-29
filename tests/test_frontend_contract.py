"""Frontend (chatbot-frontend widget) ↔ backend contract tests.

These replay the EXACT HTTP sequence the widget performs (see
chatbot-frontend/src/runtime/core/backend-api.ts) against the real backend app
and assert every response field the widget's mappers read is present and
correctly shaped. If the backend response contract drifts from what the widget
parses, one of these fails.

Widget call sequence:
  1. bootstrap   POST /api/auth/app-user-token {signed_token, force_workflow}
                 -> reads app_config.features.otp_required (gate decision)
  2. otp request POST /api/auth/otp/request  (Bearer) {phone}
                 -> reads success / expires_in / resend_after
  3. otp verify  POST /api/auth/otp/verify   (Bearer) {phone, otp}
                 -> reads access_token / refresh_token / expires_in /
                    refresh_expires_in / phone / phone_verified
  4. chat        POST /api/chat/message      (Bearer) {message}   # NO session_id
                 -> reads response / workflow / cache_hit
"""

from __future__ import annotations

import backend_proxy.main  # noqa: F401  (ensures module import path is warm)

# Reuse the in-memory harness (fake Redis + fake DB session + dep overrides).
from tests.test_otp_api import (  # noqa: F401  (harness is a pytest fixture)
    APP_ID,
    _store_creds,
    harness,
)


# ── 1. Bootstrap: the widget's only per-app gate signal ──────────────────────


def _install_fake_bootstrap(monkeypatch, *, otp_enabled: bool, app_config: dict):
    """Make /api/auth/app-user-token resolve to a crafted app without real PKI."""
    import backend_proxy.main as proxy_main
    from backend_proxy.auth import AuthenticatedUser
    from db.app_registry import AppContext

    user = AuthenticatedUser(
        actor_id="", project_name="Test App", role="app", token_id="t",
        app_id=APP_ID, actor_type="student",
        allowed_workflows=("general",), allowed_tools=(),
    )
    app_ctx = AppContext(
        app_id=APP_ID, app_name="Test App", public_key="", domain="https://example.com",
        allowed_workflows=["general"], allowed_tools=[], llm_model_override=None,
        rate_limit_rpm=60, token_quota_monthly=None, cache_ttl_seconds=3600,
        otp_enabled=otp_enabled, caching_enabled=True,
        otp_length=6, otp_ttl_seconds=300, otp_max_attempts=5,
        otp_resend_cooldown_seconds=30, otp_max_requests_per_hour=5, sms_daily_cap=500,
        is_active=True, app_config=app_config, metadata=None,
    )

    async def _fake_auth(signed_token, request_domain):
        return user, app_ctx

    async def _fake_resolvers(includes, actor_id=None):
        return {}

    monkeypatch.setattr(proxy_main, "authenticate_app_jwt", _fake_auth)
    monkeypatch.setattr(proxy_main, "run_resolvers", _fake_resolvers)


def test_bootstrap_exposes_otp_required_and_hides_secrets(harness, monkeypatch):
    # app_config carries a normal UI key AND (hypothetically) a secret-shaped key.
    _install_fake_bootstrap(
        monkeypatch,
        otp_enabled=True,
        app_config={"theme": "dark", "gupshup_password": "must-never-leak", "data_resolvers": []},
    )
    resp = harness.client.post(
        "/api/auth/app-user-token",
        json={"signed_token": "x.y.z", "force_workflow": ""},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Widget-required token fields.
    for field in ("access_token", "refresh_token", "expires_in", "refresh_expires_in"):
        assert field in body, f"missing {field}"
    # The one per-app gate signal the widget reads.
    assert body["app_config"]["features"]["otp_required"] is True
    # UI config preserved; secrets and internal keys never serialized.
    assert body["app_config"]["theme"] == "dark"
    flat = str(body).lower()
    assert "must-never-leak" not in flat
    assert "gupshup" not in flat
    assert "data_resolvers" not in body["app_config"]


def test_bootstrap_otp_not_required_when_disabled(harness, monkeypatch):
    _install_fake_bootstrap(monkeypatch, otp_enabled=False, app_config={"theme": "light"})
    resp = harness.client.post(
        "/api/auth/app-user-token", json={"signed_token": "x.y.z"}
    )
    assert resp.status_code == 200
    # features.otp_required must be strictly False so the widget skips the gate.
    assert resp.json()["app_config"]["features"]["otp_required"] is False


# ── 2. OTP request response contract ─────────────────────────────────────────


def test_otp_request_response_contract(harness):
    harness.state["sms_credentials_encrypted"] = _store_creds()
    resp = harness.client.post("/api/auth/otp/request", json={"phone": "9800000001"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert isinstance(body["expires_in"], int) and body["expires_in"] > 0
    assert isinstance(body["resend_after"], int) and body["resend_after"] >= 0


# ── 3. OTP verify response contract (widget swaps to these tokens) ───────────


def test_otp_verify_response_contract(harness):
    harness.state["sms_credentials_encrypted"] = _store_creds()
    harness.client.post("/api/auth/otp/request", json={"phone": "9800000002"})
    otp = harness.sent["otp"]
    resp = harness.client.post(
        "/api/auth/otp/verify", json={"phone": "9800000002", "otp": otp}
    )
    assert resp.status_code == 200
    body = resp.json()
    for field in ("access_token", "refresh_token", "expires_in", "refresh_expires_in"):
        assert body.get(field), f"missing {field}"
    assert body["phone"] == "9800000002"
    assert body["phone_verified"] is True


# ── 4. Chat with the widget's EXACT body ({message} only, no session_id) ─────


def test_chat_widget_body_contract(harness):
    # The widget sends only {"message": ...}; the backend derives the session
    # from the phone-verified token (phone_session_id). Must not require session_id.
    harness.holder["user"] = harness.verified_guest()
    resp = harness.client.post("/api/chat/message", json={"message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert "response" in body
    assert "workflow" in body
    assert "cache_hit" in body
    # Server-derived phone-scoped session for continuity.
    assert body["session_id"] == "phone:9998887777"


def test_chat_gate_blocks_unverified_guest_for_widget(harness):
    # An unverified guest on an OTP-enabled app is blocked (403) — the widget
    # then shows its phone screen based on features.otp_required. Widget refreshes
    # + retries on 403; the retry also 403s (still unverified), so it surfaces.
    harness.holder["user"] = harness.guest()
    resp = harness.client.post("/api/chat/message", json={"message": "hello"})
    assert resp.status_code == 403
    assert resp.json().get("code") == "PHONE_VERIFICATION_REQUIRED"
