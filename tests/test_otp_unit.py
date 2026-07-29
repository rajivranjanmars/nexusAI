"""Security unit tests for the per-app OTP / SMS configuration.

These cover the pure, dependency-free logic that guards the OTP feature:
- SMS credential encryption round-trips and fails closed (no key / bad blob).
- The message template rejects Python format-string abuse.
- OTP policy knobs are clamped to hard server-side bounds.
- The browser-facing app_config strips secret-shaped keys and never leaks
  credentials while exposing only ``features.otp_required``.
"""

from __future__ import annotations

import importlib
import os

import pytest

os.environ.setdefault("LLM_BASE_URL", "https://example.com")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_FAST_MODEL", "fast-model")
os.environ.setdefault("LLM_SMART_MODEL", "smart-model")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("PGVECTOR_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("JWT_SECRET", "test-secret")

from cryptography.fernet import Fernet

from orchestration.sms_service import (
    DEFAULT_OTP_MESSAGE_TEMPLATE,
    TemplateError,
    _render_message,
    validate_message_template,
)
from backend_proxy import otp_service


def _reload_sms_secrets(monkeypatch, key: str | None):
    """Reload shared.config + sms_secrets with SMS_SECRET_KEY set/unset."""
    if key is None:
        monkeypatch.delenv("SMS_SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("SMS_SECRET_KEY", key)
    import shared.config as config_module

    importlib.reload(config_module)
    import backend_proxy.sms_secrets as sms_secrets

    importlib.reload(sms_secrets)
    return sms_secrets


# ── SMS credential encryption ────────────────────────────────────────────────


def test_sms_credentials_round_trip(monkeypatch):
    sms_secrets = _reload_sms_secrets(monkeypatch, Fernet.generate_key().decode())
    creds = {
        "user_id": "lpude",
        "password": "s3cr3t",
        "api_url": "https://gw.example/rest",
        "message_template": "{otp} is your code",
    }
    token = sms_secrets.encrypt_sms_credentials(creds)
    assert token.startswith("gAAAA")  # Fernet token
    assert "s3cr3t" not in token
    assert sms_secrets.decrypt_sms_credentials(token) == creds


def test_sms_credentials_require_user_and_password(monkeypatch):
    sms_secrets = _reload_sms_secrets(monkeypatch, Fernet.generate_key().decode())
    with pytest.raises(ValueError):
        sms_secrets.encrypt_sms_credentials({"user_id": "only-user"})


def test_sms_credentials_fail_closed_without_key(monkeypatch):
    sms_secrets = _reload_sms_secrets(monkeypatch, None)
    assert sms_secrets.sms_encryption_configured() is False
    with pytest.raises(sms_secrets.SecretUnavailable):
        sms_secrets.encrypt_sms_credentials({"user_id": "u", "password": "p"})
    with pytest.raises(sms_secrets.SecretUnavailable):
        sms_secrets.decrypt_sms_credentials("gAAAAAanything")


def test_sms_credentials_fail_closed_on_corrupt_blob(monkeypatch):
    sms_secrets = _reload_sms_secrets(monkeypatch, Fernet.generate_key().decode())
    with pytest.raises(sms_secrets.SecretUnavailable):
        sms_secrets.decrypt_sms_credentials("not-a-valid-token")
    with pytest.raises(sms_secrets.SecretUnavailable):
        sms_secrets.decrypt_sms_credentials(None)


def test_sms_credentials_rotation(monkeypatch):
    """A blob encrypted under the old key still decrypts after rotating."""
    old_key = Fernet.generate_key().decode()
    sms_secrets = _reload_sms_secrets(monkeypatch, old_key)
    token = sms_secrets.encrypt_sms_credentials({"user_id": "u", "password": "p"})

    new_key = Fernet.generate_key().decode()
    sms_secrets = _reload_sms_secrets(monkeypatch, f"{new_key},{old_key}")
    assert sms_secrets.decrypt_sms_credentials(token)["password"] == "p"


def test_mask_user_id(monkeypatch):
    sms_secrets = _reload_sms_secrets(monkeypatch, Fernet.generate_key().decode())
    assert sms_secrets.mask_user_id("lpude-account") == "…ount"
    assert sms_secrets.mask_user_id("ab") == "**"
    assert sms_secrets.mask_user_id("") == ""


# ── SMS message-template safety ──────────────────────────────────────────────


def test_template_accepts_single_otp_placeholder():
    assert validate_message_template("Your code is {otp}. Do not share.")
    assert validate_message_template(DEFAULT_OTP_MESSAGE_TEMPLATE)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "no placeholder here",
        "{otp} and {otp} twice",
        "{otp} and {name}",
        "{otp.__class__.__init__.__globals__}",  # format-string traversal
        "{0}",
        "{}",
    ],
)
def test_template_rejects_unsafe(bad):
    with pytest.raises(TemplateError):
        validate_message_template(bad)


def test_template_render_uses_literal_replacement():
    # Even if an attacker-shaped value reached the OTP, str.replace never
    # interprets format tokens.
    assert _render_message("Code: {otp}", "123456") == "Code: 123456"


# ── OTP policy clamping ──────────────────────────────────────────────────────


def test_clamp_otp_policy_bounds_extremes():
    p = otp_service.clamp_otp_policy(
        length=1,
        ttl_seconds=10 ** 9,
        max_attempts=10_000,
        resend_cooldown_seconds=0,
        max_requests_per_hour=10_000,
        sms_daily_cap=-1,
    )
    assert p.length == 6  # never below the 6-digit floor
    assert p.ttl_seconds == 600
    assert p.max_attempts == 10
    assert p.resend_cooldown_seconds == 30
    assert p.max_requests_per_hour == 20
    assert p.sms_daily_cap == 1


def test_clamp_otp_policy_defaults_on_none_or_garbage():
    p = otp_service.clamp_otp_policy()
    assert p.length == 6
    assert p.ttl_seconds == 300
    assert p.max_attempts == 5
    p2 = otp_service.clamp_otp_policy(length="not-a-number")
    assert p2.length == 6


def test_verify_lockout_threshold_has_floor():
    p = otp_service.clamp_otp_policy(max_attempts=3, max_requests_per_hour=1)
    # 3 * 1 = 3, floored to 10.
    assert p.verify_lockout_threshold == 10


# ── Browser-facing app_config never leaks secrets ────────────────────────────


def test_client_safe_app_config_strips_secrets_and_adds_flag():
    from backend_proxy.main import _client_safe_app_config

    raw = {
        "theme": "dark",
        "header_title": "LPU",
        "data_resolvers": [{"resolver": "student_profile"}],
        "gupshup_password": "should-never-appear",
        "some_api_key": "nope",
        "features": {"existing": True},
    }
    out = _client_safe_app_config(raw, otp_required=True)
    assert out is not None
    flat = str(out).lower()
    assert "should-never-appear" not in flat
    assert "gupshup" not in flat
    assert "data_resolvers" not in out
    assert "api_key" not in flat
    assert out["theme"] == "dark"
    assert out["features"]["otp_required"] is True
    assert out["features"]["existing"] is True


def test_client_safe_app_config_non_otp_app_stays_none():
    from backend_proxy.main import _client_safe_app_config

    assert _client_safe_app_config({}, otp_required=False) is None
    # A non-OTP app with UI config gets features.otp_required=False, no secrets.
    out = _client_safe_app_config({"theme": "light"}, otp_required=False)
    assert out["features"]["otp_required"] is False


def test_client_safe_app_config_tolerates_non_dict_features():
    from backend_proxy.main import _client_safe_app_config

    # An admin-stored non-dict `features` must not crash token issuance.
    for bad in ["on", ["a", "b"], 5, True]:
        out = _client_safe_app_config({"theme": "x", "features": bad}, otp_required=True)
        assert out["features"]["otp_required"] is True


def test_client_safe_app_config_strips_broad_secret_shapes():
    from backend_proxy.main import _client_safe_app_config

    raw = {
        "theme": "dark",
        "access_token": "leak-me",
        "auth_jwt": "leak-me",
        "bearer_value": "leak-me",
        "signing_key": "leak-me",
        "session_id": "leak-me",
        "user_passwd": "leak-me",
    }
    out = _client_safe_app_config(raw, otp_required=False)
    flat = str(out).lower()
    assert "leak-me" not in flat
    assert out.get("theme") == "dark"


# ── httpx/httpcore request logging is silenced (no secret/OTP in URL logs) ────


def test_client_http_loggers_are_silenced():
    import logging

    from shared.logger import get_logger

    get_logger("warmup")  # ensures root config ran
    for name in ("httpx", "httpcore", "urllib3", "requests"):
        assert logging.getLogger(name).level == logging.WARNING
