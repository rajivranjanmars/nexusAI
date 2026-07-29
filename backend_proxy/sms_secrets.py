"""Encryption-at-rest for per-app SMS gateway credentials.

Per-app SMS credentials (e.g. a tenant's Gupshup user id + password) are stored
as a single Fernet-encrypted JSON blob in ``apps.sms_credentials_encrypted``.
The ciphertext lives in its own column and is decrypted **only** at OTP-send
time — it is never placed in ``AppContext``, ``app_config``, ``metadata``, any
API response, or the audit log.

Design decisions (see the security spec):
- Keyed by ``settings.sms_secret_key`` (a dedicated Fernet key), NOT ``jwt_secret``
  (which already double-duties as the OTP HMAC key). Separate keys so rotating
  one never invalidates the other.
- Supports key rotation: ``sms_secret_key`` may be a comma-separated list; the
  first key encrypts, all keys decrypt (``MultiFernet``).
- Fails **closed**: a missing key or an undecryptable blob raises
  ``SecretUnavailable`` so the caller aborts the OTP send (502) rather than
  sending with empty credentials or silently skipping verification.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from shared.config import settings

# Non-secret keys allowed alongside the secret credentials in the blob.
_ALLOWED_CRED_KEYS = {"user_id", "password", "api_url", "message_template"}


class SecretUnavailable(RuntimeError):
    """Raised when SMS credentials cannot be encrypted or decrypted.

    Signals a fail-closed condition: the encryption key is unset, or a stored
    blob is corrupt / encrypted under a key that is no longer configured.
    """


def sms_encryption_configured() -> bool:
    """Whether a usable SMS-secret encryption key is configured."""
    return bool((settings.sms_secret_key or "").strip())


def _keys() -> List[str]:
    raw = (settings.sms_secret_key or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _fernet() -> MultiFernet:
    """Build a MultiFernet from the configured key(s).

    Raises:
        SecretUnavailable: If no key is configured or a key is malformed.
    """
    keys = _keys()
    if not keys:
        raise SecretUnavailable("sms_secret_key is not configured")
    try:
        return MultiFernet([Fernet(k.encode("utf-8")) for k in keys])
    except (ValueError, TypeError) as exc:
        # A malformed key must not leak its value into logs/exceptions.
        raise SecretUnavailable("sms_secret_key is malformed") from exc


def encrypt_sms_credentials(credentials: Dict[str, Any]) -> str:
    """Encrypt a per-app SMS credential dict into a single Fernet token.

    Args:
        credentials: Plaintext dict with at least ``user_id`` and ``password``;
            may also carry non-secret ``api_url`` / ``message_template``.

    Returns:
        A urlsafe Fernet token string suitable for storing in the DB column.

    Raises:
        SecretUnavailable: If the encryption key is not configured.
        ValueError: If required fields are missing.
    """
    user_id = str(credentials.get("user_id", "")).strip()
    password = str(credentials.get("password", "")).strip()
    if not user_id or not password:
        raise ValueError("SMS credentials require both user_id and password")

    payload = {k: v for k, v in credentials.items() if k in _ALLOWED_CRED_KEYS and v is not None}
    token = _fernet().encrypt(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return token.decode("utf-8")


def decrypt_sms_credentials(ciphertext: Optional[str]) -> Dict[str, Any]:
    """Decrypt a stored SMS-credential blob.

    Args:
        ciphertext: The stored Fernet token, or ``None``/empty.

    Returns:
        The plaintext credential dict.

    Raises:
        SecretUnavailable: If the key is unset, the blob is empty, or it cannot
            be decrypted (corruption / rotated-away key).
    """
    if not ciphertext:
        raise SecretUnavailable("no SMS credentials configured for this app")
    try:
        raw = _fernet().decrypt(ciphertext.encode("utf-8"))
    except InvalidToken as exc:
        raise SecretUnavailable("SMS credentials could not be decrypted") from exc
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise SecretUnavailable("SMS credentials blob is not valid JSON") from exc
    if not isinstance(data, dict):
        raise SecretUnavailable("SMS credentials blob has an unexpected shape")
    return data


def mask_user_id(user_id: Optional[str]) -> str:
    """Return a non-reversible hint of a credential's user id for admin display."""
    value = (user_id or "").strip()
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return f"…{value[-4:]}"
