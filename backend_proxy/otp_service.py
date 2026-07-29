"""OTP generation, storage, and verification backed by Redis.

OTP codes are never stored in plaintext: only an HMAC (keyed by the proxy JWT
secret and bound to the app + mobile number) is persisted, so a Redis dump
cannot be replayed directly. Codes expire after a short TTL and a wrong-guess
counter caps brute-force attempts.

All behavior knobs (length, TTL, attempt/cooldown/rate caps, daily SMS budget)
are **per-app**: the caller resolves an :class:`OtpPolicy` from the app registry
and passes it in. There are no module-level env globals — a stale import can
never silently apply the wrong tenant's policy. Every value is clamped to a hard
server-side range (:func:`clamp_otp_policy`) so an admin cannot weaken OTP
security (e.g. length 1, a month-long TTL, or a zero cooldown).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass

from shared.config import settings
from shared.logger import get_logger
from shared.redis_client import get_client

logger = get_logger(__name__)

# ── Hard server-side bounds (min, max) — admin config is clamped to these ────
OTP_LENGTH_BOUNDS = (6, 8)
OTP_TTL_SECONDS_BOUNDS = (60, 600)
OTP_MAX_ATTEMPTS_BOUNDS = (3, 10)
OTP_RESEND_COOLDOWN_SECONDS_BOUNDS = (30, 600)
OTP_MAX_REQUESTS_PER_HOUR_BOUNDS = (1, 20)
SMS_DAILY_CAP_BOUNDS = (1, 100_000)

# Defaults used when a stored value is missing/garbage (match the App columns).
_OTP_DEFAULTS = {
    "length": 6,
    "ttl_seconds": 300,
    "max_attempts": 5,
    "resend_cooldown_seconds": 30,
    "max_requests_per_hour": 5,
    "sms_daily_cap": 500,
}

# Sustained failed-verify lockout floor: the per-mobile failure counter spans
# OTP regenerations, so re-requesting a fresh OTP does not reset it.
_VERIFY_LOCKOUT_FLOOR = 10

# Atomically decrement a counter but never below zero, so a refund/release can
# never drive a rolling-window counter negative (which would re-open a cap).
_DECR_FLOOR_LUA = """
local v = tonumber(redis.call('GET', KEYS[1]) or '0')
if v > 0 then return redis.call('DECR', KEYS[1]) end
return v
"""


async def _decr_floor(key: str) -> None:
    """Decrement ``key`` by one, flooring at zero."""
    client = await get_client()
    await client.eval(_DECR_FLOOR_LUA, 1, key)


@dataclass(frozen=True)
class OtpPolicy:
    """Per-app OTP behavior, already clamped to safe bounds."""

    length: int
    ttl_seconds: int
    max_attempts: int
    resend_cooldown_seconds: int
    max_requests_per_hour: int
    sms_daily_cap: int

    @property
    def verify_lockout_threshold(self) -> int:
        """Sustained wrong-guess ceiling per mobile per rolling hour.

        Set to the theoretical maximum a legitimate user could produce
        (every requested OTP fully exhausted), floored so it is never tiny.
        Exceeding it means brute force across OTP regenerations.
        """
        return max(self.max_attempts * self.max_requests_per_hour, _VERIFY_LOCKOUT_FLOOR)


def _clamp(value: object, bounds: tuple[int, int], default: int) -> int:
    """Coerce ``value`` to int and clamp into ``bounds`` (fallback ``default``)."""
    lo, hi = bounds
    try:
        ivalue = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        ivalue = default
    return max(lo, min(hi, ivalue))


def clamp_otp_policy(
    *,
    length: object = None,
    ttl_seconds: object = None,
    max_attempts: object = None,
    resend_cooldown_seconds: object = None,
    max_requests_per_hour: object = None,
    sms_daily_cap: object = None,
) -> OtpPolicy:
    """Build an :class:`OtpPolicy`, clamping every field to its hard bounds.

    ``None`` inputs fall back to the safe defaults. This is the single choke
    point for OTP policy: both the admin write path (validation) and the runtime
    read path funnel through it, so no unbounded value ever reaches Redis.
    """
    d = _OTP_DEFAULTS
    return OtpPolicy(
        length=_clamp(length if length is not None else d["length"], OTP_LENGTH_BOUNDS, d["length"]),
        ttl_seconds=_clamp(ttl_seconds if ttl_seconds is not None else d["ttl_seconds"], OTP_TTL_SECONDS_BOUNDS, d["ttl_seconds"]),
        max_attempts=_clamp(max_attempts if max_attempts is not None else d["max_attempts"], OTP_MAX_ATTEMPTS_BOUNDS, d["max_attempts"]),
        resend_cooldown_seconds=_clamp(resend_cooldown_seconds if resend_cooldown_seconds is not None else d["resend_cooldown_seconds"], OTP_RESEND_COOLDOWN_SECONDS_BOUNDS, d["resend_cooldown_seconds"]),
        max_requests_per_hour=_clamp(max_requests_per_hour if max_requests_per_hour is not None else d["max_requests_per_hour"], OTP_MAX_REQUESTS_PER_HOUR_BOUNDS, d["max_requests_per_hour"]),
        sms_daily_cap=_clamp(sms_daily_cap if sms_daily_cap is not None else d["sms_daily_cap"], SMS_DAILY_CAP_BOUNDS, d["sms_daily_cap"]),
    )


def _require_app_id(app_id: str) -> str:
    """Reject an empty app_id so security-sensitive Redis keys never collide.

    Without this, an empty app_id would collapse every app into one
    ``otp::<mobile>`` namespace (cross-tenant OTP/cooldown/budget sharing).
    """
    cleaned = (app_id or "").strip()
    if not cleaned:
        raise ValueError("app_id is required for OTP operations")
    return cleaned


def _otp_key(app_id: str, mobile: str) -> str:
    return f"otp:{app_id}:{mobile}"


def _cooldown_key(app_id: str, mobile: str) -> str:
    return f"otp:cooldown:{app_id}:{mobile}"


def _reqcount_key(app_id: str, mobile: str) -> str:
    return f"otp:reqcount:{app_id}:{mobile}"


def _faillock_key(app_id: str, mobile: str) -> str:
    return f"otp:faillock:{app_id}:{mobile}"


def _budget_key(app_id: str) -> str:
    return f"otp:budget:{app_id}"


def _hash_otp(app_id: str, mobile: str, otp: str) -> str:
    """Return an HMAC of the OTP bound to the app, mobile, and proxy secret."""
    message = f"{app_id}:{mobile}:{otp}".encode("utf-8")
    return hmac.new(settings.jwt_secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _generate_otp(length: int) -> str:
    """Return a cryptographically random zero-padded numeric OTP."""
    upper_bound = 10 ** length
    return str(secrets.randbelow(upper_bound)).zfill(length)


# Verify atomically inside Redis so concurrent guesses cannot race the
# read-modify-write attempts counter (brute-force amplification) and so the
# TTL is preserved (never resurrected for a fresh full window) on a wrong guess.
_VERIFY_LUA = """
local v = redis.call('GET', KEYS[1])
if not v then return {'OTP_EXPIRED', '0'} end
local ok, data = pcall(cjson.decode, v)
if not ok then redis.call('DEL', KEYS[1]); return {'OTP_EXPIRED', '0'} end
local max = tonumber(ARGV[2])
local attempts = tonumber(data.attempts) or 0
if attempts >= max then redis.call('DEL', KEYS[1]); return {'OTP_TOO_MANY_ATTEMPTS', '0'} end
if data.hash == ARGV[1] then redis.call('DEL', KEYS[1]); return {'OK', '0'} end
attempts = attempts + 1
data.attempts = attempts
local ttl = redis.call('TTL', KEYS[1])
if ttl == nil or ttl < 1 then ttl = 1 end
redis.call('SET', KEYS[1], cjson.encode(data), 'EX', ttl)
return {'OTP_INVALID', tostring(max - attempts)}
"""


@dataclass(frozen=True)
class OtpVerifyResult:
    """Outcome of an OTP verification attempt."""

    ok: bool
    code: str = ""
    attempts_remaining: int = 0


async def remaining_cooldown(app_id: str, mobile: str) -> int:
    """Return seconds left before another OTP may be requested (0 if allowed)."""
    app_id = _require_app_id(app_id)
    client = await get_client()
    ttl = await client.ttl(_cooldown_key(app_id, mobile))
    return ttl if ttl and ttl > 0 else 0


async def consume_sms_budget(app_id: str, policy: OtpPolicy) -> bool:
    """Reserve one unit of the app's rolling-24h SMS budget.

    Returns ``False`` (and does not overspend) once the cap is reached, so a
    single guest token cannot drain the tenant's SMS account by iterating
    distinct destination numbers.
    """
    app_id = _require_app_id(app_id)
    client = await get_client()
    key = _budget_key(app_id)
    count = await client.incr(key)
    if count == 1:
        await client.expire(key, 86400)
    if count > policy.sms_daily_cap:
        # Undo the over-limit increment so the counter reflects real sends.
        await client.decr(key)
        return False
    return True


async def refund_sms_budget(app_id: str) -> None:
    """Return a reserved SMS budget unit (used when the SMS dispatch fails)."""
    app_id = _require_app_id(app_id)
    await _decr_floor(_budget_key(app_id))


async def create_otp(app_id: str, mobile: str, policy: OtpPolicy) -> str:
    """Generate, store (hashed), and return a fresh OTP for delivery."""
    app_id = _require_app_id(app_id)
    client = await get_client()
    otp = _generate_otp(policy.length)
    payload = {"hash": _hash_otp(app_id, mobile, otp), "attempts": 0}
    await client.set(_otp_key(app_id, mobile), json.dumps(payload), ex=policy.ttl_seconds)
    await client.set(_cooldown_key(app_id, mobile), "1", ex=policy.resend_cooldown_seconds)
    logger.info("OTP generated", extra={"app_id": app_id, "mobile_suffix": mobile[-4:]})
    return otp


async def register_request(app_id: str, mobile: str, policy: OtpPolicy) -> bool:
    """Count an OTP request in a rolling hour window.

    Returns ``False`` when the per-mobile hourly cap is exceeded, so a single
    number cannot be SMS-bombed even across IPs/cooldown windows.
    """
    app_id = _require_app_id(app_id)
    client = await get_client()
    key = _reqcount_key(app_id, mobile)
    count = await client.incr(key)
    if count == 1:
        await client.expire(key, 3600)
    return count <= policy.max_requests_per_hour


async def release_request(app_id: str, mobile: str) -> None:
    """Give back a request slot (used when the SMS dispatch fails)."""
    app_id = _require_app_id(app_id)
    await _decr_floor(_reqcount_key(app_id, mobile))


async def clear_otp(app_id: str, mobile: str) -> None:
    """Delete the OTP and its resend cooldown (used to roll back a failed send)."""
    app_id = _require_app_id(app_id)
    client = await get_client()
    await client.delete(_otp_key(app_id, mobile), _cooldown_key(app_id, mobile))


async def clear_otp_only(app_id: str, mobile: str) -> None:
    """Delete the OTP but KEEP the resend cooldown.

    Used on a gateway *delivery* failure so an attacker who forces sends to fail
    still burns the per-mobile spacing/hourly slot (cannot bypass throttles by
    triggering failures), while an infrastructure fault uses the full
    :func:`clear_otp` + :func:`release_request` rollback instead.
    """
    app_id = _require_app_id(app_id)
    client = await get_client()
    await client.delete(_otp_key(app_id, mobile))


async def _verify_is_locked(app_id: str, mobile: str, policy: OtpPolicy) -> bool:
    """Whether sustained failures have locked verification for this mobile."""
    client = await get_client()
    raw = await client.get(_faillock_key(app_id, mobile))
    try:
        failures = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        failures = 0
    return failures >= policy.verify_lockout_threshold


async def _record_verify_failure(app_id: str, mobile: str) -> None:
    """Increment the sustained failed-verify counter (rolling hour window)."""
    client = await get_client()
    key = _faillock_key(app_id, mobile)
    count = await client.incr(key)
    if count == 1:
        await client.expire(key, 3600)


async def _clear_verify_failures(app_id: str, mobile: str) -> None:
    client = await get_client()
    await client.delete(_faillock_key(app_id, mobile))


async def verify_otp(app_id: str, mobile: str, otp: str, policy: OtpPolicy) -> OtpVerifyResult:
    """Verify a submitted ``otp`` against the stored hash for ``mobile``.

    The compare/attempt/delete sequence runs atomically in Redis so concurrent
    requests cannot defeat the attempts cap and the TTL is never reset. A
    sustained per-mobile failure counter (spanning OTP regenerations) locks
    verification once brute-force volume is exceeded.
    """
    app_id = _require_app_id(app_id)

    if await _verify_is_locked(app_id, mobile, policy):
        return OtpVerifyResult(ok=False, code="OTP_LOCKED")

    client = await get_client()
    code, remaining = await client.eval(
        _VERIFY_LUA,
        1,
        _otp_key(app_id, mobile),
        _hash_otp(app_id, mobile, otp),
        str(policy.max_attempts),
    )
    code_str = str(code)
    if code_str == "OK":
        await _clear_verify_failures(app_id, mobile)
        return OtpVerifyResult(ok=True)

    # Only a genuine wrong guess against a LIVE OTP accrues toward the sustained
    # lockout. OTP_EXPIRED means no OTP was outstanding for this number (e.g. a
    # verify with no prior request), so counting it would let an attacker lock
    # out ANY number without ever triggering an SMS — a targeted login DoS.
    # OTP_INVALID only occurs when a stored OTP existed and the guess mismatched,
    # which requires a real prior request (and its SMS + cooldown).
    if code_str == "OTP_INVALID":
        await _record_verify_failure(app_id, mobile)

    try:
        attempts_remaining = max(int(remaining), 0)
    except (TypeError, ValueError):
        attempts_remaining = 0
    return OtpVerifyResult(ok=False, code=code_str, attempts_remaining=attempts_remaining)
