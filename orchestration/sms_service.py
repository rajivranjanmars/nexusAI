"""Gupshup enterprise SMS gateway integration for OTP delivery.

Credentials and the message template are **per-app**: the caller resolves them
(decrypting the app's stored blob) and passes them in. There are no env globals
here — every send uses the tenant's own Gupshup account and approved copy.

The message template is treated as inert data: the OTP is substituted with a
literal ``str.replace`` (never ``str.format``), so an admin-supplied template
can never be abused as a Python format string to traverse object attributes or
break delivery. Templates are validated by :func:`validate_message_template`.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

# Platform default endpoint / copy, used only when an app omits its own.
DEFAULT_GUPSHUP_URL = "https://enterprise.smsgupshup.com/GatewayAPI/rest"
DEFAULT_OTP_MESSAGE_TEMPLATE = (
    "{otp} is your OTP for registering in LPUDE Admission portal. "
    "The OTP is Valid for 5 mins. Please do not share this OTP with anyone, "
    "Regards LPU."
)

_OTP_PLACEHOLDER = "{otp}"
_MAX_TEMPLATE_LENGTH = 320  # ~2 SMS segments; DLT templates are short.


class TemplateError(ValueError):
    """Raised when a per-app SMS template is unsafe or malformed."""


class GatewayUrlError(ValueError):
    """Raised when a per-app SMS api_url is not an allowed gateway endpoint."""


def _allowed_gateway_hosts() -> set[str]:
    raw = settings.sms_allowed_gateway_hosts or ""
    hosts = {h.strip().lower() for h in raw.split(",") if h.strip()}
    # Always allow the platform default endpoint's host.
    default_host = urlparse(DEFAULT_GUPSHUP_URL).hostname
    if default_host:
        hosts.add(default_host.lower())
    return hosts


def _host_resolves_to_private_ip(host: str) -> bool:
    """Whether ``host`` resolves to a private/loopback/link-local address."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        # Cannot resolve — treat as unsafe (fail closed).
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            return True
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    return False


def validate_sms_api_url(api_url: str | None, *, resolve: bool = True) -> str | None:
    """Validate a per-app SMS gateway URL, or return ``None`` for the default.

    Blocks SSRF / credential+OTP exfiltration: the URL must be HTTPS and its host
    must be on the configured allowlist. When ``resolve`` is true (the default,
    used at admin **write** time), it additionally rejects hosts that resolve to
    a private/loopback/link-local address (defeats DNS-rebinding to internal /
    metadata hosts). The send path passes ``resolve=False`` to avoid a blocking
    DNS lookup on the event loop every send — the host allowlist already bounds
    the destination to trusted gateways there.

    Raises:
        GatewayUrlError: If the URL is not an allowed, safe gateway endpoint.
    """
    if not api_url:
        return None
    value = api_url.strip()
    parsed = urlparse(value)
    if parsed.scheme != "https":
        raise GatewayUrlError("SMS api_url must use https.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise GatewayUrlError("SMS api_url has no host.")
    if host not in _allowed_gateway_hosts():
        raise GatewayUrlError("SMS api_url host is not in the allowed gateway list.")
    if resolve and _host_resolves_to_private_ip(host):
        raise GatewayUrlError("SMS api_url host resolves to a disallowed address.")
    return value


def validate_message_template(template: str) -> str:
    """Validate and return a per-app OTP SMS template.

    A safe template contains exactly one ``{otp}`` placeholder and no other
    braces (which would indicate a stray/format-string token). This runs at
    admin write time so a bad template is rejected before storage, and again
    defensively at send time.

    Raises:
        TemplateError: If the template is empty, too long, missing ``{otp}``,
            has more than one ``{otp}``, or contains any other brace.
    """
    value = (template or "").strip()
    if not value:
        raise TemplateError("Template must not be empty.")
    if len(value) > _MAX_TEMPLATE_LENGTH:
        raise TemplateError(f"Template must be at most {_MAX_TEMPLATE_LENGTH} characters.")
    if value.count(_OTP_PLACEHOLDER) != 1:
        raise TemplateError("Template must contain exactly one {otp} placeholder.")
    # After removing the single legitimate placeholder, no braces may remain.
    residual = value.replace(_OTP_PLACEHOLDER, "", 1)
    if "{" in residual or "}" in residual:
        raise TemplateError("Template must not contain any braces other than {otp}.")
    return value


def _render_message(template: str, otp: str) -> str:
    """Substitute the OTP into a validated template via literal replacement."""
    safe_template = validate_message_template(template)
    return safe_template.replace(_OTP_PLACEHOLDER, otp)


async def send_otp_sms(
    mobile: str,
    otp: str,
    *,
    user_id: str,
    password: str,
    api_url: str | None = None,
    message_template: str | None = None,
) -> None:
    """Send an OTP SMS to ``mobile`` via the Gupshup gateway using per-app creds.

    Args:
        mobile: Destination mobile number.
        otp: The freshly generated one-time code.
        user_id: The app's Gupshup user id (decrypted, never logged).
        password: The app's Gupshup password (decrypted, never logged).
        api_url: The app's gateway URL, or the platform default when absent.
        message_template: The app's approved copy, or the platform default.

    Raises:
        RuntimeError: If credentials are missing or the gateway reports failure.
        TemplateError: If the message template is unsafe.
    """
    if not user_id or not password:
        raise RuntimeError("SMS credentials are not configured for this app")

    # Defense in depth: even though api_url is validated at write time, never
    # POST credentials + OTP to a non-allowlisted host (guards a blob written
    # before validation existed, or a rotated allowlist). ``resolve=False`` keeps
    # this off the event loop — the host allowlist already bounds the target.
    safe_api_url = validate_sms_api_url(api_url, resolve=False) or DEFAULT_GUPSHUP_URL

    message = _render_message(message_template or DEFAULT_OTP_MESSAGE_TEMPLATE, otp)

    params = {
        "method": "SendMessage",
        "send_to": mobile,
        "msg": message,
        "msg_type": "TEXT",
        "userid": user_id,
        "auth_scheme": "plain",
        "password": password,
        "v": "1.1",
        "format": "text",
    }

    # Gupshup's GatewayAPI reads parameters from the query string, so the gateway
    # credentials and OTP travel in the request URL (over TLS). Two controls keep
    # them from leaking: client request logging (which prints the full URL) is
    # silenced in shared.logger, and ``safe_api_url`` is host-allowlisted so the
    # URL is only ever sent to a trusted gateway.
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(safe_api_url, params=params, data="")

    body = response.text.strip()
    if response.status_code >= 400:
        # Do not echo the gateway body verbatim (it can contain the full number
        # and the credentials' account context); log only the status.
        raise RuntimeError(f"Gupshup SMS failed with status {response.status_code}")

    # Gupshup text responses look like "success | <phone> | <id>" on success and
    # "error | <code> | <reason>" on failure.
    if body.lower().startswith("error"):
        raise RuntimeError("Gupshup SMS returned an error response")

    logger.info(
        "OTP SMS dispatched",
        extra={"mobile_suffix": mobile[-4:], "gateway_status": response.status_code},
    )
