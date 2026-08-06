"""
FastAPI backend proxy that sits in front of the private MCP server.

The proxy owns browser-facing JWT authentication, rate limiting, response
sanitisation, and endpoint-level business logic. The downstream MCP server
remains private and is accessed only through an internal API key.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import jwt
import uuid
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import Depends, FastAPI, Request, Response, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from starlette.responses import StreamingResponse

from db.models.app import App
from db.postgres import get_session
from db.app_registry import invalidate_cache as _invalidate_app_cache, resolve_by_app_id
from db.audit_log import list_app_audit_log, record_app_change

from backend_proxy.auth import (
    AuthenticatedUser,
    ProxyAuthError,
    authenticate_app_jwt,
    decode_token,
    issue_access_token,
    issue_refresh_token,
    mark_phone_verified,
    phone_session_id,
)
from backend_proxy.mcp_client import McpGatewayError, mcp_gateway
from backend_proxy.correlation_handler import CorrelationHandler
from backend_proxy.rate_limiter import (
    RateLimitExceeded,
    RateLimitState,
    close as close_rate_limiter,
    enforce_rate_limit,
)
from backend_proxy.resolvers import run_resolvers
from backend_proxy.sanitizer import sanitize_payload
from backend_proxy.schemas import (
    AdminAppDetailResponse,
    AdminAppSummaryResponse,
    AdminAppUpdateRequest,
    AdminLoginRequest,
    AdminLoginResponse,
    AdminRefreshRequest,
    AdminRefreshResponse,
    AdminUserProfile,
    AdminUserSummary,
    AdminUserCreateRequest,
    AdminUserUpdateRequest,
    AppAuditLogEntryResponse,
    AppBootstrapRegistrationRequest,
    AppBootstrapRegistrationResponse,
    DevAdminTokenMintRequest,
    DevAppTokenMintRequest,
    DevSignedTokenRequest,
    DevSignedTokenResponse,
    AppRegistrationRequest,
    AppRegistrationResponse,
    AppTokenRequest,
    AppTokenResponse,
    AppUserTokenMintRequest,
    ChatFeedbackRequest,
    ChatFeedbackResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    FeedbackListQuery,
    FeedbackListResponse,
    FeedbackRecordResponse,
    OtpRequestRequest,
    OtpRequestResponse,
    OtpVerifyRequest,
    OtpVerifyResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    SmsCredentialsRequest,
    SmsCredentialsResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from backend_proxy.rag_routes import router as rag_router
from backend_proxy.observability_routes import router as observability_router
from backend_proxy.health_monitor import start_health_monitor
from backend_proxy.deps import (
    admin_actor_label,
    get_admin_scope,
    get_admin_user,
    get_current_user as _get_current_user,
    require_admin_hybrid,
    require_admin_user,
    require_app_admin,
    require_super_admin,
)
from backend_proxy.admin_auth import AdminUserContext, AdminAuthError
from backend_proxy import otp_service
from backend_proxy.sms_secrets import (
    SecretUnavailable,
    decrypt_sms_credentials,
    encrypt_sms_credentials,
    mask_user_id,
    sms_encryption_configured,
)
from db.feedback_tracker import get_feedback_by_id, list_feedback, record_feedback
from orchestration.conversation_memory import append_turn, set_lead_progress
from orchestration.sms_service import GatewayUrlError, TemplateError, send_otp_sms
from orchestration.lead_capture_flags import maybe_flag_completion_celebration
from orchestration.workflow_config import get_workflow_config
from orchestration.workflow_policy import should_use_response_cache, workflow_uses_public_rag
from shared.config import settings
from shared.helper_buttons import (
    build_lead_form_buttons,
    build_public_helper_buttons,
    build_slot_buttons,
    normalize_lead_progress_payload,
)
from shared.logger import get_logger

logger = get_logger(__name__)

_VERSION = "1.1.0"
_STREAM_HIDDEN_TAGS = ("lead_progress", "upsert_lead")
_MAX_STREAM_SOURCE_URLS = 2


def _parse_allowed_origins(raw_origins: str) -> List[str]:
    """Parse configured CORS origins from an env string.

    Args:
        raw_origins: Comma-separated origin list or ``*``.

    Returns:
        List of origins suitable for ``CORSMiddleware``.
    """
    raw = raw_origins.strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _apply_rate_limit_headers(response: Response, state: RateLimitState) -> None:
    """Attach standard rate-limit headers to an HTTP response.

    Args:
        response: Outgoing response object.
        state: Rate limit window metadata.
    """
    response.headers["X-RateLimit-Limit"] = str(state.limit)
    response.headers["X-RateLimit-Remaining"] = str(state.remaining)
    response.headers["Retry-After"] = str(state.retry_after)


# Keys never sent to the browser in app_config. ``data_resolvers`` is internal
# wiring; the secret patterns are belt-and-suspenders — real secrets live in
# dedicated columns, never in app_config.
_CLIENT_CONFIG_INTERNAL_KEYS = frozenset({"data_resolvers"})
_CLIENT_CONFIG_SECRET_PATTERNS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "credential",
    "gupshup",
    "api_key",
    "apikey",
    "private",
    "_key",
    "token",
    "jwt",
    "bearer",
    "auth_",
    "session",
    "cert",
    "salt",
    "signing",
)


def _client_safe_app_config(
    raw_config: Dict[str, Any],
    *,
    otp_required: bool,
) -> Optional[Dict[str, Any]]:
    """Build the browser-facing app_config from the stored app_config.

    ``app_config`` is the tenant-defined UI channel (theme, copy, quick
    actions). It is intentionally open-schema, so this is a denylist rather than
    an allowlist: the internal ``data_resolvers`` key and any secret-shaped key
    are stripped. Secrets never actually live here — they are stored in
    dedicated columns (encrypted for SMS creds) — so this is defense-in-depth
    against a future accidental secret in app_config, and it logs loudly if one
    is ever seen.

    Also injects the single browser-facing feature flag ``features.otp_required``
    (derived from the ``otp_enabled`` column) so the widget can gate its phone
    screen without ever seeing OTP internals.
    """
    result: Dict[str, Any] = {}
    for key, value in (raw_config or {}).items():
        key_lower = str(key).lower()
        if key in _CLIENT_CONFIG_INTERNAL_KEYS:
            continue
        if any(pattern in key_lower for pattern in _CLIENT_CONFIG_SECRET_PATTERNS):
            logger.warning("Dropped secret-shaped key '%s' from client app_config", key)
            continue
        result[key] = value

    # Only surface a features block when there is something to return or OTP is
    # on, so plain non-OTP apps keep their previous (None) shape. ``features`` is
    # admin-controlled open JSON, so coerce a non-dict value rather than crash.
    if result or otp_required:
        existing = result.get("features")
        features = dict(existing) if isinstance(existing, dict) else {}
        features["otp_required"] = bool(otp_required)
        result["features"] = features

    return result or None


def _client_host(request: Request) -> str:
    """Return the best-effort client host for anonymous rate limiting.

    Args:
        request: Active FastAPI request.

    Returns:
        Request client host or ``unknown``.
    """
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _format_sse(event: str, data: Dict[str, Any]) -> str:
    """Encode one Server-Sent Events (SSE) payload.
    
    Args:
        event: The event type (e.g., 'chat', 'metadata', 'status', 'error', 'done')
        data: The data payload to serialize as JSON
        
    Returns:
        Formatted SSE string
    """
    try:
        json_data = json.dumps(data, default=str)
    except (TypeError, ValueError) as e:
        logger.warning("Failed to serialize SSE data: %s", e)
        json_data = json.dumps({"error": "Failed to serialize data"})
    
    return f"event: {event}\ndata: {json_data}\n\n"


def _chunk_text(text: str, max_chars: int = 160) -> List[str]:
    """Split response text into client-friendly chunks for SSE delivery.

    Chunk boundaries never consume whitespace, so concatenating every
    chunk in order reproduces ``text`` exactly (newlines and bullet
    formatting included) — the client just appends what it receives.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    current = ""
    for atom in re.findall(r"\S+\s*", text):
        if current and len(current) + len(atom) > max_chars:
            chunks.append(current)
            current = ""
        current += atom

    if current:
        chunks.append(current)
    return chunks or [text]


def _workflow_includes_sources(app_id: str, workflow: str, user_query: str = "") -> bool:
    """Whether a workflow should expose source URLs in user-facing replies."""
    config = get_workflow_config(app_id)
    workflow_config = config.workflow_response_config.get(
        workflow,
        config.workflow_response_config.get("general"),
    )
    return bool(workflow_config and workflow_config.include_sources)


def _append_sources_to_response(
    response: str,
    rag_context: List[Dict[str, Any]],
    *,
    app_id: str,
    workflow: str,
    user_query: str = "",
) -> str:
    """Append deduplicated source URLs when a response used RAG context."""
    if not response or not rag_context:
        return response

    if not _workflow_includes_sources(app_id, workflow, user_query):
        return response

    if "Sources:" in response:
        return response

    sources_block = _build_sources_block(rag_context)
    if not sources_block:
        return response

    return f"{response.rstrip()}\n\n{sources_block}"


def _build_sources_block(rag_context: List[Dict[str, Any]]) -> str:
    """Build the markdown sources block."""
    sources: List[str] = []
    seen: set[str] = set()
    source_index = 0
    
    for item in rag_context:
        source_url = (item or {}).get("source_url")
        if not source_url or source_url in seen:
            continue
        seen.add(source_url)
        source_index += 1
        # heading = (item or {}).get("section_heading") or source_url
        sources.append(f"[{source_index}] {source_url}")
        if len(sources) >= _MAX_STREAM_SOURCE_URLS:
            break
            
    if not sources:
        return ""
        
    return "Sources:\n" + "\n".join(sources)


def _resolve_helper_buttons(
    workflow: str,
    raw_lead_progress: Any,
    user_input: str,
    clarification: Any = None,
) -> List[Dict[str, Any]]:
    """Return the correct helper buttons for the current workflow mode.

    If a clarification question is pending (narrowing question), return
    slot buttons for the clarification axis.

    Lead form mode (``lead_capture``) shows step-specific option buttons
    plus the Go Back button.
    All other workflows show contextual topic buttons plus the persistent
    Request a Callback button.
    """
    # ponytail: clarification pending in lead_capture is impossible
    # (disambiguate_node returns early for non-answer strategies);
    # check clarification first for simpler logic.
    if (
        clarification
        and isinstance(clarification, dict)
        and clarification.get("pending") is True
        and clarification.get("axis")
        and clarification.get("options")
    ):
        return build_slot_buttons(clarification["axis"], clarification["options"])

    if workflow == "lead_capture":
        return build_lead_form_buttons(lead_progress=raw_lead_progress)
    return build_public_helper_buttons(
        lead_progress=raw_lead_progress,
        user_input=user_input,
    )



class _HiddenXmlStreamFilter:
    """Strip internal XML blocks from streamed chat deltas.

    LLM providers may split tags across arbitrary chunks, so each delta must
    pass through a small stateful buffer before it is safe to emit as chat.
    """

    def __init__(self, tags: tuple[str, ...]) -> None:
        self._buffer = ""
        self._tags = tags
        self._start_tags = {tag: f"<{tag}>" for tag in tags}
        self._end_tags = {tag: f"</{tag}>" for tag in tags}
        self.captured: List[Dict[str, str]] = []

    def feed(self, text: str) -> str:
        """Return visible text that is safe to stream as ``chat``."""
        if not text:
            return ""

        self._buffer += text
        return self._drain(flush=False)

    def flush(self) -> str:
        """Return any remaining visible text after the stream ends."""
        return self._drain(flush=True)

    def _drain(self, *, flush: bool) -> str:
        visible: List[str] = []

        while self._buffer:
            start = self._find_next_start()
            if start is None:
                emit_len = len(self._buffer) if flush else self._safe_emit_length()
                if emit_len <= 0:
                    break
                visible.append(self._buffer[:emit_len])
                self._buffer = self._buffer[emit_len:]
                break

            tag, start_index = start
            if start_index > 0:
                visible.append(self._buffer[:start_index])
                self._buffer = self._buffer[start_index:]

            start_tag = self._start_tags[tag]
            end_tag = self._end_tags[tag]
            end_index = self._buffer.find(end_tag, len(start_tag))
            if end_index < 0:
                if flush:
                    self._buffer = ""
                break

            content_start = len(start_tag)
            content = self._buffer[content_start:end_index].strip()
            self.captured.append({"tag": tag, "content": content})
            self._buffer = self._buffer[end_index + len(end_tag):]

        return "".join(visible)

    def _find_next_start(self) -> Optional[tuple[str, int]]:
        found_tag: Optional[str] = None
        found_index: Optional[int] = None
        for tag, start_tag in self._start_tags.items():
            index = self._buffer.find(start_tag)
            if index >= 0 and (found_index is None or index < found_index):
                found_tag = tag
                found_index = index

        if found_tag is None or found_index is None:
            return None
        return found_tag, found_index

    def _safe_emit_length(self) -> int:
        """Leave a suffix that could still become a hidden start tag."""
        max_overlap = 0
        for start_tag in self._start_tags.values():
            max_len = min(len(start_tag) - 1, len(self._buffer))
            for length in range(1, max_len + 1):
                if self._buffer.endswith(start_tag[:length]):
                    max_overlap = max(max_overlap, length)

        return len(self._buffer) - max_overlap


def _normalise_chat_downstream(downstream: Any, *, user_input: str = "") -> Dict[str, Any]:
    """Normalise downstream chat payload into a structured dict."""
    if isinstance(downstream, dict) and "result" in downstream and isinstance(downstream["result"], str):
        downstream = downstream["result"]

    if isinstance(downstream, str):
        try:
            downstream = json.loads(downstream)
        except (ValueError, TypeError):
            downstream = {"response": downstream, "workflow": "general", "cache_hit": False}

    if isinstance(downstream, dict):
        raw_lead_progress = downstream.get("lead_progress")
        lead_progress = normalize_lead_progress_payload(
            raw_lead_progress,
            user_input=user_input,
        )
        helper_buttons = build_public_helper_buttons(
            lead_progress=raw_lead_progress,
            user_input=user_input,
        )
        retrieval_debug = downstream.get("retrieval_debug", {})
        if isinstance(retrieval_debug, dict) and "error" in retrieval_debug:
            logger.warning("Sanitized retrieval error in metadata: %s", retrieval_debug["error"])
            retrieval_debug = {"error": "Retrieval process encountered an issue"}
        return {
            # 'cached_response' is the key used by chat_complete_prestream;
            # 'response' is used by chat_complete. Support both.
            "response": downstream.get("response", "") or downstream.get("cached_response", ""),
            "workflow": downstream.get("workflow", "general"),
            "cache_hit": downstream.get("cache_hit", False),
            "lead_progress": lead_progress,
            "helper_buttons": helper_buttons if isinstance(helper_buttons, list) else [],
            "token_usage": downstream.get("token_usage", {}),
            "error": downstream.get("error"),
            # Passthrough for streaming path
            "context": downstream.get("context", {}),
            "rag_context": downstream.get("rag_context", []),
            "answer_confidence": downstream.get("answer_confidence", 0.0),
            "retrieval_debug": retrieval_debug,
            "clarification": downstream.get("clarification"),
        }

    return {
        "response": str(downstream),
        "workflow": "general",
        "cache_hit": False,
        "lead_progress": None,
        "helper_buttons": [],
        "token_usage": {},
        "error": None,
        "clarification": None,
    }


async def _call_chat_complete(
    *,
    message: str,
    user: AuthenticatedUser,
    correlation_id: str,
    session_id: str = "",
    force_workflow: Optional[str] = None,
) -> Dict[str, Any]:
    # Enforce actor_id for lead_capture before hitting the MCP gateway
    if force_workflow == "lead_capture" and not user.actor_id:
        raise HTTPException(
            status_code=400,
            detail="lead_capture workflow requires a valid actor_id in the token.",
        )

    from mcp_server.tools import lead_tools

    await lead_tools.touch_lead_session_activity(session_id or user.actor_id)

    downstream = await mcp_gateway.call_tool(
        tool_name="chat_complete",
        arguments={
            "user_input": message,
            "actor_id": user.actor_id,
            "actor_type": user.actor_type,
            "app_id": user.app_id,
            "session_id": session_id or user.actor_id,
            "force_workflow": force_workflow,
            "verified_phone": user.phone if user.phone_verified else "",
        },
        correlation_id=correlation_id,
    )

    normalised = _normalise_chat_downstream(downstream, user_input=message)
    if normalised.get("workflow") == "lead_capture":
        normalised["lead_progress"] = await maybe_flag_completion_celebration(
            session_id=session_id or user.actor_id,
            lead_progress=normalised.get("lead_progress"),
        )

    logger.info(
        "chat_complete downstream type=%s value=%s",
        type(downstream).__name__,
        str(downstream)[:500],
    )
    return normalised


async def _stream_chat_events(
    *,
    message: str,
    user: AuthenticatedUser,
    correlation_id: str,
    session_id: str = "",
    force_workflow: Optional[str] = None,
) -> AsyncIterator[str]:
    """Yield real SSE token-level events for the chat pipeline.

    Strategy (Hybrid Bypass):
      1. Run the LangGraph pipeline up to ``build_context`` via MCP
         (classification, cache, fetch, RAG, context — all fast).
      2. If cache hit → chunk the cached response and yield.
      3. If cache miss → stream LLM reasoning tokens directly from the
         proxy process, bypassing MCP for the LLM call.
      4. After streaming completes, post-process XML tags for lead_capture
         and fire-and-forget cache store + memory append.
    """
    # Enforce actor_id for lead_capture at the proxy level
    if force_workflow == "lead_capture" and not user.actor_id:
        yield _format_sse("error", {
            "message": "lead_capture workflow requires a valid actor_id in the token.",
            "workflow": "lead_capture",
        })
        return

    from mcp_server.tools import lead_tools

    await lead_tools.touch_lead_session_activity(session_id or user.actor_id)

    yield _format_sse(
        "status",
        {"phase": "accepted", "correlation_id": correlation_id},
    )

    # ── Step 1: Run pipeline up to context (via MCP) ────────────────────
    try:
        prestream_result = await mcp_gateway.call_tool(
            tool_name="chat_complete_prestream",
            arguments={
                "user_input": message,
                "actor_id": user.actor_id,
                "actor_type": user.actor_type,
                "app_id": user.app_id,
                "session_id": session_id or user.actor_id,
                "force_workflow": force_workflow,
            },
            correlation_id=correlation_id,
        )
        prestream = _normalise_chat_downstream(prestream_result, user_input=message)
    except Exception as exc:
        logger.exception("Prestream pipeline failed: %s", exc)
        yield _format_sse("error", {
            "message": "I'm having trouble connecting to my brain right now. Please try again in a few moments.",
            "workflow": "general",
        })
        return

    if prestream.get("error"):
        yield _format_sse("error", {
            "message": "I encountered an issue gathering context. Please try again.",
            "workflow": prestream.get("workflow", "general"),
        })
        return

    workflow = prestream.get("workflow", "general")
    cache_hit = prestream.get("cache_hit", False)
    if workflow == "lead_capture":
        from mcp_server.tools import lead_tools

        await lead_tools.touch_lead_session_activity(session_id or user.actor_id)

    yield _format_sse("metadata", {
        "workflow": workflow,
        "cache_hit": cache_hit,
        "session_id": session_id or user.actor_id or correlation_id,
        "lead_progress": prestream.get("lead_progress", {}),
        "helper_buttons": _resolve_helper_buttons(
            workflow,
            prestream.get("lead_progress"),
            message,
            clarification=prestream.get("clarification"),
        ),
        "token_usage": prestream.get("token_usage", {}),
    })
    # ── Step 2: Cache hit → chunk the cached response ───────────────────
    if cache_hit:
        cached_response = prestream.get("response", "")
        for chunk in _chunk_text(cached_response):
            yield _format_sse("chat", {"text": chunk})
            await asyncio.sleep(0)

        # Emit source URLs from cache (restored via tool_results.rag_source_urls)
        rag_context = prestream.get("rag_context", [])
        sources_block = _build_sources_block(rag_context)
        if (
            sources_block
            and _workflow_includes_sources(user.app_id, workflow, message)
            and "Sources:" not in cached_response
        ):
            yield _format_sse("chat", {"text": f"\n\n{sources_block}"})

        yield _format_sse("metadata", {
            "workflow": workflow,
            "cache_hit": True,
            "lead_progress": prestream.get("lead_progress"),
            "helper_buttons": _resolve_helper_buttons(
                workflow,
                prestream.get("lead_progress"),
                message,
                clarification=prestream.get("clarification"),
            ),
            "token_usage": prestream.get("token_usage", {}),
        })
        yield _format_sse("done", {})
        return

    # ── Step 3: Cache miss → stream LLM tokens directly ────────────────
    yield _format_sse("status", {"phase": "streaming"})

    # Import the streaming reasoner (runs in the proxy process)
    from llm.reasoner import generate_response_stream
    from db.app_registry import resolve_by_app_id

    # Build the LLM context from prestream result
    llm_context = prestream.get("context") if isinstance(prestream.get("context"), dict) else {}
    llm_context["app_id"] = user.app_id
    llm_context["workflow"] = workflow
    llm_context["actor_id"] = user.actor_id
    llm_context["rag_context"] = prestream.get("rag_context", [])
    llm_context["answer_confidence"] = prestream.get("answer_confidence", 0.0)
    llm_context["lead_progress"] = prestream.get("lead_progress", {})
    llm_context["verified_phone"] = user.phone if user.phone_verified else ""
    llm_context["strict_rag"] = bool(user.app_id) and workflow_uses_public_rag(workflow)
    llm_context["guest_mode"] = not bool(user.actor_id)

    # Inject persona
    app_context = None
    try:
        app_context = resolve_by_app_id(user.app_id) if user.app_id else None
    except Exception as e:
        logger.warning("Failed to resolve app context: %s", e)
    
    app_config = app_context.app_config if app_context else {}
    if app_config and "persona" in app_config:
        llm_context["persona"] = app_config["persona"]

    full_text = ""
    usage = {}
    hidden_filter = _HiddenXmlStreamFilter(_STREAM_HIDDEN_TAGS)
    streamed_text = ""

    try:
        async for event in generate_response_stream(
            context=llm_context,
            user_query=message,
            app_id=user.app_id or "",
            actor_id=user.actor_id or "",
            actor_type=user.actor_type or "",
        ):
            if event.get("type") == "delta":
                visible_text = hidden_filter.feed(event.get("content", ""))
                if visible_text:
                    streamed_text += visible_text
                    yield _format_sse("chat", {"text": visible_text})
            elif event.get("type") == "usage":
                full_text = event.get("full_text", "")
                usage = event.get("usage", {})
    except ImportError as exc:
        logger.exception("Failed to import streaming module: %s", exc)
        yield _format_sse("error", {
            "message": "I encountered a configuration issue. Please contact support if this continues.",
            "workflow": workflow,
        })
        return
    except Exception as exc:
        logger.exception("LLM streaming failed: %s", exc)
        yield _format_sse("error", {
            "message": "Connection interrupted. Let's try re-sending your last message.",
            "workflow": workflow,
        })
        return

    trailing_text = hidden_filter.flush()
    if trailing_text:
        streamed_text += trailing_text
        yield _format_sse("chat", {"text": trailing_text})

    # ── Step 4: Post-process XML tags for lead_capture ──────────────────
    lead_progress = prestream.get("lead_progress")
    cleaned_response = full_text

    progress_match = re.search(
        r"<lead_progress>\s*(.*?)\s*</lead_progress>", full_text, re.DOTALL,
    )
    if progress_match:
        try:
            lead_progress = json.loads(progress_match.group(1))
            if workflow == "lead_capture":
                lead_progress = await maybe_flag_completion_celebration(
                    session_id=session_id or user.actor_id,
                    lead_progress=lead_progress,
                )
        except Exception:
            pass
        cleaned_response = re.sub(
            r"<lead_progress>.*?</lead_progress>", "", cleaned_response, flags=re.DOTALL,
        ).strip()
    else:
        for block in hidden_filter.captured:
            if block["tag"] != "lead_progress":
                continue
            try:
                lead_progress = json.loads(block["content"])
            except Exception:
                pass

    raw_lead_progress = lead_progress

    # ── Collect all upsert_lead blocks (findall handles multi-field messages) ─
    # Parse BEFORE resolving buttons so program_level is available for step 5.
    upsert_raw_list = re.findall(
        r"<upsert_lead>(.*?)</upsert_lead>", full_text, re.DOTALL,
    )
    if not upsert_raw_list:
        upsert_raw_list = [
            block["content"]
            for block in hidden_filter.captured
            if block["tag"] == "upsert_lead"
        ]
    for upsert_payload in upsert_raw_list:
        try:
            from mcp_server.tools import lead_tools
            payload = json.loads(upsert_payload)
            sid = payload.pop("session_id", user.actor_id)
            # Merge upserted field values into lead_progress for button resolution
            if isinstance(raw_lead_progress, dict):
                for key in ("program_level",):
                    if key in payload:
                        raw_lead_progress[key] = payload[key]
            asyncio.create_task(lead_tools.upsert_lead(session_id=sid, **payload))
        except Exception as e:
            logger.error("Failed to parse/execute upsert_lead from stream: %s", e)
    if upsert_raw_list:
        cleaned_response = re.sub(
            r"<upsert_lead>.*?</upsert_lead>", "", cleaned_response, flags=re.DOTALL,
        ).strip()

    lead_progress = normalize_lead_progress_payload(raw_lead_progress, user_input=message)
    helper_buttons = _resolve_helper_buttons(workflow, raw_lead_progress, message)

    cleaned_response = _append_sources_to_response(
        cleaned_response,
        prestream.get("rag_context", []),
        app_id=user.app_id,
        workflow=workflow,
        user_query=message,
    )
    sources_block = _build_sources_block(prestream.get("rag_context", []))
    if (
        sources_block
        and _workflow_includes_sources(user.app_id, workflow, message)
        and "Sources:" not in streamed_text
    ):
        yield _format_sse("chat", {"text": f"\n\n{sources_block}"})

    # ── Step 5: Fire-and-forget cache store + memory ────────────────────
    memory_session_id = session_id or user.actor_id
    if cleaned_response and user.app_id and memory_session_id:
        try:
            if workflow == "lead_capture" and lead_progress:
                lead_progress = {
                    **lead_progress,
                    "updated_at": datetime.now(timezone.utc).timestamp(),
                }
                asyncio.create_task(
                    set_lead_progress(user.app_id, memory_session_id, lead_progress)
                )
            asyncio.create_task(
                append_turn(user.app_id, memory_session_id, message, cleaned_response)
            )
        except Exception:
            pass

    # Cache store (awaited to ensure entry is available for subsequent queries)
    if (
        cleaned_response
        and should_use_response_cache(workflow, user.actor_id, app_id=user.app_id)
        and _caching_enabled_for_app(user.app_id)
    ):
        try:
            import time as _time
            from cache_module.vector_store import _embed
            from orchestration.workflow_config import get_workflow_config, resolve_cache_policy

            _policy = resolve_cache_policy(get_workflow_config(user.app_id), workflow)
            _cache_scope = _policy.scope

            _store_start = _time.monotonic()
            try:
                question_embedding = await _embed(message)
                _rag_context_for_cache = prestream.get("rag_context", [])

                payload = {
                    "app_id": user.app_id,
                    "workflow": workflow,
                    "question": message,
                    "response_text": cleaned_response,
                    "actor_id": user.actor_id,
                    "cache_scope": _cache_scope,
                    "tool_results": {
                        "rag_source_urls": [
                            item.get("source_url")
                            for item in _rag_context_for_cache
                            if item.get("source_url")
                        ],
                        "rag_result_count": len(_rag_context_for_cache),
                    },
                    "ttl_seconds": _policy.ttl_seconds or 86400,
                    "invalidated_by": ["student_update"] if _cache_scope == "actor" else ["rag_reingest"],
                    "question_embedding": question_embedding,
                }
                
                async with httpx.AsyncClient() as client:
                    resp = await client.post("http://mcp_server:8000/api/cache/store", json=payload, timeout=10.0)
                    resp.raise_for_status()

                _store_elapsed = round((_time.monotonic() - _store_start) * 1000, 1)
                logger.info(
                    "Cache stored for workflow=%s scope=%s in %.1fms",
                    workflow, _cache_scope, _store_elapsed,
                )
            except Exception as cache_exc:
                logger.warning("Stream cache store failed: %s", cache_exc)
        except ImportError as e:
            logger.error("Failed to import cache modules: %s", e)
        except Exception as e:
            logger.warning("Failed to create cache storage: %s", e)

    yield _format_sse("metadata", {
        "workflow": workflow,
        "cache_hit": False,
        "lead_progress": lead_progress,
        "helper_buttons": helper_buttons,
        "token_usage": usage,
    })
    yield _format_sse("done", {})



def _generate_es256_keypair() -> tuple[str, str]:
    """Generate an ES256 keypair and return (private_pem, public_pem)."""
    private_key = ec.generate_private_key(ec.SECP256R1())

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")

    return private_pem, public_pem


def _sign_app_jwt(app_id: str, private_key_pem: str, ttl_seconds: int, actor_id: str = "", actor_type: str = "student") -> str:
    """Create a short-lived ES256 app JWT for token exchange."""
    now = datetime.now(timezone.utc)
    payload = {
        "iss": app_id,
        "iat": now,
        "exp": now + timedelta(seconds=ttl_seconds),
        "actor_id": actor_id,
        "actor_type": actor_type,
    }
    return jwt.encode(payload, private_key_pem, algorithm="ES256")


def _app_to_admin_summary(app: App) -> AdminAppSummaryResponse:
    return AdminAppSummaryResponse(
        app_id=str(app.app_id),
        name=app.app_name,
        domain=app.domain,
        allowed_workflows=app.allowed_workflows,
        allowed_tools=app.allowed_tools,
        rate_limit_rpm=app.rate_limit_rpm,
        cache_ttl_seconds=app.cache_ttl_seconds,
        otp_enabled=app.otp_enabled,
        caching_enabled=app.caching_enabled,
        otp_length=app.otp_length,
        otp_ttl_seconds=app.otp_ttl_seconds,
        otp_max_attempts=app.otp_max_attempts,
        otp_resend_cooldown_seconds=app.otp_resend_cooldown_seconds,
        otp_max_requests_per_hour=app.otp_max_requests_per_hour,
        sms_daily_cap=app.sms_daily_cap,
        is_active=app.is_active,
        created_at=app.created_at.isoformat() if app.created_at else None,
        updated_at=app.updated_at.isoformat() if app.updated_at else None,
    )


def _app_to_admin_detail(app: App) -> AdminAppDetailResponse:
    return AdminAppDetailResponse(
        **_app_to_admin_summary(app).model_dump(),
        public_key=app.public_key,
        llm_model_override=app.llm_model_override,
        token_quota_monthly=app.token_quota_monthly,
        app_config=app.app_config,
        metadata=app.metadata_,
        # Only whether creds exist — never the encrypted blob or its contents.
        sms_credentials_configured=bool(app.sms_credentials_encrypted),
    )


# Maps AdminAppUpdateRequest field names to their App ORM attribute, so
# admin_update_app can apply patches and diff old/new values generically.
_APP_UPDATE_FIELD_ATTRS: Dict[str, str] = {
    "name": "app_name",
    "domain": "domain",
    "allowed_workflows": "allowed_workflows",
    "allowed_tools": "allowed_tools",
    "rate_limit_rpm": "rate_limit_rpm",
    "cache_ttl_seconds": "cache_ttl_seconds",
    "token_quota_monthly": "token_quota_monthly",
    "llm_model_override": "llm_model_override",
    "app_config": "app_config",
    "metadata": "metadata_",
    "otp_enabled": "otp_enabled",
    "caching_enabled": "caching_enabled",
    "otp_length": "otp_length",
    "otp_ttl_seconds": "otp_ttl_seconds",
    "otp_max_attempts": "otp_max_attempts",
    "otp_resend_cooldown_seconds": "otp_resend_cooldown_seconds",
    "otp_max_requests_per_hour": "otp_max_requests_per_hour",
    "sms_daily_cap": "sms_daily_cap",
    "is_active": "is_active",
    # NOTE: sms_credentials_encrypted is deliberately absent — secrets are
    # written only via the dedicated /sms-credentials endpoint, so they never
    # enter the generic PATCH old/new diff written to app_audit_log.changes.
}


def _parse_app_uuid(app_id: str) -> uuid.UUID:
    """Parse an app ID or raise a 400 with a friendly error."""
    try:
        return uuid.UUID(app_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid app_id format") from exc


def _caching_enabled_for_app(app_id: str) -> bool:
    """Whether semantic response caching is turned on for ``app_id``.

    Defaults to enabled (no app context) to match the App model's default.
    """
    if not app_id:
        return True
    app_context = resolve_by_app_id(app_id)
    return bool(app_context is None or app_context.caching_enabled)


def _otp_enabled_for_app(app_id: str) -> bool:
    """Whether guest OTP verification is turned on for ``app_id``.

    Defaults to disabled (no app context, or the app has never opted in) so
    apps that never configured Gupshup/OTP keep working as guests-only. A
    resolve error propagates (500) rather than being swallowed, so the phone
    gate never silently fails open on an infrastructure fault.
    """
    if not app_id:
        return False
    app_context = resolve_by_app_id(app_id)
    return bool(app_context and app_context.otp_enabled)


def _resolve_otp_policy(app_id: str) -> otp_service.OtpPolicy:
    """Build the clamped per-app OTP policy, falling back to safe defaults."""
    app_context = resolve_by_app_id(app_id) if app_id else None
    if not app_context:
        return otp_service.clamp_otp_policy()
    return otp_service.clamp_otp_policy(
        length=app_context.otp_length,
        ttl_seconds=app_context.otp_ttl_seconds,
        max_attempts=app_context.otp_max_attempts,
        resend_cooldown_seconds=app_context.otp_resend_cooldown_seconds,
        max_requests_per_hour=app_context.otp_max_requests_per_hour,
        sms_daily_cap=app_context.sms_daily_cap,
    )


def _rate_limit_override_for_app(app_id: str) -> Optional[int]:
    """Return the app's per-minute rate limit, if configured, else None.

    Applies the ``apps.rate_limit_rpm`` column so OTP/auth throttles are
    per-tenant rather than one shared global window.
    """
    if not app_id:
        return None
    app_context = resolve_by_app_id(app_id)
    return app_context.rate_limit_rpm if app_context else None


async def _fetch_sms_credentials(app_id: str) -> Dict[str, Any]:
    """Read and decrypt this app's SMS credentials fresh from the DB.

    The ciphertext is deliberately not part of ``AppContext`` (so it never
    enters the in-process cache or any serialized payload); it is read here, at
    OTP-send time only.

    Raises:
        SecretUnavailable: If the encryption key is unset, no credentials are
            configured, or the stored blob cannot be decrypted (fail-closed).
    """
    async with get_session() as session:
        app_record = await session.get(App, _parse_app_uuid(app_id))
        ciphertext = app_record.sms_credentials_encrypted if app_record else None
    return decrypt_sms_credentials(ciphertext)


def _validated_client_session_id(
    session_id: Optional[str], user: AuthenticatedUser
) -> Optional[str]:
    """Validate a client-supplied session_id against the reserved ``phone:`` space.

    ``phone:<number>`` is the server-derived key for a phone-verified guest
    (:func:`phone_session_id`) and is returned to the client as the response
    ``session_id``. A conforming client echoes it back on the next turn, so the
    caller's **own** phone-scoped key is accepted; any *other* ``phone:`` value is
    an attempt to read/pollute another user's conversation (IDOR) and is rejected.
    """
    if session_id and session_id.strip().lower().startswith("phone:"):
        if session_id != phone_session_id(user):
            raise HTTPException(status_code=400, detail="Invalid session_id.")
    return session_id


def _require_phone_verified(user: AuthenticatedUser) -> None:
    """Reject anonymous (no actor_id) guests that have not verified a phone.

    Verification is only required when the app has ``otp_enabled`` turned on;
    apps without OTP configured are unaffected. Fails **closed**: if a token
    names an app that can no longer be resolved (deactivated / deleted /
    unknown), the guest is denied rather than silently let through — otherwise
    deactivating an ``otp_enabled`` app would strip its phone gate for live
    tokens.
    """
    if user.actor_id or user.phone_verified:
        return
    if not user.app_id:
        # No app scope on the token — per-app OTP cannot apply.
        return
    app_context = resolve_by_app_id(user.app_id)
    if app_context is None:
        logger.warning(
            "Blocking guest chat for unresolvable app",
            extra={"event": "otp.gate.blocked", "app_id": user.app_id, "reason": "app_unavailable"},
        )
        raise ProxyAuthError(
            "This application is not available.",
            code="APP_UNAVAILABLE",
            status_code=403,
        )
    if not app_context.otp_enabled:
        return
    logger.info(
        "Blocking unverified guest chat",
        extra={"event": "otp.gate.blocked", "app_id": user.app_id, "reason": "phone_verification_required"},
    )
    raise ProxyAuthError(
        "Phone verification required before chatting.",
        code="PHONE_VERIFICATION_REQUIRED",
        status_code=403,
    )


def _authorise_tool(tool_name: str, arguments: Dict[str, Any], user: AuthenticatedUser) -> None:
    """Enforce allowlisting and ownership rules for proxied tool access.

    If the user has ``allowed_tools`` from the App Registry those are used
    exclusively.  Otherwise the legacy frozenset-based rules apply.

    Args:
        tool_name: Requested tool name.
        arguments: Tool arguments from the caller.
        user: Authenticated proxy user.

    Raises:
        ProxyAuthError: If the tool is not allowed or the user is not authorised.
    """
    # ── App-registry-aware path ──────────────────────────────────────────
    if not user.allowed_tools or tool_name not in user.allowed_tools:
        raise ProxyAuthError(
            f"Tool '{tool_name}' not in app allowlist",
            code="TOOL_NOT_ALLOWED",
            status_code=403,
        )

    # Ownership check — non-admins can only access their own data
    target_actor_id = arguments.get("actor_id") or arguments.get("student_id")
    if (
        target_actor_id
        and user.role not in ("admin", "app")
        and user.actor_id
        and target_actor_id != user.actor_id
    ):
        raise ProxyAuthError(
            "Student scope mismatch",
            code="FORBIDDEN_STUDENT_SCOPE",
            status_code=403,
        )




def create_app() -> FastAPI:
    """Build the backend proxy ASGI application.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="LPUAI Backend Proxy", version=_VERSION)
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    app.add_middleware(CorrelationHandler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_allowed_origins(settings.proxy_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(rag_router)
    app.include_router(observability_router)
    start_health_monitor(app)

    @app.exception_handler(ProxyAuthError)
    async def handle_proxy_auth_error(request: Request, exc: ProxyAuthError) -> JSONResponse:
        logger.warning(
            "Proxy auth error",
            extra={
                "path": request.url.path,
                "code": exc.code,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc), "code": exc.code},
        )

    @app.exception_handler(AdminAuthError)
    async def handle_admin_auth_error(request: Request, exc: AdminAuthError) -> JSONResponse:
        logger.warning(
            "Admin auth error",
            extra={
                "path": request.url.path,
                "code": exc.code,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc), "code": exc.code},
        )

    @app.exception_handler(RateLimitExceeded)
    async def handle_rate_limit_error(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"error": str(exc), "code": "RATE_LIMIT_EXCEEDED"},
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.exception_handler(McpGatewayError)
    async def handle_mcp_gateway_error(request: Request, exc: McpGatewayError) -> JSONResponse:
        logger.error("McpGatewayError in proxy: %s", exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": "I'm having trouble connecting to my backend services. Please try again in a few moments.",
                "code": "DOWNSTREAM_MCP_ERROR"
            },
        )

    @app.on_event("startup")
    async def on_startup() -> None:
        from mcp_server.tools import lead_tools
        from orchestration.lead_capture_inactivity import partial_lead_inactivity_loop

        # TODO: If backend_proxy runs multiple replicas, guard this scanner with a
        # Redis leader lock or move it to a single worker process.
        app.state.partial_lead_inactivity_task = asyncio.create_task(
            partial_lead_inactivity_loop(lead_tools.flush_partial_lead_after_inactivity)
        )

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        task = getattr(app.state, "partial_lead_inactivity_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await close_rate_limiter()

    @app.get("/health/version")
    async def health_version() -> JSONResponse:
        """Return proxy health and downstream wiring metadata."""
        return JSONResponse(
            {
                "status": "ok",
                "version": _VERSION,
                "server": "LPUAI Backend Proxy",
                "mcp_target": settings.proxy_mcp_sse_url,
                "environment": settings.environment,
                "git_branch": settings.git_branch,
                "build_timestamp": settings.build_timestamp,
            }
        )

    @app.post("/api/apps/register", response_model=AppRegistrationResponse)
    async def register_app(
        payload: AppRegistrationRequest,
        x_admin_key: str = Header(..., description="Admin login password required for registration")
    ) -> AppRegistrationResponse:
        """Dynamically register a new application into the Proxy DB."""
        if x_admin_key != settings.proxy_admin_login_password:
            raise HTTPException(status_code=401, detail="Invalid admin key")

        # Basic validation (could be more rigorous in production using cryptography package)
        if "-----BEGIN PUBLIC KEY-----" not in payload.public_key:
            raise HTTPException(status_code=400, detail="Invalid PEM formatted public key")

        new_app_id = uuid.uuid4()
        
        async with get_session() as session:
            new_app = App(
                app_id=new_app_id,
                app_name=payload.name,
                public_key=payload.public_key,
                domain=payload.domain,
                allowed_workflows=payload.allowed_workflows,
                allowed_tools=payload.allowed_tools,
                rate_limit_rpm=60,
                cache_ttl_seconds=3600,
                is_active=True
            )
            session.add(new_app)
            await session.commit()
            
            logger.info("Dynamically registered new application", extra={"app_id": str(new_app_id), "app_name": payload.name, "domain": payload.domain})

        return AppRegistrationResponse(
            app_id=str(new_app_id),
            name=payload.name,
            status="Registered successfully"
        )

    @app.post("/api/apps/bootstrap", response_model=AppBootstrapRegistrationResponse)
    async def bootstrap_app(
        payload: AppBootstrapRegistrationRequest,
        request: Request,
        _admin: AdminUserContext = Depends(require_super_admin),
    ) -> AppBootstrapRegistrationResponse:
        """Register a new application and return a freshly generated ES256 keypair.

        This mirrors the behavior of the local registration helper script, but as
        an HTTP endpoint for admin tooling. Requires super admin role.
        """

        private_key_pem, public_key_pem = _generate_es256_keypair()
        new_app_id = uuid.uuid4()

        async with get_session() as session:
            new_app = App(
                app_id=new_app_id,
                app_name=payload.name,
                public_key=public_key_pem,
                domain=payload.domain,
                allowed_workflows=payload.allowed_workflows,
                allowed_tools=payload.allowed_tools,
                rate_limit_rpm=60,
                cache_ttl_seconds=3600,
                is_active=True,
            )
            session.add(new_app)
            await session.commit()

        logger.info(
            "Bootstrapped new PKI application",
            extra={"app_id": str(new_app_id), "app_name": payload.name, "domain": payload.domain},
        )

        # ponytail: audit log for admin dashboard app onboarding
        from backend_proxy.admin_auth import log_admin_action
        await log_admin_action(
            admin_user_id=_admin.admin_user_id,
            action="app.bootstrap",
            target_type="app",
            target_id=str(new_app_id),
            details={"name": payload.name, "domain": payload.domain},
            ip_address=request.client.host if request.client else None,
        )

        return AppBootstrapRegistrationResponse(
            app_id=str(new_app_id),
            name=payload.name,
            domain=payload.domain,
            public_key=public_key_pem,
            private_key=private_key_pem,
            allowed_workflows=payload.allowed_workflows,
            allowed_tools=payload.allowed_tools,
            proxy_url=str(request.base_url).rstrip("/"),
        )

    @app.get("/api/admin/apps", response_model=List[AdminAppSummaryResponse])
    async def admin_list_apps(
        include_inactive: bool = True,
        _admin: AdminUserContext | AuthenticatedUser = Depends(require_admin_hybrid),
        scope: str | None = Depends(get_admin_scope),
    ) -> List[AdminAppSummaryResponse]:
        """List registered applications for admin tooling."""
        async with get_session() as session:
            stmt = select(App).order_by(App.created_at.desc())
            if not include_inactive:
                stmt = stmt.where(App.is_active == True)
            # ponytail: app_admin only sees their own app
            if scope:
                stmt = stmt.where(App.app_id == _parse_app_uuid(scope))
            result = await session.execute(stmt)
            apps = result.scalars().all()
        return [_app_to_admin_summary(app) for app in apps]

    @app.get("/api/admin/apps/{app_id}", response_model=AdminAppDetailResponse)
    async def admin_get_app(
        app_id: str,
        _admin: AdminUserContext | AuthenticatedUser = Depends(require_admin_hybrid),
        scope: str | None = Depends(get_admin_scope),
    ) -> AdminAppDetailResponse:
        """Get detailed information for one registered application."""
        if scope and _parse_app_uuid(app_id) != _parse_app_uuid(scope):
            raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")
        async with get_session() as session:
            app_record = await session.get(App, _parse_app_uuid(app_id))
            if not app_record:
                raise HTTPException(status_code=404, detail="App not found")
            return _app_to_admin_detail(app_record)

    @app.patch("/api/admin/apps/{app_id}", response_model=AdminAppDetailResponse)
    async def admin_update_app(
        app_id: str,
        payload: AdminAppUpdateRequest,
        _admin: AdminUserContext | AuthenticatedUser = Depends(require_admin_hybrid),
        scope: str | None = Depends(get_admin_scope),
    ) -> AdminAppDetailResponse:
        """Update app metadata, routing permissions, and activation state."""
        if scope and _parse_app_uuid(app_id) != _parse_app_uuid(scope):
            raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")
        async with get_session() as session:
            app_record = await session.get(App, _parse_app_uuid(app_id))
            if not app_record:
                raise HTTPException(status_code=404, detail="App not found")

            updates = payload.model_dump(exclude_unset=True)
            changes: Dict[str, Any] = {}
            for field, attr in _APP_UPDATE_FIELD_ATTRS.items():
                if field not in updates:
                    continue
                old_value = getattr(app_record, attr)
                new_value = updates[field]
                if old_value != new_value:
                    changes[field] = {"old": old_value, "new": new_value}
                setattr(app_record, attr, new_value)

            session.add(app_record)
            await session.flush()
            await session.refresh(app_record)
            _invalidate_app_cache()
            detail = _app_to_admin_detail(app_record)

        await record_app_change(
            app_id=app_id,
            changed_by=admin_actor_label(_admin),
            action="update",
            changes=changes,
        )

        # ponytail: audit log for new admin dashboard tokens only
        if scope and isinstance(_admin, AdminUserContext):
            from backend_proxy.admin_auth import log_admin_action
            await log_admin_action(
                admin_user_id=_admin.admin_user_id,
                action="app.update",
                target_type="app",
                target_id=app_id,
                details=updates,
                ip_address=request.client.host if request.client else None,
            )

        return detail

    @app.post("/api/admin/apps/{app_id}/activate", response_model=AdminAppDetailResponse)
    async def admin_activate_app(
        app_id: str,
        _admin: AdminUserContext | AuthenticatedUser = Depends(require_admin_hybrid),
        scope: str | None = Depends(get_admin_scope),
    ) -> AdminAppDetailResponse:
        """Activate a registered application."""
        if scope and _parse_app_uuid(app_id) != _parse_app_uuid(scope):
            raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")
        async with get_session() as session:
            app_record = await session.get(App, _parse_app_uuid(app_id))
            if not app_record:
                raise HTTPException(status_code=404, detail="App not found")
            was_active = app_record.is_active
            app_record.is_active = True
            session.add(app_record)
            await session.flush()
            await session.refresh(app_record)
            _invalidate_app_cache()
            detail = _app_to_admin_detail(app_record)

        if not was_active:
            await record_app_change(
                app_id=app_id,
                changed_by=admin_actor_label(_admin),
                action="activate",
                changes={"is_active": {"old": False, "new": True}},
            )
        return detail

    @app.post("/api/admin/apps/{app_id}/deactivate", response_model=AdminAppDetailResponse)
    async def admin_deactivate_app(
        app_id: str,
        _admin: AdminUserContext | AuthenticatedUser = Depends(require_admin_hybrid),
        scope: str | None = Depends(get_admin_scope),
    ) -> AdminAppDetailResponse:
        """Deactivate a registered application without deleting it."""
        if scope and _parse_app_uuid(app_id) != _parse_app_uuid(scope):
            raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")
        async with get_session() as session:
            app_record = await session.get(App, _parse_app_uuid(app_id))
            if not app_record:
                raise HTTPException(status_code=404, detail="App not found")
            was_active = app_record.is_active
            app_record.is_active = False
            session.add(app_record)
            await session.flush()
            await session.refresh(app_record)
            _invalidate_app_cache()
            detail = _app_to_admin_detail(app_record)

        if was_active:
            await record_app_change(
                app_id=app_id,
                changed_by=admin_actor_label(_admin),
                action="deactivate",
                changes={"is_active": {"old": True, "new": False}},
            )
        return detail

    @app.post("/api/admin/apps/{app_id}/cache/clear")
    async def admin_clear_app_cache(
        app_id: str,
        _admin: AdminUserContext | AuthenticatedUser = Depends(require_admin_hybrid),
        scope: str | None = Depends(get_admin_scope),
    ) -> Dict[str, Any]:
        """Clear semantic response cache for an app."""
        if scope and _parse_app_uuid(app_id) != _parse_app_uuid(scope):
            raise HTTPException(status_code=403, detail="Access denied: app_id mismatch")

        from cache_module.semantic_cache import clear_cache
        count = await clear_cache(app_id=app_id)

        logger.info(
            "Cache cleared",
            extra={"event": "cache.cleared", "app_id": app_id, "count": count, "admin": admin_actor_label(_admin)},
        )
        return {"app_id": app_id, "cleared": count}

    @app.get("/api/admin/apps/{app_id}/audit-log", response_model=List[AppAuditLogEntryResponse])
    async def admin_get_app_audit_log(
        app_id: str,
        limit: int = 100,
        _admin: AuthenticatedUser = Depends(require_admin_user),
    ) -> List[AppAuditLogEntryResponse]:
        """Return the change history (who/when/what) for one registered app."""
        async with get_session() as session:
            app_record = await session.get(App, _parse_app_uuid(app_id))
            if not app_record:
                raise HTTPException(status_code=404, detail="App not found")
        entries = await list_app_audit_log(app_id=app_id, limit=min(max(limit, 1), 500))
        return [AppAuditLogEntryResponse(**entry) for entry in entries]

    @app.put(
        "/api/admin/apps/{app_id}/sms-credentials",
        response_model=SmsCredentialsResponse,
    )
    async def admin_set_sms_credentials(
        app_id: str,
        payload: SmsCredentialsRequest,
        _admin: AuthenticatedUser = Depends(require_admin_user),
    ) -> SmsCredentialsResponse:
        """Store this app's SMS gateway credentials, encrypted at rest.

        Write-only: the plaintext is never echoed back, the audit entry records
        only that credentials changed (never the values), and the request is
        refused if no encryption key is configured (never store what we cannot
        protect or rotate).
        """
        # api_url and message_template are validated in SmsCredentialsRequest
        # (both surface a 422). Here we only guard the server-side prerequisite.
        if not sms_encryption_configured():
            raise HTTPException(
                status_code=400,
                detail="SMS credential encryption is not configured on this server.",
            )

        try:
            ciphertext = encrypt_sms_credentials(payload.model_dump(exclude_none=True))
        except SecretUnavailable as exc:
            raise HTTPException(
                status_code=400,
                detail="SMS credential encryption is not configured on this server.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async with get_session() as session:
            app_record = await session.get(App, _parse_app_uuid(app_id))
            if not app_record:
                raise HTTPException(status_code=404, detail="App not found")
            app_record.sms_credentials_encrypted = ciphertext
            session.add(app_record)
            await session.flush()
            _invalidate_app_cache()

        # Redacted audit entry: record that credentials changed, never values.
        await record_app_change(
            app_id=app_id,
            changed_by=_admin.actor_id or "unknown-admin",
            action="sms_credentials_update",
            changes={"sms_credentials": "updated"},
        )
        logger.info(
            "SMS credentials updated",
            extra={"event": "otp.creds.updated", "app_id": app_id, "changed_by": _admin.actor_id or "unknown-admin"},
        )
        return SmsCredentialsResponse(
            configured=True,
            user_id_hint=mask_user_id(payload.user_id),
        )

    async def _issue_proxy_tokens_for_signed_app(
        *,
        signed_token: str,
        request_domain: str,
        force_workflow: str = "",
    ) -> AppTokenResponse:
        user, app_ctx = await authenticate_app_jwt(
            signed_token=signed_token,
            request_domain=request_domain,
        )

        # Validate force_workflow against allowed_workflows
        if force_workflow and force_workflow not in app_ctx.allowed_workflows:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow '{force_workflow}' is not in allowed_workflows for this app. "
                       f"Allowed: {app_ctx.allowed_workflows}",
            )

        if force_workflow:
            user = AuthenticatedUser(
                actor_id=user.actor_id,
                project_name=user.project_name,
                role=user.role,
                token_id=user.token_id,
                app_id=user.app_id,
                actor_type=user.actor_type,
                allowed_workflows=user.allowed_workflows,
                allowed_tools=user.allowed_tools,
                force_workflow=force_workflow,
            )

        access_token, access_ttl_s = issue_access_token(user)
        refresh_token, refresh_ttl_s = issue_refresh_token(user)

        raw_config = app_ctx.app_config or {}
        includes = raw_config.get("data_resolvers", [])
        resolved_data = await run_resolvers(includes, actor_id=user.actor_id)

        client_config = _client_safe_app_config(
            raw_config, otp_required=bool(app_ctx.otp_enabled)
        )

        logger.info(
            "Issued app tokens",
            extra={
                "app_id": app_ctx.app_id,
                "app_name": app_ctx.app_name,
                "actor_id": user.actor_id or "(none)",
                "force_workflow": force_workflow or "(none)",
            },
        )

        return AppTokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=access_ttl_s,
            refresh_expires_in=refresh_ttl_s,
            app_id=app_ctx.app_id,
            app_name=app_ctx.app_name,
            role=user.role,
            app_config=client_config,
            data=resolved_data,
            force_workflow=force_workflow or None,
        )

    @app.post("/api/auth/app-token", response_model=AppTokenResponse)
    async def issue_app_token(
        payload: AppTokenRequest,
        request: Request,
        response: Response,
    ) -> AppTokenResponse:
        """Issue proxy tokens for an application using a signed app JWT.

        Args:
            payload: App token request body.
            request: Active FastAPI request.
            response: Outgoing response object.

        Returns:
            Access and refresh tokens scoped to the registered application.
        """
        rate_limit = await enforce_rate_limit(
            subject=_client_host(request),
            scope="auth-app-token",
        )
        _apply_rate_limit_headers(response, rate_limit)

        request_domain = request.headers.get("origin") or request.headers.get("referer") or ""
        # Strip trailing slash if present on referer
        if request_domain.endswith("/"):
            request_domain = request_domain[:-1]

        return await _issue_proxy_tokens_for_signed_app(
            signed_token=payload.signed_token,
            request_domain=request_domain,
        )

    @app.post("/api/auth/app-user-token", response_model=AppTokenResponse)
    async def mint_app_user_token(
        payload: AppUserTokenMintRequest,
        request: Request,
        response: Response,
    ) -> AppTokenResponse:
        """Issue app-user tokens from a signed app JWT.

        This is the browser-safe production flow: the app backend mints a short-
        lived signed JWT and the client exchanges it here. Private-key signing
        convenience lives under ``/api/auth/dev/*`` endpoints.
        """
        rate_limit = await enforce_rate_limit(
            subject=_client_host(request),
            scope="auth-app-token",
        )
        _apply_rate_limit_headers(response, rate_limit)

        request_domain = request.headers.get("origin") or request.headers.get("referer") or ""
        if request_domain.endswith("/"):
            request_domain = request_domain[:-1]

        return await _issue_proxy_tokens_for_signed_app(
            signed_token=payload.signed_token.strip(),
            request_domain=request_domain,
            force_workflow=payload.force_workflow or "",
        )

    @app.post("/api/auth/dev/app-user-token", response_model=AppTokenResponse)
    async def mint_dev_app_user_token(
        payload: DevAppTokenMintRequest,
        request: Request,
        response: Response,
    ) -> AppTokenResponse:
        """Development-only convenience endpoint that signs an app JWT from a private key."""
        rate_limit = await enforce_rate_limit(
            subject=_client_host(request),
            scope="auth-app-token",
        )
        _apply_rate_limit_headers(response, rate_limit)

        if "-----BEGIN PRIVATE KEY-----" not in payload.private_key:
            raise HTTPException(status_code=400, detail="Invalid PEM formatted private key")

        signed_token = _sign_app_jwt(
            app_id=payload.app_id,
            private_key_pem=payload.private_key,
            ttl_seconds=payload.jwt_ttl_seconds,
            actor_id=payload.actor_id,
            actor_type=payload.actor_type,
        )

        request_domain = payload.origin
        if not request_domain:
            request_domain = request.headers.get("origin") or request.headers.get("referer") or ""
        if request_domain.endswith("/"):
            request_domain = request_domain[:-1]

        return await _issue_proxy_tokens_for_signed_app(
            signed_token=signed_token,
            request_domain=request_domain,
            force_workflow=payload.force_workflow or "",
        )

    @app.post("/api/auth/dev/admin-token", response_model=AppTokenResponse)
    async def mint_dev_admin_token(
        payload: DevAdminTokenMintRequest,
        request: Request,
        response: Response,
    ) -> AppTokenResponse:
        """Development-only convenience endpoint that signs an admin JWT from a private key."""
        rate_limit = await enforce_rate_limit(
            subject=_client_host(request),
            scope="auth-app-token",
        )
        _apply_rate_limit_headers(response, rate_limit)

        if "-----BEGIN PRIVATE KEY-----" not in payload.private_key:
            raise HTTPException(status_code=400, detail="Invalid PEM formatted private key")

        signed_token = _sign_app_jwt(
            app_id=payload.app_id,
            private_key_pem=payload.private_key,
            ttl_seconds=payload.jwt_ttl_seconds,
            actor_id=payload.actor_id,
            actor_type="admin",
        )

        request_domain = payload.origin
        if not request_domain:
            request_domain = request.headers.get("origin") or request.headers.get("referer") or ""
        if request_domain.endswith("/"):
            request_domain = request_domain[:-1]

        return await _issue_proxy_tokens_for_signed_app(
            signed_token=signed_token,
            request_domain=request_domain,
        )

    # ── Admin Dashboard Auth ──────────────────────────────────────────────

    @app.post("/api/auth/admin/login", response_model=AdminLoginResponse)
    async def admin_login(
        payload: AdminLoginRequest,
        request: Request,
        response: Response,
    ) -> AdminLoginResponse:
        """Authenticate an admin dashboard user with email + password."""
        from backend_proxy.admin_auth import admin_login as _admin_login

        rate_limit = await enforce_rate_limit(
            subject=_client_host(request),
            scope="admin-login",
        )
        _apply_rate_limit_headers(response, rate_limit)

        try:
            result = await _admin_login(payload.email, payload.password)
        except Exception as exc:
            # ponytail: AdminAuthError is a subclass of Exception with status_code
            if hasattr(exc, "status_code"):
                raise HTTPException(
                    status_code=getattr(exc, "status_code"),
                    detail={"code": getattr(exc, "code", "AUTH_ERROR"), "message": str(exc)},
                )
            raise HTTPException(status_code=401, detail=str(exc))

        return AdminLoginResponse(**result)

    @app.post("/api/auth/admin/refresh", response_model=AdminRefreshResponse)
    async def admin_refresh(
        payload: AdminRefreshRequest,
        request: Request,
        response: Response,
    ) -> AdminRefreshResponse:
        """Refresh an admin access token."""
        from backend_proxy.admin_auth import admin_refresh as _admin_refresh

        rate_limit = await enforce_rate_limit(
            subject=_client_host(request),
            scope="admin-refresh",
        )
        _apply_rate_limit_headers(response, rate_limit)

        try:
            result = await _admin_refresh(payload.refresh_token)
        except Exception as exc:
            if hasattr(exc, "status_code"):
                raise HTTPException(
                    status_code=getattr(exc, "status_code"),
                    detail={"code": getattr(exc, "code", "AUTH_ERROR"), "message": str(exc)},
                )
            raise HTTPException(status_code=401, detail=str(exc))

        return AdminRefreshResponse(**result)

    @app.get("/api/auth/admin/me", response_model=AdminUserProfile)
    async def admin_me(
        request: Request,
        admin_user: AdminUserContext = Depends(get_admin_user),
    ) -> AdminUserProfile:
        """Return the current admin user's profile."""
        return AdminUserProfile(
            id=admin_user.admin_user_id,
            email=admin_user.email,
            display_name=admin_user.display_name,
            admin_role=admin_user.admin_role,
            app_id=admin_user.app_id or None,
        )

    # ── Admin User Management (super_admin only) ─────────────────────────

    @app.get("/api/admin/users", response_model=List[AdminUserSummary])
    async def admin_list_users(
        _admin: AdminUserContext = Depends(require_super_admin),
    ) -> List[AdminUserSummary]:
        """List all admin users (super_admin only)."""
        from db.models.admin_user import AdminUser

        async with get_session() as session:
            result = await session.execute(
                select(AdminUser).order_by(AdminUser.created_at.desc())
            )
            users = result.scalars().all()

        return [
            AdminUserSummary(
                id=str(u.id),
                email=u.email,
                display_name=u.display_name,
                role=u.role,
                app_id=str(u.app_id) if u.app_id else None,
                is_active=u.is_active,
                last_login_at=u.last_login_at.isoformat() if u.last_login_at else None,
                created_at=u.created_at.isoformat() if u.created_at else None,
            )
            for u in users
        ]

    @app.post("/api/admin/users", response_model=AdminUserSummary)
    async def admin_create_user(
        payload: AdminUserCreateRequest,
        request: Request,
        _admin: AdminUserContext = Depends(require_super_admin),
    ) -> AdminUserSummary:
        """Create a new admin user."""
        from db.models.admin_user import AdminUser

        async with get_session() as session:
            # Check duplicate email
            existing = await session.execute(
                select(AdminUser).where(AdminUser.email == payload.email)
            )
            if existing.scalar_one_or_none():
                raise HTTPException(status_code=409, detail="Email already exists")

            admin = AdminUser(
                email=payload.email,
                password_hash=AdminUser.hash_password(payload.password),
                display_name=payload.display_name,
                role=payload.role,
                app_id=uuid.UUID(payload.app_id) if payload.app_id else None,
            )
            session.add(admin)
            await session.flush()
            await session.refresh(admin)

            from backend_proxy.admin_auth import log_admin_action
            await log_admin_action(
                admin_user_id=_admin.admin_user_id,
                action="user.create",
                target_type="admin_user",
                target_id=str(admin.id),
                details={"email": admin.email, "role": admin.role},
                ip_address=request.client.host if request.client else None,
            )

            return AdminUserSummary(
                id=str(admin.id),
                email=admin.email,
                display_name=admin.display_name,
                role=admin.role,
                app_id=str(admin.app_id) if admin.app_id else None,
                is_active=admin.is_active,
                last_login_at=None,
                created_at=admin.created_at.isoformat() if admin.created_at else None,
            )

    @app.get("/api/admin/users/{user_id}", response_model=AdminUserSummary)
    async def admin_get_user(
        user_id: str,
        _admin: AdminUserContext = Depends(require_super_admin),
    ) -> AdminUserSummary:
        """Get a single admin user by ID."""
        from db.models.admin_user import AdminUser

        async with get_session() as session:
            admin = await session.get(AdminUser, uuid.UUID(user_id))
            if not admin:
                raise HTTPException(status_code=404, detail="User not found")

        return AdminUserSummary(
            id=str(admin.id),
            email=admin.email,
            display_name=admin.display_name,
            role=admin.role,
            app_id=str(admin.app_id) if admin.app_id else None,
            is_active=admin.is_active,
            last_login_at=admin.last_login_at.isoformat() if admin.last_login_at else None,
            created_at=admin.created_at.isoformat() if admin.created_at else None,
        )

    @app.patch("/api/admin/users/{user_id}", response_model=AdminUserSummary)
    async def admin_update_user(
        user_id: str,
        payload: AdminUserUpdateRequest,
        request: Request,
        _admin: AdminUserContext = Depends(require_super_admin),
    ) -> AdminUserSummary:
        """Update an admin user (role, active status, app assignment)."""
        from db.models.admin_user import AdminUser

        async with get_session() as session:
            admin = await session.get(AdminUser, uuid.UUID(user_id))
            if not admin:
                raise HTTPException(status_code=404, detail="User not found")

            updates = payload.model_dump(exclude_unset=True)
            if "display_name" in updates:
                admin.display_name = updates["display_name"]
            if "role" in updates:
                admin.role = updates["role"]
            if "app_id" in updates:
                admin.app_id = uuid.UUID(updates["app_id"]) if updates["app_id"] else None
            if "is_active" in updates:
                admin.is_active = updates["is_active"]

            session.add(admin)
            await session.flush()
            await session.refresh(admin)

        from backend_proxy.admin_auth import log_admin_action
        await log_admin_action(
            admin_user_id=_admin.admin_user_id,
            action="user.update",
            target_type="admin_user",
            target_id=user_id,
            details=updates,
            ip_address=request.client.host if request.client else None,
        )

        return AdminUserSummary(
            id=str(admin.id),
            email=admin.email,
            display_name=admin.display_name,
            role=admin.role,
            app_id=str(admin.app_id) if admin.app_id else None,
            is_active=admin.is_active,
            last_login_at=admin.last_login_at.isoformat() if admin.last_login_at else None,
            created_at=admin.created_at.isoformat() if admin.created_at else None,
        )

    @app.delete("/api/admin/users/{user_id}", status_code=204)
    async def admin_delete_user(
        user_id: str,
        request: Request,
        _admin: AdminUserContext = Depends(require_super_admin),
    ) -> None:
        """Soft-delete (deactivate) an admin user."""
        from db.models.admin_user import AdminUser

        async with get_session() as session:
            admin = await session.get(AdminUser, uuid.UUID(user_id))
            if not admin:
                raise HTTPException(status_code=404, detail="User not found")
            admin.is_active = False
            session.add(admin)
            await session.commit()

        from backend_proxy.admin_auth import log_admin_action
        await log_admin_action(
            admin_user_id=_admin.admin_user_id,
            action="user.deactivate",
            target_type="admin_user",
            target_id=user_id,
            ip_address=request.client.host if request.client else None,
        )

    # ── Audit Log (super_admin only) ──────────────────────────────────────

    @app.get("/api/admin/audit-log")
    async def admin_audit_log(
        limit: int = 50,
        offset: int = 0,
        action: str | None = None,
        admin_user_id: str | None = None,
        _admin: AdminUserContext = Depends(require_super_admin),
    ) -> dict:
        """Query audit log entries (paginated, filterable)."""
        from db.models.audit_log import AdminAuditLog

        limit = max(1, min(limit, 500))
        offset = max(0, offset)

        async with get_session() as session:
            from sqlalchemy import select, func as sql_func

            stmt = select(AdminAuditLog).order_by(AdminAuditLog.created_at.desc())
            count_stmt = select(sql_func.count(AdminAuditLog.id))

            if action:
                stmt = stmt.where(AdminAuditLog.action == action)
                count_stmt = count_stmt.where(AdminAuditLog.action == action)
            if admin_user_id:
                stmt = stmt.where(AdminAuditLog.admin_user_id == uuid.UUID(admin_user_id))
                count_stmt = count_stmt.where(AdminAuditLog.admin_user_id == uuid.UUID(admin_user_id))

            total = (await session.execute(count_stmt)).scalar() or 0
            rows = (await session.execute(stmt.limit(limit).offset(offset))).scalars().all()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "entries": [
                {
                    "id": e.id,
                    "admin_user_id": str(e.admin_user_id) if e.admin_user_id else None,
                    "action": e.action,
                    "target_type": e.target_type,
                    "target_id": e.target_id,
                    "details": e.details,
                    "ip_address": e.ip_address,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in rows
            ],
        }

    @app.post("/api/auth/dev/signed-token", response_model=DevSignedTokenResponse)
    async def mint_dev_signed_token(
        payload: DevSignedTokenRequest,
        request: Request,
        response: Response,
    ) -> DevSignedTokenResponse:
        """Development-only convenience endpoint that returns a raw ES256 app-signed JWT.

        Unlike the other dev endpoints this does **not** exchange the signed
        token for access/refresh proxy tokens.  The returned JWT can be passed
        to ``/api/auth/app-token`` or ``/api/auth/app-user-token`` manually.
        """
        rate_limit = await enforce_rate_limit(
            subject=_client_host(request),
            scope="auth-app-token",
        )
        _apply_rate_limit_headers(response, rate_limit)

        if "-----BEGIN PRIVATE KEY-----" not in payload.private_key:
            raise HTTPException(status_code=400, detail="Invalid PEM formatted private key")

        signed_token = _sign_app_jwt(
            app_id=payload.app_id,
            private_key_pem=payload.private_key,
            ttl_seconds=payload.jwt_ttl_seconds,
            actor_id=payload.actor_id,
            actor_type=payload.actor_type,
        )

        return DevSignedTokenResponse(
            signed_token=signed_token,
            expires_in=payload.jwt_ttl_seconds,
            app_id=payload.app_id,
        )

    @app.post("/api/auth/refresh", response_model=RefreshTokenResponse)
    async def refresh_token(
        payload: RefreshTokenRequest,
        request: Request,
        response: Response,
    ) -> RefreshTokenResponse:
        """Refresh an access token using a valid refresh JWT.

        Args:
            payload: Refresh request body.
            request: Active FastAPI request.
            response: Outgoing response object.

        Returns:
            Fresh access token response.
        """
        rate_limit = await enforce_rate_limit(
            subject=_client_host(request),
            scope="auth-refresh",
        )
        _apply_rate_limit_headers(response, rate_limit)

        user = decode_token(payload.refresh_token, expected_type="refresh")
        access_token, access_ttl_s = issue_access_token(user)
        return RefreshTokenResponse(
            access_token=access_token,
            expires_in=access_ttl_s,
        )

    @app.post("/api/auth/otp/request", response_model=OtpRequestResponse)
    async def request_otp(
        payload: OtpRequestRequest,
        request: Request,
        response: Response,
        user: AuthenticatedUser = Depends(_get_current_user),
    ) -> OtpRequestResponse:
        """Generate and SMS an OTP to a guest's mobile number."""
        # A real app_id is required — never fall back to a shared "default"
        # namespace for security-sensitive OTP/budget keys.
        if not user.app_id or not _otp_enabled_for_app(user.app_id):
            logger.info(
                "OTP request rejected for non-OTP app",
                extra={"event": "otp.disabled_attempt", "app_id": user.app_id or ""},
            )
            raise HTTPException(
                status_code=403,
                detail="OTP verification is not enabled for this app.",
            )
        app_id = user.app_id
        policy = _resolve_otp_policy(app_id)
        mobile_suffix = payload.phone[-4:]

        rate_limit = await enforce_rate_limit(
            subject=_client_host(request),
            scope="auth-otp-request",
            rate_limit_override=_rate_limit_override_for_app(app_id),
        )
        _apply_rate_limit_headers(response, rate_limit)

        cooldown = await otp_service.remaining_cooldown(app_id, payload.phone)
        if cooldown > 0:
            logger.info(
                "OTP request throttled (cooldown)",
                extra={"event": "otp.request.throttled", "app_id": app_id, "reason": "cooldown", "mobile_suffix": mobile_suffix},
            )
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {cooldown}s before requesting a new OTP.",
                headers={"Retry-After": str(cooldown)},
            )

        if not await otp_service.register_request(app_id, payload.phone, policy):
            logger.warning(
                "OTP request throttled (hourly cap)",
                extra={"event": "otp.request.throttled", "app_id": app_id, "reason": "hourly_cap", "mobile_suffix": mobile_suffix},
            )
            raise HTTPException(
                status_code=429,
                detail="Too many OTP requests for this number. Please try again later.",
            )

        # Per-app daily SMS budget (cost/abuse cap across distinct numbers).
        if not await otp_service.consume_sms_budget(app_id, policy):
            await otp_service.release_request(app_id, payload.phone)
            logger.warning(
                "OTP daily SMS budget exhausted",
                extra={"event": "otp.budget.exhausted", "app_id": app_id},
            )
            raise HTTPException(
                status_code=429,
                detail="This service has reached its OTP limit for now. Please try again later.",
            )

        logger.info(
            "OTP requested",
            extra={"event": "otp.requested", "app_id": app_id, "mobile_suffix": mobile_suffix},
        )
        otp = await otp_service.create_otp(app_id, payload.phone, policy)
        try:
            credentials = await _fetch_sms_credentials(app_id)
            await send_otp_sms(
                payload.phone,
                otp,
                user_id=str(credentials.get("user_id", "")),
                password=str(credentials.get("password", "")),
                api_url=credentials.get("api_url"),
                message_template=credentials.get("message_template"),
            )
        except (SecretUnavailable, TemplateError, GatewayUrlError, SQLAlchemyError) as exc:
            # Server-side fault we own — missing/undecryptable key, a bad stored
            # template/gateway URL, or a DB error. Nothing was billed and the
            # guest did nothing wrong, so fully roll back (incl. cooldown + hourly
            # slot). Fail closed: the gate stays shut, OTP is never bypassed.
            logger.error(
                "OTP send blocked by server-side fault for app %s: %s", app_id, exc,
                extra={"event": "otp.failclosed", "app_id": app_id},
            )
            await otp_service.clear_otp(app_id, payload.phone)
            await otp_service.release_request(app_id, payload.phone)
            await otp_service.refund_sms_budget(app_id)
            raise HTTPException(
                status_code=502,
                detail="OTP delivery is not configured. Please contact support.",
            )
        except Exception as exc:  # noqa: BLE001
            # Gateway *delivery* failure (attacker-forcible, e.g. an undeliverable
            # number). KEEP the per-mobile cooldown + hourly slot consumed so
            # forced failures cannot bypass those throttles; only the daily budget
            # is refunded since no SMS was actually delivered.
            logger.error(
                "OTP dispatch failed: %s", exc,
                extra={"event": "otp.sms_failed", "app_id": app_id},
            )
            await otp_service.clear_otp_only(app_id, payload.phone)
            await otp_service.refund_sms_budget(app_id)
            raise HTTPException(
                status_code=502,
                detail="Could not send the OTP right now. Please try again.",
            )

        logger.info(
            "OTP sent",
            extra={"event": "otp.sent", "app_id": app_id, "mobile_suffix": mobile_suffix},
        )
        return OtpRequestResponse(
            expires_in=policy.ttl_seconds,
            resend_after=policy.resend_cooldown_seconds,
        )

    @app.post("/api/auth/otp/verify", response_model=OtpVerifyResponse)
    async def verify_otp(
        payload: OtpVerifyRequest,
        request: Request,
        response: Response,
        user: AuthenticatedUser = Depends(_get_current_user),
    ) -> OtpVerifyResponse:
        """Verify a submitted OTP and re-issue phone-verified proxy tokens."""
        if not user.app_id or not _otp_enabled_for_app(user.app_id):
            raise HTTPException(
                status_code=403,
                detail="OTP verification is not enabled for this app.",
            )
        app_id = user.app_id
        policy = _resolve_otp_policy(app_id)

        rate_limit = await enforce_rate_limit(
            subject=_client_host(request),
            scope="auth-otp-verify",
            rate_limit_override=_rate_limit_override_for_app(app_id),
        )
        _apply_rate_limit_headers(response, rate_limit)

        result = await otp_service.verify_otp(app_id, payload.phone, payload.otp, policy)
        if not result.ok:
            messages = {
                "OTP_EXPIRED": "Your OTP has expired. Please request a new one.",
                "OTP_TOO_MANY_ATTEMPTS": "Too many incorrect attempts. Please request a new OTP.",
                "OTP_LOCKED": "Too many attempts. Please try again later.",
                "OTP_INVALID": (
                    f"Incorrect OTP. {result.attempts_remaining} attempt(s) left."
                    if result.attempts_remaining
                    else "Incorrect OTP."
                ),
            }
            status_code = 429 if result.code == "OTP_LOCKED" else 400
            event = "otp.verify.locked" if result.code == "OTP_LOCKED" else "otp.verify.failed"
            logger.info(
                "OTP verification failed",
                extra={"event": event, "app_id": app_id, "code": result.code, "mobile_suffix": payload.phone[-4:]},
            )
            raise HTTPException(
                status_code=status_code,
                detail=messages.get(result.code, "OTP verification failed."),
            )

        verified_user = mark_phone_verified(user, payload.phone)
        access_token, access_ttl_s = issue_access_token(verified_user)
        refresh_token, refresh_ttl_s = issue_refresh_token(verified_user)
        logger.info(
            "Phone verified",
            extra={"event": "otp.verify.success", "app_id": app_id, "mobile_suffix": payload.phone[-4:]},
        )
        return OtpVerifyResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=access_ttl_s,
            refresh_expires_in=refresh_ttl_s,
            phone=payload.phone,
        )

    @app.post("/api/chat/message", response_model=ChatMessageResponse)
    async def chat_message(
        payload: ChatMessageRequest,
        request: Request,
        response: Response,
        user: AuthenticatedUser = Depends(_get_current_user),
    ) -> ChatMessageResponse:
        """Run the full AI chat pipeline through the private MCP server.

        Invokes the ``chat_complete`` tool which runs the LangGraph
        orchestration pipeline: intent classification → cache check →
        data fetch → context build → LLM reasoning → cache store.

        Args:
            payload: Chat message request body.
            request: Active FastAPI request.
            response: Outgoing response object.
            user: Authenticated proxy user.

        Returns:
            AI-generated response with workflow metadata.
        """
        _require_phone_verified(user)

        rate_limit = await enforce_rate_limit(
            subject=user.actor_id or user.phone or _client_host(request),
            scope="chat-message",
        )
        _apply_rate_limit_headers(response, rate_limit)

        if payload.stream:
            raise HTTPException(
                status_code=400,
                detail="Use POST /api/chat/message/stream for SSE streaming responses.",
            )

        effective_session_id = (
            _validated_client_session_id(payload.session_id, user)
            or phone_session_id(user)
            or user.actor_id
            or request.state.correlation_id
        )
        downstream = await _call_chat_complete(
            message=payload.message,
            user=user,
            correlation_id=request.state.correlation_id,
            session_id=effective_session_id,
            force_workflow=user.force_workflow or None,
        )
        ds_workflow = downstream.get("workflow", "general")
        return ChatMessageResponse(
            response=downstream.get("response", ""),
            workflow=ds_workflow,
            session_id=effective_session_id,
            cache_hit=downstream.get("cache_hit", False),
            lead_progress=downstream.get("lead_progress"),
            helper_buttons=_resolve_helper_buttons(
                ds_workflow,
                downstream.get("lead_progress"),
                payload.message,
            ),
            token_usage=downstream.get("token_usage", {}),
        )

    @app.post("/api/chat/message/stream")
    async def chat_message_stream(
        payload: ChatMessageRequest,
        request: Request,
        user: AuthenticatedUser = Depends(_get_current_user),
    ) -> StreamingResponse:
        """Run the chat pipeline and stream SSE events to the client."""
        _require_phone_verified(user)

        rate_limit = await enforce_rate_limit(
            subject=user.actor_id or user.phone or _client_host(request),
            scope="chat-message",
        )

        stream = StreamingResponse(
            _stream_chat_events(
                message=payload.message,
                user=user,
                correlation_id=request.state.correlation_id,
                session_id=_validated_client_session_id(payload.session_id, user)
                or phone_session_id(user)
                or user.actor_id
                or request.state.correlation_id,
                force_workflow=user.force_workflow or None,
            ),
            media_type="text/event-stream",
        )
        stream.headers["Cache-Control"] = "no-cache"
        stream.headers["Connection"] = "keep-alive"
        stream.headers["X-Accel-Buffering"] = "no"
        _apply_rate_limit_headers(stream, rate_limit)
        return stream

    @app.post("/api/chat/feedback", response_model=ChatFeedbackResponse)
    async def chat_feedback(
        payload: ChatFeedbackRequest,
        request: Request,
        response: Response,
        user: AuthenticatedUser = Depends(_get_current_user),
    ) -> ChatFeedbackResponse:
        """Store one feedback submission for a chat response."""

        if not user.app_id:
            raise HTTPException(status_code=400, detail="Feedback requires an app-scoped token.")

        rate_limit = await enforce_rate_limit(
            subject=user.actor_id or payload.session_id or _client_host(request),
            scope="chat-feedback",
        )
        _apply_rate_limit_headers(response, rate_limit)

        feedback_id = await record_feedback(
            app_id=user.app_id,
            actor_id=user.actor_id,
            session_id=payload.session_id,
            feedback_flag=payload.feedback_flag,
            feedback_text=payload.feedback_text,
            user_message=payload.user_message,
            response=payload.response,
        )
        return ChatFeedbackResponse(id=feedback_id)

    @app.get("/api/feedback/all", response_model=FeedbackListResponse)
    async def admin_list_feedback(
        request: Request,
        query: FeedbackListQuery = Depends(),
        _admin: AdminUserContext | AuthenticatedUser = Depends(require_admin_hybrid),
    ) -> FeedbackListResponse:
        """Return feedback rows for admins in paginated or recent-N mode."""

        if query.last is not None and ({"page", "page_size"} & set(request.query_params.keys())):
            raise HTTPException(
                status_code=400,
                detail="last cannot be combined with pagination parameters",
            )
        try:
            query.validate_date_range()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        result = await list_feedback(
            page=query.page,
            page_size=query.page_size,
            last=query.last,
            actor_id=query.actor_id,
            start_date=query.start_date,
            end_date=query.end_date,
        )
        return FeedbackListResponse(**result)

    @app.get("/api/feedback/{feedback_id}", response_model=FeedbackRecordResponse)
    async def admin_get_feedback(
        feedback_id: str,
        _admin: AdminUserContext | AuthenticatedUser = Depends(require_admin_hybrid),
    ) -> FeedbackRecordResponse:
        """Return one stored feedback record for admin review."""

        feedback = await get_feedback_by_id(feedback_id)
        if feedback is None:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return FeedbackRecordResponse(**feedback)

    @app.post("/api/mcp/tools/{tool_name}", response_model=ToolCallResponse)
    async def proxy_tool_call(
        tool_name: str,
        payload: ToolCallRequest,
        request: Request,
        response: Response,
        user: AuthenticatedUser = Depends(_get_current_user),
    ) -> ToolCallResponse:
        """Call an allowlisted MCP tool through the proxy.

        Args:
            tool_name: Downstream tool name.
            payload: Tool request body.
            request: Active FastAPI request.
            response: Outgoing response object.
            user: Authenticated proxy user.

        Returns:
            Sanitised downstream tool payload.
        """
        # Gate here too so a guest can't bypass /api/chat/* via a chat-capable tool.
        _require_phone_verified(user)
        _authorise_tool(tool_name, payload.arguments, user)
        rate_limit = await enforce_rate_limit(
            subject=f"{user.actor_id or user.phone or _client_host(request)}:{tool_name}",
            scope="tool-call",
        )
        _apply_rate_limit_headers(response, rate_limit)

        result = await mcp_gateway.call_tool(
            tool_name=tool_name,
            arguments=payload.arguments,
            correlation_id=request.state.correlation_id,
        )

        return ToolCallResponse(
            tool=tool_name,
            data=sanitize_payload(result),
        )

    return app


app = create_app()


if __name__ == "__main__":
    logger.info(
        "Starting LPUAI backend proxy",
        extra={"port": settings.proxy_port},
    )
    import uvicorn
    uvicorn.run(app, host=settings.proxy_host, port=settings.proxy_port)
