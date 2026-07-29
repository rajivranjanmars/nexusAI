"""
OpenAI-compatible async LLM client.

Works transparently with OpenRouter, OpenAI, Together AI, Groq, Ollama, or
any provider that exposes an OpenAI-compatible ``/v1/chat/completions``
endpoint.  Two convenience methods are exposed:

* ``call_fast``  — routes to ``LLM_FAST_MODEL``  (classification, cheap tasks)
* ``call_smart`` — routes to ``LLM_SMART_MODEL`` (reasoning, complex tasks)

Both support an optional ``model_override`` for per-call flexibility and
include 30 s timeout + 3-attempt exponential-backoff retry.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from shared.config import settings
from shared.logger import get_logger
from shared.openai_client import get_llm_client

logger = get_logger(__name__)

# ── Retry configuration ─────────────────────────────────────────────────────

_MAX_RETRIES: int = 3
_TIMEOUT_SECONDS: int = 30
_BACKOFF_BASE: float = 1.0  # seconds


# ── Internal request helper ────────────────────────────────────────────────


async def _chat(
    model: str,
    prompt: str,
    system: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.1,
    *,
    app_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    actor_type: Optional[str] = None,
    action: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Send a chat completion request with retry + exponential back-off.

    Args:
        model: Model identifier (provider-specific).
        prompt: User message content.
        system: System message content.

    Returns:
        Tuple of ``(response_text, usage_dict)``.  ``usage_dict`` contains
        ``prompt_tokens``, ``completion_tokens``, ``total_tokens``, and
        ``model``.

    Raises:
        openai.OpenAIError: After all retries are exhausted.
    """
    import openai
    client = get_llm_client()
    last_exc: BaseException | None = None

    t0 = time.monotonic()
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            content = response.choices[0].message.content or ""

            # Extract usage metadata from the response.
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
            completion_tokens = getattr(response.usage, "completion_tokens", 0)
            total_tokens = getattr(response.usage, "total_tokens", 0)

            usage: Dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model": model,
            }

            logger.debug(
                "LLM response received",
                extra={
                    "model": model,
                    "attempt": attempt,
                    "tokens": total_tokens,
                    "latency_ms": latency_ms,
                },
            )

            # ── Fire-and-forget: persist token usage ──────────────────────
            try:
                from db.token_tracker import record_usage
                from llm.cost_estimator import estimate_cost

                model_tier = "fast" if action == "detect_workflow" else "smart"
                estimated_cost = estimate_cost(model, prompt_tokens, completion_tokens)

                asyncio.create_task(
                    record_usage(
                        app_id=app_id,
                        actor_id=actor_id,
                        actor_type=actor_type,
                        action=action,
                        model=model,
                        model_tier=model_tier,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        estimated_cost_usd=estimated_cost,
                        cache_hit=False,
                        latency_ms=latency_ms,
                        correlation_id=correlation_id,
                    )
                )
            except Exception as track_exc:
                logger.warning("Token usage tracking failed: %s", track_exc)

            return content.strip(), usage

        except (openai.APITimeoutError, openai.RateLimitError, openai.APIConnectionError) as exc:
            last_exc = exc
            wait = _BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt,
                _MAX_RETRIES,
                wait,
                exc,
            )
            await asyncio.sleep(wait)

        except openai.OpenAIError as exc:
            logger.error("LLM call failed with non-retryable error: %s", exc)
            raise

    # All retries exhausted
    raise openai.OpenAIError(  # type: ignore[call-arg]
        f"LLM call failed after {_MAX_RETRIES} retries: {last_exc}"
    )


# ── Public API ──────────────────────────────────────────────────────────────


async def call_fast(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model_override: Optional[str] = None,
    max_tokens: Optional[int] = None,
    *,
    app_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    actor_type: Optional[str] = None,
    action: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Send a request to the **fast** (cheap) model.

    Args:
        prompt: User message.
        system: System prompt.
        model_override: Override the default fast model for this call.

    Returns:
        Tuple of ``(response_text, usage_dict)``.
    """
    model = model_override or settings.llm_fast_model
    limit = max_tokens or settings.llm_max_tokens_fast
    logger.debug("call_fast → %s (limit: %s)", model, limit)
    return await _chat(
        model=model, prompt=prompt, system=system, max_tokens=limit,
        app_id=app_id, actor_id=actor_id, actor_type=actor_type,
        action=action, correlation_id=correlation_id,
    )


async def call_smart(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model_override: Optional[str] = None,
    max_tokens: Optional[int] = None,
    *,
    app_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    actor_type: Optional[str] = None,
    action: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Send a request to the **smart** (reasoning) model.

    Args:
        prompt: User message.
        system: System prompt.
        model_override: Override the default smart model for this call.

    Returns:
        Tuple of ``(response_text, usage_dict)``.
    """
    model = model_override or settings.llm_smart_model
    limit = max_tokens or settings.llm_max_tokens_smart
    logger.debug("call_smart → %s (limit: %s)", model, limit)
    return await _chat(
        model=model, prompt=prompt, system=system, max_tokens=limit,
        app_id=app_id, actor_id=actor_id, actor_type=actor_type,
        action=action, correlation_id=correlation_id,
    )


# ── Streaming API ───────────────────────────────────────────────────────────


async def _chat_stream(
    model: str,
    prompt: str,
    system: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.1,
    *,
    app_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    actor_type: Optional[str] = None,
    action: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Yield text deltas from a streaming chat completion.

    Each yielded dict is either:
      - ``{"type": "delta", "content": "..."}``   – a token fragment
      - ``{"type": "usage",  "usage": {...}, "full_text": "..."}`` – final usage
    """
    client = get_llm_client()
    t0 = time.monotonic()

    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
    )

    accumulated = ""
    usage: Dict[str, Any] = {}

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            accumulated += delta
            yield {"type": "delta", "content": delta}
        if getattr(chunk, "usage", None):
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens or 0,
                "completion_tokens": chunk.usage.completion_tokens or 0,
                "total_tokens": chunk.usage.total_tokens or 0,
                "model": model,
            }

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Fire-and-forget token tracking
    if usage:
        try:
            from db.token_tracker import record_usage
            from llm.cost_estimator import estimate_cost

            model_tier = "fast" if action == "detect_workflow" else "smart"
            estimated_cost = estimate_cost(
                model,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
            asyncio.create_task(
                record_usage(
                    app_id=app_id,
                    actor_id=actor_id,
                    actor_type=actor_type,
                    action=action,
                    model=model,
                    model_tier=model_tier,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    estimated_cost_usd=estimated_cost,
                    cache_hit=False,
                    latency_ms=latency_ms,
                    correlation_id=correlation_id,
                )
            )
        except Exception as track_exc:
            logger.warning("Token usage tracking (stream) failed: %s", track_exc)

    yield {"type": "usage", "usage": usage, "full_text": accumulated.strip()}


async def call_smart_stream(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model_override: Optional[str] = None,
    max_tokens: Optional[int] = None,
    *,
    app_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    actor_type: Optional[str] = None,
    action: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream a request to the **smart** (reasoning) model.

    Yields the same event dicts as ``_chat_stream``.
    """
    model = model_override or settings.llm_smart_model
    limit = max_tokens or settings.llm_max_tokens_smart
    logger.debug("call_smart_stream → %s (limit: %s)", model, limit)
    async for event in _chat_stream(
        model=model, prompt=prompt, system=system, max_tokens=limit,
        app_id=app_id, actor_id=actor_id, actor_type=actor_type,
        action=action, correlation_id=correlation_id,
    ):
        yield event
