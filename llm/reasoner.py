"""
Deep reasoning and response generation.

Uses the **smart** LLM model (via ``call_smart``) to produce a high-quality,
context-aware response for the student's query.

Returns a ``(response_text, usage_dict)`` tuple so that token usage can be
propagated into ``WorkflowState.token_usage`` for tracking and billing.

The streaming variant ``generate_response_stream`` yields token-level deltas
so the backend proxy can deliver true SSE streaming to the client.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Tuple

from shared.logger import get_logger
from llm.llm_client import call_smart, call_smart_stream
from llm.prompt_builder import PromptBuilder

logger = get_logger(__name__)


def _build_prompts(
    context: Dict[str, Any],
    user_query: str,
) -> Tuple[str, str]:
    """Build system + user prompts from context and query.

    Returns:
        Tuple of ``(system_prompt, user_prompt)``.
        When strict RAG has no chunks, ``system_prompt`` is ``""`` and
        ``user_prompt`` contains the grounded fallback text.
    """
    strict_rag = bool(context.get("strict_rag", False))
    workflow = context.get("workflow", "general")
    app_id = context.get("app_id", "")
    
    # Initialize prompt builder
    builder = PromptBuilder(app_id, workflow)
    workflow_config = builder.get_workflow_response_config()
    
    # Prepare student data
    student_data = json.dumps(
        context.get("student", context.get("student_data", {})),
        indent=2,
        default=str,
    )

    # Process RAG context
    rag_context = context.get("rag_context", [])
    answer_confidence = float(context.get("answer_confidence", 1.0) or 0.0)
    if rag_context:
        rag_context, website_content = builder.render_rag_content(rag_context, workflow_config)
        retrieved_context = website_content or json.dumps(rag_context, indent=2, default=str)
    else:
        website_content = ""
        retrieved_context = json.dumps(
            context.get("semantic_results", list(context.values())),
            indent=2,
            default=str,
        )

    # Prepare conversation history
    history = context.get("conversation_history", [])
    history_limit = workflow_config.history_limit
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'You'}: {m['content']}"
        for m in history[-history_limit:]
    ) or "(no prior conversation)"

    # Handle admission inquiry workflow
    if workflow == "admission_inquiry":
        from shared.prompt_loader import render_prompt
        system_prompt, user_prompt = render_prompt(
            "workflow_guest_admissions",
            query=user_query,
            conversation_history=history_text,
            website_content=website_content or "(none)",
            website_content_status="available" if rag_context else "not available",
        )
        # Apply app-specific system prompt configuration
        system_prompt = builder.build_system_prompt(system_prompt)
        return system_prompt, user_prompt

    # Handle strict RAG with no chunks — except lead_capture which should
    # remain conversational even when a specific turn has no RAG data.
    if strict_rag and not rag_context and workflow != "lead_capture":
        logger.info("Strict RAG enabled but no relevant chunks found; returning grounded fallback")
        return ("", _build_no_rag_fallback(workflow))

    # Build prompts based on mode
    # lead_capture and enrollment are conversational workflows — they must always
    # go through build_regular_prompt so their specific yaml templates are loaded.
    if strict_rag and workflow not in ("lead_capture", "enrollment"):
        system_prompt, user_prompt = builder.build_strict_rag_prompt(
            user_query,
            rag_context,
            workflow_config,
            answer_confidence,
            history_text,
        )
    else:
        system_prompt, user_prompt = builder.build_regular_prompt(
            user_query,
            student_data,
            retrieved_context,
            history_text,
            rag_context,
            workflow_config,
            answer_confidence,
            actor_id=context.get("actor_id", ""),
            lead_progress=context.get("lead_progress", {}),
            verified_phone=context.get("verified_phone", ""),
        )

    # Apply app-specific system prompt configuration
    system_prompt = builder.build_system_prompt(system_prompt)

    # Inject conversation history into the user prompt
    if history:
        user_prompt = f"Conversation so far:\n{history_text}\n\n{user_prompt}"

    return system_prompt, user_prompt


def _build_no_rag_fallback(workflow: str) -> str:
    """Return a workflow-aware fallback when strict RAG has no usable chunks."""
    if workflow == "enrollment":
        return (
            "I can help with admissions, eligibility, programs, fees, and next steps. "
            "Tell me which course or admission topic you want to explore first."
        )
    if workflow == "lead_capture":
        return (
            "I can help you with admissions and get you connected to the right next step. "
            "Could you share what program or level you are interested in?"
        )
    return (
        "I can help with admissions, programs, fees, policies, and general university questions. "
        "Tell me what you want to know, and I will guide you."
    )


# ── Non-streaming API ───────────────────────────────────────────────────────


async def generate_response(
    context: Dict[str, Any],
    user_query: str,
    *,
    app_id: str = "",
    actor_id: str = "",
    actor_type: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Generate a reasoned response using the smart LLM model.

    Args:
        context: Merged student context dict (student data + semantic results).
        user_query: The student's original question.
        app_id: Originating application identifier for usage tracking.
        actor_id: Actor identifier for usage tracking.
        actor_type: Actor type for usage tracking.

    Returns:
        Tuple of ``(response_text, usage_dict)``.
    """
    system_prompt, user_prompt = _build_prompts(context, user_query)

    # Early-return for strict RAG fallback (system_prompt is empty)
    if not system_prompt:
        return user_prompt, {}

    response, usage = await call_smart(
        prompt=user_prompt,
        system=system_prompt,
        app_id=app_id,
        actor_id=actor_id,
        actor_type=actor_type,
        action="llm_reasoning",
    )
    logger.info("Generated reasoned response (%d chars)", len(response))
    return response, usage


# ── Streaming API ───────────────────────────────────────────────────────────


async def generate_response_stream(
    context: Dict[str, Any],
    user_query: str,
    *,
    app_id: str = "",
    actor_id: str = "",
    actor_type: str = "",
) -> AsyncIterator[Dict[str, Any]]:
    """Stream a reasoned response token-by-token.

    Yields the same event dicts as ``call_smart_stream``:
      - ``{"type": "delta", "content": "..."}``
      - ``{"type": "usage",  "usage": {...}, "full_text": "..."}``

    For strict-RAG fallback (no chunks found), yields a single delta + usage.
    """
    system_prompt, user_prompt = _build_prompts(context, user_query)

    # Early-return for strict RAG fallback
    if not system_prompt:
        yield {"type": "delta", "content": user_prompt}
        yield {"type": "usage", "usage": {}, "full_text": user_prompt}
        return

    async for event in call_smart_stream(
        prompt=user_prompt,
        system=system_prompt,
        app_id=app_id,
        actor_id=actor_id,
        actor_type=actor_type,
        action="llm_reasoning",
    ):
        yield event
