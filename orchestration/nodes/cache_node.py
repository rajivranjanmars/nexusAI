"""
Node: Semantic cache check and store for workflow responses.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, cast

from cache_module.semantic_cache import check_cache, store_response
from cache_module.vector_store import _embed
from db.app_registry import resolve_by_app_id
from orchestration.state import WorkflowState
from orchestration.workflow_policy import normalize_workflow_name, should_use_response_cache
from orchestration.workflow_config import get_workflow_config, resolve_cache_policy
from shared.logger import get_logger

logger = get_logger(__name__)

CacheScope = Literal["shared", "actor"]


def _caching_enabled_for_state(state: WorkflowState) -> bool:
    """Whether semantic response caching is turned on for this app.

    Defaults to enabled (no app context, e.g. no ``app_id``) so caching keeps
    working for callers outside the App Registry.
    """
    app_id = state.get("app_id")
    if not app_id:
        return True
    app_context = resolve_by_app_id(app_id)
    return bool(app_context is None or app_context.caching_enabled)


def _cache_scope_for_state(state: WorkflowState) -> CacheScope:
    workflow = normalize_workflow_name(state.get("detected_workflow") or "general")
    app_id = state.get("app_id")
    config = get_workflow_config(app_id)
    policy = resolve_cache_policy(config, workflow)
    return policy.scope


def _resolve_cache_config(state: WorkflowState) -> tuple[int, float]:
    app_id = state.get("app_id")
    workflow = normalize_workflow_name(state.get("detected_workflow") or "general")
    config = get_workflow_config(app_id)
    policy = resolve_cache_policy(config, workflow)

    # Per-workflow TTL > app-level TTL > hardcoded default
    ttl_seconds = policy.ttl_seconds or 86400
    if not policy.ttl_seconds and app_id:
        app_context = resolve_by_app_id(app_id)
        if app_context:
            ttl_seconds = app_context.cache_ttl_seconds

    similarity_threshold = policy.similarity_threshold or config.cache_default_similarity_threshold
    return ttl_seconds, similarity_threshold


def _build_tool_results_snapshot(state: WorkflowState, cache_scope: CacheScope) -> Dict[str, Any]:
    rag_context = state.get("rag_context") or []
    tool_calls = state.get("tool_calls") or []
    actor_id = state.get("actor_id") or state.get("student_id")

    snapshot: Dict[str, Any] = {
        "cache_scope": cache_scope,
        "workflow": state.get("detected_workflow"),
        "rag_source_urls": [item.get("source_url") for item in rag_context if item.get("source_url")],
        "rag_result_count": len(rag_context),
        "tool_call_count": len(tool_calls),
    }
    if cache_scope == "actor" and actor_id:
        snapshot["actor_id"] = actor_id
    return snapshot


async def cache_check_node(state: WorkflowState) -> WorkflowState:
    workflow = state.get("detected_workflow")
    user_input = state.get("user_input")
    if not workflow or not user_input:
        state["cache_hit"] = False
        return state

    actor_id = state.get("actor_id") or state.get("student_id")
    app_id = state.get("app_id")
    if not should_use_response_cache(workflow, actor_id, app_id=app_id):
        logger.info(
            "cache_check_node: skipped — cache disabled for workflow=%s",
            workflow,
        )
        state["cache_hit"] = False
        return state

    if not _caching_enabled_for_state(state):
        state["cache_hit"] = False
        return state

    cache_scope = _cache_scope_for_state(state)
    _, similarity_threshold = _resolve_cache_config(state)

    try:
        question_embedding = state.get("query_embedding")
        if not question_embedding:
            question_embedding = await _embed(user_input)
            state["query_embedding"] = question_embedding

        hit = await check_cache(
            app_id=state.get("app_id"),
            workflow=workflow,
            question=user_input,
            actor_id=actor_id,
            cache_scope=cache_scope,
            threshold=similarity_threshold,
            question_embedding=question_embedding,
        )
        if hit:
            state["cache_hit"] = True
            state["llm_response"] = str(hit["response_text"])
            # Restore source URLs from cached tool_results so they are
            # available for the proxy to emit in the SSE stream.
            tool_results = hit.get("tool_results") or {}
            cached_urls = tool_results.get("rag_source_urls") or []
            if cached_urls:
                state["rag_context"] = [{"source_url": u} for u in cached_urls]
            state["metadata"] = {
                **(state.get("metadata") or {}),
                "cache_scope": cache_scope,
                "cache_similarity": hit.get("similarity"),
            }
            logger.info(
                "cache_check_node: cache HIT for workflow=%s scope=%s similarity=%.3f",
                workflow,
                cache_scope,
                hit.get("similarity", 0.0),
            )
            return state
        logger.info(
            "cache_check_node: cache MISS for workflow=%s scope=%s threshold=%.2f",
            workflow,
            cache_scope,
            similarity_threshold,
        )
    except Exception as exc:
        logger.warning("cache_check_node failed open: %s", exc)

    state["cache_hit"] = False
    return state


async def cache_store_node(state: WorkflowState) -> WorkflowState:
    if state.get("cache_hit"):
        return state

    workflow = state.get("detected_workflow")
    user_input = state.get("user_input")
    response_text = state.get("llm_response")
    if not workflow or not user_input or not response_text:
        return state

    actor_id = state.get("actor_id") or state.get("student_id")
    app_id = state.get("app_id")
    if not should_use_response_cache(workflow, actor_id, app_id=app_id):
        return state

    if not _caching_enabled_for_state(state):
        return state

    if state.get("error"):
        return state

    cache_scope = _cache_scope_for_state(state)
    ttl_seconds, _ = _resolve_cache_config(state)
    invalidated_by = ["student_update"] if cache_scope == "actor" else ["rag_reingest"]

    try:
        await store_response(
            app_id=state.get("app_id"),
            workflow=workflow,
            question=user_input,
            response_text=response_text,
            actor_id=actor_id,
            cache_scope=cache_scope,
            tool_results=_build_tool_results_snapshot(state, cache_scope),
            ttl_seconds=ttl_seconds,
            invalidated_by=invalidated_by,
            question_embedding=cast(list[float] | None, state.get("query_embedding")),
        )
        state["metadata"] = {
            **(state.get("metadata") or {}),
            "cache_scope": cache_scope,
            "cache_ttl_seconds": ttl_seconds,
        }
        logger.info(
            "cache_store_node: stored cache entry for workflow=%s scope=%s ttl=%s",
            workflow,
            cache_scope,
            ttl_seconds,
        )
    except Exception as exc:
        logger.warning("cache_store_node failed open: %s", exc)

    return state
