"""Workflow policy helpers for runtime gating and normalization."""

from __future__ import annotations

from typing import FrozenSet, Iterable, Optional

from db.app_registry import resolve_by_app_id

WORKFLOW_ALIASES = {
    "grade_inquiry": "student_grade_inquiry",
}

PUBLIC_RAG_WORKFLOWS: FrozenSet[str] = frozenset({
    "enrollment",
    "admission_inquiry",
    "policy_lookup",
    "resource_request",
    "lead_capture"
})

CONVERSATIONAL_PARENT_WORKFLOWS: FrozenSet[str] = frozenset({
    "general",
})

ACTOR_REQUIRED_WORKFLOWS: FrozenSet[str] = frozenset({
    "student_grade_inquiry",
    "data_mutation",
})


def normalize_workflow_name(workflow: str | None) -> str:
    """Map historical aliases to the canonical workflow name."""
    cleaned = (workflow or "").strip().lower()
    return WORKFLOW_ALIASES.get(cleaned, cleaned)


def normalize_workflow_names(workflows: Iterable[str]) -> FrozenSet[str]:
    """Normalize a workflow iterable into canonical names."""
    return frozenset(
        normalized
        for normalized in (normalize_workflow_name(item) for item in workflows)
        if normalized
    )


def resolve_allowed_workflows(app_id: Optional[str]) -> FrozenSet[str]:
    """Return the normalized workflow allowlist for an app."""
    if not app_id:
        return frozenset()

    app_context = resolve_by_app_id(app_id)
    if not app_context or not app_context.allowed_workflows:
        return frozenset()

    return normalize_workflow_names(app_context.allowed_workflows)


def workflow_requires_actor(workflow: str | None) -> bool:
    """Whether a workflow should never run for guest traffic."""
    return normalize_workflow_name(workflow) in ACTOR_REQUIRED_WORKFLOWS


def workflow_uses_public_rag(workflow: str | None) -> bool:
    """Whether a workflow should stay grounded to public RAG content."""
    return normalize_workflow_name(workflow) in PUBLIC_RAG_WORKFLOWS


def workflow_is_conversational_parent(workflow: str | None) -> bool:
    """Whether a workflow is intended for lightweight triage and clarification."""
    return normalize_workflow_name(workflow) in CONVERSATIONAL_PARENT_WORKFLOWS


def should_use_response_cache(
    workflow: str | None,
    actor_id: str | None,
    app_id: str | None = None,
) -> bool:
    """Decide whether to use the semantic response cache for this workflow.

    Delegates to the per-workflow cache policy defined in the YAML
    configuration (via ``resolve_cache_policy``).  Actor-scoped workflows
    are silently disabled when no ``actor_id`` is present.
    """
    from orchestration.workflow_config import get_workflow_config, resolve_cache_policy

    normalized = normalize_workflow_name(workflow)
    if not normalized:
        return False

    config = get_workflow_config(app_id)
    policy = resolve_cache_policy(config, normalized)

    if policy.policy == "disabled":
        return False
    if policy.scope == "actor" and not actor_id:
        return False
    return True


def resolve_effective_workflow(
    workflow: str | None,
    *,
    app_id: Optional[str],
    actor_id: Optional[str],
) -> tuple[str, Optional[str]]:
    """Clamp a workflow to the app/session policy and return an optional reason."""
    normalized = normalize_workflow_name(workflow) or "general"
    allowed = resolve_allowed_workflows(app_id)

    if not actor_id and workflow_requires_actor(normalized):
        fallback = _fallback_workflow(allowed, actor_id)
        return fallback, (
            f"Workflow '{normalized}' requires actor data; downgraded to '{fallback}'."
        )

    if allowed and normalized not in allowed:
        fallback = _fallback_workflow(allowed, actor_id)
        return fallback, (
            f"Workflow '{normalized}' is not allowed for this app; downgraded to '{fallback}'."
        )

    return normalized, None


def _fallback_workflow(allowed: FrozenSet[str], actor_id: Optional[str]) -> str:
    preferred = (
        ["general", "enrollment", "policy_lookup", "admission_inquiry", "resource_request"]
        if actor_id
        else ["general", "enrollment", "admission_inquiry", "policy_lookup", "resource_request"]
    )
    for candidate in preferred:
        if not allowed or candidate in allowed:
            return candidate
    if not actor_id:
        return "general"
    return sorted(allowed)[0] if allowed else "general"
