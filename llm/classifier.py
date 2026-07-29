"""
Intent / workflow classifier.

Uses the fast LLM model (via `call_fast`) to determine which workflow
should handle a given user message, using the dynamic WorkflowConfig.

Falls back to a rule-based classifier when:
- LLM fails
- Output is invalid
- Confidence is below threshold
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from shared.logger import get_logger
from llm.llm_client import call_fast
from orchestration.workflow_config import get_workflow_config, WorkflowConfig
from orchestration.workflow_policy import normalize_workflow_name, resolve_effective_workflow

logger = get_logger(__name__)

_EMPTY_USAGE: Dict[str, Any] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "model": "rule_based",
}

async def classify_intent(
    user_input: str,
    *,
    app_id: str = "",
    actor_id: str = "",
    actor_type: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Classify user input into a workflow label using dynamic config."""

    config = get_workflow_config(app_id)

    try:
        system_prompt = config.classify_system_prompt
        user_prompt = config.classify_user_template.replace("{user_input}", user_input)

        raw, usage = await call_fast(
            prompt=user_prompt,
            system=system_prompt,
            app_id=app_id,
            actor_id=actor_id,
            actor_type=actor_type,
            action="detect_workflow",
        )

        label, confidence = _parse_llm_output(raw)

        if label in config.labels and confidence >= config.confidence_threshold:
            effective_label, policy_reason = resolve_effective_workflow(
                label,
                app_id=app_id or None,
                actor_id=actor_id or None,
            )
            if policy_reason:
                logger.info(policy_reason, extra={"app_id": app_id, "actor_id": actor_id})
            logger.info(
                "LLM classification success",
                extra={
                    "label": effective_label,
                    "confidence": confidence,
                    "model": usage.get("model"),
                    "app_id": app_id
                },
            )
            return effective_label, usage

        logger.warning(
            "LLM classification low confidence or invalid label",
            extra={
                "label": label,
                "confidence": confidence,
                "raw": raw,
                "app_id": app_id
            },
        )

    except Exception as exc:
        logger.warning("LLM classifier failed: %s", exc)

    # Fallback
    return _rule_based_fallback(user_input, config, app_id=app_id, actor_id=actor_id)


def _parse_llm_output(raw: str) -> Tuple[str, float]:
    """Parse structured LLM output safely."""
    try:
        parsed = json.loads(raw)
        label = normalize_workflow_name(str(parsed.get("label", "")).strip().lower())
        confidence = float(parsed.get("confidence", 0.0))
        return label, confidence
    except Exception:
        # Fallback for non-JSON outputs
        cleaned = normalize_workflow_name(raw.strip().lower().replace('"', "").replace("'", ""))
        return cleaned, 0.0


def _rule_based_fallback(
    user_input: str,
    config: WorkflowConfig,
    *,
    app_id: str = "",
    actor_id: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Simple keyword-based fallback classifier using dynamic rules."""
    text = user_input.lower()

    for label, rules in config.rules.items():
        keywords = rules.get("keywords", [])
        if any(keyword in text for keyword in keywords):
            logger.info(
                "Rule-based classification",
                extra={"label": label},
            )
            effective_label, policy_reason = resolve_effective_workflow(
                label,
                app_id=app_id or None,
                actor_id=actor_id or None,
            )
            if policy_reason:
                logger.info(policy_reason, extra={"app_id": app_id, "actor_id": actor_id})
            return effective_label, _EMPTY_USAGE

    logger.info("Rule-based classification defaulted to 'general'")
    return normalize_workflow_name("general"), _EMPTY_USAGE
