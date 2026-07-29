"""
MCP Prompt templates — reusable prompt definitions.

These prompts are exposed as MCP Prompt resources so that clients can
discover and invoke them without hard-coding prompt text.
"""

from __future__ import annotations

from typing import Dict, List

from shared.logger import get_logger
from shared.prompt_loader import load_prompt, PromptNotFoundError
from orchestration.workflow_config import get_workflow_config

logger = get_logger(__name__)

# ── Prompt definitions ─────────────────────────────────────────────────────

# Note: The MCP layer exposes Prompts. This function dynamically loads the
# prompts that map to active workflows defined in the dynamic WorkflowConfig.


def _build_dict_for_mcp(workflow_name: str) -> Dict[str, str]:
    prompt_id = f"workflow_{workflow_name}"
    try:
        data = load_prompt(prompt_id)
        template = f"{data.get('system', '').strip()}\n\n{data.get('user_template', '').strip()}"
        return {
            "name": workflow_name,
            "description": data.get("description", ""),
            "template": template
        }
    except PromptNotFoundError:
        return {}


def get_workflow_prompt(workflow_name: str) -> Dict[str, str]:
    """Retrieve a prompt template by workflow name.

    Args:
        workflow_name: One of the registered workflow keys.

    Returns:
        A dict with ``name``, ``description``, and ``template`` keys.
    """
    prompt = _build_dict_for_mcp(workflow_name)
    if not prompt:
        logger.warning("Unknown workflow prompt '%s', falling back to general", workflow_name)
        return _build_dict_for_mcp("general")
    return prompt


def list_workflow_prompts() -> List[Dict[str, str]]:
    """Return all available workflow prompt definitions."""
    config = get_workflow_config(None) # Use global defaults
    return [_build_dict_for_mcp(w) for w in config.labels if _build_dict_for_mcp(w)]
