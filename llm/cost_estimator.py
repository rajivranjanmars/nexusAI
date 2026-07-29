"""
LLM Cost Estimator — dynamic, env-var-driven pricing.

Reads model pricing from the ``LLM_COST_MAP`` environment variable as a
JSON string.  Falls back to sensible defaults for common OpenAI /
OpenRouter models.

Expected ``LLM_COST_MAP`` format (JSON):

    {
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1":      {"input": 2.00, "output": 8.00}
    }

Prices are expressed as USD per **1 million tokens**.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from shared.logger import get_logger

logger = get_logger(__name__)

# ── Default cost map (USD per 1M tokens) ────────────────────────────────────
_DEFAULT_COST_MAP: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # OpenRouter / Groq / other common models
    "meta-llama/llama-4-maverick": {"input": 0.20, "output": 0.60},
    "meta-llama/llama-4-scout": {"input": 0.15, "output": 0.40},
    "google/gemini-2.5-flash-preview": {"input": 0.15, "output": 0.60},
    "google/gemini-2.5-pro-preview": {"input": 1.25, "output": 10.00},
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
}

_cost_map: Optional[Dict[str, Dict[str, float]]] = None


def _load_cost_map() -> Dict[str, Dict[str, float]]:
    """Load and cache the cost map from env or defaults."""
    global _cost_map
    if _cost_map is not None:
        return _cost_map

    raw = os.environ.get("LLM_COST_MAP", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                _cost_map = {**_DEFAULT_COST_MAP, **parsed}
                logger.info(
                    "Loaded LLM_COST_MAP from environment (%d model entries, %d from env)",
                    len(_cost_map),
                    len(parsed),
                )
                return _cost_map
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Invalid LLM_COST_MAP env var, using defaults: %s", exc)

    _cost_map = _DEFAULT_COST_MAP.copy()
    return _cost_map


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Optional[float]:
    """Estimate cost in USD for a single LLM invocation.

    Args:
        model: Model identifier as returned by the LLM provider.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD, or ``None`` if the model is unknown.
    """
    cost_map = _load_cost_map()

    # Try exact match first, then prefix match for versioned model names
    pricing = cost_map.get(model)
    if pricing is None:
        for key in cost_map:
            if model.startswith(key) or key.startswith(model):
                pricing = cost_map[key]
                break

    if pricing is None:
        return None

    input_cost = (prompt_tokens / 1_000_000) * pricing.get("input", 0)
    output_cost = (completion_tokens / 1_000_000) * pricing.get("output", 0)
    return round(input_cost + output_cost, 6)


def get_cost_map() -> Dict[str, Dict[str, float]]:
    """Return the active cost map for admin display."""
    return _load_cost_map().copy()
