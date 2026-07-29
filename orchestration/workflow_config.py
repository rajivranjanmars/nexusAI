"""
Registry resolver for workflow configuration.

Loads platform-wide defaults from config/workflow_defaults.yaml and
merges app-specific overrides from the database app_config.
"""

from __future__ import annotations

import time
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, FrozenSet, Tuple

from shared.logger import get_logger
from db.app_registry import resolve_by_app_id
from orchestration.workflow_policy import normalize_workflow_name, normalize_workflow_names

logger = get_logger(__name__)

_DEFAULTS_PATH = Path(__file__).parent.parent / "config" / "workflow_defaults.yaml"
_CACHE_TTL_SECONDS = 300

@dataclass(frozen=True)
class ResponseStyleConfig:
    include_sources: bool
    max_rag_chars: int
    format_instructions: str

@dataclass(frozen=True)
class WorkflowResponseConfig:
    default_style: str
    include_sources: bool
    max_rag_chars: int
    history_limit: int
    system_suffix: Optional[str] = None

@dataclass(frozen=True)
class RetrievalConfig:
    """Configurable constants for the RAG retrieval pipeline."""

    dense_top_k: int
    sparse_top_k: int
    rrf_k: int
    rerank_input_size: int
    rerank_output_size: int
    min_final_score: float
    max_chunks_per_source: int
    scoring_weights: Dict[str, float]
    lexical_boosts: Dict[str, float]
    noisy_path_tokens: FrozenSet[str]
    path_allowlist: Tuple[str, ...]
    path_blocklist: Tuple[str, ...]
    path_allowlist_boost: float
    path_blocklist_penalty: float


CacheScope = Literal["shared", "actor"]


@dataclass(frozen=True)
class WorkflowCachePolicy:
    """Per-workflow cache behaviour."""

    policy: str          # "enabled" | "disabled"
    scope: CacheScope    # "shared" | "actor"
    similarity_threshold: Optional[float] = None  # per-workflow override
    ttl_seconds: Optional[int] = None             # per-workflow override


@dataclass(frozen=True)
class WorkflowConfig:
    labels: FrozenSet[str]
    confidence_threshold: float
    rules: Dict[str, Dict[str, List[str]]]  # label -> {"keywords": [...]}
    routes: Dict[str, str]                  # label -> next_node
    classify_system_prompt: str
    classify_user_template: str
    cache_default_policy: str
    cache_default_scope: CacheScope
    cache_default_similarity_threshold: float
    cache_workflow_policies: Dict[str, WorkflowCachePolicy]
    response_styles: Dict[str, ResponseStyleConfig]
    workflow_response_config: Dict[str, WorkflowResponseConfig]
    retrieval_config: RetrievalConfig

_defaults_cache: Optional[dict] = None
_config_cache: Dict[str, tuple[WorkflowConfig, float]] = {}

# ── Retrieval config defaults ───────────────────────────────────────────────
_DEFAULT_RETRIEVAL_CONFIG = {
    "dense_top_k": 30,
    "sparse_top_k": 30,
    "rrf_k": 60,
    "rerank_input_size": 25,
    "rerank_output_size": 6,
    "min_final_score": 0.35,
    "max_chunks_per_source": 2,
    "scoring_weights": {"dense": 0.45, "sparse": 0.35, "rrf": 0.20},
    "lexical_boosts": {
        "structured_data": 0.75,
        "strong_program_match": 0.20,
        "weak_program_match": 0.08,
        "program_miss": -0.15,
        "fee_match": 0.08,
        "fee_miss": -0.05,
        "noisy_path": -0.12,
        "programme_path": 0.04,
        "heading_match": 0.04,
    },
    "noisy_path_tokens": ["archive", "convocation", "pictures", "gallery"],
    "path_allowlist": [],
    "path_blocklist": [],
    "path_allowlist_boost": 0.10,
    "path_blocklist_penalty": -0.10,
}


def _parse_retrieval_config(raw: dict) -> RetrievalConfig:
    """Build a ``RetrievalConfig`` from a raw dict, filling in safe defaults."""

    d = _DEFAULT_RETRIEVAL_CONFIG
    return RetrievalConfig(
        dense_top_k=int(raw.get("dense_top_k", d["dense_top_k"])),
        sparse_top_k=int(raw.get("sparse_top_k", d["sparse_top_k"])),
        rrf_k=int(raw.get("rrf_k", d["rrf_k"])),
        rerank_input_size=int(raw.get("rerank_input_size", d["rerank_input_size"])),
        rerank_output_size=int(raw.get("rerank_output_size", d["rerank_output_size"])),
        min_final_score=float(raw.get("min_final_score", d["min_final_score"])),
        max_chunks_per_source=int(raw.get("max_chunks_per_source", d["max_chunks_per_source"])),
        scoring_weights={**d["scoring_weights"], **dict(raw.get("scoring_weights", {}))},
        lexical_boosts={**d["lexical_boosts"], **dict(raw.get("lexical_boosts", {}))},
        noisy_path_tokens=frozenset(
            str(t).lower() for t in raw.get("noisy_path_tokens", d["noisy_path_tokens"])
        ),
        path_allowlist=tuple(str(p) for p in raw.get("path_allowlist", d["path_allowlist"])),
        path_blocklist=tuple(str(p) for p in raw.get("path_blocklist", d["path_blocklist"])),
        path_allowlist_boost=float(raw.get("path_allowlist_boost", d["path_allowlist_boost"])),
        path_blocklist_penalty=float(raw.get("path_blocklist_penalty", d["path_blocklist_penalty"])),
    )


def _load_defaults() -> dict:
    """Load the platform defaults YAML lazily."""
    global _defaults_cache
    if _defaults_cache is None:
        try:
            with open(_DEFAULTS_PATH, "r", encoding="utf-8") as f:
                _defaults_cache = yaml.safe_load(f)
                logger.info("Loaded workflow defaults from %s", _DEFAULTS_PATH)
        except Exception as exc:
            logger.error("Failed to load workflow defaults: %s", exc)
            # Minimal fail-safe fallback
            _defaults_cache = {
                "labels": ["general"],
                "confidence_threshold": 0.5,
                "rules": {},
                "routes": {"general": "fetch_data"},
                "classify_prompt": {
                    "system": "Respond with JSON: {\"label\": \"general\", \"confidence\": 1.0}",
                    "user_template": "{user_input}"
                }
            }
    return _defaults_cache


def get_workflow_config(app_id: Optional[str] = None) -> WorkflowConfig:
    """
    Resolve the workflow configuration for an app.
    
    1. Loads platform defaults.
    2. Deep merges overrides from AppContext.app_config["workflow_config"].
    3. Caches the resulting WorkflowConfig.
    """
    cache_key = app_id or "default"
    cached = _config_cache.get(cache_key)
    if cached and cached[1] > time.time():
        return cached[0]

    defaults = _load_defaults()
    
    # Base defaults
    labels_set = set(normalize_workflow_names(defaults.get("labels", [])))
    conf_threshold = float(defaults.get("confidence_threshold", 0.6))
    rules = {
        normalize_workflow_name(label): label_rules
        for label, label_rules in dict(defaults.get("rules", {})).items()
    }
    routes = {
        normalize_workflow_name(label): route
        for label, route in dict(defaults.get("routes", {})).items()
    }
    classify_system = defaults.get("classify_prompt", {}).get("system", "")
    classify_user = defaults.get("classify_prompt", {}).get("user_template", "{user_input}")
    
    # Retrieval config defaults
    retrieval_raw = dict(defaults.get("retrieval_config", {}))
    retrieval_cfg = _parse_retrieval_config(retrieval_raw)

    # Cache defaults
    cache_cfg = defaults.get("cache_config", {})
    cache_default_policy = str(cache_cfg.get("default_policy", "enabled"))
    cache_default_scope: CacheScope = "actor" if cache_cfg.get("default_scope") == "actor" else "shared"
    cache_default_sim_threshold = float(cache_cfg.get("default_similarity_threshold", 0.92))
    cache_workflow_policies: Dict[str, WorkflowCachePolicy] = {}
    for wf_name, wf_pol in dict(cache_cfg.get("workflow_policies", {})).items():
        normalized = normalize_workflow_name(wf_name)
        cache_workflow_policies[normalized] = WorkflowCachePolicy(
            policy=str(wf_pol.get("policy", cache_default_policy)),
            scope=wf_pol.get("scope", cache_default_scope),
            similarity_threshold=float(wf_pol["similarity_threshold"]) if "similarity_threshold" in wf_pol else None,
            ttl_seconds=int(wf_pol["ttl_seconds"]) if "ttl_seconds" in wf_pol else None,
        )
    
    # Response style defaults
    response_styles = {}
    for style_name, style_config in defaults.get("response_styles", {}).items():
        response_styles[style_name] = ResponseStyleConfig(
            include_sources=style_config.get("include_sources", True),
            max_rag_chars=style_config.get("max_rag_chars", 4000),
            format_instructions=style_config.get("format_instructions", "")
        )
    
    # Workflow response config defaults
    workflow_response_config = {}
    for workflow_name, wf_config in defaults.get("workflow_response_config", {}).items():
        normalized_name = normalize_workflow_name(workflow_name)
        workflow_response_config[normalized_name] = WorkflowResponseConfig(
            default_style=wf_config.get("default_style", "balanced"),
            include_sources=wf_config.get("include_sources", True),
            max_rag_chars=wf_config.get("max_rag_chars", 4000),
            history_limit=wf_config.get("history_limit", 6),
            system_suffix=wf_config.get("system_suffix")
        )

    # App overrides
    if app_id:
        app_context = resolve_by_app_id(app_id)
        if app_context and app_context.app_config:
            wf_config = app_context.app_config.get("workflow_config", {})
            
            # Merge labels (union)
            if "labels" in wf_config:
                labels_set.update(normalize_workflow_names(wf_config["labels"]))
                
            # Merge threshold
            if "confidence_threshold" in wf_config:
                conf_threshold = float(wf_config["confidence_threshold"])
                
            # Merge rules
            if "rules" in wf_config:
                for label, label_rules in wf_config["rules"].items():
                    rules[normalize_workflow_name(label)] = label_rules  # overwrite entirely for that label
                    
            # Merge routes
            if "routes" in wf_config:
                for label, route in wf_config["routes"].items():
                    routes[normalize_workflow_name(label)] = route
                    
            # Merge prompts
            if "classify_prompt" in wf_config:
                if "system" in wf_config["classify_prompt"]:
                    classify_system = wf_config["classify_prompt"]["system"]
                if "user_template" in wf_config["classify_prompt"]:
                    classify_user = wf_config["classify_prompt"]["user_template"]
                    
            # Merge cache config
            if "cache_config" in wf_config:
                ccfg = wf_config["cache_config"]
                if "default_similarity_threshold" in ccfg:
                    cache_default_sim_threshold = float(ccfg["default_similarity_threshold"])
                if "default_policy" in ccfg:
                    cache_default_policy = str(ccfg["default_policy"])
                if "default_scope" in ccfg:
                    cache_default_scope = ccfg["default_scope"]
                for wf_name, wf_pol in dict(ccfg.get("workflow_policies", {})).items():
                    normalized = normalize_workflow_name(wf_name)
                    existing = cache_workflow_policies.get(normalized)
                    cache_workflow_policies[normalized] = WorkflowCachePolicy(
                        policy=str(wf_pol.get("policy", existing.policy if existing else cache_default_policy)),
                        scope=wf_pol.get("scope", existing.scope if existing else cache_default_scope),
                        similarity_threshold=float(wf_pol["similarity_threshold"]) if "similarity_threshold" in wf_pol else (existing.similarity_threshold if existing else None),
                        ttl_seconds=int(wf_pol["ttl_seconds"]) if "ttl_seconds" in wf_pol else (existing.ttl_seconds if existing else None),
                    )
            
            # Merge response styles
            if "response_styles" in wf_config:
                for style_name, style_config in wf_config["response_styles"].items():
                    response_styles[style_name] = ResponseStyleConfig(
                        include_sources=style_config.get("include_sources", True),
                        max_rag_chars=style_config.get("max_rag_chars", 4000),
                        format_instructions=style_config.get("format_instructions", "")
                    )
            
            # Merge workflow response config
            if "workflow_response_config" in wf_config:
                for workflow_name, wf_resp_config in wf_config["workflow_response_config"].items():
                    normalized_name = normalize_workflow_name(workflow_name)
                    workflow_response_config[normalized_name] = WorkflowResponseConfig(
                        default_style=wf_resp_config.get("default_style", "balanced"),
                        include_sources=wf_resp_config.get("include_sources", True),
                        max_rag_chars=wf_resp_config.get("max_rag_chars", 4000),
                        history_limit=wf_resp_config.get("history_limit", 6),
                        system_suffix=wf_resp_config.get("system_suffix")
                    )

            # Merge retrieval config (app overrides shallow-merge into defaults)
            if "retrieval_config" in wf_config:
                merged_raw = {**retrieval_raw, **wf_config["retrieval_config"]}
                # Deep-merge nested dicts (scoring_weights, lexical_boosts)
                for nested_key in ("scoring_weights", "lexical_boosts"):
                    if nested_key in wf_config["retrieval_config"]:
                        merged_raw[nested_key] = {
                            **retrieval_raw.get(nested_key, {}),
                            **wf_config["retrieval_config"][nested_key],
                        }
                retrieval_cfg = _parse_retrieval_config(merged_raw)

    # Render template variables
    labels_pipe_separated = " | ".join(sorted(labels_set))
    classify_system = classify_system.replace("{labels_pipe_separated}", labels_pipe_separated)

    resolved = WorkflowConfig(
        labels=frozenset(labels_set),
        confidence_threshold=conf_threshold,
        rules=rules,
        routes=routes,
        classify_system_prompt=classify_system,
        classify_user_template=classify_user,
        cache_default_policy=cache_default_policy,
        cache_default_scope=cache_default_scope,
        cache_default_similarity_threshold=cache_default_sim_threshold,
        cache_workflow_policies=cache_workflow_policies,
        response_styles=response_styles,
        workflow_response_config=workflow_response_config,
        retrieval_config=retrieval_cfg,
    )

    # Cache it
    _config_cache[cache_key] = (resolved, time.time() + _CACHE_TTL_SECONDS)
    return resolved

def invalidate_workflow_config_cache() -> None:
    """Clear the runtime cache."""
    _config_cache.clear()


def resolve_cache_policy(
    config: WorkflowConfig,
    workflow: str,
) -> WorkflowCachePolicy:
    """Return the effective cache policy for *workflow*.

    Falls back to the config-level defaults when the workflow is not
    explicitly listed in ``cache_workflow_policies``.
    """
    normalized = normalize_workflow_name(workflow) or "general"
    if normalized in config.cache_workflow_policies:
        return config.cache_workflow_policies[normalized]
    return WorkflowCachePolicy(
        policy=config.cache_default_policy,
        scope=config.cache_default_scope,
    )
