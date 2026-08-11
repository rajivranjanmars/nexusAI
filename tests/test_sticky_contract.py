"""Tests for workflow stickiness configuration contract.

Tests that:
1. sticky_workflow() correctly identifies the sticky workflow from config
2. WorkflowResponseConfig defaults have correct stickiness values
3. lead_capture is configured as sticky in YAML defaults
4. Other workflows are non-sticky
"""

import pytest
from orchestration.workflow_config import (
    get_workflow_config,
    sticky_workflow,
    WorkflowResponseConfig,
)


def test_sticky_workflow_returns_lead_capture_for_defaults():
    """sticky_workflow(get_workflow_config("")) returns "lead_capture"."""
    config = get_workflow_config("")
    sticky = sticky_workflow(config)
    assert sticky == "lead_capture"


def test_workflow_response_config_defaults_non_sticky():
    """WorkflowResponseConfig defaults: sticky is False, abandon_after_seconds == 600."""
    cfg = WorkflowResponseConfig(
        default_style="balanced",
        include_sources=True,
        max_rag_chars=4000,
        history_limit=6,
    )
    assert cfg.sticky is False
    assert cfg.abandon_after_seconds == 600


def test_lead_capture_entry_is_sticky():
    """The lead_capture entry parsed from yaml has sticky is True."""
    config = get_workflow_config("")
    lead_cfg = config.workflow_response_config.get("lead_capture")
    assert lead_cfg is not None
    assert lead_cfg.sticky is True
    assert lead_cfg.abandon_after_seconds == 600


def test_all_other_workflows_are_non_sticky():
    """Every other workflow in workflow_response_config has sticky is False."""
    config = get_workflow_config("")
    for name, cfg in config.workflow_response_config.items():
        if name != "lead_capture":
            assert cfg.sticky is False, f"Workflow {name} should not be sticky"


def test_sticky_workflow_returns_none_when_all_non_sticky():
    """sticky_workflow returns None for a hand-built config with no sticky workflows."""
    from orchestration.workflow_config import WorkflowConfig, RetrievalConfig, DisambiguationConfig

    # Build a minimal config with all non-sticky workflows
    cfg = WorkflowConfig(
        labels=frozenset(["general"]),
        confidence_threshold=0.6,
        rules={},
        routes={},
        classify_system_prompt="",
        classify_user_template="",
        cache_default_policy="enabled",
        cache_default_scope="shared",
        cache_default_similarity_threshold=0.92,
        cache_workflow_policies={},
        response_styles={},
        workflow_response_config={
            "general": WorkflowResponseConfig(
                default_style="balanced",
                include_sources=True,
                max_rag_chars=4000,
                history_limit=6,
                sticky=False,
                abandon_after_seconds=600,
            ),
        },
        retrieval_config=RetrievalConfig(
            dense_top_k=30,
            sparse_top_k=30,
            rrf_k=60,
            rerank_input_size=25,
            rerank_output_size=6,
            min_final_score=0.35,
            max_chunks_per_source=2,
            scoring_weights={},
            lexical_boosts={},
            noisy_path_tokens=frozenset(),
            path_allowlist=(),
            path_blocklist=(),
            path_allowlist_boost=0.0,
            path_blocklist_penalty=0.0,
        ),
        disambiguation=DisambiguationConfig(
            min_gain=0.35,
            axes=(),
        ),
    )
    assert sticky_workflow(cfg) is None
