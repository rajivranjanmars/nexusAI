"""
Test that workflow_lead_capture prompt correctly removes the model-driven exit instruction
and verifies the prompt contract for lead capture workflow transitions.
"""

import json
from shared.prompt_loader import load_prompt, render_prompt


def test_prompt_structure():
    """Verify prompt has exactly the required top-level keys."""
    prompt = load_prompt("workflow_lead_capture")

    expected_keys = {"id", "version", "description", "system", "user_template"}
    actual_keys = set(prompt.keys())

    assert actual_keys == expected_keys, f"Keys mismatch. Expected {expected_keys}, got {actual_keys}"


def test_removed_model_driven_exit_from_system_text():
    """Verify that harmful model-driven exit instructions are removed from the rendered system text."""
    prompt = load_prompt("workflow_lead_capture")
    system_text = prompt["system"]

    # Verify the typo "leap_capture" does not appear anywhere
    assert "leap_capture" not in system_text, "System text still contains typo 'leap_capture'"

    # Verify the model-driven exit instruction is not in the system text
    assert "change the workflow" not in system_text, "System text still contains 'change the workflow' instruction"
    assert "Exit the" not in system_text, "System text still contains 'Exit the' instruction"


def test_completion_trigger_survives():
    """Verify that the actual exit trigger (status='completed') remains in the prompt."""
    prompt = load_prompt("workflow_lead_capture")
    system_text = prompt["system"]

    # The real exit mechanism must survive: setting status to "completed"
    assert '"status": "completed"' in system_text, "System text does not contain the actual exit trigger 'status: completed'"


def test_render_with_app_kwargs():
    """
    Verify prompt renders successfully with the exact kwargs the app passes.
    This tests that no brace-escaping was broken and all required variables are present.
    """
    # Mock the kwargs the app actually passes (from llm/prompt_builder.py line 266-274)
    kwargs = {
        "user_input": "Hi, I'm interested in BCA",
        "student_data": "Name: John Doe, Phone: +919876543210",
        "context": "User is interested in pursuing a BCA degree",
        "actor_id": "session_12345",
        "lead_progress": json.dumps({"status": "in_progress", "fields_collected": ["name"]}, default=str),
        "verified_phone": "+919876543210",
    }

    # Should not raise PromptRenderError for missing variables
    system_rendered, user_rendered = render_prompt("workflow_lead_capture", **kwargs)

    # Verify rendered content makes sense
    assert system_rendered, "System prompt rendered empty"
    assert user_rendered, "User prompt rendered empty"

    # Verify the rendered text still has no harmful instructions
    assert "leap_capture" not in system_rendered, "Rendered system contains typo 'leap_capture'"
    assert "change the workflow" not in system_rendered, "Rendered system contains 'change the workflow'"
    assert "Exit the" not in system_rendered, "Rendered system contains 'Exit the'"

    # Verify the exit trigger is in the rendered output
    assert '"status": "completed"' in system_rendered, "Rendered system missing the actual exit trigger"
