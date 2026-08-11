"""Tests for build_slot_buttons function."""

import pytest

from shared.helper_buttons import build_lead_form_buttons, build_slot_buttons, GO_BACK_BUTTON


def test_basic_slot_buttons():
    """Three programs -> 3 option buttons + GO_BACK."""
    buttons = build_slot_buttons("program", ["mba", "bba", "bca"])
    assert len(buttons) == 4  # 3 options + GO_BACK

    assert buttons[0]["id"] == "slot_program_mba"
    assert buttons[0]["text"] == "MBA"
    assert buttons[0]["value"] == "mba"

    assert buttons[1]["id"] == "slot_program_bba"
    assert buttons[1]["text"] == "BBA"
    assert buttons[1]["value"] == "bba"

    assert buttons[2]["id"] == "slot_program_bca"
    assert buttons[2]["text"] == "BCA"
    assert buttons[2]["value"] == "bca"

    assert buttons[3] == GO_BACK_BUTTON


def test_truncate_to_max_options():
    """More than max_options -> truncated to max_options (+ GO_BACK)."""
    buttons = build_slot_buttons("program", ["mba", "bba", "bca", "mca", "dca"], max_options=3)
    assert len(buttons) == 4  # 3 options + GO_BACK
    assert buttons[0]["id"] == "slot_program_mba"
    assert buttons[1]["id"] == "slot_program_bba"
    assert buttons[2]["id"] == "slot_program_bca"
    # mca and dca should be dropped
    assert buttons[3] == GO_BACK_BUTTON


def test_blank_axis():
    """Blank axis -> []."""
    assert build_slot_buttons("", ["mba", "bba"]) == []
    assert build_slot_buttons("   ", ["mba", "bba"]) == []


def test_empty_options():
    """Empty options -> []."""
    assert build_slot_buttons("program", []) == []


def test_skip_blank_options():
    """Options containing blank strings -> skipped."""
    buttons = build_slot_buttons("program", ["mba", "", "bba", "  ", "bca"])
    assert len(buttons) == 4  # 3 options + GO_BACK
    assert buttons[0]["id"] == "slot_program_mba"
    assert buttons[1]["id"] == "slot_program_bba"
    assert buttons[2]["id"] == "slot_program_bca"
    assert buttons[3] == GO_BACK_BUTTON


def test_case_insensitive_dedup():
    """Case-insensitive duplicates -> collapsed, first occurrence kept."""
    buttons = build_slot_buttons("program", ["MBA", "mba", "BBA"])
    assert len(buttons) == 3  # 2 options + GO_BACK
    assert buttons[0]["id"] == "slot_program_mba"
    assert buttons[0]["text"] == "MBA"  # Preserves original case in text
    assert buttons[0]["value"] == "MBA"  # Preserves original in value
    assert buttons[1]["id"] == "slot_program_bba"
    assert buttons[1]["text"] == "BBA"
    assert buttons[1]["value"] == "BBA"
    assert buttons[2] == GO_BACK_BUTTON


def test_punctuation_in_option():
    """Option with punctuation -> id slug handles it correctly."""
    buttons = build_slot_buttons("program", ["b.com"])
    assert len(buttons) == 2  # 1 option + GO_BACK
    assert buttons[0]["id"] == "slot_program_b_com"
    assert buttons[0]["text"] == "B.COM"
    assert buttons[0]["value"] == "b.com"
    assert buttons[1] == GO_BACK_BUTTON


def test_regression_lead_form_buttons_program_level():
    """Regression: build_lead_form_buttons still works for program_level."""
    lead_progress = {"current_step": "program_level"}
    buttons = build_lead_form_buttons(lead_progress)
    assert len(buttons) == 4  # 3 program level buttons + GO_BACK
    assert buttons[0]["id"] == "program_level_bachelors"
    assert buttons[1]["id"] == "program_level_masters"
    assert buttons[2]["id"] == "program_level_diploma"
    assert buttons[3] == GO_BACK_BUTTON
