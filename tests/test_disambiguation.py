"""
Tests for shared.disambiguation module.

Covers URL extraction, partitioning, gain calculation, ranking, and axis selection.
"""

import pytest

from shared.disambiguation import (
    Axis,
    axis_gain,
    extract_axis_value,
    partition_by_axis,
    rank_axes,
    top_axis,
)


# ============================================================================
# extract_axis_value tests
# ============================================================================


def test_extract_axis_value_basic() -> None:
    """Extract value from basic URL with path prefix."""
    url = "https://x.com/programmes/mba/fee-structure/"
    result = extract_axis_value(url, "/programmes/")
    assert result == "mba"


def test_extract_axis_value_mba_capitalized() -> None:
    """Lowercase the extracted value."""
    url = "https://x.com/programmes/MBA"
    result = extract_axis_value(url, "/programmes/")
    assert result == "mba"


def test_extract_axis_value_missing_prefix() -> None:
    """Return None when prefix is absent."""
    url = "https://x.com/admissions/apply/"
    result = extract_axis_value(url, "/programmes/")
    assert result is None


def test_extract_axis_value_nothing_after_prefix() -> None:
    """Return None when nothing follows the prefix."""
    url = "https://x.com/programmes/"
    result = extract_axis_value(url, "/programmes/")
    assert result is None


def test_extract_axis_value_with_query_string() -> None:
    """Ignore query string when extracting."""
    url = "https://x.com/programmes/mba?param=value"
    result = extract_axis_value(url, "/programmes/")
    assert result == "mba"


def test_extract_axis_value_case_insensitive_prefix() -> None:
    """Path prefix matching is case-insensitive."""
    url = "https://x.com/PROGRAMMES/mba/"
    result = extract_axis_value(url, "/programmes/")
    assert result == "mba"


def test_extract_axis_value_with_fragment() -> None:
    """Ignore fragment when extracting."""
    url = "https://x.com/programmes/mba#section"
    result = extract_axis_value(url, "/programmes/")
    assert result == "mba"


# ============================================================================
# Partitioning tests
# ============================================================================


def test_partition_by_axis_basic() -> None:
    """Partition chunks by axis value."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "MBA content"},
        {"source_url": "https://x.com/programmes/mba/fees", "score": 0.8, "content": "MBA fees"},
        {"source_url": "https://x.com/programmes/bca/overview", "score": 0.85, "content": "BCA content"},
    ]
    axis = Axis(name="program", path_prefix="/programmes/")
    result = partition_by_axis(chunks, axis)

    assert len(result) == 2
    assert len(result["mba"]) == 2
    assert len(result["bca"]) == 1


def test_partition_by_axis_excludes_none() -> None:
    """Chunks with None axis value are excluded."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "MBA content"},
        {"source_url": "https://x.com/admissions/apply", "score": 0.7, "content": "Admissions content"},
    ]
    axis = Axis(name="program", path_prefix="/programmes/")
    result = partition_by_axis(chunks, axis)

    assert len(result) == 1
    assert "mba" in result


# ============================================================================
# axis_gain tests
# ============================================================================


def test_axis_gain_different_content() -> None:
    """Gain > 0 when programs have different content."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "MBA overview and structure"},
        {"source_url": "https://x.com/programmes/mba/fees", "score": 0.8, "content": "MBA fees structure"},
        {"source_url": "https://x.com/programmes/bca/overview", "score": 0.85, "content": "BCA completely different content"},
        {"source_url": "https://x.com/programmes/bca/fees", "score": 0.75, "content": "BCA different fees"},
    ]
    axis = Axis(name="program", path_prefix="/programmes/")
    gain = axis_gain(chunks, axis)

    assert 0.0 < gain <= 1.0


def test_axis_gain_identical_content() -> None:
    """Gain == 0 when all partitions have identical normalized content."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "Same   content"},
        {"source_url": "https://x.com/programmes/bca/overview", "score": 0.85, "content": "same content"},  # different case
    ]
    axis = Axis(name="program", path_prefix="/programmes/")
    gain = axis_gain(chunks, axis)

    assert gain == 0.0


def test_axis_gain_single_partition() -> None:
    """Gain == 0 when only one partition exists."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "MBA content 1"},
        {"source_url": "https://x.com/programmes/mba/fees", "score": 0.8, "content": "MBA content 2"},
    ]
    axis = Axis(name="program", path_prefix="/programmes/")
    gain = axis_gain(chunks, axis)

    assert gain == 0.0


def test_axis_gain_no_prefix_match() -> None:
    """Gain == 0 when no chunks match the prefix."""
    chunks = [
        {"source_url": "https://x.com/admissions/apply", "score": 0.9, "content": "Admissions content 1"},
        {"source_url": "https://x.com/admissions/deadline", "score": 0.8, "content": "Admissions content 2"},
    ]
    axis = Axis(name="program", path_prefix="/programmes/")
    gain = axis_gain(chunks, axis)

    assert gain == 0.0


def test_axis_gain_zero_scores() -> None:
    """Gain == 0 when all scores are zero (no ZeroDivisionError)."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.0, "content": "MBA content 1"},
        {"source_url": "https://x.com/programmes/bca/overview", "score": 0.0, "content": "BCA content 2"},
    ]
    axis = Axis(name="program", path_prefix="/programmes/")
    gain = axis_gain(chunks, axis)

    assert gain == 0.0


# ============================================================================
# rank_axes tests
# ============================================================================


def test_rank_axes_sorted_descending() -> None:
    """Axes are sorted by gain descending."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "MBA content 1"},
        {"source_url": "https://x.com/programmes/bca/overview", "score": 0.85, "content": "BCA different content"},
        {"source_url": "https://x.com/duration/1yr/overview", "score": 0.7, "content": "1 year content"},
        {"source_url": "https://x.com/duration/2yr/overview", "score": 0.6, "content": "2 year content"},
    ]
    program_axis = Axis(name="program", path_prefix="/programmes/")
    duration_axis = Axis(name="duration", path_prefix="/duration/")

    ranked = rank_axes(chunks, [program_axis, duration_axis])

    # Check that results are sorted by gain descending
    gains = [gain for _, gain in ranked]
    assert gains == sorted(gains, reverse=True)


def test_rank_axes_tie_break_by_order() -> None:
    """Axes with equal gain break ties by declaration order."""
    # Create chunks where both axes have identical gain structure
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "MBA content 1"},
        {"source_url": "https://x.com/programmes/bca/overview", "score": 0.85, "content": "BCA different content"},
        {"source_url": "https://x.com/format/online/overview", "score": 0.8, "content": "Online content 1"},
        {"source_url": "https://x.com/format/offline/overview", "score": 0.75, "content": "Offline different content"},
    ]
    program_axis = Axis(name="program", path_prefix="/programmes/")
    format_axis = Axis(name="format", path_prefix="/format/")

    # program first
    ranked1 = rank_axes(chunks, [program_axis, format_axis])
    names1 = [name for name, _ in ranked1]

    # format first
    ranked2 = rank_axes(chunks, [format_axis, program_axis])
    names2 = [name for name, _ in ranked2]

    # When axes have equal gain, declaration order should be preserved
    # The one declared first should appear first in the result
    if ranked1[0][1] == ranked1[1][1]:  # If they have equal gain
        assert names1[0] == "program"
        assert names2[0] == "format"


# ============================================================================
# top_axis tests
# ============================================================================


def test_top_axis_returns_best_axis_and_options() -> None:
    """top_axis returns (axis_name, options) for highest-ranked axis."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "MBA content 1"},
        {"source_url": "https://x.com/programmes/mba/fees", "score": 0.8, "content": "MBA content 2"},
        {"source_url": "https://x.com/programmes/bca/overview", "score": 0.85, "content": "BCA different content"},
    ]
    axis = Axis(name="program", path_prefix="/programmes/")

    result = top_axis(chunks, [axis])

    assert result is not None
    assert result[0] == "program"
    assert len(result[1]) == 2
    assert set(result[1]) == {"mba", "bca"}


def test_top_axis_options_sorted_by_mass() -> None:
    """Options are sorted by descending score mass, ties broken by first appearance."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "MBA content 1"},
        {"source_url": "https://x.com/programmes/mba/fees", "score": 0.8, "content": "MBA content 2"},  # mba mass: 1.7
        {"source_url": "https://x.com/programmes/bca/overview", "score": 0.5, "content": "BCA different content"},  # bca mass: 0.5
    ]
    axis = Axis(name="program", path_prefix="/programmes/")

    result = top_axis(chunks, [axis])

    assert result is not None
    assert result[1] == ["mba", "bca"]  # mba has higher mass


def test_top_axis_below_min_gain() -> None:
    """Returns None if top axis's gain <= min_gain."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "Same content"},
        {"source_url": "https://x.com/programmes/bca/overview", "score": 0.85, "content": "same content"},
    ]
    axis = Axis(name="program", path_prefix="/programmes/")

    result = top_axis(chunks, [axis], min_gain=0.5)

    assert result is None


def test_top_axis_no_matching_prefix() -> None:
    """Returns None if no chunks match any axis."""
    chunks = [
        {"source_url": "https://x.com/admissions/apply", "score": 0.9, "content": "Admissions content"},
    ]
    axis = Axis(name="program", path_prefix="/programmes/")

    result = top_axis(chunks, [axis])

    assert result is None


def test_top_axis_identical_content_different_axes() -> None:
    """Returns None if identical content (gain == 0) even with different axes."""
    chunks = [
        {"source_url": "https://x.com/programmes/mba/overview", "score": 0.9, "content": "Same"},
        {"source_url": "https://x.com/programmes/bca/overview", "score": 0.85, "content": "same"},
    ]
    axis1 = Axis(name="program", path_prefix="/programmes/")
    axis2 = Axis(name="location", path_prefix="/location/")

    result = top_axis(chunks, [axis1, axis2])

    assert result is None
