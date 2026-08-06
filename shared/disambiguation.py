"""
Disambiguation module: ranks which facet of the corpus to ask the user about.

Given a set of retrieved chunks, identifies the axis (facet) that would most
usefully narrow down the results. Gain of 0.0 means "do not ask about this axis".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Axis:
    """A facet of the corpus you could ask the user to narrow down."""

    name: str  # e.g. "program"
    path_prefix: str  # e.g. "/programmes/"


def extract_axis_value(source_url: str, path_prefix: str) -> str | None:
    """
    Extract the axis value from a source URL.

    The axis value is the single URL path segment that immediately follows
    path_prefix. Lowercases the result. Returns None if path_prefix is absent,
    or if nothing non-empty follows it.

    Args:
        source_url: The full URL to parse.
        path_prefix: The path segment to find (case-insensitive).

    Returns:
        The lowercased segment following path_prefix, or None.

    Examples:
        "https://x.com/programmes/mba/fee-structure/" + "/programmes/" -> "mba"
        "https://x.com/programmes/MBA" + "/programmes/" -> "mba"
        "https://x.com/admissions/apply/" + "/programmes/" -> None
        "https://x.com/programmes/" + "/programmes/" -> None
    """
    # Remove fragment and query string
    url_path = source_url.split("?")[0].split("#")[0]

    # Find path_prefix case-insensitively
    lower_url = url_path.lower()
    lower_prefix = path_prefix.lower()

    prefix_idx = lower_url.find(lower_prefix)
    if prefix_idx == -1:
        return None

    # Start after the prefix
    start_idx = prefix_idx + len(path_prefix)
    remainder = url_path[start_idx:]

    # Extract the first path segment
    # Split by "/" and take the first non-empty segment
    segments = [s for s in remainder.split("/") if s]
    if not segments:
        return None

    return segments[0].lower()


def partition_by_axis(
    chunks: Sequence[Mapping[str, Any]], axis: Axis
) -> dict[str, list[Mapping[str, Any]]]:
    """
    Group chunks by their axis value.

    Chunks whose axis value is None are EXCLUDED entirely.

    Args:
        chunks: Sequence of chunk dicts with "source_url" key.
        axis: The axis to partition by.

    Returns:
        Dict mapping axis value (str) to list of chunks in that partition.
    """
    result: dict[str, list[Mapping[str, Any]]] = {}

    for chunk in chunks:
        source_url = chunk.get("source_url", "")
        value = extract_axis_value(source_url, axis.path_prefix)
        if value is None:
            continue
        if value not in result:
            result[value] = []
        result[value].append(chunk)

    return result


def axis_gain(chunks: Sequence[Mapping[str, Any]], axis: Axis) -> float:
    """
    Compute the value of asking the user about this axis.

    Returns a float in [0.0, 1.0]. Gain 0.0 means "do not ask".

    Two steps:
    1. SPLIT — normalized score-mass entropy over the partitions.
    2. DIFFER-GATE — if answer would not change, split is worthless.

    Args:
        chunks: Sequence of chunk dicts with "source_url" and "score" keys.
        axis: The axis to evaluate.

    Returns:
        Float in [0.0, 1.0] indicating utility of asking about this axis.
    """
    partitions = partition_by_axis(chunks, axis)

    # Step 1: SPLIT — compute normalized entropy
    if len(partitions) < 2:
        return 0.0

    # Compute mass for each partition
    masses: dict[str, float] = {}
    for value, partition in partitions.items():
        mass = sum(max(0.0, chunk.get("score", 0.0)) for chunk in partition)
        masses[value] = mass

    total = sum(masses.values())
    if total <= 0.0:
        return 0.0

    # Compute entropy
    entropy = 0.0
    num_partitions = len(partitions)
    for mass in masses.values():
        if mass > 0.0:
            p = mass / total
            entropy -= p * math.log2(p)

    # Normalize by max entropy (uniform distribution)
    split = entropy / math.log2(num_partitions)

    # Step 2: DIFFER-GATE — compare top chunks' normalized content
    top_chunks: dict[str, Mapping[str, Any]] = {}
    for value, partition in partitions.items():
        # Find max by score (stable: first occurrence wins ties)
        top_chunk = max(partition, key=lambda c: c.get("score", 0.0))
        top_chunks[value] = top_chunk

    # Normalize content: whitespace collapse + lowercase
    def normalize_content(content: str) -> str:
        return " ".join(content.split()).casefold()

    normalized_contents = {
        value: normalize_content(chunk.get("content", ""))
        for value, chunk in top_chunks.items()
    }

    # Check if all are identical
    content_values = list(normalized_contents.values())
    if content_values and all(c == content_values[0] for c in content_values):
        return 0.0

    return split


def rank_axes(
    chunks: Sequence[Mapping[str, Any]], axes: Sequence[Axis]
) -> list[tuple[str, float]]:
    """
    Rank axes by their gain, sorted descending.

    Ties broken by declaration order (index in `axes` argument).

    Args:
        chunks: Sequence of chunk dicts.
        axes: Sequence of Axis objects to evaluate.

    Returns:
        List of (axis.name, gain) tuples sorted by gain descending,
        ties broken by declaration order.
    """
    results: list[tuple[str, float, int]] = []
    for idx, axis in enumerate(axes):
        gain = axis_gain(chunks, axis)
        results.append((axis.name, gain, idx))

    # Sort by gain descending, then by index ascending (stable sort preserves ties)
    results.sort(key=lambda x: (-x[1], x[2]))

    return [(name, gain) for name, gain, _ in results]


def top_axis(
    chunks: Sequence[Mapping[str, Any]],
    axes: Sequence[Axis],
    min_gain: float = 0.0,
) -> tuple[str, list[str]] | None:
    """
    Get the highest-ranked axis and its options.

    If the top axis's gain <= min_gain, returns None.
    Otherwise returns (axis_name, options) where options are sorted by
    descending score mass, ties broken by first appearance in input order.

    Args:
        chunks: Sequence of chunk dicts.
        axes: Sequence of Axis objects.
        min_gain: Minimum gain threshold. If top axis <= this, return None.

    Returns:
        Tuple of (axis_name, sorted_options) or None if no axis meets threshold.
    """
    ranked = rank_axes(chunks, axes)
    if not ranked or ranked[0][1] <= min_gain:
        return None

    axis_name, _gain = ranked[0]

    # Find the axis object by name
    axis_obj = next((a for a in axes if a.name == axis_name), None)
    if axis_obj is None:
        return None

    # Compute mass for each partition, preserving input order
    partition_keys_in_order: list[str] = []
    partition_masses: dict[str, float] = {}

    for chunk in chunks:
        source_url = chunk.get("source_url", "")
        value = extract_axis_value(source_url, axis_obj.path_prefix)
        if value is None:
            continue
        if value not in partition_masses:
            partition_keys_in_order.append(value)
            partition_masses[value] = 0.0
        partition_masses[value] += max(0.0, chunk.get("score", 0.0))

    # Sort by mass descending, ties broken by first appearance
    options = sorted(
        partition_keys_in_order,
        key=lambda v: -partition_masses[v],
    )

    return (axis_name, options)
