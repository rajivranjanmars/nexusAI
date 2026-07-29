"""
Test Suite for MCP Server Workflows — Thorough Testing.

This script tests the MCP tools, workflow detection, context operations,
and LangGraph workflow execution. Run with: python test_mcp_workflows.py

Assumes Docker services are running (docker-compose up).
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Dict

# This file is an operator-run live-service diagnostic, not an isolated pytest
# module. Its functions intentionally return booleans and require seeded DB,
# Redis, and LLM services, so pytest must not collect them as unit tests.
__test__ = False

# MCP Tools
from mcp_server.tools import student_tools, context_tools, workflow_tools

# Orchestration
from orchestration.router import route_after_detection
from orchestration.state import WorkflowState
from orchestration.graphs.student_query_graph import build_student_query_graph

# Shared
from shared.logger import get_logger

logger = get_logger(__name__)


async def test_student_tools() -> bool:
    """Test get_student and update_student tools."""
    print("Testing student tools...")

    # Test get_student
    try:
        student = await student_tools.get_student("STU123")
        assert student["student_id"] == "STU123"
        assert student["first_name"] == "Jane"
        print("✓ get_student works")
    except Exception as e:
        print(f"✗ get_student failed: {e}")
        return False

    # Test update_student
    try:
        updates = {"gpa": 4.0}
        result = await student_tools.update_student("STU123", updates)
        assert result["gpa"] == 4.0
        print("✓ update_student works")
    except Exception as e:
        print(f"✗ update_student failed: {e}")
        return False

    return True


async def test_workflow_detection() -> bool:
    """Test detect_workflow tool."""
    print("Testing workflow detection...")

    test_queries = [
        ("What is my GPA?", "grade_inquiry"),
        ("How do I enroll in Biology?", "enrollment"),
        ("What are the drop deadlines?", "policy_lookup"),
        ("I need a math tutor", "resource_request"),
        ("Change my major to CS", "data_mutation"),
        ("Hello", "general"),
    ]

    for query, expected in test_queries:
        try:
            detected = await workflow_tools.detect_workflow(query, "STU123")
            if detected == expected:
                print(f"✓ '{query}' -> {detected}")
            else:
                print(f"✗ '{query}' -> {detected} (expected {expected})")
                return False
        except Exception as e:
            print(f"✗ detect_workflow failed for '{query}': {e}")
            return False

    return True


async def test_context_operations() -> bool:
    """Test get_context and build_context tools."""
    print("Testing context operations...")

    # Test build_context
    try:
        data = {"recent_query": "What is my GPA?", "session_id": "test"}
        success = await context_tools.build_context("STU123", data)
        assert success is True
        print("✓ build_context works")
    except Exception as e:
        print(f"✗ build_context failed: {e}")
        return False

    # Test get_context
    try:
        context = await context_tools.get_context("STU123", "GPA information")
        assert "session" in context
        assert "semantic_results" in context
        assert isinstance(context["semantic_results"], list)
        print("✓ get_context works")
    except Exception as e:
        print(f"✗ get_context failed: {e}")
        return False

    return True


async def test_workflow_routing() -> bool:
    """Test the workflow router."""
    print("Testing workflow routing...")

    test_cases = [
        ("grade_inquiry", "fetch_data"),
        ("enrollment", "fetch_data"),
        ("general", "build_context"),
    ]

    for workflow, expected_node in test_cases:
        try:
            state = WorkflowState(detected_workflow=workflow)
            next_node = route_after_detection(state)
            if next_node == expected_node:
                print(f"✓ {workflow} -> {next_node}")
            else:
                print(f"✗ {workflow} -> {next_node} (expected {expected_node})")
                return False
        except Exception as e:
            print(f"✗ routing failed for {workflow}: {e}")
            return False

    return True


async def test_workflow_execution() -> bool:
    """Test LangGraph workflow execution."""
    print("Testing workflow execution...")

    # Build the graph
    student_query_graph = build_student_query_graph()

    # Test student query graph with grade_inquiry
    try:
        initial_state = WorkflowState(
            detected_workflow="grade_inquiry",
            user_input="What is my GPA?",
            student_id="STU123",
        )

        # Run the graph
        result = await student_query_graph.ainvoke(initial_state)

        # Check that it completed
        assert "final_response" in result
        assert isinstance(result["final_response"], str)
        print("✓ Workflow execution works")
    except Exception as e:
        print(f"✗ Workflow execution failed: {e}")
        return False

    return True


async def run_tests() -> None:
    """Run all tests."""
    print("Starting MCP Workflows Test Suite")
    print("=" * 50)

    tests = [
        ("Student Tools", test_student_tools),
        ("Workflow Detection", test_workflow_detection),
        ("Context Operations", test_context_operations),
        ("Workflow Routing", test_workflow_routing),
        ("Workflow Execution", test_workflow_execution),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n{name}:")
        if await test_func():
            passed += 1
        print()

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_tests())
