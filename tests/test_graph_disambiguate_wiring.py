"""Test router functions and graph wiring for the disambiguate node."""

import pytest
from orchestration.state import WorkflowState
from orchestration.graphs.prestream_graph import _route_after_disambiguate as prestream_router
from orchestration.graphs.student_query_graph import _route_after_disambiguate as student_router
from orchestration.graphs.student_query_graph import build_student_query_graph
from orchestration.graphs.enrollment_graph import _route_after_disambiguate as enrollment_router
from orchestration.graphs.enrollment_graph import build_enrollment_graph
from langgraph.graph import END


class TestDisambiguateRouter:
    """Test _route_after_disambiguate router function."""

    @pytest.mark.parametrize("router_func", [prestream_router, student_router, enrollment_router])
    def test_router_returns_end_on_narrowing_question(self, router_func):
        """When llm_response is set AND prechecked_response is True, return END."""
        state: WorkflowState = {
            "llm_response": "What is your preferred course?",
            "metadata": {"prechecked_response": True},
        }
        assert router_func(state) == END

    @pytest.mark.parametrize("router_func", [prestream_router, student_router, enrollment_router])
    def test_router_returns_build_context_when_no_prechecked_response(self, router_func):
        """When prechecked_response is not set, return 'build_context'."""
        state: WorkflowState = {
            "llm_response": "Some response",
            "metadata": {},
        }
        assert router_func(state) == "build_context"

    @pytest.mark.parametrize("router_func", [prestream_router, student_router, enrollment_router])
    def test_router_returns_build_context_when_no_llm_response(self, router_func):
        """When llm_response is not set, return 'build_context'."""
        state: WorkflowState = {
            "metadata": {"prechecked_response": True},
        }
        assert router_func(state) == "build_context"

    @pytest.mark.parametrize("router_func", [prestream_router, student_router, enrollment_router])
    def test_router_returns_build_context_on_empty_state(self, router_func):
        """When state is empty, return 'build_context'."""
        state: WorkflowState = {}
        assert router_func(state) == "build_context"

    @pytest.mark.parametrize("router_func", [prestream_router, student_router, enrollment_router])
    def test_router_returns_build_context_when_metadata_missing(self, router_func):
        """When metadata is None, return 'build_context'."""
        state: WorkflowState = {
            "llm_response": "Some response",
        }
        assert router_func(state) == "build_context"


class TestGraphBuilding:
    """Test that graphs build successfully and contain the disambiguate node."""

    def test_student_query_graph_builds(self):
        """Verify student_query_graph builds without raising."""
        graph = build_student_query_graph()
        assert graph is not None

    def test_enrollment_graph_builds(self):
        """Verify enrollment_graph builds without raising."""
        graph = build_enrollment_graph()
        assert graph is not None

    def test_student_query_graph_has_disambiguate_node(self):
        """Verify disambiguate node is in student_query_graph."""
        graph = build_student_query_graph()
        assert hasattr(graph, "nodes"), "Graph should have nodes attribute"
        assert "disambiguate" in graph.nodes, "disambiguate should be in graph nodes"

    def test_enrollment_graph_has_disambiguate_node(self):
        """Verify disambiguate node is in enrollment_graph."""
        graph = build_enrollment_graph()
        assert hasattr(graph, "nodes"), "Graph should have nodes attribute"
        assert "disambiguate" in graph.nodes, "disambiguate should be in graph nodes"
