"""
FastMCP application entrypoint — "LPUAI Central".

Initialises the FastMCP server, registers all tools, resources, and prompt
templates from their respective sub-packages, and starts the Streamable HTTP
(SSE) transport on port 8000.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Literal

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from shared.config import settings
from shared.logger import get_logger
from shared.reranker import warmup as warmup_reranker

# ── Tool / resource / prompt imports ────────────────────────────────────────
from mcp_server.tools import student_tools, context_tools, workflow_tools, chat_tools, lead_tools
from mcp_server.resources import student_resource
from mcp_server.prompts.workflow_prompts import (
    get_workflow_prompt,
    list_workflow_prompts,
)

logger = get_logger(__name__)

from shared.prompt_loader import load_prompt, PromptNotFoundError

def _validate_prompts() -> None:
    expected_prompts = [
        "classify_intent",
        "extract_entities",
        "reasoning_synthesis",
        "policy_response",
        "study_plan_generator",
        "data_mutation_confirm"
    ]
    for p in expected_prompts:
        try:
            load_prompt(p)
        except PromptNotFoundError as e:
            logger.warning("Startup prompt validation warning: %s", e)

_validate_prompts()

# ── FastMCP app ─────────────────────────────────────────────────────────────

_SERVER_START = time.time()

mcp = FastMCP(
    name="LPU AI Nexus",
    instructions=(
        "Central AI module for an educational platform.  Provides tools for "
        "student data access, context retrieval, workflow detection, and "
        "AI-powered reasoning."
    ),
)

# ── Register MCP Tools ─────────────────────────────────────────────────────


@mcp.tool()
async def get_student(student_id: str) -> str:
    """Retrieve a student record by ID."""
    result = await student_tools.get_student(student_id)
    return json.dumps(result, default=str)


@mcp.tool()
async def update_student(student_id: str, updates: str) -> str:
    """Update a student record. `updates` is a JSON string of field→value pairs."""
    parsed: Dict[str, Any] = json.loads(updates)
    result = await student_tools.update_student(student_id, parsed)
    return json.dumps(result, default=str)


@mcp.tool()
async def detect_workflow(user_input: str, student_id: str) -> str:
    """Classify user intent and return the workflow name."""
    return await workflow_tools.detect_workflow(user_input, student_id)


@mcp.tool()
async def get_context(student_id: str, query: str) -> str:
    """Fetch merged context from Redis cache + pgvector semantic search."""
    result = await context_tools.get_context(student_id, query)
    return json.dumps(result, default=str)


@mcp.tool()
async def build_context(student_id: str, data: str) -> str:
    """Embed and store student context. `data` is a JSON string."""
    parsed: Dict[str, Any] = json.loads(data)
    success = await context_tools.build_context(student_id, parsed)
    return json.dumps({"success": success})


@mcp.tool()
async def chat_complete(
    user_input: str,
    actor_id: str,
    actor_type: str = "student",
    app_id: str = "",
    session_id: str = "",
    force_workflow: str | None = None,
    verified_phone: str = "",
) -> str:
    """Run the full AI chat pipeline: classify intent, fetch data, reason, and respond."""
    result = await chat_tools.chat_complete(
        user_input=user_input,
        actor_id=actor_id,
        actor_type=actor_type,
        app_id=app_id,
        session_id=session_id,
        force_workflow=force_workflow,
        verified_phone=verified_phone,
    )
    return json.dumps(result, default=str)


@mcp.tool()
async def chat_complete_prestream(
    user_input: str,
    actor_id: str,
    actor_type: str = "student",
    app_id: str = "",
    session_id: str = "",
    force_workflow: str | None = None,
) -> str:
    """Run the pipeline up to context building for streaming. Returns prepared state."""
    result = await chat_tools.chat_complete_prestream(
        user_input=user_input,
        actor_id=actor_id,
        actor_type=actor_type,
        app_id=app_id,
        session_id=session_id,
        force_workflow=force_workflow,
    )
    return json.dumps(result, default=str)


@mcp.tool()
async def upsert_lead(
    session_id: str,
    name: str | None = None,
    mobile: str | None = None,
    email: str | None = None,
    program_level: str | None = None,
    program_names: str | None = None,
    state: str | None = None,
    city: str | None = None,
    address: str | None = None,
    status: str | None = None,
) -> str:
    """Upsert lead information to the external lead storage."""
    result = await lead_tools.upsert_lead(
        session_id=session_id,
        name=name,
        mobile=mobile,
        email=email,
        program_level=program_level,
        program_names=program_names,
        state=state,
        city=city,
        address=address,
        status=status,
    )
    return json.dumps(result, default=str)


# ── Register MCP Resources ────────────────────────────────────────────────


@mcp.resource("student://{student_id}/profile")
async def student_profile(student_id: str) -> str:
    """Read-only student profile resource."""
    result = await student_resource.read_student_profile(student_id)
    return json.dumps(result, default=str)


# ── Register MCP Prompts ──────────────────────────────────────────────────


@mcp.prompt()
async def workflow_prompt(workflow_name: str) -> str:
    """Return the prompt template for a given workflow type."""
    prompt_def = get_workflow_prompt(workflow_name)
    return prompt_def["template"]


# ── Custom HTTP Routes ────────────────────────────────────────────────────


@mcp.custom_route("/health/version", methods=["GET"])
async def health_version(request: Request) -> JSONResponse:
    """Return the health status and version of the MCP server."""
    uptime_s = round(time.time() - _SERVER_START, 2)
    return JSONResponse({
        "status": "ok",
        "version": "1.0.0",
        "environment": settings.environment,
        "git_branch": settings.git_branch,
        "build_timestamp": settings.build_timestamp,
        "server": "LPU AI NEXUS",
        "transport": "sse",
        "uptime_seconds": uptime_s,
    })


@mcp.custom_route("/api/cache/store", methods=["POST"])
async def api_cache_store(request: Request) -> JSONResponse:
    """Store semantic cache entries via an internal endpoint.
    
    This must be handled by the MCP Server since ChromaDB PersistentClient
    cannot be safely shared across processes. The backend proxy delegates
    its cache writing to this endpoint.
    """
    try:
        from cache_module.semantic_cache import store_response
        data = await request.json()
        await store_response(**data)
        return JSONResponse({"status": "stored"})
    except Exception as exc:
        logger.error("Failed to store cache via API: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


# ── ASGI app factory (with HTTP middleware) ─────────────────────────────────


def create_app():
    """Build the Starlette ASGI app for MCP server.
    
    Note: API authentication is now handled by the backend proxy.
    This MCP server is internal/private and only accessed from the proxy.
    No middleware needed since correlation IDs are handled by the proxy.
    """
    try:
        warmup_reranker()
    except Exception as exc:
        logger.warning("Reranker warmup failed during startup: %s", exc)
    app = mcp.http_app(transport="sse")
    # Explicitly ensure no middleware is applied to avoid conflicts with SSE transport
    return app


# ── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting LPUAI Central MCP server: NEXUS",
        extra={"port": settings.mcp_server_port, "transport": "sse"},
    )
    app = create_app()
    uvicorn.run(app, host=settings.mcp_server_host, port=settings.mcp_server_port)
