"""
MCP client gateway used by the backend proxy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import TextContent

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ToolInvocation:
    """Downstream MCP tool invocation descriptor."""

    name: str
    arguments: Dict[str, Any]


class McpGatewayError(Exception):
    """Raised when the downstream MCP call fails."""


class McpGateway:
    """Thin wrapper around the official Python MCP client."""

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        correlation_id: str,
    ) -> Any:
        """Invoke a single MCP tool.

        Args:
            tool_name: Tool to call on the private MCP server.
            arguments: Tool arguments.
            correlation_id: Correlation ID propagated to the downstream service.

        Returns:
            Parsed downstream payload.
        """
        results = await self.call_tools(
            invocations=[ToolInvocation(name=tool_name, arguments=arguments)],
            correlation_id=correlation_id,
        )
        return results[tool_name]

    async def call_tools(
        self,
        invocations: List[ToolInvocation],
        correlation_id: str,
    ) -> Dict[str, Any]:
        """Invoke multiple MCP tools over a single proxied session.

        Args:
            invocations: Ordered tool calls to execute.
            correlation_id: Correlation ID propagated to the downstream service.

        Returns:
            Mapping of tool name to parsed payload.

        Raises:
            McpGatewayError: If downstream session setup or tool execution fails.
        """
        headers = {
            "Authorization": f"Bearer {settings.proxy_mcp_api_key}",
            "X-Correlation-ID": correlation_id,
        }
        responses: Dict[str, Any] = {}

        try:
            async with sse_client(
                settings.proxy_mcp_sse_url,
                headers=headers,
                timeout=settings.proxy_mcp_http_timeout_seconds,
                sse_read_timeout=settings.proxy_mcp_sse_read_timeout_seconds,
            ) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()

                    for invocation in invocations:
                        logger.info(
                            "Calling downstream MCP tool",
                            extra={
                                "tool_name": invocation.name,
                                "correlation_id": correlation_id,
                            },
                        )
                        result = await session.call_tool(
                            invocation.name,
                            arguments=invocation.arguments,
                        )
                        if result.isError:
                            raise McpGatewayError(
                                f"Downstream MCP tool returned an error: {invocation.name}"
                            )
                        responses[invocation.name] = self._normalise_result(result)
        except Exception as exc:
            logger.exception(
                "Failed to execute downstream MCP calls",
                extra={"correlation_id": correlation_id},
            )
            if isinstance(exc, McpGatewayError):
                raise
            raise McpGatewayError("Failed to reach downstream MCP service") from exc

        return responses

    @staticmethod
    def _normalise_result(result: Any) -> Any:
        """Convert a tool result into JSON-friendly data.

        Args:
            result: ``CallToolResult`` from the MCP client.

        Returns:
            Parsed structured content, JSON-decoded text, or raw text.
        """
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured

        text_chunks = [
            block.text
            for block in getattr(result, "content", [])
            if isinstance(block, TextContent)
        ]
        if not text_chunks:
            return None

        if len(text_chunks) == 1:
            return McpGateway._maybe_parse_json(text_chunks[0])

        return [McpGateway._maybe_parse_json(chunk) for chunk in text_chunks]

    @staticmethod
    def _maybe_parse_json(text: str) -> Any:
        """Parse JSON text when possible.

        Args:
            text: Candidate JSON string.

        Returns:
            Parsed JSON value or the original text.
        """
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text


mcp_gateway = McpGateway()
