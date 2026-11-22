"""Converts MCP `Tool` objects (as returned by `MCPServer.list_tools()`)
into the provider-facing `ToolSpec` used by both agent modes."""

from __future__ import annotations

from mcp.types import Tool

from cmbench.providers.base import ToolSpec


def mcp_tool_to_spec(tool: Tool) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=tool.description or "",
        parameters=tool.input_schema,
    )


def mcp_tools_to_specs(tools: list[Tool]) -> list[ToolSpec]:
    return [mcp_tool_to_spec(t) for t in tools]
