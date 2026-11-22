"""Re-export of the mcp package's in-process tool-call error type.

The rest of the codebase imports `ToolExecutionError` from here instead
of reaching into `mcp.server.mcpserver.exceptions` directly, since that
path already moved once (`mcp.server.fastmcp` -> `mcp.server.mcpserver`
between mcp 1.x and 2.x) -- one place to update if it moves again.
"""

from __future__ import annotations

from mcp.server.mcpserver.exceptions import UnexpectedToolError as ToolExecutionError

__all__ = ["ToolExecutionError"]
