"""48 realistic-but-unused "noise" tools, added to the MCP server for the
60-tool variant (tests H6: does a larger tool list cost more input
tokens in `classic` mode?). None of T1-T6's graders ever calls one of
these; they exist purely so the model sees 60 tool definitions instead
of 12.

Each tool is generated from a template via `exec` so it gets a real,
distinct name, signature and docstring that MCPServer can introspect for
its JSON schema -- a hand-written function per tool would be 48x the
same five lines.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

_NOISE_RESOURCES: tuple[str, ...] = (
    "warehouse",
    "shipment",
    "supplier",
    "employee",
    "return_request",
    "payment",
    "subscription",
    "contract",
    "lead",
    "opportunity",
    "review",
    "coupon",
    "currency_rate",
    "audit_log",
    "notification",
    "webhook",
    "api_key",
    "feature_flag",
    "inventory_item",
    "support_agent",
    "sla_policy",
    "campaign",
    "discount",
    "tax_rate",
)  # 24 resources x (list_ + get_) = 48 tools

_LIST_TEMPLATE = '''
def {list_name}(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """List {resource} records. Not used by any benchmark task; always empty."""
    return []
'''

_GET_TEMPLATE = '''
def {get_name}(record_id: str) -> dict[str, Any]:
    """Fetch one {resource} record by id. Not used by any benchmark task."""
    raise LookupError(f"{resource} {{record_id!r}} not found")
'''


def register_noise_tools(server: MCPServer) -> None:
    """Register the 48 noise tools on `server`."""
    for resource in _NOISE_RESOURCES:
        _register_pair(server, resource)


def _register_pair(server: MCPServer, resource: str) -> None:
    list_name = f"list_{resource}s"
    get_name = f"get_{resource}"

    namespace: dict[str, Any] = {"Any": Any}
    exec(_LIST_TEMPLATE.format(list_name=list_name, resource=resource), namespace)
    exec(_GET_TEMPLATE.format(get_name=get_name, resource=resource), namespace)

    server.tool()(namespace[list_name])
    server.tool()(namespace[get_name])
