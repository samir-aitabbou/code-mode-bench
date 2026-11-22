import asyncio
from pathlib import Path

import pytest

from cmbench.mcp_server.exceptions_compat import ToolExecutionError
from cmbench.mcp_server.server import build_server

EXPECTED_12_TOOLS = {
    "list_customers",
    "get_customer",
    "list_products",
    "get_product",
    "list_orders",
    "get_order",
    "list_invoices",
    "get_invoice",
    "search_tickets",
    "get_ticket",
    "update_customer_note",
    "send_email",
}


def test_server_exposes_exactly_12_tools(seeded_db: Path) -> None:
    server = build_server(seeded_db, with_noise_tools=False)
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == EXPECTED_12_TOOLS


def test_server_with_noise_exposes_60_tools(seeded_db: Path) -> None:
    server = build_server(seeded_db, with_noise_tools=True)
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert len(names) == 60
    assert names >= EXPECTED_12_TOOLS
    assert "list_warehouses" in names
    assert "get_warehouse" in names


def test_call_tool_get_customer(seeded_db: Path) -> None:
    server = build_server(seeded_db)

    async def run() -> dict[str, object]:
        result = await server.call_tool("get_customer", {"customer_id": "C0042"})
        return result.structured_content  # type: ignore[return-value]

    structured = asyncio.run(run())
    assert structured["id"] == "C0042"


def test_call_tool_list_orders_limit(seeded_db: Path) -> None:
    server = build_server(seeded_db)

    async def run() -> list[object]:
        result = await server.call_tool("list_orders", {"limit": 4})
        return result.structured_content["result"]  # type: ignore[index]

    rows = asyncio.run(run())
    assert len(rows) == 4


def test_call_tool_unknown_id_raises(seeded_db: Path) -> None:
    server = build_server(seeded_db)

    async def run() -> None:
        await server.call_tool("get_customer", {"customer_id": "C9999"})

    with pytest.raises(ToolExecutionError):
        asyncio.run(run())
