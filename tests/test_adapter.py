import asyncio
from pathlib import Path

from cmbench.mcp_server.adapter import mcp_tools_to_specs
from cmbench.mcp_server.server import build_server


def test_adapter_converts_all_tools(seeded_db: Path) -> None:
    server = build_server(seeded_db)
    mcp_tools = asyncio.run(server.list_tools())
    specs = mcp_tools_to_specs(mcp_tools)

    assert len(specs) == len(mcp_tools)
    names = {s.name for s in specs}
    assert "get_customer" in names
    assert "list_orders" in names

    get_customer = next(s for s in specs if s.name == "get_customer")
    assert get_customer.parameters["required"] == ["customer_id"]
    assert get_customer.description
