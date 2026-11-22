from typing import Any

from cmbench.modes.typed_api import generate_tool_stub, generate_tools_module
from cmbench.providers.base import ToolSpec


def _exec_stub(stub_source: str, calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    def _call(name: str, kwargs: dict[str, Any]) -> str:
        calls.append((name, kwargs))
        return "ok"

    namespace: dict[str, Any] = {"_call": _call, "Any": Any}
    exec(compile(stub_source, "<stub>", "exec"), namespace)
    return namespace


def test_required_and_optional_params() -> None:
    tool = ToolSpec(
        name="get_customer",
        description="Fetch one customer by id.",
        parameters={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    )
    stub = generate_tool_stub(tool)
    assert "def get_customer(customer_id: str) -> Any:" in stub

    calls: list[tuple[str, dict[str, Any]]] = []
    namespace = _exec_stub(stub, calls)
    namespace["get_customer"](customer_id="C0001")
    assert calls == [("get_customer", {"customer_id": "C0001"})]


def test_optional_params_with_defaults_and_anyof() -> None:
    tool = ToolSpec(
        name="list_orders",
        description="List orders.",
        parameters={
            "type": "object",
            "properties": {
                "customer_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                "limit": {"type": "integer", "default": 5000},
            },
            "required": [],
        },
    )
    stub = generate_tool_stub(tool)
    assert "customer_id: str | None = None" in stub
    assert "limit: int = 5000" in stub

    calls: list[tuple[str, dict[str, Any]]] = []
    namespace = _exec_stub(stub, calls)
    namespace["list_orders"]()
    assert calls == [("list_orders", {"customer_id": None, "limit": 5000})]


def test_generated_module_is_valid_python_for_12_tools(seeded_db: Any) -> None:
    import asyncio

    from cmbench.mcp_server.adapter import mcp_tools_to_specs
    from cmbench.mcp_server.server import build_server

    server = build_server(seeded_db)
    mcp_tools = asyncio.run(server.list_tools())
    specs = mcp_tools_to_specs(mcp_tools)

    module_source = generate_tools_module(specs)
    compile(module_source, "<tools>", "exec")  # must not raise

    for spec in specs:
        assert f"def {spec.name}(" in module_source
