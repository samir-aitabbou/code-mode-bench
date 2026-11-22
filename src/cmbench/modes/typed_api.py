"""Generates the typed `tools` module shown to the model in codemode
(CLAUDE.md §3.3): one Python function per MCP tool, with real type
annotations and the same default values as the tool's JSON schema, so
what the model reads in the prompt is exactly what runs in the sandbox.
"""

from __future__ import annotations

from typing import Any

from cmbench.providers.base import ToolSpec

_JSON_TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
}


def _schema_to_annotation(schema: dict[str, Any]) -> str:
    if "anyOf" in schema:
        parts = [_schema_to_annotation(s) for s in schema["anyOf"]]
        return " | ".join(dict.fromkeys(parts))  # dedupe while keeping order
    json_type = schema.get("type")
    if json_type == "null":
        return "None"
    if json_type == "array":
        item_schema = schema.get("items")
        inner = _schema_to_annotation(item_schema) if item_schema else "Any"
        return f"list[{inner}]"
    if json_type == "object":
        return "dict[str, Any]"
    if isinstance(json_type, str) and json_type in _JSON_TYPE_MAP:
        return _JSON_TYPE_MAP[json_type]
    return "Any"


def generate_tool_stub(tool: ToolSpec) -> str:
    """Render one tool as a typed Python function definition that calls
    `_call(name, kwargs)` -- the bridge RPC helper injected by
    `cmbench.sandbox.bridge.build_child_script` at sandbox runtime."""
    schema = tool.parameters
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))

    required_params = [n for n in properties if n in required]
    optional_params = [n for n in properties if n not in required]

    param_strs: list[str] = []
    for name in required_params:
        annotation = _schema_to_annotation(properties[name])
        param_strs.append(f"{name}: {annotation}")
    for name in optional_params:
        sub = properties[name]
        annotation = _schema_to_annotation(sub)
        default = sub.get("default")
        param_strs.append(f"{name}: {annotation} = {default!r}")

    signature = ", ".join(param_strs)
    kwargs_literal = "{" + ", ".join(f'"{n}": {n}' for n in properties) + "}"
    doc = tool.description.replace('"""', "'").strip()

    return (
        f"def {tool.name}({signature}) -> Any:\n"
        f'    """{doc}"""\n'
        f"    return _call({tool.name!r}, {kwargs_literal})\n"
    )


def generate_tools_module(tools: list[ToolSpec]) -> str:
    """Render the full `tools` module source shown to the model."""
    header = (
        '"""Typed wrappers over the MCP CRM tools, generated for this trial. '
        "Calls are forwarded through the sandbox bridge to the real MCP "
        'server; the CRM data they see/change is the same as in classic mode."""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
    )
    body = "\n".join(generate_tool_stub(t) for t in tools)
    return header + body
