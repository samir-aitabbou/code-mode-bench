"""The RPC bridge between the sandboxed child process (running the
model's generated code) and the harness process (which holds the real
MCP client/tool implementation).

Protocol: line-based JSON over the child's stdout/stdin. A line starting
with `REQUEST_PREFIX` is a tool call from the child; everything else on
stdout is the child program's genuine output and is returned to the
model (truncated). The harness answers a request by writing one
`RESPONSE_PREFIX` line to the child's stdin.

This module only builds/parses the protocol and the child-side script;
`cmbench.sandbox.runner` owns the actual subprocess and the OS-level
sandboxing (timeout, memory, no network).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

REQUEST_PREFIX = "##CMBENCH_RPC_REQUEST##"
RESPONSE_PREFIX = "##CMBENCH_RPC_RESPONSE##"


@dataclass
class RpcRequest:
    call_id: str
    tool: str
    arguments: dict[str, Any]


@dataclass
class RpcResponse:
    call_id: str
    result: Any = None
    error: str | None = None


def encode_request(req: RpcRequest) -> str:
    payload = {"call_id": req.call_id, "tool": req.tool, "arguments": req.arguments}
    return REQUEST_PREFIX + json.dumps(payload)


def try_decode_request(line: str) -> RpcRequest | None:
    if not line.startswith(REQUEST_PREFIX):
        return None
    payload = json.loads(line[len(REQUEST_PREFIX) :])
    return RpcRequest(
        call_id=payload["call_id"], tool=payload["tool"], arguments=payload["arguments"]
    )


def encode_response(resp: RpcResponse) -> str:
    payload = {"call_id": resp.call_id, "result": resp.result, "error": resp.error}
    return RESPONSE_PREFIX + json.dumps(payload)


def try_decode_response(line: str) -> RpcResponse | None:
    if not line.startswith(RESPONSE_PREFIX):
        return None
    payload = json.loads(line[len(RESPONSE_PREFIX) :])
    return RpcResponse(
        call_id=payload["call_id"], result=payload.get("result"), error=payload.get("error")
    )


_CHILD_SCRIPT_TEMPLATE = '''
import json
import sys
import types

_REQUEST_PREFIX = {request_prefix!r}
_RESPONSE_PREFIX = {response_prefix!r}


def _call(tool, kwargs):
    import uuid

    call_id = str(uuid.uuid4())
    request = {{"call_id": call_id, "tool": tool, "arguments": kwargs}}
    print(_REQUEST_PREFIX + json.dumps(request), flush=True)
    while True:
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError("sandbox bridge closed unexpectedly")
        line = line.rstrip("\\n")
        if not line.startswith(_RESPONSE_PREFIX):
            continue
        payload = json.loads(line[len(_RESPONSE_PREFIX) :])
        if payload["call_id"] != call_id:
            continue
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        return payload.get("result")


_tools_module = types.ModuleType("tools")
_tools_module.__dict__["_call"] = _call
exec(compile({tools_module_source!r}, "<tools>", "exec"), _tools_module.__dict__)
sys.modules["tools"] = _tools_module

exec(
    compile({user_code!r}, "<model_code>", "exec"),
    {{"tools": _tools_module, "__name__": "__main__"}},
)
'''


def build_child_script(tools_module_source: str, user_code: str) -> str:
    """Build the full Python source run inside the sandboxed subprocess.

    Both `tools_module_source` and `user_code` are embedded via `repr()`
    (not raw string splicing), so arbitrary model-generated code -- any
    quotes, backslashes, or triple-quoted strings -- can never break out
    of its slot in the template.
    """
    return _CHILD_SCRIPT_TEMPLATE.format(
        request_prefix=REQUEST_PREFIX,
        response_prefix=RESPONSE_PREFIX,
        tools_module_source=tools_module_source,
        user_code=user_code,
    )
