"""Spawns the sandboxed subprocess that runs model-generated Python
code, and services its tool-call requests over the bridge protocol
(`cmbench.sandbox.bridge`) by calling `execute_tool`.

Isolation on Linux: no network access (best-effort via `unshare -rn`,
falling back to running unisolated with a logged warning if that fails
-- e.g. unprivileged user namespaces disabled) and a wall-clock
timeout. A memory limit is applied inside the child via
`resource.setrlimit` on POSIX. There is no sandboxing on Windows (used
only for local development; the real benchmark runs are all on the
Linux VM).
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cmbench.sandbox.bridge import (
    RpcResponse,
    build_child_script,
    encode_response,
    try_decode_request,
)

logger = logging.getLogger(__name__)

ToolExecutor = Callable[[str, dict[str, Any]], Any]

TRUNCATE_CHARS = 8000
MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    timed_out: bool
    exit_code: int | None
    num_tool_calls: int
    isolation: str


def _child_env() -> dict[str, str]:
    """Copy the current environment but drop anything that looks like a
    secret, so the sandboxed child never sees API keys etc."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(marker in k.upper() for marker in _SECRET_ENV_MARKERS)
    }


def _isolation_prefix() -> tuple[list[str], str]:
    """Return (command prefix, description) used to launch the child with
    no network access, or ([], description) if unavailable."""
    if sys.platform != "linux":
        return [], "none (non-Linux)"
    unshare = shutil.which("unshare")
    if unshare is None:
        return [], "none (unshare not found)"
    try:
        probe = subprocess.run([unshare, "-rn", "true"], capture_output=True, timeout=5)  # noqa: S603
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("unshare -rn probe failed to run (%s); no network isolation", exc)
        return [], "none (unshare -rn probe errored)"
    if probe.returncode != 0:
        logger.warning(
            "unshare -rn probe failed (rc=%s, stderr=%s); running sandbox without "
            "network isolation",
            probe.returncode,
            probe.stderr.decode(errors="replace"),
        )
        return [], "none (unshare -rn probe failed)"
    return [unshare, "-rn", "--"], "unshare -rn"


def _preexec_limits() -> Callable[[], None] | None:
    if sys.platform == "win32":
        return None

    def _set_limits() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))

    return _set_limits


def run_sandboxed(
    tools_module_source: str,
    user_code: str,
    execute_tool: ToolExecutor,
    *,
    timeout_seconds: float = 120.0,
) -> SandboxResult:
    """Run `user_code` (with a `tools` module built from
    `tools_module_source`) in a subprocess, servicing its tool calls via
    `execute_tool`, and return its captured output."""
    script = build_child_script(tools_module_source, user_code)
    isolation_prefix, isolation_desc = _isolation_prefix()

    process = subprocess.Popen(  # noqa: S603
        [*isolation_prefix, sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=_child_env(),
        preexec_fn=_preexec_limits(),
    )
    assert process.stdin is not None
    assert process.stdout is not None

    stdout_chunks: list[str] = []
    num_tool_calls = 0
    deadline = time.monotonic() + timeout_seconds
    timed_out = False

    try:
        while True:
            if time.monotonic() >= deadline:
                timed_out = True
                process.kill()
                break
            line = process.stdout.readline()
            if line == "":
                if process.poll() is not None:
                    break
                continue
            request = try_decode_request(line.rstrip("\n"))
            if request is None:
                stdout_chunks.append(line)
                continue
            num_tool_calls += 1
            try:
                result = execute_tool(request.tool, request.arguments)
                response = RpcResponse(call_id=request.call_id, result=result)
            except Exception as exc:  # noqa: BLE001 -- surfaced to the model as a tool error
                response = RpcResponse(call_id=request.call_id, error=str(exc))
            process.stdin.write(encode_response(response) + "\n")
            process.stdin.flush()
    finally:
        with contextlib.suppress(OSError):
            process.stdin.close()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

    stderr = process.stderr.read() if process.stderr else ""
    stdout = "".join(stdout_chunks)
    return SandboxResult(
        stdout=stdout[:TRUNCATE_CHARS],
        stderr=stderr[:TRUNCATE_CHARS],
        timed_out=timed_out,
        exit_code=process.returncode,
        num_tool_calls=num_tool_calls,
        isolation=isolation_desc,
    )
