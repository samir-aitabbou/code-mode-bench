from typing import Any

from cmbench.sandbox.runner import run_sandboxed


def test_sandbox_runs_code_and_captures_stdout() -> None:
    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("not expected to be called")

    result = run_sandboxed("", "print('hello')", execute_tool, timeout_seconds=10)
    assert result.stdout == "hello\n"
    assert result.exit_code == 0
    assert not result.timed_out
    assert result.num_tool_calls == 0


def test_sandbox_round_trips_tool_calls() -> None:
    tools_src = 'def add(a, b):\n    return _call("add", {"a": a, "b": b})\n'
    user_code = "import tools\nprint(tools.add(a=2, b=3))\n"

    calls: list[tuple[str, dict[str, Any]]] = []

    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        calls.append((name, args))
        return args["a"] + args["b"]

    result = run_sandboxed(tools_src, user_code, execute_tool, timeout_seconds=10)
    assert result.stdout == "5\n"
    assert result.num_tool_calls == 1
    assert calls == [("add", {"a": 2, "b": 3})]


def test_sandbox_surfaces_tool_error_to_child() -> None:
    tools_src = 'def boom():\n    return _call("boom", {})\n'
    user_code = "import tools\ntools.boom()\n"

    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        raise ValueError("not found")

    result = run_sandboxed(tools_src, user_code, execute_tool, timeout_seconds=10)
    assert result.exit_code != 0
    assert "not found" in result.stderr


def test_sandbox_timeout() -> None:
    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("not expected to be called")

    result = run_sandboxed(
        "", "import time\nprint('before')\ntime.sleep(5)\n", execute_tool, timeout_seconds=0.5
    )
    assert result.timed_out
    assert "before" in result.stdout


def test_sandbox_truncates_long_output() -> None:
    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("not expected to be called")

    user_code = "print('x' * 20000)"
    result = run_sandboxed("", user_code, execute_tool, timeout_seconds=10)
    assert len(result.stdout) <= 8000
