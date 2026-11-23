from typing import Any

from cmbench.modes.codemode import run_codemode_trial
from cmbench.providers.base import ToolSpec

from .fakes import StubProvider, assistant_text, assistant_tool_call

MCP_TOOLS = [
    ToolSpec(
        name="get_customer",
        description="Fetch one customer.",
        parameters={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    )
]


def test_codemode_trial_feeds_back_malformed_tool_call_arguments_for_retry() -> None:
    """Regression test, same bug as classic.py's equivalent (PROGRESS.md):
    a malformed execute_python tool call (ToolCall.parse_error set by
    OpenAICompatProvider.chat()) must be fed back as a tool-execution
    error, never passed to run_sandboxed with empty/garbage code."""
    provider = StubProvider(
        responses=[
            assistant_tool_call(
                "execute_python",
                {},
                parse_error="tool call arguments were not valid JSON (...); raw text: 'bad'",
            ),
            assistant_text('FINAL_ANSWER: "ok"'),
        ]
    )

    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("execute_tool must not be called for a malformed tool call")

    outcome = run_codemode_trial(
        provider, "system prompt", "task prompt", MCP_TOOLS, execute_tool, max_turns=15
    )

    assert outcome.had_error is True
    assert outcome.num_code_executions == 0
    assert outcome.stopped_reason == "final_answer"
    tool_message = outcome.transcript[3]
    assert tool_message.role == "tool"
    assert tool_message.content is not None
    assert "not valid JSON" in tool_message.content


def test_codemode_trial_executes_code_and_finishes() -> None:
    code = "import tools\nc = tools.get_customer(customer_id='C0001')\nprint(c['email'])"
    provider = StubProvider(
        responses=[
            assistant_tool_call(
                "execute_python", {"code": code}, prompt_tokens=200, completion_tokens=40
            ),
            assistant_text(
                'FINAL_ANSWER: "a@example.com"', prompt_tokens=250, completion_tokens=10
            ),
        ]
    )

    calls: list[tuple[str, dict[str, Any]]] = []

    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        calls.append((name, args))
        return {"id": "C0001", "email": "a@example.com"}

    outcome = run_codemode_trial(
        provider, "system prompt", "task prompt", MCP_TOOLS, execute_tool, max_turns=15
    )

    assert outcome.stopped_reason == "final_answer"
    assert outcome.final_answer_text == '"a@example.com"'
    assert outcome.num_requests == 2
    assert outcome.num_code_executions == 1
    assert outcome.num_tool_calls == 1
    assert calls == [("get_customer", {"customer_id": "C0001"})]
    # the generated tools module is embedded in the system prompt the model saw
    assert "def get_customer(customer_id: str) -> Any:" in (outcome.transcript[0].content or "")


def test_codemode_trial_reports_code_error_without_crashing_harness() -> None:
    code = "raise RuntimeError('boom')"
    provider = StubProvider(
        responses=[
            assistant_tool_call("execute_python", {"code": code}),
            assistant_text("FINAL_ANSWER: null"),
        ]
    )

    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("not expected")

    outcome = run_codemode_trial(provider, "sys", "task", MCP_TOOLS, execute_tool, max_turns=15)

    tool_messages = [m for m in outcome.transcript if m.role == "tool"]
    assert len(tool_messages) == 1
    assert "boom" in (tool_messages[0].content or "")
    assert outcome.stopped_reason == "final_answer"


def test_codemode_trial_hits_max_turns() -> None:
    responses = [assistant_tool_call("execute_python", {"code": "pass"}) for _ in range(2)]
    provider = StubProvider(responses=responses)

    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("not expected")

    outcome = run_codemode_trial(provider, "sys", "task", MCP_TOOLS, execute_tool, max_turns=2)

    assert outcome.stopped_reason == "max_turns"
    assert outcome.num_code_executions == 2
