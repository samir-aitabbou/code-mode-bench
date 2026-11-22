from typing import Any

from cmbench.modes.classic import run_classic_trial
from cmbench.providers.base import ToolSpec

from .fakes import StubProvider, assistant_text, assistant_tool_call

TOOLS = [
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


def test_classic_trial_calls_tool_then_finishes() -> None:
    provider = StubProvider(
        responses=[
            assistant_tool_call(
                "get_customer",
                {"customer_id": "C0001"},
                prompt_tokens=100,
                completion_tokens=20,
            ),
            assistant_text(
                'FINAL_ANSWER: {"email": "a@example.com"}',
                prompt_tokens=150,
                completion_tokens=10,
            ),
        ]
    )

    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        assert name == "get_customer"
        assert args == {"customer_id": "C0001"}
        return {"id": "C0001", "email": "a@example.com"}

    outcome = run_classic_trial(
        provider, "system prompt", "task prompt", TOOLS, execute_tool, max_turns=15
    )

    assert outcome.stopped_reason == "final_answer"
    assert outcome.final_answer_text == '{"email": "a@example.com"}'
    assert outcome.num_requests == 2
    assert outcome.num_tool_calls == 1
    assert outcome.total_usage.prompt_tokens == 250
    assert outcome.total_usage.completion_tokens == 30
    # transcript: system, user, assistant(tool_call), tool, assistant(final)
    roles = [m.role for m in outcome.transcript]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_classic_trial_hits_max_turns_without_final_answer() -> None:
    responses = [assistant_tool_call("get_customer", {"customer_id": "C0001"}) for _ in range(3)]
    provider = StubProvider(responses=responses)

    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        return {"id": "C0001"}

    outcome = run_classic_trial(provider, "sys", "task", TOOLS, execute_tool, max_turns=3)

    assert outcome.stopped_reason == "max_turns"
    assert outcome.final_answer_text is None
    assert outcome.num_requests == 3


def test_classic_trial_tool_error_is_surfaced_as_tool_message() -> None:
    provider = StubProvider(
        responses=[
            assistant_tool_call("get_customer", {"customer_id": "C9999"}),
            assistant_text("FINAL_ANSWER: null"),
        ]
    )

    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        raise ValueError("customer not found")

    outcome = run_classic_trial(provider, "sys", "task", TOOLS, execute_tool, max_turns=15)

    tool_messages = [m for m in outcome.transcript if m.role == "tool"]
    assert len(tool_messages) == 1
    assert "customer not found" in (tool_messages[0].content or "")
    assert outcome.stopped_reason == "final_answer"


def test_classic_trial_nudges_when_no_tool_call_and_no_final_answer() -> None:
    provider = StubProvider(
        responses=[
            assistant_text("I am thinking about it."),
            assistant_text("FINAL_ANSWER: 42"),
        ]
    )

    def execute_tool(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("not expected")

    outcome = run_classic_trial(provider, "sys", "task", TOOLS, execute_tool, max_turns=15)

    assert outcome.stopped_reason == "final_answer"
    assert outcome.final_answer_text == "42"
    user_messages = [m for m in outcome.transcript if m.role == "user"]
    assert len(user_messages) == 2  # original task + the reminder
