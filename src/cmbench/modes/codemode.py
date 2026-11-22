"""Code Mode agent loop: the model sees a single tool, `execute_python`,
and a typed `tools` module (generated from the same MCP tool schemas
`classic` mode uses). Its code runs in a sandbox (`cmbench.sandbox.runner`)
with no network access; only stdout/stderr (truncated) comes back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cmbench.modes.common import FINAL_ANSWER_MARKER, REMINDER_TEXT
from cmbench.modes.typed_api import generate_tools_module
from cmbench.providers.base import ChatMessage, Provider, ToolSpec, Usage
from cmbench.sandbox.runner import SandboxResult, ToolExecutor, run_sandboxed

logger = logging.getLogger(__name__)

EXECUTE_PYTHON_TOOL = ToolSpec(
    name="execute_python",
    description=(
        "Execute Python code in a sandbox with no network access. The code "
        "can call the functions in the `tools` module (given to you above) "
        "to read/write the CRM. Only stdout/stderr (truncated to 8000 "
        "characters) is returned to you; nothing else about the sandbox "
        "persists between calls except the CRM data itself."
    ),
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source code to run."}},
        "required": ["code"],
    },
)


@dataclass
class TrialOutcome:
    final_answer_text: str | None
    transcript: list[ChatMessage] = field(default_factory=list)
    num_requests: int = 0
    num_code_executions: int = 0
    num_tool_calls: int = 0  # tool calls made by the sandboxed code, summed across executions
    total_usage: Usage = field(default_factory=lambda: Usage(0, 0))
    per_request_latencies: list[float] = field(default_factory=list)
    stopped_reason: str = "max_turns"  # "final_answer" | "max_turns"


def _format_sandbox_result(result: SandboxResult) -> str:
    parts = [f"stdout:\n{result.stdout}"]
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr}")
    if result.timed_out:
        parts.append("(execution timed out)")
    elif result.exit_code not in (0, None):
        parts.append(f"(process exited with code {result.exit_code})")
    return "\n".join(parts)


def run_codemode_trial(
    provider: Provider,
    system_prompt: str,
    task_prompt: str,
    mcp_tools: list[ToolSpec],
    execute_tool: ToolExecutor,
    *,
    max_turns: int = 15,
    temperature: float = 0.0,
    max_code_exec_seconds: float = 120.0,
) -> TrialOutcome:
    """Run one codemode-mode trial to completion (final answer or
    max_turns). Exceptions from `provider.chat` propagate to the caller
    (see `cmbench.runner`), same as in classic mode."""
    tools_module_source = generate_tools_module(mcp_tools)
    system_with_api = (
        f"{system_prompt}\n\n"
        "You interact with the CRM only by writing Python code executed via "
        "the `execute_python` tool. The following `tools` module is "
        "available inside that sandbox:\n\n```python\n"
        f"{tools_module_source}\n```"
    )
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system_with_api),
        ChatMessage(role="user", content=task_prompt),
    ]
    outcome = TrialOutcome(final_answer_text=None)

    for _turn in range(max_turns):
        response = provider.chat(messages, tools=[EXECUTE_PYTHON_TOOL], temperature=temperature)
        outcome.num_requests += 1
        outcome.total_usage = Usage(
            prompt_tokens=outcome.total_usage.prompt_tokens + response.usage.prompt_tokens,
            completion_tokens=outcome.total_usage.completion_tokens
            + response.usage.completion_tokens,
        )
        outcome.per_request_latencies.append(response.latency_seconds)
        messages.append(response.message)

        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                code = str(tc.arguments.get("code", ""))
                outcome.num_code_executions += 1
                result = run_sandboxed(
                    tools_module_source,
                    code,
                    execute_tool,
                    timeout_seconds=max_code_exec_seconds,
                )
                outcome.num_tool_calls += result.num_tool_calls
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=_format_sandbox_result(result),
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )
            continue

        content = response.message.content or ""
        if FINAL_ANSWER_MARKER in content:
            outcome.final_answer_text = content.split(FINAL_ANSWER_MARKER, 1)[1].strip()
            outcome.stopped_reason = "final_answer"
            break

        messages.append(ChatMessage(role="user", content=REMINDER_TEXT))

    outcome.transcript = messages
    return outcome
