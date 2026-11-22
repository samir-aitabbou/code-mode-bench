"""Classic MCP tool-calling agent loop: the model is given the tool
definitions directly and calls them one at a time, with each full
result returned into its context.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cmbench.modes.common import FINAL_ANSWER_MARKER, REMINDER_TEXT
from cmbench.providers.base import ChatMessage, Provider, ToolSpec, Usage

logger = logging.getLogger(__name__)

ToolExecutor = Callable[[str, dict[str, Any]], Any]


@dataclass
class TrialOutcome:
    final_answer_text: str | None
    transcript: list[ChatMessage] = field(default_factory=list)
    num_requests: int = 0
    num_tool_calls: int = 0
    total_usage: Usage = field(default_factory=lambda: Usage(0, 0))
    per_request_latencies: list[float] = field(default_factory=list)
    stopped_reason: str = "max_turns"  # "final_answer" | "max_turns"


def run_classic_trial(
    provider: Provider,
    system_prompt: str,
    task_prompt: str,
    tools: list[ToolSpec],
    execute_tool: ToolExecutor,
    *,
    max_turns: int = 15,
    temperature: float = 0.0,
) -> TrialOutcome:
    """Run one classic-mode trial to completion (final answer or
    max_turns). Exceptions from `provider.chat` (e.g. a context-length
    error from vLLM) propagate to the caller, which is responsible for
    classifying the failure (see `cmbench.runner`)."""
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=task_prompt),
    ]
    outcome = TrialOutcome(final_answer_text=None)

    for _turn in range(max_turns):
        response = provider.chat(messages, tools=tools, temperature=temperature)
        outcome.num_requests += 1
        outcome.total_usage = Usage(
            prompt_tokens=outcome.total_usage.prompt_tokens + response.usage.prompt_tokens,
            completion_tokens=outcome.total_usage.completion_tokens
            + response.usage.completion_tokens,
        )
        outcome.per_request_latencies.append(response.latency_seconds)
        messages.append(response.message)

        if response.message.tool_calls:
            outcome.num_tool_calls += len(response.message.tool_calls)
            for tc in response.message.tool_calls:
                try:
                    result = execute_tool(tc.name, tc.arguments)
                    result_text = str(result)
                except Exception as exc:  # noqa: BLE001 -- a tool error is a normal trial outcome
                    result_text = f"ERROR: {exc}"
                messages.append(
                    ChatMessage(
                        role="tool", content=result_text, tool_call_id=tc.id, name=tc.name
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
