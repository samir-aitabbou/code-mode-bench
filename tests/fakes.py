"""Test doubles shared across the suite. Never imported by `src/cmbench`
-- this is how modes/classic.py and modes/codemode.py get tested in CI
without a GPU or any real API (CLAUDE.md: "tests qui passent en CI sans
GPU (fournisseur simulé)").
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from cmbench.providers.base import (
    ChatMessage,
    ChatResponse,
    Provider,
    ToolCall,
    ToolSpec,
    Usage,
)


@dataclass
class StubProvider(Provider):
    """Returns pre-scripted responses in order, one per `chat()` call.
    An item may be an `Exception` instance instead of a `ChatResponse`,
    in which case `chat()` raises it (used to simulate e.g.
    `UsageMissingError` or a context-length error from the real vLLM
    provider, without needing a live server)."""

    responses: list[ChatResponse | Exception]
    calls: list[list[ChatMessage]] = field(default_factory=list)
    # Simulates a slow vLLM response, e.g. to test the trial wall-clock
    # timeout without a real server.
    delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        self._iter: Iterator[ChatResponse | Exception] = iter(self.responses)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
    ) -> ChatResponse:
        self.calls.append(list(messages))
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        try:
            item = next(self._iter)
        except StopIteration as exc:
            raise AssertionError("StubProvider ran out of scripted responses") from exc
        if isinstance(item, Exception):
            raise item
        return item


def usage(prompt_tokens: int = 10, completion_tokens: int = 5) -> Usage:
    return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def assistant_text(
    text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5
) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role="assistant", content=text),
        usage=usage(prompt_tokens, completion_tokens),
        latency_seconds=0.01,
    )


def assistant_tool_call(
    name: str,
    arguments: dict[str, object],
    *,
    call_id: str = "call_1",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    parse_error: str | None = None,
) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id=call_id, name=name, arguments=arguments, parse_error=parse_error)
            ],
        ),
        usage=usage(prompt_tokens, completion_tokens),
        latency_seconds=0.01,
    )
