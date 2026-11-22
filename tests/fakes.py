"""Test doubles shared across the suite. Never imported by `src/cmbench`
-- this is how modes/classic.py and modes/codemode.py get tested in CI
without a GPU or any real API (CLAUDE.md: "tests qui passent en CI sans
GPU (fournisseur simulé)").
"""

from __future__ import annotations

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
    """Returns pre-scripted responses in order, one per `chat()` call."""

    responses: list[ChatResponse]
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._iter: Iterator[ChatResponse] = iter(self.responses)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
    ) -> ChatResponse:
        self.calls.append(list(messages))
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise AssertionError("StubProvider ran out of scripted responses") from exc


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
) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        ),
        usage=usage(prompt_tokens, completion_tokens),
        latency_seconds=0.01,
    )
