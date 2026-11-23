"""Common interface for chat-completion providers.

Only one provider ships today (`openai_compat.OpenAICompatProvider`,
talking to a local vLLM server), but `modes/classic.py` and
`modes/codemode.py` are written against this interface only, so a
future cloud provider (e.g. Anthropic, the extension noted in
PLAN.md §10) can be added without touching either mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Set instead of raising when the model's tool-call arguments were not
    # valid JSON (observed with gpt-oss-20b's Harmony tool-call format: a
    # malformed function-call payload that vLLM's own tool parser already
    # tolerates -- it falls back to passing the raw text through -- but
    # that raw text is not valid JSON for *us* to decode as `arguments`).
    # `arguments` is {} in this case. Modes must check this before calling
    # the tool and feed it back as a normal tool-execution error (the same
    # retry path as "unknown tool" or a bad-argument TypeError), instead of
    # letting a bare JSONDecodeError escape and abort the whole trial.
    parse_error: str | None = None


@dataclass
class ChatMessage:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool" messages
    name: str | None = None  # tool name, set on role="tool" messages


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the tool's arguments


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class UsageMissingError(RuntimeError):
    """Raised when a provider response carries no token usage.

    Per CLAUDE.md §3.4, a trial without real usage data is invalid --
    it must never be estimated, so callers should mark the trial
    invalid rather than catch and ignore this.
    """


@dataclass
class ChatResponse:
    message: ChatMessage
    usage: Usage
    latency_seconds: float


class Provider(ABC):
    """A chat-completion backend. One request = one call to `chat()`."""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
    ) -> ChatResponse:
        """Send one chat-completion request and return the parsed response.

        Must raise `UsageMissingError` if the backend's response carries
        no token usage.
        """
        raise NotImplementedError
