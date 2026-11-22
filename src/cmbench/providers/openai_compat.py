"""Provider talking to a local vLLM server through its OpenAI-compatible
API. Uses a dummy API key -- vLLM does not check it, but the `openai`
SDK requires a non-empty string. Never points at a real OpenAI/Anthropic
endpoint (see CLAUDE.md §0.1: no paid API in this benchmark).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from cmbench.providers.base import (
    ChatMessage,
    ChatResponse,
    Provider,
    ToolCall,
    ToolSpec,
    Usage,
    UsageMissingError,
)


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        if m.role == "tool":
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def _to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


@dataclass
class OpenAICompatProvider(Provider):
    """Chat-completion provider for vLLM's OpenAI-compatible endpoint."""

    base_url: str
    model: str
    api_key: str = "EMPTY"

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
    ) -> ChatResponse:
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = _to_openai_tools(tools)
            extra["tool_choice"] = "auto"

        start = time.monotonic()
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(messages),  # type: ignore[arg-type]
            temperature=temperature,
            **extra,
        )
        latency = time.monotonic() - start

        if completion.usage is None:
            raise UsageMissingError(
                f"vLLM response for model={self.model!r} has no usage field; "
                "this trial must be marked invalid, never estimated (CLAUDE.md §3.4)"
            )

        choice = completion.choices[0]
        raw_message = choice.message
        tool_calls = []
        for tc in raw_message.tool_calls or []:
            if tc.type != "function":
                raise ValueError(
                    f"unexpected tool_call type {tc.type!r} from vLLM; only "
                    "function-style tool calls are supported"
                )
            tool_calls.append(
                ToolCall(
                    id=tc.id or str(uuid.uuid4()),
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                )
            )
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=raw_message.content,
                tool_calls=tool_calls,
            ),
            usage=Usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            ),
            latency_seconds=latency,
        )
