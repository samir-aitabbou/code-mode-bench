"""Unit tests for OpenAICompatProvider that don't need a live vLLM server:
mocks `self._client.chat.completions.create` directly. Excluded from
nothing special -- CLAUDE.md's "no GPU/API in CI" is about not making real
network calls, which this test never does.
"""

from __future__ import annotations

from types import SimpleNamespace

from cmbench.providers.base import ChatMessage
from cmbench.providers.openai_compat import OpenAICompatProvider


def _fake_completion(*, content: str | None, tool_calls: list[SimpleNamespace]) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=[SimpleNamespace(message=message)],
    )


def test_chat_sets_parse_error_on_malformed_tool_call_arguments() -> None:
    """Regression test for the gpt-oss-20b bug (PROGRESS.md): vLLM's own
    "openai" tool parser can return 200 OK with non-JSON `arguments` text
    (it tolerates a malformed Harmony tool call on its side, falling back
    to the raw text). Previously `json.loads` on that raw text raised
    uncaught, aborting the whole trial. It must now be caught and
    recorded as ToolCall.parse_error, with empty `arguments`."""
    provider = OpenAICompatProvider(base_url="http://localhost:8000/v1", model="gpt-oss-20b")
    bad_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="list_invoices", arguments="not json"),
    )
    provider._client.chat.completions.create = lambda **kwargs: _fake_completion(  # type: ignore[method-assign]
        content=None, tool_calls=[bad_call]
    )

    response = provider.chat([ChatMessage(role="user", content="hi")])

    assert len(response.message.tool_calls) == 1
    tc = response.message.tool_calls[0]
    assert tc.arguments == {}
    assert tc.parse_error is not None
    assert "not valid JSON" in tc.parse_error
    assert "not json" in tc.parse_error


def test_chat_parses_well_formed_tool_call_arguments_normally() -> None:
    provider = OpenAICompatProvider(base_url="http://localhost:8000/v1", model="gpt-oss-20b")
    good_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="get_customer", arguments='{"customer_id": "C0001"}'),
    )
    provider._client.chat.completions.create = lambda **kwargs: _fake_completion(  # type: ignore[method-assign]
        content=None, tool_calls=[good_call]
    )

    response = provider.chat([ChatMessage(role="user", content="hi")])

    tc = response.message.tool_calls[0]
    assert tc.arguments == {"customer_id": "C0001"}
    assert tc.parse_error is None
