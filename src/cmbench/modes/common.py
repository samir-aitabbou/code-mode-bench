"""Shared constants between the classic and codemode agent loops, so the
final-answer protocol (CLAUDE.md §3.5) is identical in both modes."""

from __future__ import annotations

from typing import Any

FINAL_ANSWER_MARKER = "FINAL_ANSWER:"

REMINDER_TEXT = (
    "Reminder: when you have the final answer, reply with exactly "
    f"'{FINAL_ANSWER_MARKER} <json>' and nothing else -- no tool call, no extra text."
)


class TrialAborted(Exception):
    """Raised by a mode's trial loop when `provider.chat()` itself raises
    (a vLLM/HTTP-level failure: context overflow, a malformed tool call
    the parser rejects, a client-side request timeout, ...), instead of
    letting the exception erase every turn completed so far. Carries the
    partial `TrialOutcome` (transcript, usage, turns/tool-calls so far)
    so `cmbench.runner` can still save a trace and report real counters
    for the failed trial -- previously these were always zeroed out and
    the trace was always empty, which made failed trials unreviewable.
    Always raised as `raise TrialAborted(outcome) from exc`, so the
    original exception is available via `__cause__`.
    """

    def __init__(self, outcome: Any) -> None:
        super().__init__("trial aborted mid-way by a provider-level error")
        self.outcome = outcome
