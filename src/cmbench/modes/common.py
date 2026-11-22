"""Shared constants between the classic and codemode agent loops, so the
final-answer protocol (CLAUDE.md §3.5) is identical in both modes."""

from __future__ import annotations

FINAL_ANSWER_MARKER = "FINAL_ANSWER:"

REMINDER_TEXT = (
    "Reminder: when you have the final answer, reply with exactly "
    f"'{FINAL_ANSWER_MARKER} <json>' and nothing else -- no tool call, no extra text."
)
