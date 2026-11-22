"""Loads task statements from tasks.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_TASKS_PATH = Path(__file__).with_name("tasks.yaml")


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    statement: str
    answer_format: str

    def build_prompt(self) -> str:
        """The full task prompt shown to the model: the statement plus the
        shared FINAL_ANSWER instruction."""
        return (
            f"{self.statement}\n\n"
            "When you have the final answer, reply with exactly "
            f"'FINAL_ANSWER: <json>' and nothing else, where <json> is "
            f"{self.answer_format}"
        )


def load_tasks(path: Path = DEFAULT_TASKS_PATH) -> dict[str, Task]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks = {}
    for raw in data["tasks"]:
        task = Task(
            id=raw["id"],
            title=raw["title"],
            statement=raw["statement"].strip(),
            answer_format=raw["answer_format"].strip(),
        )
        tasks[task.id] = task
    return tasks
