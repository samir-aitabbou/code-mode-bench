import json
import sqlite3
from pathlib import Path

from cmbench.providers.base import UsageMissingError
from cmbench.runner import (
    TrialSpec,
    build_trial_specs,
    load_completed_trial_ids,
    run_trial,
)
from cmbench.tasks.loader import load_tasks

from .fakes import StubProvider, assistant_text, assistant_tool_call

BASE_KWARGS = {
    "tasks": ["T1", "T2", "T3", "T4", "T5", "T6"],
    "modes": ["classic", "codemode"],
    "main_repetitions": 10,
    "tool_bloat_tasks": ["T3", "T5"],
    "tool_bloat_repetitions": 5,
    "seed": 42,
}


def test_build_trial_specs_total_count_matches_plan() -> None:
    specs = build_trial_specs("gpt-oss-20b", **BASE_KWARGS)
    assert len(specs) == 140  # 6*2*10 + 2*2*5, per PLAN.md §6
    assert len({s.trial_id for s in specs}) == 140  # all ids unique


def test_build_trial_specs_shuffled_deterministically() -> None:
    a = build_trial_specs("gpt-oss-20b", **BASE_KWARGS)
    b = build_trial_specs("gpt-oss-20b", **BASE_KWARGS)
    assert [s.trial_id for s in a] == [s.trial_id for s in b]


def test_build_trial_specs_order_differs_per_model() -> None:
    a = build_trial_specs("gpt-oss-20b", **BASE_KWARGS)
    b = build_trial_specs("qwen3.5-9b", **BASE_KWARGS)
    ids_a = [s.trial_id.replace("gpt-oss-20b", "X") for s in a]
    ids_b = [s.trial_id.replace("qwen3.5-9b", "X") for s in b]
    assert ids_a != ids_b  # different shuffle order


def test_load_completed_trial_ids(tmp_path: Path) -> None:
    jsonl = tmp_path / "model.jsonl"
    jsonl.write_text(
        json.dumps({"trial_id": "a"}) + "\n" + json.dumps({"trial_id": "b"}) + "\n",
        encoding="utf-8",
    )
    assert load_completed_trial_ids(jsonl) == {"a", "b"}


def test_load_completed_trial_ids_missing_file(tmp_path: Path) -> None:
    assert load_completed_trial_ids(tmp_path / "missing.jsonl") == set()


def test_run_trial_classic_success(tmp_path: Path, seeded_db: Path) -> None:
    conn = sqlite3.connect(seeded_db)
    expected = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE customer_id='C0042' AND status='open'"
    ).fetchone()[0]

    provider = StubProvider(
        responses=[
            assistant_tool_call("get_customer", {"customer_id": "C0042"}),
            assistant_text(f"FINAL_ANSWER: {expected}"),
        ]
    )
    spec = TrialSpec("gpt-oss-20b", "classic", "T1", 12, 0)
    tasks = load_tasks()

    record, transcript = run_trial(spec, provider, tasks, seeded_db, tmp_path)

    assert record.valid
    assert record.success
    assert record.failure_type is None
    assert record.num_requests == 2
    assert record.num_tool_calls == 1
    assert record.trial_id == spec.trial_id
    assert transcript  # non-empty
    assert (tmp_path / f"{spec.trial_id}.db").exists()


def test_run_trial_codemode_success(tmp_path: Path, seeded_db: Path) -> None:
    code = (
        "import tools\n"
        "c = tools.get_customer(customer_id='C0042')\n"
        "n = tools.search_tickets(customer_id='C0042', status='open', limit=1000)\n"
        "print(len(n))\n"
    )
    conn = sqlite3.connect(seeded_db)
    expected = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE customer_id='C0042' AND status='open'"
    ).fetchone()[0]

    provider = StubProvider(
        responses=[
            assistant_tool_call("execute_python", {"code": code}),
            assistant_text(f"FINAL_ANSWER: {expected}"),
        ]
    )
    spec = TrialSpec("gpt-oss-20b", "codemode", "T1", 12, 0)
    tasks = load_tasks()

    record, _transcript = run_trial(spec, provider, tasks, seeded_db, tmp_path)

    assert record.valid
    assert record.success
    assert record.num_code_executions == 1


def test_run_trial_wrong_answer(tmp_path: Path, seeded_db: Path) -> None:
    provider = StubProvider(responses=[assistant_text("FINAL_ANSWER: 999999")])
    spec = TrialSpec("gpt-oss-20b", "classic", "T1", 12, 0)
    tasks = load_tasks()

    record, _ = run_trial(spec, provider, tasks, seeded_db, tmp_path)

    assert record.valid
    assert not record.success
    assert record.failure_type == "wrong_answer"


def test_run_trial_format_error(tmp_path: Path, seeded_db: Path) -> None:
    provider = StubProvider(responses=[assistant_text("FINAL_ANSWER: not valid json!!")])
    spec = TrialSpec("gpt-oss-20b", "classic", "T1", 12, 0)
    tasks = load_tasks()

    record, _ = run_trial(spec, provider, tasks, seeded_db, tmp_path)

    assert record.valid
    assert not record.success
    assert record.failure_type == "format_error"


def test_run_trial_max_turns_without_tool_error(tmp_path: Path, seeded_db: Path) -> None:
    provider = StubProvider(responses=[assistant_text("still thinking...") for _ in range(15)])
    spec = TrialSpec("gpt-oss-20b", "classic", "T1", 12, 0)
    tasks = load_tasks()

    record, _ = run_trial(spec, provider, tasks, seeded_db, tmp_path, max_turns=15)

    assert record.valid
    assert not record.success
    assert record.failure_type == "max_turns"


def test_run_trial_max_turns_with_tool_error_is_classified_as_tool_error(
    tmp_path: Path, seeded_db: Path
) -> None:
    provider = StubProvider(
        responses=[
            assistant_tool_call("get_customer", {"customer_id": "C9999"}) for _ in range(3)
        ]
    )
    spec = TrialSpec("gpt-oss-20b", "classic", "T1", 12, 0)
    tasks = load_tasks()

    record, _ = run_trial(spec, provider, tasks, seeded_db, tmp_path, max_turns=3)

    assert record.valid
    assert not record.success
    assert record.failure_type == "tool_error"


def test_run_trial_usage_missing_marks_invalid(tmp_path: Path, seeded_db: Path) -> None:
    provider = StubProvider(responses=[UsageMissingError("no usage field")])
    spec = TrialSpec("gpt-oss-20b", "classic", "T1", 12, 0)
    tasks = load_tasks()

    record, transcript = run_trial(spec, provider, tasks, seeded_db, tmp_path)

    assert not record.valid
    assert not record.success
    # The partial transcript (system + user messages, no turns completed
    # yet) is preserved rather than discarded -- see modes.common.TrialAborted.
    assert [m.role for m in transcript] == ["system", "user"]


def test_run_trial_context_overflow_classified(tmp_path: Path, seeded_db: Path) -> None:
    provider = StubProvider(
        responses=[RuntimeError("This model's maximum context length is 8192 tokens")]
    )
    spec = TrialSpec("gpt-oss-20b", "classic", "T1", 12, 0)
    tasks = load_tasks()

    record, _ = run_trial(spec, provider, tasks, seeded_db, tmp_path)

    assert record.valid
    assert not record.success
    assert record.failure_type == "context_overflow"
