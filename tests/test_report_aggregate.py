import json
from pathlib import Path

from cmbench.report.aggregate import (
    by_cell,
    codemode_effect,
    context_overflow_by_model,
    failure_breakdown,
    load_trials,
    tool_bloat_effect,
    write_all,
)

MODEL = "fake-model"


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "trial_id": "x",
        "model_id": MODEL,
        "mode": "classic",
        "task_id": "T1",
        "tool_count": 12,
        "rep_index": 0,
        "valid": True,
        "success": True,
        "failure_type": None,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "num_requests": 1,
        "latency_total_s": 1.0,
        "gpu_seconds": 1.0,
        "cost_usd_estimate": 0.001,
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_trials_reads_every_jsonl_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_jsonl(raw_dir / f"{MODEL}.jsonl", [_record(trial_id="a"), _record(trial_id="b")])

    df = load_trials(raw_dir)

    assert len(df) == 2
    assert set(df["trial_id"]) == {"a", "b"}


def test_by_cell_computes_success_rate_and_medians(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        _record(trial_id="r0", success=True, prompt_tokens=100),
        _record(trial_id="r1", success=False, failure_type="wrong_answer", prompt_tokens=200),
    ]
    _write_jsonl(raw_dir / f"{MODEL}.jsonl", rows)
    df = load_trials(raw_dir)

    cells = by_cell(df)

    assert len(cells) == 1
    row = cells.iloc[0]
    assert row["n"] == 2
    assert row["successes"] == 1
    assert row["success_rate"] == 0.5
    assert row["median_prompt_tokens"] == 150.0


def test_codemode_effect_pairs_classic_and_codemode_for_main_series(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        _record(trial_id="c0", mode="classic", prompt_tokens=200, success=True),
        _record(trial_id="m0", mode="codemode", prompt_tokens=100, success=True),
        # 60-tool variant must be excluded from the main-series comparison.
        _record(trial_id="c1", mode="classic", tool_count=60, prompt_tokens=999, success=True),
        _record(trial_id="m1", mode="codemode", tool_count=60, prompt_tokens=999, success=True),
    ]
    _write_jsonl(raw_dir / f"{MODEL}.jsonl", rows)
    cells = by_cell(load_trials(raw_dir))

    effect = codemode_effect(cells)

    assert len(effect) == 1
    row = effect.iloc[0]
    assert row["prompt_tokens_pct_change"] == -50.0  # 100 vs 200 = -50%


def test_codemode_effect_flags_token_comparison_as_not_comparable_when_classic_fails(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        # classic always fails (success_rate 0) -- its median tokens come
        # from aborted trials, not completed ones (ANALYSE-CLAUDE.md
        # point C), so the pct-change against codemode must be flagged.
        _record(trial_id="c0", mode="classic", prompt_tokens=50, success=False,
                failure_type="context_overflow"),
        _record(trial_id="m0", mode="codemode", prompt_tokens=500, success=True),
    ]
    _write_jsonl(raw_dir / f"{MODEL}.jsonl", rows)
    cells = by_cell(load_trials(raw_dir))

    effect = codemode_effect(cells)

    assert len(effect) == 1
    row = effect.iloc[0]
    assert row["classic_success_rate"] == 0.0
    assert row["tokens_comparable"] is False or row["tokens_comparable"] == 0


def test_tool_bloat_effect_compares_12_vs_60_tools(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        _record(trial_id="a", task_id="T3", tool_count=12, prompt_tokens=100, success=True),
        _record(trial_id="b", task_id="T3", tool_count=60, prompt_tokens=150, success=True),
    ]
    _write_jsonl(raw_dir / f"{MODEL}.jsonl", rows)
    cells = by_cell(load_trials(raw_dir))

    bloat = tool_bloat_effect(cells)

    assert len(bloat) == 1
    row = bloat.iloc[0]
    assert row["median_prompt_tokens_12tools"] == 100.0
    assert row["median_prompt_tokens_60tools"] == 150.0
    assert row["prompt_tokens_pct_change"] == 50.0


def test_failure_breakdown_counts_only_failures(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        _record(trial_id="ok", success=True),
        _record(trial_id="f1", success=False, failure_type="timeout"),
        _record(trial_id="f2", success=False, failure_type="timeout"),
        _record(trial_id="f3", success=False, failure_type="tool_error"),
    ]
    _write_jsonl(raw_dir / f"{MODEL}.jsonl", rows)
    df = load_trials(raw_dir)

    counts = failure_breakdown(df)

    assert dict(zip(counts["failure_type"], counts["count"], strict=True)) == {
        "timeout": 2,
        "tool_error": 1,
    }


def test_context_overflow_by_model_counts_only_that_failure_type(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        _record(trial_id="o1", mode="classic", success=False, failure_type="context_overflow"),
        _record(trial_id="o2", mode="codemode", success=False, failure_type="wrong_answer"),
    ]
    _write_jsonl(raw_dir / f"{MODEL}.jsonl", rows)
    df = load_trials(raw_dir)

    counts = context_overflow_by_model(df)

    assert len(counts) == 1
    assert counts.iloc[0]["mode"] == "classic"
    assert counts.iloc[0]["count"] == 1


def test_write_all_creates_every_summary_csv(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "summary"
    rows = [_record(trial_id="a"), _record(trial_id="b", mode="codemode")]
    _write_jsonl(raw_dir / f"{MODEL}.jsonl", rows)

    outputs = write_all(raw_dir, out_dir)

    for path in outputs.values():
        assert path.exists()
