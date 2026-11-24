"""JSONL trial records -> CSV summary tables (results/summary/).

Every table here is derived purely from `results/raw/*.jsonl` -- no
numbers are computed by hand, so a re-run of `cmbench report` always
reflects the data on disk exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cmbench.metrics import median, wilson_interval

MAIN_TOOL_COUNT = 12


def load_trials(raw_dir: Path) -> pd.DataFrame:
    """Load every `*.jsonl` trial file in `raw_dir` into one DataFrame."""
    rows: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no trial records found under {raw_dir}")
    return pd.DataFrame(rows)


def by_cell(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, mode, task, tool_count): success rate (Wilson
    CI), median tokens/requests/latency/GPU time -- the main results
    table (CLAUDE.md §5.4)."""
    records = []
    group_cols = ["model_id", "mode", "task_id", "tool_count"]
    for key, group in df.groupby(group_cols, sort=True):
        model_id, mode, task_id, tool_count = key
        n = len(group)
        successes = int(group["success"].sum())
        ci = wilson_interval(successes, n)
        valid = group[group["valid"]]
        records.append(
            {
                "model_id": model_id,
                "mode": mode,
                "task_id": task_id,
                "tool_count": tool_count,
                "n": n,
                "successes": successes,
                "success_rate": ci.point,
                "success_rate_low": ci.low,
                "success_rate_high": ci.high,
                "median_prompt_tokens": median(list(valid["prompt_tokens"].dropna())),
                "median_completion_tokens": median(list(valid["completion_tokens"].dropna())),
                "median_num_requests": median(list(valid["num_requests"].dropna())),
                "median_latency_s": median(list(valid["latency_total_s"].dropna())),
                "median_gpu_seconds": median(list(valid["gpu_seconds"].dropna())),
                "total_gpu_seconds": float(group["gpu_seconds"].sum()),
                "total_tokens": int(group["total_tokens"].sum()),
            }
        )
    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)


def codemode_effect(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (model, task), % change of codemode vs classic on the main
    (12-tool) series: input tokens, output tokens, success rate. One of
    the report's required views (CLAUDE.md §5.4)."""
    main = cells[cells["tool_count"] == MAIN_TOOL_COUNT]
    records = []
    for (model_id, task_id), group in main.groupby(["model_id", "task_id"], sort=True):
        rows = {row["mode"]: row for _, row in group.iterrows()}
        if "classic" not in rows or "codemode" not in rows:
            continue
        classic, codemode = rows["classic"], rows["codemode"]

        def pct_change(
            field: str, classic: pd.Series = classic, codemode: pd.Series = codemode
        ) -> float | None:
            base = classic[field]
            if not base:
                return None
            return float(100.0 * (codemode[field] - base) / base)

        records.append(
            {
                "model_id": model_id,
                "task_id": task_id,
                "classic_success_rate": classic["success_rate"],
                "codemode_success_rate": codemode["success_rate"],
                "success_rate_diff_pp": 100.0
                * (codemode["success_rate"] - classic["success_rate"]),
                "prompt_tokens_pct_change": pct_change("median_prompt_tokens"),
                "completion_tokens_pct_change": pct_change("median_completion_tokens"),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "model_id",
                "task_id",
                "classic_success_rate",
                "codemode_success_rate",
                "success_rate_diff_pp",
                "prompt_tokens_pct_change",
                "completion_tokens_pct_change",
            ]
        )
    return pd.DataFrame(records).sort_values(["model_id", "task_id"]).reset_index(drop=True)


def tool_bloat_effect(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (model, mode, task in {T3, T5}), 12 vs 60 tools: how much the
    input-token median grows with more tool definitions in context
    (H6, CLAUDE.md §5.4)."""
    bloat_tasks = cells[cells["task_id"].isin(["T3", "T5"])]
    records = []
    for (model_id, mode, task_id), group in bloat_tasks.groupby(
        ["model_id", "mode", "task_id"], sort=True
    ):
        rows = {row["tool_count"]: row for _, row in group.iterrows()}
        if 12 not in rows or 60 not in rows:
            continue
        tools12, tools60 = rows[12], rows[60]
        base = tools12["median_prompt_tokens"]
        pct_change = 100.0 * (tools60["median_prompt_tokens"] - base) / base if base else None
        records.append(
            {
                "model_id": model_id,
                "mode": mode,
                "task_id": task_id,
                "median_prompt_tokens_12tools": tools12["median_prompt_tokens"],
                "median_prompt_tokens_60tools": tools60["median_prompt_tokens"],
                "prompt_tokens_pct_change": pct_change,
                "success_rate_12tools": tools12["success_rate"],
                "success_rate_60tools": tools60["success_rate"],
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "model_id",
                "mode",
                "task_id",
                "median_prompt_tokens_12tools",
                "median_prompt_tokens_60tools",
                "prompt_tokens_pct_change",
                "success_rate_12tools",
                "success_rate_60tools",
            ]
        )
    return pd.DataFrame(records).sort_values(["model_id", "mode", "task_id"]).reset_index(
        drop=True
    )


def failure_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Count of each failure_type, per (model, mode) -- feeds the
    failure-type pie/bar chart and the Failure analysis section."""
    failures = df[~df["success"]]
    counts = (
        failures.groupby(["model_id", "mode", "failure_type"], sort=True)
        .size()
        .reset_index(name="count")
    )
    return counts.sort_values(["model_id", "mode", "count"], ascending=[True, True, False])


def context_overflow_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """H7: context_overflow failures per (model, mode), classic vs
    codemode -- expected to be classic-only if H7 holds."""
    overflow = df[df["failure_type"] == "context_overflow"]
    counts = overflow.groupby(["model_id", "mode"], sort=True).size().reset_index(name="count")
    return counts


def write_all(raw_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Load the raw trials and write every summary CSV. Returns a dict of
    {name: path} for the caller (e.g. render.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_trials(raw_dir)
    cells = by_cell(df)

    outputs = {
        "trials": out_dir / "trials.csv",
        "by_cell": out_dir / "by_cell.csv",
        "codemode_effect": out_dir / "codemode_effect.csv",
        "tool_bloat_effect": out_dir / "tool_bloat_effect.csv",
        "failure_breakdown": out_dir / "failure_breakdown.csv",
        "context_overflow_by_model": out_dir / "context_overflow_by_model.csv",
    }
    df.to_csv(outputs["trials"], index=False)
    cells.to_csv(outputs["by_cell"], index=False)
    codemode_effect(cells).to_csv(outputs["codemode_effect"], index=False)
    tool_bloat_effect(cells).to_csv(outputs["tool_bloat_effect"], index=False)
    failure_breakdown(df).to_csv(outputs["failure_breakdown"], index=False)
    context_overflow_by_model(df).to_csv(outputs["context_overflow_by_model"], index=False)
    return outputs
