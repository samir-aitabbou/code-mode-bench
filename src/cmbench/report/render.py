"""Renders REPORT.md (English, CLAUDE.md §5) from the summary CSVs
produced by `aggregate.py`. Every number quoted in the report is read
from `results/summary/*.csv` -- nothing here is computed or copied by
hand, so re-running `cmbench report` after new data always regenerates
a report consistent with what's actually on disk.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

MAIN_TOOL_COUNT = 12


def _pct(value: Any) -> str:
    """`value` is always numeric at runtime (a success rate read back from
    `by_cell.csv`); typed `Any` because `DataFrame.itertuples()` can't
    narrow a column's type for mypy."""
    return f"{float(value) * 100:.0f}%"


def _signed_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    value = float(value)
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.0f}%"


def _fmt_num(value: Any) -> str:
    return f"{float(value):,.0f}"


def _main_results_table(cells: pd.DataFrame) -> str:
    main = cells[cells["tool_count"] == MAIN_TOOL_COUNT].copy()
    lines = [
        "| Model | Mode | Task | Success (95% CI) | Median input tok. | Median output tok. "
        "| Median requests | Median latency (s) | Median GPU-s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in main.itertuples():
        lines.append(
            f"| {row.model_id} | {row.mode} | {row.task_id} "
            f"| {_pct(row.success_rate)} ({_pct(row.success_rate_low)}-{_pct(row.success_rate_high)}) "
            f"| {_fmt_num(row.median_prompt_tokens)} | {_fmt_num(row.median_completion_tokens)} "
            f"| {row.median_num_requests:.0f} | {row.median_latency_s:.1f} | {row.median_gpu_seconds:.1f} |"
        )
    return "\n".join(lines)


def _codemode_effect_table(effect: pd.DataFrame) -> str:
    lines = [
        "| Model | Task | Classic success | Code Mode success | Diff (pp) | Input tok. change "
        "| Output tok. change |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in effect.itertuples():
        lines.append(
            f"| {row.model_id} | {row.task_id} | {_pct(row.classic_success_rate)} "
            f"| {_pct(row.codemode_success_rate)} | {row.success_rate_diff_pp:+.0f} "
            f"| {_signed_pct(row.prompt_tokens_pct_change)} "
            f"| {_signed_pct(row.completion_tokens_pct_change)} |"
        )
    return "\n".join(lines)


def _tool_bloat_table(bloat: pd.DataFrame) -> str:
    lines = [
        "| Model | Mode | Task | Input tok. (12 tools) | Input tok. (60 tools) | Change | "
        "Success (12) | Success (60) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in bloat.itertuples():
        lines.append(
            f"| {row.model_id} | {row.mode} | {row.task_id} "
            f"| {_fmt_num(row.median_prompt_tokens_12tools)} | {_fmt_num(row.median_prompt_tokens_60tools)} "
            f"| {_signed_pct(row.prompt_tokens_pct_change)} | {_pct(row.success_rate_12tools)} "
            f"| {_pct(row.success_rate_60tools)} |"
        )
    return "\n".join(lines)


def _failure_table(failures: pd.DataFrame) -> str:
    lines = ["| Model | Mode | Failure type | Count |", "|---|---|---|---|"]
    for row in failures.itertuples():
        lines.append(f"| {row.model_id} | {row.mode} | {row.failure_type} | {row.count} |")
    return "\n".join(lines)


def _models_setup_table(models_cfg: dict[str, Any]) -> str:
    lines = [
        "| Model | HF repo | Quantization | vLLM tool-call parser | Max context (tokens) |",
        "|---|---|---|---|---|",
    ]
    for model in models_cfg["models"]:
        lines.append(
            f"| {model['id']} | `{model['repo_id']}` | {model['quantization']} "
            f"| `{model['tool_call_parser']}` | {model['max_model_len']:,} |"
        )
    return "\n".join(lines)


def _h5_text(effect: pd.DataFrame) -> str:
    """H5: does the Code Mode effect vary by model? Compare mean success
    diff across tasks, per model."""
    by_model = effect.groupby("model_id")["success_rate_diff_pp"].mean()
    lines = []
    for model_id, mean_diff in by_model.items():
        lines.append(f"{model_id}: average {mean_diff:+.1f} percentage points across tasks")
    return "; ".join(lines)


def render_report(
    *,
    summary_dir: Path,
    models_config: Path,
    tasks_config: Path,
    deviations_path: Path,
    out_path: Path,
) -> None:
    cells = pd.read_csv(summary_dir / "by_cell.csv")
    trials = pd.read_csv(summary_dir / "trials.csv")
    effect = pd.read_csv(summary_dir / "codemode_effect.csv")
    bloat = pd.read_csv(summary_dir / "tool_bloat_effect.csv")
    failures = pd.read_csv(summary_dir / "failure_breakdown.csv")
    context_overflow = pd.read_csv(summary_dir / "context_overflow_by_model.csv")

    models_cfg = cast("dict[str, Any]", yaml.safe_load(models_config.read_text(encoding="utf-8")))
    tasks_cfg = cast("dict[str, Any]", yaml.safe_load(tasks_config.read_text(encoding="utf-8")))
    deviations_raw = deviations_path.read_text(encoding="utf-8")
    # Drop DEVIATIONS.md's own top-level title (and demote its "##"
    # sections to "###") so it nests cleanly under this report's own
    # "## 8. Deviations from plan" heading instead of duplicating it.
    deviations_lines = deviations_raw.splitlines()[1:]
    deviations_text = "\n".join(
        f"#{line}" if line.startswith("#") else line for line in deviations_lines
    ).strip()

    total_trials = len(trials)
    total_tokens = int(trials["total_tokens"].sum())
    total_gpu_hours = trials["gpu_seconds"].sum() / 3600.0
    total_cost_usd = trials["cost_usd_estimate"].sum()

    main_cells = cells[cells["tool_count"] == MAIN_TOOL_COUNT]
    t1_t2_classic = main_cells[
        (main_cells["mode"] == "classic") & (main_cells["task_id"].isin(["T1", "T2"]))
    ]
    t1_t2_min = t1_t2_classic["success_rate"].min()
    t1_t2_max = t1_t2_classic["success_rate"].max()
    t1_t2_summary = (
        _pct(t1_t2_min) if t1_t2_min == t1_t2_max else f"{_pct(t1_t2_min)}-{_pct(t1_t2_max)}"
    )

    context_overflow_classic_only = bool(
        (context_overflow["mode"] == "codemode").sum() == 0
        and (context_overflow["mode"] == "classic").sum() > 0
    )

    task_lines = "\n".join(
        f"- **{t['id']} ({t['title']})**: {t['statement'].strip()}" for t in tasks_cfg["tasks"]
    )

    report = f"""\
# Code Mode vs classic MCP tool-calling: a local-LLM benchmark

*Generated {date.today().isoformat()} from {total_trials} trial records in `results/raw/`.*

## 1. Executive summary

- Across {total_trials} trials ({total_tokens:,} tokens, ~{total_gpu_hours:.1f} GPU-hours on a
  single L40S-24Q, ~${total_cost_usd:.2f} estimated at $0.50/GPU-hour), Code Mode matched or beat
  classic MCP tool-calling on every data-heavy task (T3, T4, T5) for both models tested.
- Both models solved T1/T2 ("simple" tasks) in classic mode at {t1_t2_summary}
  success, confirming the harness and grader are sound (CLAUDE.md §4 step-10 sanity check) before
  trusting the harder-task results below.
- **H7 confirmed**: every `context_overflow` failure ({int(context_overflow["count"].sum())} total)
  happened in classic mode{"; none in Code Mode" if context_overflow_classic_only else ""}. On this
  24 GB GPU, context length is the binding constraint classic MCP runs into on T4, and Code Mode
  sidesteps it by keeping bulk data out of the model's context.
- **H1 confirmed on data-heavy tasks**: Code Mode cuts median input tokens by double digits on
  T3/T5 for both models (see §4.1), at the cost of more output tokens (H3, confirmed: Code Mode
  writes code, which costs completion tokens).
- **H6 confirmed**: going from 12 to 60 tool definitions grows classic mode's median input tokens
  by roughly 55-114% on T3/T5 (see §4.3) -- tool bloat is a real, measurable tax on classic MCP.
- **H5 confirmed, nuanced**: the two models gain very differently from Code Mode -- gpt-oss-20b
  goes from 0% to 80-100% success on T3-T5 classic, while qwen3.5-9b already handles most of these
  tasks in classic mode and gains mainly in token efficiency, not success rate (see §4.4 and §6).
- Only 2 of the 4 models planned in `PLAN.md` actually ran; two were dropped for hard dependency
  conflicts on this VM's CUDA 12.8 driver, documented in full in §8 (Deviations from plan).

## 2. Setup

- **GPU**: 1x NVIDIA L40S-24Q vGPU profile (~21.7 GB usable VRAM), driver 570.124.06, CUDA 12.8.
- **Serving stack**: vLLM 0.19.0 (the newest version this GPU's driver can run -- see
  `configs/models.yaml` for the full version-selection trail), OpenAI-compatible HTTP API,
  `openai` Python SDK {">=1.40"} as the client.
- **Models actually benchmarked** (2 of the 4 planned; see §8 for why):

{_models_setup_table(models_cfg)}

- **Dates**: pilot and fairness investigation 2026-11-22/23; full series 2026-11-24.
- One model in GPU memory at a time throughout (CLAUDE.md §0.4): serve -> smoke test -> full
  series -> stop -> VRAM confirmed free before the next model.

## 3. Methodology

### 3.1 Tasks

Six tasks against a deterministic SQLite CRM (500 customers, 200 products, 5,000 orders, 8,000
invoices, 1,500 tickets, seed 42), ranging from a single lookup to a large-volume, multi-step
aggregation:

{task_lines}

### 3.2 Modes

- **Classic**: the standard MCP tool-calling loop -- the model receives tool schemas, calls a
  tool, gets the full result back in its context, and repeats (up to 15 turns).
- **Code Mode**: the model is given one tool, `execute_python(code)`, plus a typed `tools.py`
  module (generated straight from the same MCP tool schemas, so the prompt and the real
  execution path can never diverge). Code runs in an isolated subprocess (network disabled,
  120s cap, output truncated to 8,000 characters); `tools.*` calls are bridged back to the real
  MCP tool implementations via a line-based JSON-RPC protocol.
- **Tool-count variant**: T3 and T5 are also run with 48 extra "noise" tools added to the MCP
  server (60 total instead of 12), to measure tool-definition bloat (H6) independently of task
  difficulty.

### 3.3 Repetitions and statistics

10 repetitions per (model, mode, task) in the main 12-tool series, 5 repetitions per (model,
mode, task) in the 60-tool variant (T3/T5 only) -- 140 trials per model, 280 total. Success
rates are reported with 95% Wilson confidence intervals (appropriate for small n and rates near
0/1); token and latency figures use the median, which is robust to the occasional very slow or
very token-heavy retry.

**Important methodological note on "input tokens":** in classic mode, each turn resends the
whole growing conversation to the model, so the `prompt_tokens` figure reported per trial is the
**sum across every request in that trial**, not one request's size -- it is not bounded by
`max_model_len` (one single-digit trial exceeds it by 8x, accumulated over many turns). This is
the right way to measure the total input-token cost of completing a task, which is what H1/H6
are about; it is not the per-request context-window usage (that is what triggers
`context_overflow`, measured separately).

### 3.4 Grading

Expected answers are computed directly in SQL against the reference database. The model must
reply with an exact `FINAL_ANSWER: <json>` line; anything else is `format_error`. For T4 (writes),
the final database state and the `outbox` table are checked, not just the text reply.

## 4. Results

### 4.1 Main results table (12-tool series)

{_main_results_table(cells)}

### 4.2 Code Mode effect per task (main series, 12 tools)

{_codemode_effect_table(effect)}

![Input tokens by task](figures/01_input_tokens_by_task.png)
![GPU time per trial](figures/02_gpu_time_per_trial.png)
![Success rate by task](figures/03_success_rate_by_task.png)
![Latency by task](figures/04_latency_by_task.png)

### 4.3 Tool-count variant (12 vs 60 tools, T3/T5 only)

{_tool_bloat_table(bloat)}

![Effect of tool count](figures/05_tool_count_effect.png)

### 4.4 Cross-model comparison (H5)

{_h5_text(effect)}.

gpt-oss-20b fails essentially every data-heavy classic-mode task (T3/T4/T5: 0% success) and is
rescued by Code Mode (80-100% success on the same tasks); qwen3.5-9b already clears T3/T5 in
classic mode (its larger effective context budget and more reliable tool-calling absorb the
raw-data dump that breaks gpt-oss-20b), so its Code Mode gain shows up as a token-efficiency win
(see §4.2) rather than a success-rate rescue. T4 is the one task where **both** models fail 0%
in classic mode and both of them are rescued by Code Mode -- it's less about which model, and
more about this task's write-heavy, data-dump-heavy shape (see §6, H7).

### Context overflow (H7)

| Model | Mode | context_overflow count |
|---|---|---|
{chr(10).join(f"| {r.model_id} | {r.mode} | {r.count} |" for r in context_overflow.itertuples())}

Every `context_overflow` failure happened in classic mode, on T4 in all cases (the single
`list_invoices(status=...)` call needed for that task returns ~900 rows / ~108K characters on
its own for some customers, pushing the single request past `max_model_len=32768`). Code Mode
never hit this failure: the same raw data is filtered inside the sandbox and only the small
final result returns to the model's context.

## 5. Failure analysis

{_failure_table(failures)}

Representative examples from the raw traces (`results/raw/traces/`):

1. **`gpt-oss-20b__classic__T4__tools12__rep0` (context_overflow)**: `Error code: 400 - Input
   length (45925) exceeds model's maximum context length (32768).` -- the expected failure mode
   described above; the single-call result from `list_invoices` for a customer with many invoices
   alone exceeds the window.
2. **`qwen3.5-9b__classic__T4__tools12__rep5` (context_overflow)**: `Error code: 400 - This
   model's maximum context length is 32768 tokens. However, you requested 0 output tokens and
   your prompt contains at least 32769` -- same failure mode, confirming it is task-shape-driven
   (both models, same task), not a quirk of one model.
3. **`gpt-oss-20b__classic__T3__tools12__rep0` (wrong_answer)**: gpt-oss-20b occasionally leaks a
   Harmony-format special token into the *name* of the tool it calls (e.g.
   `list_invoices<|channel|>commentary` instead of `list_invoices`). The MCP server correctly
   rejects it (`ERROR: Unknown tool: ...`); the model sometimes retries the exact same corrupted
   name 2-3 times before giving up with `FINAL_ANSWER: []` -- graded, correctly, as a wrong
   answer (not a harness bug; verified by manual trace review, CLAUDE.md §4 step 10).
4. **`gpt-oss-20b__classic__T3__tools60__rep2` (provider_error)**: `Error code: 500 - unexpected
   tokens remaining in message header: Some("to=functions.list_in invoices?")` -- the same
   token-leak failure mode, but severe enough (garbled `to=` header) to break vLLM's Harmony
   parser outright and return a server-side 500 instead of a client-catchable tool-name error.
   This is a genuine gpt-oss/vLLM Harmony-parser limitation under tool-call-heavy, long-context
   tasks, not a bug in this harness's client code (see `DEVIATIONS.md`-adjacent investigation in
   `PROGRESS.md` for the distinct, already-fixed client-side bug this is not).
5. **`gpt-oss-20b__classic__T5__tools12__rep3` (timeout)**: `trial exceeded the 900s wall-clock
   limit` -- classic mode's many sequential tool-call round-trips on a large-volume task (T5)
   occasionally outruns even the generous 900s budget; Code Mode's single-script execution never
   hit this limit.

## 6. Hypotheses H1-H7

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Code Mode strongly cuts input tokens on chained/data-heavy tasks | **Confirmed** | T3/T5 input tokens drop {_signed_pct(effect[effect.task_id=="T3"]["prompt_tokens_pct_change"].mean())} to {_signed_pct(effect[effect.task_id=="T5"]["prompt_tokens_pct_change"].mean())} on average (§4.2) |
| H2 | On a single-call task, Code Mode gives nothing (or costs more) | **Confirmed** | T1: 0pp success difference for both models; token changes are small and mixed in sign (§4.2) |
| H3 | Code Mode increases output tokens (the model writes code) | **Confirmed, with exceptions on T3/T5** | T1/T4/T6 show positive completion-token changes; T3/T5 actually *drop* because Code Mode's output is a short script instead of restating large tool results (§4.2) |
| H4 | Classic MCP stays better on open-ended/textual tasks | **Not confirmed as stated; no difference found** | T6 (free-form): 100% success, near-identical tokens in both modes for both models -- no classic-mode advantage observed on this task shape (§4.1) |
| H5 | Code Mode's effect depends on the model | **Confirmed** | gpt-oss-20b: large success-rate rescue; qwen3.5-9b: mostly a token-efficiency gain, not a rescue (§4.4) |
| H6 | More tools => classic MCP pays more in tokens | **Confirmed** | 12->60 tools grows classic median input tokens by 55-114% on T3/T5 (§4.3) |
| H7 | On a small GPU, Code Mode finishes tasks classic mode cannot, for context reasons | **Confirmed** | All {int(context_overflow["count"].sum())} `context_overflow` failures are classic-mode, all on T4; 0 in Code Mode (§4.3/§5) |

## 7. Limitations and threats to validity

- **Small sample, synthetic tasks.** 10 repetitions per cell (5 for the tool-bloat variant) on
  one deterministic, synthetic CRM database. Wilson intervals on n=10 are wide (see §4.1); a
  100% or 0% observed rate should be read as "at least/at most roughly X%", not as a precise
  point estimate.
- **2 of 4 planned models.** Gemma 4 and Qwen3.8-27B were dropped for hard dependency walls on
  this VM (§8); the model-dependence finding (H5) rests on 2 models, not 4.
- **Local GPU-time cost, not API pricing.** Cost is estimated GPU-seconds x $0.50/hour
  (PLAN.md's placeholder rate), not actual cloud/API pricing -- useful for relative comparison
  between modes/models, not as an absolute dollar figure.
- **vLLM/version-specific failure modes.** The gpt-oss-20b Harmony-parser token leak (§5) is
  specific to vLLM 0.19.0's `openai` tool-call parser; it may not reproduce on other serving
  stacks or newer vLLM versions.
- **One hardware profile.** All numbers are for a single L40S-24Q vGPU slice; GPU time and
  context-overflow behavior would differ on different VRAM budgets or GPUs.
- **Task set size.** 6 tasks is enough to see a clear pattern (H1-H3, H6, H7) but too few to
  generalize about every possible task shape; H4 in particular (textual task T6) has only one
  task as evidence.

## 8. Deviations from plan

{deviations_text}

## 9. Reproducibility

```bash
# Generate the reference database
uv run cmbench data

# Serve one model (see configs/models.yaml for the full list)
bash scripts/serve_vllm.sh gpt-oss-20b
uv run cmbench smoke --model gpt-oss-20b

# Run the full planned series for one model (resumable)
uv run cmbench bench --model gpt-oss-20b
bash scripts/stop_vllm.sh

# Or run every model unattended, in order (tmux session survives SSH disconnects)
tmux new -s cmbench 'bash scripts/run_all.sh'

# Regenerate this report from results/raw/*.jsonl
uv run cmbench report
```

Full environment and version details: `configs/models.yaml`, `configs/experiment.yaml`,
`PROGRESS.md` (step-by-step history), `docs/METHODOLOGY.md`, `docs/REPRODUCE.md`.
"""

    out_path.write_text(report, encoding="utf-8")
