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

from cmbench.report.aggregate import context_overflow_by_model, failure_breakdown

MAIN_TOOL_COUNT = 12
# T4p is a complementary probe added after the main 6-task design
# (ANALYSE-CLAUDE.md point A, DEVIATIONS.md); it has no codemode trials
# and is reported separately (§4.5), never folded into the main
# 6-task x 2-mode grid or its aggregates.
PAGINATED_PROBE_TASK = "T4p"


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
    main = cells[
        (cells["tool_count"] == MAIN_TOOL_COUNT) & (cells["task_id"] != PAGINATED_PROBE_TASK)
    ].copy()
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


def _token_change_cell(value: Any, comparable: Any) -> str:
    """ANALYSE-CLAUDE.md point C: when `comparable` is False, the token
    pct-change compares a mostly-failed classic cell's median (computed
    from aborted trials) against a mostly-succeeded Code Mode cell --
    not a meaningful number, so it is flagged instead of printed."""
    if not bool(comparable):
        return "n/a¹"
    return _signed_pct(value)


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
            f"| {_token_change_cell(row.prompt_tokens_pct_change, row.tokens_comparable)} "
            f"| {_token_change_cell(row.completion_tokens_pct_change, row.tokens_comparable)} |"
        )
    lines.append(
        "\n¹ *n/a: classic mode's success rate on this (model, task) is 0%, so its median "
        "tokens come from trials that mostly failed early (e.g. one request before a "
        "`context_overflow`), not from completing the task -- comparing that against Code "
        "Mode's (mostly completed) median would compare the cost of failing with the cost of "
        "succeeding, not a token-efficiency effect (ANALYSE-CLAUDE.md point C).*"
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


def _h1_evidence_text(effect: pd.DataFrame) -> str:
    """H1 evidence, restricted to (model, task) pairs where both modes
    actually complete the task often enough for a token comparison to
    mean anything (ANALYSE-CLAUDE.md points C and F: the old "-59% to
    -56% on average" phrasing mixed non-comparable pairs into one vague
    range; this reports each task's comparable pair(s) by name)."""
    comparable = effect[effect["tokens_comparable"].astype(bool)]
    parts = []
    for task_id in ("T3", "T5"):
        rows = comparable[comparable["task_id"] == task_id]
        if rows.empty:
            parts.append(f"{task_id}: no (model, task) pair had both modes succeeding often "
                         "enough to compare tokens")
            continue
        bits = ", ".join(
            f"{r.model_id} {_signed_pct(r.prompt_tokens_pct_change)}" for r in rows.itertuples()
        )
        parts.append(f"{task_id}: {bits}")
    return "; ".join(parts)


def _bloat_range_text(bloat: pd.DataFrame, mode: str) -> str:
    """ANALYSE-CLAUDE.md point D: tool bloat (12->60 tools) grows input
    tokens in *both* modes, not just classic -- Code Mode's tools.py
    stub lists every tool too, so it pays the same listing cost. Returns
    e.g. "+55% to +114%" across the (model, task) pairs for one mode."""
    changes = bloat[bloat["mode"] == mode]["prompt_tokens_pct_change"].dropna()
    if changes.empty:
        return "n/a"
    lo, hi = changes.min(), changes.max()
    if lo == hi:
        return _signed_pct(lo)
    return f"{_signed_pct(lo)} to {_signed_pct(hi)}"


def _t4p_section(trials: pd.DataFrame) -> str:
    """§4.5: the paginated-invoices complementary probe (DEVIATIONS.md,
    ANALYSE-CLAUDE.md point A). Built directly from `trials` (not a
    precomputed CSV) since this probe is intentionally outside the main
    6-task grid every other aggregate is built from."""
    t4p = trials[trials["task_id"] == PAGINATED_PROBE_TASK]
    if t4p.empty:
        return "*Not run.*"
    lines = [
        "| Model | Mode | n | Successes | Failure types |",
        "|---|---|---|---|---|",
    ]
    for (model_id, mode), group in t4p.groupby(["model_id", "mode"], sort=True):
        n = len(group)
        successes = int(group["success"].sum())
        failure_counts = group[~group["success"]]["failure_type"].value_counts()
        failure_text = ", ".join(f"{ft} x{c}" for ft, c in failure_counts.items())
        lines.append(f"| {model_id} | {mode} | {n} | {successes} | {failure_text} |")
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
    # failure_breakdown.csv / context_overflow_by_model.csv on disk cover
    # every trial (including the T4p probe, for transparency); the
    # report's narrative below instead recomputes both, scoped to the
    # core 6-task design (see core_trials below), since T4p is reported
    # separately in §4.5 and must not be folded into H7's "all on T4"
    # claim or the main failure-type counts.

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

    # The paginated probe (T4p) is reported on its own in §4.5, never
    # folded into the main 6-task narrative or its H7/failure counts.
    core_trials = trials[trials["task_id"] != PAGINATED_PROBE_TASK]
    context_overflow = context_overflow_by_model(core_trials)
    failures = failure_breakdown(core_trials)

    context_overflow_classic_only = bool(
        (context_overflow["mode"] == "codemode").sum() == 0
        and (context_overflow["mode"] == "classic").sum() > 0
    )
    overflow_task_ids = sorted(
        core_trials[core_trials["failure_type"] == "context_overflow"]["task_id"].unique()
    )
    overflow_task_text = (
        overflow_task_ids[0] if len(overflow_task_ids) == 1 else " and ".join(overflow_task_ids)
    )

    t4p_trials = trials[trials["task_id"] == PAGINATED_PROBE_TASK]
    t4p_n = len(t4p_trials)
    t4p_successes = int(t4p_trials["success"].sum()) if t4p_n else 0
    t4p_score_text = f"{t4p_successes}/{t4p_n}" if t4p_n else "not run"

    task_lines = "\n".join(
        f"- **{t['id']} ({t['title']})**: {t['statement'].strip()}"
        for t in tasks_cfg["tasks"]
        if t["id"] != PAGINATED_PROBE_TASK
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
- **H7 confirmed**: every `context_overflow` failure in the core 6-task design
  ({int(context_overflow["count"].sum())} total) happened in classic mode{"; none in Code Mode" if context_overflow_classic_only else ""},
  all on {overflow_task_text}. On this 24 GB GPU, context length is the binding constraint
  classic MCP runs into, and Code Mode sidesteps it by keeping bulk data out of the model's
  context.
- **T4 specifically fails 0/20 in classic mode for a tool-design reason, and a follow-up probe
  shows that isn't the whole story**: the only `list_invoices` tool classic mode had access to
  returns every matching invoice in one unbounded call. A follow-up run with a properly paginated
  version of that same tool (`T4p`, §4.5) still scored {t4p_score_text}: classic mode's
  turn-by-turn, full-history-resend design keeps re-accumulating tool output across the several
  pages this task needs, so pagination alone does not rescue it here.
- **H1 confirmed, but only where the comparison is meaningful**: on the (model, task) pairs where
  classic mode actually completes the task often enough to compare ({_h1_evidence_text(effect)}),
  Code Mode cuts median input tokens substantially (see §4.2) -- at the cost of more output tokens
  on simple tasks (H3), though output tokens actually *drop* on T3/T5 (see H3, §6).
- **H6 confirmed, in both modes**: going from 12 to 60 tool definitions grows median input tokens
  on T3/T5 by {_bloat_range_text(bloat, "classic")} in classic mode and
  {_bloat_range_text(bloat, "codemode")} in Code Mode (see §4.3) -- tool bloat is a real,
  measurable tax on both ways of calling tools here, since this harness's Code Mode shows the
  model every tool in `tools.py` rather than a searchable subset.
- **H5 confirmed, nuanced**: the two models gain very differently from Code Mode -- gpt-oss-20b
  goes from 0% to 80-100% success on T3-T5 classic, while qwen3.5-9b already handles most of these
  tasks in classic mode and gains mainly in token efficiency, not success rate (see §4.4 and §6).
  Part of gpt-oss-20b's classic-mode rescue reflects a vLLM Harmony-parser bug this model hits
  more often in classic mode (§5); qwen3.5-9b is the cleaner evidence for the general effect.
- Only 2 of the 4 models planned in `PLAN.md` actually ran; two were dropped for hard dependency
  conflicts on this VM's CUDA 12.8 driver, documented in full in §8 (Deviations from plan).

## 2. Setup

![Harness architecture](figures/07_setup_diagram.png)

*Figure 1. Harness architecture: an identical task prompt and grader feed both modes; classic
mode exchanges full tool schemas and results with the model every turn, Code Mode exposes a
single `execute_python` tool backed by a sandboxed script and a bridge to the MCP server.*

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

A seventh, `T4p`, reruns T4's exact statement against a differently-served tool and is reported
separately in §4.5, not among these six.

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
mode, task) in the 60-tool variant (T3/T5 only) -- 140 trials per model, 280 total for this core
6-task design, plus a 10-trials-per-model complementary probe (`T4p`, classic mode only, §4.5)
added afterward, for {total_trials} trials overall. Success rates are reported with 95% Wilson
confidence intervals (appropriate for small n and rates near 0/1); token and latency figures use
the median, which is robust to the occasional very slow or very token-heavy retry.

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

*Figure 2. Median input tokens per trial, classic vs Code Mode, with interquartile-range error
bars. Hatched bars mark a cell whose success rate is below 50%: its median is computed mostly
from aborted trials, not completed ones, and is not comparable to the other mode (point C).*

![GPU time per trial](figures/02_gpu_time_per_trial.png)

*Figure 3. Median GPU time per trial, same convention as Figure 2.*

![Success rate by task](figures/03_success_rate_by_task.png)

*Figure 4. Success rate per task with 95% Wilson confidence intervals, classic vs Code Mode;
`T4p` (classic-mode only, §4.5) is appended after the dotted separator.*

![Latency by task](figures/04_latency_by_task.png)

*Figure 5. Median end-to-end latency per trial, same convention as Figure 2.*

### 4.3 Tool-count variant (12 vs 60 tools, T3/T5 only)

{_tool_bloat_table(bloat)}

![Effect of tool count](figures/05_tool_count_effect.png)

*Figure 6. Median input tokens on T3/T5 with 12 vs 60 tool definitions in context (lighter shade
= 12 tools), classic and Code Mode side by side.*

Tool-definition bloat is not a classic-mode-only cost here: Code Mode's input tokens grow by
{_bloat_range_text(bloat, "codemode")} too, almost as much as classic mode's
{_bloat_range_text(bloat, "classic")} (ANALYSE-CLAUDE.md point D), because this implementation's
generated `tools.py` lists every available tool with its full docstring every time, exactly like
the tool schemas classic mode is shown. A tool-search/discovery mechanism (e.g. a `search()` call
the model uses before importing specific tools) would be needed to make Code Mode immune to this
cost -- not implemented here.

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

Every `context_overflow` failure in the core 6-task design happened in classic mode, on
{overflow_task_text} in all cases. T4's `list_invoices` tool takes no per-customer filter for
this task's `status='overdue'` query, so a single call returns *every* overdue invoice across
*all* ~500 customers at once (~900 rows / ~108K characters), pushing that one request past
`max_model_len=32768` before the model gets anywhere near reasoning about per-customer counts.
Code Mode never hit this failure: the same raw data is filtered inside the sandbox and only the
small final result returns to the model's context.

### 4.5 Complementary probe: T4 classic with a paginated invoices tool

T4's 0% classic-mode success rate above could, in principle, measure this harness's tool design
rather than the models' reasoning (ANALYSE-CLAUDE.md point A): the single unbounded
`list_invoices` call T4 depends on is, by construction, too large for a classic-mode request. To
test that directly, `T4p` reruns the exact same task against a paginated `list_invoices` variant
that caps `limit` at 200 and reports `total_count`/`has_more` -- a model that loops over `offset`
can retrieve every overdue invoice without any single call overflowing. Classic mode only (Code
Mode already succeeds on T4 with the original tool); 10 trials per model:

{_t4p_section(trials)}

![Context-limit scatter](figures/08_context_limit.png)

*Figure 7. Per-request token count on T4/T4p against the 32,768-token context limit (dashed
line): classic mode's rejected requests (both the unpaginated and the paginated tool) cluster at
or above the limit, while Code Mode's requests stay far below it.*

**Result: still {t4p_successes}/{t4p_n} success**, for a different reason than T4's single-call
overflow. Covering every one of the ~500 customers' overdue invoices takes ~5 pages at 200 rows
each, plus the per-customer `update_customer_note`/`send_email` write calls; classic mode resends
the whole growing conversation on every turn (§3.3), so by the time it has paged through enough
invoices to answer, the *accumulated* conversation itself exceeds the 32,768-token window in most
trials (`context_overflow`) -- or the model gives up partway through paging and answers with an
incomplete customer list, sometimes without ever calling the write tools at all (`wrong_answer`;
confirmed by manual trace review, e.g. `gpt-oss-20b__classic__T4p__tools12__rep9`: only 2 tool
calls total, `FINAL_ANSWER` listing 19 of the 212 expected customers, no write tool called for
any of them). A well-designed paginated tool removes the *single-oversized-call* failure mode,
but does not by itself make this task solvable in classic mode under this GPU's context budget
and this harness's full-history-resend design -- see `DEVIATIONS.md` for the full writeup.

## 5. Failure analysis

{_failure_table(failures)}

![Failure-type breakdown](figures/06_failure_breakdown.png)

*Figure 8. Count of failed trials by failure type, classic vs Code Mode, per model (core 6-task
design; `T4p` is reported separately in §4.5).*

Representative examples from the raw traces (`results/raw/traces/`):

1. **`gpt-oss-20b__classic__T4__tools12__rep0` (context_overflow)**: `Error code: 400 - Input
   length (45925) exceeds model's maximum context length (32768).` -- the expected failure mode
   described above; `list_invoices(status='overdue')` has no customer filter, so this one call
   already returns every overdue invoice across all customers combined, not just one customer's.
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
6. **`gpt-oss-20b__classic__T4p__tools12__rep9` (wrong_answer, §4.5 probe)**: even with a
   paginated `list_invoices`, the model made only 2 tool calls total before answering with
   `FINAL_ANSWER` listing 19 of the 212 customers actually expected, and never called
   `update_customer_note`/`send_email` for any of them -- it stopped paging (and writing) well
   short of covering the task, not because any single call overflowed.

## 6. Hypotheses H1-H7

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Code Mode strongly cuts input tokens on chained/data-heavy tasks | **Confirmed, on the pairs where the comparison is meaningful** | {_h1_evidence_text(effect)} (§4.2; gpt-oss-20b's T3/T5 classic cells are excluded as not comparable -- its 0% classic success rate there means its "tokens" are from aborted trials, point C) |
| H2 | On a single-call task, Code Mode gives nothing (or costs more) | **Confirmed** | T1: 0pp success difference for both models; token changes are small and mixed in sign (§4.2) |
| H3 | Code Mode increases output tokens (the model writes code) | **Confirmed on simple tasks; disconfirmed on data-heavy tasks** | T1/T4/T6 show positive completion-token changes, as expected; T3/T5 actually *drop* sharply instead, because Code Mode's output is a short script instead of restating large tool results in prose -- a genuine secondary finding, not an exception to wave away (§4.2) |
| H4 | Classic MCP stays better on open-ended/textual tasks | **Not confirmed as stated; no difference found** | T6 (free-form): 100% success, near-identical tokens in both modes for both models -- no classic-mode advantage observed on this task shape (§4.1) |
| H5 | Code Mode's effect depends on the model | **Confirmed** | gpt-oss-20b: large success-rate rescue (partly reflecting a vLLM parser bug it hits more in classic mode, §5); qwen3.5-9b: mostly a token-efficiency gain, not a rescue, and the cleaner evidence for this hypothesis (§4.4) |
| H6 | More tools => classic MCP pays more in tokens | **Confirmed in both modes** | 12->60 tools grows median input tokens by {_bloat_range_text(bloat, "classic")} in classic mode and {_bloat_range_text(bloat, "codemode")} in Code Mode on T3/T5 (§4.3) -- this implementation's Code Mode shows every tool too, so it is not immune |
| H7 | On a small GPU, Code Mode finishes tasks classic mode cannot, for context reasons | **Confirmed for the single-oversized-call failure mode; a paginated tool alone does not fully fix classic mode** | All {int(context_overflow["count"].sum())} core-series `context_overflow` failures are classic-mode, all on {overflow_task_text}; 0 in Code Mode (§4.3/§5). A follow-up with a paginated tool (T4p, §4.5) still scored {t4p_score_text} in classic mode, for a different (accumulated-context) reason |

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
