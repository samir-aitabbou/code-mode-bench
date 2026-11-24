# Methodology

This document expands on `REPORT.md` §3 with the full detail of how each
measurement is taken. See `PLAN.md` (project root) for the original
experimental design and its rationale.

## Reference data

A deterministic SQLite database (`src/cmbench/data/seed.py`, seed 42) with
500 customers, 200 products, 5,000 orders, 8,000 invoices, 1,500 support
tickets, and an empty `outbox` table (for T4's "send an email" writes). The
generator verifies, at generation time, that the answers to T3/T4/T5 are
unambiguous (no ties among the top-3/top-5 results, at least 5 customers
qualify for T4) and nudges the data deterministically if a future reseed
ever produced a tie.

The database is reset to its template state before every trial (a file
copy), so trials never see another trial's writes.

## Tasks

See `src/cmbench/tasks/tasks.yaml` for the exact statements and expected
answer formats; `REPORT.md` §3.1 lists all six with their one-line summary.
Expected answers are computed once, directly in SQL against the reference
database (`src/cmbench/tasks/graders.py`), independent of either mode's
implementation -- the grader never runs the model's own tool calls to
"check its own work."

## Modes

### Classic MCP tool-calling

The model is given the 12 (or 60, tool-bloat variant) MCP tool schemas as
native OpenAI-style tool definitions. Each turn: the model emits a tool
call, the harness runs it against the real MCP server and returns the full
JSON result as a tool message, up to 15 turns. A tool error (unknown tool,
bad arguments) is fed back to the model as a normal tool result so it can
retry -- a run only becomes `tool_error` if the trial then runs out of
turns (15) with an error still present in the transcript; otherwise it is
classified by whatever actually ends the trial (success, `wrong_answer`,
`max_turns` with no error present, etc).

### Code Mode

The model receives exactly one tool, `execute_python(code: str)`, plus a
typed `tools.py` module in its system prompt. That module is generated
directly from the same MCP JSON schemas used by classic mode
(`src/cmbench/modes/typed_api.py`), so the two modes are guaranteed to
expose equivalent capabilities -- Code Mode is never secretly given an
easier or different set of tools.

Code runs in an isolated subprocess (`src/cmbench/sandbox/runner.py`):
120-second cap, no network (`unshare -rn`, falling back to
`docker run --network none` where user namespaces are unavailable, then to
no isolation with a logged warning as a last resort), stdout/stderr
truncated to 8,000 characters. `tools.*` calls inside that subprocess are
bridged back to the harness's real MCP tool implementations over a
line-based JSON-RPC protocol (`src/cmbench/sandbox/bridge.py`) -- the
sandboxed code never talks to the database or network directly.

## Tool-count (bloat) variant

T3 and T5 are additionally run with 48 extra, realistic-but-irrelevant
"noise" tools added to the MCP server (`src/cmbench/mcp_server/extra_tools.py`),
60 total instead of 12, to separate the cost of tool *definitions* in
context from the cost of the task itself (H6).

## What gets measured, per trial

- `prompt_tokens` / `completion_tokens` / `total_tokens`: read from each
  API response's `usage` field, summed across every request in the trial
  (classic mode resends the whole growing conversation each turn, so this
  is the total input-token cost of the task, not one request's size -- see
  `REPORT.md` §3.3 for why this matters for H1/H6).
- `num_requests`: count of LLM API calls in the trial.
- `num_tool_calls` / `num_code_executions`: count of MCP tool calls
  (classic) or sandboxed code executions (Code Mode).
- `latency_total_s`: wall-clock seconds for the whole trial.
- `gpu_seconds`: GPU time actually spent generating tokens (used for the
  cost estimate, `gpu_seconds / 3600 * gpu_hourly_usd`, default
  $0.50/hour -- always presented as an estimate, not real cloud pricing).
- `gpu_util_pct` / `gpu_mem_used_mib` / `gpu_mem_free_mib` /
  `gpu_compute_procs`: an `nvidia-smi` snapshot taken right after the
  trial finishes. This exists because `redacted-gpu-vm` is a shared VM;
  `gpu_compute_procs > 1` flags a trial that ran alongside another user's
  GPU job, so a reader can tell our own load apart from external
  contention when reading the raw results. `None` when `nvidia-smi` isn't
  available (e.g. the fake-provider CI tests on a GPU-less machine) --
  never fabricated.
- `valid`: `False` only if the API response was missing its `usage` field
  entirely (never estimated -- such a trial is excluded from every token
  aggregate, though its failure is still counted).
- `success` / `failure_type`: see "Grading and failure types" below.

## Grading and failure types

The model must end with exactly `FINAL_ANSWER: <json>` on its own; any
other final message is `format_error`. For T4 (writes), grading checks the
final database state (customer notes) and the `outbox` table, not just the
text reply -- a model could claim success in text without actually having
made the required tool calls.

Failure types, in the order a trial is checked against them:

1. `timeout` -- the trial exceeded its wall-clock budget
   (`limits.max_trial_seconds` in `configs/experiment.yaml`, 900s).
2. `context_overflow` -- the provider returned a context-length-exceeded
   error (classic mode only in practice; Code Mode keeps raw data out of
   the model's context).
3. `provider_error` -- any other API-level error (e.g. a malformed
   tool-call header the server itself could not parse).
4. `max_turns` -- 15 turns used up with no `FINAL_ANSWER` and no tool
   error anywhere in the transcript.
5. `tool_error` -- same as `max_turns`, but at least one tool call in the
   transcript returned an error; distinguishes "the model never got
   workable data" from "the model just kept thinking."
6. `format_error` -- a final message that isn't a parseable
   `FINAL_ANSWER: <json>`.
7. `wrong_answer` -- a well-formed `FINAL_ANSWER` that doesn't match the
   SQL-computed expected value (or, for T4, the expected database state).

## Statistics

- **Success rate**: Wilson score interval, 95% confidence
  (`src/cmbench/metrics.py::wilson_interval`) -- preferred over the normal
  approximation for small n (10 per cell, 5 for the tool-bloat variant)
  and rates near 0% or 100%, both common here.
- **Token/latency/GPU-time**: median per cell, robust to the occasional
  very slow or very token-heavy retry that a mean would be dominated by.
- **Paired comparisons** (e.g. Code Mode vs classic, same task): percent
  change of medians, reported per (model, task) rather than pooled across
  models, since H5 is specifically about whether the effect depends on
  the model.

## Reproducing the figures and report

`uv run cmbench report` regenerates every summary CSV under
`results/summary/`, every figure under `results/figures/`, and `REPORT.md`
itself, purely from `results/raw/*.jsonl` -- nothing in the report is
hand-edited or hand-computed. See `docs/REPRODUCE.md` for the full
environment setup and the commands to regenerate the raw trial data
itself.
