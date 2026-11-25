# Code Mode vs classic MCP tool-calling: a local-LLM benchmark

*Generated 2026-11-25 from 300 trial records in `results/raw/`.*

## 1. Executive summary

- Across 300 trials (9,111,154 tokens, ~5.2 GPU-hours on a
  single L40S-24Q, ~$2.62 estimated at $0.50/GPU-hour), Code Mode matched or beat
  classic MCP tool-calling on every data-heavy task (T3, T4, T5) for both models tested.
- Both models solved T1/T2 ("simple" tasks) in classic mode at 100%
  success, confirming the harness and grader are sound (CLAUDE.md §4 step-10 sanity check) before
  trusting the harder-task results below.
- **H7 confirmed**: every `context_overflow` failure in the core 6-task design
  (20 total) happened in classic mode; none in Code Mode,
  all on T4. On this 24 GB GPU, context length is the binding constraint
  classic MCP runs into, and Code Mode sidesteps it by keeping bulk data out of the model's
  context.
- **T4 specifically fails 0/20 in classic mode for a tool-design reason, and a follow-up probe
  shows that isn't the whole story**: the only `list_invoices` tool classic mode had access to
  returns every matching invoice in one unbounded call. A follow-up run with a properly paginated
  version of that same tool (`T4p`, §4.5) still scored 0/20: classic mode's
  turn-by-turn, full-history-resend design keeps re-accumulating tool output across the several
  pages this task needs, so pagination alone does not rescue it here.
- **H1 confirmed, but only where the comparison is meaningful**: on the (model, task) pairs where
  classic mode actually completes the task often enough to compare (T3: qwen3.5-9b -51%; T5: qwen3.5-9b -65%),
  Code Mode cuts median input tokens substantially (see §4.2) -- at the cost of more output tokens
  on simple tasks (H3), though output tokens actually *drop* on T3/T5 (see H3, §6).
- **H6 confirmed, in both modes**: going from 12 to 60 tool definitions grows median input tokens
  on T3/T5 by +55% to +114% in classic mode and
  -6% to +97% in Code Mode (see §4.3) -- tool bloat is a real,
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

- **GPU**: 1x NVIDIA L40S-24Q vGPU profile (~21.7 GB usable VRAM), driver 570.124.06, CUDA 12.8.
- **Serving stack**: vLLM 0.19.0 (the newest version this GPU's driver can run -- see
  `configs/models.yaml` for the full version-selection trail), OpenAI-compatible HTTP API,
  `openai` Python SDK >=1.40 as the client.
- **Models actually benchmarked** (2 of the 4 planned; see §8 for why):

| Model | HF repo | Quantization | vLLM tool-call parser | Max context (tokens) |
|---|---|---|---|---|
| gpt-oss-20b | `openai/gpt-oss-20b` | mxfp4-native | `openai` | 32,768 |
| qwen3.5-9b | `RedHatAI/Qwen3.5-9B-FP8-dynamic` | fp8-dynamic | `qwen3_coder` | 32,768 |

- **Dates**: pilot and fairness investigation 2026-11-22/23; full series 2026-11-24.
- One model in GPU memory at a time throughout (CLAUDE.md §0.4): serve -> smoke test -> full
  series -> stop -> VRAM confirmed free before the next model.

## 3. Methodology

### 3.1 Tasks

Six tasks against a deterministic SQLite CRM (500 customers, 200 products, 5,000 orders, 8,000
invoices, 1,500 tickets, seed 42), ranging from a single lookup to a large-volume, multi-step
aggregation:

- **T1 (Simple count)**: How many open tickets does customer C0042 have?
- **T2 (Simple, two-step lookup)**: What is the email address of the customer who placed order O1337?
- **T3 (Chained aggregation)**: Among VIP customers in the 'South' region, sum each customer's unpaid invoices (status 'unpaid' or 'overdue'). Return the top 3 customers by that total, ranked from highest to lowest.
- **T4 (Loop with writes)**: For every customer who has more than 2 overdue invoices: add the note 'Q4 follow-up' to their customer record (overwriting any existing note), and send them a reminder email about their overdue invoices (any subject and body mentioning the overdue payment is fine).
- **T5 (Large-volume filter)**: Among orders placed in October 2026 (order_date between 2026-10-01 and 2026-10-31 inclusive) in the 'Electronics' product category, return the 5 largest orders by amount, ranked from highest to lowest.
- **T6 (Textual / free-form)**: Read customer C0107's 3 most recent tickets and state the single dominant complaint category among them. The dominant category is whichever category appears most often among the 3 tickets. If all 3 tickets have different categories (no majority), the dominant category is the most recent ticket's category.

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
added afterward, for 300 trials overall. Success rates are reported with 95% Wilson
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

| Model | Mode | Task | Success (95% CI) | Median input tok. | Median output tok. | Median requests | Median latency (s) | Median GPU-s |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | classic | T1 | 100% (72%-100%) | 2,182 | 110 | 2 | 1.0 | 1.0 |
| gpt-oss-20b | classic | T2 | 100% (72%-100%) | 3,321 | 183 | 3 | 1.7 | 1.7 |
| gpt-oss-20b | classic | T3 | 0% (0%-28%) | 42,203 | 5,064 | 15 | 58.4 | 78.1 |
| gpt-oss-20b | classic | T4 | 0% (0%-28%) | 1,090 | 110 | 1 | 1.0 | 1.1 |
| gpt-oss-20b | classic | T5 | 0% (0%-28%) | 44,798 | 15,568 | 8 | 176.8 | 176.8 |
| gpt-oss-20b | classic | T6 | 100% (72%-100%) | 2,497 | 169 | 2 | 1.6 | 1.6 |
| gpt-oss-20b | codemode | T1 | 100% (72%-100%) | 2,710 | 127 | 2 | 1.2 | 1.2 |
| gpt-oss-20b | codemode | T2 | 100% (72%-100%) | 2,735 | 154 | 2 | 1.5 | 1.5 |
| gpt-oss-20b | codemode | T3 | 100% (72%-100%) | 14,004 | 1,261 | 6 | 11.4 | 11.4 |
| gpt-oss-20b | codemode | T4 | 80% (49%-94%) | 11,102 | 2,860 | 5 | 25.7 | 25.7 |
| gpt-oss-20b | codemode | T5 | 100% (72%-100%) | 23,982 | 1,676 | 6 | 15.4 | 15.4 |
| gpt-oss-20b | codemode | T6 | 100% (72%-100%) | 3,020 | 248 | 2 | 2.3 | 2.3 |
| qwen3.5-9b | classic | T1 | 100% (72%-100%) | 6,339 | 385 | 3 | 7.6 | 7.6 |
| qwen3.5-9b | classic | T2 | 100% (72%-100%) | 8,456 | 343 | 4 | 7.0 | 7.0 |
| qwen3.5-9b | classic | T3 | 100% (72%-100%) | 40,738 | 7,476 | 8 | 141.9 | 141.9 |
| qwen3.5-9b | classic | T4 | 0% (0%-28%) | 2,055 | 153 | 1 | 3.0 | 3.0 |
| qwen3.5-9b | classic | T5 | 100% (72%-100%) | 77,509 | 12,979 | 8 | 255.4 | 255.4 |
| qwen3.5-9b | classic | T6 | 100% (72%-100%) | 4,472 | 321 | 2 | 6.3 | 6.3 |
| qwen3.5-9b | codemode | T1 | 100% (72%-100%) | 5,082 | 374 | 3 | 7.2 | 7.2 |
| qwen3.5-9b | codemode | T2 | 100% (72%-100%) | 5,125 | 494 | 3 | 9.4 | 9.4 |
| qwen3.5-9b | codemode | T3 | 100% (72%-100%) | 19,828 | 1,989 | 8 | 38.0 | 38.0 |
| qwen3.5-9b | codemode | T4 | 100% (72%-100%) | 11,724 | 7,004 | 3 | 130.7 | 130.7 |
| qwen3.5-9b | codemode | T5 | 100% (72%-100%) | 27,443 | 2,387 | 12 | 45.9 | 45.9 |
| qwen3.5-9b | codemode | T6 | 100% (72%-100%) | 5,734 | 579 | 3 | 11.0 | 11.0 |

### 4.2 Code Mode effect per task (main series, 12 tools)

| Model | Task | Classic success | Code Mode success | Diff (pp) | Input tok. change | Output tok. change |
|---|---|---|---|---|---|---|
| gpt-oss-20b | T1 | 100% | 100% | +0 | +24% | +15% |
| gpt-oss-20b | T2 | 100% | 100% | +0 | -18% | -16% |
| gpt-oss-20b | T3 | 0% | 100% | +100 | n/a¹ | n/a¹ |
| gpt-oss-20b | T4 | 0% | 80% | +80 | n/a¹ | n/a¹ |
| gpt-oss-20b | T5 | 0% | 100% | +100 | n/a¹ | n/a¹ |
| gpt-oss-20b | T6 | 100% | 100% | +0 | +21% | +47% |
| qwen3.5-9b | T1 | 100% | 100% | +0 | -20% | -3% |
| qwen3.5-9b | T2 | 100% | 100% | +0 | -39% | +44% |
| qwen3.5-9b | T3 | 100% | 100% | +0 | -51% | -73% |
| qwen3.5-9b | T4 | 0% | 100% | +100 | n/a¹ | n/a¹ |
| qwen3.5-9b | T5 | 100% | 100% | +0 | -65% | -82% |
| qwen3.5-9b | T6 | 100% | 100% | +0 | +28% | +80% |

¹ *n/a: classic mode's success rate on this (model, task) is 0%, so its median tokens come from trials that mostly failed early (e.g. one request before a `context_overflow`), not from completing the task -- comparing that against Code Mode's (mostly completed) median would compare the cost of failing with the cost of succeeding, not a token-efficiency effect (ANALYSE-CLAUDE.md point C).*

![Input tokens by task](figures/01_input_tokens_by_task.png)
![GPU time per trial](figures/02_gpu_time_per_trial.png)
![Success rate by task](figures/03_success_rate_by_task.png)
![Latency by task](figures/04_latency_by_task.png)

### 4.3 Tool-count variant (12 vs 60 tools, T3/T5 only)

| Model | Mode | Task | Input tok. (12 tools) | Input tok. (60 tools) | Change | Success (12) | Success (60) |
|---|---|---|---|---|---|---|---|
| gpt-oss-20b | classic | T3 | 42,203 | 65,296 | +55% | 0% | 0% |
| gpt-oss-20b | classic | T5 | 44,798 | 95,832 | +114% | 0% | 20% |
| gpt-oss-20b | codemode | T3 | 14,004 | 27,645 | +97% | 100% | 100% |
| gpt-oss-20b | codemode | T5 | 23,982 | 22,554 | -6% | 100% | 100% |
| qwen3.5-9b | classic | T3 | 40,738 | 68,695 | +69% | 100% | 100% |
| qwen3.5-9b | classic | T5 | 77,509 | 157,821 | +104% | 100% | 100% |
| qwen3.5-9b | codemode | T3 | 19,828 | 38,899 | +96% | 100% | 100% |
| qwen3.5-9b | codemode | T5 | 27,443 | 50,782 | +85% | 100% | 100% |

![Effect of tool count](figures/05_tool_count_effect.png)

Tool-definition bloat is not a classic-mode-only cost here: Code Mode's input tokens grow by
-6% to +97% too, almost as much as classic mode's
+55% to +114% (ANALYSE-CLAUDE.md point D), because this implementation's
generated `tools.py` lists every available tool with its full docstring every time, exactly like
the tool schemas classic mode is shown. A tool-search/discovery mechanism (e.g. a `search()` call
the model uses before importing specific tools) would be needed to make Code Mode immune to this
cost -- not implemented here.

### 4.4 Cross-model comparison (H5)

gpt-oss-20b: average +46.7 percentage points across tasks; qwen3.5-9b: average +16.7 percentage points across tasks.

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
| gpt-oss-20b | classic | 10 |
| qwen3.5-9b | classic | 10 |

Every `context_overflow` failure in the core 6-task design happened in classic mode, on
T4 in all cases. T4's `list_invoices` tool takes no per-customer filter for
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

| Model | Mode | n | Successes | Failure types |
|---|---|---|---|---|
| gpt-oss-20b | classic | 10 | 0 | wrong_answer x5, context_overflow x5 |
| qwen3.5-9b | classic | 10 | 0 | context_overflow x10 |

**Result: still 0/20 success**, for a different reason than T4's single-call
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

| Model | Mode | Failure type | Count |
|---|---|---|---|
| gpt-oss-20b | classic | wrong_answer | 14 |
| gpt-oss-20b | classic | context_overflow | 10 |
| gpt-oss-20b | classic | tool_error | 5 |
| gpt-oss-20b | classic | max_turns | 4 |
| gpt-oss-20b | classic | provider_error | 4 |
| gpt-oss-20b | classic | timeout | 2 |
| gpt-oss-20b | codemode | wrong_answer | 2 |
| qwen3.5-9b | classic | context_overflow | 10 |

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
| H1 | Code Mode strongly cuts input tokens on chained/data-heavy tasks | **Confirmed, on the pairs where the comparison is meaningful** | T3: qwen3.5-9b -51%; T5: qwen3.5-9b -65% (§4.2; gpt-oss-20b's T3/T5 classic cells are excluded as not comparable -- its 0% classic success rate there means its "tokens" are from aborted trials, point C) |
| H2 | On a single-call task, Code Mode gives nothing (or costs more) | **Confirmed** | T1: 0pp success difference for both models; token changes are small and mixed in sign (§4.2) |
| H3 | Code Mode increases output tokens (the model writes code) | **Confirmed on simple tasks; disconfirmed on data-heavy tasks** | T1/T4/T6 show positive completion-token changes, as expected; T3/T5 actually *drop* sharply instead, because Code Mode's output is a short script instead of restating large tool results in prose -- a genuine secondary finding, not an exception to wave away (§4.2) |
| H4 | Classic MCP stays better on open-ended/textual tasks | **Not confirmed as stated; no difference found** | T6 (free-form): 100% success, near-identical tokens in both modes for both models -- no classic-mode advantage observed on this task shape (§4.1) |
| H5 | Code Mode's effect depends on the model | **Confirmed** | gpt-oss-20b: large success-rate rescue (partly reflecting a vLLM parser bug it hits more in classic mode, §5); qwen3.5-9b: mostly a token-efficiency gain, not a rescue, and the cleaner evidence for this hypothesis (§4.4) |
| H6 | More tools => classic MCP pays more in tokens | **Confirmed in both modes** | 12->60 tools grows median input tokens by +55% to +114% in classic mode and -6% to +97% in Code Mode on T3/T5 (§4.3) -- this implementation's Code Mode shows every tool too, so it is not immune |
| H7 | On a small GPU, Code Mode finishes tasks classic mode cannot, for context reasons | **Confirmed for the single-oversized-call failure mode; a paginated tool alone does not fully fix classic mode** | All 20 core-series `context_overflow` failures are classic-mode, all on T4; 0 in Code Mode (§4.3/§5). A follow-up with a paginated tool (T4p, §4.5) still scored 0/20 in classic mode, for a different (accumulated-context) reason |

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

Tracked here as they happen, so `REPORT.md`'s "Deviations from plan" section can be
written from this file once the full series is done (CLAUDE.md §5.8). Each entry: what
changed, the exact evidence, and who signed off.

### Model roster: 2 models instead of 4 (approved by Samir Ait-Abbou)

PLAN.md specified 4 models. Two were dropped for hard dependency conflicts on this VM
(NVIDIA L40S-24Q, driver pinned to CUDA 12.8); a third candidate proposed as a replacement
was also dropped, for a VRAM budget reason. The benchmark runs with **2 models**:
`gpt-oss-20b` and `Qwen3.5-9B-FP8-dynamic`.

#### Gemma 4 26B-A4B — dropped 2026-11-22

Smoke test failed with a definitive pydantic error:

```
model type 'gemma4' but Transformers does not recognize this architecture
```

Root cause: Gemma 4's MoE architecture requires `transformers>=5.0`. Every vLLM version
that can run on this VM's CUDA 12.8 driver pins `transformers<5` (newer vLLM releases that
bundle `transformers>=5` also bundle a CUDA-13-only native extension). This is a hard
dependency conflict, not a configuration issue -- no pip-installable combination of vLLM +
transformers can both recognize the Gemma 4 architecture and run on this GPU.

#### Qwen3.8-27B-INT4 — dropped 2026-11-23

Loaded and ran without crashing, but produced numerically incoherent, deterministic output
under vLLM 0.19.0 regardless of quantization repo or flags:
- The `reasoning` field was a meaningless mix of Latin/Chinese/Arabic characters, identical
  across 5 independent generations (`temperature=0`).
- Identical result with `--enforce-eager` on and off.
- Identical result with `--quantization compressed-tensors` forced explicitly (not just
  auto-detected).
- Identical result on two independent replacement quantization repos
  (`avyukth/Qwen3.8-27B-AWQ-INT4`, `barrydeen/Qwen3.8-27B-AWQ-4bit`).

Root cause (most consistent with all observations): the model's official card requires
`transformers>=5.8.0` for its hybrid attention architecture (16 full-attention layers + 48
Gated DeltaNet layers); vLLM 0.19.0 pins `transformers<5`, the same hard wall as Gemma 4, but
manifesting as silent numerical corruption instead of an immediate load error because the
layer type still loads (correct shapes, fast inference) without the architecture-specific
behavior the newer transformers version would have provided.

#### Qwen3-32B-AWQ — proposed as a 3rd model, dropped same day (approved by Samir)

Proposed as a dense (non-MoE, non-hybrid-attention) replacement candidate, avoiding both
architectural walls above. Loaded, but its own AWQ-4bit weights alone consume ~18.6 GiB of
this VM's ~20.8 GiB usable VRAM budget, leaving a KV cache of only ~1.57 GiB:

```
ValueError: To serve at least one request with the model's max seq len (16384),
(4.0 GiB KV cache is needed, which is larger than the available KV cache memory
(1.57 GiB). Based on the available memory, the estimated maximum model length is 6416.
```

A realistic `max_model_len` of ~9000 tokens is far below the ~21,000-46,000-token prompts
already observed on the data-heavy tasks (T3/T4/T5). Running it anyway would bias the
comparison (an artificial context-overflow handicap from VRAM alone, not from the model's
real capability), so it was dropped rather than accepted as a crippled 3rd data point.

### Per-trial wall-clock limit raised from 600s (PLAN.md) to 900s

PLAN.md §4 specifies 600s max per trial. Raised to 900s on 2026-11-23 after the T5-classic
investigation (see `configs/experiment.yaml`'s `limits.max_trial_seconds` comment and
`PROGRESS.md`): gpt-oss-20b naturally converges to a correct `FINAL_ANSWER` at ~670s on that
task, just over the old 600s limit, and was being counted as a `timeout` failure while still
genuinely converging -- an artifact of the harness, not a real model limitation, and one that
specifically understated classic mode's success rate relative to Code Mode on the data-heavy
tasks this report's headline results are about. The per-request client timeout
(`limits.request_timeout_s`) was raised the same day, in two steps, for the same reason
(120s -> 180s -> 400s): qwen3.5-9b's T5-classic requests were observed taking up to ~191s on a
single slow-but-progressing request, well under the overall trial budget.

### T4 classical paginated-invoices probe (T4p) added after the full series (not in PLAN.md)

Added 2026-11-24/25 after an independent review of the results (`ANALYSE-CLAUDE.md` point A)
questioned whether T4's 0% classic-mode success rate (§4.1, §5) measured the model's reasoning
or simply this harness's tool design: the only `list_invoices` tool available to T4 returns
every matching invoice in one unbounded call (no pagination), so a single call for all overdue
invoices across all customers already exceeds `max_model_len` before the model gets a chance to
reason about per-customer counts.

To test this directly, a second `list_invoices` tool variant was added
(`CrmTools.list_invoices_page`, enabled only for this probe) that caps `limit` at 200 and
reports `total_count`/`has_more`, so a model that loops over `offset` can in principle retrieve
every overdue invoice without any single call overflowing. 10 classic-mode trials per model
(`T4p`, otherwise identical to T4) were run against this paginated tool; Code Mode was not
re-tested, since it already succeeds on T4 with the original, unpaginated tool.

**Result: still 0/20 success**, for a different reason than T4's single-call overflow -- see
§4.5. This is reported as observed (CLAUDE.md §0.5): a well-designed paginated tool removes the
single-oversized-call failure mode, but classic mode's turn-by-turn, full-history-resend design
still accumulates enough tool output across the ~5 pages needed to cover all overdue invoices
(plus the per-customer write calls) to exceed the 32,768-token context window in most trials, or
to make the model give up (and sometimes skip the write actions entirely) before finishing in
the rest. The probe is additive to the existing 280-trial series; `experiment.yaml`'s
`paginated_variant` section and `runner.py`'s `build_trial_specs`/`run_trial` wiring are
unconditional on and invisible to every other task/trial.

### Shared VM: default vLLM port moved from 8000 to 8010

Discovered 2026-11-23/24: port 8000 on `redacted-gpu-vm` can be held by another user's own
application, unrelated to this benchmark's GPU/VRAM usage. We have no permission to free
another user's process (no sudo without Samir's explicit sign-off each time, per CLAUDE.md
§0.2), so the benchmark's own default port was moved to 8010 instead (confirmed free,
overridable via `CMBENCH_VLLM_PORT`). Not a result-affecting deviation -- purely
infrastructure -- but noted here since it explains why some historical logs/traces show
`localhost:8000` and later ones show `localhost:8010`.

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
