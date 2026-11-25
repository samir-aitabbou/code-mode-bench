# Deviations from plan

Tracked here as they happen, so `REPORT.md`'s "Deviations from plan" section can be
written from this file once the full series is done (CLAUDE.md §5.8). Each entry: what
changed, the exact evidence, and who signed off.

## Model roster: 2 models instead of 4 (approved by Samir Ait-Abbou)

PLAN.md specified 4 models. Two were dropped for hard dependency conflicts on this VM
(NVIDIA L40S-24Q, driver pinned to CUDA 12.8); a third candidate proposed as a replacement
was also dropped, for a VRAM budget reason. The benchmark runs with **2 models**:
`gpt-oss-20b` and `Qwen3.5-9B-FP8-dynamic`.

### Gemma 4 26B-A4B — dropped 2026-11-22

Smoke test failed with a definitive pydantic error:

```
model type 'gemma4' but Transformers does not recognize this architecture
```

Root cause: Gemma 4's MoE architecture requires `transformers>=5.0`. Every vLLM version
that can run on this VM's CUDA 12.8 driver pins `transformers<5` (newer vLLM releases that
bundle `transformers>=5` also bundle a CUDA-13-only native extension). This is a hard
dependency conflict, not a configuration issue -- no pip-installable combination of vLLM +
transformers can both recognize the Gemma 4 architecture and run on this GPU.

### Qwen3.8-27B-INT4 — dropped 2026-11-23

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

### Qwen3-32B-AWQ — proposed as a 3rd model, dropped same day (approved by Samir)

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

## Per-trial wall-clock limit raised from 600s (PLAN.md) to 900s

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

## T4 classical paginated-invoices probe (T4p) added after the full series (not in PLAN.md)

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

## Shared VM: default vLLM port moved from 8000 to 8010

Discovered 2026-11-23/24: port 8000 on the benchmark VM can be held by another user's own
application, unrelated to this benchmark's GPU/VRAM usage. We have no permission to free
another user's process (no sudo without Samir's explicit sign-off each time, per CLAUDE.md
§0.2), so the benchmark's own default port was moved to 8010 instead (confirmed free,
overridable via `CMBENCH_VLLM_PORT`). Not a result-affecting deviation -- purely
infrastructure -- but noted here since it explains why some historical logs/traces show
`localhost:8000` and later ones show `localhost:8010`.
