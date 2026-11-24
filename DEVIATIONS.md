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

## Shared VM: default vLLM port moved from 8000 to 8010

Discovered 2026-11-23/24: port 8000 on `redacted-gpu-vm` can be held by another user's own
application, unrelated to this benchmark's GPU/VRAM usage. We have no permission to free
another user's process (no sudo without Samir's explicit sign-off each time, per CLAUDE.md
§0.2), so the benchmark's own default port was moved to 8010 instead (confirmed free,
overridable via `CMBENCH_VLLM_PORT`). Not a result-affecting deviation -- purely
infrastructure -- but noted here since it explains why some historical logs/traces show
`localhost:8000` and later ones show `localhost:8010`.
