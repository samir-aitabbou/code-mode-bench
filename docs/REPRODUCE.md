# Reproducing the benchmark

## What you need

- **A GPU with at least ~22 GB of free VRAM.** This benchmark was run on a
  single NVIDIA L40S-24Q vGPU profile (~21.7 GB usable); a full 24 GB
  consumer/datacenter GPU should work equally well. Less VRAM will likely
  fail to load the larger model (`qwen3.5-9b`'s FP8 weights + a usable KV
  cache) even at `--max-model-len 32768`.
- **A CUDA driver that supports CUDA 12.8 or newer.** vLLM 0.19.0
  (pinned, see `configs/models.yaml` for why) needs it. If your driver
  supports a newer CUDA version, a newer vLLM may also work, but has not
  been tested here.
- **Python >= 3.11**, [`uv`](https://docs.astral.sh/uv/) for dependency
  management, and `tmux` (for the unattended full-series run).
- No API keys, no cloud account, no internet access needed once the model
  weights are downloaded from Hugging Face.

## Setup

```bash
git clone <this repo>
cd code-mode-bench

# Application dependencies + dev tools (pytest, ruff, mypy)
uv sync --extra dev

# A separate venv for vLLM itself (kept apart so the application's own
# dependencies never constrain vLLM's, and vice versa)
uv venv .venv-vllm --python 3.11
source .venv-vllm/bin/activate
pip install vllm==0.19.0   # see configs/models.yaml for why this exact version
deactivate

# Generate the deterministic reference database
uv run cmbench data
```

## Quickstart: one trial, one model

```bash
# Serve a model (reads configs/models.yaml, writes results/vllm.pid)
bash scripts/serve_vllm.sh gpt-oss-20b

# Sanity check: 1 trial of T1 in each mode
uv run cmbench smoke --model gpt-oss-20b

# A handful of trials, ad-hoc
uv run cmbench run --model gpt-oss-20b --mode codemode --task T3 --reps 3

# Done with this model
bash scripts/stop_vllm.sh
```

## Running the full planned series

Per model (resumable -- already-completed trial ids are skipped):

```bash
bash scripts/serve_vllm.sh gpt-oss-20b
uv run cmbench smoke --model gpt-oss-20b
uv run cmbench bench --model gpt-oss-20b   # 140 trials: see configs/experiment.yaml
bash scripts/stop_vllm.sh
```

Or every model in `configs/models.yaml`, unattended, one at a time
(serve -> smoke -> full series -> stop -> wait for VRAM, repeating per
model -- never more than one model's vLLM server in GPU memory at once):

```bash
tmux new -s cmbench 'bash scripts/run_all.sh'
# Detach (Ctrl-b d) -- the series survives an SSH disconnect.
# Check on it later:
tmux attach -t cmbench
cat results/run_status.json
tail -f results/run.log
```

**Expected duration**: on the hardware this was run on (1x L40S-24Q),
the full 280-trial series (2 models x 140 trials, including the 60-tool
variant) took ~4.9 GPU-hours of actual compute, plus model-switch
overhead -- call it half a day including the two model loads, or less if
your GPU is faster. Duration scales with the number of models in
`configs/models.yaml` and the repetition counts in
`configs/experiment.yaml`.

## Regenerating the report

```bash
uv run cmbench report
```

Regenerates `results/summary/*.csv`, `results/figures/*.png`, and
`REPORT.md`, purely from whatever trial records currently exist under
`results/raw/*.jsonl`. Safe to re-run any time (e.g. after adding more
trials, or a different model's results) -- nothing is hand-edited.

## Tests and CI

```bash
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/cmbench/
```

All of this runs without a GPU and without any API key: the test suite
uses a fake, in-process LLM provider (`tests/fakes.py`) that never makes a
real network call.
