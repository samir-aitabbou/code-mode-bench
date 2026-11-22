#!/usr/bin/env bash
# Runs the full benchmark, unattended, on the VM alone (no PC, no Claude
# Code needed once this starts -- CLAUDE.md §0.7). For each model in
# configs/models.yaml (in `order`): serve -> smoke test -> full series ->
# stop -> wait for VRAM. A model that fails its smoke test is skipped,
# never blocks the rest. Meant to run inside `tmux new -s cmbench`.
set -uo pipefail  # no -e: one model's failure must not kill the whole run

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"
STATUS_FILE="$RESULTS_DIR/run_status.json"
LOG_FILE="$RESULTS_DIR/run.log"
VENV_VLLM="${CMBENCH_VLLM_VENV:-$HOME/code-mode-bench/.venv-vllm}"

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_DIR"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"
}

write_status() {
  # $1=model_id  $2=state (serving|smoke_failed|running|stopped|skipped|done)
  python3 - "$1" "$2" "$STATUS_FILE" <<'PY'
import json, sys, datetime
model_id, state, path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    data = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    data = {"models": {}}
data["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
data["current_model"] = model_id
data["models"][model_id] = state
json.dump(data, open(path, "w"), indent=2)
PY
}

log "run_all.sh starting"
rm -f "$RESULTS_DIR/DONE"

MODELS=$(uv run cmbench list-models)
for MODEL_ID in $MODELS; do
  log "=== model: $MODEL_ID ==="
  write_status "$MODEL_ID" "serving"

  if ! bash scripts/serve_vllm.sh "$MODEL_ID" 2>&1 | tee -a "$LOG_FILE"; then
    log "model $MODEL_ID failed to start vLLM -- skipping, continuing with next model"
    write_status "$MODEL_ID" "serve_failed"
    bash scripts/stop_vllm.sh 2>&1 | tee -a "$LOG_FILE" || true
    continue
  fi

  log "smoke-testing $MODEL_ID"
  if ! uv run cmbench smoke --model "$MODEL_ID" 2>&1 | tee -a "$LOG_FILE"; then
    log "model $MODEL_ID failed its smoke test -- skipping, continuing with next model"
    write_status "$MODEL_ID" "smoke_failed"
    bash scripts/stop_vllm.sh 2>&1 | tee -a "$LOG_FILE" || true
    continue
  fi

  log "running full series for $MODEL_ID"
  write_status "$MODEL_ID" "running"
  uv run cmbench bench --model "$MODEL_ID" 2>&1 | tee -a "$LOG_FILE"

  log "stopping vLLM for $MODEL_ID"
  write_status "$MODEL_ID" "stopping"
  bash scripts/stop_vllm.sh 2>&1 | tee -a "$LOG_FILE"
  write_status "$MODEL_ID" "done"
  log "=== model $MODEL_ID done ==="
done

touch "$RESULTS_DIR/DONE"
log "run_all.sh finished, see $RESULTS_DIR/DONE"
