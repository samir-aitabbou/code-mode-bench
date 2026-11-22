#!/usr/bin/env bash
# Start vLLM for one model, from configs/models.yaml, in the project's own
# venv (NOT Samir's pre-existing ~/venv-vllm). Waits for /v1/models to
# answer before returning. Usage: scripts/serve_vllm.sh <model_id>
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"  # so `uv` is found even in a non-interactive shell

MODEL_ID="${1:?usage: serve_vllm.sh <model_id>}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_VLLM="${CMBENCH_VLLM_VENV:-$HOME/code-mode-bench/.venv-vllm}"
LOG_DIR="$PROJECT_DIR/results"
PID_FILE="$LOG_DIR/vllm.pid"
LOG_FILE="$LOG_DIR/vllm_${MODEL_ID}.log"
PORT="${CMBENCH_VLLM_PORT:-8000}"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "A vLLM server (pid $(cat "$PID_FILE")) is already running under our control." >&2
  echo "Run scripts/stop_vllm.sh first." >&2
  exit 1
fi

cd "$PROJECT_DIR"
source "$VENV_VLLM/bin/activate"

CMD=$(uv run cmbench serve-cmd --model "$MODEL_ID" --port "$PORT")
echo "Starting: $CMD"
# shellcheck disable=SC2086
nohup $CMD > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "vLLM pid $(cat "$PID_FILE"), logging to $LOG_FILE"

echo "Waiting for http://localhost:$PORT/v1/models ..."
for _ in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "vLLM is up for model $MODEL_ID."
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "vLLM process died while loading. Check $LOG_FILE." >&2
    exit 1
  fi
  sleep 5
done

echo "Timed out waiting for vLLM to come up. Check $LOG_FILE." >&2
exit 1
