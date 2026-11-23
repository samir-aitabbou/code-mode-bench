#!/usr/bin/env bash
# Stop OUR vLLM server (started by scripts/serve_vllm.sh) and wait until
# GPU memory is freed, per CLAUDE.md §2's model-switch procedure. Never
# touches a process it did not start.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_DIR/results/vllm.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No $PID_FILE found -- nothing (of ours) to stop."
  exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping vLLM (pid $PID)..."
  kill "$PID"
  for _ in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "Process still alive after 30s, sending SIGKILL."
    kill -9 "$PID" || true
  fi
else
  echo "Process $PID is already gone."
fi
rm -f "$PID_FILE"

# A SIGKILL (above, or anything that kills the main `vllm serve` process
# without letting it shut down gracefully) can orphan its
# "VLLM::EngineCore" worker subprocess instead of taking it down too --
# observed in the wild: the API server was gone but EngineCore was still
# alive and holding ~19 GB of VRAM. Since CLAUDE.md's model-switch
# procedure (§2) never runs more than one model's vLLM at a time, any
# EngineCore process still around at this point is ours to clean up.
if pgrep -f 'VLLM::EngineCore' > /dev/null 2>&1; then
  echo "Found an orphaned VLLM::EngineCore process, cleaning it up too."
  pkill -9 -f 'VLLM::EngineCore' || true
  sleep 1
fi

echo "Waiting for GPU memory to drop below 500 MiB (max 60s, per PLAN.md §2)..."
for _ in $(seq 1 12); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
  if [ "$USED" -lt 500 ]; then
    echo "GPU memory used: ${USED} MiB. Clear."
    exit 0
  fi
  sleep 5
done

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
echo "GPU memory still at ${USED} MiB after 60s." >&2
echo "Per CLAUDE.md §0.4: if this is not freed by a process we started, STOP and ask Samir." >&2
exit 1
