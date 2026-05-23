#!/usr/bin/env bash
# Same env as the 60.7 tok/s run (sparse + on-device + fp8) plus per-kernel CUDA
# event timing. NOTE: ktime adds a per-call stream sync, so tok/s in this run is
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime
export CUDA_PATH=$CUDA_HOME
export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_runtime=info,skein_cli=info
export SKEIN_PERF=1
export SKEIN_SPARSE_MOE=1
export SKEIN_ONDEVICE_MOE=1
export SKEIN_MOE_FP8=1
export SKEIN_NO_EVENT_TRACKING=1
export SKEIN_GRAPH_DIRTY=1

ART=/tmp/art_big/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
MAX_NEW=${1:-32}
LOG=/tmp/serve_dirty.log
MAX_ATTEMPTS=8

: > "$LOG"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "==== ATTEMPT $attempt $(date -Is) ====" >> "$LOG"
  ./target/release/skein serve \
    --artifact "$ART" \
    --workload cluster/sample_trace.jsonl \
    --cost cluster/cost_constants.toml \
    --prompt "The capital of France is" \
    --max-new-tokens "$MAX_NEW" \
    --gpus 0,1 >> "$LOG" 2>&1
  rc=$?
  echo "==== ATTEMPT $attempt EXIT=$rc $(date -Is) ====" >> "$LOG"
  if [ "$rc" -eq 0 ]; then
    echo "==== SUCCESS on attempt $attempt ====" >> "$LOG"
    exit 0
  fi
  if ! tail -40 "$LOG" | grep -q 'viable initial genome'; then
    echo "==== NON-GENOME FAILURE — stopping ====" >> "$LOG"
    exit "$rc"
  fi
done
echo "==== EXHAUSTED $MAX_ATTEMPTS ATTEMPTS ====" >> "$LOG"
exit 1
