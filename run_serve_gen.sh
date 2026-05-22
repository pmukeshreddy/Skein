#!/usr/bin/env bash
# 2-GPU greedy generation for the prompt "The capital of France is".
# Based on run_serve_verdict.sh: same env + retry-on-flaky-genome logic, but
# generates MAX_NEW tokens (default 32) so we get real throughput, not just 2.
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime
export CUDA_PATH=$CUDA_HOME
export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_compile::dyn_runtime=debug,skein_runtime=info,skein_cli=info
export SKEIN_PERF=1

ART=artifacts/mixtral_rtx6000_fixed_cache/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
MAX_NEW=${1:-32}
LOG=/tmp/serve_gen.log
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
