#!/usr/bin/env bash
# Priority-1 correctness run: one greedy generation across both GPUs.
# serve JIT-searches kernels at boot with an entropy-seeded RNG, so the
# "viable initial genome" search is flaky. Retry until a run gets past the
# search into inference; failures are fast (~10s), success commits to the
# full forward.
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
# NVRTC needs the CUDA headers (cuda_bf16.h) to compile bf16 kernels. The
# resolver in luminal_cuda_lite appends "$CUDA_HOME/include" to --include-path.
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime
export CUDA_PATH=$CUDA_HOME
export CUDA_ROOT=$CUDA_HOME
# Quiet egglog; keep skein INFO for rank logs + SKEIN_PERF; turn on
# dyn_runtime DEBUG so per-block (set_tensor) + position progress is visible.
export RUST_LOG=warn,skein_compile::dyn_runtime=debug,skein_runtime=info,skein_cli=info
export SKEIN_PERF=1

ART=artifacts/mixtral_rtx6000_fixed_cache/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
LOG=/tmp/serve.log
MAX_ATTEMPTS=8

: > "$LOG"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "==== ATTEMPT $attempt $(date -Is) ====" >> "$LOG"
  ./target/release/skein serve \
    --artifact "$ART" \
    --workload cluster/sample_trace.jsonl \
    --cost cluster/cost_constants.toml \
    --prompt "The capital of France is" \
    --max-new-tokens 2 \
    --gpus 0,1 >> "$LOG" 2>&1
  rc=$?
  echo "==== ATTEMPT $attempt EXIT=$rc $(date -Is) ====" >> "$LOG"
  if [ "$rc" -eq 0 ]; then
    echo "==== SUCCESS on attempt $attempt ====" >> "$LOG"
    exit 0
  fi
  # Only retry the flaky genome-search failure; bail on anything else.
  if ! tail -40 "$LOG" | grep -q 'viable initial genome'; then
    echo "==== NON-GENOME FAILURE — stopping ====" >> "$LOG"
    exit "$rc"
  fi
done
echo "==== EXHAUSTED $MAX_ATTEMPTS ATTEMPTS ====" >> "$LOG"
exit 1
