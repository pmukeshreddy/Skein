#!/usr/bin/env bash
# Profiling pass: 1 short prompt, 1 forward, per-segment + per-collective timing.
# Emits SKEIN_SEG (per segment: host_in/gpu_launch/host_out us) and
# SKEIN_PERF_STEP (seg_us vs comm_us per forward pass).
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime
export CUDA_PATH=$CUDA_HOME
export CUDA_ROOT=$CUDA_HOME
# Only our perf markers at info; suppress egglog + dyn_runtime debug spam.
export RUST_LOG=warn,skein_runtime=info,skein_cli=info

ART=artifacts/mixtral_rtx6000_fixed_cache/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
LOG=/tmp/profile.log
: > "$LOG"
echo "profile start $(date -Is)" | tee -a "$LOG"
./target/release/skein serve \
  --artifact "$ART" \
  --workload cluster/sample_trace.jsonl \
  --cost cluster/cost_constants.toml \
  --prompt "Hello" \
  --max-new-tokens 2 \
  --gpus 0,1 >> "$LOG" 2>&1
echo "PROFILE_EXIT=$? $(date -Is)" | tee -a "$LOG"
