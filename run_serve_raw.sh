#!/usr/bin/env bash
set -u; cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime; export CUDA_PATH=$CUDA_HOME; export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_runtime=info,skein_cli=info
export SKEIN_PERF=1 SKEIN_SPARSE_MOE=1 SKEIN_ONDEVICE_MOE=1 SKEIN_MOE_FP8=1
export SKEIN_RAW_LAUNCH=1
ART=/tmp/art_big/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
LOG=/tmp/serve_raw.log; : > "$LOG"
for a in $(seq 1 6); do
  timeout 240 ./target/release/skein serve --artifact "$ART" --workload cluster/sample_trace.jsonl --cost cluster/cost_constants.toml --prompt "The capital of France is" --max-new-tokens "${1:-16}" --gpus 0,1 >> "$LOG" 2>&1
  rc=$?; [ "$rc" -eq 124 ] && { echo HANG; exit 124; }; [ "$rc" -eq 0 ] && { echo OK; exit 0; }
  tail -30 "$LOG" | grep -q 'viable initial genome' || { echo "FAIL rc=$rc"; exit $rc; }
done
