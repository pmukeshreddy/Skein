#!/usr/bin/env bash
set -u; cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime; export CUDA_PATH=$CUDA_HOME; export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_runtime=info,skein_cli=info
export SKEIN_PERF=1
export SKEIN_PIPELINE=1
ART=artifacts/mixtral_rtx6000_fixed_cache/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
LOG=/tmp/serve_pipe_only.log; : > "$LOG"
for attempt in $(seq 1 6); do
  ./target/release/skein serve --artifact "$ART" --workload cluster/sample_trace.jsonl --cost cluster/cost_constants.toml --prompt "The capital of France is" --max-new-tokens 32 --gpus 0,1 >> "$LOG" 2>&1
  rc=$?; echo "EXIT=$rc" >> "$LOG"; [ "$rc" -eq 0 ] && { echo SUCCESS >> "$LOG"; exit 0; }
  tail -40 "$LOG" | grep -q 'viable initial genome' || exit "$rc"
done
