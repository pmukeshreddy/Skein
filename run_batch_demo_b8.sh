#!/usr/bin/env bash
# Continuous-batching proof: serve the max_batch=8 artifact and submit 4
# DIFFERENT-LENGTH prompts at once. With admission allowing >1 in flight, short
# prompts begin decoding while long ones are still prefilling -> co-batching
# (max_concurrent_inflight>1) and mixed prefill+decode steps (mixed_batch_steps>0).
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime
export CUDA_PATH=$CUDA_HOME; export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_runtime=info,skein_cli=info

ART=artifacts/mixtral_batch8
PROMPTS="The capital of France is,Hello there my friend how are you doing today friend"
LOG=/tmp/batch_demo_b8.log
MAX_ATTEMPTS=8
: > "$LOG"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "==== ATTEMPT $attempt $(date -Is) ====" >> "$LOG"
  ./target/release/skein serve \
    --artifact "$ART" \
    --workload cluster/sample_trace.jsonl \
    --cost cluster/cost_constants.toml \
    --batch-demo --cuda-graphs \
    --demo-prompts "$PROMPTS" \
    --max-new-tokens 8 >> "$LOG" 2>&1
  rc=$?
  echo "==== ATTEMPT $attempt EXIT=$rc $(date -Is) ====" >> "$LOG"
  [ "$rc" -eq 0 ] && { echo "==== SUCCESS attempt $attempt ====" >> "$LOG"; exit 0; }
  tail -40 "$LOG" | grep -q 'viable initial genome' || { echo "==== NON-GENOME FAILURE ====" >> "$LOG"; exit "$rc"; }
done
echo "==== EXHAUSTED ====" >> "$LOG"; exit 1
