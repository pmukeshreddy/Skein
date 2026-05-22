#!/usr/bin/env bash
# Real continuous batching: SKEIN_SPREAD_DEVICES=1 places tp shard d on GPU d
# (~47 GB each, ~49 GB free), so the single-process batch driver has headroom to
# co-batch. max_batch=8 artifact + 4 different-length prompts -> short ones decode
# while long ones still prefill (mixed_batch_steps>0, max_concurrent_inflight>1).
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime
export CUDA_PATH=$CUDA_HOME; export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_runtime=info,skein_cli=info
export SKEIN_SPREAD_DEVICES=1

ART=artifacts/mixtral_batch8
PROMPTS="Hi,Hello there my friend how are you,The history of the Roman Empire spans many centuries including the rise and fall of many emperors,Paris the capital of France is a city known across the world for its art and its long winding river"
LOG=/tmp/batch_demo_spread.log
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
    --max-new-tokens 12 >> "$LOG" 2>&1
  rc=$?
  echo "==== ATTEMPT $attempt EXIT=$rc $(date -Is) ====" >> "$LOG"
  [ "$rc" -eq 0 ] && { echo "==== SUCCESS attempt $attempt ====" >> "$LOG"; exit 0; }
  tail -40 "$LOG" | grep -q 'viable initial genome' || { echo "==== NON-GENOME FAILURE ====" >> "$LOG"; exit "$rc"; }
done
echo "==== EXHAUSTED ====" >> "$LOG"; exit 1
