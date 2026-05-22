#!/usr/bin/env bash
# Prove paged-KV prefix-cache reuse: run the SAME long prompt twice (cold then
# warm). gpu_rank's SKEIN_PREFIX_DEMO path logs cold vs warm prefix_hit and
# whether the warm tokens are byte-identical. A hit needs a shared prefix >=
# page_size tokens, so the prompt is long on purpose.
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime
export CUDA_PATH=$CUDA_HOME; export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_runtime=info,skein_cli=info
export SKEIN_PERF=1
export SKEIN_PREFIX_DEMO=1

ART=artifacts/mixtral_rtx6000_fixed_cache/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
PROMPT="The history of the Roman Empire spans many centuries and includes the rise and fall of countless emperors, generals, senators, and ordinary citizens who together shaped the ancient Mediterranean world in profound and lasting ways across art, law, language, and warfare."
LOG=/tmp/prefix_demo.log
MAX_ATTEMPTS=8
: > "$LOG"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "==== ATTEMPT $attempt $(date -Is) ====" >> "$LOG"
  ./target/release/skein serve \
    --artifact "$ART" \
    --workload cluster/sample_trace.jsonl \
    --cost cluster/cost_constants.toml \
    --prompt "$PROMPT" \
    --max-new-tokens 8 \
    --gpus 0,1 >> "$LOG" 2>&1
  rc=$?
  echo "==== ATTEMPT $attempt EXIT=$rc $(date -Is) ====" >> "$LOG"
  [ "$rc" -eq 0 ] && { echo "==== SUCCESS attempt $attempt ====" >> "$LOG"; exit 0; }
  tail -40 "$LOG" | grep -q 'viable initial genome' || { echo "==== NON-GENOME FAILURE ====" >> "$LOG"; exit "$rc"; }
done
echo "==== EXHAUSTED ====" >> "$LOG"; exit 1
