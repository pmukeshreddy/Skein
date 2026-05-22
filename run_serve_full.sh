#!/usr/bin/env bash
# ONE run of the fast production path with ALL compatible features ON:
#   --gpus 0,1         -> multi-process NCCL collectives (NOT host-mediated)
#   SKEIN_BATCHED_PREFILL=54 -> whole 54-tok prompt prefilled in ONE forward
#   SKEIN_PREFIX_DEMO=1 -> cold then warm (prefix-cache reuse) in the same proc
#   paged KV + real Luminal CUDA graphs are always on in this path
# Reports combined SKEIN_PERF (cold + warm) with real cuda_graph counts.
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime
export CUDA_PATH=$CUDA_HOME; export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_runtime=info,skein_cli=info
export SKEIN_PERF=1
export SKEIN_PREFIX_DEMO=1
export SKEIN_BATCHED_PREFILL=54   # = prompt token count

ART=artifacts/mixtral_rtx6000_fixed_cache/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
PROMPT="The history of the Roman Empire spans many centuries and includes the rise and fall of countless emperors, generals, senators, and ordinary citizens who together shaped the ancient Mediterranean world in profound and lasting ways across art, law, language, and warfare."
LOG=/tmp/serve_full.log
MAX_ATTEMPTS=8
: > "$LOG"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "==== ATTEMPT $attempt $(date -Is) ====" >> "$LOG"
  ./target/release/skein serve \
    --artifact "$ART" \
    --workload cluster/sample_trace.jsonl \
    --cost cluster/cost_constants.toml \
    --prompt "$PROMPT" \
    --max-new-tokens 32 \
    --gpus 0,1 >> "$LOG" 2>&1
  rc=$?
  echo "==== ATTEMPT $attempt EXIT=$rc $(date -Is) ====" >> "$LOG"
  [ "$rc" -eq 0 ] && { echo "==== SUCCESS attempt $attempt ====" >> "$LOG"; exit 0; }
  tail -40 "$LOG" | grep -q 'viable initial genome' || { echo "==== NON-GENOME FAILURE ====" >> "$LOG"; exit "$rc"; }
done
echo "==== EXHAUSTED ====" >> "$LOG"; exit 1
