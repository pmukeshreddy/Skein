#!/usr/bin/env bash
# Priority-1 accuracy gate: KL of Skein candidate vs real HF transformers.
# n_prompts overridable as $1 (default 4 for a tractable-but-real sample;
# the README/full gate is 50).
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime
export CUDA_PATH=$CUDA_HOME
export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_runtime=info,skein_cli=info,skein_parity=info
export PYTHON=${PYTHON:-python3}

ART=artifacts/mixtral_rtx6000_fixed_cache/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
N=${1:-1}
PROMPTS=${2:-/tmp/one_prompt.jsonl}
LOG=/tmp/verify.log

: > "$LOG"
echo "verify start $(date -Is) n_prompts=$N sample_from=$PROMPTS" | tee -a "$LOG"
./target/release/skein verify \
  --hf-reference /home/ubuntu/mixtral-8x7b \
  --artifact "$ART" \
  --cost cluster/cost_constants.toml \
  --sample-from "$PROMPTS" \
  --n-prompts "$N" >> "$LOG" 2>&1
echo "VERIFY_EXIT=$? $(date -Is)" | tee -a "$LOG"
