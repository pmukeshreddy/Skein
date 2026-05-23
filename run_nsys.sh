#!/usr/bin/env bash
# Full-timeline profile of a short decode (v1 default path, no ktime/v2 overhead)
# so we can see where the token's ~16.5ms actually goes: attention vs MoE vs
# logits kernels, NCCL all-reduce time, and GPU idle gaps (host orchestration).
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime
export CUDA_PATH=$CUDA_HOME
export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_runtime=info,skein_cli=info
export SKEIN_PERF=1
export SKEIN_SPARSE_MOE=1
export SKEIN_ONDEVICE_MOE=1
export SKEIN_MOE_FP8=1

NSYS=/home/ubuntu/nsight/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys
ART=/tmp/art_big/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
MAX_NEW=${1:-8}
OUT=/tmp/skein_prof

"$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output "$OUT" \
  --force-overwrite true \
  ./target/release/skein serve \
    --artifact "$ART" \
    --workload cluster/sample_trace.jsonl \
    --cost cluster/cost_constants.toml \
    --prompt "The capital of France is" \
    --max-new-tokens "$MAX_NEW" \
    --gpus 0,1
echo "==== nsys exit=$? ===="
