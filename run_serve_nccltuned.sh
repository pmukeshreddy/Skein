#!/usr/bin/env bash
set -u
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH=/home/ubuntu/cuda12/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/home/ubuntu/cuda12/nvidia/cuda_runtime; export CUDA_PATH=$CUDA_HOME; export CUDA_ROOT=$CUDA_HOME
export RUST_LOG=warn,skein_runtime=info,skein_cli=info
export SKEIN_PERF=1 SKEIN_SPARSE_MOE=1 SKEIN_ONDEVICE_MOE=1 SKEIN_MOE_FP8=1 SKEIN_NO_EVENT_TRACKING=1 SKEIN_GRAPH_DIRTY=1
# NCCL tuning (no root): try to cut the host-staged all-reduce latency.
export NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 NCCL_BUFFSIZE=1048576
ART=/tmp/art_big/7923e3a4a6acb152f3856e4bd3ad7d69530433508ec92b04243f002453dc5012
LOG=/tmp/serve_nccltuned.log; : > "$LOG"
for a in $(seq 1 6); do
  ./target/release/skein serve --artifact "$ART" --workload cluster/sample_trace.jsonl --cost cluster/cost_constants.toml --prompt "The capital of France is" --max-new-tokens "${1:-32}" --gpus 0,1 >> "$LOG" 2>&1
  rc=$?; [ "$rc" -eq 0 ] && { echo "OK"; exit 0; }; tail -30 "$LOG" | grep -q 'viable initial genome' || { echo "FAIL rc=$rc"; exit $rc; }
done
