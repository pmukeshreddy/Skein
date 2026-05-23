#!/usr/bin/env bash
# Pipelined PP serve: drive N concurrent streams with 1F1B overlap so BOTH GPUs
# compute at once. Aggregate decode tok/s is logged as SKEIN_PERF_PIPE.
# Usage: ./run_serve_pipe.sh [artifact] [max_new_tokens] [n_streams]
set -eo pipefail
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/ubuntu/skein_env.sh
set -u

ART="${1:-$(echo /tmp/art_pp/*/ | tr -d ' ')}"
ART="${ART%/}"
N="${2:-32}"
STREAMS="${3:-2}"

export RUST_LOG=warn,skein_runtime=info,skein_cli=info
export SKEIN_SPARSE_MOE=1 SKEIN_ONDEVICE_MOE=1 SKEIN_MOE_FP8=1
export SKEIN_NO_EVENT_TRACKING=1
export SKEIN_PIPELINE_STREAMS="$STREAMS"

echo "serving artifact: $ART  (max_new_tokens=$N, pipeline_streams=$STREAMS)"
exec ./target/release/skein serve \
  --artifact "$ART" \
  --workload cluster/sample_trace.jsonl \
  --cost cluster/cost_constants.toml \
  --prompt "The capital of France is" \
  --max-new-tokens "$N" \
  --gpus 0,1
