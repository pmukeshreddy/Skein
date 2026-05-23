#!/usr/bin/env bash
# Serve the pipeline-parallel artifact across both GPUs and measure decode tok/s.
# Same GLUMoE fp8 gates as the proven TP serve; NO capture, NO batched prefill
# (token-by-token prefill keeps the seq=1 SendRecv shape correct).
set -eo pipefail
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/ubuntu/skein_env.sh
set -u

# Artifact: arg 1, or the single dir under /tmp/art_pp.
ART="${1:-$(echo /tmp/art_pp/*/ | tr -d ' ')}"
ART="${ART%/}"
N="${2:-64}"

export RUST_LOG=warn,skein_runtime=info,skein_cli=info
export SKEIN_SPARSE_MOE=1 SKEIN_ONDEVICE_MOE=1 SKEIN_MOE_FP8=1
export SKEIN_NO_EVENT_TRACKING=1

echo "serving artifact: $ART  (max_new_tokens=$N)"
exec ./target/release/skein serve \
  --artifact "$ART" \
  --workload cluster/sample_trace.jsonl \
  --cost cluster/cost_constants.toml \
  --prompt "The capital of France is" \
  --max-new-tokens "$N" \
  --gpus 0,1
