#!/usr/bin/env bash
# Pipeline-parallel compile: pp=2, tp=1, ep=1 on the 2x RTX PRO 6000 (no NVLink).
# Forces the PP placement so extract_plan emits SendRecv stage boundaries
# instead of 64 per-step all-reduces. Serve re-lowers the sparse/fp8 MoE at load.
set -eo pipefail
cd /home/ubuntu/Skein
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/ubuntu/skein_env.sh
set -u

export SKEIN_FORCE_PP=2
export SKEIN_FORCE_TP=1
export SKEIN_FORCE_EP=1

exec target/release/skein compile \
  --model   configs/mixtral_8x7b_config.json \
  --cluster cluster/rtx6000_2x.toml \
  --trace   cluster/sample_trace.jsonl \
  --drift   models/mixtral_8x7b_drift.toml \
  --cost    cluster/cost_constants.toml \
  --weights /home/ubuntu/mixtral-8x7b \
  --out     /tmp/art_pp \
  --search-budget 100
