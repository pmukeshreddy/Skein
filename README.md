# Skein

A compiler for distributed LLM inference. Skein finds the optimal placement across your cluster — TP/PP/EP, KV layout, per-layer quantization, CUDA Graphs, prefix cache, continuous batching — and compiles it down to per-device kernels.

## Pipeline

```
Model Config    →    Plan Search    →    Compile    →    Parity Gate    ──pass──→    Serve
Cluster·Workload    TP·PP·EP·dtype DP   kernel search     KL drift               CUDA Graphs·paged KV
                           ↑                                   │
                           └──────────────fail─────────────────┘
```

## Results

Mixtral 8×7B · 2× RTX PRO 6000 Blackwell · PP=2 TP=1 EP=1 · **168 tok/s** decode throughput

## Crates

| Crate | Role |
|---|---|
| `skein_ir` | typed IR — `Plan`, `ClusterSpec`, `Workload`, HF importer |
| `skein_cost` | 5-term cost model (compute + comm + memory + bubble + launch) |
| `skein_extract` | enumeration over TP/PP/EP axes + DP over per-layer dtype |
| `skein_emit` | `Plan` → per-device graph + sharded weights + topology |
| `skein_compile` | kernel search + artifact format |
| `skein_parity` | KL/MSE parity gate (bf16 reference vs candidate) |
| `skein_runtime` | paged KV, continuous batcher, CUDA Graphs, hot-swap, server |
| `skein_cli` | `extract` · `compile` · `verify` · `serve` · `bench` · `calibrate` |
| `skein_calibrate` | offline calibration of cost constants + drift table |

## Build

```
make build
make test
make lint
```

## Run

```
cargo build --release

./target/release/skein extract \
    --model   configs/mixtral_8x7b_config.json \
    --cluster cluster/rtx6000_2x.toml \
    --trace   cluster/sample_trace.jsonl \
    --drift   models/mixtral_8x7b_drift.toml \
    --cost    cluster/cost_constants.toml \
    --out     artifacts/plan.json

./target/release/skein compile  --model ... --cluster cluster/rtx6000_2x.toml --weights ... --out artifacts/
./target/release/skein verify   --artifact artifacts/LATEST
./target/release/skein serve    --artifact artifacts/LATEST --port 8080
./target/release/skein bench    --artifact artifacts/LATEST --baseline <vllm-endpoint>

./target/release/skein calibrate \
    --hardware rtx_pro_6000_blackwell \
    --model    configs/mixtral_8x7b_config.json \
    --corpus   crates/skein_calibrate/corpus/mixtral_8x7b.toml
```

See `docs/` for design details.
