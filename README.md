# Skein

A compiler for distributed LLM inference. Skein decides **what** to run across your cluster — TP/PP/EP placement, KV layout, per-layer quantization, CUDA Graphs, prefix cache, continuous batching — and compiles it down to optimized per-device kernels.

## Pipeline

```mermaid
flowchart TD
    subgraph S1["Stage 1 — Plan Search  (no GPU needed)"]
        IN[Model Config · Cluster Spec · Workload Trace] --> PARSE
        CAL[Calibration\ncost constants + drift table] --> COST
        COST[5-Term Cost Model\ncompute · comm · memory · bubble · launch] --> SEARCH
        PARSE[Parse & Validate] --> SEARCH
        SEARCH[Enumerate TP · PP · EP placements\nDP search over per-layer dtype] --> PLAN
        PLAN([Winning Plan])
    end

    subgraph S2["Stage 2 — Compile  (GPU required)"]
        PLAN --> LOWER
        LOWER[Lower Plan to per-device graphs\nshard weights across GPUs] --> KERNELS
        KERNELS[Search for optimal kernels\nper device] --> ART
        ART([Compiled Artifact])
    end

    subgraph S3["Stage 3 — Parity Gate"]
        ART --> PAR
        PAR{Run bf16 reference vs candidate\nmeasure KL drift per layer}
        PAR -->|drift ≤ SLO| OK([Artifact Accepted])
        PAR -->|drift > SLO — raise drift table + re-search| SEARCH
    end

    subgraph S4["Stage 4 — Serve"]
        OK --> BATCHER
        BATCHER[Continuous Batcher\nadmit requests · mix prefill + decode]
        BATCHER --> KV[Paged KV Cache\nradix tree · prefix reuse]
        KV --> DECODE[Decode Forward]
        DECODE --> GRAPHS[CUDA Graphs\ncapture · cache · replay]
        GRAPHS --> HOTSWAP[Hot-swap\nswap artifact without dropping server]
        HOTSWAP --> SERVER([HTTP Inference Server])
    end
```

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
make build    # CUDA by default; CPU fallback when nvcc is absent
make test
make lint
```

## Run

```
cargo build --release

# Plan search — no GPU needed
./target/release/skein extract \
    --model   configs/mixtral_8x7b_config.json \
    --cluster cluster/rtx6000_2x.toml \
    --trace   cluster/sample_trace.jsonl \
    --drift   models/mixtral_8x7b_drift.toml \
    --cost    cluster/cost_constants.toml \
    --out     artifacts/plan.json

# Full pipeline
./target/release/skein compile  --model ... --cluster cluster/rtx6000_2x.toml --weights ... --out artifacts/
./target/release/skein verify   --artifact artifacts/LATEST
./target/release/skein serve    --artifact artifacts/LATEST --port 8080
./target/release/skein bench    --artifact artifacts/LATEST --baseline <vllm-endpoint>

# Calibrate once per (hardware, model) pair
./target/release/skein calibrate \
    --hardware rtx_pro_6000_blackwell \
    --model    configs/mixtral_8x7b_config.json \
    --corpus   crates/skein_calibrate/corpus/mixtral_8x7b.toml
```

See `docs/` for the detailed design of each stage.
