# Skein

A parallelization-plan compiler for distributed LLM inference, built on top of
[Luminal](https://github.com/luminal-ai/luminal). Skein decides **what** to run
(TP/PP/EP placement, KV layout, batching, per-component quantization, CUDA
Graphs, speculative decoding, prefix cache) and Luminal decides **how** each
kernel runs.

Skein never writes CUDA kernels. It lowers a winning `Plan` to per-device
`luminal::Graph` instances and lets Luminal's search-based compiler take it
from there.

## Architecture

```
HF config.json ─┐
cluster.toml   ─┼──► skein_ir ──► skein_extract ──► Plan
trace.jsonl    ─┘    (typed IR)   (enum+DP search)
                                            │
                                            ▼
                            skein_emit (per-device luminal::Graph)
                                            │
                                            ▼
                          skein_compile (Luminal search → Runtime)
                                            │
                                            ▼
                          skein_parity (KL drift vs bf16 reference)
                                            │
                                            ▼
                          skein_runtime (paged KV, batcher, hot-swap)
```

Nine crates, all in `crates/`:

| Crate | Role |
|---|---|
| `skein_ir`        | typed IR, `Plan`/`ClusterSpec`/`Workload` types, HF importer |
| `skein_cost`      | analytic 5-term cost model (compute + comm + memory + bubble + launch) |
| `skein_extract`   | enumeration over global axes + DP over per-layer dtype |
| `skein_emit`      | `Plan` → per-device `luminal::Graph` + sharded weights + topology |
| `skein_compile`   | wrapper around Luminal's `cx.search(...)` + artifact format |
| `skein_parity`    | logit-level KL/MSE parity gate (Skein-bf16 vs candidate) |
| `skein_runtime`   | paged KV, continuous batcher, CUDA Graphs cache, hot-swap, server |
| `skein_cli`       | `compile`, `serve`, `bench`, `calibrate`, `verify`, `extract` |
| `skein_calibrate` | offline calibration of cost constants + drift table |

The production target is NVIDIA H100. The default build enables the `cuda`
feature, selecting the `CudaComputeRuntime` and the NCCL / CUDA Graphs / RDMA
runtime paths. Building with `--no-default-features` selects the CPU
`NativeComputeRuntime` and the in-process collective backend — used for CI and
for developing the planning + runtime logic without a GPU. See `docs/` for the
detailed design of each stage.

## Validation metrics (vs vLLM on 2× H100 SXM5, Mixtral 8x7B)

1. **Throughput at P99 SLO** — sustained tokens/sec at P99 TTFT ≤ 500 ms AND
   P99 TPOT ≤ 50 ms.
   **Acceptance:** ≥ 1.4× vLLM default AND ≥ 1.15× vLLM guide-tuned.
2. **Goodput on bimodal workload** — 50% short-chat + 50% long-RAG, interleaved.
   **Acceptance:** matches or beats vllm-disaggregated AND beats both static
   vLLM configs.
3. **Drift compliance rate** — fraction of prompts whose output KL divergence
   from the bf16 reference is ≤ `workload.slo.max_accuracy_drift`.
   **Acceptance:** ≥ 99%.

If any metric fails the project stops and reports which architectural decision
to revisit. We do not paper over with prose.

## Plan search (no GPU required)

Plan extraction is pure analysis — it runs anywhere, including a CPU-only
build. Build and run a search on the bundled Mixtral config:

```
cargo build --release --no-default-features
./target/release/skein extract \
    --model   configs/mixtral_8x7b_config.json \
    --cluster cluster/h100_2x.toml \
    --trace   cluster/sample_trace.jsonl \
    --drift   models/mixtral_8x7b_drift.toml \
    --cost    cluster/cost_constants.toml \
    --out     plan.json
```

This runs the full enumeration + DP search and writes `plan.json` — no GPU
required, typically under 1 s. Pass `--output json` to emit the report as a
single JSON object (suitable for `jq`):

```
./target/release/skein extract ... --output json | jq .
```

## Full pipeline (H100)

The default build targets CUDA:

```
cargo build --release
./target/release/skein compile  --model ... --cluster ... --weights ... --out artifacts/
./target/release/skein verify   --artifact artifacts/LATEST --reference <bf16-artifact>
./target/release/skein serve    --artifact artifacts/LATEST --port 8080
./target/release/skein bench    --artifact artifacts/LATEST --baseline <vllm-endpoint> \
                                --metrics throughput,goodput,drift_compliance
```

Calibration (run once per `(hardware, model)` pair, reuse across compiles):

```
./target/release/skein calibrate \
    --hardware h100_sxm5 \
    --model    configs/mixtral_8x7b_config.json \
    --corpus   crates/skein_calibrate/corpus/mixtral_8x7b.toml
```

## Build (make targets)

```
make build        # cuda by default; --no-default-features when nvcc is absent
make test
make lint

make compile-mixtral   # GPU pipeline targets; abort cleanly without CUDA
make verify-mixtral
make bench
```
