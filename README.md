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
                                            ▼  (Phase B)
                          skein_compile (Luminal search → Runtime)
                                            │
                                            ▼
                          skein_parity (KL drift vs HF reference)
                                            │
                                            ▼
                          skein_runtime (paged KV, batcher, hot-swap)
```

Nine crates, all in `crates/`:

| Crate | Phase | Role |
|---|---|---|
| `skein_ir`        | A | typed IR, `Plan`/`ClusterSpec`/`Workload` types, HF importer |
| `skein_cost`      | A | analytic 5-term cost model (compute + comm + memory + bubble + launch) |
| `skein_extract`   | A | enumeration over global axes + DP over per-layer dtype |
| `skein_emit`      | A | `Plan` → per-device `luminal::Graph` + sharded weights + topology |
| `skein_compile`   | B | ~20-line wrapper around Luminal's `cx.search(...)` |
| `skein_parity`    | B | logit-level KL/MSE check vs HF `transformers` reference |
| `skein_runtime`   | A/B | paged KV, continuous batcher, CUDA Graphs cache, hot-swap |
| `skein_cli`       | A/B | `compile`, `serve`, `bench`, `calibrate`, `verify`, `extract` |
| `skein_calibrate` | B | offline calibration of cost constants + drift table |

Phase A is everything that compiles on macOS without CUDA. Phase B is the H100
side and only builds with `--features cuda`. See `docs/design.md` for the
detailed split.

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

## Phase A quick start (Mac, no GPU)

Build:

```
cargo build --release
```

Run a plan search on the bundled Mixtral fixture:

```
./target/release/skein extract \
    --model   configs/mixtral_8x7b_config.json \
    --cluster cluster/h100_2x.toml \
    --trace   cluster/sample_trace.jsonl \
    --drift   models/mixtral_8x7b_drift.toml \
    --cost    cluster/cost_constants.toml \
    --out     plan.json
```

This runs the full enumeration + DP search and writes `plan.json` — no
GPU required. Expect under 1 s on a modern Mac. Pass `--output json` to
emit the report as a single JSON object (suitable for `jq`):

```
./target/release/skein extract ... --output json | jq .
```

The other subcommands — `compile`, `verify`, `serve`, `calibrate`,
`bench` — require `--features cuda` on an NVIDIA host. Invoking them on
Phase A returns a clear `RequiresCuda` error with the exact rebuild
command.

## Phase B (H100 server)

```
cargo build --release --features cuda
./target/release/skein compile  --model ... --cluster ... --weights ... --out artifacts/
./target/release/skein verify   --artifact artifacts/LATEST --reference <hf_path>
./target/release/skein serve    --artifact artifacts/LATEST --port 8080
./target/release/skein bench    --artifact artifacts/LATEST --baseline <vllm-endpoint> \
                                --metrics throughput,goodput,drift_compliance
```

Calibration (run once per `(hardware, model)` pair, reuse across
compiles):

```
./target/release/skein calibrate \
    --hardware h100_sxm5 \
    --model    configs/mixtral_8x7b_config.json \
    --corpus   crates/skein_calibrate/corpus/mixtral_8x7b.toml
```

## Build (alternative make targets)

Mac (Phase A only):

```
make build
make test
make lint
```

H100 (full pipeline):

```
make build        # auto-enables --features cuda when nvcc is on PATH
make compile-mixtral
make verify-mixtral
make bench
```
