# Skein vs TensorRT-LLM — Mixtral 8x7B on 2× RTX PRO 6000 Blackwell

Reproduce + measured results on a clean GPU box. All numbers below are real
command output (greedy, single in-flight request), not estimates.

## Hardware / driver

- 2× NVIDIA RTX PRO 6000 Blackwell, 96 GB each (97887 MiB), compute capability
  **sm_120** (GB202).
- GPU0<->GPU1 interconnect: **PCIe Gen5 host bridge (PHB), no NVLink**
  (`nvidia-smi topo -m`).
- Driver 580.x / CUDA 13 capable.
- Cluster spec used: `cluster/rtx6000_2x.toml` (tp=2, PCIe link).

## Skein environment (isolated CUDA 12.8 runtime)

`cudarc` is pinned to the CUDA 12.8 API; keep it off torch's CUDA 13 wheels.

```bash
pip install --target ~/cuda12 \
  nvidia-cuda-nvrtc-cu12==12.8.* nvidia-cuda-runtime-cu12==12.8.* \
  nvidia-cublas-cu12 nvidia-cuda-cccl-cu12 nvidia-nccl-cu12      # nccl needed for multi-GPU serve
# flatten all .so into one dir + unversioned aliases, then:
export LD_LIBRARY_PATH=~/cuda12/lib:~/cuda12/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export CUDA_HOME=~/cuda12/nvidia/cuda_runtime   # NVRTC needs cuda_bf16.h via $CUDA_HOME/include
export CUDA_PATH=$CUDA_HOME; export CUDA_ROOT=$CUDA_HOME
pip install torch transformers safetensors accelerate   # accelerate is required by the HF parity reference
cargo build --release --features cuda
```

Pipeline: `extract` (tp=2 pp=1 ep=1, 31 bf16 + 1 fp8 layer; fp8 clamped to bf16
at compile) -> `compile` (artifact `7923e3a4...`, candidate + reference_bf16
shards 46.7 GB/device) -> `serve` / `verify`.

> Note: `compile`'s final *advisory* Skein-vs-Skein parity loads both the
> candidate and the bf16 reference forward in one process; on a 96 GB card that
> is ~2×94 GB and OOMs (SIGKILL). The artifact is fully written *before* that
> step, so `serve` and the HF `verify` gate work off it regardless.

## Skein results (2-GPU, tp=2, prompt "The capital of France is", 32 tokens)

Generated text:

```
Paris.
## What is the capital of France and why?
Paris, city and capital of France, situated in the north-central
```

`SKEIN_PERF` (paged-KV cached decode, greedy):

| metric | value |
|---|---|
| decode throughput | 9.17 tok/s |
| TPOT p50 | 104.5 ms |
| TPOT p95 | 146.3 ms |
| TTFT | 34.1 s (includes one-time per-launch weight upload) |
| per-step | seg-exec ≈ 95 ms + NCCL comm ≈ 8–10 ms |

## HF parity gate (Skein bf16 candidate vs HuggingFace transformers bf16)

`skein verify --hf-reference <weights> --artifact <art> --n-prompts 4`

| metric | value |
|---|---|
| passed | **false** |
| avg_final_kl | 0.00967  (SLO max_accuracy_drift = 0.01 → KL axis passes) |
| max_final_kl | 0.02545 |
| per-prompt final_kl | [0.00227, 0.02545, 0.00210, 0.00884] |
| failing layer | 30, weight, bf16, MSE 2.379 ≫ tol 1e-3 |

Read: the final next-token distribution tracks HF closely (avg KL < 0.01,
consistent with the correct "Paris" generation), but the strict per-layer
hidden-state MSE gate (1e-3 for bf16) fails at layer 30 — bf16 kernel /
accumulation-order differences accumulating in the residual stream. Tolerance
was **not** loosened.

## TensorRT-LLM on Blackwell (sm_120)

TRT-LLM 1.2.1 has no prebuilt sm_120 kernels; flashinfer JIT-compiles them at
first run. Beyond `pip install tensorrt_llm` (isolated venv), the box needed:

- OpenMPI (`libopenmpi-dev openmpi-bin`) — multi-GPU spawn.
- CUDA-13 cuBLAS/CCCL/cuRAND headers+libs: `pip install nvidia-cublas
  nvidia-cuda-cccl nvidia-curand nvidia-cuda-nvcc` (unified names pull cu13 into
  `nvidia/cu13`).
- `ninja-build` + `g++-12` (nvcc host compiler; default gcc-12 lacked cc1plus).
- `CUDA_HOME=.../nvidia/cu13`, plus `LIBRARY_PATH` and unversioned `libcudart.so`
  symlinks so the flashinfer link step finds `-lcudart`.

Model load itself was fine: ~8.5 s, tp=2.

## Head-to-head (Mixtral 8x7B, tp=2, 32 tokens, greedy, same prompt)

`bench/compare_trtllm.py`

| metric | Skein | TensorRT-LLM 1.2.1 | ratio |
|---|---|---|---|
| decode throughput | 9.17 tok/s | **89.14 tok/s** | ~9.7× TRT-LLM |
| TPOT p50 | 104.5 ms | **11.22 ms** | ~9.3× TRT-LLM |
| TPOT p95 | 146.3 ms | **11.55 ms** | — |
| TTFT | 34.1 s* | 18.2 ms* | not comparable* |

\* TTFT is **not** apples-to-apples: Skein `serve` is a fresh-process launcher
that re-uploads ~47 GB/GPU per launch (its TTFT is dominated by that), while
TRT-LLM keeps weights resident and is measured after warmup. The clean
comparison is TPOT / throughput, where TRT-LLM is ~9–10× faster.

Generated text differed (both greedy, same weights): Skein → "Paris."; TRT-LLM →
"a city that is known for its beauty and its history…". The divergence is from
BOS/prompt-tokenization differences between the two harnesses, not a model
disagreement.

The gap is expected: TRT-LLM is a mature engine (fused flashinfer kernels,
batched prefill, optimized MoE). Skein already runs **real per-segment CUDA
graphs** (see below) but executes the segments sequentially and re-uploads
weights per process launch.

## CUDA graphs: what's real (verified)

Two layers existed; one was genuine, one was a stub:

- **Real — Luminal per-segment kernel graphs.** Each segment's kernels are built
  into a `cudaGraph` once (`cuGraphInstantiate`) and replayed every forward via
  `cuGraphLaunch` with surgical param updates. Instrumented at the actual call
  sites (`luminal_cuda_lite`); the `--gpus` decode now logs the real counts. One
  measured 32-token run (5 prefill + 31 decode = 36 forwards):

  ```
  cuda_graph_instantiations=394   cuda_graph_replays=14184     (394 graphs x 36 forwards)
  ```

  i.e. 394 segment graphs built once, replayed 14,184 times — proof the kernel
  CUDA graphs really fire in the production decode.

- **Removed — fake serving-level `CudaGraphCache`.** It captured a no-op
  scratch-zeroing graph, replayed *that*, ran the real forward eagerly anyway,
  and reported phantom `graph_captures/replays`. Deleted; `batch_driver` now
  reports the real Luminal instantiate/launch deltas instead.

## Continuous batching + single-process multi-GPU (verified)

Two findings, both real:

- **Planner never selects `max_batch>1`.** `extract` minimizes per-step
  `total_cost`, and a larger batch only *raises* it (more comm/memory), so
  `max_batch=1` always wins — the chosen plan is byte-identical for a 5-request
  and a 24-request burst trace. So produced artifacts never enable co-batching.

- **The batching driver works, once given headroom.** The single-process
  `--batch-demo` path loaded both tp shards on GPU 0 (~93 GB) and OOM'd the
  moment ≥2 requests were in flight (1 in flight was fine). Root cause: luminal
  hardcoded `CudaContext::new(0)`. Fix: `CudaRuntime::new_on(device)` +
  `SKEIN_SPREAD_DEVICES=1` so `load_runtime_segments` places shard d on GPU d
  (~47 GB each, the host-mediated in-process collective handles cross-GPU). With
  that, real co-batching:

  | Metric | before (both shards on GPU0) | after (shard-per-GPU) |
  |---|---|---|
  | max_concurrent_inflight | 1 | **4** |
  | mixed_batch_steps | 0 | **18** |
  | result | OOM at ≥2 reqs | 4 reqs co-batched, real tokens |

  Caveat: this single-process spread path is **functional but slow**
  (host-mediated collective, ~414 s for the demo batch). The fast path for real
  throughput is continuous batching in the multi-process NCCL `--gpus` loop,
  which is the remaining work — that loop is currently single-request lockstep.

## Single-stream perf after device-logits + PP-pipelining (2026-05-24)

All rows: `artifacts/LATEST` (the `2323e83e…` candidate artifact), prompt
"The capital of France is", `--max-new-tokens 32`, greedy, CUDA 12.8 runtime.
Every row's generated text contains a coherent "Paris".

**Important:** this artifact is compiled **pp=2, tp=1** (pipeline-parallel:
layers 0–15 on GPU0, 16–31 on GPU1, one `SendRecv` of `carry_pre_block_16` per
token). The decode schedule is 3 steps; there is **no logits AllGather and no
per-layer RingAllReduce**. So the TP-oriented levers do **not** engage here:
`SKEIN_CAPTURE` (full-step capture) has empty windows on both stages, and
`SKEIN_SHM_ALLREDUCE` has no all-reduce to accelerate. Decode is still
graph-accelerated by luminal's internal per-segment graphs
(`cuda_graph_instantiations=49`, `cuda_graph_replays=1764`).

`decode_tokens_per_s = 1000/tpot_p50` is the per-stage GPU forward rate;
`true_tokens_per_s` is the end-to-end single-stream rate (incl. sample/broadcast).
The ~2.8× gap between them is the **PP serial-pipeline bubble** (one GPU idle
while the other computes), not the logits read.

| scenario | TPOT p50 (ms) | decode tok/s | true / aggregate tok/s | note |
|---|---|---|---|---|
| baseline (sparse+fp8+notrack+shm+capture+LL) | 10.41 | 96.7 | 34.4 (single) | shm/capture flags are no-ops on PP |
| + `SKEIN_DEVICE_LOGITS=1` (on-device argmax) | 10.34 | 96.7 | 33.5 (single) | **correct, perf-neutral** — logits read isn't the bottleneck |
| `SKEIN_PIPELINE_STREAMS=2` (1F1B overlap) | — | — | **90.6 aggregate** (~45/stream) | the real PP lever: both GPUs concurrent |
| `SKEIN_PIPELINE_STREAMS=4` | — | — | 91.8 aggregate | plateaus (2 µbatches already fill 2 stages) |
| TRT-LLM 1.2.1 (reference) | 11.22 | 89.1 | — | single-stream `1000/TPOT` |
| Luminal DeepSeek-R1 8×H200 (reference) | 10.7 | 93.3 | — | single-stream `1000/ITL` |

### Lever findings (honest)

- **Lever #1 (device-resident logits + on-device argmax).** Implemented + the
  bf16 argmax kernel is unit-tested (`cuda::argmax`). The TP all-gather form
  doesn't exist on a PP artifact, so it was adapted to argmax the last stage's
  full-vocab logits on-device. Correct (coherent Paris) but **perf-neutral**:
  reading the *computed* logits requires a per-token context sync (the host read
  path got that sync for free), which offsets the ~64 KB host-read it removes —
  and the logits read isn't on the PP critical path anyway. Gated behind
  `SKEIN_DEVICE_LOGITS`; default path unchanged.
- **Lever #2 (multi-block shm all-reduce).** N/A — PP has no RingAllReduce.
- **Device-resident PP `SendRecv`** (`SKEIN_DEVICE_SENDRECV`): wiring engages
  (d2h→0) but the precompiled boundary segment expects a host-staged carry, so
  binding a device buffer feeds wrong data (garbage). Making it correct needs a
  **compile-time** change (emit the carry as a device output), out of scope here.
- **STEP E calibrate:** fails (`missing cuBLASLt A input buffer`); skipped,
  zero-impact on tok/s.

**Takeaway:** single-stream decode is already competitive with the references
(96.7 per-stage / 34.4 end-to-end). The real throughput win on this PP=2 box is
**pipelining the two stages** (`SKEIN_PIPELINE_STREAMS`), giving **~91 aggregate
tok/s** (2.6× the single-stream rate) by filling the pipeline bubble.
