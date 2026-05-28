# Compile Pipeline

`skein compile` is now the full artifact pipeline:

1. Load model config, cluster topology, workload trace, drift table, and cost constants.
2. Run `skein_extract::extract_plan` to choose the lowest-cost feasible `Plan`.
3. Lower each device with `skein_emit::lower_per_device`.
4. Compile every lowered segment through `compile_with_luminal::<R>` using the selected runtime.
5. Materialize per-device safetensors shards and write a reloadable `SkeinArtifact`.
6. Build a bf16 reference artifact for the same model and run advisory Skein-vs-Skein parity.
7. Write `parity_report.json`, update the drift table on parity failure, and update `LATEST`.

The artifact directory is content-addressed by `Plan::content_hash()`:

```text
artifacts/<plan_hash>/
  plan.json
  topology.json
  metadata.json
  parity_report.json
  reference_bf16/
  device_0/
    graph_segments.bin
    weights.safetensors
    io.json
```

## Parity Policy

Compile follows the NVIDIA Model Optimizer pattern: artifact
production and accuracy gating are separate by default. Parity runs inside
compile because Skein uses the result to refine the drift table, but a failed
parity report does not fail the compile unless `--enforce-parity` is set.

Default behavior:

- artifact is written;
- `parity_report.json` is written;
- drift table is updated monotonically on failure;
- `LATEST` points at the compiled artifact;
- command exits zero.

CI behavior:

- pass `--enforce-parity`;
- the same report and drift update happen;
- parity failure returns `CliError::ParityFailed`.

`skein verify --enforce` is the standalone validation gate for an existing
artifact. Without `--reference`, verify uses `<artifact>/reference_bf16`,
which compile writes during advisory parity.

## Failure Modes

- Input loading errors stop before search.
- No feasible plan stops before lowering.
- Lowering or Luminal compile errors stop before artifact activation.
- Existing content-addressed artifact directories are reloaded rather than
  silently overwritten.
- Parity failures update the drift table and only become hard failures when
  `--enforce-parity` or `skein verify --enforce` is used.

## CPU and GPU builds

The default build selects `CudaComputeRuntime` for compile / verify /
calibrate on an NVIDIA host. Building with `--no-default-features` selects
`NativeComputeRuntime`, which exercises the same pipeline with CPU timings
and native Luminal execution.

The CPU build is for pipeline correctness and CI; the GPU build is the
production target for calibrated cost constants and final numeric
validation.

## Runtime

**Paged KV cache.** Each rank owns a local page pool managed by a free-list + LRU allocator. A radix prefix trie enables cross-request KV reuse: pages survive request completion and are matched against the next request's prompt prefix so shared tokens skip K/V recomputation. TP head-sharding and PP layer-slicing mean KV is already partitioned — no cross-rank KV synchronisation during decode.

**Prefill / decode bifurcation.** Two compiled graphs at different sequence lengths share the same resident weight device pointers (no second weight copy). The prefill graph (gated by `SKEIN_BATCHED_PREFILL=N`) runs once over N tokens, collecting K/V outputs as `[N, kv_dim]` tensors rather than writing them slot-by-slot. Those tensors are written into the decode runner's paged cache and decode continues from position N.

**Comm-compute overlap.** Host-staged collective round trips are removed by a device-resident bf16 RingAllReduce path: the producer segment's output buffer is all-reduced in place by device pointer, collapsing D2H traffic and host tensor materializations. The NCCL receive buffer is cached across steps and per-call synchronisation is removed so the collective issues without a host flush on the critical path. Under `SKEIN_SHM_ALLREDUCE`, a kernel-only alternative runs per-layer all-reduces over a cross-process `cuMemHostRegister` shared-memory window with no NCCL collective and no proxy thread. Under `SKEIN_CAPTURE`, the full forward — all layer compute segments and their all-reduces — is captured into one CUDA Graph and replayed with a single `cuGraphLaunch` per token.

**Kernel fusion.** Elementwise ops are fused automatically by egglog rewrite rules that bracket consecutive fusible ops into a single NVRTC kernel. GLUMoE runs gate + up + SwiGLU + down in two hand-written kernels using float4 vectorised loads and `__shfl_down_sync` warp-shuffle reduction. Decode attention fuses Q·Kᵀ, scaling, softmax, and ·V into one kernel with an early exit at the actual context length.
