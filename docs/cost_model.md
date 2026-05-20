# Cost model

`skein_cost` scores a `Plan` analytically, in microseconds per call, against
a representative decode-step workload. It runs on any host (no GPU required)
and does **not** measure real kernels — that is `skein_calibrate`'s job.

## Five additive terms

All in microseconds, per device. Final wall-clock = **max over devices**, not
sum.

### 1. Compute time

```
compute_us(layer, plan, device) =
    flops(layer, batch, seq, kv) /
    (peak_tflops[device.kind][weight_dtype] × 1e12
     × efficiency[op_kind][weight_dtype]
     × compute_divisor(layer.kind, tp, ep))
    × 1e6
```

- `flops` is the textbook arithmetic intensity for the op (multiply-add
  counted as 2). See `compute::layer_flops`.
- `peak_tflops` and `efficiency` come from `cost_constants.toml`.
- `compute_divisor` splits matmul work across the TP/EP group:
  attention / MLP / lmhead / embedding → `tp`; MoE → `tp × ep`;
  RmsNorm → 1 (replicated).

The decode-step workload (`seq=1`, `kv=decode_kv_tokens`) is used because
throughput at a fixed latency SLO — the headline validation metric — is
determined by decode-time per-step wall-clock.

### 2. Communication time

```
comm_us(collective) =
    path.total_latency_us
  + (factor(n) × bytes) / (path.min_bandwidth_gbps × collective_efficiency × 1e9)
    × 1e6
```

The topology graph walker finds the **critical path** for a collective: the
pair of participants whose latency-weighted shortest path is the longest.
Min bandwidth along that path is the bandwidth bottleneck.

`factor(n)` is the standard ring-algorithm bytes-on-the-wire multiplier:

| Collective       | factor       | Source |
|------------------|--------------|--------|
| `RingAllReduce`  | `2(n-1)/n`   | Patarasuk & Yuan 2009 (NCCL doc) |
| `AllGather`      | `(n-1)/n`    | same |
| `ReduceScatter`  | `(n-1)/n`    | same |
| `AllToAll`       | `(n-1)/n`    | NCCL doc |
| `Broadcast`      | `1`          | one-shot send |
| `SendRecv`       | `1`          | one-shot send |

For collectives the Plan does not issue (e.g. TP collectives when `tp = 1`)
`comm::collectives_on_device` returns an empty list — the term contributes
zero, no special case needed.

### 3. Memory penalty

```
mem_us(plan, device) =
    0                                      if peak <= cap
    (peak - cap) / 1e9 × overshoot_us_per_gb   otherwise
```

`overshoot_us_per_gb = 1_000_000.0` by default — one second per GB of
overshoot. Any OOM Plan is therefore strictly worse than any fitting Plan.
This makes memory fit a hard constraint without breaking the additive
structure of `total_cost`.

`peak` sums weights + KV cache + activation envelope, after TP/EP/PP
sharding. See `memory::peak_memory_bytes`.

### 4. Pipeline bubble

```
bubble_us(plan, stage_compute_us) =
    0                                      if pp <= 1
    ((pp - 1) / pp) × stage_compute_us     otherwise
```

Standard 1F1B / GPipe formula. `pp = 2` → 0.5 × stage. `pp = 4` →
0.75 × stage.

### 5. Launch overhead

```
launch_us(plan, kernels) =
    launch_us_per_kernel × kernels × (1 - coverage)
```

CUDA Graph coverage table:

| Plan batching                        | CUDA Graphs | Coverage |
|--------------------------------------|-------------|----------|
| `Static(_)`                          | enabled     | 1.00     |
| `Continuous(_)`, ≥ 4 capture classes | enabled     | 0.95     |
| `Continuous(_)`, fewer classes       | enabled     | 0.70     |
| `ContinuousChunked(...)`             | enabled     | 0.40     |
| anything                             | disabled    | 0.00     |

Assumption: a captured CUDA Graph drops the *per-kernel* launch cost to ~0.
Coverage is the fraction of decode steps whose `(batch_size, kv_class)` pair
matches a captured graph. The numbers above are envelopes — replace with
measured coverage during `skein calibrate`.

## Per-device max dominates

The wall-clock for one forward step is the slowest device's completion time,
not the average. TP groups must rendezvous at every AllReduce; PP stages run
in lockstep at micro-batch boundaries. The cost model accordingly returns
`max_d per_device_cost(d)`, never `sum`, never `mean`.

This rule is what makes the cost model rank an asymmetric Plan honestly: a
pipeline-parallel split that puts more blocks on stage 0 is correctly scored
as "as slow as stage 0".

## Recalibration

`cost_constants.toml` ships with envelope numbers — datasheet peaks and
educated efficiency guesses. To replace them with measured values:

```
skein calibrate --hardware <name> --model <config.json> \
    --out-cost cluster/cost_constants.toml \
    --out-drift models/<model>_drift.toml
```

`skein_calibrate` runs sample Luminal compiles on a representative corpus,
measures kernel runtimes for each `(op, dtype)`, fits an efficiency
constant, and writes it back. It also fits the drift table consumed by the
DP in `skein_extract`. Producing production constants needs the target GPU.

## What the cost model does **not** include

- **Accuracy.** Drift stays as a DP constraint in `skein_extract`'s inner
  loop. The cost function is purely a latency model.
- **Prefill cost.** The decode-step model is what determines steady-state
  throughput; the prefill cost is a one-off paid per request, not what the
  Plan search ranks against.
- **CUDA kernel selection.** That's Luminal's job. Skein delegates kernel
  optimization entirely to Luminal's search-based compiler.
