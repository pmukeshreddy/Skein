# Search algorithms

`skein_extract` decides the joint optimal `Plan` by enumerating global axis
configurations and dynamically programming the per-layer dtype map for each.
This document explains the structure, the cardinality, and why we picked
enumeration + DP over alternatives.

## Two-level structure

```
extract_plan
├── enumerate_global_configs       (Vec<GlobalConfig>, ~84k entries)
├── for each global:
│     ├── constraints::reject       (hard pruning)
│     ├── budgets_after_global      (memory + drift budget for the DP)
│     ├── layer_dtype_dp            (inner knapsack-style DP)
│     ├── compose_plan              (GlobalConfig + DtypeMap)
│     └── cost_model.total_cost     (final ranking)
└── best Plan wins
```

`extract_plan` is the only public entry point. Every intermediate stage is
deterministic, side-effect-free, and unit-testable.

## Outer enumeration — cardinality

The outer iterates the product of:

| Axis          | Variants                                                 | Count |
|---------------|----------------------------------------------------------|-------|
| Parallelism   | `tp ∈ {1,2,4,8} × pp ∈ {1,2,4} × ep ∈ {1,2,4,8}`*        | ~30   |
| KV layout     | `Contiguous \| Paged(16/32/64/128)` × `kv_shard ∈ {f,t}` | 10    |
| Batching      | `Static(M) \| Continuous(M) \| ContinuousChunked(M,C)`   | ~35   |
| CUDA Graphs   | enabled or disabled                                      | 2     |
| Spec decode   | enabled or disabled                                      | 2     |
| Prefix cache  | disabled \| LRU \| LFU                                   | 3     |

\* `ep_max` collapses to 1 for non-MoE models.

Raw product: `30 × 10 × 35 × 2 × 2 × 3 ≈ 126,000`. With the practical
restriction `tp × pp × ep ≤ num_devices` the parallelism axis shrinks
further; on the canonical Mixtral 2× H100 setup the survivor count is
~50–200 (the exact figure is logged by `tracing::info!` at the end of every
search and asserted in `enumeration.rs`'s test).

**P/D disaggregation** is enumerated as a single value (`disaggregation =
None`) in Phase A. The DriftTable, runtime, and transfer topology needed to
score a true `(prefill, decode)` pair land in Phase B; pre-emitting the axis
now would only add dead Plan branches.

## Hard constraints

Each is a pure predicate over `(GlobalConfig, Cluster, Graph, CostModel)`:

| Constraint                    | What it prevents                                              |
|-------------------------------|---------------------------------------------------------------|
| `divisibility_tp`             | TP shards `hidden`; indivisible TP leaves a ragged tile.      |
| `divisibility_ep`             | EP shards experts; an MoE with `num_experts % ep != 0` fails. |
| `device_count`                | `tp × pp × ep > num_devices` is unrealizable.                 |
| `global_memory_fits`          | Even with the *cheapest* dtype map (int4/fp8 KV) the per-device peak exceeds the smallest cap; no DP choice can rescue. |

Failures increment per-reason counters surfaced in `NoFeasiblePlan` and the
final `tracing::info!`, so an empty search is debuggable without re-running
under a profiler.

## Inner DP — state space and complexity

State: `dp[block_idx][mem_bucket][drift_bucket] = min cumulative compute time`.

For each block, the DP iterates the active 24-combo set:

- weights: `bf16, fp8_e4m3, int8, int4`
- activations: `bf16, fp8_e4m3`
- kv_cache: `bf16, fp8_e4m3, int8`

⇒ 24 combos per block. The dropped dtypes (`fp16`, `fp8_e5m2`, int8
activations, int4 activations/KV) duplicate another option's
cost/drift profile or are unsupported in the runtime's activation/KV path.
Restricting the combo set keeps the DP fast without losing the Pareto
frontier.

For Mixtral (32 blocks) at the default `(memory_buckets=100,
drift_buckets=50)`:

```
operations = num_blocks × (mem_buckets+1) × (drift_buckets+1) × combos
           = 32 × 101 × 51 × 24
           ≈ 3.96M per outer candidate
```

With ~100 surviving outer candidates the total work is ~400M memory-bound
floating-point comparisons. In practice the DP completes in ~3 s on a
single Mac core; the end-to-end `extract_plan` budget is < 5 s.

To tune for tighter Plans at the cost of search wall-time, raise
`memory_buckets` and `drift_buckets` in `cluster/cost_constants.toml`.
Doubling both quadruples the DP work.

## Why enumeration + DP, not ILP

Galvatron (2022) showed that for inference-style parallel-placement
problems the global axis space is small (~hundreds, not millions) and each
candidate's per-layer optimization is naturally a DP rather than a global
ILP. The Skein problem matches that shape:

- The outer axes are discrete with small per-axis variant counts. Direct
  enumeration is simpler than the LP relaxation an ILP solver would need.
- The inner per-layer problem has a clean `(memory_used, drift_used)` state
  and additive transition cost — textbook knapsack DP territory.
- Skein has hard rules against ILP-solver dependencies (`good_lp`,
  HiGHS, CBC, Gurobi). The DP runs in pure Rust with no FFI.

Alpa's ILP-based approach handles per-operator sharding in training, where
the placement space is much larger. Skein doesn't shard per operator — TP
groups handle the same op uniformly — so the ILP heavy machinery is
unnecessary.

## Why accuracy is in the DP, not a separate filter

A two-pass extraction (find min-cost Plan, then filter for drift) discards
information: the DP can pick a slightly slower per-layer dtype combo that
keeps drift inside the SLO. Two-pass would either reject the whole Plan or
silently keep a violating one. By keying `drift_used` as DP state, every
finishing cell is guaranteed `drift ≤ SLO`; the lowest-cost finisher is the
optimal drift-feasible Plan in one pass.

This also keeps the DP's transition costs honest. Cost is purely compute
time; the SLO is enforced as a state coordinate, never as a fudge in the
cost function. The cost-vs-drift Pareto frontier is implicit in the DP
table; the search picks the right point on it without tuning weights.

## Recalibration touch points

- `cluster/cost_constants.toml` — peak FLOPS, efficiency, collective
  bandwidth, launch overhead. Adjust on Phase B from real H100 measurements.
- `models/<model>_drift.toml` — per-(layer, component, dtype) drift. The
  Phase A file is a placeholder calibrated to relative ordering only; the
  absolute numbers should be measured against the HF bf16 reference and
  written back by `skein_calibrate` in Phase B.
