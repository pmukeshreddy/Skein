# Lowering — `skein_emit`

`skein_emit` takes the winning `Plan` and produces, for every device in the
cluster:

1. A `luminal::Graph` whose tensor declarations already carry the sharded
   shapes — Luminal's compile pass operates on these shapes and is not aware
   of TP/EP.
2. A `WeightShard` listing the source byte ranges this device owns. The
   bytes are *not* moved by `lower_per_device`; that's
   `weights::write_weight_shard`'s job.
3. An `IoManifest` (`io.json`) the runtime uses at load time to validate
   that an artifact's tensor names + shapes + dtypes match what it expects.

Plus a *cluster-wide* `topology.json` describing the collectives the
runtime must issue between graph invocations.

## ShardRole resolution rules

Per `(layer, param, device)`, exactly one of:

| Variant                          | Meaning                                                                 |
|----------------------------------|-------------------------------------------------------------------------|
| `Replicated`                     | Full tensor present on this device.                                     |
| `TpOutputShard {g, idx}`         | Row-axis slice (column-parallel weight). Output activation is sharded. |
| `TpInputShard {g, idx}`          | Column-axis slice (row-parallel weight). Partial sums need AllReduce.  |
| `ExpertOwned {expert_idx}`       | MoE expert weight assigned to this device's EP shard.                   |
| `ExpertElsewhere {expert_idx}`   | Expert owned by another device — skip lowering.                         |
| `PipelineStageElsewhere`         | Layer's block is on a PP stage not owned by this device.                |
| `NoParams`                       | Marker / collective / no weights to lower.                              |

Resolution priority (early exits short-circuit):

1. **Pipeline (PP).** If the layer has a `block_idx` and that block's stage
   ≠ this device's stage → `PipelineStageElsewhere`. Pre/post layers
   (embed, final norm, lm_head) live on stage 0 / stage `pp - 1` by
   convention.
2. **Expert (EP).** For MoE expert weights with `ep > 1`, the expert index
   is bucketed by `expert_idx % ep`. The matching `ep_idx` device returns
   `ExpertOwned`; others return `ExpertElsewhere`.
3. **Tensor (TP).** Non-expert weights consult the param-name table below
   to pick column- or row-parallel.
4. Otherwise → `Replicated`.

### EP + TP simplification

When `ep > 1`, expert weights go to `ExpertOwned` / `ExpertElsewhere` and do
**not** further sub-shard via TP. Stacking TP-within-expert on top of EP
would require runtime support for nested all-reduce + all-to-all (not yet
wired). With `ep == 1`, expert weights fall through to TP rules normally
(w1/w3 column-parallel, w2 row-parallel).

## Column vs row parallel for Mixtral

| HF safetensors key suffix                              | Direction        |
|--------------------------------------------------------|------------------|
| `self_attn.q_proj.weight`                              | column-parallel  |
| `self_attn.k_proj.weight`                              | column-parallel  |
| `self_attn.v_proj.weight`                              | column-parallel  |
| `self_attn.o_proj.weight`                              | row-parallel     |
| `mlp.gate_proj.weight`, `mlp.up_proj.weight`           | column-parallel  |
| `mlp.down_proj.weight`                                 | row-parallel     |
| `block_sparse_moe.experts.<N>.w1`/`w3.weight`          | column-parallel  |
| `block_sparse_moe.experts.<N>.w2.weight`               | row-parallel     |
| `block_sparse_moe.gate.weight`                         | replicated (tiny router) |
| `model.embed_tokens.weight`, `lm_head.weight`          | replicated       |
| `*_layernorm.weight`, `model.norm.weight`              | replicated       |

## Weight slicing — byte-range arithmetic

Each `WeightSlice` carries a `ShardStrategy` that names the read pattern:

| Strategy            | Source byte ranges                                                            |
|---------------------|-------------------------------------------------------------------------------|
| `Whole`             | One range `[0, total_bytes)`.                                                 |
| `OuterAxisSlice {start, end}` | Contiguous `[start × row_stride, end × row_stride)`.                |
| `InnerAxisSlice {start, end}` | `num_rows` strided ranges: row `r` contributes `[r × row_stride + start × elem_bytes, r × row_stride + end × elem_bytes)`. |
| `ExpertOwned {idx}` | Mixtral stores experts as separate keys, so this degenerates to `Whole` on the per-expert key. Reserved for stacked-expert layouts. |

`row_stride = inner_dim × elem_bytes`. `elem_bytes = ceil(dtype.bits / 8)`.
Tests assert that the union of `start_idx ∈ {0, group_size-1}` ranges covers
every source byte exactly once (no gaps, no overlap).

## Topology emission order

`emit_topology` walks decoder blocks in execution order. For each block:

1. PP `SendRecv` (if this block is the first on its stage, when `pp > 1`).
2. EP `AllToAll` dispatch (if `ep > 1` and the block has an MoE layer).
3. TP `RingAllReduce` after the attention block's `o_proj` (if `tp > 1`).
4. EP `AllToAll` combine (if `ep > 1` and the block has an MoE layer).
5. TP `RingAllReduce` after the MoE/MLP block's `down_proj` (if `tp > 1`).

Sequence indices are dense (`0..N`). Tests pin this — `topology_deterministic_for_mixtral` serializes the same `Topology` twice
and asserts byte equality.

## Determinism

`emit_topology` and `build_device_graph` are deterministic by construction:

- The IR is walked in source order (`Graph::layers` is a `Vec`, preserved).
- `shard_role_for_param` is pure.
- `HashMap` is only used for output indexing (`LoweredGraph::declared`,
  `WeightShard`'s lookup); serialization order is `Vec`-driven, never
  hash-driven.

Determinism matters for the artifact-hashing path:
`blake3(serde_json::to_vec(&Plan)?) → artifacts/<plan_hash>/` only works if
the per-device graph + topology + weight shard are also reproducible.

## Responsibility split: `skein_emit` vs `skein_compile`

`skein_emit` builds the full per-device op graphs (declarations + matmul /
residual / norm edges, segmented at collective boundaries) and the
weight-shard / IO-manifest / topology metadata. It never runs Luminal's
search. `skein_compile` takes those graphs and runs
`cx.build_search_space::<R>()` + `cx.search(...)` per segment, then executes
through the selected `ComputeRuntime`.

| Concern                          | `skein_emit`                          | `skein_compile`                      |
|----------------------------------|---------------------------------------|--------------------------------------|
| `Graph::new()` + op wiring       | ✅ full per-segment op graphs          | —                                    |
| `cx.build_search_space()`        | —                                     | ✅                                    |
| `cx.search(runtime, budget)`     | —                                     | ✅ (`Native` or `Cuda` runtime)       |
| `write_weight_shard`             | ✅ single-file + multi-shard (index)   | —                                    |
| `IoManifest` / `topology.json`   | ✅ deterministic JSON                  | consumed at load                     |

The op graph implements RoPE, the causal mask, and top-k MoE routing (each
with a hand-computed test under `skein_emit/tests/`).

## Segment-per-collective architecture

`LoweredGraph` holds a sequence of [`Segment`]s — one Luminal graph per
collective-bracketed region of the device's forward pass — plus a
[`SequenceStep`] schedule the runtime walks per token. Each segment is a
complete `luminal::Graph` that `skein_compile` searches and compiles
independently; the runtime feeds tensors across segments by logical name
(see "Handoff naming" below) and issues the NCCL collective recorded in
the matching `SequenceStep::Collective`.

### Segmentation rule

For each device, enumerate the collective points (the device's
participation in the cluster-wide collective schedule) in execution
order. **A new segment starts whenever a collective fires.** N
collectives ⇒ N+1 segments per device.

The per-block collective ordering matches `topology::emit_topology`'s
emission order, so a cluster-wide topology snapshot lines up with the
per-device sequencing step-for-step:

1. EP dispatch (if `ep > 1` and the block has MoE).
2. TP `RingAllReduce` after attention's `o_proj` (if `tp > 1`).
3. EP combine (if `ep > 1` and the block has MoE).
4. TP `RingAllReduce` after the MoE/MLP's `down_proj` (if `tp > 1` and
   the block has FFN).

### Segment-count formulas (Mixtral 8x7B, 32 decoder blocks)

| Plan       | Collectives / block | Segments per device |
|-----------:|--------------------:|--------------------:|
| `tp=1 ep=1 pp=1` | 0 | 1 |
| `tp=2 ep=1` | 2 | 65 |
| `tp=2 ep=2` | 4 | 129 |

The `tp=1` case is uniform with the algorithm — zero collectives, one
segment containing the whole forward pass. There is **no** special-case
"keep one big graph" path; segmentation is the only path, `tp=1` just
falls through cleanly.

### Per-block wiring at `tp=2 ep=1`

- Segment A: `input_layernorm → q/k/v_proj → attention → o_proj` produces
  `block_N_attn_out`.
- Collective: `RingAllReduce(block_N_attn_out)`.
- Segment B: `residual(carry_pre_block_N + block_N_attn_out) →
  post_attention_layernorm → SwiGLU experts → down_proj` produces
  `block_N_ffn_out`.
- Collective: `RingAllReduce(block_N_ffn_out)`.
- Segment C (= block `N+1`'s Segment A): opens with the next residual
  `carry_post_attn_block_N + block_N_ffn_out`, then begins block `N+1`'s
  `input_layernorm`.

### Handoff naming

| Logical name | Origin | Role |
|--------------|--------|------|
| `input_tokens` | segment 0's only input | runtime feeds token IDs |
| `logits` | last segment's only output | runtime reads final logits |
| `block_N_attn_out` | upstream segment of TP attention AllReduce | collective tensor |
| `block_N_ffn_out` | upstream segment of TP MoE/MLP AllReduce | collective tensor |
| `block_N_moe_dispatch` | upstream segment of EP dispatch | AllToAll tensor |
| `block_N_moe_combine` | upstream segment of EP combine | AllToAll tensor |
| `carry_pre_block_N` | live across attn AllReduce | residual into post-AR residual add |
| `carry_post_attn_block_N` | live across MoE AllReduce | residual into post-MoE residual add |

The same logical name appears in the upstream segment's
`output_handoff`, the matching `SequenceStep::Collective.tensor` field
(for collective handoffs), and the downstream segment's `input_handoff`.
Test 2 of `tests/segmentation.rs` pins this invariant for every
collective in the schedule.

### Cross-segment data flow

The runtime — not Luminal — threads buffers across segments. Each
segment is a self-contained `luminal::Graph` whose Input ops correspond
to its `input_handoff` list; the runtime allocates the post-collective
buffer (the AllReduce'd value, the AllToAll'd shard) and routes it into
the next segment's matching Input op by logical name.

Carry tensors are simpler: the runtime keeps the buffer live across the
collective without modifying it, and feeds the same bytes into the next
segment's Input op. The `cut_segment_at_collective` helper in
`op_wiring::DeviceWiring` makes this explicit by re-declaring carries as
Input ops on the new segment's graph.

### What is NOT allowed

- **No collapsing segments.** The collective IS the segment boundary;
  collapsing breaks the runtime's collective-execution contract.
- **No reaching across segments inside Luminal.** Cross-segment data
  flow is exclusively via logical names through the runtime; never via
  Luminal graph edges.
- **No special-casing `tp=1`.** The segmentation algorithm is uniform;
  `tp=1` produces one segment because zero collectives fire, not
  because we shortcut anything.
