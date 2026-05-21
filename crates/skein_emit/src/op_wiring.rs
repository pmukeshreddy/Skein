//! Op wiring + multi-segment lowering.
//!
//! Lowers a device's forward pass into `Vec<Segment>` + `Vec<SequenceStep>`,
//! driven by [`crate::handoff::device_collective_points`]. Every collective
//! the device participates in becomes a segment boundary; the segments on
//! either side carry the matching [`HandoffTensor`] entries so the runtime
//! can route buffers across them.
//!
//! `tp = ep = pp = 1` produces exactly one segment per device containing the
//! whole forward pass.
//!
//! ## Per-block wiring at tp=2 ep=1
//!
//! Per Mixtral block, the cluster-wide topology emits two collectives:
//! `block_N_attn_out` (RingAllReduce after attention's `o_proj`) and
//! `block_N_ffn_out` (RingAllReduce after the MoE's `down_proj`). The
//! segmentation gives two break-points per block, and the wiring inside
//! each segment is the natural Mixtral op chain split at those points:
//!
//! - Segment A: `input_layernorm → q/k/v/o_proj` → `block_N_attn_out`.
//! - Collective: `RingAllReduce(block_N_attn_out)`.
//! - Segment B: `residual(carry + block_N_attn_out) →
//!   post_attention_layernorm → moe gate → SwiGLU experts → down_proj`
//!   → `block_N_ffn_out`.
//! - Collective: `RingAllReduce(block_N_ffn_out)`.
//! - The next segment opens with `residual(carry + block_N_ffn_out)`
//!   and starts block `N+1`'s `input_layernorm`.
//!
//! ## Carry tensors
//!
//! Residual adds fire after each TP AllReduce on the replicated
//! tensor, which means the pre-collective hidden state has to skip
//! across the collective. Two carries per block:
//!
//! - `carry_pre_block_N` — the input to block N's `input_layernorm`,
//!   needed for the residual after `block_N_attn_out`.
//! - `carry_post_attn_block_N` — the post-attn residual sum, needed for
//!   the residual after `block_N_ffn_out`.
//!
//! ## Per-op semantics
//!
//! The op graph implements the core Mixtral decoder math:
//!
//! - **RoPE** on Q/K ([`wire_attention_math`] via [`rope_tables`] /
//!   [`apply_rope`]), rotate-half / NeoX layout, with a `position_offset` so
//!   decode tokens are rotated by their absolute KV position.
//! - **Causal mask** added to the attention scores before the softmax
//!   ([`causal_bias`]).
//! - **Top-k expert routing** ([`top_k_route`]): the router softmax is
//!   restricted to the `meta.top_k` highest-logit experts and renormalized,
//!   so each expert is weighted by its true top-k gate probability.
//! - **KV cache** ([`attention_with_kv_cache`]): variable-length prefill and
//!   incremental decode against a stored K/V cache, with the shifted causal
//!   mask and absolute RoPE positions. Tested: decoding token-by-token while
//!   accumulating the cache reproduces the full-prefill output exactly.
//! - **EP routing** ([`moe_dispatch_combine`]): GShard capacity-based
//!   scatter/gather — `dispatch^T @ hidden` scatters tokens into per-expert
//!   capacity slots; `combine @ expert_out` gathers them back weighted by the
//!   renormalized top-k gates. Tested via the dispatch∘combine round-trip.
//!
//! ## Deferred lowering (TODO)
//!
//! The structural lowering — segment boundaries, collective ordering,
//! handoff naming, weight sharding — is complete and tested. The per-op math
//! above is implemented and unit-tested on `NativeRuntime`; the remaining work
//! is runtime *integration*, best validated against a GPU reference:
//!
//! - **TODO(ep-routing):** wire [`moe_dispatch_combine`] into the multi-segment
//!   schedule — lay the dispatched buffer out as `[ep, …]` for the AllToAll,
//!   run each rank's expert shard on its received tokens, and AllToAll the
//!   results back. The collective schedule + handoffs already reserve the
//!   dispatch/combine boundaries.
//! - **TODO(kv-runtime):** drive [`attention_with_kv_cache`] from the serving
//!   loop — allocate paged K/V via `PagedKVAllocator`, feed the per-step
//!   `past` offset and cache pages as graph inputs, and write `k_full`/`v_full`
//!   back after each step.

use std::collections::HashMap;

use luminal::hlir::Input;
use luminal::prelude::{
    DType, Expression, Graph as LuminalGraph, GraphTensor, NodeIndex, ShapeTracker,
};

use skein_cost::Cluster;
use skein_ir::ir::{Graph, LayerKind, ModelMeta, Param};
use skein_ir::plan::Plan;

use crate::error::EmitError;
use crate::graph_builder::{DeclaredTensor, shard_param_dims, to_luminal_dtype};
use crate::handoff::{
    CollectivePoint, EMBED_OUT, INPUT_TOKENS, LOGITS, carry_post_attn, carry_pre_block,
    collective_attn_out, collective_ffn_out, collective_moe_combine,
    device_collective_points,
};
use crate::segment::{HandoffTensor, Segment, SequenceStep};
use crate::shard_role::{ShardRole, shard_role_for_param};

/// Logical name of the runtime-fed absolute-position scalar (`[1]`, f32) the
/// cached-decode attention reads each step (for RoPE + the validity mask).
const POSITION_INPUT: &str = "position";
/// Fixed KV-cache capacity (slots) baked into every attention segment. The cache
/// is a static `[batch, KV_CACHE_CAP, n_kv*head_dim]` buffer — no dynamic /
/// zero-length dim (which the CUDA backend can't represent). The runtime feeds
/// the full buffer each step, writes the new token's K/V into slot `position`,
/// and the graph masks slots `> position`. Caps the max sequence length per
/// request; raise if longer contexts are needed (cost: O(CAP) attention/step).
/// Cache handoffs are named `kvcache_{k|v}_{block}` so
/// `skein_runtime::kv_cache::parse_kvcache_name` routes them to the per-layer cache.
pub const KV_CACHE_CAP: usize = 2048;

// ---------------------------------------------------------------------------
// Public entry: wire_segments
// ---------------------------------------------------------------------------

/// Lower a device's forward pass to a sequence of [`Segment`]s + a
/// [`SequenceStep`] schedule. One [`Segment`] per collective-bracketed
/// region; for a device participating in `N` collectives the result has
/// `N + 1` segments and `2N + 1` sequencing steps.
///
/// For `tp = ep = pp = 1` returns exactly one segment per device,
/// containing the whole forward pass.
pub fn wire_segments(
    plan: &Plan,
    cluster: &Cluster,
    ir: &Graph,
    device_idx: u32,
) -> Result<(Vec<Segment>, Vec<SequenceStep>), EmitError> {
    let points = device_collective_points(plan, ir, device_idx);
    let mut wiring = DeviceWiring::new(plan, cluster, ir, device_idx)?;

    let mut point_iter = points.iter();
    // `wire_initial` consumes the vocab-parallel embedding AllReduce (if any),
    // each block consumes its TP/EP collectives, and `wire_final` consumes the
    // logits AllGather (if any).
    wiring.wire_initial(&mut point_iter)?;
    for block in 0..ir.meta.num_layers {
        wiring.wire_block(block, &mut point_iter)?;
    }
    wiring.wire_final(&mut point_iter)?;

    // Every collective point must have been consumed.
    assert!(point_iter.next().is_none(), "collective points exhausted");
    Ok((wiring.segments, wiring.sequencing))
}

// ---------------------------------------------------------------------------
// DeviceWiring orchestrator
// ---------------------------------------------------------------------------

/// Live tensor inside the current segment's graph.
struct LiveTensor {
    tensor: GraphTensor,
    shape: Vec<usize>,
    dtype: skein_ir::types::Dtype,
}

/// Stateful orchestrator that walks the IR for one device, opens/closes
/// segments at collective boundaries, and accumulates the
/// `Vec<Segment>` + `Vec<SequenceStep>` `wire_segments` returns.
struct DeviceWiring<'a> {
    plan: &'a Plan,
    cluster: &'a Cluster,
    ir: &'a Graph,
    device_idx: u32,

    segments: Vec<Segment>,
    sequencing: Vec<SequenceStep>,

    // Current segment fields.
    cur_idx: usize,
    cur_cx: LuminalGraph,
    cur_declared: HashMap<String, DeclaredTensor>,
    cur_op_nodes: HashMap<String, NodeIndex>,
    cur_input_handoff: Vec<HandoffTensor>,
    // Cached-decode KV outputs (the new token's K/V) produced inside the
    // current segment; merged into its `output_handoff` when the segment closes
    // so the runtime appends them to the per-layer cache.
    cur_extra_outputs: Vec<HandoffTensor>,
    // The current segment's `position` Input, declared lazily on first use and
    // shared by every attention block in the segment (reset per segment).
    cur_position: Option<GraphTensor>,
    live: HashMap<String, LiveTensor>,

    // Static shape dimensions baked into every tensor — matches
    // `topology::emit_topology`'s `[batch, 1, hidden]` shape contract.
    batch: usize,
    seq: usize,
    hidden: usize,
    activation_dtype: skein_ir::types::Dtype,

    // When `SKEIN_DEBUG_TAPS` is set in the environment, insert extra named
    // `.output()` taps for layer-0 intermediates (post_input_ln, q/k proj,
    // attn_out, post_attn_ln, router_logits, moe_out) so the parity walk can
    // dump them op-by-op for HF bisection. Off by default: production graphs
    // (the KL gate) are byte-identical to the untapped lowering.
    debug_taps: bool,
}

impl<'a> DeviceWiring<'a> {
    fn new(
        plan: &'a Plan,
        cluster: &'a Cluster,
        ir: &'a Graph,
        device_idx: u32,
    ) -> Result<Self, EmitError> {
        if device_idx >= cluster.num_devices() {
            return Err(EmitError::DeviceOutOfRange {
                idx: device_idx,
                total: cluster.num_devices(),
            });
        }
        let batch = plan.batching.max_batch() as usize;
        let seq = 1usize; // decode-mode static seq; matches topology shape contract.
        let hidden = ir.meta.hidden;
        let activation_dtype = plan
            .dtype_map
            .per_layer
            .first()
            .map(|p| p.activation)
            .unwrap_or(skein_ir::types::Dtype::Bf16);

        Ok(Self {
            plan,
            cluster,
            ir,
            device_idx,
            segments: Vec::new(),
            sequencing: Vec::new(),
            cur_idx: 0,
            cur_cx: LuminalGraph::new(),
            cur_declared: HashMap::new(),
            cur_op_nodes: HashMap::new(),
            cur_input_handoff: Vec::new(),
            cur_extra_outputs: Vec::new(),
            cur_position: None,
            live: HashMap::new(),
            batch,
            seq,
            hidden,
            activation_dtype,
            debug_taps: std::env::var_os("SKEIN_DEBUG_TAPS").is_some(),
        })
    }

    /// Insert a named `.output()` tap so the parity walk can capture this
    /// tensor by name. Only fires under `SKEIN_DEBUG_TAPS` (see field docs).
    /// The `.output()` marks the node as a retained output, which suppresses
    /// fusion across it — acceptable for the debug dump, never used in the
    /// production graph.
    fn debug_tap(&mut self, name: &str, t: GraphTensor) {
        if !self.debug_taps {
            return;
        }
        // Cast to f32 before .output(): the runtime f32 read path mis-reads some
        // non-f32 op-output buffers (bf16 matmul output read raw → halved; f32
        // output mis-tagged bf16 → interleaved zeros). An explicit f32 cast
        // yields a buffer the read handles correctly, so taps are trustworthy.
        let tapped = t.cast(DType::F32).output();
        self.cur_op_nodes.insert(name.to_string(), tapped.id);
    }

    // -----------------------------------------------------------------------
    // Segment lifecycle
    // -----------------------------------------------------------------------

    /// Close the current segment with `output_handoff` and start a fresh
    /// one. Pushes an `ExecuteSegment` step into `sequencing`.
    fn close_current_segment(&mut self, mut output_handoff: Vec<HandoffTensor>) {
        let cx = std::mem::replace(&mut self.cur_cx, LuminalGraph::new());
        let declared = std::mem::take(&mut self.cur_declared);
        let op_nodes = std::mem::take(&mut self.cur_op_nodes);
        let input_handoff = std::mem::take(&mut self.cur_input_handoff);
        // Cached-decode K/V the runtime appends to the per-layer cache. These are
        // outputs of this segment regardless of where the segment boundary fell.
        output_handoff.append(&mut self.cur_extra_outputs);
        self.cur_position = None;

        self.sequencing.push(SequenceStep::ExecuteSegment {
            device_idx: self.device_idx,
            segment_idx: self.cur_idx,
        });
        self.segments.push(Segment {
            idx: self.cur_idx,
            graph: cx,
            declared,
            op_nodes,
            input_handoff,
            output_handoff,
        });
        self.cur_idx += 1;
        self.live.clear();
    }

    /// Open a fresh segment that takes `inputs` as its `input_handoff`.
    /// Each input becomes a `cx.named_tensor` in the new graph and lands
    /// in `self.live` under its logical name.
    fn open_new_segment(&mut self, inputs: Vec<HandoffSpec>) {
        for spec in inputs {
            let dims: Vec<Expression> = spec.shape.iter().copied().map(Expression::from).collect();
            let lum_dtype = to_luminal_dtype(spec.dtype);
            // `named_tensor` returns a handle with `DType::default()` (F32);
            // we must both (a) mutate the underlying Input op's dtype AND
            // (b) re-stamp the handle's dtype via `as_dtype` so downstream
            // ops type-check against the right value.
            let t = self
                .cur_cx
                .named_tensor(spec.logical_name.clone(), dims)
                .as_dtype(lum_dtype);
            self.cur_cx.get_op_mut::<Input>(t.id).dtype = lum_dtype;
            // `named_tensor` stamps `input_meta` with the F32 default; the
            // runtime narrows staged f32 host data to the input's dtype using
            // *this* map, so it must reflect the real dtype — otherwise a bf16
            // input keeps an f32 buffer that the bf16 codegen reads as
            // interleaved (every-other-zero) garbage.
            self.cur_cx
                .input_meta
                .insert(t.id, (spec.logical_name.clone(), lum_dtype));
            self.cur_input_handoff.push(HandoffTensor {
                logical_name: spec.logical_name.clone(),
                luminal_id: t.id,
                shape: spec.shape.clone(),
                dtype: spec.dtype,
            });
            self.live.insert(
                spec.logical_name,
                LiveTensor {
                    tensor: t,
                    shape: spec.shape,
                    dtype: spec.dtype,
                },
            );
        }
    }

    /// Close-then-open: finalize the current segment with the live tensors
    /// the next stage will need, push the collective step, and start a
    /// fresh segment whose `input_handoff` re-introduces those tensors.
    /// `collective_tensor` is the [`CollectivePoint::tensor`] that flows
    /// through the collective; `carries` are the additional live tensors
    /// that simply pass through and need re-introducing on the other side.
    fn cut_segment_at_collective(
        &mut self,
        point: &CollectivePoint,
        collective_tensor: &str,
        carries: &[&str],
    ) {
        // Build output_handoff = collective tensor + carries.
        let mut output_handoff = Vec::new();
        let coll_live = self
            .live
            .get(collective_tensor)
            .unwrap_or_else(|| panic!("collective tensor {collective_tensor} missing from live"));
        let coll_output = coll_live.tensor.output();
        // The collective tensor's real handoff dtype (e.g. f32 for materialized
        // matmul outputs), used both for the output handoff AND the
        // re-introduction below so the receiving segment's Input matches the
        // buffer the collective produced (point.dtype is the planner's original
        // activation dtype and can disagree).
        let coll_dtype = coll_live.dtype;
        output_handoff.push(HandoffTensor {
            logical_name: collective_tensor.to_string(),
            luminal_id: coll_output.id,
            shape: coll_live.shape.clone(),
            dtype: coll_dtype,
        });
        let mut carry_specs: Vec<HandoffSpec> = Vec::new();
        for c in carries {
            let live = self
                .live
                .get(*c)
                .unwrap_or_else(|| panic!("carry {c} missing from live"));
            // Materialize the carry as f32 for the cross-segment handoff. A
            // carry is a passed-through Input (the residual stream skipping the
            // collective); the runtime's f32 read path does not recognize a
            // bf16 passed-through Input as bf16 and reads its bytes as raw f32,
            // halving it — corrupting the residual on the far side. An explicit
            // f32 cast yields a buffer the read handles correctly; it is cast
            // back to the activation dtype where the residual consumes it.
            let carry_output = live.tensor.cast(DType::F32).output();
            output_handoff.push(HandoffTensor {
                logical_name: (*c).to_string(),
                luminal_id: carry_output.id,
                shape: live.shape.clone(),
                dtype: skein_ir::types::Dtype::F32,
            });
            carry_specs.push(HandoffSpec {
                logical_name: (*c).to_string(),
                shape: live.shape.clone(),
                dtype: skein_ir::types::Dtype::F32,
            });
        }

        self.close_current_segment(output_handoff);
        self.sequencing.push(SequenceStep::Collective {
            collective: point.kind,
            participants: point.participants.clone(),
            tensor: point.tensor.clone(),
            shape: point.shape.clone(),
            dtype: point.dtype,
        });

        // Open the next segment with the collective output + carries
        // re-introduced as fresh Input ops.
        let mut next_inputs = Vec::new();
        next_inputs.push(HandoffSpec {
            logical_name: point.tensor.clone(),
            shape: point.shape.clone(),
            dtype: coll_dtype,
        });
        next_inputs.extend(carry_specs);
        self.open_new_segment(next_inputs);
    }

    // -----------------------------------------------------------------------
    // Initial / final segments
    // -----------------------------------------------------------------------

    fn wire_initial<'b>(
        &mut self,
        point_iter: &mut impl Iterator<Item = &'b CollectivePoint>,
    ) -> Result<(), EmitError> {
        // Segment 0 declares `input_tokens` as its sole input handoff.
        let input_shape = vec![self.batch, self.seq];
        let tokens = self
            .cur_cx
            .named_tensor(INPUT_TOKENS, (self.batch, self.seq));
        let tokens = tokens.as_dtype(DType::Int);
        self.cur_cx.get_op_mut::<Input>(tokens.id).dtype = DType::Int;
        self.cur_cx
            .input_meta
            .insert(tokens.id, (INPUT_TOKENS.to_string(), DType::Int));
        self.cur_input_handoff.push(HandoffTensor {
            logical_name: INPUT_TOKENS.to_string(),
            luminal_id: tokens.id,
            shape: input_shape.clone(),
            dtype: skein_ir::types::Dtype::Int8, // marker — runtime treats it as token ids
        });
        self.cur_op_nodes
            .insert(INPUT_TOKENS.to_string(), tokens.id);

        let placement = skein_cost::cluster::Placement::from_plan(self.plan);
        let hidden_shape = vec![self.batch, self.seq, self.hidden];

        if placement.tp > 1 {
            // Vocab-parallel: this rank holds vocab rows
            // `[vocab_start, vocab_start + vocab_local)`. The masked lookup
            // zeros tokens outside the slice; the embedding AllReduce sums the
            // per-rank partials into the full embedding.
            let embed_w = self.weight("model.embed_tokens.weight")?;
            let vocab_local = self
                .cur_declared
                .get("model.embed_tokens.weight")
                .expect("embed table just declared")
                .shape[0];
            let tp_idx = (self.device_idx % (placement.tp * placement.ep)) / placement.ep;
            let vocab_start = tp_idx as usize * vocab_local;
            let local = vocab_parallel_embed(
                tokens,
                embed_w,
                vocab_start,
                vocab_local,
                self.batch,
                self.seq,
                self.hidden,
            );
            self.cur_op_nodes
                .insert("hidden_after_embed".to_string(), local.id);
            self.live.insert(
                EMBED_OUT.to_string(),
                LiveTensor {
                    tensor: local,
                    shape: hidden_shape.clone(),
                    dtype: self.activation_dtype,
                },
            );
            let point = point_iter
                .next()
                .expect("vocab-parallel embedding AllReduce point present");
            assert_eq!(point.tensor, EMBED_OUT);
            self.cut_segment_at_collective(point, EMBED_OUT, &[]);
            // After the AllReduce the full embedding feeds block 0.
            let reduced = self
                .live
                .get(EMBED_OUT)
                .expect("embed_out re-introduced after AllReduce")
                .tensor;
            self.live.insert(
                carry_pre_block(0),
                LiveTensor {
                    tensor: reduced,
                    shape: hidden_shape,
                    dtype: self.activation_dtype,
                },
            );
        } else {
            // Replicated embedding table → plain lookup, no collective.
            let embed_w = self.weight("model.embed_tokens.weight")?;
            let hidden = embedding_lookup(tokens, embed_w, self.batch, self.seq, self.hidden);
            self.cur_op_nodes
                .insert("hidden_after_embed".to_string(), hidden.id);
            self.live.insert(
                carry_pre_block(0),
                LiveTensor {
                    tensor: hidden,
                    shape: hidden_shape,
                    dtype: self.activation_dtype,
                },
            );
        }
        Ok(())
    }

    fn wire_final<'b>(
        &mut self,
        point_iter: &mut impl Iterator<Item = &'b CollectivePoint>,
    ) -> Result<(), EmitError> {
        let last_carry_name = carry_pre_block(self.ir.meta.num_layers);
        let final_hidden_live = self
            .live
            .remove(&last_carry_name)
            .expect("last carry tensor missing from live");
        let final_hidden = final_hidden_live.tensor;

        let final_norm_w = self.weight("model.norm.weight")?;
        let normed = rms_norm(final_hidden, final_norm_w, self.ir.meta.rms_norm_eps);

        let placement = skein_cost::cluster::Placement::from_plan(self.plan);
        let lm_head_w = self.weight("lm_head.weight")?;
        // With a vocab-sharded LM head this produces `[batch, seq, vocab/tp]`;
        // replicated, it produces the full `[batch, seq, vocab]`. Materialize as
        // f32: the logits are a bf16 *matmul* output read back through the
        // runtime handoff path (`get_data_f32`), which does not recognize a raw
        // matmul-output buffer as bf16 and would read the bytes as f32 — halving
        // the vocab. An explicit f32 cast fixes the read (and the final logits
        // should be f32 anyway, matching the HF reference).
        let logits_local = normed.matmul(lm_head_w.permute((1, 0))).cast(DType::F32);
        let full_shape = vec![self.batch, self.seq, self.ir.meta.vocab];

        if placement.tp > 1 {
            let vocab_local = self
                .cur_declared
                .get("lm_head.weight")
                .expect("lm_head just declared")
                .shape[0];
            self.live.insert(
                LOGITS.to_string(),
                LiveTensor {
                    tensor: logits_local,
                    shape: vec![self.batch, self.seq, vocab_local],
                    dtype: skein_ir::types::Dtype::F32,
                },
            );
            // AllGather concatenates the per-rank logit shards (rank-major,
            // matching the vocab-slice ordering) into the full logits.
            let point = point_iter
                .next()
                .expect("vocab-parallel logits AllGather point present");
            assert_eq!(point.tensor, LOGITS);
            self.cut_segment_at_collective(point, LOGITS, &[]);
            let gathered = self
                .live
                .get(LOGITS)
                .expect("logits re-introduced after AllGather")
                .tensor
                .output();
            self.cur_op_nodes.insert(LOGITS.to_string(), gathered.id);
            self.close_current_segment(vec![HandoffTensor {
                logical_name: LOGITS.to_string(),
                luminal_id: gathered.id,
                shape: full_shape,
                dtype: skein_ir::types::Dtype::F32,
            }]);
        } else {
            let logits = logits_local.output();
            self.cur_op_nodes.insert(LOGITS.to_string(), logits.id);
            self.close_current_segment(vec![HandoffTensor {
                logical_name: LOGITS.to_string(),
                luminal_id: logits.id,
                shape: full_shape,
                dtype: skein_ir::types::Dtype::F32,
            }]);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Per-block wiring
    // -----------------------------------------------------------------------

    fn wire_block<'b>(
        &mut self,
        block: usize,
        point_iter: &mut impl Iterator<Item = &'b CollectivePoint>,
    ) -> Result<(), EmitError> {
        let placement = skein_cost::cluster::Placement::from_plan(self.plan);
        let has_moe = self
            .ir
            .layers
            .iter()
            .any(|l| l.block_idx == Some(block) && matches!(l.kind, LayerKind::Moe(_)));
        let has_mlp = self
            .ir
            .layers
            .iter()
            .any(|l| l.block_idx == Some(block) && matches!(l.kind, LayerKind::Mlp(_)));

        // Pull the input carry — output of the previous block's tail. Under EP
        // the hidden state is replicated across the EP group, so there is no
        // dispatch collective before MoE (dense expert parallel; the combine
        // all-reduce after MoE sums the per-rank partial expert outputs).
        let carry_in = carry_pre_block(block);

        // (1) Pre-attn ops: input_layernorm + q/k/v + attn → block_N_attn_out.
        let hidden = self
            .live
            .get(&carry_in)
            .expect("pre-block carry missing")
            .tensor;
        if block == 0 {
            self.debug_tap("dbg_l0_embed_in", hidden);
        }
        let norm1_w = self.weight(&format!("model.layers.{block}.input_layernorm.weight"))?;
        let normed_1 = rms_norm(hidden, norm1_w, self.ir.meta.rms_norm_eps);
        if block == 0 {
            self.debug_tap("dbg_l0_post_input_ln", normed_1);
        }
        let attn_out = self.wire_block_attention(block, normed_1)?;

        // Record attn_out as block_N_attn_out in live (will either be the
        // collective tensor or just an internal name when tp=1). When this is a
        // cross-segment collective handoff (tp>1), materialize it as f32: the
        // o_proj output is a bf16 *matmul* buffer, and the runtime handoff read
        // (`get_data_f32`) does not recognize a raw matmul-output buffer as
        // bf16, so it reads the raw bytes as f32 and HALVES the tensor —
        // corrupting the AllReduce. An explicit f32 cast produces a buffer the
        // read handles correctly; it is cast back to the activation dtype after
        // the collective so the residual stays bf16.
        let attn_name = collective_attn_out(block);
        let (attn_live, attn_handoff_dtype) = if placement.tp > 1 {
            (attn_out.cast(DType::F32), skein_ir::types::Dtype::F32)
        } else {
            (attn_out, self.activation_dtype)
        };
        self.live.insert(
            attn_name.clone(),
            LiveTensor {
                tensor: attn_live,
                shape: vec![self.batch, self.seq, self.hidden],
                dtype: attn_handoff_dtype,
            },
        );

        // (3) TP attn AllReduce.
        if placement.tp > 1 {
            let point = point_iter.next().expect("tp attn point not present");
            assert_eq!(point.tensor, attn_name);
            self.cut_segment_at_collective(point, &attn_name, &[&carry_in]);
        }

        // (4) Residual after attn (in the current segment, post-collective).
        let attn_full_raw = self.live.get(&attn_name).expect("attn out missing").tensor;
        // Cast the f32 collective handoff back to the activation dtype for the
        // bf16 residual add (no-op when tp==1).
        let attn_full = if placement.tp > 1 {
            attn_full_raw.cast(to_luminal_dtype(self.activation_dtype))
        } else {
            attn_full_raw
        };
        if block == 0 {
            self.debug_tap("dbg_l0_attn_out", attn_full);
        }
        let carry_in_raw = self.live.get(&carry_in).expect("carry missing").tensor;
        // When tp>1 the carry crossed the attn-collective cut and was carried as
        // f32 (so the runtime read does not halve it); cast back to the
        // activation dtype for the bf16 residual add.
        let carry_in_live = if placement.tp > 1 {
            carry_in_raw.cast(to_luminal_dtype(self.activation_dtype))
        } else {
            carry_in_raw
        };
        let after_attn = carry_in_live + attn_full;
        let carry_post = carry_post_attn(block);
        self.live.insert(
            carry_post.clone(),
            LiveTensor {
                tensor: after_attn,
                shape: vec![self.batch, self.seq, self.hidden],
                dtype: self.activation_dtype,
            },
        );

        // (5) post_attention_layernorm + MoE → block_N_ffn_out.
        let norm2_w = self.weight(&format!(
            "model.layers.{block}.post_attention_layernorm.weight"
        ))?;
        let normed_2 = rms_norm(after_attn, norm2_w, self.ir.meta.rms_norm_eps);
        if block == 0 {
            self.debug_tap("dbg_l0_post_attn_ln", normed_2);
        }
        let ffn_out = if has_moe {
            self.wire_block_moe(block, normed_2)?
        } else if has_mlp {
            self.wire_block_mlp(block, normed_2)?
        } else {
            // No FFN layer — just pass through. Defensive; Mixtral has
            // MoE in every block.
            normed_2
        };
        // Same matmul-output handoff fix as attn: the MoE down-proj output is a
        // bf16 matmul buffer; materialize it as f32 when it crosses a collective
        // (EP combine or TP ffn AllReduce) so the runtime read does not halve it.
        let ffn_is_collective = (placement.tp > 1 && (has_moe || has_mlp)) || (placement.ep > 1 && has_moe);
        let ffn_name = collective_ffn_out(block);
        let (ffn_live, ffn_handoff_dtype) = if ffn_is_collective {
            (ffn_out.cast(DType::F32), skein_ir::types::Dtype::F32)
        } else {
            (ffn_out, self.activation_dtype)
        };
        self.live.insert(
            ffn_name.clone(),
            LiveTensor {
                tensor: ffn_live,
                shape: vec![self.batch, self.seq, self.hidden],
                dtype: ffn_handoff_dtype,
            },
        );

        // (6) EP combine after MoE: all-reduce (sum) the per-rank partial
        // expert outputs across the EP group. `ffn_out` is this rank's sum over
        // the experts it owns; summing across the EP group reconstructs the
        // full MoE output (dense expert parallel).
        if placement.ep > 1 && has_moe {
            let point = point_iter
                .next()
                .expect("ep combine point not present in iter");
            assert_eq!(point.tensor, collective_moe_combine(block));
            let combine_name = collective_moe_combine(block);
            self.live.insert(
                combine_name.clone(),
                LiveTensor {
                    tensor: ffn_live,
                    shape: vec![self.batch, self.seq, self.hidden],
                    dtype: ffn_handoff_dtype,
                },
            );
            self.cut_segment_at_collective(point, &combine_name, &[&carry_post]);
            // After the all-reduce the combined tensor is the full MoE output;
            // re-register it under ffn_name for the downstream TP ffn-AllReduce
            // (when tp > 1) and the post-block residual.
            let combined = self
                .live
                .get(&combine_name)
                .expect("combined missing")
                .tensor;
            self.live.insert(
                ffn_name.clone(),
                LiveTensor {
                    tensor: combined,
                    shape: vec![self.batch, self.seq, self.hidden],
                    dtype: ffn_handoff_dtype,
                },
            );
        }

        // (7) TP MoE AllReduce.
        if placement.tp > 1 && (has_moe || has_mlp) {
            let point = point_iter.next().expect("tp ffn point not present");
            assert_eq!(point.tensor, ffn_name);
            self.cut_segment_at_collective(point, &ffn_name, &[&carry_post]);
        }

        // (8) Final residual + emit next-block carry.
        let ffn_full_raw = self.live.get(&ffn_name).expect("ffn out missing").tensor;
        // Cast the f32 collective handoff back to the activation dtype for the
        // bf16 residual add (no-op when ffn never crossed a collective).
        let ffn_full = if ffn_is_collective {
            ffn_full_raw.cast(to_luminal_dtype(self.activation_dtype))
        } else {
            ffn_full_raw
        };
        if block == 0 {
            self.debug_tap("dbg_l0_moe_out", ffn_full);
        }
        let carry_post_raw = self
            .live
            .get(&carry_post)
            .expect("post-attn carry missing")
            .tensor;
        // carry_post crossed the ffn (and/or EP) collective cut as f32; cast
        // back to the activation dtype for the bf16 residual add.
        let carry_post_live = if ffn_is_collective {
            carry_post_raw.cast(to_luminal_dtype(self.activation_dtype))
        } else {
            carry_post_raw
        };
        let after_block = carry_post_live + ffn_full;
        let next_carry = carry_pre_block(block + 1);
        self.live.insert(
            next_carry,
            LiveTensor {
                tensor: after_block,
                shape: vec![self.batch, self.seq, self.hidden],
                dtype: self.activation_dtype,
            },
        );
        let after_block_output = after_block.output();
        self.cur_op_nodes
            .insert(format!("hidden_after_block_{block}"), after_block_output.id);

        // Stale live entries are cleared on segment close; for ops within
        // the current segment we leave them in place (they're harmless).
        Ok(())
    }

    fn wire_block_attention(
        &mut self,
        block: usize,
        normed: GraphTensor,
    ) -> Result<GraphTensor, EmitError> {
        let head_dim = self.ir.meta.head_dim;
        let q_w = self.weight(&format!("model.layers.{block}.self_attn.q_proj.weight"))?;
        let k_w = self.weight(&format!("model.layers.{block}.self_attn.k_proj.weight"))?;
        let v_w = self.weight(&format!("model.layers.{block}.self_attn.v_proj.weight"))?;
        let o_w = self.weight(&format!("model.layers.{block}.self_attn.o_proj.weight"))?;
        // Local head counts from sharded Q/K weight row counts.
        let q_decl = self
            .cur_declared
            .get(&format!("model.layers.{block}.self_attn.q_proj.weight"))
            .expect("q_proj just declared");
        let k_decl = self
            .cur_declared
            .get(&format!("model.layers.{block}.self_attn.k_proj.weight"))
            .expect("k_proj just declared");
        let n_heads_local = q_decl.shape[0] / head_dim;
        let n_kv_heads_local = k_decl.shape[0] / head_dim;
        let kv_dim = n_kv_heads_local * head_dim;
        let batch = self.batch;
        let act = self.activation_dtype;
        let lum_act = to_luminal_dtype(act);
        let rope_theta = self.ir.meta.rope_theta;

        // Cached decode: project the current token's q/k/v (seq = 1) and attend
        // over a fixed-capacity KV cache (slots 0..position) plus the current
        // token. RoPE/scores/softmax/Attn·V run in fp32 — q/k/v, the cache, and
        // `position` are all f32 — both for numerical parity with HF's fp32
        // softmax and so the cache round-trips losslessly through the runtime's
        // f32 cache. Only the final attention output is cast back to the
        // activation dtype for `o_proj`. This replaces the old stateless `seq=1`,
        // position-0 lowering.
        let q = normed.matmul(q_w.permute((1, 0))).cast(DType::F32);
        let k_new = normed.matmul(k_w.permute((1, 0))).cast(DType::F32);
        let v_new = normed.matmul(v_w.permute((1, 0))).cast(DType::F32);
        if block == 0 {
            self.debug_tap("dbg_l0_q_proj", q);
            self.debug_tap("dbg_l0_k_proj", k_new);
            self.debug_tap("dbg_l0_v_proj", v_new);
        }

        // Runtime-fed fixed-capacity cache [batch, KV_CACHE_CAP, kv_dim] (static,
        // never empty/null) + the absolute `position`. `SegmentRunner` feeds the
        // full cache buffer each step and writes the new K/V into slot `position`.
        let cache_dims = || {
            vec![
                Expression::from(batch),
                Expression::from(KV_CACHE_CAP),
                Expression::from(kv_dim),
            ]
        };
        let k_cache = self.runtime_input(
            &format!("kvcache_k_{block}"),
            cache_dims(),
            vec![batch, KV_CACHE_CAP, kv_dim],
            DType::F32,
        );
        let v_cache = self.runtime_input(
            &format!("kvcache_v_{block}"),
            cache_dims(),
            vec![batch, KV_CACHE_CAP, kv_dim],
            DType::F32,
        );
        let position = self.position_input();

        let (attn, k_store, v_store) = attention_fixed_cache(
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            position,
            n_heads_local,
            n_kv_heads_local,
            head_dim,
            KV_CACHE_CAP,
            rope_theta,
        );

        // Hand the new (rotated) key + value back to the runtime to write into
        // slot `position` of this layer's fixed cache for the next step.
        let k_out = k_store.output();
        let v_out = v_store.output();
        self.cur_extra_outputs.push(HandoffTensor {
            logical_name: format!("kvcache_k_{block}"),
            luminal_id: k_out.id,
            shape: vec![batch, 1, kv_dim],
            dtype: act,
        });
        self.cur_extra_outputs.push(HandoffTensor {
            logical_name: format!("kvcache_v_{block}"),
            luminal_id: v_out.id,
            shape: vec![batch, 1, kv_dim],
            dtype: act,
        });

        let attn = attn.cast(lum_act);
        Ok(attn.matmul(o_w.permute((1, 0))))
    }

    /// Declare a runtime-fed Input in the current segment: registers it as an
    /// `input_handoff` (so `SegmentRunner` feeds it by logical name) and in
    /// `op_nodes`. `dims` may carry a dynamic-dim char; `shape_repr` is the
    /// concrete representative shape used for the compile-time search staging.
    fn runtime_input(
        &mut self,
        name: &str,
        dims: Vec<Expression>,
        shape_repr: Vec<usize>,
        dtype: DType,
    ) -> GraphTensor {
        let t = self
            .cur_cx
            .named_tensor(name.to_string(), dims)
            .as_dtype(dtype);
        self.cur_cx.get_op_mut::<Input>(t.id).dtype = dtype;
        self.cur_cx
            .input_meta
            .insert(t.id, (name.to_string(), dtype));
        self.cur_input_handoff.push(HandoffTensor {
            logical_name: name.to_string(),
            luminal_id: t.id,
            shape: shape_repr,
            // Marker only: the runtime feeds KV cache + position as raw f32, and
            // `segment_input_zero_bytes` sizes their search buffers by name.
            dtype: self.activation_dtype,
        });
        self.cur_op_nodes.insert(name.to_string(), t.id);
        t
    }

    /// The current segment's `position` scalar Input (`[1]`, f32), declared once
    /// and reused by every attention block in the same segment.
    fn position_input(&mut self) -> GraphTensor {
        if let Some(p) = self.cur_position {
            return p;
        }
        let p = self.runtime_input(
            POSITION_INPUT,
            vec![Expression::from(1usize)],
            vec![1],
            DType::F32,
        );
        self.cur_position = Some(p);
        p
    }

    /// MoE feed-forward for one block, with top-k expert routing.
    ///
    /// The router gate produces per-expert logits; [`top_k_route`] restricts
    /// the softmax to the `meta.top_k` highest-logit experts (top-2 for
    /// Mixtral) and renormalizes over them, so each expert is weighted by its
    /// true top-k gate probability (0 for non-selected experts).
    ///
    /// Compute note: every owned expert is still *evaluated* and then weighted
    /// (non-selected weights are ~0, so the **output** matches true top-k
    /// routing). Under expert parallelism (`ep > 1`) each device owns a
    /// disjoint subset of experts (see [`Self::owns_expert`]) and therefore
    /// produces a *partial* MoE output; the EP combine all-reduce in
    /// [`Self::wire_block`] sums these partials across the EP group into the
    /// full output. This is the *dense* expert-parallel scheme. Sparse compute
    /// — routing only each token's selected experts via the GShard
    /// dispatch/combine in [`moe_dispatch_combine`] — is a future throughput
    /// optimization, not a correctness requirement.
    fn wire_block_moe(
        &mut self,
        block: usize,
        normed: GraphTensor,
    ) -> Result<GraphTensor, EmitError> {
        let n_experts = self
            .ir
            .meta
            .num_experts
            .expect("wire_block_moe called on non-MoE block");
        let gate_w = self.weight(&format!(
            "model.layers.{block}.block_sparse_moe.gate.weight"
        ))?;
        let routing_logits = normed.matmul(gate_w.permute((1, 0)));
        if block == 0 {
            self.debug_tap("dbg_l0_router_logits", routing_logits);
        }
        let n = normed.dims().len();
        let top_k = self.ir.meta.top_k.unwrap_or(n_experts).clamp(1, n_experts);
        let routing_probs = top_k_route(routing_logits, top_k, n_experts, n - 1);

        let mut acc: Option<GraphTensor> = None;
        for e in 0..n_experts {
            // Only declare experts this device owns (under EP). For tp-only
            // plans every expert is replicated. The shard role check
            // routes correctly.
            let owns = self.owns_expert(block, e);
            if !owns {
                continue;
            }
            let w1 = self.weight(&format!(
                "model.layers.{block}.block_sparse_moe.experts.{e}.w1.weight"
            ))?;
            let w2 = self.weight(&format!(
                "model.layers.{block}.block_sparse_moe.experts.{e}.w2.weight"
            ))?;
            let w3 = self.weight(&format!(
                "model.layers.{block}.block_sparse_moe.experts.{e}.w3.weight"
            ))?;
            let gate_val = normed.matmul(w1.permute((1, 0))).silu();
            let up_val = normed.matmul(w3.permute((1, 0)));
            let down = (gate_val * up_val).matmul(w2.permute((1, 0)));
            let mut prob_e = routing_probs.slice((.., .., e..e + 1));
            prob_e.shape.expand(down.dims());
            let weighted = down * prob_e;
            acc = Some(match acc {
                None => weighted,
                Some(a) => a + weighted,
            });
        }
        Ok(acc.expect("at least one expert owned"))
    }

    fn wire_block_mlp(
        &mut self,
        block: usize,
        normed: GraphTensor,
    ) -> Result<GraphTensor, EmitError> {
        // Simple SwiGLU MLP for non-MoE blocks. Mixtral has none, but
        // included for completeness so a future model with MLP blocks
        // doesn't trip the wiring.
        let w1 = self.weight(&format!("model.layers.{block}.mlp.gate_proj.weight"))?;
        let w2 = self.weight(&format!("model.layers.{block}.mlp.down_proj.weight"))?;
        let w3 = self.weight(&format!("model.layers.{block}.mlp.up_proj.weight"))?;
        let gate_val = normed.matmul(w1.permute((1, 0))).silu();
        let up_val = normed.matmul(w3.permute((1, 0)));
        Ok((gate_val * up_val).matmul(w2.permute((1, 0))))
    }

    fn owns_expert(&self, block: usize, expert_idx: usize) -> bool {
        // Find the MoE param matching this expert and check shard role.
        // Conservative: if shard_role_for_param says ExpertElsewhere we
        // skip; otherwise we own it.
        let name = format!("model.layers.{block}.block_sparse_moe.experts.{expert_idx}.w1.weight");
        for layer in &self.ir.layers {
            if layer.block_idx != Some(block) {
                continue;
            }
            for param in &layer.params {
                if param.name == name {
                    let role = shard_role_for_param(
                        self.plan,
                        self.cluster,
                        self.ir,
                        self.device_idx,
                        layer,
                        param,
                    );
                    return !matches!(role, ShardRole::ExpertElsewhere { .. });
                }
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Weight declaration + handle lookup
    // -----------------------------------------------------------------------

    fn weight(&mut self, name: &str) -> Result<GraphTensor, EmitError> {
        if let Some(d) = self.cur_declared.get(name) {
            return Ok(handle_for_declared(&mut self.cur_cx, d));
        }
        // Look up the param in IR.
        for layer in &self.ir.layers {
            for param in &layer.params {
                if param.name == name {
                    let role = shard_role_for_param(
                        self.plan,
                        self.cluster,
                        self.ir,
                        self.device_idx,
                        layer,
                        param,
                    );
                    if matches!(
                        role,
                        ShardRole::PipelineStageElsewhere
                            | ShardRole::ExpertElsewhere { .. }
                            | ShardRole::NoParams
                    ) {
                        return Err(EmitError::UnknownParamPattern { name: name.into() });
                    }
                    let d = declare_param_into(
                        &mut self.cur_cx,
                        self.plan,
                        layer.block_idx,
                        param,
                        &role,
                    )?;
                    let handle = handle_for_declared(&mut self.cur_cx, &d);
                    self.cur_declared.insert(name.to_string(), d);
                    return Ok(handle);
                }
            }
        }
        Err(EmitError::UnknownParamPattern { name: name.into() })
    }

}

/// Shape + dtype + logical name spec for a tensor to introduce as an
/// `Input` op when opening a new segment.
struct HandoffSpec {
    logical_name: String,
    shape: Vec<usize>,
    dtype: skein_ir::types::Dtype,
}

// ---------------------------------------------------------------------------
// Per-op helpers. `embedding_lookup` takes explicit static dims rather than
// inferring them from the GraphTensor, because segments use static shapes.
// ---------------------------------------------------------------------------

/// Output-correct top-k routing weights: a softmax restricted to the `k`
/// highest-logit experts along `axis`, renormalized over the selected set.
/// `k >= n_experts` degenerates to a plain softmax.
///
/// `softmax(top-k logits)` is mathematically equal to renormalizing the
/// top-k entries of the full softmax, which is exactly Mixtral's router.
///
/// Note: exact ties at the k-th logit would select more than `k` experts;
/// for real bf16 gate logits this does not occur in practice.
pub fn top_k_route(logits: GraphTensor, k: usize, n_experts: usize, axis: usize) -> GraphTensor {
    if k >= n_experts {
        return logits.softmax(axis);
    }
    // A logit floor that softmax maps to ~0 once subtracted from the kept set.
    const NEG: f32 = -1e30;
    // Comparison masks come back as F32; cast the additive terms back to the
    // logits' dtype (e.g. Bf16) before combining.
    let dt = logits.dtype;
    // Find the k-th largest logit along `axis` by iteratively masking out the
    // running maxima, then keep every logit >= that threshold.
    let mut threshold = logits.max(axis).expand_dim(axis, n_experts);
    let mut removed = logits;
    for _ in 1..k {
        let is_max = removed.ge(threshold).cast(DType::F32);
        removed += (is_max * NEG).cast(dt);
        threshold = removed.max(axis).expand_dim(axis, n_experts);
    }
    let keep = logits.ge(threshold).cast(DType::F32);
    let masked = logits + ((1.0 - keep) * NEG).cast(dt);
    masked.softmax(axis)
}

/// GShard-style capacity routing tensors for expert parallelism. Given the
/// router `gate_logits` `[tokens, n_experts]`, returns `(dispatch, combine)`
/// each shaped `[tokens, n_experts * capacity]`:
///
/// - `dispatched = dispatch.permute((1, 0)).matmul(hidden)` →
///   `[n_experts * capacity, hidden]` scatters each token into a capacity
///   slot of each of its top-k experts.
/// - `out = combine.matmul(expert_out)` (with `expert_out`
///   `[n_experts * capacity, hidden]`) → `[tokens, hidden]` gathers the
///   expert outputs back, weighted by the renormalized top-k gate
///   probabilities.
///
/// Capacity slots are assigned globally per expert via an exclusive cumulative
/// count; tokens past an expert's `capacity` are dropped (their slots never
/// fill), matching the standard capacity-factor behaviour. This is the
/// per-EP-rank math: across `ep` devices the `[n_experts*capacity, hidden]`
/// buffer is reshaped to `[ep, …]` and AllToAll'd so each rank receives the
/// tokens destined for its expert shard.
pub fn moe_dispatch_combine(
    gate_logits: GraphTensor,
    top_k: usize,
    n_experts: usize,
    capacity: usize,
) -> (GraphTensor, GraphTensor) {
    let probs = gate_logits.softmax(1); // [T, E]
    let topk_idx = gate_logits.topk_indexes(top_k, 1); // [T, K] (expert ids)
    let topk_gate = probs.gather_elements(topk_idx, 1); // [T, K]
    // Renormalize the gate weights over just the selected experts.
    let gate_norm = topk_gate / topk_gate.sum(1).expand_dim(1, top_k); // [T, K]

    // Flatten the (token, k) assignments into one axis N = T*K, ordered token-
    // major, so capacity slots are assigned in a stable global order.
    let assign_expert = topk_idx.merge_dims(0, 1).cast(DType::F32); // [N]
    let assign_gate = gate_norm.merge_dims(0, 1); // [N]
    let n = assign_expert.dims()[0];
    let cx = gate_logits.graph();

    // Expert one-hot [N, E].
    let experts = cx.arange(n_experts).cast(DType::F32); // [E]
    let onehot_e = assign_expert
        .expand_dim(1, n_experts)
        .eq(experts.expand_dim(0, n))
        .cast(DType::F32); // [N, E]
    // Exclusive cumulative count per expert → the slot each assignment lands in.
    let pos = ((onehot_e.cumsum(0) - onehot_e) * onehot_e).sum(1); // [N]
    // Capacity one-hot [N, C]; a position >= capacity yields an all-zero row
    // (the token is dropped).
    let slots = cx.arange(capacity).cast(DType::F32); // [C]
    let onehot_c = pos
        .expand_dim(1, capacity)
        .eq(slots.expand_dim(0, n))
        .cast(DType::F32); // [N, C]

    // Per-assignment dispatch [N, E, C] = expert one-hot ⊗ capacity one-hot.
    let disp_assign = onehot_e.expand_dim(2, capacity) * onehot_c.expand_dim(1, n_experts);
    let comb_assign = disp_assign * assign_gate.expand_dim(1, n_experts).expand_dim(2, capacity);

    // [N, E, C] → [N, E*C] → [T, K, E*C] → sum over the K assignments.
    let dispatch = disp_assign.merge_dims(1, 2).split_dims(0, top_k).sum(1);
    let combine = comb_assign.merge_dims(1, 2).split_dims(0, top_k).sum(1);
    (dispatch, combine)
}

/// `x / sqrt(mean(x²) + eps) * weight` — i.e. RmsNorm.
pub fn rms_norm(input: GraphTensor, weight: GraphTensor, eps: f32) -> GraphTensor {
    let last = input.shape.last_axis();
    let normed = input.std_norm(last, eps);
    let dims = input.dims();
    let n = dims.len();
    normed * weight.expand_lhs(&dims[..n - 1])
}

/// Token-id `[batch, seq]` → embedded `[batch, seq, hidden]`. Static dims
/// are passed in explicitly because segments use static shapes rather than
/// dynamic `'b'` / `'s'` dimensions.
pub fn embedding_lookup(
    tokens: GraphTensor,
    embed_weight: GraphTensor,
    batch: usize,
    seq: usize,
    hidden: usize,
) -> GraphTensor {
    let cols = tokens
        .graph()
        .arange(hidden)
        .expand_dim(0, batch)
        .expand_dim(1, seq);
    embed_weight.gather((tokens * hidden).expand_dim(2, hidden) + cols)
}

/// Vocab-parallel embedding lookup. `local_table` is this TP rank's vocab
/// slice `[vocab_local, hidden]` covering global ids
/// `[vocab_start, vocab_start + vocab_local)`. A token outside this rank's
/// slice contributes a zero row; an `AllReduce(sum)` across the TP group then
/// reconstructs the full embedding, since each id is owned by exactly one
/// rank. Returns `[batch, seq, hidden]` in the table's dtype.
pub fn vocab_parallel_embed(
    tokens: GraphTensor,
    local_table: GraphTensor,
    vocab_start: usize,
    vocab_local: usize,
    batch: usize,
    seq: usize,
    hidden: usize,
) -> GraphTensor {
    let cols = tokens
        .graph()
        .arange(hidden)
        .expand_dim(0, batch)
        .expand_dim(1, seq);

    // local_id = token - vocab_start, clamped to [0, vocab_local-1] so the
    // gather is always in-bounds; out-of-range tokens are masked to 0 below.
    let token_f = tokens.cast(DType::F32);
    let local_clamped = (token_f - vocab_start as f32)
        .clip(0.0, (vocab_local.saturating_sub(1)) as f32)
        .cast(DType::Int);
    let flat = (local_clamped * hidden).expand_dim(2, hidden) + cols;
    let embeds = local_table.gather(flat);

    // in_range = (token >= vocab_start) & (token < vocab_start + vocab_local).
    let lo = token_f * 0.0 + vocab_start as f32;
    let hi = token_f * 0.0 + (vocab_start + vocab_local) as f32;
    let in_range = token_f.ge(lo).cast(DType::F32) * token_f.lt(hi).cast(DType::F32);
    let mask = in_range.expand_dim(2, hidden).cast(embeds.dtype);
    embeds * mask
}

/// Attention against a paged KV cache, for variable-length prefill and
/// incremental decode.
///
/// `q`/`k_new`/`v_new` are the current chunk's projections
/// (`[batch, seq, n_heads*head_dim]` and `[batch, seq, n_kv_heads*head_dim]`);
/// `k_cache`/`v_cache` hold the previously stored keys/values
/// (`[batch, past, n_kv_heads*head_dim]`, already RoPE-rotated). RoPE is
/// applied to `q`/`k_new` at absolute positions `past..past+seq`; the new
/// keys/values are appended to the cache and attention runs over all
/// `past+seq` keys under a causal mask where query `past+i` attends keys
/// `0..=past+i`. Returns `(attn_out [batch, seq, n_heads*head_dim], k_full,
/// v_full)` — the runtime appends `k_full`/`v_full` to the paged cache.
///
/// `past == 0` is the prefill case (no cache concat); `seq == 1`, `past > 0`
/// is a decode step.
#[allow(clippy::too_many_arguments)]
pub fn attention_with_kv_cache(
    q: GraphTensor,
    k_new: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    past: usize,
    rope_theta: f32,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let kv_groups = n_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let batch = q.dims()[0];
    let seq = q.dims()[1];

    // Head-split [batch, seq, heads, head_dim].
    let q = q.split_dims(2, head_dim);
    let k_new = k_new.split_dims(2, head_dim);
    let v_new = v_new.split_dims(2, head_dim);

    // RoPE on the new q/k at absolute positions past..past+seq.
    let (cos_t, sin_t) = rope_tables(q.graph(), seq, head_dim, rope_theta, past);
    let q = apply_rope(q, cos_t, sin_t, head_dim);
    let k_new = apply_rope(k_new, cos_t, sin_t, head_dim);

    // Append the new (rotated) k / (raw) v to the cache. `past == 0` skips the
    // concat to avoid zero-length cache tensors on the prefill step.
    let (k_full_hs, v_full_hs) = if past == 0 {
        (k_new, v_new)
    } else {
        let k_cache_hs = k_cache.split_dims(2, head_dim);
        let v_cache_hs = v_cache.split_dims(2, head_dim);
        (
            k_cache_hs.concat_along(k_new, 1),
            v_cache_hs.concat_along(v_new, 1),
        )
    };
    let total = past + seq; // total key positions = past + current chunk

    // GQA attention. q:[b,seq,Hkv,groups,d]->[b,Hkv,groups,seq,d];
    // k_full:[b,T,Hkv,d]->[b,Hkv,groups,d,T]; v_full:->[b,Hkv,groups,T,d].
    let q = q.split_dims(2, kv_groups).permute((0, 2, 3, 1, 4));
    let k_full = k_full_hs.permute((0, 2, 3, 1)).expand_dim(2, kv_groups);
    let v_full = v_full_hs.permute((0, 2, 1, 3)).expand_dim(2, kv_groups);

    let scores = q.matmul(k_full) * scale; // [b, Hkv, groups, seq, T]
    // Causal mask: query at absolute position `past+i` attends key `j` iff
    // `j <= past+i`. That is the last `seq` rows of the `[T, T]` lower triangle
    // (`tril`), reusing the tested mask construction.
    let cx = scores.graph();
    let tri = cx.tril(total, 0).cast(DType::F32); // [T, T] lower triangle
    let allowed = if past == 0 {
        tri // total == seq here → [seq, T]
    } else {
        tri.slice((past.., ..)) // last `seq` rows → [seq, T]
    };
    let bias = ((allowed - 1.0) * 1.0e9)
        .expand_dim(0, batch)
        .expand_dim(1, n_kv_heads)
        .expand_dim(2, kv_groups)
        .cast(scores.dtype);
    let weights = (scores + bias).softmax(4);
    let attn = weights.matmul(v_full); // [b, Hkv, groups, seq, d]

    // [b, Hkv, groups, seq, d] -> [b, seq, Hkv, groups, d] -> [b, seq, n_heads*d].
    // `merge_dims` composes with the permute's strides (a plain `ShapeTracker`
    // reassignment would misread the strided data when groups/heads > 1). The
    // caller applies `o_proj` to this `[batch, seq, n_heads*head_dim]` output.
    let attn = attn
        .permute((0, 3, 1, 2, 4))
        .merge_dims(3, 4) // [b, seq, Hkv, groups*d]
        .merge_dims(2, 3); // [b, seq, Hkv*groups*d] = [b, seq, n_heads*d]

    // Flatten the full cache back to [batch, T, n_kv_heads*head_dim] for the
    // runtime to store.
    let k_store = k_full_hs.merge_dims(2, 3);
    let v_store = v_full_hs.merge_dims(2, 3);
    (attn, k_store, v_store)
}

/// Single decode step against a fixed-capacity KV cache, with a **runtime**
/// position (what the serving loop needs: a static graph can't bake the
/// per-step position into a constant).
///
/// `q` is the current token `[batch, 1, n_heads*head_dim]`. `k_cache`/`v_cache`
/// are `[batch, max_cache, n_kv_heads*head_dim]`, holding the running cache
/// (keys already RoPE-rotated when they were stored; slots `0..=position`
/// valid). `position` is a runtime scalar `[1]` — the current token's absolute
/// index. Returns `[batch, 1, n_heads*head_dim]` (the caller applies
/// `o_proj`).
///
/// The serving loop owns the cache: it RoPE-rotates the new key, writes the
/// new k/v into the cache at `position` (via `PagedKVAllocator`), feeds the
/// updated cache + `position` here, and reads back the attention output.
///
/// Validated on the CUDA path. (Luminal's CPU `NativeRuntime` search hits an
/// internal scheduling error when this runtime-position mask meets the
/// matmul-derived attention weights; the equivalent compile-time-`past` cache
/// math is covered green by `tests/kv_cache.rs::decode_with_cache_matches_full_prefill`.)
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_with_cache(
    q: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    position: GraphTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_cache: usize,
    rope_theta: f32,
) -> GraphTensor {
    let kv_groups = n_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let batch = q.dims()[0];
    let half = head_dim / 2;

    // Create all `arange` tensors up front in one borrow of the graph — never
    // hold a `&mut Graph` across the tensor ops below (luminal threads the
    // graph through raw pointers, so two live mutable views corrupt it).
    let (inv_freq, slots) = {
        let cx = q.graph();
        let idx = cx.arange(half).cast(DType::F32);
        let inv_freq = (idx * (-2.0 / head_dim as f32 * rope_theta.ln())).exp(); // [half]
        let slots = cx.arange(max_cache).cast(DType::F32).expand_dim(0, 1); // [1, C]
        (inv_freq, slots)
    };

    // RoPE the current query at the runtime `position`.
    let pos_f = position.cast(DType::F32); // [1]
    // angles[1, half] = position * inv_freq.
    let angles = pos_f.expand_dim(1, half) * inv_freq.expand_dim(0, 1);
    let emb = angles.concat_along(angles, 1); // [1, head_dim]
    let q_hs = q.split_dims(2, head_dim); // [b, 1, n_heads, d]
    let qd = q_hs.dims();
    let dt = q.dtype;
    let cos_b = emb.cos().expand_dim(0, qd[0]).expand_dim(2, qd[2]).cast(dt); // [b, 1, n_heads, d]
    let sin_b = emb.sin().expand_dim(0, qd[0]).expand_dim(2, qd[2]).cast(dt);
    let q_hs = q_hs * cos_b + rotate_half(q_hs, head_dim) * sin_b;

    // GQA reshape. q:[b,1,Hkv,groups,d]->[b,Hkv,groups,1,d];
    // k_cache:[b,C,Hkv,d]->[b,Hkv,groups,d,C]; v_cache:->[b,Hkv,groups,C,d].
    let q5 = q_hs.split_dims(2, kv_groups).permute((0, 2, 3, 1, 4));
    let k5 = k_cache
        .split_dims(2, head_dim)
        .permute((0, 2, 3, 1))
        .expand_dim(2, kv_groups);
    let v5 = v_cache
        .split_dims(2, head_dim)
        .permute((0, 2, 1, 3))
        .expand_dim(2, kv_groups);

    let scores = q5.matmul(k5) * scale; // [b, Hkv, groups, 1, C]
    // Mask: attend cache slot j iff j <= position (runtime). Stale slots beyond
    // the current length get a large negative additive bias before the softmax.
    let allowed = slots.le(pos_f.expand_dim(1, max_cache)).cast(DType::F32); // [1, C]
    let bias = ((allowed - 1.0) * 1.0e9)
        .expand_dim(0, batch)
        .expand_dim(1, n_kv_heads)
        .expand_dim(2, kv_groups)
        .cast(scores.dtype); // [b, Hkv, groups, 1, C]
    let weights = (scores + bias).softmax(4);
    let attn = weights.matmul(v5); // [b, Hkv, groups, 1, d]

    attn.permute((0, 3, 1, 2, 4)) // [b, 1, Hkv, groups, d]
        .merge_dims(3, 4)
        .merge_dims(2, 3) // [b, 1, n_heads*d]
}

/// Single-token cached decode against a **growing, runtime-fed** KV cache.
///
/// This is the serving counterpart used by the runtime [`KvCache`]
/// (`skein_runtime`): each decode step the runtime feeds the accumulated past
/// (`k_cache`/`v_cache`, shape `[batch, past, n_kv_heads*head_dim]`, where
/// `past` is a *dynamic* dimension that grows by one each step) and the current
/// absolute `position` (`[1]`); the graph returns the attention output plus the
/// new token's **rotated** key and its value, which the runtime appends to the
/// cache for the next step.
///
/// The single query is the latest token, so it attends to *every* cached key
/// plus its own — no causal mask is needed (unlike prefill). RoPE is applied to
/// q and the new k at the runtime `position`, matching the rotation already
/// stored for the cached keys.
///
/// Numerics validated on GPU (the CPU `NativeRuntime` cannot execute the block;
/// the dynamic-`past` concat in particular — including the `past == 0` first
/// step — should be checked on hardware).
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_step(
    q: GraphTensor,
    k_new: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    position: GraphTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let kv_groups = n_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let half = head_dim / 2;

    // Runtime-position RoPE tables for a single position. Build the `arange`
    // in one borrow of the graph (luminal threads it through raw pointers).
    let inv_freq = {
        let cx = q.graph();
        let idx = cx.arange(half).cast(DType::F32);
        (idx * (-2.0 / head_dim as f32 * rope_theta.ln())).exp() // [half]
    };
    let pos_f = position.cast(DType::F32); // [1]
    let angles = pos_f.expand_dim(1, half) * inv_freq.expand_dim(0, 1); // [1, half]
    let emb = angles.concat_along(angles, 1); // [1, head_dim]

    // Rotate the current q and k at `position`.
    let q_hs = q.split_dims(2, head_dim); // [b, 1, n_heads, d]
    let qd = q_hs.dims();
    let q_cos = emb
        .cos()
        .expand_dim(0, qd[0])
        .expand_dim(2, qd[2])
        .cast(q.dtype);
    let q_sin = emb
        .sin()
        .expand_dim(0, qd[0])
        .expand_dim(2, qd[2])
        .cast(q.dtype);
    let q_hs = q_hs * q_cos + rotate_half(q_hs, head_dim) * q_sin;

    let k_hs = k_new.split_dims(2, head_dim); // [b, 1, n_kv, d]
    let kd = k_hs.dims();
    let k_cos = emb
        .cos()
        .expand_dim(0, kd[0])
        .expand_dim(2, kd[2])
        .cast(k_new.dtype);
    let k_sin = emb
        .sin()
        .expand_dim(0, kd[0])
        .expand_dim(2, kd[2])
        .cast(k_new.dtype);
    let k_hs = k_hs * k_cos + rotate_half(k_hs, head_dim) * k_sin; // [b, 1, n_kv, d]
    let v_hs = v_new.split_dims(2, head_dim); // [b, 1, n_kv, d]

    // Concatenate the past cache with the current token along the sequence
    // axis. `past == 0` (first step) yields an empty cache; the concat then
    // reduces to the current token.
    let k_cache_hs = k_cache.split_dims(2, head_dim); // [b, past, n_kv, d]
    let v_cache_hs = v_cache.split_dims(2, head_dim);
    let k_full_hs = k_cache_hs.concat_along(k_hs, 1); // [b, past+1, n_kv, d]
    let v_full_hs = v_cache_hs.concat_along(v_hs, 1);

    // GQA attention; single query → no causal mask.
    let q5 = q_hs.split_dims(2, kv_groups).permute((0, 2, 3, 1, 4)); // [b,n_kv,groups,1,d]
    let k5 = k_full_hs.permute((0, 2, 3, 1)).expand_dim(2, kv_groups); // [b,n_kv,groups,d,T]
    let v5 = v_full_hs.permute((0, 2, 1, 3)).expand_dim(2, kv_groups); // [b,n_kv,groups,T,d]
    let scores = q5.matmul(k5) * scale; // [b,n_kv,groups,1,T]
    let weights = scores.softmax(4);
    let attn = weights.matmul(v5); // [b,n_kv,groups,1,d]
    let attn = attn
        .permute((0, 3, 1, 2, 4))
        .merge_dims(3, 4)
        .merge_dims(2, 3); // [b, 1, n_heads*d]

    // The new (rotated) key and value, flattened for the runtime to append.
    let k_store = k_hs.merge_dims(2, 3); // [b, 1, n_kv*d]
    let v_store = v_hs.merge_dims(2, 3);
    (attn, k_store, v_store)
}

/// Cached-decode attention against a **fixed-capacity** KV cache of `max_cache`
/// slots — the production-shaped design that avoids any dynamic / zero-length
/// dimension (luminal's CUDA runtime cannot represent the empty cache at decode
/// position 0 with a growing `past` dim: it resolves to a null device buffer).
///
/// `k_cache`/`v_cache` are static `[batch, max_cache, n_kv*head_dim]` runtime-fed
/// buffers holding the rotated past keys / raw values in slots `0..position`;
/// `position` is the current token's absolute index (`[1]`, runtime scalar). The
/// new token's rotated K / raw V are written into slot `position` *in-graph* via
/// an arithmetic select (no scatter op, no dynamic dim), attention masks slots
/// `> position`, and the new K/V are returned flattened for the runtime to
/// persist into slot `position` of its fixed buffer for the next step. Single
/// query, so the only mask is the validity mask. Assumes `batch == 1` (the
/// served plan's `max_batch`); the select's `[1, C]` masks broadcast as
/// `[batch=1, C, ...]`.
#[allow(clippy::too_many_arguments)]
pub fn attention_fixed_cache(
    q: GraphTensor,        // [b, 1, n_heads*d]
    k_new: GraphTensor,    // [b, 1, n_kv*d]
    v_new: GraphTensor,    // [b, 1, n_kv*d]
    k_cache: GraphTensor,  // [b, C, n_kv*d]
    v_cache: GraphTensor,  // [b, C, n_kv*d]
    position: GraphTensor, // [1]
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_cache: usize,
    rope_theta: f32,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let kv_groups = n_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let half = head_dim / 2;
    let batch = q.dims()[0];
    let kv_dim = n_kv_heads * head_dim;

    // Runtime-position RoPE tables for the single current position.
    let inv_freq = {
        let cx = q.graph();
        let idx = cx.arange(half).cast(DType::F32);
        (idx * (-2.0 / head_dim as f32 * rope_theta.ln())).exp() // [half]
    };
    let pos_f = position.cast(DType::F32); // [1]
    let angles = pos_f.expand_dim(1, half) * inv_freq.expand_dim(0, 1); // [1, half]
    let emb = angles.concat_along(angles, 1); // [1, head_dim]

    // Rotate q and k_new at `position`.
    let q_hs = q.split_dims(2, head_dim); // [b,1,n_heads,d]
    let qd = q_hs.dims();
    let q_cos = emb.cos().expand_dim(0, qd[0]).expand_dim(2, qd[2]).cast(q.dtype);
    let q_sin = emb.sin().expand_dim(0, qd[0]).expand_dim(2, qd[2]).cast(q.dtype);
    let q_hs = q_hs * q_cos + rotate_half(q_hs, head_dim) * q_sin;

    let k_hs = k_new.split_dims(2, head_dim); // [b,1,n_kv,d]
    let kd = k_hs.dims();
    let k_cos = emb
        .cos()
        .expand_dim(0, kd[0])
        .expand_dim(2, kd[2])
        .cast(k_new.dtype);
    let k_sin = emb
        .sin()
        .expand_dim(0, kd[0])
        .expand_dim(2, kd[2])
        .cast(k_new.dtype);
    let k_hs = k_hs * k_cos + rotate_half(k_hs, head_dim) * k_sin; // [b,1,n_kv,d]
    let v_hs = v_new.split_dims(2, head_dim); // [b,1,n_kv,d]

    // New token flattened — returned for the runtime to write into slot `position`.
    let k_store = k_hs.merge_dims(2, 3); // [b,1,n_kv*d]
    let v_store = v_hs.merge_dims(2, 3);

    // Slot indices and the current-slot / validity selectors.
    //   is_cur[slot] = (slot <= pos) - (slot <= pos-1)  == 1 iff slot == pos
    //   allowed[slot] = (slot <= pos)                    == 1 for valid slots
    let slots = {
        let cx = q.graph();
        cx.arange(max_cache).cast(DType::F32).expand_dim(0, 1) // [1, C]
    };
    let pos_c = pos_f.expand_dim(1, max_cache); // [1, C]
    let posm1_c = (pos_f - 1.0).expand_dim(1, max_cache); // [1, C]
    // `le` yields Bool; cast to f32 for the arithmetic select / mask.
    let is_cur = slots.le(pos_c).cast(DType::F32) - slots.le(posm1_c).cast(DType::F32); // [1, C]

    // Write the new token into slot `position` via select (batch == 1 → the
    // [1, C] masks broadcast across the [b, C, kv_dim] cache):
    //   full = cache * (1 - is_cur) + new_broadcast * is_cur
    let mut is_cur_b = is_cur.expand_dim(2, kv_dim); // [1, C, kv_dim]
    is_cur_b.shape.expand(k_cache.dims()); // [b, C, kv_dim]
    let keep = (is_cur_b * -1.0) + 1.0; // 1 - is_cur
    let mut k_new_b = k_store; // [b, 1, kv_dim]
    k_new_b.shape.expand(k_cache.dims()); // broadcast slot dim 1 -> C
    let mut v_new_b = v_store;
    v_new_b.shape.expand(v_cache.dims());
    let k_full = k_cache * keep + k_new_b * is_cur_b; // [b, C, kv_dim]
    let v_full = v_cache * keep + v_new_b * is_cur_b;

    // GQA attention over the fixed cache, masking slots > position.
    let k_full_hs = k_full.split_dims(2, head_dim); // [b, C, n_kv, d]
    let v_full_hs = v_full.split_dims(2, head_dim);
    let q5 = q_hs.split_dims(2, kv_groups).permute((0, 2, 3, 1, 4)); // [b,n_kv,groups,1,d]
    let k5 = k_full_hs.permute((0, 2, 3, 1)).expand_dim(2, kv_groups); // [b,n_kv,groups,d,C]
    let v5 = v_full_hs.permute((0, 2, 1, 3)).expand_dim(2, kv_groups); // [b,n_kv,groups,C,d]
    let scores = q5.matmul(k5) * scale; // [b,n_kv,groups,1,C]
    let allowed = slots.le(pos_c).cast(DType::F32);
    let bias = ((allowed - 1.0) * 1.0e9)
        .expand_dim(0, batch)
        .expand_dim(1, n_kv_heads)
        .expand_dim(2, kv_groups)
        .cast(scores.dtype); // [b,n_kv,groups,1,C]
    let weights = (scores + bias).softmax(4);
    let attn = weights.matmul(v5); // [b,n_kv,groups,1,d]
    let attn = attn
        .permute((0, 3, 1, 2, 4))
        .merge_dims(3, 4)
        .merge_dims(2, 3); // [b, 1, n_heads*d]

    (attn, k_store, v_store)
}

/// Self-attention with GQA expansion. Looks up Q/K/V/O weights from a
/// `declared` map and delegates the op graph to [`wire_attention_math`]
/// (which applies RoPE and the causal mask). The segment orchestrator
/// normally calls [`wire_attention_math`] directly with per-segment handles;
/// this map-driven entry point is kept for the hand-computed attention test.
pub fn wire_attention(
    cx: &mut LuminalGraph,
    declared: &HashMap<String, DeclaredTensor>,
    meta: &ModelMeta,
    block: usize,
    input: GraphTensor,
) -> Result<GraphTensor, EmitError> {
    let q_name = format!("model.layers.{block}.self_attn.q_proj.weight");
    let k_name = format!("model.layers.{block}.self_attn.k_proj.weight");
    let v_name = format!("model.layers.{block}.self_attn.v_proj.weight");
    let o_name = format!("model.layers.{block}.self_attn.o_proj.weight");
    // Derive local head counts from declared (sharded) Q/K weight rows.
    let n_heads_local = declared
        .get(&q_name)
        .map(|d| d.shape[0] / meta.head_dim)
        .unwrap_or(meta.num_attention_heads);
    let n_kv_heads_local = declared
        .get(&k_name)
        .map(|d| d.shape[0] / meta.head_dim)
        .unwrap_or(meta.num_kv_heads);
    let q_w = handle_for(cx, declared, &q_name)?;
    let k_w = handle_for(cx, declared, &k_name)?;
    let v_w = handle_for(cx, declared, &v_name)?;
    let o_w = handle_for(cx, declared, &o_name)?;
    Ok(wire_attention_math(
        n_heads_local,
        n_kv_heads_local,
        meta.head_dim,
        meta.rope_theta,
        0,
        input,
        q_w,
        k_w,
        v_w,
        o_w,
    ))
}

/// The op-graph half of [`wire_attention`]: no `declared` map needed,
/// just hand-built Luminal ops over caller-provided weight handles. Applies
/// RoPE to Q/K and an additive causal mask to the attention scores.
///
/// Takes the **device-local** head counts. Under TP the Q/K/V weights
/// are column-parallel: each device sees only `n_heads / tp` and
/// `n_kv_heads / tp` heads, and the final shape reassignment must use
/// those local values, not the cluster-wide `ModelMeta::num_*_heads`.
///
/// `position_offset` is the absolute position of this chunk's first token:
/// `0` for prefill, or the current KV-cache length for a decode step (see
/// [`rope_tables`]).
#[allow(clippy::too_many_arguments)]
pub fn wire_attention_math(
    n_heads_local: usize,
    n_kv_heads_local: usize,
    head_dim: usize,
    rope_theta: f32,
    position_offset: usize,
    input: GraphTensor,
    q_w: GraphTensor,
    k_w: GraphTensor,
    v_w: GraphTensor,
    o_w: GraphTensor,
) -> GraphTensor {
    let kv_groups = n_heads_local / n_kv_heads_local;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let batch = input.dims()[0];
    let seq = input.dims()[1];

    // Project, then split the projection into [batch, seq, heads, head_dim].
    let q = input.matmul(q_w.permute((1, 0))).split_dims(2, head_dim);
    let k = input.matmul(k_w.permute((1, 0))).split_dims(2, head_dim);
    let v = input.matmul(v_w.permute((1, 0)));

    // RoPE on Q and K (rotate-half / NeoX layout, as Mixtral uses).
    let (cos_t, sin_t) = rope_tables(input.graph(), seq, head_dim, rope_theta, position_offset);
    let q = apply_rope(q, cos_t, sin_t, head_dim);
    let k = apply_rope(k, cos_t, sin_t, head_dim);

    // GQA reshape: expand each KV head across its group of query heads.
    let q = q.split_dims(2, kv_groups).permute((0, 2, 3, 1, 4)); // [b, kvh, kvg, seq, hd]
    let k = k.permute((0, 2, 3, 1)); // [b, kvh, hd, seq]
    let v = v.split_dims(2, head_dim).permute((0, 2, 1, 3)); // [b, kvh, seq, hd]
    let k = k.expand_dim(2, kv_groups);
    let v = v.expand_dim(2, kv_groups);

    // scores: [b, kvh, kvg, q_seq, k_seq].
    let scores = q.matmul(k) * scale;
    // Additive causal mask: query at position i may not attend to keys j > i.
    let bias = causal_bias(input.graph(), seq)
        .expand_dim(0, batch)
        .expand_dim(1, n_kv_heads_local)
        .expand_dim(2, kv_groups)
        .cast(scores.dtype);
    let scores = scores + bias;
    let weights = scores.softmax(4);
    let attn = weights.matmul(v);

    // [b, kvh, kvg, seq, d] -> [b, seq, kvh, kvg, d] -> [b, seq, n_heads*d].
    // `merge_dims` composes with the permute's strides; a plain `ShapeTracker`
    // reassignment would misread the strided data whenever `kv_groups > 1`
    // (e.g. real Mixtral GQA), interleaving heads across sequence positions.
    let attn = attn
        .permute((0, 3, 1, 2, 4))
        .merge_dims(3, 4) // [b, seq, kvh, kvg*d]
        .merge_dims(2, 3); // [b, seq, kvh*kvg*d] = [b, seq, n_heads_local*d]

    attn.matmul(o_w.permute((1, 0)))
}

/// RoPE `(cos, sin)` tables of shape `[seq, head_dim]` in the rotate-half
/// (NeoX) layout Mixtral uses. `inv_freq[i] = theta^(-2i/head_dim)` and the
/// angle table is `outer(positions, inv_freq)` duplicated across the two
/// halves of `head_dim`.
///
/// `position_offset` is the absolute position of the first token in this
/// chunk: `0` for prefill (positions `0..seq`), and the current KV-cache
/// length for a decode step (positions `offset..offset+seq`). This is what
/// makes decode-step RoPE correct — the query/key for a decode token must be
/// rotated by its true position in the full sequence, not by `0`.
pub fn rope_tables(
    cx: &mut LuminalGraph,
    seq: impl Into<Expression>,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> (GraphTensor, GraphTensor) {
    let seq = seq.into();
    let half = head_dim / 2;
    // inv_freq[i] = exp(-(2i/head_dim) * ln(theta)).
    let idx = cx.arange(half).cast(DType::F32);
    let inv_freq = (idx * (-2.0 / head_dim as f32 * theta.ln())).exp(); // [half]
    // positions = offset, offset+1, .., offset+seq-1.
    let positions = cx.arange(seq).cast(DType::F32) + position_offset as f32; // [seq]
    // outer product → [seq, half].
    let angles = positions.expand_dim(1, half) * inv_freq.expand_dim(0, seq);
    // NeoX layout duplicates the angles across both halves → [seq, head_dim].
    let emb = angles.concat_along(angles, 1);
    (emb.cos(), emb.sin())
}

/// Apply RoPE to `x` of shape `[batch, seq, heads, head_dim]`. `cos`/`sin`
/// are `[seq, head_dim]` and broadcast across batch and heads.
pub fn apply_rope(
    x: GraphTensor,
    cos: GraphTensor,
    sin: GraphTensor,
    head_dim: usize,
) -> GraphTensor {
    let dims = x.dims();
    let batch = dims[0];
    let heads = dims[2];
    // The cos/sin tables are F32; match `x`'s dtype (e.g. Bf16) before mul.
    let dt = x.dtype;
    let cos_b = cos.expand_dim(0, batch).expand_dim(2, heads).cast(dt); // [b, seq, heads, hd]
    let sin_b = sin.expand_dim(0, batch).expand_dim(2, heads).cast(dt);
    x * cos_b + rotate_half(x, head_dim) * sin_b
}

/// `rotate_half([x1, x2]) = [-x2, x1]` along the last (`head_dim`) axis,
/// where `x1`/`x2` are the two contiguous halves of `head_dim`.
fn rotate_half(x: GraphTensor, head_dim: usize) -> GraphTensor {
    let half = head_dim / 2;
    let x1 = x.slice((.., .., .., ..half));
    let x2 = x.slice((.., .., .., half..));
    (x2 * -1.0).concat_along(x1, 3)
}

/// Additive causal attention bias of shape `[seq, seq]`: `0` on/below the
/// diagonal (allowed) and a large negative value above it (future keys).
fn causal_bias(cx: &mut LuminalGraph, seq: impl Into<Expression>) -> GraphTensor {
    let seq = seq.into();
    // tril(seq, 0) is 1 on/below the diagonal, 0 above.
    let lower = cx.tril(seq, 0).cast(DType::F32);
    (lower - 1.0) * 1.0e9
}

/// Reconstruct a `GraphTensor` from a previously-declared weight tensor.
pub fn handle_for(
    cx: &mut LuminalGraph,
    declared: &HashMap<String, DeclaredTensor>,
    name: &str,
) -> Result<GraphTensor, EmitError> {
    let d = declared
        .get(name)
        .ok_or_else(|| EmitError::UnknownParamPattern { name: name.into() })?;
    Ok(handle_for_declared(cx, d))
}

fn handle_for_declared(cx: &mut LuminalGraph, d: &DeclaredTensor) -> GraphTensor {
    let dims: Vec<Expression> = d.shape.iter().copied().map(Expression::from).collect();
    GraphTensor::from_id(
        d.id,
        ShapeTracker::new(dims),
        cx as *mut _,
        to_luminal_dtype(d.dtype),
    )
}

/// Declare one `Param` into `cx` with sharded shape + plan dtype.
/// Counterpart to `graph_builder::declare_param`, factored out so the
/// segment orchestrator can declare on demand into each segment's graph
/// rather than relying on a single pre-built declared map.
fn declare_param_into(
    cx: &mut LuminalGraph,
    plan: &Plan,
    block_idx: Option<usize>,
    param: &Param,
    role: &ShardRole,
) -> Result<DeclaredTensor, EmitError> {
    let dims_usize = shard_param_dims(param, role)?;
    let dims_expr: Vec<Expression> = dims_usize.iter().copied().map(Expression::from).collect();
    let dtype_skein = match block_idx {
        Some(b) => skein_cost::compute::weight_dtype(plan, Some(b)),
        None => skein_cost::compute::weight_dtype(plan, None),
    };
    let dtype_lum = to_luminal_dtype(dtype_skein);
    let tensor = cx.named_tensor(param.name.clone(), dims_expr);
    let id = tensor.id;
    cx.get_op_mut::<Input>(id).dtype = dtype_lum;
    // Keep input_meta consistent with the Input op dtype: the runtime narrows
    // staged f32 weight data to this dtype, and the codegen reads the buffer at
    // this dtype. A stale F32 entry (named_tensor's default) leaves a bf16
    // weight as an f32 buffer that the bf16 codegen reads as interleaved zeros.
    cx.input_meta.insert(id, (param.name.clone(), dtype_lum));
    Ok(DeclaredTensor {
        id,
        shape: dims_usize,
        dtype: dtype_skein,
        role: *role,
    })
}
