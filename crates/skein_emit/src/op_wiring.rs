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
//! ## Deferred lowering (TODO)
//!
//! The structural lowering — segment boundaries, collective ordering,
//! handoff naming, weight sharding — is complete and tested. The following
//! per-op semantics are intentionally not yet lowered; each needs work that
//! is best validated against a GPU reference (see `docs/lowering.md`):
//!
//! - **TODO(rope):** rotary position embeddings on Q/K. Requires plumbing a
//!   position-id input through segment 0 and the IO manifest, then applying
//!   the rotation in [`wire_attention_math`].
//! - **TODO(causal-mask):** additive causal mask before the attention
//!   softmax. A no-op for the current static `seq = 1` decode-mode lowering
//!   (a single query attends to all cached keys); required once prefill
//!   (`seq > 1`) graphs are emitted.
//! - **TODO(moe-topk):** top-k expert sparsification. [`DeviceWiring::wire_block_moe`]
//!   currently computes a *dense* mixture (every owned expert weighted by its
//!   full-softmax gate probability). Mixtral routes to the top-k experts and
//!   renormalizes their gate weights.
//! - **TODO(ep-routing):** expert-parallel token routing. At `ep > 1` the
//!   dispatch/combine collectives are wired with the correct shapes and
//!   handoff names via [`DeviceWiring::reshape_for_handoff`], but the actual
//!   per-token scatter to / gather from remote experts is not yet emitted.

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
    CollectivePoint, INPUT_TOKENS, LOGITS, carry_post_attn, carry_pre_block, collective_attn_out,
    collective_ffn_out, collective_moe_combine, collective_moe_dispatch, device_collective_points,
};
use crate::segment::{HandoffTensor, Segment, SequenceStep};
use crate::shard_role::{ShardRole, shard_role_for_param};

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
    wiring.wire_initial()?;

    let mut point_iter = points.iter();
    for block in 0..ir.meta.num_layers {
        wiring.wire_block(block, &mut point_iter)?;
    }

    // No more collectives expected after the last block.
    assert!(point_iter.next().is_none(), "collective points exhausted");

    wiring.wire_final()?;
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
    live: HashMap<String, LiveTensor>,

    // Static shape dimensions baked into every tensor — matches
    // `topology::emit_topology`'s `[batch, 1, hidden]` shape contract.
    batch: usize,
    seq: usize,
    hidden: usize,
    activation_dtype: skein_ir::types::Dtype,
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
            live: HashMap::new(),
            batch,
            seq,
            hidden,
            activation_dtype,
        })
    }

    // -----------------------------------------------------------------------
    // Segment lifecycle
    // -----------------------------------------------------------------------

    /// Close the current segment with `output_handoff` and start a fresh
    /// one. Pushes an `ExecuteSegment` step into `sequencing`.
    fn close_current_segment(&mut self, output_handoff: Vec<HandoffTensor>) {
        let cx = std::mem::replace(&mut self.cur_cx, LuminalGraph::new());
        let declared = std::mem::take(&mut self.cur_declared);
        let op_nodes = std::mem::take(&mut self.cur_op_nodes);
        let input_handoff = std::mem::take(&mut self.cur_input_handoff);

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
        output_handoff.push(HandoffTensor {
            logical_name: collective_tensor.to_string(),
            luminal_id: coll_output.id,
            shape: coll_live.shape.clone(),
            dtype: coll_live.dtype,
        });
        let mut carry_specs: Vec<HandoffSpec> = Vec::new();
        for c in carries {
            let live = self
                .live
                .get(*c)
                .unwrap_or_else(|| panic!("carry {c} missing from live"));
            let carry_output = live.tensor.output();
            output_handoff.push(HandoffTensor {
                logical_name: (*c).to_string(),
                luminal_id: carry_output.id,
                shape: live.shape.clone(),
                dtype: live.dtype,
            });
            carry_specs.push(HandoffSpec {
                logical_name: (*c).to_string(),
                shape: live.shape.clone(),
                dtype: live.dtype,
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
            dtype: point.dtype,
        });
        next_inputs.extend(carry_specs);
        self.open_new_segment(next_inputs);
    }

    // -----------------------------------------------------------------------
    // Initial / final segments
    // -----------------------------------------------------------------------

    fn wire_initial(&mut self) -> Result<(), EmitError> {
        // Segment 0 declares `input_tokens` as its sole input handoff.
        let input_shape = vec![self.batch, self.seq];
        let tokens = self
            .cur_cx
            .named_tensor(INPUT_TOKENS, (self.batch, self.seq));
        let tokens = tokens.as_dtype(DType::Int);
        self.cur_cx.get_op_mut::<Input>(tokens.id).dtype = DType::Int;
        self.cur_input_handoff.push(HandoffTensor {
            logical_name: INPUT_TOKENS.to_string(),
            luminal_id: tokens.id,
            shape: input_shape.clone(),
            dtype: skein_ir::types::Dtype::Int8, // marker — runtime treats it as token ids
        });
        self.cur_op_nodes
            .insert(INPUT_TOKENS.to_string(), tokens.id);

        // Embedding lookup → [batch, seq, hidden].
        let embed_w = self.weight("model.embed_tokens.weight")?;
        let hidden = embedding_lookup(tokens, embed_w, self.batch, self.seq, self.hidden);
        self.cur_op_nodes
            .insert("hidden_after_embed".to_string(), hidden.id);

        // Push into live as the input to block 0.
        let carry0 = carry_pre_block(0);
        self.live.insert(
            carry0,
            LiveTensor {
                tensor: hidden,
                shape: vec![self.batch, self.seq, self.hidden],
                dtype: self.activation_dtype,
            },
        );
        Ok(())
    }

    fn wire_final(&mut self) -> Result<(), EmitError> {
        let last_carry_name = carry_pre_block(self.ir.meta.num_layers);
        let final_hidden_live = self
            .live
            .remove(&last_carry_name)
            .expect("last carry tensor missing from live");
        let final_hidden = final_hidden_live.tensor;

        let final_norm_w = self.weight("model.norm.weight")?;
        let normed = rms_norm(final_hidden, final_norm_w, self.ir.meta.rms_norm_eps);

        let lm_head_w = self.weight("lm_head.weight")?;
        let logits = normed.matmul(lm_head_w.permute((1, 0))).output();
        self.cur_op_nodes.insert(LOGITS.to_string(), logits.id);

        // Final segment's output_handoff = [logits].
        let output_handoff = vec![HandoffTensor {
            logical_name: LOGITS.to_string(),
            luminal_id: logits.id,
            shape: vec![self.batch, self.seq, self.ir.meta.vocab],
            dtype: self.activation_dtype,
        }];
        self.close_current_segment(output_handoff);
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

        // Pull the input carry — output of the previous block's tail.
        let carry_in = carry_pre_block(block);
        let hidden = self
            .live
            .get(&carry_in)
            .expect("pre-block carry missing")
            .tensor;

        // (1) EP dispatch (before attn collective in topology order).
        if placement.ep > 1 && has_moe {
            let point = point_iter
                .next()
                .expect("ep dispatch point not present in iter");
            assert_eq!(point.tensor, collective_moe_dispatch(block));
            // Structural handoff only: re-view `hidden` to the dispatch
            // collective's shape so the segment graph is complete and the
            // collective has a typed input. TODO(ep-routing): emit the real
            // per-token scatter to remote experts (see module docstring).
            let dispatch_tensor = self.reshape_for_handoff(hidden, &point.shape);
            self.live.insert(
                point.tensor.clone(),
                LiveTensor {
                    tensor: dispatch_tensor,
                    shape: point.shape.clone(),
                    dtype: point.dtype,
                },
            );
            self.cut_segment_at_collective(point, &point.tensor, &[&carry_in]);
        }

        // (2) Pre-attn ops: input_layernorm + q/k/v + attn → block_N_attn_out.
        let hidden = self
            .live
            .get(&carry_in)
            .expect("pre-block carry missing after optional dispatch cut")
            .tensor;
        let norm1_w = self.weight(&format!("model.layers.{block}.input_layernorm.weight"))?;
        let normed_1 = rms_norm(hidden, norm1_w, self.ir.meta.rms_norm_eps);
        let attn_out = self.wire_block_attention(block, normed_1)?;

        // Record attn_out as block_N_attn_out in live (will either be the
        // collective tensor or just an internal name when tp=1).
        let attn_name = collective_attn_out(block);
        self.live.insert(
            attn_name.clone(),
            LiveTensor {
                tensor: attn_out,
                shape: vec![self.batch, self.seq, self.hidden],
                dtype: self.activation_dtype,
            },
        );

        // (3) TP attn AllReduce.
        if placement.tp > 1 {
            let point = point_iter.next().expect("tp attn point not present");
            assert_eq!(point.tensor, attn_name);
            self.cut_segment_at_collective(point, &attn_name, &[&carry_in]);
        }

        // (4) Residual after attn (in the current segment, post-collective).
        let attn_full = self.live.get(&attn_name).expect("attn out missing").tensor;
        let carry_in_live = self.live.get(&carry_in).expect("carry missing").tensor;
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
        let ffn_out = if has_moe {
            self.wire_block_moe(block, normed_2)?
        } else if has_mlp {
            self.wire_block_mlp(block, normed_2)?
        } else {
            // No FFN layer — just pass through. Defensive; Mixtral has
            // MoE in every block.
            normed_2
        };
        let ffn_name = collective_ffn_out(block);
        self.live.insert(
            ffn_name.clone(),
            LiveTensor {
                tensor: ffn_out,
                shape: vec![self.batch, self.seq, self.hidden],
                dtype: self.activation_dtype,
            },
        );

        // (6) EP combine after MoE.
        if placement.ep > 1 && has_moe {
            let point = point_iter
                .next()
                .expect("ep combine point not present in iter");
            assert_eq!(point.tensor, collective_moe_combine(block));
            // Same structural handoff as dispatch — re-view ffn_out into
            // [batch, top_k, hidden] for the collective tensor.
            // TODO(ep-routing): emit the real gather of expert outputs.
            let combine_tensor = self.reshape_for_handoff(ffn_out, &point.shape);
            self.live.insert(
                point.tensor.clone(),
                LiveTensor {
                    tensor: combine_tensor,
                    shape: point.shape.clone(),
                    dtype: point.dtype,
                },
            );
            self.cut_segment_at_collective(point, &point.tensor, &[&carry_post, &ffn_name]);
            // After the cut, fold the combined tensor back to ffn_name's
            // shape and re-register under ffn_name for the downstream
            // ffn-AllReduce step.
            let combined = self
                .live
                .get(&point.tensor)
                .expect("combined missing")
                .tensor;
            let folded = self.reshape_for_handoff(combined, &[self.batch, self.seq, self.hidden]);
            self.live.insert(
                ffn_name.clone(),
                LiveTensor {
                    tensor: folded,
                    shape: vec![self.batch, self.seq, self.hidden],
                    dtype: self.activation_dtype,
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
        let ffn_full = self.live.get(&ffn_name).expect("ffn out missing").tensor;
        let carry_post_live = self
            .live
            .get(&carry_post)
            .expect("post-attn carry missing")
            .tensor;
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
        Ok(wire_attention_math(
            n_heads_local,
            n_kv_heads_local,
            head_dim,
            normed,
            q_w,
            k_w,
            v_w,
            o_w,
        ))
    }

    /// MoE feed-forward for one block.
    ///
    /// TODO(moe-topk): this lowers a *dense* mixture — every expert the
    /// device owns is evaluated and weighted by its full-softmax gate
    /// probability. Mixtral selects the top-k experts per token and
    /// renormalizes their gate weights over the selected set. Implementing
    /// the top-k selection needs a top-k / masked-softmax op and should be
    /// validated against a GPU reference before it replaces the dense path.
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
        let n = normed.dims().len();
        let routing_probs = routing_logits.softmax(n - 1);

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

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Re-view `t` under `target_shape` via direct `ShapeTracker`
    /// assignment, producing a typed input/output for a collective handoff.
    /// Used only on the EP dispatch/combine boundaries — see the
    /// `TODO(ep-routing)` note in the module docstring.
    fn reshape_for_handoff(&self, mut t: GraphTensor, target_shape: &[usize]) -> GraphTensor {
        let dims: Vec<Expression> = target_shape.iter().copied().map(Expression::from).collect();
        t.shape = ShapeTracker::new(dims);
        t
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

/// `x / sqrt(mean(x²) + eps) * weight` — i.e. RmsNorm.
fn rms_norm(input: GraphTensor, weight: GraphTensor, eps: f32) -> GraphTensor {
    let last = input.shape.last_axis();
    let normed = input.std_norm(last, eps);
    let dims = input.dims();
    let n = dims.len();
    normed * weight.expand_lhs(&dims[..n - 1])
}

/// Token-id `[batch, seq]` → embedded `[batch, seq, hidden]`. Static dims
/// are passed in explicitly because segments use static shapes rather than
/// dynamic `'b'` / `'s'` dimensions.
fn embedding_lookup(
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

/// Self-attention with GQA expansion. Looks up Q/K/V/O weights from a
/// `declared` map and delegates the op graph to [`wire_attention_math`].
/// The segment orchestrator normally calls [`wire_attention_math`] directly
/// with per-segment handles; this map-driven entry point is kept for the
/// hand-computed attention test.
///
/// TODO(rope) / TODO(causal-mask): rotary embeddings and the causal mask
/// are not yet applied here — see the module docstring.
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
        input,
        q_w,
        k_w,
        v_w,
        o_w,
    ))
}

/// The op-graph half of [`wire_attention`]: no `declared` map needed,
/// just hand-built Luminal ops over caller-provided weight handles.
///
/// Takes the **device-local** head counts. Under TP the Q/K/V weights
/// are column-parallel: each device sees only `n_heads / tp` and
/// `n_kv_heads / tp` heads, and the final shape reassignment must use
/// those local values, not the cluster-wide `ModelMeta::num_*_heads`.
#[allow(clippy::too_many_arguments)]
pub fn wire_attention_math(
    n_heads_local: usize,
    n_kv_heads_local: usize,
    head_dim: usize,
    input: GraphTensor,
    q_w: GraphTensor,
    k_w: GraphTensor,
    v_w: GraphTensor,
    o_w: GraphTensor,
) -> GraphTensor {
    let kv_groups = n_heads_local / n_kv_heads_local;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = input.matmul(q_w.permute((1, 0)));
    let k = input.matmul(k_w.permute((1, 0)));
    let v = input.matmul(v_w.permute((1, 0)));

    let q = q
        .split_dims(2, head_dim)
        .split_dims(2, kv_groups)
        .permute((0, 2, 3, 1, 4));
    let k = k.split_dims(2, head_dim).permute((0, 2, 3, 1));
    let v = v.split_dims(2, head_dim).permute((0, 2, 1, 3));

    // TODO(rope): apply rotary position embeddings to `q` and `k` here,
    // before the score matmul, once position ids are plumbed through.

    let k = k.expand_dim(2, kv_groups);
    let v = v.expand_dim(2, kv_groups);

    let scores = q.matmul(k) * scale;
    // TODO(causal-mask): add an additive causal mask to `scores` before the
    // softmax for prefill (`seq > 1`). It is a no-op for the current
    // `seq = 1` decode-mode lowering.
    let weights = scores.softmax(4);
    let attn = weights.matmul(v);

    let mut attn = attn.permute((0, 3, 1, 2, 4));
    let dims = attn.dims();
    let b = dims[0];
    let s = dims[1];
    attn.shape = ShapeTracker::new((b, s, Expression::from(n_heads_local * head_dim)));

    attn.matmul(o_w.permute((1, 0)))
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
    Ok(DeclaredTensor {
        id,
        shape: dims_usize,
        dtype: dtype_skein,
        role: *role,
    })
}
