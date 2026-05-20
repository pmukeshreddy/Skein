//! Handoff naming conventions + per-device collective schedule.
//!
//! Two responsibilities:
//!
//! - **Naming.** Every cross-segment tensor has a single canonical
//!   `logical_name`. These names are how the runtime routes buffers between
//!   segments and collectives — get them wrong and the runtime feeds the
//!   wrong data into the wrong segment.
//! - **Schedule.** Per-device list of [`CollectivePoint`]s, derived
//!   from `(Plan, IR)` without going through cluster-wide
//!   `emit_topology`. This is per-device by construction: every device
//!   participating in the plan sees the same number of collectives per
//!   block, regardless of which TP/EP shard it sits in. `emit_topology`
//!   emits only one representative group per stage; the runtime needs the
//!   full per-device view, and this is where we compute it.

use skein_cost::collectives::CollectiveKind;
use skein_ir::ir::{Graph, LayerKind};
use skein_ir::plan::Plan;
use skein_ir::types::Dtype;

/// Logical name for the tensor a TP attention AllReduce operates on.
pub fn collective_attn_out(block: usize) -> String {
    format!("block_{block}_attn_out")
}

/// Logical name for the tensor a TP MoE/MLP AllReduce operates on.
pub fn collective_ffn_out(block: usize) -> String {
    format!("block_{block}_ffn_out")
}

/// Logical name for the EP-dispatch AllToAll's tensor.
pub fn collective_moe_dispatch(block: usize) -> String {
    format!("block_{block}_moe_dispatch")
}

/// Logical name for the EP-combine AllToAll's tensor.
pub fn collective_moe_combine(block: usize) -> String {
    format!("block_{block}_moe_combine")
}

/// Residual carry from the start of a decoder block's pre-attn segment.
/// Lives across the attn AllReduce (the residual add fires after the
/// AllReduce on every device's replicated copy).
pub fn carry_pre_block(block: usize) -> String {
    format!("carry_pre_block_{block}")
}

/// Residual carry between a block's attn AllReduce and its MoE/MLP
/// AllReduce. The post-attention residual sum that the post-attn
/// norm + MoE compute on.
pub fn carry_post_attn(block: usize) -> String {
    format!("carry_post_attn_block_{block}")
}

/// Initial input — segment 0 takes this as its only input handoff.
pub const INPUT_TOKENS: &str = "input_tokens";

/// Final output — the last segment's `output_handoff` exposes this.
pub const LOGITS: &str = "logits";

/// Vocab-parallel embedding partial-sum tensor. Each TP rank produces the
/// embedding for the tokens whose ids land in its vocab slice (zeros
/// elsewhere); a RingAllReduce over the TP group reconstructs the full
/// embedding before block 0.
pub const EMBED_OUT: &str = "embed_out";

/// One collective the device participates in. Ordered execution order:
/// outer iteration is decoder block, inner iteration is within-block
/// ordering matching `topology::emit_topology` so the cluster-wide
/// `Topology` view and the per-device sequencing line up step-for-step.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectivePoint {
    pub block: usize,
    pub kind: CollectiveKind,
    /// Logical name of the tensor the collective operates on (and that
    /// the segments on either side reference).
    pub tensor: String,
    /// Shape after the collective. For AllReduce this is the operand
    /// shape; for AllToAll it's the post-transpose shape.
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    /// Device-rank list participating in this collective. Computed by
    /// the caller using `Placement` from the Plan.
    pub participants: Vec<u32>,
}

/// Compute the device's collective schedule. Same ordering as
/// `topology::emit_topology`'s per-block iteration:
///   1. TP AllReduce after attention (if `tp > 1`)
///   2. EP combine all-reduce (if `ep > 1` and block has MoE)
///   3. TP AllReduce after MoE/MLP (if `tp > 1` and block has FFN)
///
/// Expert parallelism uses the *dense* scheme: each EP rank owns a disjoint
/// subset of experts and evaluates them on the (EP-replicated) hidden state,
/// producing a partial MoE output. The EP combine is a sum-all-reduce over
/// the EP group that reconstructs the full MoE output — numerically identical
/// to single-device routing. No dispatch collective is needed because the
/// hidden state is already replicated across the EP group.
///
/// Every device in the plan sees the same count and ordering — they
/// differ only in which TP / EP groups they sit in, which the
/// `participants` field captures.
pub fn device_collective_points(plan: &Plan, ir: &Graph, device_idx: u32) -> Vec<CollectivePoint> {
    let placement = skein_cost::cluster::Placement::from_plan(plan);
    let hidden = ir.meta.hidden;
    let batch = plan.batching.max_batch() as usize;
    let vocab = ir.meta.vocab;
    let mut points = Vec::new();

    let embed_dtype = plan
        .dtype_map
        .per_layer
        .first()
        .map(|p| p.activation)
        .unwrap_or(Dtype::Bf16);

    // Vocab-parallel embedding AllReduce: reconstructs the full embedding from
    // the per-rank vocab-slice partial sums before block 0. Only the first
    // pipeline stage runs the embedding.
    if placement.tp > 1 && device_on_first_stage(device_idx, &placement) {
        points.push(CollectivePoint {
            block: 0,
            kind: CollectiveKind::RingAllReduce,
            tensor: EMBED_OUT.to_string(),
            shape: vec![batch, 1, hidden],
            dtype: embed_dtype,
            participants: tp_group_for_device(device_idx, &placement),
        });
    }

    for block in 0..ir.meta.num_layers {
        let activation_dtype = plan
            .dtype_map
            .per_layer
            .get(block)
            .map(|p| p.activation)
            .unwrap_or(Dtype::Bf16);

        let has_moe = ir
            .layers
            .iter()
            .any(|l| l.block_idx == Some(block) && matches!(l.kind, LayerKind::Moe(_)));
        let has_mlp = ir
            .layers
            .iter()
            .any(|l| l.block_idx == Some(block) && matches!(l.kind, LayerKind::Mlp(_)));

        // 1. TP AllReduce after attn.
        if placement.tp > 1 {
            points.push(CollectivePoint {
                block,
                kind: CollectiveKind::RingAllReduce,
                tensor: collective_attn_out(block),
                shape: vec![batch, 1, hidden],
                dtype: activation_dtype,
                participants: tp_group_for_device(device_idx, &placement),
            });
        }

        // 2. EP combine: sum the per-rank partial expert outputs across the
        //    EP group. The full MoE output shape is [batch, seq=1, hidden].
        if placement.ep > 1 && has_moe {
            points.push(CollectivePoint {
                block,
                kind: CollectiveKind::RingAllReduce,
                tensor: collective_moe_combine(block),
                shape: vec![batch, 1, hidden],
                dtype: activation_dtype,
                participants: ep_group_for_device(device_idx, &placement),
            });
        }

        // 3. TP AllReduce after MoE/MLP.
        if placement.tp > 1 && (has_moe || has_mlp) {
            points.push(CollectivePoint {
                block,
                kind: CollectiveKind::RingAllReduce,
                tensor: collective_ffn_out(block),
                shape: vec![batch, 1, hidden],
                dtype: activation_dtype,
                participants: tp_group_for_device(device_idx, &placement),
            });
        }
    }

    // Vocab-parallel logits AllGather: concatenates each rank's vocab-slice
    // logit shard into the full `[batch, 1, vocab]` logits. Only the last
    // pipeline stage runs the LM head.
    if placement.tp > 1 && device_on_last_stage(device_idx, ir.meta.num_layers, &placement) {
        points.push(CollectivePoint {
            block: ir.meta.num_layers.saturating_sub(1),
            kind: CollectiveKind::AllGather,
            tensor: LOGITS.to_string(),
            shape: vec![batch, 1, vocab],
            dtype: embed_dtype,
            participants: tp_group_for_device(device_idx, &placement),
        });
    }

    points
}

/// Whether `device_idx` sits on pipeline stage 0 (which runs the embedding).
fn device_on_first_stage(device_idx: u32, placement: &skein_cost::cluster::Placement) -> bool {
    device_idx / (placement.tp * placement.ep) == 0
}

/// Whether `device_idx` sits on the last pipeline stage (which runs the LM
/// head). With `pp == 1` every device is on the single (last) stage.
fn device_on_last_stage(
    device_idx: u32,
    _num_layers: usize,
    placement: &skein_cost::cluster::Placement,
) -> bool {
    let stage = device_idx / (placement.tp * placement.ep);
    stage == placement.pp - 1
}

/// TP group this device belongs to. With the lexicographic placement
/// `device_idx = stage * tp * ep + tp_idx * ep + ep_idx`, the TP group
/// fixes `(stage, ep_idx)` and varies `tp_idx ∈ [0, tp)`.
fn tp_group_for_device(device_idx: u32, placement: &skein_cost::cluster::Placement) -> Vec<u32> {
    let stage = device_idx / (placement.tp * placement.ep);
    let ep_idx = device_idx % placement.ep;
    (0..placement.tp)
        .map(|t| stage * placement.tp * placement.ep + t * placement.ep + ep_idx)
        .collect()
}

/// EP group this device belongs to. Fixes `(stage, tp_idx)` and varies
/// `ep_idx ∈ [0, ep)`.
fn ep_group_for_device(device_idx: u32, placement: &skein_cost::cluster::Placement) -> Vec<u32> {
    let stage = device_idx / (placement.tp * placement.ep);
    let tp_idx = (device_idx % (placement.tp * placement.ep)) / placement.ep;
    (0..placement.ep)
        .map(|e| stage * placement.tp * placement.ep + tp_idx * placement.ep + e)
        .collect()
}
