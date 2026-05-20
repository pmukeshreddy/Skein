//! Cluster-wide collective list — the `topology.json` the runtime issues
//! between graph invocations.
//!
//! The collectives are derived from the Plan in the same way
//! `skein_cost::comm::collectives_on_device` does, but here we enumerate
//! the cluster-wide list rather than the per-device subset. Execution
//! order: outermost iteration is the decoder block index (0..num_blocks),
//! inner iteration is within-block ordering (TP allreduces fire after the
//! Attention block's o_proj, then again after the MoE/MLP's down_proj;
//! EP all-to-alls fire around the MoE expert FFN; PP send/recvs fire at
//! stage boundaries).
//!
//! Determinism: same Plan + IR → byte-identical `Topology`. The artifact
//! hashing path (`Plan::content_hash`) depends on this.

use serde::{Deserialize, Serialize};

use skein_cost::Cluster;
use skein_cost::cluster::{Placement, block_to_stage};
use skein_cost::collectives::CollectiveKind;
use skein_ir::ir::{Graph, LayerKind};
use skein_ir::plan::Plan;
use skein_ir::types::Dtype;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Topology {
    pub collectives: Vec<TopologyEntry>,
    /// Inter-pool KV transfer. `None` when there is no prefill/decode
    /// disaggregation; populated when `Plan::disaggregation` is `Some(...)`.
    pub kv_transfer: Option<KvTransferProtocol>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyEntry {
    pub sequence_idx: u64,
    pub kind: CollectiveKind,
    pub participants: Vec<u32>,
    pub tensor_name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    /// Human-readable label naming the IR node this collective fires after.
    /// Useful for debugging traces; not consumed by the runtime.
    pub after_node: String,
}

/// Inter-pool KV transfer for prefill/decode disaggregation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KvTransferProtocol {
    RdmaWrite { layer_overlap: bool },
    NcclByLayer,
}

pub fn emit_topology(plan: &Plan, _cluster: &Cluster, ir: &Graph) -> Topology {
    let placement = Placement::from_plan(plan);
    let num_blocks = ir.meta.num_layers;
    let hidden = ir.meta.hidden;
    let batch = plan.batching.max_batch() as usize;

    let mut collectives: Vec<TopologyEntry> = Vec::new();
    let mut seq: u64 = 0;

    let io_dtype = plan
        .dtype_map
        .per_layer
        .first()
        .map(|p| p.activation)
        .unwrap_or(Dtype::Bf16);

    // Vocab-parallel embedding AllReduce on the first stage's TP group.
    if placement.tp > 1 {
        collectives.push(TopologyEntry {
            sequence_idx: seq,
            kind: CollectiveKind::RingAllReduce,
            participants: tp_group_leaders(0, &placement),
            tensor_name: "embed_out".to_string(),
            shape: vec![batch, 1, hidden],
            dtype: io_dtype,
            after_node: "embed_tokens".to_string(),
        });
        seq += 1;
    }

    // Iterate decoder blocks in execution order. Each block:
    //   (1) optional EP AllToAll dispatch before MoE
    //   (2) TP AllReduce after attention
    //   (3) optional EP AllToAll combine after MoE
    //   (4) TP AllReduce after MoE/MLP
    // Plus PP SendRecv at every stage boundary.
    for block in 0..num_blocks {
        let stage = block_to_stage(block, num_blocks, placement.pp);
        let attn_dtype = plan
            .dtype_map
            .per_layer
            .get(block)
            .map(|p| p.activation)
            .unwrap_or(Dtype::Bf16);

        // PP boundary: emit one SendRecv per stage edge crossing into this
        // block. Specifically, when this block is the first on its stage,
        // the previous stage's last device sends its hidden state.
        if placement.pp > 1 && stage > 0 {
            let first_on_stage = first_block_on_stage(stage, num_blocks, placement.pp);
            if block == first_on_stage {
                // Pair the tp/ep coords across PP groups: one Send/Recv per
                // (tp_idx, ep_idx) pair, enumerating every member of the PP
                // group. We emit one TopologyEntry covering the canonical
                // group leaders.
                let prev_leader = (stage - 1) * placement.tp * placement.ep;
                let curr_leader = stage * placement.tp * placement.ep;
                collectives.push(TopologyEntry {
                    sequence_idx: seq,
                    kind: CollectiveKind::SendRecv,
                    participants: vec![prev_leader, curr_leader],
                    tensor_name: format!("block_{block}_pp_boundary"),
                    shape: vec![batch, 1, hidden],
                    dtype: attn_dtype,
                    after_node: format!("stage_{}_to_{}", stage - 1, stage),
                });
                seq += 1;
            }
        }

        // EP dispatch (before MoE FFN, for MoE blocks only).
        if placement.ep > 1 {
            if let Some(moe) = ir
                .layers
                .iter()
                .find(|l| l.block_idx == Some(block) && matches!(l.kind, LayerKind::Moe(_)))
            {
                let LayerKind::Moe(cfg) = &moe.kind else {
                    unreachable!()
                };
                let ep_group = ep_group_leaders(stage, &placement);
                let bytes_shape = vec![batch, cfg.top_k, hidden];
                collectives.push(TopologyEntry {
                    sequence_idx: seq,
                    kind: CollectiveKind::AllToAll,
                    participants: ep_group.clone(),
                    tensor_name: format!("block_{block}_moe_dispatch"),
                    shape: bytes_shape.clone(),
                    dtype: attn_dtype,
                    after_node: format!("block_{block}_router"),
                });
                seq += 1;
            }
        }

        // TP AllReduce after the attention block's o_proj.
        if placement.tp > 1 {
            let tp_group = tp_group_leaders(stage, &placement);
            collectives.push(TopologyEntry {
                sequence_idx: seq,
                kind: CollectiveKind::RingAllReduce,
                participants: tp_group.clone(),
                tensor_name: format!("block_{block}_attn_out"),
                shape: vec![batch, 1, hidden],
                dtype: attn_dtype,
                after_node: format!("block_{block}_o_proj"),
            });
            seq += 1;
        }

        // EP combine (after MoE FFN).
        if placement.ep > 1 {
            if let Some(moe) = ir
                .layers
                .iter()
                .find(|l| l.block_idx == Some(block) && matches!(l.kind, LayerKind::Moe(_)))
            {
                let LayerKind::Moe(cfg) = &moe.kind else {
                    unreachable!()
                };
                let ep_group = ep_group_leaders(stage, &placement);
                collectives.push(TopologyEntry {
                    sequence_idx: seq,
                    kind: CollectiveKind::AllToAll,
                    participants: ep_group,
                    tensor_name: format!("block_{block}_moe_combine"),
                    shape: vec![batch, cfg.top_k, hidden],
                    dtype: attn_dtype,
                    after_node: format!("block_{block}_expert_out"),
                });
                seq += 1;
            }
        }

        // TP AllReduce after the MoE/MLP block's down projection.
        if placement.tp > 1 {
            let has_moe_or_mlp = ir.layers.iter().any(|l| {
                l.block_idx == Some(block)
                    && matches!(l.kind, LayerKind::Moe(_) | LayerKind::Mlp(_))
            });
            if has_moe_or_mlp {
                let tp_group = tp_group_leaders(stage, &placement);
                collectives.push(TopologyEntry {
                    sequence_idx: seq,
                    kind: CollectiveKind::RingAllReduce,
                    participants: tp_group,
                    tensor_name: format!("block_{block}_ffn_out"),
                    shape: vec![batch, 1, hidden],
                    dtype: attn_dtype,
                    after_node: format!("block_{block}_down_proj"),
                });
                seq += 1;
            }
        }
    }

    // Vocab-parallel logits AllGather on the last stage's TP group.
    if placement.tp > 1 {
        let last_stage = placement.pp - 1;
        collectives.push(TopologyEntry {
            sequence_idx: seq,
            kind: CollectiveKind::AllGather,
            participants: tp_group_leaders(last_stage, &placement),
            tensor_name: "logits".to_string(),
            shape: vec![batch, 1, ir.meta.vocab],
            dtype: io_dtype,
            after_node: "lm_head".to_string(),
        });
    }

    Topology {
        collectives,
        kv_transfer: None,
    }
}

/// First block_idx assigned to `stage` under even pipeline split.
fn first_block_on_stage(stage: u32, num_blocks: usize, pp: u32) -> usize {
    let pp = pp as usize;
    let base = num_blocks / pp;
    let rem = num_blocks % pp;
    let stage = stage as usize;
    let extra = stage.min(rem);
    stage * base + extra
}

/// Device indices that form the TP group for the given pipeline stage. We
/// use the lexicographic placement: stage occupies device indices
/// `[stage * tp * ep, (stage + 1) * tp * ep)`. The TP group leaders are the
/// `ep_idx = 0` members of each TP shard within the stage.
fn tp_group_leaders(stage: u32, placement: &Placement) -> Vec<u32> {
    (0..placement.tp)
        .map(|t| stage * placement.tp * placement.ep + t * placement.ep)
        .collect()
}

/// Device indices forming the EP group for the leader (`tp_idx = 0`) of
/// `stage`. The EP group is `ep_idx ∈ [0, ep)` at fixed `tp_idx`.
fn ep_group_leaders(stage: u32, placement: &Placement) -> Vec<u32> {
    (0..placement.ep)
        .map(|e| stage * placement.tp * placement.ep + e)
        .collect()
}
