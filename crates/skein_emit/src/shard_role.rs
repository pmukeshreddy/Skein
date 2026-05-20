//! `ShardRole` — the per-`(param, device)` lowering decision.
//!
//! Resolution priority (early exits stop further work):
//!
//! 1. **Pipeline (PP)**. If the parameter belongs to a decoder block that's
//!    not assigned to this device's PP stage → `PipelineStageElsewhere`.
//!    Pre/post layers (embed, final norm, lm_head) live on stage 0 / stage
//!    pp-1 by convention.
//! 2. **Expert (EP)**, MoE-only. If the parameter is an MoE expert weight
//!    and `ep > 1`, the expert index is bucketed by `expert_idx % ep`. The
//!    device whose `ep_idx` matches owns it (`ExpertOwned`); the others
//!    see `ExpertElsewhere`. EP supersedes TP for expert weights — we do
//!    not stack TP-within-expert on top of EP (see `docs/lowering.md`).
//! 3. **Tensor (TP)**, applies to non-expert weights. Direction is read
//!    from the parameter's HF safetensors key:
//!    - `self_attn.q_proj`, `k_proj`, `v_proj`, `mlp.gate_proj`/`up_proj`,
//!      `expert.w1`/`w3`, `embed_tokens`, `lm_head` → column-parallel
//!      → `TpOutputShard` (split rows of `[out, in]`).
//!    - `self_attn.o_proj`, `mlp.down_proj`, `expert.w2` → row-parallel
//!      → `TpInputShard` (split columns).
//!    - All others (norms, MoE router gate) → `Replicated`.
//!
//!    `embed_tokens` and `lm_head` are vocab-parallel (column-parallel on
//!    their `[vocab, hidden]` row axis); the runtime reconstructs the full
//!    embedding with an AllReduce and the full logits with an AllGather.
//! 4. **Otherwise** → `Replicated`.
//!
//! The function is pure: no I/O, no global state, deterministic.

use skein_cost::cluster::{Cluster, Placement, block_to_stage};
use skein_ir::ir::{Layer, LayerKind, Param};
use skein_ir::plan::Plan;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardRole {
    /// Parameter is fully replicated on this device.
    Replicated,
    /// Output axis (rows of `[out, in]`) sharded across `group_size` devices;
    /// this device owns shard `group_index`.
    TpOutputShard { group_size: u32, group_index: u32 },
    /// Input axis (columns of `[out, in]`) sharded; this device owns shard
    /// `group_index`.
    TpInputShard { group_size: u32, group_index: u32 },
    /// MoE expert weight owned by this device.
    ExpertOwned { expert_idx: u32 },
    /// MoE expert weight owned by another device — skip lowering.
    ExpertElsewhere { expert_idx: u32 },
    /// The parameter's owning block belongs to a pipeline stage that's not
    /// on this device — skip lowering.
    PipelineStageElsewhere,
    /// Layer has no parameters (e.g. a marker or collective-only layer).
    NoParams,
}

/// Resolve the role of `param` (a member of `layer`) on `device_idx`.
pub fn shard_role_for_param(
    plan: &Plan,
    // Cluster is in the signature so future per-cluster placement variants
    // (e.g. tier-aware sharding) don't change every caller. Unused today.
    _cluster: &Cluster,
    ir: &skein_ir::ir::Graph,
    device_idx: u32,
    layer: &Layer,
    param: &Param,
) -> ShardRole {
    let placement = Placement::from_plan(plan);
    let Some((stage, tp_idx, ep_idx)) = placement.device_coords(device_idx) else {
        // Device is outside the placement (idle) — same effect as PP-elsewhere.
        return ShardRole::PipelineStageElsewhere;
    };
    let num_blocks = ir.meta.num_layers;

    // 1. Pipeline filter.
    let layer_stage = match layer.block_idx {
        Some(b) => block_to_stage(b, num_blocks, placement.pp),
        None => match &layer.kind {
            LayerKind::Embedding(_) => 0,
            LayerKind::Lmhead(_) => placement.pp - 1,
            // Final RmsNorm / other non-block layers live on stage 0 by
            // convention. Match `skein_cost` so the two stay aligned.
            _ => 0,
        },
    };
    if layer_stage != stage {
        return ShardRole::PipelineStageElsewhere;
    }

    let kind = param_kind_for(&param.name);

    // 2. Expert routing (MoE only).
    if let ParamKind::MoeExpert { expert_idx, role } = kind {
        if expert_idx >= ir.meta.num_experts.unwrap_or(0) as u32 {
            // Out-of-range expert index — treat as elsewhere to avoid silent
            // miswiring; callers can spot the issue via `EmitError` if they
            // try to declare the tensor downstream.
            return ShardRole::ExpertElsewhere { expert_idx };
        }
        if placement.ep > 1 {
            if expert_idx % placement.ep == ep_idx {
                return ShardRole::ExpertOwned { expert_idx };
            } else {
                return ShardRole::ExpertElsewhere { expert_idx };
            }
        }
        // ep == 1 — no expert sharding; fall through to TP rules using the
        // expert's role (w1/w3 column-parallel, w2 row-parallel).
        return tp_role_for(role, placement.tp, tp_idx);
    }

    // 3. TP for non-expert weights.
    if let ParamKind::TpDirectional(role) = kind {
        return tp_role_for(role, placement.tp, tp_idx);
    }

    // 4. Replicated (norms, MoE router gate, embed/lm_head).
    ShardRole::Replicated
}

/// Convert a [`TpDirection`] + placement into a [`ShardRole`]. With `tp == 1`
/// every param is replicated.
fn tp_role_for(role: TpDirection, tp: u32, tp_idx: u32) -> ShardRole {
    if tp == 1 {
        return ShardRole::Replicated;
    }
    match role {
        TpDirection::ColumnParallel => ShardRole::TpOutputShard {
            group_size: tp,
            group_index: tp_idx,
        },
        TpDirection::RowParallel => ShardRole::TpInputShard {
            group_size: tp,
            group_index: tp_idx,
        },
    }
}

/// Parameter category derived from its safetensors key. Drives the sharding
/// direction. Adding a new architecture means extending this function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// Standard TP-directional weight (Q/K/V/O, MLP w1/w2/w3, embed, lm_head).
    TpDirectional(TpDirection),
    /// MoE expert weight. `role` is the within-expert TP direction; the
    /// `expert_idx` is the routing key.
    MoeExpert { expert_idx: u32, role: TpDirection },
    /// Norm or MoE router gate — always replicated.
    Replicated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpDirection {
    /// Splits the *output* axis (rows of `[out, in]`). The output activation
    /// is sharded; the runtime issues an AllGather/no-op afterwards.
    ColumnParallel,
    /// Splits the *input* axis (columns). Partial sums require AllReduce.
    RowParallel,
}

/// Classify a Mixtral parameter name. The Mixtral importer in `skein_ir`
/// uses these exact HF safetensors keys; if a future architecture changes
/// the layout, extend this function rather than adding a special case in
/// `shard_role_for_param`.
pub fn param_kind_for(name: &str) -> ParamKind {
    // MoE experts: model.layers.<N>.block_sparse_moe.experts.<E>.<w1|w2|w3>.weight
    if let Some(rest) = name.split_once(".block_sparse_moe.experts.").map(|t| t.1) {
        let mut parts = rest.split('.');
        let idx_str = parts.next();
        let role_str = parts.next();
        if let (Some(idx), Some(role)) = (idx_str, role_str) {
            if let Ok(expert_idx) = idx.parse::<u32>() {
                let role = match role {
                    "w1" | "w3" => TpDirection::ColumnParallel,
                    "w2" => TpDirection::RowParallel,
                    // Unknown sub-key in an expert path — replicate as a
                    // safe default; callers can spot via test failure.
                    _ => return ParamKind::Replicated,
                };
                return ParamKind::MoeExpert { expert_idx, role };
            }
        }
        // Malformed expert key — replicate rather than crash.
        return ParamKind::Replicated;
    }

    // MoE router gate is small and replicated.
    if name.contains(".block_sparse_moe.gate") {
        return ParamKind::Replicated;
    }

    // Attention projections.
    if name.ends_with(".self_attn.q_proj.weight")
        || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight")
    {
        return ParamKind::TpDirectional(TpDirection::ColumnParallel);
    }
    if name.ends_with(".self_attn.o_proj.weight") {
        return ParamKind::TpDirectional(TpDirection::RowParallel);
    }

    // Dense MLP (non-MoE; reserved for future Llama-family models).
    if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight") {
        return ParamKind::TpDirectional(TpDirection::ColumnParallel);
    }
    if name.ends_with(".mlp.down_proj.weight") {
        return ParamKind::TpDirectional(TpDirection::RowParallel);
    }

    // Embedding and LM head are vocab-parallel: their vocab axis is the row
    // (output) axis of `[vocab, hidden]`, so column-parallel splits the vocab.
    // The embedding lookup masks out-of-range tokens and the runtime issues an
    // AllReduce; the LM head produces a logit shard that the runtime gathers.
    if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
        return ParamKind::TpDirectional(TpDirection::ColumnParallel);
    }

    // RmsNorm / final norm — replicated.
    if name.ends_with("_layernorm.weight") || name == "model.norm.weight" {
        return ParamKind::Replicated;
    }

    // Unknown — replicate as a safe default. Surface via test if a new
    // pattern slips through.
    ParamKind::Replicated
}
