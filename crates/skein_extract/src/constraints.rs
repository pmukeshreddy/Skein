//! Hard-constraint predicates. Each is a pure, deterministic function over
//! `(GlobalConfig, Cluster, Graph, CostModel)` — no I/O, no side effects.
//!
//! These predicates determine which outer candidates *cannot* satisfy any
//! per-layer dtype assignment. Soft constraints (memory + drift budgets)
//! belong inside the inner DP, where they get a chance to find a tighter
//! dtype map.

use skein_cost::cluster::Placement;
use skein_cost::memory::{block_kv_bytes, block_weight_bytes, non_decoder_weight_bytes};
use skein_cost::{Cluster, CostModel, WorkloadCtx};
use skein_ir::ir::Graph;
use skein_ir::types::Dtype;

use crate::candidate::GlobalConfig;

/// Reasons a global candidate was rejected. The outer loop's counters use
/// the dominant reason to annotate `NoFeasiblePlan` errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// `hidden_size % tp != 0`.
    TpDivisibility,
    /// `num_experts % ep != 0` (MoE only).
    EpDivisibility,
    /// `tp × pp × ep > num_devices`.
    DeviceCount,
    /// Even with the most aggressive dtype map (int4 weights, fp8 KV) the
    /// per-device peak memory exceeds the smallest device cap.
    GlobalMemory,
}

/// Apply every hard constraint. Returns `None` if the candidate survives,
/// or `Some(reason)` annotating the first violation.
pub fn reject(
    global: &GlobalConfig,
    cluster: &Cluster,
    ir: &Graph,
    cost_model: &CostModel,
) -> Option<RejectReason> {
    if !divisibility_tp(global, ir) {
        return Some(RejectReason::TpDivisibility);
    }
    if !divisibility_ep(global, ir) {
        return Some(RejectReason::EpDivisibility);
    }
    if !device_count(global, cluster) {
        return Some(RejectReason::DeviceCount);
    }
    if !global_memory_fits(global, cluster, ir, cost_model) {
        return Some(RejectReason::GlobalMemory);
    }
    None
}

/// Convenience wrapper for tests / docs: `true` iff every constraint passes.
pub fn satisfies_hard_constraints(
    global: &GlobalConfig,
    cluster: &Cluster,
    ir: &Graph,
    cost_model: &CostModel,
) -> bool {
    reject(global, cluster, ir, cost_model).is_none()
}

// --- individual predicates ---

/// TP shards weights along `hidden`. Indivisible TP would leave a ragged
/// remainder; reject.
pub fn divisibility_tp(global: &GlobalConfig, ir: &Graph) -> bool {
    ir.meta.hidden % global.parallelism.tp as usize == 0
}

/// EP shards MoE experts across devices. For non-MoE models the EP axis is
/// always 1, so this trivially passes.
pub fn divisibility_ep(global: &GlobalConfig, ir: &Graph) -> bool {
    let Some(n_experts) = ir.meta.num_experts else {
        // Not an MoE model; ep must be 1 — `enumerate_parallelism` already
        // restricts this, but check defensively.
        return global.parallelism.ep == 1;
    };
    n_experts % global.parallelism.ep as usize == 0
}

/// `tp × pp × ep` must not exceed the cluster's device count.
pub fn device_count(global: &GlobalConfig, cluster: &Cluster) -> bool {
    let used = global.parallelism.tp * global.parallelism.pp * global.parallelism.ep;
    used <= cluster.num_devices()
}

/// Optimistic memory check: even with the cheapest possible dtype map
/// (int4 weights, fp8_e4m3 KV) the per-device peak fits the tightest cap.
///
/// "Cheapest" here means smallest memory. If that fails, no inner DP choice
/// can rescue the global config — reject it before paying the DP cost.
pub fn global_memory_fits(
    global: &GlobalConfig,
    cluster: &Cluster,
    ir: &Graph,
    cost_model: &CostModel,
) -> bool {
    let placement = Placement {
        tp: global.parallelism.tp,
        pp: global.parallelism.pp,
        ep: global.parallelism.ep,
    };
    let wl = WorkloadCtx {
        batch: global.batch.max_batch(),
        seq_len: 1,
        kv_len: cost_model
            .constants()
            .representative_workload
            .decode_kv_tokens,
    };

    // Cheapest possible dtype map.
    let weight_dt = Dtype::Int4;
    let kv_dt = Dtype::Fp8E4m3;

    // Pipeline parallelism distributes whole decoder blocks across `pp` stages,
    // so each device holds ~num_blocks/pp of them. Divide the all-blocks sum by
    // `pp` to get the per-device total (pp=1 is a no-op). Non-decoder weights
    // (embedding on stage 0, lm_head on the last stage) are not distributed and
    // stay whole — the conservative bound.
    let pp = placement.pp.max(1) as u64;
    let mut total: u64 = 0;
    for b in 0..ir.meta.num_layers {
        total = total.saturating_add(block_weight_bytes(b, ir, weight_dt, placement) / pp);
        total = total.saturating_add(block_kv_bytes(ir, kv_dt, placement, &wl, global.kv_shard) / pp);
    }
    total = total.saturating_add(non_decoder_weight_bytes(ir, placement));

    total <= cluster.min_device_memory_bytes()
}
