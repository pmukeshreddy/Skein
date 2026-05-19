//! Memory + drift budgets the outer hands to the inner DP.
//!
//! For each surviving global config:
//!
//! - **Memory budget**: device cap minus the things the DP can't change —
//!   non-decoder weights (embedding, final norm, lm_head). KV cache and
//!   decoder-block weights are budgeted *inside* the DP, where the per-block
//!   choice actually affects them.
//! - **Drift budget**: the full `workload.slo.max_accuracy_drift`. The DP
//!   accumulates per-block drift contributions and must finish at or below
//!   this ceiling.

use skein_cost::cluster::Placement;
use skein_cost::memory::non_decoder_weight_bytes;
use skein_cost::{Cluster, CostModel};
use skein_ir::ir::Graph;
use skein_ir::workload::Workload;

use crate::candidate::GlobalConfig;

#[derive(Debug, Clone, Copy)]
pub struct Budgets {
    /// Maximum total decoder-block weight + KV memory per device.
    pub memory_bytes: u64,
    /// Maximum sum of per-block drift across the DP's accumulator.
    pub drift: f64,
}

pub fn budgets_after_global(
    global: &GlobalConfig,
    cluster: &Cluster,
    workload: &Workload,
    ir: &Graph,
    _cost_model: &CostModel,
) -> Budgets {
    let placement = Placement {
        tp: global.parallelism.tp,
        pp: global.parallelism.pp,
        ep: global.parallelism.ep,
    };
    let cap = cluster.min_device_memory_bytes();
    let reserved = non_decoder_weight_bytes(ir, placement);
    let memory_bytes = cap.saturating_sub(reserved);
    Budgets {
        memory_bytes,
        drift: workload.slo.max_accuracy_drift,
    }
}
