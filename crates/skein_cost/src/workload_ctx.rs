//! Workload context — the `(batch, seq_len, kv_len)` triple the cost model
//! evaluates one forward pass at. Built from `Plan::batching` and
//! `CostConstants::representative_workload`.
//!
//! The cost model evaluates a *decode-step* cost (one token produced per
//! sequence per step) because that determines throughput at a fixed
//! latency SLO. `seq_len` is fixed to 1 and `kv_len` is taken from
//! `representative_workload.decode_kv_tokens`.

use skein_ir::plan::Plan;

use crate::constants::CostConstants;

#[derive(Debug, Clone, Copy)]
pub struct WorkloadCtx {
    /// Concurrent sequences in flight — `Plan::batching.max_batch()`.
    pub batch: u32,
    /// Tokens computed per forward pass. 1 for the decode-step model.
    pub seq_len: u32,
    /// Existing KV cache length at the time of the modelled step.
    pub kv_len: u32,
}

impl WorkloadCtx {
    /// Derive the representative decode-step context from a `Plan` and the
    /// global `CostConstants`.
    pub fn decode_step(plan: &Plan, constants: &CostConstants) -> Self {
        WorkloadCtx {
            batch: plan.batching.max_batch(),
            seq_len: 1,
            kv_len: constants.representative_workload.decode_kv_tokens,
        }
    }
}
