//! Metric structs the runtime records per step / per request.

use crate::types::RequestId;

#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub step_idx: u64,
    pub compute_us: f64,
    /// Phase B: real NCCL/comm time. Phase A: always 0 — the forward-pass
    /// driver isn't running, so no comm happens.
    pub comm_us: f64,
    pub kv_pages_in_use: u32,
    pub batch_size: u32,
    pub uniform_decode_size: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct RequestMetrics {
    pub request_id: RequestId,
    pub ttft_ms: f64,
    pub per_token_latency_ms: Vec<f64>,
    pub total_wall_ms: f64,
    pub prefix_cache_hit_tokens: u32,
    pub output_tokens: u32,
    /// Wall-clock millisecond timestamp when the request completed. Used
    /// by `ProfileHooks::recent_traces` to bound a recency window.
    pub completed_at_ms: u64,
}
