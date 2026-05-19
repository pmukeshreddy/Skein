//! Continuous batcher: SLO-aware admission, in-flight set, retire, chunked
//! prefill state machine.

use std::sync::{Arc, Mutex};

use skein_cost::CostConstants;
use skein_ir::plan::Plan;
use skein_ir::types::BatchPolicy;
use skein_ir::workload::Workload;

use crate::error::RuntimeError;
use crate::kv::PagedKVAllocator;
use crate::token_stream::TokenSender;
use crate::types::{IncomingRequest, RequestId};

pub mod admission;
pub mod chunking;
pub mod inflight;
pub mod queue;
pub mod retire;

pub use admission::{AdmissionDecision, LatencyEstimator, RejectReason};
pub use chunking::{ChunkPlan, plan_chunks};
pub use inflight::{InflightRequest, InflightSet, Phase};

#[derive(Debug, Clone)]
pub struct StepBatch {
    pub prefill_requests: Vec<RequestId>,
    pub decode_requests: Vec<RequestId>,
    pub total_kv_pages: u32,
    /// Set when every decode in the batch shares the same `(batch_size,
    /// kv_class)` pair — Phase B's CUDA Graphs dispatch uses this to pick
    /// a captured graph.
    pub uniform_decode_size: Option<u32>,
}

pub struct ContinuousBatcher {
    policy: BatchPolicy,
    estimator: LatencyEstimator,
    queue: queue::AdmissionQueue,
    inflight: InflightSet,
    workload: Workload,
    kv: Arc<Mutex<PagedKVAllocator>>,
}

impl ContinuousBatcher {
    pub fn new(
        plan: &Plan,
        workload: &Workload,
        cost_constants: &CostConstants,
        kv: Arc<Mutex<PagedKVAllocator>>,
    ) -> Self {
        Self {
            policy: plan.batching,
            estimator: LatencyEstimator::from_cost_constants(cost_constants),
            queue: queue::AdmissionQueue::new(),
            inflight: InflightSet::new(),
            workload: workload.clone(),
            kv,
        }
    }

    /// Decide what to do with an incoming request without admitting it.
    /// `admit` is the mutating counterpart.
    pub fn decide(&self, request: &IncomingRequest, now_ms: u64) -> AdmissionDecision {
        admission::decide(
            request,
            now_ms,
            self.policy.max_batch(),
            self.inflight.len() as u32,
            &self.workload.slo,
            &self.estimator,
        )
    }

    /// Admit a request, returning the same decision as `decide`. On
    /// `Admit` the request is moved into the in-flight set and a token
    /// streamer sender is stored. On `Delay`/`Reject` the request goes
    /// into the queue (Delay) or is dropped (Reject).
    pub fn admit(
        &mut self,
        request: IncomingRequest,
        sender: TokenSender,
        now_ms: u64,
    ) -> AdmissionDecision {
        let decision = self.decide(&request, now_ms);
        match &decision {
            AdmissionDecision::Admit => {
                self.inflight.insert(request, sender, self.policy, now_ms);
            }
            AdmissionDecision::Delay { until_ms } => {
                self.queue.push(request, sender, *until_ms);
            }
            AdmissionDecision::Reject { .. } => {}
        }
        decision
    }

    /// Compose the next step's batch from the in-flight set. Prefills in
    /// `Phase::Prefill` contribute one chunk per step (chunked when the
    /// Plan's `BatchPolicy::ContinuousChunked`); decodes in `Phase::Decode`
    /// each contribute one token.
    pub fn next_step_batch(&mut self) -> StepBatch {
        let mut prefill_requests = Vec::new();
        let mut decode_requests = Vec::new();
        for (id, req) in self.inflight.iter() {
            match req.phase {
                Phase::Prefill => prefill_requests.push(*id),
                Phase::Decode => decode_requests.push(*id),
            }
        }
        let total_kv_pages = self.kv.lock().map(|a| a.in_use_pages()).unwrap_or(0);
        let uniform_decode_size = if decode_requests.is_empty() {
            None
        } else {
            Some(decode_requests.len() as u32)
        };
        StepBatch {
            prefill_requests,
            decode_requests,
            total_kv_pages,
            uniform_decode_size,
        }
    }

    /// Mark a request retired. Releases its KV pages and closes the
    /// token streamer.
    pub fn retire(&mut self, request_id: RequestId) -> Result<(), RuntimeError> {
        retire::retire(request_id, &self.kv, &mut self.inflight)
    }

    pub fn inflight_len(&self) -> usize {
        self.inflight.len()
    }

    pub fn queued_len(&self) -> usize {
        self.queue.len()
    }

    pub fn workload(&self) -> &Workload {
        &self.workload
    }

    /// Borrow the in-flight set for hot-swap drain inspection.
    pub fn inflight_ref(&self) -> &InflightSet {
        &self.inflight
    }
}
