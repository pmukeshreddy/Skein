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
    /// kv_class)` pair — the CUDA Graphs dispatcher uses this to pick a
    /// captured graph.
    pub uniform_decode_size: Option<u32>,
}

/// Result of accepting a freshly-generated token for a request. The forward
/// driver uses this to stream the token and decide whether to retire.
pub struct AcceptOutcome {
    /// A clone of the request's token-stream sender. The driver sends the
    /// token on this *after* releasing the batcher lock (the send is async).
    pub sender: TokenSender,
    /// `true` once the request has emitted its full `max_output_tokens`.
    pub is_final: bool,
}

pub struct ContinuousBatcher {
    policy: BatchPolicy,
    estimator: LatencyEstimator,
    queue: queue::AdmissionQueue,
    inflight: InflightSet,
    workload: Workload,
    kv: Arc<Mutex<PagedKVAllocator>>,
    /// Benchmark concurrency override (env `SKEIN_CB_MAX_BATCH`). The compiled
    /// plan pins `max_batch = 1` for single-stream latency, which caps the
    /// scheduler to one in-flight request. This override raises the *scheduler's*
    /// concurrency so the continuous batcher actually interleaves N requests
    /// (mixed prefill/decode, shared paged KV). It is independent of the
    /// per-forward graph batch width (still 1): requests are run sequentially per
    /// step but join/leave continuously. `None` = use the plan's `max_batch`.
    max_batch_override: Option<u32>,
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
            max_batch_override: std::env::var("SKEIN_CB_MAX_BATCH")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&m| m >= 1),
        }
    }

    /// Decide what to do with an incoming request without admitting it.
    /// `admit` is the mutating counterpart.
    pub fn decide(&self, request: &IncomingRequest, now_ms: u64) -> AdmissionDecision {
        // Benchmark override: admit up to `m` concurrent immediately (no SLO
        // delay); requeue with `until_ms: 0` so the next `promote_ready` retries
        // as soon as a slot frees — no wall-clock advance required.
        if let Some(m) = self.max_batch_override {
            return if (self.inflight.len() as u32) < m {
                AdmissionDecision::Admit
            } else {
                AdmissionDecision::Delay { until_ms: 0 }
            };
        }
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

    /// The running token sequence for an in-flight request:
    /// `prompt_tokens` followed by every token generated so far. The forward
    /// driver feeds this to the executor; returns `None` if the request is
    /// not in flight.
    pub fn current_sequence(&self, id: RequestId) -> Option<Vec<u32>> {
        let req = self.inflight.get(id)?;
        let mut seq = Vec::with_capacity(req.request.prompt_tokens.len() + req.generated_tokens.len());
        seq.extend_from_slice(&req.request.prompt_tokens);
        seq.extend_from_slice(&req.generated_tokens);
        Some(seq)
    }

    /// Record a token the executor produced for `id`: append it to the
    /// request's output, advance `Prefill -> Decode` on the first token, and
    /// report whether the request has now emitted its full output. Returns
    /// `None` if the request is not in flight (e.g. already retired).
    pub fn accept_token(&mut self, id: RequestId, token: u32) -> Option<AcceptOutcome> {
        let policy_is_chunked = matches!(self.policy, BatchPolicy::ContinuousChunked { .. });
        let req = self.inflight.get_mut(id)?;
        req.generated_tokens.push(token);
        req.output_tokens_emitted += 1;
        // A prefill request transitions to decode once it emits its first
        // token. (Chunked prefill advances chunk-by-chunk elsewhere; here we
        // only flip when not mid-chunk.)
        if req.phase == Phase::Prefill && (!policy_is_chunked || req.chunks.is_none()) {
            req.phase = Phase::Decode;
        }
        let is_final = req.output_tokens_emitted >= req.request.max_output_tokens;
        Some(AcceptOutcome {
            sender: req.sender.clone(),
            is_final,
        })
    }

    /// Pull queued (delayed) requests whose `ready_at_ms` has passed and
    /// re-run admission against current load. Admitted requests move into the
    /// in-flight set; still-delayed ones are requeued; rejected ones have
    /// their stream closed. Called once per driver step.
    pub fn promote_ready(&mut self, now_ms: u64) {
        let mut budget = self.queue.len();
        while budget > 0 {
            budget -= 1;
            let Some(entry) = self.queue.pop_ready(now_ms) else {
                break;
            };
            // Honor the benchmark override (and the SLO/load policy otherwise) by
            // re-deciding through the same path as `admit`.
            let decision = self.decide(&entry.request, now_ms);
            match decision {
                AdmissionDecision::Admit => {
                    self.inflight
                        .insert(entry.request, entry.sender, self.policy, now_ms);
                }
                AdmissionDecision::Delay { until_ms } => {
                    self.queue.push(entry.request, entry.sender, until_ms);
                }
                AdmissionDecision::Reject { .. } => {
                    entry.sender.close();
                }
            }
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
