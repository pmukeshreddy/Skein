//! `ContinuousBatchDriver` — single-process continuous-batching execution over
//! paged KV, with optional CUDA-graph decode replay.
//!
//! This is the execution driver the [`ContinuousBatcher`] drives. It owns all
//! devices' paged [`SegmentRunner`]s (via [`LocalTopology`]) and runs the tp=2
//! Mixtral forward in one process. Each in-flight request has its own paged-KV
//! page table; the driver switches the active request per step
//! ([`SegmentRunner::activate_request`]) so several requests share the page pool
//! with cross-request prefix reuse.
//!
//! Scheduling (genuine continuous batching / iteration-level scheduling):
//!   * requests are admitted dynamically through the `ContinuousBatcher`;
//!   * every iteration composes a step from the in-flight set
//!     ([`ContinuousBatcher::next_step_batch`]) — a *mix* of prefilling and
//!     decoding requests;
//!   * each request advances one token (chunked prefill, chunk = 1, since the
//!     compiled graph is seq=1) using its own pages; new tokens are recorded
//!     back into the batcher, KV is grown, and finished requests are retired
//!     (freeing their pages for prefix reuse);
//!   * when every decoding request shares the same `(batch, kv_class)`
//!     (`uniform_decode_size`), the decode step is captured/replayed through the
//!     [`CudaGraphCache`] to cut per-launch overhead.
//!
//! The compiled graph processes one sequence per forward, so a batch step runs
//! its requests sequentially; this is iteration-level (continuous) batching at
//! the scheduler, which is what lets requests join/leave every step and share
//! the paged KV pool — independent of the per-forward batch width.

use std::collections::HashMap;
use std::time::Instant;

use skein_compile::{ComputeRuntime, SkeinArtifact, load_runtime_segments};

use crate::batcher::ContinuousBatcher;
use crate::cuda::cuda_graphs::{CudaGraphCache, GraphOutcome};
use crate::distributed::rank_executor::LocalSegments;
use crate::distributed::{LocalTopology, SegmentRunner};
use crate::error::RuntimeError;
use crate::token_stream::TokenStreamer;
use crate::types::{IncomingRequest, RequestId};

const INPUT_TOKENS: &str = "input_tokens";
const LOGITS: &str = "logits";

/// One submitted request plus its live decode state.
struct ReqState {
    order: usize,
    prompt: Vec<u32>,
    max_new: usize,
    /// Next slot to write (== absolute position of the token fed this step).
    position: usize,
    /// Token id to feed on the next step (prompt token during prefill, then the
    /// last generated token during decode).
    next_input: u32,
    matched_prefix: usize,
    prefill_steps: usize,
    generated: Vec<u32>,
}

/// Result for one completed request.
#[derive(Debug, Clone)]
pub struct BatchOutput {
    pub order: usize,
    pub request_id: u64,
    pub tokens: Vec<u32>,
    pub prefix_hit_tokens: usize,
    pub prefill_steps: usize,
    pub prompt_len: usize,
}

/// Aggregate driver telemetry.
#[derive(Debug, Clone, Default)]
pub struct DriverMetrics {
    pub batch_steps: u64,
    pub forward_steps: u64,
    pub prefill_forward_steps: u64,
    pub decode_forward_steps: u64,
    pub max_concurrent_inflight: usize,
    pub mixed_batch_steps: u64,
    pub graph_captures: u64,
    pub graph_replays: u64,
    pub total_compute_us: f64,
}

pub struct ContinuousBatchDriver {
    topo: LocalTopology,
    batcher: ContinuousBatcher,
    vocab: u32,
    state: HashMap<RequestId, ReqState>,
    /// Keep token streamers alive (the batcher holds senders); we drain tokens
    /// here so the channels don't fill, but the demo reads `BatchOutput`.
    _streamers: Vec<TokenStreamer>,
    graph_cache: Option<CudaGraphCache>,
    next_order: usize,
    metrics: DriverMetrics,
    outputs: Vec<BatchOutput>,
}

impl ContinuousBatchDriver {
    /// Load all device segments in this process and build the driver. `R` is the
    /// compute runtime (CUDA on the GPU build).
    pub fn load<R: ComputeRuntime + 'static>(
        artifact_dir: &std::path::Path,
        batcher: ContinuousBatcher,
        search_budget: usize,
        enable_cuda_graphs: bool,
    ) -> Result<Self, RuntimeError> {
        let artifact = SkeinArtifact::load(artifact_dir)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
        let vocab = artifact.plan.model_meta.vocab as u32;
        let sequencing = artifact.sequencing.clone();
        let per_device = load_runtime_segments::<R>(&artifact, search_budget)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
        let runners: Vec<SegmentRunner> = per_device.into_iter().map(SegmentRunner::new).collect();
        let topo = LocalTopology::new(runners, sequencing);
        let graph_cache = if enable_cuda_graphs {
            Some(CudaGraphCache::new()?)
        } else {
            None
        };
        Ok(Self {
            topo,
            batcher,
            vocab,
            state: HashMap::new(),
            _streamers: Vec::new(),
            graph_cache,
            next_order: 0,
            metrics: DriverMetrics::default(),
            outputs: Vec::new(),
        })
    }

    pub fn vocab(&self) -> u32 {
        self.vocab
    }

    pub fn metrics(&self) -> &DriverMetrics {
        &self.metrics
    }

    /// Admit a request: through the batcher (SLO / inflight) and the paged
    /// allocator on every device (prefix match + page allocation). Returns the
    /// prefix-cache hit length (consistent across devices).
    pub fn submit(&mut self, prompt: Vec<u32>, max_new: usize, now_ms: u64) -> Result<RequestId, RuntimeError> {
        let id = RequestId::next();
        let (streamer, sender) = TokenStreamer::paired();
        let incoming = IncomingRequest {
            id,
            prompt_tokens: prompt.clone(),
            max_output_tokens: max_new as u32,
            arrival_ms: now_ms,
        };
        let _decision = self.batcher.admit(incoming, sender, now_ms);
        self._streamers.push(streamer);

        // Admit on every device's paged allocator (deterministic → identical
        // matched length across devices, so they stay in lockstep).
        let mut matched = 0usize;
        for runner in self.topo.runners_mut() {
            matched = runner.admit_request(id, &prompt)?;
        }
        let prefill_start = matched.min(prompt.len().saturating_sub(1));
        self.state.insert(
            id,
            ReqState {
                order: self.next_order,
                prompt: prompt.clone(),
                max_new,
                position: prefill_start,
                next_input: prompt.get(prefill_start).copied().unwrap_or(0),
                matched_prefix: matched,
                prefill_steps: prompt.len() - prefill_start,
                generated: Vec::new(),
            },
        );
        self.next_order += 1;
        Ok(id)
    }

    /// Drive until every in-flight request has finished. Returns per-request
    /// outputs in submission order.
    pub fn run_to_completion(&mut self, now_ms: u64) -> Result<Vec<BatchOutput>, RuntimeError> {
        loop {
            self.batcher.promote_ready(now_ms);
            let step = self.batcher.next_step_batch();
            let ids: Vec<RequestId> = step
                .prefill_requests
                .iter()
                .chain(step.decode_requests.iter())
                .copied()
                .collect();
            if ids.is_empty() {
                break;
            }
            let inflight = ids.len();
            self.metrics.max_concurrent_inflight = self.metrics.max_concurrent_inflight.max(inflight);
            self.metrics.batch_steps += 1;
            if !step.prefill_requests.is_empty() && !step.decode_requests.is_empty() {
                self.metrics.mixed_batch_steps += 1;
            }

            // CUDA-graph eligibility: a pure-decode step where every request is
            // a single-token decode. `uniform_decode_size` is the batcher's
            // signal; the cache replays a captured graph for that key.
            let uniform = step.uniform_decode_size.filter(|_| step.prefill_requests.is_empty());

            for id in ids {
                self.run_one_request_step(id, uniform)?;
            }
        }
        // Stable order for the caller.
        self.outputs.sort_by_key(|o| o.order);
        Ok(std::mem::take(&mut self.outputs))
    }

    /// Run one forward step for request `id`: switch its paged KV active, feed
    /// its current input token at its position, walk the schedule (or replay a
    /// captured CUDA graph for a uniform decode step), then update its state and
    /// the batcher.
    fn run_one_request_step(
        &mut self,
        id: RequestId,
        uniform: Option<u32>,
    ) -> Result<(), RuntimeError> {
        let (position, next_input, prompt_len, in_prefill) = {
            let st = self.state.get(&id).ok_or(RuntimeError::UnknownRequest(id.0))?;
            (
                st.position,
                st.next_input,
                st.prompt.len(),
                st.position + 1 < st.prompt.len(),
            )
        };
        let is_decode = position >= prompt_len;

        // Switch every device to this request's pages; grow pages for a decode
        // slot (prefill slots were allocated at admit).
        for runner in self.topo.runners_mut() {
            runner.activate_request(id, position)?;
            if is_decode {
                runner.advance_kv()?;
            }
            runner.set_input_tokens(INPUT_TOKENS, vec![next_input as i32]);
            runner.set_position(position);
        }

        // Execute the schedule. For a uniform pure-decode step, route through the
        // CUDA-graph cache (capture on first sight of the key, replay after) so
        // decode steps replay a captured graph instead of re-recording kernels.
        // The cache itself decides replay vs capture vs eager and runs the
        // schedule closure when it cannot replay.
        let started = Instant::now();
        let graph_key = if is_decode { uniform.map(|k| (k, 0u32)) } else { None };
        match (graph_key, self.graph_cache.as_mut()) {
            (Some(key), Some(cache)) => {
                let topo = &mut self.topo;
                let outcome = cache.run_decode_step(key, &mut || topo.run_step().map_err(rt))?;
                match outcome {
                    GraphOutcome::Replayed => self.metrics.graph_replays += 1,
                    GraphOutcome::Captured => self.metrics.graph_captures += 1,
                    GraphOutcome::Eager => {}
                }
            }
            _ => self.topo.run_step().map_err(rt)?,
        }
        self.metrics.total_compute_us += started.elapsed().as_secs_f64() * 1e6;
        self.metrics.forward_steps += 1;
        if is_decode {
            self.metrics.decode_forward_steps += 1;
        } else {
            self.metrics.prefill_forward_steps += 1;
        }

        let logits = self.topo.runner(0).read(LOGITS).map_err(rt)?;
        let next = argmax(&logits, self.vocab as usize);

        // Update request state + batcher.
        let st = self.state.get_mut(&id).ok_or(RuntimeError::UnknownRequest(id.0))?;
        st.position += 1;
        if in_prefill {
            // Still consuming the prompt — feed the next prompt token, no emit.
            st.next_input = st.prompt[st.position];
            return Ok(());
        }
        // We just produced a token (first generated, or a decode token).
        st.generated.push(next);
        st.next_input = next;
        let _ = self.batcher.accept_token(id, next); // flips prefill->decode, counts output
        let done = st.generated.len() >= st.max_new;
        if done {
            let out = BatchOutput {
                order: st.order,
                request_id: id.0,
                tokens: st.generated.clone(),
                prefix_hit_tokens: st.matched_prefix,
                prefill_steps: st.prefill_steps,
                prompt_len,
            };
            self.outputs.push(out);
            self.state.remove(&id);
            let _ = self.batcher.retire(id);
            for runner in self.topo.runners_mut() {
                let _ = runner.release_request(id);
            }
        }
        Ok(())
    }
}

fn rt(e: crate::distributed::RankExecError) -> RuntimeError {
    RuntimeError::ServerInit(e.to_string())
}

fn argmax(values: &[f32], vocab: usize) -> u32 {
    let n = values.len().min(vocab.max(1));
    values
        .iter()
        .take(n)
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
