//! `ContinuousBatchDriver` — single-process continuous-batching execution over
//! paged KV. The per-segment kernel CUDA graphs are Luminal's `CudaGraphOp`
//! (built once, replayed via `cuGraphLaunch` every forward); the driver reports
//! the real instantiate/launch counts from Luminal's counters.
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
//!   * each forward's kernels replay Luminal's per-segment CUDA graphs;
//!     `metrics.graph_captures/replays` are the real `cuGraphInstantiate` /
//!     `cuGraphLaunch` deltas measured around each step.
//!
//! The compiled graph processes one sequence per forward, so a batch step runs
//! its requests sequentially; this is iteration-level (continuous) batching at
//! the scheduler, which is what lets requests join/leave every step and share
//! the paged KV pool — independent of the per-forward batch width.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use skein_compile::{ComputeRuntime, SkeinArtifact, load_runtime_segments};
use skein_cost::collectives::CollectiveKind;
use skein_emit::segment::SequenceStep;

use crate::batcher::ContinuousBatcher;
use skein_compile::cuda_graph_exec_stats;
use crate::distributed::gpu_rank::{install_gate_buffer_on, schedule_top_k};
use crate::distributed::rank_executor::{LocalSegments, ResolvedSequenceStep};
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
    /// When set, accumulate real Luminal CUDA-graph instantiate/launch counts
    /// (deltas around each forward) into `metrics.graph_captures/replays`.
    track_graphs: bool,
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
        let num_layers = artifact.plan.model_meta.num_layers;

        // Sparse MoE re-lowers each device into gate/FFN-split segments with
        // MoeRoute steps the stored (dense) schedule lacks, so re-derive the
        // schedule to match the rebuilt segments (no recompile; same weights).
        let sparse = std::env::var_os("SKEIN_SPARSE_MOE").is_some();
        let ondevice = std::env::var_os("SKEIN_ONDEVICE_MOE").is_some();
        let spread = std::env::var_os("SKEIN_SPREAD_DEVICES").is_some();
        let schedule: Vec<SequenceStep> = if sparse {
            artifact
                .rebuild_sequencing()
                .map_err(|e| RuntimeError::ServerInit(e.to_string()))?
        } else {
            artifact.sequencing.clone()
        };

        // Only size-changing / final collectives (logits AllGather, Broadcast)
        // stay host-materialized; RingAllReduce tensors and internal activations
        // are device-resident (all-reduced in place across GPUs by the resolved
        // walk). The driver reads `logits` host-side, so mark it host too.
        let mut host_tensors: HashSet<String> = schedule
            .iter()
            .filter_map(|s| match s {
                SequenceStep::Collective {
                    collective, tensor, ..
                } if !matches!(collective, CollectiveKind::RingAllReduce) => Some(tensor.clone()),
                _ => None,
            })
            .collect();
        host_tensors.insert(LOGITS.to_string());

        let per_device = load_runtime_segments::<R>(&artifact, search_budget)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
        let mut runners: Vec<SegmentRunner> =
            per_device.into_iter().map(SegmentRunner::new).collect();

        // Per-device sparse bootstrap (mirrors the multi-process RankServer):
        // mark host tensors, free search arenas + materialize the resident expert
        // weights, resolve the schedule per runner, and install the resident
        // gate-scalar buffer on the runner's own GPU.
        let mut resolved_per: Vec<Vec<ResolvedSequenceStep>> = Vec::with_capacity(runners.len());
        for (d, runner) in runners.iter_mut().enumerate() {
            runner.set_host_tensors(host_tensors.clone());
            // Per-request device KV: each in-flight request gets its own KV
            // buffer so concurrent decode/prefill don't clobber each other.
            runner.set_paged_device_kv(true);
            if sparse || ondevice {
                runner.clear_intermediates();
                runner.materialize_weights();
            }
            let r = runner
                .resolve_schedule(&schedule)
                .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
            if let Some(top_k) = schedule_top_k(&r) {
                // Runner `d` lives on physical GPU `d` only when SKEIN_SPREAD_DEVICES
                // is set (else all on device 0). Place the gate buffer accordingly.
                let device = if spread { d } else { 0 };
                install_gate_buffer_on(runner, &r, d as u32, num_layers, top_k, device)?;
            }
            resolved_per.push(r);
        }

        let mut topo = LocalTopology::new(runners, schedule);
        topo.set_resolved(resolved_per);
        Ok(Self {
            topo,
            batcher,
            vocab,
            state: HashMap::new(),
            _streamers: Vec::new(),
            track_graphs: enable_cuda_graphs,
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

    /// Synchronous batched decode of `prompts.len()` real sequences in lockstep
    /// (must equal the graph's batch width). Truncates every prompt to the common
    /// min length L (no padding → clean synchronous attention), prefills all rows
    /// token-by-token to position L-1, then decodes `max_new` tokens with every
    /// row sharing the position. Returns (per-row generated tokens, decode seconds
    /// covering the `max_new` batched forwards). Real generation, real timing.
    pub fn run_batched_lockstep(
        &mut self,
        prompts: &[Vec<u32>],
        max_new: usize,
    ) -> Result<(Vec<Vec<u32>>, f64), RuntimeError> {
        let n = prompts.len();
        let l = prompts.iter().map(|p| p.len()).min().unwrap_or(1).max(1);
        let vocab = self.vocab as usize;
        for r in self.topo.runners_mut() {
            r.set_paged_device_kv(false);
            r.set_decode_batch(n);
        }
        let mut cur: Vec<i32> = prompts.iter().map(|p| p[0] as i32).collect();
        let mut genr: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut decode_start = Instant::now();
        for pos in 0..(l - 1 + max_new) {
            if pos == l - 1 {
                decode_start = Instant::now();
            }
            for runner in self.topo.runners_mut() {
                runner.set_input_tokens(INPUT_TOKENS, cur.clone());
                runner.set_position(pos);
            }
            self.topo.run_step().map_err(rt)?;
            let logits = self.topo.runner(0).read(LOGITS).map_err(rt)?;
            // The logits AllGather concatenates the TP ranks' vocab-parallel
            // partials RANK-major: the buffer is [num_ranks, batch, vocab_local]
            // (each rank's full [batch, vocab_local] block back to back), not
            // [batch, vocab]. De-interleave per row to argmax over the full vocab.
            let ranks = self.topo.num_devices().max(1);
            let vl = vocab / ranks; // vocab_local
            let mut next = vec![0i32; n];
            for (r, slot) in next.iter_mut().enumerate() {
                let mut best_i = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for rk in 0..ranks {
                    let base = (rk * n + r) * vl;
                    for j in 0..vl {
                        let v = logits[(base + j).min(logits.len().saturating_sub(1))];
                        if v > best_v {
                            best_v = v;
                            best_i = rk * vl + j;
                        }
                    }
                }
                *slot = best_i as i32;
            }
            if pos < l - 1 {
                for (r, c) in cur.iter_mut().enumerate() {
                    *c = prompts[r][pos + 1] as i32;
                }
            } else {
                for r in 0..n {
                    genr[r].push(next[r] as u32);
                    cur[r] = next[r];
                }
            }
        }
        Ok((genr, decode_start.elapsed().as_secs_f64()))
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

            for id in ids {
                self.run_one_request_step(id)?;
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
    fn run_one_request_step(&mut self, id: RequestId) -> Result<(), RuntimeError> {
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

        // Execute the schedule. The per-segment kernel CUDA graphs live inside
        // Luminal's `CudaGraphOp` (built once, replayed via `cuGraphLaunch` on
        // every later forward). We measure the *real* instantiate/launch deltas
        // around the step from Luminal's counters — no separate serving-level
        // graph, no fabricated counts.
        let started = Instant::now();
        let g0 = if self.track_graphs { cuda_graph_exec_stats() } else { (0, 0) };
        self.topo.run_step().map_err(rt)?;
        if self.track_graphs {
            let g1 = cuda_graph_exec_stats();
            self.metrics.graph_captures += g1.0.saturating_sub(g0.0);
            self.metrics.graph_replays += g1.1.saturating_sub(g0.1);
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
