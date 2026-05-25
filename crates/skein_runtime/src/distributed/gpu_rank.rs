//! GPU multi-rank serving: one process per GPU, joined over NCCL.
//!
//! This is the assembly of the distributed pieces into a runnable multi-GPU
//! forward path. Each rank process (launched by [`super::launcher`] with
//! `CUDA_VISIBLE_DEVICES` set so its device 0 is its physical GPU):
//!
//! 1. exchanges the NCCL unique id over the shared rendezvous file,
//! 2. joins the communicator (`ncclCommInitRank`),
//! 3. loads **only its own device's** compiled segments onto its GPU,
//! 4. drives the global schedule with [`RankExecutor`], meeting peers at each
//!    collective over NCCL.
//!
//! Lockstep decode: after the final logits all-gather every rank in the TP
//! group holds identical full logits, so every rank samples the same next
//! token deterministically — no per-token broadcast needed. The only cross-rank
//! input is the prompt, which rank 0 broadcasts at request start
//! ([`RankServer::broadcast_prompt`]).
//!
//! `#[cfg(feature = "cuda")]`-gated: it needs `CudaComputeRuntime` + NCCL, so
//! it is built and validated on the GPU host.

use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use skein_compile::{
    CudaComputeRuntime, DEFAULT_SEARCH_BUDGET, SkeinArtifact, cuda_graph_exec_stats,
    load_device_prefill_segments, load_device_runtime_segments,
};
use skein_cost::collectives::CollectiveKind;
use skein_emit::segment::SequenceStep;

use crate::kv_cache::KvKind;

use crate::cuda::nccl::NcclCollective;
use crate::distributed::{
    CollectiveError, LocalSegments, RankCollective, RankExecutor, ResolvedSequenceStep,
    SegmentRunner, WorldLayout,
};
use crate::error::RuntimeError;
use crate::speculative;
use crate::tokenizer::SkeinTokenizer;

const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(120);
const INPUT_TOKENS: &str = "input_tokens";
const LOGITS: &str = "logits";

/// One rank's GPU serving state.
pub struct RankServer {
    layout: WorldLayout,
    executor: RankExecutor<SegmentRunner>,
    collective: NcclCollective,
    /// Decode schedule with all names pre-resolved to ids (artifact stays
    /// String-keyed; this is built once at bootstrap). The hot path walks this.
    schedule_resolved: Vec<ResolvedSequenceStep>,
    vocab: u32,
    /// Optional batched-prefill executor (seq=N graph sharing the decode graph's
    /// weights by device pointer). Gated by `SKEIN_BATCHED_PREFILL=<seq>`. When
    /// present and the prompt length matches `prefill_seq`, the whole prompt is
    /// prefilled in ONE forward instead of token-by-token.
    prefill: Option<RankExecutor<SegmentRunner>>,
    /// The prefill executor's own resolved schedule (same names → same ids, but
    /// resolved against the prefill runner's resident weights). `Some` iff
    /// `prefill` is `Some`.
    prefill_schedule_resolved: Option<Vec<ResolvedSequenceStep>>,
    prefill_seq: usize,
    num_layers: usize,
    /// SKEIN_CAPTURE: decode-step counter and the schedule index where the logits
    /// all-gather (a host sync, can't be captured) begins. The pre-split steps are
    /// captured into a full-step graph and replayed; the post-split (all-gather +
    /// logits read) runs on the host every step.
    decode_step: usize,
    /// Full-step CUDA-graph capture window `[capture_lo, capture_hi)` into
    /// `schedule_resolved`: the contiguous device-only region replayed per token.
    /// Under PP the boundary `SendRecv` (host-staged) bounds it — the sender's
    /// send follows its compute (`[0, sr)`), the receiver's recv precedes it
    /// (`[sr+1, len)`). Under TP it ends after the last RingAllReduce (which is
    /// routed onto the capture stream via shm). The pre-region `[0, lo)` and the
    /// post-region `[hi, len)` run on the host each step, outside capture.
    capture_lo: usize,
    capture_split: usize,
    /// Pipeline-parallel coordination: under PP only the last stage computes
    /// logits, so it samples the next token and broadcasts it to the earlier
    /// stages (which need it for their next embed). `pp == 1` => TP, every rank
    /// has full logits and samples locally (no broadcast).
    pp: u32,
    is_last_stage: bool,
    last_stage_root: usize,
    /// Compiled decode batch width (`plan.batching.max_batch()`): the number of
    /// rows the decode graph processes per forward. `generate_batched` packs this
    /// many sequences into each captured step.
    batch_width: usize,
    /// SKEIN_DEVICE_LOGITS: the next-token index computed on-device by the logits
    /// all-gather + argmax (greedy, TP path). Set by `forward_step` after the
    /// forward, consumed by `sample_and_sync` in place of a host argmax. `None`
    /// when the lever is off or on the PP path.
    device_token: std::cell::Cell<Option<u32>>,
}

impl RankServer {
    /// Bootstrap this rank: NCCL id exchange → `ncclCommInitRank` → load this
    /// rank's device segments onto its GPU. The process must already have
    /// `CUDA_VISIBLE_DEVICES` set (the launcher does this).
    pub fn bootstrap(
        artifact_dir: &Path,
        layout: WorldLayout,
        rendezvous_path: &Path,
    ) -> Result<Self, RuntimeError> {
        // 1. NCCL unique id: leader generates + publishes; others fetch.
        let id_bytes = if layout.is_leader() {
            let bytes = NcclCollective::new_id_bytes().map_err(to_rt)?;
            super::rendezvous::publish(rendezvous_path, &bytes).map_err(to_rt)?;
            bytes
        } else {
            super::rendezvous::fetch(rendezvous_path, RENDEZVOUS_TIMEOUT).map_err(to_rt)?
        };

        // 2. Join the communicator on this rank's GPU (device 0 of this proc).
        let collective =
            NcclCollective::init(layout.rank, layout.world_size, &id_bytes).map_err(to_rt)?;

        // 3. Load only this rank's device segments (rank == device index).
        let artifact = SkeinArtifact::load(artifact_dir)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
        let segments = load_device_runtime_segments::<CudaComputeRuntime>(
            &artifact,
            layout.rank,
            DEFAULT_SEARCH_BUDGET,
        )
        .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;

        let mut executor = RankExecutor::new(layout.rank, SegmentRunner::new(segments));

        // With sparse MoE the segments are re-lowered with a gate/FFN split +
        // MoeRoute steps the serialized (dense) schedule lacks, so re-derive the
        // schedule to match the rebuilt segments — no recompile, existing weights
        // are reused. Dense path uses the stored schedule unchanged.
        let schedule: Vec<SequenceStep> = if std::env::var_os("SKEIN_SPARSE_MOE").is_some() {
            artifact
                .rebuild_sequencing()
                .map_err(|e| RuntimeError::ServerInit(e.to_string()))?
        } else {
            artifact.sequencing.clone()
        };

        // Only all-gather / broadcast collective tensors stay host (size-changing
        // / final; their read/write host path is kept). RingAllReduce tensors and
        // every internal activation handoff stay device-resident — RingAllReduce
        // is all-reduced in place on the device by the rank executor.
        let mut host_tensors: HashSet<String> = schedule
            .iter()
            .filter_map(|s| match s {
                SequenceStep::Collective {
                    collective, tensor, ..
                } if !matches!(collective, CollectiveKind::RingAllReduce) => Some(tensor.clone()),
                _ => None,
            })
            .collect();
        // `forward_step` reads the final logits host-side via `read(LOGITS)`, which
        // only sees `f32_slots`. Under TP the logits all-gather already host-stages
        // them; under PP with tp=1 there is no all-gather (logits are the last
        // stage's plain segment output → device_slots), so mark LOGITS host here to
        // materialize it to f32_slots. Idempotent if the all-gather already added it.
        //
        // SKEIN_DEVICE_LOGITS: keep LOGITS device-resident (bf16) so the all-gather
        // + argmax run on-device (see rank_executor's device-logits path); marking
        // it host would force the full-vocab f32 D2H this lever removes. Only the
        // TP path is changed — PP (no all-gather) still needs the host read.
        let device_logits = std::env::var_os("SKEIN_DEVICE_LOGITS").is_some();
        if device_logits {
            // Keep LOGITS device-resident (bf16). TP (pp=1): the AllGather output
            // stays in device_slots for the on-device gather+argmax. PP: the last
            // stage's full-vocab logits stay device-resident so the next token is
            // argmaxed on the GPU (read-only) instead of read full-vocab to host.
            // The collective filter above may have added it (TP AllGather); remove
            // it either way so it is not materialized to f32_slots.
            host_tensors.remove(LOGITS);
        } else {
            host_tensors.insert(LOGITS.to_string());
        }
        // SKEIN_DEVICE_SENDRECV: keep the pipeline-parallel boundary handoff
        // (the SendRecv carry tensors) device-resident so the producer stage's
        // output stays in `device_slots` (ncclSend straight from it) and the
        // consumer stage binds the ncclRecv'd device buffer — no host round-trip.
        if std::env::var_os("SKEIN_DEVICE_SENDRECV").is_some() {
            for s in &schedule {
                if let SequenceStep::Collective {
                    collective: CollectiveKind::SendRecv,
                    tensor,
                    ..
                } = s
                {
                    host_tensors.remove(tensor);
                }
            }
        }
        executor.runner_mut().set_host_tensors(host_tensors.clone());

        // Sparse / on-device MoE: the FFN segments declare every expert (so they
        // load GPU-resident — sparse binds the top-k slots; on-device GLUMoE
        // `gather`s them by device index), so lazy-on-execute would never upload
        // the experts. Free the per-segment search arenas FIRST (otherwise they
        // coexist with the ~46 GB/card of weights and OOM — the on-device path
        // hit signal-9 OOM at serve without this), then materialize so the
        // route/gather resolves each expert's resident device pointer.
        if std::env::var_os("SKEIN_SPARSE_MOE").is_some()
            || std::env::var_os("SKEIN_ONDEVICE_MOE").is_some()
        {
            executor.runner_mut().clear_intermediates();
            executor.runner_mut().materialize_weights();
        }

        // Batched prefill (gated): build a seq=N prefill graph that SHARES the
        // decode graph's resident weights by device pointer (no 2nd 47GB copy).
        // The decode + prefill graphs have the same segmentation, so they reuse
        // the same global schedule. Prompt length must equal `prefill_seq`.
        let prefill_seq = std::env::var("SKEIN_BATCHED_PREFILL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 1)
            .unwrap_or(0);
        let mut prefill = if prefill_seq > 1 {
            // Decode weights must be GPU-resident before the prefill graph shares
            // their pointers.
            executor.runner_mut().materialize_weights();
            let prefill_segs = load_device_prefill_segments::<CudaComputeRuntime>(
                &artifact,
                layout.rank,
                DEFAULT_SEARCH_BUDGET,
                prefill_seq,
                executor.runner().segments(),
            )
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
            let mut pe = RankExecutor::new(layout.rank, SegmentRunner::new(prefill_segs));
            pe.runner_mut().set_host_tensors(host_tensors.clone());
            pe.runner_mut().clear_intermediates(); // free prefill search arenas
            Some(pe)
        } else {
            None
        };

        // Free the per-segment search/compile arenas now (re-allocated lazily on
        // execute). Otherwise both graphs' ~15 GB of arenas stay resident on top
        // of the 47 GB weights and the first prefill execute OOMs the 96 GB card.
        if prefill.is_some() {
            executor.runner_mut().clear_intermediates();
        }

        // Resolve the String-keyed schedule to ids ONCE, now that names are
        // interned and (for sparse MoE) weights are resident — so the decode hot
        // path does zero string hashing per token. Decode and prefill resolve
        // against their own runners (same names → same ids; expert weight
        // pointers resolved per runner).
        let schedule_resolved = executor
            .runner_mut()
            .resolve_schedule(&schedule)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
        let prefill_schedule_resolved = match prefill.as_mut() {
            Some(pe) => Some(
                pe.runner_mut()
                    .resolve_schedule(&schedule)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?,
            ),
            None => None,
        };

        // Sparse MoE: install one resident bf16 gate-scalar buffer per runner and
        // bind each FFN segment's gate inputs to fixed offsets ONCE. After this,
        // route_moe writes a layer's gates with one async H2D instead of a tiny
        // per-slot H2D each (no MoeRoute steps on the dense path → no buffer).
        let num_layers = artifact.plan.model_meta.num_layers;
        let rank = layout.rank as u32;
        if let Some(top_k) = schedule_top_k(&schedule_resolved) {
            install_gate_buffer(executor.runner_mut(), &schedule_resolved, rank, num_layers, top_k)?;
        }
        if let (Some(pe), Some(pr)) = (prefill.as_mut(), prefill_schedule_resolved.as_ref()) {
            if let Some(top_k) = schedule_top_k(pr) {
                install_gate_buffer(pe.runner_mut(), pr, rank, num_layers, top_k)?;
            }
        }

        // Optional: isolate pure all-reduce latency (no decode work => no rank
        // drift) to separate NCCL transport cost from in-context drift. Both
        // ranks reach this together (post NCCL init), so the tight loop stays
        // synced. hidden=4096 bf16 is the decode all-reduce size.
        if std::env::var_os("SKEIN_ALLREDUCE_BENCH").is_some() {
            collective.bench_all_reduce(4096, 2000);
            collective.bench_all_reduce(4096, 2000);
        }

        // Split point for SKEIN_CAPTURE: capture everything up to and including the
        // LAST RingAllReduce (the 32 layers — all device-resident, no host sync).
        // The tail after it (final norm, logits projection, the logits all-gather,
        // and the logits read) stays on the host: those segments produce host
        // handoffs whose output-capture does a D2H, which is illegal mid-capture.
        let capture_split = schedule_resolved
            .iter()
            .rposition(|s| {
                matches!(
                    s,
                    ResolvedSequenceStep::Collective {
                        collective: CollectiveKind::RingAllReduce,
                        ..
                    }
                )
            })
            .map(|i| i + 1)
            .unwrap_or(schedule_resolved.len());

        // Pipeline-parallel stage coords (ranks are stage-major: rank = stage*tp*ep
        // + tp_idx*ep + ep_idx, so stage = rank / (tp*ep)).
        let pp = artifact.plan.parallelism.pp;
        let tp = artifact.plan.parallelism.tp;
        let ep = artifact.plan.parallelism.ep;
        let group = (tp * ep).max(1) as usize;
        let stage = (layout.rank as usize) / group;
        let is_last_stage = stage as u32 == pp.saturating_sub(1);
        let last_stage_root = (pp.saturating_sub(1) * tp * ep) as usize;

        // PP capture window: the compute region between this rank's boundary
        // SendRecv handoffs. The recv (this rank as receiver) precedes the
        // compute → capture starts after it; the send (this rank as sender)
        // follows the compute → capture ends before it. Stage 0: `[0, send)`.
        // Last stage: `[recv+1, len)`. TP (pp==1): `[0, capture_split)`.
        let (capture_lo, capture_hi) = if pp > 1 {
            let mut lo = 0usize;
            let mut hi = schedule_resolved.len();
            for (i, s) in schedule_resolved.iter().enumerate() {
                if let ResolvedSequenceStep::Collective {
                    collective: CollectiveKind::SendRecv,
                    participants,
                    ..
                } = s
                {
                    let sender = participants.first().copied().unwrap_or(0) as usize;
                    let receiver = participants.get(1).copied().unwrap_or(0) as usize;
                    if receiver == layout.rank {
                        lo = i + 1;
                    }
                    if sender == layout.rank && i < hi {
                        hi = i;
                    }
                }
            }
            (lo, hi)
        } else {
            (0, capture_split)
        };

        // Clamp the window to a device-only run: stop before the first segment
        // whose output is host-materialized (the boundary carry on stage 0, the
        // LM-head logits on the last stage). Its device→host copy is illegal
        // during stream capture; excluded, it runs in the un-captured post-region
        // (where the SendRecv send / logits read already live). This makes the
        // ~15/16-block compute region capturable without touching the carry/
        // logits/SendRecv host paths.
        let capture_hi = {
            let runner = executor.runner();
            let mut hi = capture_hi;
            for i in capture_lo..capture_hi {
                if let ResolvedSequenceStep::ExecuteSegment {
                    device_idx,
                    segment_idx,
                } = &schedule_resolved[i]
                    && *device_idx as usize == layout.rank
                    && runner.segment_has_host_output(*segment_idx)
                {
                    hi = i;
                    break;
                }
            }
            hi
        };

        // SKEIN_DEVICE_SENDRECV / SKEIN_DEVICE_LOGITS make a PP boundary/last-stage
        // tensor device-resident, which flips `segment_has_host_output` for that
        // stage to false and would otherwise extend the full-step capture window to
        // cover a stage's segment. skein's stream-capture of a stage conflicts with
        // luminal's internal per-segment CUDA graphs (CUDA_ERROR_STREAM_CAPTURE_-
        // UNSUPPORTED), and PP already decodes without full-step capture, so keep
        // the window empty — these levers win on host-traffic, not stage capture.
        let capture_hi = if pp > 1
            && (std::env::var_os("SKEIN_DEVICE_SENDRECV").is_some()
                || std::env::var_os("SKEIN_DEVICE_LOGITS").is_some())
        {
            capture_lo
        } else {
            capture_hi
        };

        if std::env::var_os("SKEIN_FI_LOG").is_some() {
            eprintln!(
                "SKEIN_SPLIT capture_lo={capture_lo} capture_hi={capture_hi} capture_split={capture_split} len={} rank={}",
                schedule_resolved.len(),
                layout.rank
            );
            for (i, s) in schedule_resolved.iter().enumerate() {
                let desc = match s {
                    ResolvedSequenceStep::ExecuteSegment { device_idx, segment_idx } => {
                        format!("ExecuteSegment dev={device_idx} seg={segment_idx}")
                    }
                    ResolvedSequenceStep::Collective { collective, participants, tensor, elems } => {
                        format!("Collective {collective:?} parts={participants:?} tensor={tensor:?} elems={elems}")
                    }
                    ResolvedSequenceStep::MoeRoute { ffn_segment_idx, block, .. } => {
                        format!("MoeRoute ffn_seg={ffn_segment_idx} block={block}")
                    }
                };
                eprintln!("SKEIN_SCHED[{i}] rank={} {desc}", layout.rank);
            }
        }
        Ok(Self {
            layout,
            executor,
            collective,
            schedule_resolved,
            vocab: artifact.plan.model_meta.vocab as u32,
            prefill,
            prefill_schedule_resolved,
            prefill_seq,
            num_layers: artifact.plan.model_meta.num_layers,
            decode_step: 0,
            capture_lo,
            capture_split: capture_hi,
            pp,
            is_last_stage,
            last_stage_root,
            batch_width: artifact.plan.batching.max_batch() as usize,
            device_token: std::cell::Cell::new(None),
        })
    }

    pub fn batch_width(&self) -> usize {
        self.batch_width
    }

    pub fn layout(&self) -> WorldLayout {
        self.layout
    }

    pub fn vocab(&self) -> u32 {
        self.vocab
    }

    /// The runtime KV paging page size (tokens per page) on this rank.
    pub fn kv_page_size(&self) -> usize {
        self.executor.runner().kv().page_size()
    }

    /// Distribute the prompt from rank 0 to every rank so they decode in
    /// lockstep. Rank 0 passes `Some(prompt)`; the others pass `None` and
    /// receive it. Length is broadcast first (variable-length prompts), then
    /// the token ids.
    pub fn broadcast_prompt(&self, prompt: Option<&[u32]>) -> Result<Vec<u32>, RuntimeError> {
        // Broadcast the length as a 1-element buffer.
        let mut len_buf = vec![prompt.map(|p| p.len()).unwrap_or(0) as f32];
        self.collective.broadcast(&mut len_buf, 0).map_err(to_rt)?;
        let len = len_buf[0] as usize;

        // Broadcast the token ids (as f32; ids are small, exact in f32).
        let mut tok_buf = match prompt {
            Some(p) => p.iter().map(|&t| t as f32).collect::<Vec<_>>(),
            None => vec![0.0; len],
        };
        if tok_buf.len() != len {
            tok_buf.resize(len, 0.0);
        }
        self.collective.broadcast(&mut tok_buf, 0).map_err(to_rt)?;
        Ok(tok_buf.into_iter().map(|f| f as u32).collect())
    }

    /// Run one cached-decode step: feed the single current `token` at absolute
    /// `position`, driving this rank's segments + NCCL collectives over the
    /// schedule. The attention segments read the accumulated KV cache (past =
    /// `position` tokens) and append this token's K/V. Returns this rank's logits
    /// — full vocab on TP ranks after the logits all-gather.
    pub fn forward_step(&mut self, token: u32, position: usize) -> Result<Vec<f32>, RuntimeError> {
        self.forward_step_tokens(&[token], position)
    }

    /// Batched cached-decode step: feed `tokens` (one per batch row, length =
    /// the compiled graph's batch width) at the shared absolute `position`,
    /// driving this rank's segments + collectives over the schedule with the
    /// SAME full-step CUDA-graph capture/replay as the single-token path. Returns
    /// this rank's logits (full vocab after the all-gather, `[batch, vocab]`
    /// rank-major across TP ranks → de-interleave per row at the call site).
    pub fn forward_step_tokens(&mut self, tokens: &[u32], position: usize) -> Result<Vec<f32>, RuntimeError> {
        {
            let runner = self.executor.runner_mut();
            runner.set_input_tokens(INPUT_TOKENS, tokens.iter().map(|&t| t as i32).collect());
            runner.set_position(position);
        }
        // SKEIN_CAPTURE full-step graph: capture the device-only compute window
        // `[capture_lo, capture_hi)` once (after a few warmup steps so the arena /
        // handoff buffers are stable), then replay it per token with one
        // cuGraphLaunch. The host pre-region (the receiver's boundary recv) and
        // post-region (the sender's boundary send, or TP's logits all-gather +
        // read) run on the host every step, outside the capture.
        let capture_at: usize = std::env::var("SKEIN_CAPTURE_AT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let lo = self.capture_lo;
        // SKEIN_CAPTURE_HI: bisection knob — cap the captured window's upper bound
        // so the segments in [hi, capture_split) run via run_segment (host) instead
        // of the graph. Lets us find the first segment whose replay is wrong.
        let cap_hi_override: Option<usize> = std::env::var("SKEIN_CAPTURE_HI")
            .ok()
            .and_then(|v| v.parse().ok());
        let hi = match cap_hi_override {
            Some(h) => h.min(self.capture_split),
            None => self.capture_split,
        };
        let len = self.schedule_resolved.len();
        // Capture only when there is a proper device-only window. Under PP the
        // window is always device-only; under TP require it to stop before the
        // host logits tail (`hi < len`).
        let window_ok = hi > lo && (self.pp > 1 || hi < len);
        let capture = std::env::var_os("SKEIN_CAPTURE").is_some()
            && window_ok
            && std::env::var_os("SKEIN_NO_SPLIT").is_none();
        // Captured-graph mode: run_segment must skip the per-token `input_tokens`
        // feed and `set_decode_position` memcpy so they aren't recorded into the
        // full-step graph (a recorded host→device copy bakes in a now-freed host
        // source and replays garbage). flush_step_device_inputs writes them into
        // the persistent device buffers every step instead.
        self.executor.runner_mut().set_capturing(capture);
        if capture {
            self.decode_step += 1;
            // Refresh this step's per-token device inputs (token id + decode
            // position) into their persistent device buffers, OUTSIDE the captured
            // region, so the captured kernels read fresh values on every replay.
            if std::env::var_os("SKEIN_NO_FLUSH").is_none() {
                let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
                self.executor
                    .runner_mut()
                    .flush_step_device_inputs(&toks_i32, position);
            }
            // Host pre-region (e.g. last stage's boundary recv of the carry).
            if lo > 0 {
                self.executor
                    .run(&self.schedule_resolved[..lo], &self.collective)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
            }
            // Captured/replayed compute region.
            let dbg = std::env::var_os("SKEIN_FI_LOG").is_some() && self.layout.rank == 0;
            if self.executor.runner().has_captured() {
                if dbg && self.decode_step <= capture_at + 3 {
                    eprintln!("SKEIN_PHASE step={} REPLAY pos={position}", self.decode_step);
                }
                let ok = self.executor.runner().replay_captured();
                if dbg && !ok {
                    eprintln!("SKEIN_PHASE step={} REPLAY-RETURNED-FALSE", self.decode_step);
                }
            } else if self.decode_step == capture_at {
                if dbg {
                    eprintln!("SKEIN_PHASE step={} CAPTURE pos={position}", self.decode_step);
                }
                self.executor.runner().begin_capture();
                self.executor
                    .run(&self.schedule_resolved[lo..hi], &self.collective)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                self.executor.runner().end_capture();
                // Stream capture RECORDS the region without executing it, so the
                // captured buffers still hold the previous step's values. Replay
                // the just-captured graph once now so THIS step actually computes
                // its output — otherwise the host post-region (and the KV-cache
                // write for this position) run on stale data, permanently
                // corrupting the cache slot at the capture position and degrading
                // every subsequent decode step.
                self.executor.runner().replay_captured();
            } else {
                self.executor
                    .run(&self.schedule_resolved[lo..hi], &self.collective)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
            }
            // Host post-region (e.g. stage 0's boundary send; TP's logits tail).
            if hi < len {
                self.executor
                    .run(&self.schedule_resolved[hi..], &self.collective)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
            }
        } else {
            self.executor
                .run(&self.schedule_resolved, &self.collective)
                .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
        }
        // Under PP only the last stage produces logits; earlier stages return
        // empty (the next token is broadcast to them by `sample_and_sync`).
        if self.pp > 1 && !self.is_last_stage {
            return Ok(Vec::new());
        }
        // SKEIN_DEVICE_LOGITS: compute the next token on-device and skip the
        // full-vocab host logits read.
        if std::env::var_os("SKEIN_DEVICE_LOGITS").is_some() {
            // TP (pp=1): the executor's AllGather hook already argmaxed the gathered
            // logits and stashed the token on the collective.
            if let Some(tok) = self.collective.take_device_token() {
                self.device_token.set(Some(tok));
                return Ok(Vec::new());
            }
            // PP last stage (no AllGather): the full-vocab logits are this stage's
            // own device-resident output — argmax them on the GPU (read-only) and
            // return the 4-byte token via `device_token`.
            if let Some((_stale_ptr, e)) = self.executor.runner().output_device_ptr(LOGITS) {
                // `output_device_ptr` is a stale output slot for a host-staged
                // handoff. D2D-copy the *computed* logits into a scratch buffer
                // (the live-value path KV writes use), then argmax that. Avoids the
                // full-vocab host read entirely; only the 4-byte token comes back.
                let dest = self.collective.logits_scratch_ptr(e);
                let copied = dest != 0
                    && self
                        .executor
                        .runner()
                        .copy_output_to_device(LOGITS, dest, e * 2);
                if std::env::var_os("SKEIN_FI_LOG").is_some() {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static ONCE: AtomicBool = AtomicBool::new(false);
                    if !ONCE.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "SKEIN_DEVICE_LOGITS(PP): elems={e} vocab={} copied={copied}",
                            self.vocab
                        );
                    }
                }
                if copied {
                    let tok = unsafe { self.collective.logits_argmax_local_device(dest, e) }
                        .map_err(to_rt)?;
                    self.device_token.set(Some(tok));
                    return Ok(Vec::new());
                }
            }
        }
        self.executor
            .runner()
            .read(LOGITS)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))
    }

    /// Sample the next token and make every rank agree on it. Under TP each rank
    /// already holds full logits, so it samples locally. Under PP only the last
    /// stage has logits, so it samples and broadcasts the token to the earlier
    /// stages (which need it for their next embed). Collective: all ranks call it.
    fn sample_and_sync(&self, logits: &[f32]) -> Result<u32, RuntimeError> {
        if self.pp <= 1 {
            // SKEIN_DEVICE_LOGITS: the next token was argmaxed on-device this step;
            // consume it (clearing the slot) instead of a host argmax over a
            // full-vocab f32 vector that was never materialized.
            if let Some(tok) = self.device_token.take() {
                return Ok(tok);
            }
            return Ok(argmax(logits));
        }
        // PP: only the last stage has logits. Under SKEIN_DEVICE_LOGITS it already
        // argmaxed them on-device (device_token); else host argmax. Broadcast the
        // chosen token to the earlier stages (which need it for their next embed).
        let next = if self.is_last_stage {
            self.device_token.take().unwrap_or_else(|| argmax(logits))
        } else {
            0
        };
        let mut buf = [next as f32];
        self.collective
            .broadcast(&mut buf, self.last_stage_root)
            .map_err(to_rt)?;
        Ok(buf[0] as u32)
    }

    /// Batched prefill: process the WHOLE prompt (`prefill_seq` tokens) in ONE
    /// forward through the seq=N prefill graph (which shares the decode graph's
    /// weights), write the N tokens' K/V into the decode runner's paged cache
    /// (slots 0..N), and return the last token's logits. Replaces N sequential
    /// `forward_step` prefill calls. Requires `self.prefill` to be present and
    /// `prompt_tokens.len() == self.prefill_seq`.
    fn forward_prefill(&mut self, prompt_tokens: &[u32]) -> Result<Vec<f32>, RuntimeError> {
        let n = prompt_tokens.len();
        // Take the prefill executor out so its &mut doesn't alias self.executor
        // below; put it back before returning.
        let mut prefill = self.prefill.take().expect("prefill executor present");
        {
            let runner = prefill.runner_mut();
            runner.set_input_tokens(
                INPUT_TOKENS,
                prompt_tokens.iter().map(|&t| t as i32).collect(),
            );
            runner.set_prefill_capture(true);
        }
        let prefill_schedule = self
            .prefill_schedule_resolved
            .as_ref()
            .expect("prefill schedule present when prefill executor is");
        let run_res = prefill
            .run(prefill_schedule, &self.collective)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()));
        if let Err(e) = run_res {
            self.prefill = Some(prefill);
            return Err(e);
        }
        // Land the prompt's K/V into the decode runner's paged cache (the active
        // request's pages cover slots 0..n). Each kvcache_{k|v}_{layer} output is
        // [n, kv_dim]; write it row by row into slots 0..n.
        for layer in 0..self.num_layers {
            for (tag, kind) in [("k", KvKind::Key), ("v", KvKind::Value)] {
                let name = format!("kvcache_{tag}_{layer}");
                if let Some(kv) = prefill.runner().read_handoff(&name) {
                    if n == 0 || kv.len() % n != 0 {
                        continue;
                    }
                    let kvd = kv.len() / n;
                    for slot in 0..n {
                        self.executor.runner_mut().write_kv_slot(
                            kind,
                            layer,
                            slot,
                            &kv[slot * kvd..(slot + 1) * kvd],
                        );
                    }
                }
            }
        }
        // SKEIN_DEVICE_LOGITS (TP): the prefill all-gather already argmaxed the
        // first token on-device and stashed it; take it and return empty logits
        // (consumed via `device_token` in `sample_and_sync`). No host logits read
        // — under the lever LOGITS is device-resident, so `read_handoff` would only
        // see this rank's un-gathered shard.
        if self.pp <= 1 && std::env::var_os("SKEIN_DEVICE_LOGITS").is_some() {
            if let Some(tok) = self.collective.take_device_token() {
                self.device_token.set(Some(tok));
                prefill.runner_mut().clear_intermediates();
                self.prefill = Some(prefill);
                return Ok(Vec::new());
            }
        }
        let logits = prefill.runner().read_handoff(LOGITS);
        // Free the prefill arena before decode so it isn't resident alongside the
        // decode arena + the 47 GB weights.
        prefill.runner_mut().clear_intermediates();
        self.prefill = Some(prefill);
        logits.ok_or_else(|| RuntimeError::ServerInit("prefill produced no logits".into()))
    }

    /// Compute next-token logits for an entire `seq` from scratch: resets the KV
    /// cache and prefills `seq` token-by-token, returning the logits after its
    /// last token. Used by the speculative loop, which probes arbitrary candidate
    /// sequences and so cannot reuse the running cache.
    pub fn forward_full(&mut self, seq: &[u32]) -> Result<Vec<f32>, RuntimeError> {
        self.executor.runner_mut().reset_kv_cache();
        let mut logits = Vec::new();
        for (pos, &tok) in seq.iter().enumerate() {
            logits = self.forward_step(tok, pos)?;
        }
        Ok(logits)
    }

    /// Greedy lockstep cached decode of `max_new_tokens` from `prompt_tokens`,
    /// backed by the **paged KV cache with cross-request prefix reuse**. The
    /// request is admitted through the paged allocator: any prompt prefix whose
    /// KV is still resident in cached pages (from an earlier request) is reused
    /// — those tokens are NOT recomputed; prefill starts after the matched
    /// prefix. The remaining prompt tokens are prefilled (writing their KV into
    /// freshly allocated pages); the logits after the last prompt token predict
    /// the first generated token. Decode then feeds one token per step, growing
    /// pages as the sequence crosses page boundaries. Every rank runs this
    /// identically (same prompt → same admit → same matched prefix → same
    /// logits → same argmax), so ranks stay in lockstep without per-token
    /// communication, and each rank's paged allocator evolves identically.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
    ) -> Result<GenResult, RuntimeError> {
        // Release any prior active request (its pages stay cached for reuse).
        self.executor.runner_mut().reset_kv_cache();
        if prompt_tokens.is_empty() {
            return Ok(GenResult::default());
        }

        // Snapshot real Luminal CUDA-graph counters so we can report how many
        // graph instantiations vs. replays this whole generation actually did.
        let graph_start = cuda_graph_exec_stats();

        // Admit through the paged allocator: prefix match + page allocation.
        let request_id = crate::types::RequestId::next();
        let matched = self
            .executor
            .runner_mut()
            .begin_request(request_id, prompt_tokens)?;
        // Always run at least the final prompt token so we get its logits, even
        // if the whole prompt was prefix-matched.
        let prefill_start_pos = matched.min(prompt_tokens.len() - 1);

        // Prefill the un-cached suffix. Time it as TTFT (prompt seen -> first
        // token's logits ready). Pages for the prompt were allocated at admit.
        let mut position = prefill_start_pos;
        let mut logits = Vec::new();
        let prefill_start = Instant::now();
        // Batched prefill (gated): if the prefill graph is loaded, there is no
        // resident prefix to reuse (prefill_start_pos == 0), and the prompt
        // length matches the prefill graph's seq, process the WHOLE prompt in
        // one forward instead of the per-token loop below.
        let batched = self.prefill.is_some()
            && prefill_start_pos == 0
            && prompt_tokens.len() == self.prefill_seq;
        let prefill_steps = if batched {
            logits = self.forward_prefill(prompt_tokens)?;
            position = prompt_tokens.len();
            1
        } else {
            for &tok in &prompt_tokens[prefill_start_pos..] {
                logits = self.forward_step(tok, position)?;
                position += 1;
            }
            prompt_tokens.len() - prefill_start_pos
        };
        let ttft = prefill_start.elapsed();

        // Decode: argmax the current logits, emit, and feed it back as the next
        // token at the running position. Grow the request's pages by one token
        // per step (paged KV). Time each decode step (TPOT).
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut step_times: Vec<Duration> = Vec::new();
        // Snapshot host<->device traffic counters across the decode loop so we
        // can report H2D/D2H bytes + host materializations PER GENERATED TOKEN.
        let perf_before = crate::perf_counters::snapshot();
        // True end-to-end decode wall: includes sample_and_sync (the broadcast
        // that, under PP, waits for the last stage to compute the token). The
        // per-step `step_times` below time only forward_step, which under PP is
        // just THIS rank's stage — so the leader's tpot understates the true rate.
        let decode_wall = Instant::now();
        for i in 0..max_new_tokens {
            let next = self.sample_and_sync(&logits)?;
            generated.push(next);
            if i + 1 < max_new_tokens {
                // Allocate the page covering this new token's slot before writing.
                self.executor.runner_mut().advance_kv()?;
                let t = Instant::now();
                logits = self.forward_step(next, position)?;
                step_times.push(t.elapsed());
                position += 1;
            }
        }
        if self.layout.is_leader() && std::env::var_os("SKEIN_PERF").is_some() {
            let w = decode_wall.elapsed().as_secs_f64();
            let toks = generated.len().saturating_sub(1) as f64;
            eprintln!(
                "SKEIN_PERF_TRUE: single-stream end-to-end decode (incl. sample_and_sync) tokens={toks} wall_s={w:.4} true_tokens_per_s={:.2}",
                if w > 0.0 { toks / w } else { 0.0 }
            );
        }

        // Per-token host<->device traffic over the decode loop (the round-trips
        // the device-resident handoff work will remove).
        let perf_pt =
            crate::perf_counters::snapshot().per_token(perf_before, step_times.len() as u64);

        let pages_in_use = self.executor.runner().kv_pages_in_use();
        // Release the request: its pages return to the cache so the next request
        // sharing this prompt's prefix can reuse them.
        self.executor.runner_mut().end_request();

        // Real CUDA-graph activity for this generation (delta of the Luminal
        // counters): instantiations are first-build / shape-change rebuilds,
        // launches are graph replays. A healthy decode shows launches growing
        // by ~(segments x decode_steps) while instantiations stay near the
        // one-time build count.
        let graph_end = cuda_graph_exec_stats();
        let graph_instantiates = graph_end.0.saturating_sub(graph_start.0);
        let graph_launches = graph_end.1.saturating_sub(graph_start.1);

        let mut ms: Vec<f64> = step_times.iter().map(|d| d.as_secs_f64() * 1e3).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| -> f64 {
            if ms.is_empty() {
                0.0
            } else {
                ms[((p * (ms.len() as f64 - 1.0)).round() as usize).min(ms.len() - 1)]
            }
        };
        let decode_total: f64 = ms.iter().sum::<f64>() / 1e3;
        let decode_toks = step_times.len() as f64;
        let tput = if decode_total > 0.0 {
            decode_toks / decode_total
        } else {
            0.0
        };
        let result = GenResult {
            tokens: generated,
            prefix_hit_tokens: matched,
            prefill_steps,
            prompt_tokens: prompt_tokens.len(),
            ttft_ms: ttft.as_secs_f64() * 1e3,
            tpot_p50_ms: pct(0.50),
            tpot_p95_ms: pct(0.95),
            decode_tokens_per_s: tput,
            kv_pages_in_use: pages_in_use,
            graph_instantiates,
            graph_launches,
        };
        if self.layout.is_leader() {
            tracing::info!(
                prompt_tokens = result.prompt_tokens,
                prefix_cache_hit_tokens = result.prefix_hit_tokens,
                prefill_steps = result.prefill_steps,
                kv_pages_in_use = result.kv_pages_in_use,
                ttft_ms = result.ttft_ms,
                decode_steps = step_times.len(),
                tpot_p50_ms = result.tpot_p50_ms,
                tpot_p95_ms = result.tpot_p95_ms,
                decode_tokens_per_s = result.decode_tokens_per_s,
                cuda_graph_instantiations = result.graph_instantiates,
                cuda_graph_replays = result.graph_launches,
                h2d_bytes_per_token = perf_pt.h2d_bytes,
                d2h_bytes_per_token = perf_pt.d2h_bytes,
                host_materializations_per_token = perf_pt.host_materializations,
                segment_launches_per_token = perf_pt.segment_launches,
                "SKEIN_PERF: paged-KV cached-decode timing (single in-flight request, greedy)"
            );
        }
        Ok(result)
    }

    /// Batched cached-decode of `prompts.len()` sequences in lockstep through the
    /// batch=N decode graph, WITH the full-step CUDA-graph capture (one
    /// `cuGraphLaunch`/token) + shm device all-reduce — the product path. Every
    /// row shares the decode position (the graph's `position` is a scalar), so the
    /// prompts are truncated to their common min length L for a clean synchronous
    /// prefill, then `max_new` tokens are decoded with all rows advancing together.
    /// Returns (per-row generated tokens, decode seconds over the `max_new`
    /// captured forwards). TP only (pp==1); each rank holds vocab-parallel logits
    /// gathered rank-major `[ranks, N, vocab_local]`, de-interleaved per row here.
    pub fn generate_batched(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
    ) -> Result<(Vec<Vec<u32>>, f64), RuntimeError> {
        let n = prompts.len();
        if n == 0 {
            return Ok((Vec::new(), 0.0));
        }
        let vocab = self.vocab as usize;
        let ranks = (self.layout.world_size as usize).max(1);
        let vl = (vocab / ranks).max(1); // vocab_local per rank

        // FULL-CONTEXT mixed-length batching (the default): every row feeds ONE
        // token per step — its own prompt token while prefilling, then its own
        // generated token once decoding — so after `step` steps every row holds
        // exactly `step` tokens and the graph's shared `position` scalar is
        // correct for ALL rows (no per-row positions, no recompile, NO truncation
        // to a common length). A short-prompt row simply starts generating while
        // longer-prompt rows are still consuming their prompt.
        let plen: Vec<usize> = prompts.iter().map(|p| p.len().max(1)).collect();
        let lmax = *plen.iter().max().unwrap();

        self.executor.runner_mut().reset_kv_cache();
        self.executor.runner_mut().set_paged_device_kv(false);
        self.executor.runner_mut().set_decode_batch(n);

        let mut cur: Vec<u32> = prompts.iter().map(|p| p[0]).collect();
        let mut genr: Vec<Vec<u32>> = vec![Vec::new(); n];
        // Decode wall = the final `max_new_tokens` steps, during which the
        // longest-prompt row is decoding (all rows have finished prefill by then),
        // so it is a clean steady-state batched-decode measurement at full context.
        let mut decode_start = Instant::now();
        let total_steps = (lmax - 1) + max_new_tokens;
        // STEP-6 proof counters. ONE `forward_step_tokens` call == ONE batched
        // graph step that advances ALL `n` rows together; it is NOT a loop of
        // `n` single-row forwards. `forward_steps` counts graph steps (not
        // per-request forwards); `decode_tokens_emitted` counts generated tokens.
        // Invariant proven in the leader log below: decode_tokens_emitted ≈
        // active_decode_batch_size × decode_forward_steps — each steady-state
        // graph step yields N decode tokens, not N steps yielding one each.
        let mut forward_steps = 0usize;
        let mut decode_forward_steps = 0usize;
        let mut decode_tokens_emitted = 0usize;
        let batch_log = std::env::var_os("SKEIN_BATCH_LOG").is_some() && self.layout.is_leader();
        for pos in 0..total_steps {
            if pos == lmax - 1 {
                decode_start = Instant::now();
            }
            let logits = self.forward_step_tokens(&cur, pos)?;
            forward_steps += 1;
            // De-interleave the rank-major all-gathered logits per row and argmax
            // over the full vocab. (Every TP rank holds the same gathered logits,
            // so all ranks pick identical next tokens → stays in lockstep.)
            let mut next = vec![0u32; n];
            for (r, slot) in next.iter_mut().enumerate() {
                let mut best_v = f32::NEG_INFINITY;
                let mut best_i = 0usize;
                for rk in 0..ranks {
                    let base = (rk * n + r) * vl;
                    for j in 0..vl {
                        let idx = base + j;
                        if idx < logits.len() && logits[idx] > best_v {
                            best_v = logits[idx];
                            best_i = rk * vl + j;
                        }
                    }
                }
                *slot = best_i as u32;
            }
            // Per-row update: still in this row's prompt → feed its next prompt
            // token (no emit); otherwise this output is one of its generated
            // tokens → record (until it has max_new) and feed it back.
            let mut active_decode = 0usize; // rows that emitted a decode token THIS step
            for r in 0..n {
                if pos + 1 < plen[r] {
                    cur[r] = prompts[r][pos + 1];
                } else {
                    if genr[r].len() < max_new_tokens {
                        genr[r].push(next[r]);
                        active_decode += 1;
                    }
                    cur[r] = next[r];
                }
            }
            decode_tokens_emitted += active_decode;
            if pos >= lmax - 1 {
                decode_forward_steps += 1;
            }
            if batch_log {
                eprintln!(
                    "SKEIN_BATCH_LOG step pos={pos} forward_steps=1 \
                     active_decode_batch_size={active_decode} decode_tokens_this_step={active_decode}"
                );
            }
        }
        let decode_s = decode_start.elapsed().as_secs_f64();
        self.executor.runner_mut().set_decode_batch(1);
        if self.layout.is_leader() {
            // PROOF that batching is real: `forward_steps` graph steps produced
            // `decode_tokens_emitted` tokens with up to N rows advancing PER step
            // — not one graph step per request-token. In the steady-state window
            // (pos >= lmax-1) every step's active_decode_batch_size == N.
            let tps = if decode_s > 0.0 {
                decode_tokens_emitted as f64 / decode_s
            } else {
                0.0
            };
            tracing::info!(
                active_decode_batch_size = n,
                forward_steps,
                decode_forward_steps,
                decode_tokens_emitted,
                decode_s,
                aggregate_decode_tokens_per_s = tps,
                "SKEIN_PERF_BATCHED: ONE forward_step_tokens per step drives all N rows (batched decode)"
            );
        }
        Ok((genr, decode_s))
    }

    /// 1F1B pipeline-parallel decode across the two PP stages: drive
    /// `prompts.len()` concurrent requests (microbatches) so BOTH stages — and
    /// thus both GPUs — compute simultaneously on different microbatches, instead
    /// of the single-stream path where the stages run in turns and only one GPU
    /// is busy per token. Targets pp==2 (stage 0 = sender rank, stage 1 = last
    /// stage). Needs >= 2 microbatches: the pipeline must have slack for a
    /// microbatch's sampled token to travel back from stage 1 to stage 0 before
    /// that microbatch's next forward — so stage 0 issues another microbatch's
    /// stage in the meantime. Falls back to sequential single-stream otherwise.
    ///
    /// The token feedback (stage 1 -> stage 0) and the activation handoff (stage
    /// 0 -> stage 1) are exchanged in ONE NCCL group per step
    /// ([`RankCollective::send_recv_f32`]); two separate blocking calls would
    /// deadlock. Both ranks build the identical round-robin job order so the
    /// send/recv streams stay matched.
    ///
    /// Returns one [`GenResult`] per prompt (tokens populated on stage 0 — which
    /// receives every sampled token — and the last stage). The leader logs the
    /// aggregate decode throughput = (microbatches * decode-tokens-each) / decode
    /// wall time.
    pub fn generate_pipelined(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
    ) -> Result<Vec<GenResult>, RuntimeError> {
        let n = prompts.len();

        // This rank's compute-only steps (schedule minus the boundary SendRecv)
        // and the carry handoff (id, element count, sender rank, receiver rank).
        let compute: Vec<ResolvedSequenceStep> = self
            .schedule_resolved
            .iter()
            .filter(|s| {
                !matches!(
                    s,
                    ResolvedSequenceStep::Collective {
                        collective: CollectiveKind::SendRecv,
                        ..
                    }
                )
            })
            .cloned()
            .collect();
        let carry = self.schedule_resolved.iter().find_map(|s| match s {
            ResolvedSequenceStep::Collective {
                collective: CollectiveKind::SendRecv,
                participants,
                tensor,
                elems,
            } => Some((
                *tensor,
                *elems,
                participants[0] as usize,
                participants[1] as usize,
            )),
            _ => None,
        });
        // Not a 2-stage PP plan, or too few microbatches to fill the pipeline →
        // run each prompt single-stream (still correct, just no stage overlap).
        let needs_fallback = self.pp != 2 || n < 2 || carry.is_none();
        if needs_fallback {
            let mut out = Vec::with_capacity(n);
            for p in prompts {
                out.push(self.generate(p, max_new_tokens)?);
            }
            return Ok(out);
        }
        let (carry_id, carry_elems, sender, receiver) = carry.unwrap();
        let is_first = self.layout.rank == sender;
        let is_last = self.layout.rank == receiver;

        struct Mb {
            id: crate::types::RequestId,
            base_position: usize,
            first_token: u32,
            generated: Vec<u32>,
        }
        let mut mbs: Vec<Mb> = Vec::with_capacity(n);

        // --- PREFILL: lockstep through both stages (full schedule incl. SendRecv),
        // one microbatch at a time. Establishes each request's KV + first token. ---
        self.executor.runner_mut().reset_kv_cache();
        for p in prompts.iter() {
            let id = crate::types::RequestId::next();
            // admit (keeps prior microbatches in-flight) + activate, NOT
            // begin_request — begin_request releases the previous active request,
            // which would evict every earlier microbatch before decode.
            let matched = self.executor.runner_mut().admit_request(id, p)?;
            self.executor.runner_mut().activate_request(id, matched)?;
            let start = matched.min(p.len().saturating_sub(1));
            let mut pos = start;
            let mut logits = Vec::new();
            for &tok in &p[start..] {
                logits = self.forward_step(tok, pos)?;
                pos += 1;
            }
            let first = self.sample_and_sync(&logits)?;
            mbs.push(Mb {
                id,
                base_position: pos,
                first_token: first,
                generated: vec![first],
            });
        }

        // --- DECODE: 1F1B pipeline. jobs[i] = microbatch index, round-robin so a
        // microbatch's token has `n-1` other-microbatch jobs to return in. ---
        let per_mb = max_new_tokens.saturating_sub(1);
        let mut jobs: Vec<usize> = Vec::with_capacity(n * per_mb);
        for _ in 0..per_mb {
            for m in 0..n {
                jobs.push(m);
            }
        }
        let njobs = jobs.len();
        let decode_start = Instant::now();

        let pipe_timing = std::env::var_os("SKEIN_PIPE_TIMING").is_some();
        let (mut t_setup, mut t_compute, mut t_read, mut t_comm) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        if njobs == 0 {
            // no decode steps beyond the prefill token
        } else if is_first {
            // Stage 0: run stage-0 compute, then send the activation forward and
            // receive the previous job's sampled token back (one NCCL group).
            let mut results: Vec<u32> = vec![0u32; njobs];
            for i in 0..njobs {
                let m = jobs[i];
                let k = i / n;
                let id = mbs[m].id;
                let input = if k == 0 { mbs[m].first_token } else { results[i - n] };
                let pos = mbs[m].base_position + k;
                let ts = Instant::now();
                {
                    let r = self.executor.runner_mut();
                    r.activate_request(id, pos)?;
                    r.advance_kv()?;
                    r.set_input_tokens(INPUT_TOKENS, vec![input as i32]);
                    r.set_position(pos);
                }
                if pipe_timing { t_setup += ts.elapsed().as_secs_f64() * 1e3; }
                let tc = Instant::now();
                self.executor
                    .run(&compute, &self.collective)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                if pipe_timing { t_compute += tc.elapsed().as_secs_f64() * 1e3; }
                let tr = Instant::now();
                let buf = self
                    .executor
                    .runner()
                    .read_by_id(carry_id)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                if pipe_timing { t_read += tr.elapsed().as_secs_f64() * 1e3; }
                let tm = Instant::now();
                if i == 0 {
                    self.collective.send_f32(&buf, receiver).map_err(to_rt)?;
                } else {
                    let tok = self
                        .collective
                        .send_recv_f32(&buf, receiver, receiver, 1)
                        .map_err(to_rt)?;
                    results[i - 1] = tok[0] as u32;
                }
                if pipe_timing { t_comm += tm.elapsed().as_secs_f64() * 1e3; }
            }
            // Drain the last job's token (stage 1 sends it after its final job).
            let tok = self.collective.recv_f32(receiver, 1).map_err(to_rt)?;
            results[njobs - 1] = tok[0] as u32;
            if pipe_timing {
                eprintln!(
                    "SKEIN_PIPE_TIMING stage0: setup={t_setup:.1} compute={t_compute:.1} read_carry={t_read:.1} comm={t_comm:.1} (ms total over {njobs} jobs)"
                );
            }
            for (m, mb) in mbs.iter_mut().enumerate() {
                for k in 0..per_mb {
                    mb.generated.push(results[k * n + m]);
                }
            }
        } else if is_last {
            // Stage 1: receive the activation (and send the previous job's token
            // back, same group), run stage-1 compute, sample.
            let mut last_token: u32 = 0;
            for i in 0..njobs {
                let m = jobs[i];
                let k = i / n;
                let id = mbs[m].id;
                let pos = mbs[m].base_position + k;
                let tm = Instant::now();
                let buf = if i == 0 {
                    self.collective
                        .recv_f32(sender, carry_elems)
                        .map_err(to_rt)?
                } else {
                    self.collective
                        .send_recv_f32(&[last_token as f32], sender, sender, carry_elems)
                        .map_err(to_rt)?
                };
                if pipe_timing { t_comm += tm.elapsed().as_secs_f64() * 1e3; }
                let ts = Instant::now();
                {
                    let r = self.executor.runner_mut();
                    r.write_by_id(carry_id, buf)
                        .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                    r.activate_request(id, pos)?;
                    r.advance_kv()?;
                    r.set_position(pos);
                }
                if pipe_timing { t_setup += ts.elapsed().as_secs_f64() * 1e3; }
                let tc = Instant::now();
                self.executor
                    .run(&compute, &self.collective)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                if pipe_timing { t_compute += tc.elapsed().as_secs_f64() * 1e3; }
                let tr = Instant::now();
                let logits = self
                    .executor
                    .runner()
                    .read(LOGITS)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                if pipe_timing { t_read += tr.elapsed().as_secs_f64() * 1e3; }
                let next = argmax(&logits);
                mbs[m].generated.push(next);
                last_token = next;
            }
            // Drain: send the final token back to stage 0.
            self.collective
                .send_f32(&[last_token as f32], sender)
                .map_err(to_rt)?;
            if pipe_timing {
                eprintln!(
                    "SKEIN_PIPE_TIMING stage1: comm={t_comm:.1} setup={t_setup:.1} compute={t_compute:.1} read_logits={t_read:.1} (ms total over {njobs} jobs)"
                );
            }
        }

        let decode_elapsed = decode_start.elapsed();
        let decode_tokens = (n * per_mb) as f64;
        let agg_tps = if decode_elapsed.as_secs_f64() > 0.0 {
            decode_tokens / decode_elapsed.as_secs_f64()
        } else {
            0.0
        };
        for mb in &mbs {
            let _ = self.executor.runner_mut().release_request(mb.id);
        }

        if self.layout.is_leader() {
            tracing::info!(
                microbatches = n,
                max_new_tokens,
                decode_tokens_total = decode_tokens as u64,
                decode_wall_s = decode_elapsed.as_secs_f64(),
                aggregate_decode_tokens_per_s = agg_tps,
                "SKEIN_PERF_PIPE: 1F1B pipelined decode (both stages run concurrently)"
            );
        }

        let mut out = Vec::with_capacity(n);
        for mb in mbs {
            out.push(GenResult {
                tokens: mb.generated,
                decode_tokens_per_s: agg_tps,
                ..Default::default()
            });
        }
        Ok(out)
    }

    /// Dynamic PP=2 continuous batching. Keeps up to `max_slots` requests in
    /// flight, 1F1B-overlapped so stage 0 (GPU0) computes one request's first half
    /// while stage 1 (GPU1) finishes the previous request's second half. New
    /// requests are admitted from `prompts` (a queue) as slots free up; finished
    /// requests are retired. Unlike [`generate_pipelined`] (a *fixed* set decoded
    /// together), the active set changes over time — requests join/leave.
    ///
    /// Per-request positions ARE supported (each 1F1B job is a single-request
    /// forward at that request's own position via `activate_request`), so slots at
    /// different absolute positions co-exist — no same-position limitation.
    ///
    /// Determinism (why both ranks stay matched without extra coordination): both
    /// read the same `prompts` queue and both observe every sampled token (stage 1
    /// samples; stage 0 receives it back over the 1F1B exchange), so admit/retire
    /// decisions are byte-identical on both ranks — same trick as
    /// [`run_generation_batched`]. Each admit does a lockstep prefill (a brief
    /// pipeline bubble). pp==2 only; falls back to sequential [`generate`].
    pub fn run_continuous_pipelined(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
        max_slots: usize,
    ) -> Result<Vec<GenResult>, RuntimeError> {
        let n_total = prompts.len();
        let compute: Vec<ResolvedSequenceStep> = self
            .schedule_resolved
            .iter()
            .filter(|s| {
                !matches!(
                    s,
                    ResolvedSequenceStep::Collective { collective: CollectiveKind::SendRecv, .. }
                )
            })
            .cloned()
            .collect();
        let carry = self.schedule_resolved.iter().find_map(|s| match s {
            ResolvedSequenceStep::Collective {
                collective: CollectiveKind::SendRecv,
                participants,
                tensor,
                elems,
            } => Some((*tensor, *elems, participants[0] as usize, participants[1] as usize)),
            _ => None,
        });
        let needs_fallback = self.pp != 2 || carry.is_none() || max_slots < 2 || n_total == 0;
        if needs_fallback {
            let mut out = Vec::with_capacity(n_total);
            for p in prompts {
                out.push(self.generate(p, max_new_tokens)?);
            }
            return Ok(out);
        }
        let (carry_id, carry_elems, sender, receiver) = carry.unwrap();
        let is_first = self.layout.rank == sender;
        let is_last = self.layout.rank == receiver;

        struct Slot {
            idx: usize,
            id: crate::types::RequestId,
            pos: usize,
            last_token: u32,
            generated: Vec<u32>,
        }
        let mut active: Vec<Slot> = Vec::new();
        let mut next_prompt = 0usize;
        let mut results: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut completed = 0usize;

        self.executor.runner_mut().reset_kv_cache();
        // Per-request paged KV device buffers: each in-flight request gets its OWN
        // KV buffer (keyed by RequestId via req_kv_buffer). WITHOUT this the runner
        // binds a single shared KV buffer, so concurrent slots clobber each other's
        // KV (the first slot in a wave ends up decoding the last slot's KV).
        self.executor.runner_mut().set_paged_device_kv(true);
        let wall_start = Instant::now();
        let mut prefill_s = 0.0f64; // lockstep-prefill (admit) wall — the continuous-batching bubble
        let mut decode_s = 0.0f64; // steady-state 1F1B decode wall (apples-to-apples throughput metric)
        let mut decode_tokens_total = 0usize; // steady-state decode tokens (excludes the prefill first token)

        while completed < n_total {
            // (1) ADMIT: fill free slots from the queue, lockstep-prefilling each.
            let admit_t = Instant::now();
            while active.len() < max_slots && next_prompt < n_total {
                let p = &prompts[next_prompt];
                if p.is_empty() {
                    next_prompt += 1;
                    continue;
                }
                let id = crate::types::RequestId::next();
                let matched = self.executor.runner_mut().admit_request(id, p)?;
                self.executor.runner_mut().activate_request(id, matched)?;
                let start = matched.min(p.len().saturating_sub(1));
                let mut pos = start;
                let mut logits = Vec::new();
                for &tok in &p[start..] {
                    logits = self.forward_step(tok, pos)?;
                    pos += 1;
                }
                let first = self.sample_and_sync(&logits)?;
                active.push(Slot { idx: next_prompt, id, pos, last_token: first, generated: vec![first] });
                next_prompt += 1;
            }
            prefill_s += admit_t.elapsed().as_secs_f64();
            if active.is_empty() {
                break;
            }

            // (2) DECODE WAVE: run `steps` 1F1B rounds over the active slots, where
            // `steps` = tokens until the soonest slot reaches max_new (so a slot
            // frees promptly for a new admit). Round-robin job order m + k*n keeps
            // stage0/stage1 in lockstep; both ranks update slots identically.
            let n = active.len();
            let steps = active
                .iter()
                .map(|s| max_new_tokens.saturating_sub(s.generated.len()))
                .filter(|&r| r > 0)
                .min()
                .unwrap_or(0);
            let dec_t = Instant::now();
            if steps > 0 {
                let njobs = steps * n;
                if is_first {
                    let mut res: Vec<u32> = vec![0u32; njobs];
                    for i in 0..njobs {
                        let m = i % n;
                        let k = i / n;
                        let input = if k == 0 { active[m].last_token } else { res[i - n] };
                        let pos = active[m].pos + k;
                        let id = active[m].id;
                        {
                            let r = self.executor.runner_mut();
                            r.activate_request(id, pos)?;
                            r.advance_kv()?;
                            r.set_input_tokens(INPUT_TOKENS, vec![input as i32]);
                            r.set_position(pos);
                        }
                        self.executor
                            .run(&compute, &self.collective)
                            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                        let buf = self
                            .executor
                            .runner()
                            .read_by_id(carry_id)
                            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                        if i == 0 {
                            self.collective.send_f32(&buf, receiver).map_err(to_rt)?;
                        } else {
                            let tok = self
                                .collective
                                .send_recv_f32(&buf, receiver, receiver, 1)
                                .map_err(to_rt)?;
                            res[i - 1] = tok[0] as u32;
                        }
                    }
                    let tok = self.collective.recv_f32(receiver, 1).map_err(to_rt)?;
                    res[njobs - 1] = tok[0] as u32;
                    for k in 0..steps {
                        for m in 0..n {
                            let t = res[k * n + m];
                            active[m].generated.push(t);
                            active[m].last_token = t;
                        }
                    }
                    for s in active.iter_mut() {
                        s.pos += steps;
                    }
                } else if is_last {
                    let mut last_token: u32 = 0;
                    for i in 0..njobs {
                        let m = i % n;
                        let k = i / n;
                        let pos = active[m].pos + k;
                        let id = active[m].id;
                        let buf = if i == 0 {
                            self.collective.recv_f32(sender, carry_elems).map_err(to_rt)?
                        } else {
                            self.collective
                                .send_recv_f32(&[last_token as f32], sender, sender, carry_elems)
                                .map_err(to_rt)?
                        };
                        {
                            let r = self.executor.runner_mut();
                            r.write_by_id(carry_id, buf)
                                .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                            r.activate_request(id, pos)?;
                            r.advance_kv()?;
                            r.set_position(pos);
                        }
                        self.executor
                            .run(&compute, &self.collective)
                            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                        let logits = self
                            .executor
                            .runner()
                            .read(LOGITS)
                            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                        let next = argmax(&logits);
                        active[m].generated.push(next);
                        active[m].last_token = next;
                        last_token = next;
                    }
                    self.collective.send_f32(&[last_token as f32], sender).map_err(to_rt)?;
                    for s in active.iter_mut() {
                        s.pos += steps;
                    }
                }
                decode_tokens_total += steps * n;
            }
            decode_s += dec_t.elapsed().as_secs_f64();

            // (3) RETIRE finished slots; free their pages for a later admit.
            let mut still: Vec<Slot> = Vec::with_capacity(active.len());
            for s in active.drain(..) {
                if s.generated.len() >= max_new_tokens {
                    results.push((s.idx, s.generated.clone()));
                    completed += 1;
                    let _ = self.executor.runner_mut().release_request(s.id);
                } else {
                    still.push(s);
                }
            }
            active = still;

            if self.layout.is_leader() {
                // Headline = decode-only (excludes the lockstep-prefill bubble), the
                // apples-to-apples match to the fixed SKEIN_PIPELINE_STREAMS metric.
                let agg = if decode_s > 0.0 { decode_tokens_total as f64 / decode_s } else { 0.0 };
                let inflight: Vec<usize> = active.iter().map(|s| s.idx).collect();
                eprintln!(
                    "SKEIN_PERF_CONTINUOUS_PP active_streams={n} \
                     aggregate_decode_tokens_per_s={agg:.2} per_user_tokens_per_s={:.2} \
                     stage0_req={inflight:?} stage1_req={inflight:?} completed_reqs={completed}",
                    agg / (n as f64).max(1.0),
                );
            }
        }

        results.sort_by_key(|(idx, _)| *idx);
        let agg = if decode_s > 0.0 { decode_tokens_total as f64 / decode_s } else { 0.0 };
        let end_to_end = wall_start.elapsed().as_secs_f64();
        let e2e_tps = if end_to_end > 0.0 { decode_tokens_total as f64 / end_to_end } else { 0.0 };
        if self.layout.is_leader() {
            eprintln!(
                "SKEIN_PERF_CONTINUOUS_PP FINAL completed_reqs={completed} total_decode_tokens={decode_tokens_total} \
                 decode_s={decode_s:.4} aggregate_decode_tokens_per_s={agg:.2} \
                 prefill_bubble_s={prefill_s:.4} end_to_end_s={end_to_end:.4} end_to_end_tokens_per_s={e2e_tps:.2} \
                 max_slots={max_slots}"
            );
        }
        let mut out = Vec::with_capacity(n_total);
        for (_idx, toks) in results {
            out.push(GenResult { tokens: toks, decode_tokens_per_s: agg, ..Default::default() });
        }
        Ok(out)
    }

    /// Argmax `stage_mb` rows of a [`stage_mb`, vocab] PP last-stage logits buffer
    /// (TP=1 → no rank interleave) and broadcast the chosen tokens from the last
    /// stage to the earlier stage (which needs them for its next embed). Collective.
    fn sample_batched_pp(&self, logits: &[f32], stage_mb: usize) -> Result<Vec<u32>, RuntimeError> {
        let vocab = self.vocab as usize;
        let next: Vec<u32> = if self.is_last_stage {
            (0..stage_mb)
                .map(|r| {
                    let lo = r * vocab;
                    let hi = ((r + 1) * vocab).min(logits.len());
                    if lo < hi { argmax(&logits[lo..hi]) } else { 0 }
                })
                .collect()
        } else {
            vec![0u32; stage_mb]
        };
        let mut buf: Vec<f32> = next.iter().map(|&t| t as f32).collect();
        self.collective.broadcast(&mut buf, self.last_stage_root).map_err(to_rt)?;
        Ok(buf.iter().map(|&f| f as u32).collect())
    }

    /// PP=2 dynamic continuous batching with STAGE MICROBATCH = `stage_mb`: each
    /// 1F1B job is a BATCHED forward of `stage_mb` requests, so stage 1 emits
    /// `stage_mb` real tokens per pipeline tick (vs 1 in run_continuous_pipelined).
    /// Goal: break the 1-token/tick per-stage ceiling — if a batched tick costs
    /// < `stage_mb`× a single tick, aggregate decode throughput exceeds the
    /// microbatch=1 ~93. Requires a FORCE_BATCH=`stage_mb` graph (the kvcache +
    /// PP carry tensors are batch-sized).
    ///
    /// KV isolation: each in-flight microbatch gets its OWN batched KV via a
    /// per-microbatch key (`kv_key`) + `set_decode_batch(stage_mb)`; req_kv_buffer
    /// allocates a per-key batch buffer, so concurrent microbatches don't clobber.
    /// Rows in one microbatch share the scalar position (same-length assumed).
    /// PP=2 only (falls back to sequential generate).
    pub fn run_continuous_pipelined_mb(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
        max_microbatches: usize,
        stage_mb: usize,
    ) -> Result<Vec<GenResult>, RuntimeError> {
        let n_total = prompts.len();
        let vocab = self.vocab as usize;
        let compute: Vec<ResolvedSequenceStep> = self
            .schedule_resolved
            .iter()
            .filter(|s| {
                !matches!(
                    s,
                    ResolvedSequenceStep::Collective { collective: CollectiveKind::SendRecv, .. }
                )
            })
            .cloned()
            .collect();
        let carry = self.schedule_resolved.iter().find_map(|s| match s {
            ResolvedSequenceStep::Collective {
                collective: CollectiveKind::SendRecv,
                participants,
                tensor,
                elems,
            } => Some((*tensor, *elems, participants[0] as usize, participants[1] as usize)),
            _ => None,
        });
        let needs_fallback = self.pp != 2 || carry.is_none() || stage_mb < 1 || n_total == 0;
        if needs_fallback {
            let mut out = Vec::with_capacity(n_total);
            for p in prompts {
                out.push(self.generate(p, max_new_tokens)?);
            }
            return Ok(out);
        }
        let (carry_id, carry_elems, sender, receiver) = carry.unwrap();
        let is_first = self.layout.rank == sender;
        let is_last = self.layout.rank == receiver;
        let mb_log = std::env::var_os("SKEIN_CONTINUOUS_PP_MB_LOG").is_some();
        if mb_log {
            eprintln!(
                "MB_DBG rank={} carry_elems={carry_elems} stage_mb={stage_mb} is_first={is_first} is_last={is_last}",
                self.layout.rank
            );
        }

        self.executor.runner_mut().reset_kv_cache();
        self.executor.runner_mut().set_decode_batch(stage_mb);
        // Per-microbatch paged KV: each microbatch gets its OWN batched KV buffer
        // (keyed by kv_key) so concurrent microbatches don't clobber each other.
        self.executor.runner_mut().set_paged_device_kv(true);

        struct Mb {
            idxs: Vec<usize>,        // original prompt indices (real rows only)
            kv_key: crate::types::RequestId,
            pos: usize,              // shared position of all rows
            last: Vec<u32>,          // last token per row (len stage_mb)
            generated: Vec<Vec<u32>>, // generated tokens per row (len stage_mb)
        }
        let mut active: Vec<Mb> = Vec::new();
        let mut next_idx = 0usize;
        let mut results: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut completed = 0usize; // real requests retired
        let mut prefill_s = 0.0f64;
        let mut decode_s = 0.0f64;
        let mut decode_tokens_total = 0usize; // real emitted tokens (excludes padded rows)
        let mut max_active = 0usize;
        let mut tick = 0u64;
        let wall_start = Instant::now();

        while completed < n_total {
            // (1) ADMIT: form microbatches of stage_mb prompts; lockstep batched prefill.
            let admit_t = Instant::now();
            while active.len() < max_microbatches && next_idx < n_total {
                let mut idxs: Vec<usize> = Vec::new();
                let mut mbp: Vec<Vec<u32>> = Vec::new();
                while mbp.len() < stage_mb && next_idx < n_total {
                    if !prompts[next_idx].is_empty() {
                        idxs.push(next_idx);
                        mbp.push(prompts[next_idx].clone());
                    }
                    next_idx += 1;
                }
                if mbp.is_empty() {
                    continue;
                }
                // Pad to stage_mb rows (repeat last) so the batch graph always gets
                // stage_mb rows; padded rows are never emitted/retired.
                while mbp.len() < stage_mb {
                    mbp.push(mbp[mbp.len() - 1].clone());
                }
                let kv_key = crate::types::RequestId::next();
                let matched = self.executor.runner_mut().admit_request(kv_key, &mbp[0])?;
                self.executor.runner_mut().activate_request(kv_key, matched)?;
                let lmax = mbp.iter().map(|p| p.len()).max().unwrap();
                let mut logits = Vec::new();
                for pos in 0..lmax {
                    let toks: Vec<u32> =
                        (0..stage_mb).map(|r| mbp[r].get(pos).copied().unwrap_or(0)).collect();
                    logits = self.forward_step_tokens(&toks, pos)?;
                }
                let first = self.sample_batched_pp(&logits, stage_mb)?;
                let generated: Vec<Vec<u32>> = (0..stage_mb).map(|r| vec![first[r]]).collect();
                active.push(Mb { idxs, kv_key, pos: lmax, last: first, generated });
            }
            prefill_s += admit_t.elapsed().as_secs_f64();
            if mb_log {
                eprintln!("MB_DBG rank={} PREFILL_DONE active_microbatches={}", self.layout.rank, active.len());
            }
            if active.is_empty() {
                break;
            }
            max_active = max_active.max(active.len());

            // (2) DECODE WAVE: 1F1B over the active microbatches; `steps` rounds.
            let n_mb = active.len();
            let steps = active
                .iter()
                .map(|mb| max_new_tokens.saturating_sub(mb.generated[0].len()))
                .filter(|&r| r > 0)
                .min()
                .unwrap_or(0);
            let dec_t = Instant::now();
            if steps > 0 {
                let njobs = steps * n_mb;
                if is_first {
                    // res[job*stage_mb + r] = token for microbatch job's row r.
                    let mut res: Vec<u32> = vec![0u32; njobs * stage_mb];
                    for i in 0..njobs {
                        let m = i % n_mb;
                        let k = i / n_mb;
                        let pos = active[m].pos + k;
                        let kv_key = active[m].kv_key;
                        let input: Vec<u32> = if k == 0 {
                            active[m].last.clone()
                        } else {
                            res[(i - n_mb) * stage_mb..(i - n_mb + 1) * stage_mb].to_vec()
                        };
                        {
                            let r = self.executor.runner_mut();
                            r.activate_request(kv_key, pos)?;
                            r.advance_kv()?;
                            r.set_input_tokens(INPUT_TOKENS, input.iter().map(|&t| t as i32).collect());
                            r.set_position(pos);
                        }
                        self.executor
                            .run(&compute, &self.collective)
                            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                        let buf = self
                            .executor
                            .runner()
                            .read_by_id(carry_id)
                            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                        if i == 0 {
                            if mb_log {
                                eprintln!("MB_DBG rank={} stage0 job0 carry_buf_len={} (carry_elems={carry_elems})", self.layout.rank, buf.len());
                            }
                            self.collective.send_f32(&buf, receiver).map_err(to_rt)?;
                        } else {
                            let toks = self
                                .collective
                                .send_recv_f32(&buf, receiver, receiver, stage_mb)
                                .map_err(to_rt)?;
                            for r in 0..stage_mb {
                                res[(i - 1) * stage_mb + r] = toks[r] as u32;
                            }
                        }
                    }
                    let toks = self.collective.recv_f32(receiver, stage_mb).map_err(to_rt)?;
                    for r in 0..stage_mb {
                        res[(njobs - 1) * stage_mb + r] = toks[r] as u32;
                    }
                    for k in 0..steps {
                        for m in 0..n_mb {
                            let job = k * n_mb + m;
                            for r in 0..stage_mb {
                                let t = res[job * stage_mb + r];
                                active[m].generated[r].push(t);
                                active[m].last[r] = t;
                            }
                        }
                    }
                    for mb in active.iter_mut() {
                        mb.pos += steps;
                    }
                } else if is_last {
                    let mut last_tokens = vec![0u32; stage_mb];
                    for i in 0..njobs {
                        let m = i % n_mb;
                        let k = i / n_mb;
                        let pos = active[m].pos + k;
                        let kv_key = active[m].kv_key;
                        let buf = if i == 0 {
                            self.collective.recv_f32(sender, carry_elems).map_err(to_rt)?
                        } else {
                            let send: Vec<f32> = last_tokens.iter().map(|&t| t as f32).collect();
                            self.collective
                                .send_recv_f32(&send, sender, sender, carry_elems)
                                .map_err(to_rt)?
                        };
                        {
                            let r = self.executor.runner_mut();
                            r.write_by_id(carry_id, buf)
                                .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                            r.activate_request(kv_key, pos)?;
                            r.advance_kv()?;
                            r.set_position(pos);
                        }
                        self.executor
                            .run(&compute, &self.collective)
                            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                        let logits = self
                            .executor
                            .runner()
                            .read(LOGITS)
                            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                        let next: Vec<u32> = (0..stage_mb)
                            .map(|r| {
                                let lo = r * vocab;
                                let hi = ((r + 1) * vocab).min(logits.len());
                                if lo < hi { argmax(&logits[lo..hi]) } else { 0 }
                            })
                            .collect();
                        for r in 0..stage_mb {
                            active[m].generated[r].push(next[r]);
                            active[m].last[r] = next[r];
                        }
                        last_tokens = next;
                    }
                    let send: Vec<f32> = last_tokens.iter().map(|&t| t as f32).collect();
                    self.collective.send_f32(&send, sender).map_err(to_rt)?;
                    for mb in active.iter_mut() {
                        mb.pos += steps;
                    }
                }
                // Real emitted tokens this wave = steps × (real rows across microbatches).
                let real_rows: usize = active.iter().map(|mb| mb.idxs.len()).sum();
                decode_tokens_total += steps * real_rows;
                tick += njobs as u64;
            }
            decode_s += dec_t.elapsed().as_secs_f64();

            if self.layout.is_leader() && std::env::var_os("SKEIN_CONTINUOUS_PP_MB_LOG").is_some() {
                let s0: Vec<&Vec<usize>> = active.iter().map(|mb| &mb.idxs).collect();
                eprintln!(
                    "SKEIN_PERF_CONTINUOUS_PP_MB tick={tick} stage_microbatch={stage_mb} \
                     active_microbatches={n_mb} stage0_mb={s0:?} stage1_mb={s0:?} emitted_tokens_per_tick={stage_mb}"
                );
            }

            // (3) RETIRE finished microbatches (all real rows hit max_new).
            let mut still: Vec<Mb> = Vec::with_capacity(active.len());
            for mb in active.drain(..) {
                if mb.generated[0].len() >= max_new_tokens {
                    for (r, &idx) in mb.idxs.iter().enumerate() {
                        results.push((idx, mb.generated[r].clone()));
                        completed += 1;
                    }
                    let _ = self.executor.runner_mut().release_request(mb.kv_key);
                } else {
                    still.push(mb);
                }
            }
            active = still;
        }

        results.sort_by_key(|(idx, _)| *idx);
        let agg = if decode_s > 0.0 { decode_tokens_total as f64 / decode_s } else { 0.0 };
        let end_to_end = wall_start.elapsed().as_secs_f64();
        let e2e = if end_to_end > 0.0 { decode_tokens_total as f64 / end_to_end } else { 0.0 };
        if self.layout.is_leader() {
            eprintln!(
                "SKEIN_PERF_CONTINUOUS_PP_MB_FINAL stage_microbatch={stage_mb} total_prompts={n_total} \
                 completed_reqs={completed} max_active_microbatches={max_active} decode_tokens={decode_tokens_total} \
                 decode_s={decode_s:.4} decode_only_aggregate_tps={agg:.2} \
                 prefill_bubble_s={prefill_s:.4} end_to_end_burst_s={end_to_end:.4} end_to_end_burst_tps={e2e:.2} \
                 per_user_avg_tps={:.2}",
                if n_total > 0 { agg / n_total as f64 } else { 0.0 }
            );
        }
        let mut out = Vec::with_capacity(n_total);
        for (_idx, toks) in results {
            out.push(GenResult { tokens: toks, decode_tokens_per_s: agg, ..Default::default() });
        }
        Ok(out)
    }
}

/// One generation's tokens plus paged-KV / timing telemetry.
#[derive(Debug, Clone, Default)]
pub struct GenResult {
    pub tokens: Vec<u32>,
    /// Prompt tokens reused from the prefix cache (0 = cold).
    pub prefix_hit_tokens: usize,
    /// Prompt tokens actually prefilled this run (= prompt_len - prefix-skipped).
    pub prefill_steps: usize,
    pub prompt_tokens: usize,
    pub ttft_ms: f64,
    pub tpot_p50_ms: f64,
    pub tpot_p95_ms: f64,
    pub decode_tokens_per_s: f64,
    pub kv_pages_in_use: u32,
    /// Real Luminal `cuGraphInstantiate` count during this generation.
    pub graph_instantiates: u64,
    /// Real Luminal `cuGraphLaunch` (graph replay) count during this generation.
    pub graph_launches: u64,
}

/// End-to-end distributed greedy generation for one prompt across the rank
/// group — the runnable multi-GPU path. Launch one process per GPU (see
/// [`crate::distributed::launcher`]); every process calls this. Rank 0
/// tokenizes + broadcasts the prompt and returns the decoded completion; the
/// other ranks follow in lockstep and return `None`.
pub fn run_generation(
    artifact_dir: &Path,
    layout: WorldLayout,
    rendezvous_path: &Path,
    prompt: &str,
    max_new_tokens: usize,
    tokenizer: Option<&SkeinTokenizer>,
) -> Result<Option<String>, RuntimeError> {
    let mut server = RankServer::bootstrap(artifact_dir, layout, rendezvous_path)?;

    // Rank 0 encodes the prompt; the broadcast hands it to every rank so they
    // decode the same sequence in lockstep.
    let prompt_tokens: Vec<u32> = if layout.is_leader() {
        match tokenizer {
            Some(tok) => tok.encode(prompt)?,
            None => {
                let vocab = server.vocab().max(1);
                prompt.bytes().map(|b| (b as u32) % vocab).collect()
            }
        }
    } else {
        Vec::new()
    };
    let prompt_tokens = server.broadcast_prompt(if layout.is_leader() {
        Some(&prompt_tokens)
    } else {
        None
    })?;

    // Cold run (no prefix cache populated yet).
    let cold = server.generate(&prompt_tokens, max_new_tokens)?;

    // Prefix-cache demonstration: re-run the same prompt. The paged allocator's
    // radix tree now holds the cold run's pages, so the shared prompt prefix is
    // served from cache — fewer prefill steps — and the generated tokens are
    // byte-identical, proving paged-KV prefix reuse is correct and active.
    // (Page-granular: a hit needs a shared prefix >= page_size tokens.)
    if std::env::var_os("SKEIN_PREFIX_DEMO").is_some() {
        let warm = server.generate(&prompt_tokens, max_new_tokens)?;
        if layout.is_leader() {
            let identical = cold.tokens == warm.tokens;
            tracing::info!(
                prompt_tokens = cold.prompt_tokens,
                page_size = server.kv_page_size(),
                cold_prefix_hit = cold.prefix_hit_tokens,
                cold_prefill_steps = cold.prefill_steps,
                warm_prefix_hit = warm.prefix_hit_tokens,
                warm_prefill_steps = warm.prefill_steps,
                warm_kv_pages = warm.kv_pages_in_use,
                tokens_identical = identical,
                "SKEIN_PREFIX_DEMO: cold vs warm (prefix-cache reuse)"
            );
            if !identical {
                tracing::error!(
                    cold = ?cold.tokens,
                    warm = ?warm.tokens,
                    "SKEIN_PREFIX_DEMO: warm tokens differ from cold — reuse INCORRECT"
                );
            }
        }
    }

    if layout.is_leader() {
        let text = match tokenizer {
            Some(tok) => tok.decode(&cold.tokens)?,
            None => format!("{:?}", cold.tokens),
        };
        Ok(Some(text))
    } else {
        Ok(None)
    }
}

/// Batched decode of many prompts in one captured forward per token — the
/// product path (batch>1 graph + full-step CUDA-graph capture + shm device
/// all-reduce). Every rank process calls this; all read the same prompts and run
/// identical lockstep batched decode. The prompts are padded/truncated by
/// repetition to the compiled batch width. Rank 0 returns a report (aggregate +
/// per-row decode throughput + decoded rows); other ranks return `None`.
pub fn run_generation_batched(
    artifact_dir: &Path,
    layout: WorldLayout,
    rendezvous_path: &Path,
    prompts: &[Vec<u32>],
    max_new_tokens: usize,
    tokenizer: Option<&SkeinTokenizer>,
) -> Result<Option<String>, RuntimeError> {
    let mut server = RankServer::bootstrap(artifact_dir, layout, rendezvous_path)?;
    let gb = server.batch_width().max(1);

    // Keep original indices so the report maps back to the user's prompt order.
    let mut indexed: Vec<(usize, Vec<u32>)> = prompts
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.is_empty())
        .map(|(i, p)| (i, p.clone()))
        .collect();
    if indexed.is_empty() {
        return Err(RuntimeError::ServerInit("no prompts for batched generation".into()));
    }
    let total_unique = indexed.len();
    // Length-bucket: sort by prompt length so each fixed-width chunk has
    // similar-length prompts. Mixed-length batching wastes work when a short
    // prompt shares a chunk with a very long one (it finishes early but still
    // rides every forward); bucketing minimises that.
    indexed.sort_by_key(|(_, p)| p.len());

    let mut all_rows: Vec<(usize, Vec<u32>)> = Vec::new(); // (original idx, tokens)
    let mut total_decode_s = 0.0f64;
    let mut total_unique_tokens = 0usize;
    let mut chunk_lines = String::new();
    let mut chunk_idx = 0;
    for chunk in indexed.chunks(gb) {
        let unique_here = chunk.len();
        let mut bp: Vec<Vec<u32>> = chunk.iter().map(|(_, p)| p.clone()).collect();
        let src_len = bp.len();
        while bp.len() < gb {
            let i = bp.len() % src_len;
            bp.push(bp[i].clone());
        }
        let (genr, decode_s) = server.generate_batched(&bp, max_new_tokens)?;
        total_decode_s += decode_s;
        total_unique_tokens += unique_here * max_new_tokens;
        if layout.is_leader() {
            let agg = (gb * max_new_tokens) as f64 / decode_s.max(1e-9);
            let maxlen = chunk.iter().map(|(_, p)| p.len()).max().unwrap_or(0);
            chunk_lines.push_str(&format!(
                "  chunk{chunk_idx}: batch={gb} unique={unique_here} maxlen={maxlen} decode_s={decode_s:.3} chunk_aggregate_decode_tok/s={agg:.1}\n"
            ));
            for (r, g) in genr.iter().enumerate().take(unique_here) {
                all_rows.push((chunk[r].0, g.clone()));
            }
        }
        chunk_idx += 1;
    }
    all_rows.sort_by_key(|(idx, _)| *idx);

    if layout.is_leader() {
        // Sustained throughput across all prompts = unique tokens / total decode
        // wall (chunks run sequentially → this is the real system throughput for
        // the 20 prompts on this 2-GPU box).
        let sustained = total_unique_tokens as f64 / total_decode_s.max(1e-9);
        let per_row = max_new_tokens as f64 / (total_decode_s / chunk_idx.max(1) as f64).max(1e-9);
        let mut s = format!(
            "=== Batched decode + full-step CUDA-graph capture (TP, shm all-reduce) ===\n  \
             unique_prompts={total_unique} batch_width={gb} chunks={chunk_idx} max_new={max_new_tokens} \
             total_decode_s={total_decode_s:.3}\n  \
             SUSTAINED aggregate_tokens_per_s={sustained:.1} (per_row≈{per_row:.1})\n{chunk_lines}"
        );
        for (idx, g) in all_rows.iter().take(total_unique) {
            let txt = match tokenizer {
                Some(t) => t.decode(g).unwrap_or_default(),
                None => format!("{:?}", &g[..g.len().min(8)]),
            };
            let head: String = txt.chars().take(64).collect();
            s.push_str(&format!("  prompt{idx}: {head:?}\n"));
        }
        Ok(Some(s))
    } else {
        Ok(None)
    }
}

/// Pipelined multi-stream generation: drive `n_streams` concurrent copies of the
/// prompt through the PP stages with 1F1B overlap so both GPUs compute at once.
/// Reports aggregate decode throughput. Rank 0 returns the (first stream's)
/// decoded text; other ranks return `None`. Falls back to single-stream inside
/// [`RankServer::generate_pipelined`] when the plan isn't 2-stage PP or
/// `n_streams < 2`.
pub fn run_generation_pipelined(
    artifact_dir: &Path,
    layout: WorldLayout,
    rendezvous_path: &Path,
    prompt: &str,
    max_new_tokens: usize,
    n_streams: usize,
    tokenizer: Option<&SkeinTokenizer>,
) -> Result<Option<String>, RuntimeError> {
    let mut server = RankServer::bootstrap(artifact_dir, layout, rendezvous_path)?;

    let prompt_tokens: Vec<u32> = if layout.is_leader() {
        match tokenizer {
            Some(tok) => tok.encode(prompt)?,
            None => {
                let vocab = server.vocab().max(1);
                prompt.bytes().map(|b| (b as u32) % vocab).collect()
            }
        }
    } else {
        Vec::new()
    };
    let prompt_tokens = server.broadcast_prompt(if layout.is_leader() {
        Some(&prompt_tokens)
    } else {
        None
    })?;

    // Same prompt on every stream: each is an independent request (its own KV
    // pages + decode trajectory) so the aggregate reflects real concurrent work.
    let prompts: Vec<Vec<u32>> = vec![prompt_tokens; n_streams.max(1)];
    let results = server.generate_pipelined(&prompts, max_new_tokens)?;

    if layout.is_leader() {
        let tokens = results.first().map(|r| r.tokens.as_slice()).unwrap_or(&[]);
        let text = match tokenizer {
            Some(tok) => tok.decode(tokens)?,
            None => format!("{tokens:?}"),
        };
        Ok(Some(text))
    } else {
        Ok(None)
    }
}

/// DYNAMIC PP=2 continuous-batching generation (SKEIN_CONTINUOUS_PP=N). Drives a
/// queue of `queue_len` requests through up to `max_slots` 1F1B-overlapped slots:
/// requests join as slots free, finished ones retire — a real continuous-serving
/// loop, not the fixed-stream `generate_pipelined`. Every rank builds the same
/// queue (the prompt is broadcast then replicated), so both stages stay matched.
/// Rank 0 returns the first request's decoded text + logs SKEIN_PERF_CONTINUOUS_PP.
pub fn run_generation_continuous_pp(
    artifact_dir: &Path,
    layout: WorldLayout,
    rendezvous_path: &Path,
    prompts: &[Vec<u32>],
    max_new_tokens: usize,
    max_slots: usize,
    stage_mb: usize,
    tokenizer: Option<&SkeinTokenizer>,
) -> Result<Option<String>, RuntimeError> {
    let mut server = RankServer::bootstrap(artifact_dir, layout, rendezvous_path)?;
    // `prompts` is already tokenized identically on every rank (caller reads the
    // same SKEIN_PROMPTS_FILE / replicates the same prompt), so both PP stages
    // build the same queue and stay matched.
    let results = if stage_mb >= 2 {
        server.run_continuous_pipelined_mb(prompts, max_new_tokens, max_slots, stage_mb)?
    } else {
        server.run_continuous_pipelined(prompts, max_new_tokens, max_slots)?
    };

    if layout.is_leader() {
        let mut s = String::new();
        for (i, r) in results.iter().enumerate() {
            let first = r.tokens.first().copied().unwrap_or(0);
            let text = match tokenizer {
                Some(tok) => tok.decode(&r.tokens).unwrap_or_default(),
                None => format!("{:?}", &r.tokens),
            };
            let head: String = text.chars().take(80).collect();
            s.push_str(&format!("prompt{i} first_token={first} text={head:?}\n"));
        }
        Ok(Some(s))
    } else {
        Ok(None)
    }
}

/// Speculative-decode generation across the rank group: a small `draft` model
/// proposes tokens, the `target` model verifies them (exact speculative
/// sampling, [`crate::speculative`]). Both models are bootstrapped as their own
/// rank servers; the loop drives them via [`RankServer::forward_step`]. Rank 0
/// returns the decoded text. Best-effort GPU-staged code — validate on hardware.
pub fn run_speculative_generation(
    target_artifact: &Path,
    draft_artifact: &Path,
    layout: WorldLayout,
    rendezvous_dir: &Path,
    prompt: &str,
    max_new_tokens: usize,
    draft_lookahead: usize,
    tokenizer: Option<&SkeinTokenizer>,
) -> Result<Option<String>, RuntimeError> {
    // Each model gets its own communicator (separate rendezvous file).
    let mut target =
        RankServer::bootstrap(target_artifact, layout, &rendezvous_dir.join("target.rdv"))?;
    let mut draft =
        RankServer::bootstrap(draft_artifact, layout, &rendezvous_dir.join("draft.rdv"))?;

    let prompt_tokens: Vec<u32> = if layout.is_leader() {
        match tokenizer {
            Some(tok) => tok.encode(prompt)?,
            None => {
                let vocab = target.vocab().max(1);
                prompt.bytes().map(|b| (b as u32) % vocab).collect()
            }
        }
    } else {
        Vec::new()
    };
    let prompt_tokens = target.broadcast_prompt(if layout.is_leader() {
        Some(&prompt_tokens)
    } else {
        None
    })?;

    // Forwards: running sequence -> next-token logits. Cached decode needs a
    // position per token and a per-request cache; the speculative loop probes
    // arbitrary candidate sequences, so each call recomputes from scratch
    // (`forward_full` resets the cache and prefills the whole seq). Correct but
    // O(n) per call — a cache-rollback fast path is a follow-up. Errors degrade
    // to an empty distribution (logged) so the loop's signature stays infallible.
    let mut target_fwd = |seq: &[u32]| match target.forward_full(seq) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%e, "target forward failed");
            Vec::new()
        }
    };
    let mut draft_fwd = |seq: &[u32]| match draft.forward_full(seq) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%e, "draft forward failed");
            Vec::new()
        }
    };

    // Deterministic uniform stream (splitmix-style LCG) for the accept/reject
    // coin flips. Swap for a seeded RNG when non-greedy sampling is wanted.
    let mut rng_state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut uniforms = || {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 40) as f32) / ((1u64 << 24) as f32)
    };

    let generated = speculative::generate(
        &prompt_tokens,
        max_new_tokens,
        draft_lookahead,
        &mut target_fwd,
        &mut draft_fwd,
        speculative::argmax,
        &mut uniforms,
    )
    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;

    if layout.is_leader() {
        let text = match tokenizer {
            Some(tok) => tok.decode(&generated)?,
            None => format!("{generated:?}"),
        };
        Ok(Some(text))
    } else {
        Ok(None)
    }
}

fn to_rt(e: CollectiveError) -> RuntimeError {
    RuntimeError::ServerInit(e.to_string())
}

/// The MoE `top_k` from a resolved schedule (all MoeRoute steps share it), or
/// `None` on the dense path (no MoeRoute steps → no gate buffer needed).
pub(crate) fn schedule_top_k(schedule: &[ResolvedSequenceStep]) -> Option<usize> {
    schedule.iter().find_map(|s| match s {
        ResolvedSequenceStep::MoeRoute { top_k, .. } => Some(*top_k),
        _ => None,
    })
}

/// Allocate this runner's resident bf16 gate buffer (`num_layers * top_k` slots)
/// on a fresh default-stream and bind its FFN gate inputs to fixed offsets once.
pub(crate) fn install_gate_buffer(
    runner: &mut SegmentRunner,
    schedule: &[ResolvedSequenceStep],
    rank: u32,
    num_layers: usize,
    top_k: usize,
) -> Result<(), RuntimeError> {
    // Multi-process ranks see their GPU as device 0 (CUDA_VISIBLE_DEVICES), so
    // the gate buffer goes on device 0. The single-process LocalTopology, which
    // places runner `d` on physical device `d`, must instead call
    // [`install_gate_buffer_on`] with the explicit device index.
    install_gate_buffer_on(runner, schedule, rank, num_layers, top_k, 0)
}

/// Device-explicit variant of [`install_gate_buffer`]: allocate the gate buffer
/// on `device` (the runner's physical GPU). The multi-process path passes the
/// CUDA-visible device 0; the single-process LocalTopology passes the device idx.
pub(crate) fn install_gate_buffer_on(
    runner: &mut SegmentRunner,
    schedule: &[ResolvedSequenceStep],
    rank: u32,
    num_layers: usize,
    top_k: usize,
    device: usize,
) -> Result<(), RuntimeError> {
    let total = num_layers * top_k;
    let ctx = cudarc::driver::CudaContext::new(device)
        .map_err(|e| RuntimeError::ServerInit(format!("gate buffer ctx: {e}")))?;
    let stream = ctx.default_stream();
    let buf = stream
        .alloc_zeros::<half::bf16>(total)
        .map_err(|e| RuntimeError::ServerInit(format!("gate buffer alloc: {e}")))?;
    runner.set_gate_buffer(stream, buf, top_k);
    runner
        .bind_gate_inputs(schedule, rank)
        .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
    Ok(())
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}
