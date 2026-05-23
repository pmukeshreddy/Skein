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
    capture_split: usize,
    /// Pipeline-parallel coordination: under PP only the last stage computes
    /// logits, so it samples the next token and broadcasts it to the earlier
    /// stages (which need it for their next embed). `pp == 1` => TP, every rank
    /// has full logits and samples locally (no broadcast).
    pp: u32,
    is_last_stage: bool,
    last_stage_root: usize,
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
        host_tensors.insert(LOGITS.to_string());
        executor.runner_mut().set_host_tensors(host_tensors.clone());

        // Sparse MoE: the FFN segments declare every expert (so they load
        // GPU-resident) but compute only the bound slots, so lazy-on-execute
        // would never upload the experts. Free the per-segment search arenas
        // FIRST (otherwise they coexist with the ~46 GB/card of weights and OOM),
        // then materialize so the MoeRoute step can resolve each selected
        // expert's resident device pointer.
        if std::env::var_os("SKEIN_SPARSE_MOE").is_some() {
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
            capture_split,
            pp,
            is_last_stage,
            last_stage_root,
        })
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
        {
            let runner = self.executor.runner_mut();
            runner.set_input_tokens(INPUT_TOKENS, vec![token as i32]);
            runner.set_position(position);
        }
        // SKEIN_CAPTURE full-step graph: capture the pre-all-gather region once
        // (after a few warmup steps so the arena/handoff buffers are stable), then
        // replay it per token with one cuGraphLaunch; the logits all-gather + read
        // run on the host every step.
        const CAPTURE_AT: usize = 2;
        let capture =
            std::env::var_os("SKEIN_CAPTURE").is_some() && self.capture_split < self.schedule_resolved.len();
        if capture {
            self.decode_step += 1;
            let split = self.capture_split;
            if self.executor.runner().has_captured() {
                self.executor.runner().replay_captured();
            } else if self.decode_step == CAPTURE_AT {
                self.executor.runner().begin_capture();
                self.executor
                    .run(&self.schedule_resolved[..split], &self.collective)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
                self.executor.runner().end_capture();
            } else {
                self.executor
                    .run(&self.schedule_resolved[..split], &self.collective)
                    .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
            }
            self.executor
                .run(&self.schedule_resolved[split..], &self.collective)
                .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
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
            return Ok(argmax(logits));
        }
        let next = if self.is_last_stage { argmax(logits) } else { 0 };
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
                    results[i - 1] = tok[0] as u32;
                }
            }
            // Drain the last job's token (stage 1 sends it after its final job).
            let tok = self.collective.recv_f32(receiver, 1).map_err(to_rt)?;
            results[njobs - 1] = tok[0] as u32;
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
                let buf = if i == 0 {
                    self.collective
                        .recv_f32(sender, carry_elems)
                        .map_err(to_rt)?
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
                mbs[m].generated.push(next);
                last_token = next;
            }
            // Drain: send the final token back to stage 0.
            self.collective
                .send_f32(&[last_token as f32], sender)
                .map_err(to_rt)?;
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
fn schedule_top_k(schedule: &[ResolvedSequenceStep]) -> Option<usize> {
    schedule.iter().find_map(|s| match s {
        ResolvedSequenceStep::MoeRoute { top_k, .. } => Some(*top_k),
        _ => None,
    })
}

/// Allocate this runner's resident bf16 gate buffer (`num_layers * top_k` slots)
/// on a fresh default-stream and bind its FFN gate inputs to fixed offsets once.
fn install_gate_buffer(
    runner: &mut SegmentRunner,
    schedule: &[ResolvedSequenceStep],
    rank: u32,
    num_layers: usize,
    top_k: usize,
) -> Result<(), RuntimeError> {
    let total = num_layers * top_k;
    let ctx = cudarc::driver::CudaContext::new(0)
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
