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

use std::path::Path;
use std::time::{Duration, Instant};

use skein_compile::{
    CudaComputeRuntime, DEFAULT_SEARCH_BUDGET, SkeinArtifact, load_device_runtime_segments,
};
use skein_emit::segment::SequenceStep;

use crate::cuda::nccl::NcclCollective;
use crate::distributed::{
    CollectiveError, LocalSegments, RankCollective, RankExecutor, SegmentRunner, WorldLayout,
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
    schedule: Vec<SequenceStep>,
    vocab: u32,
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

        let executor = RankExecutor::new(layout.rank, SegmentRunner::new(segments));
        Ok(Self {
            layout,
            executor,
            collective,
            schedule: artifact.sequencing.clone(),
            vocab: artifact.plan.model_meta.vocab as u32,
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
        let runner = self.executor.runner_mut();
        runner.set_input_tokens(INPUT_TOKENS, vec![token as i32]);
        runner.set_position(position);
        self.executor
            .run(&self.schedule, &self.collective)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
        self.executor
            .runner()
            .read(LOGITS)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))
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
        for &tok in &prompt_tokens[prefill_start_pos..] {
            logits = self.forward_step(tok, position)?;
            position += 1;
        }
        let ttft = prefill_start.elapsed();
        let prefill_steps = prompt_tokens.len() - prefill_start_pos;

        // Decode: argmax the current logits, emit, and feed it back as the next
        // token at the running position. Grow the request's pages by one token
        // per step (paged KV). Time each decode step (TPOT).
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut step_times: Vec<Duration> = Vec::new();
        for i in 0..max_new_tokens {
            let next = argmax(&logits);
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

        let pages_in_use = self.executor.runner().kv_pages_in_use();
        // Release the request: its pages return to the cache so the next request
        // sharing this prompt's prefix can reuse them.
        self.executor.runner_mut().end_request();

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
                "SKEIN_PERF: paged-KV cached-decode timing (single in-flight request, greedy)"
            );
        }
        Ok(result)
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

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}
