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
use std::time::Duration;

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

    /// Run one forward step for `tokens` (feeds the last token id), driving this
    /// rank's segments + NCCL collectives over the schedule. Returns this rank's
    /// logits — full vocab on TP/last-stage ranks after the logits all-gather.
    pub fn forward_step(&mut self, tokens: &[u32]) -> Result<Vec<f32>, RuntimeError> {
        let last = *tokens.last().unwrap_or(&0) as i32;
        self.executor
            .runner_mut()
            .set_input_tokens(INPUT_TOKENS, vec![last]);
        self.executor
            .run(&self.schedule, &self.collective)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
        self.executor
            .runner()
            .read(LOGITS)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))
    }

    /// Greedy lockstep decode of `max_new_tokens` from `prompt_tokens`. Every
    /// rank runs this identically (same prompt → same logits → same argmax), so
    /// the ranks stay in step without per-token communication. Returns the
    /// generated token ids (identical on every rank).
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, RuntimeError> {
        let mut tokens = prompt_tokens.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let logits = self.forward_step(&tokens)?;
            let next = argmax(&logits);
            tokens.push(next);
            generated.push(next);
        }
        Ok(generated)
    }
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

    let generated = server.generate(&prompt_tokens, max_new_tokens)?;

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

    // Forwards: running sequence -> next-token logits. Errors degrade to an
    // empty distribution (logged) so the loop's signature stays infallible.
    let mut target_fwd = |seq: &[u32]| match target.forward_step(seq) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%e, "target forward failed");
            Vec::new()
        }
    };
    let mut draft_fwd = |seq: &[u32]| match draft.forward_step(seq) {
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
