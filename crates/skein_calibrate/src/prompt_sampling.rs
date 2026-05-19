//! Drift-prompt sampling.
//!
//! Reads a JSONL trace file (one `{"prompt": "..."}` object per line) and
//! samples `n_drift_prompts` according to the configured strategy. Phase A
//! runs entirely on Mac — only the *GPU drift measurement* on the sampled
//! prompts is Phase B.
//!
//! Determinism is part of the contract: same `(trace, n, strategy, seed)` →
//! byte-identical output. Drift calibration depends on this so re-runs
//! don't introduce noise that the parity gate then chases.

use std::path::Path;

use serde::Deserialize;

use crate::corpus::{DriftPromptSource, SamplingStrategy};
use crate::error::CalibrationError;

#[derive(Debug, Deserialize)]
struct TraceLine {
    prompt: String,
}

/// Sample drift prompts per the configured source. Reads the trace from
/// disk, applies the sampling strategy, returns the chosen prompts.
pub fn sample_drift_prompts(source: &DriftPromptSource) -> Result<Vec<String>, CalibrationError> {
    let prompts = read_prompts_jsonl(&source.workload_trace_path)?;
    if prompts.is_empty() {
        return Err(CalibrationError::CorpusInvalid {
            reason: format!(
                "drift-prompt trace at {} has no `{{\"prompt\": ...}}` entries",
                source.workload_trace_path.display()
            ),
        });
    }
    let n = source.n_drift_prompts.min(prompts.len());
    let sampled = match source.sampling_strategy {
        SamplingStrategy::FirstN => first_n(prompts, n),
        SamplingStrategy::UniformRandom => uniform_random(prompts, n, source.sampling_seed),
        SamplingStrategy::StratifiedByLength => stratified_by_length(prompts, n),
    };
    Ok(sampled)
}

pub fn default_public_prompt_source() -> DriftPromptSource {
    DriftPromptSource {
        workload_trace_path: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("corpus")
            .join("drift_prompts.jsonl"),
        n_drift_prompts: 128,
        sampling_strategy: SamplingStrategy::FirstN,
        sampling_seed: 0,
    }
}

/// Read a JSONL file where each line is a `{"prompt": "..."}` object.
/// Blank lines are tolerated. Other JSON shapes are an error.
pub fn read_prompts_jsonl(path: &Path) -> Result<Vec<String>, CalibrationError> {
    let s = std::fs::read_to_string(path).map_err(|source| CalibrationError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = Vec::new();
    for (idx, line) in s.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: TraceLine =
            serde_json::from_str(line).map_err(|e| CalibrationError::CorpusInvalid {
                reason: format!(
                    "line {} of {} does not parse as {{\"prompt\": ...}}: {e}",
                    idx + 1,
                    path.display()
                ),
            })?;
        out.push(parsed.prompt);
    }
    Ok(out)
}

fn first_n(prompts: Vec<String>, n: usize) -> Vec<String> {
    prompts.into_iter().take(n).collect()
}

/// Deterministic shuffle via BLAKE3 of `(seed, original_index)`. Sort by
/// the resulting 64-bit prefix; ties broken by index. Same seed → same
/// permutation across hosts and architectures.
fn uniform_random(prompts: Vec<String>, n: usize, seed: u64) -> Vec<String> {
    let mut indexed: Vec<(u64, usize, String)> = prompts
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let mut buf = [0u8; 16];
            buf[..8].copy_from_slice(&seed.to_le_bytes());
            buf[8..].copy_from_slice(&(i as u64).to_le_bytes());
            let h = blake3::hash(&buf);
            let key = u64::from_le_bytes(h.as_bytes()[..8].try_into().expect("8 bytes"));
            (key, i, p)
        })
        .collect();
    indexed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    indexed.into_iter().take(n).map(|(_, _, p)| p).collect()
}

/// Sort prompts by `(byte length, content)`, then sample one from each of
/// `n` equally-ranked buckets. Picks the middle of each bucket so the
/// sampling spans short → long prompts evenly.
fn stratified_by_length(mut prompts: Vec<String>, n: usize) -> Vec<String> {
    prompts.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    let len = prompts.len();
    if n == 0 || len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Bucket i covers [i*len/n, (i+1)*len/n). Pick its midpoint.
        let lo = i * len / n;
        let hi = (i + 1) * len / n;
        let mid = lo + (hi - lo) / 2;
        out.push(prompts[mid.min(len - 1)].clone());
    }
    out
}
