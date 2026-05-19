//! Drift-prompt sampling tests.
//!
//! Two contracts to verify:
//!
//! 1. Sampling is deterministic — same `(trace, n, strategy, seed)` →
//!    byte-identical output across runs.
//! 2. Each strategy produces a sensible sample. FirstN preserves trace
//!    order; UniformRandom doesn't (and changes with seed);
//!    StratifiedByLength spans short → long.

mod common;

use std::path::PathBuf;

use skein_calibrate::corpus::{DriftPromptSource, SamplingStrategy};
use skein_calibrate::prompt_sampling::{read_prompts_jsonl, sample_drift_prompts};

fn fixture_path() -> PathBuf {
    common::repo_root().join("crates/skein_calibrate/tests/fixtures/drift_prompts.jsonl")
}

fn mk_source(strategy: SamplingStrategy, n: usize, seed: u64) -> DriftPromptSource {
    DriftPromptSource {
        workload_trace_path: fixture_path(),
        n_drift_prompts: n,
        sampling_strategy: strategy,
        sampling_seed: seed,
    }
}

#[test]
fn fixture_loads_all_prompts() {
    let prompts = read_prompts_jsonl(&fixture_path()).unwrap();
    assert_eq!(prompts.len(), 10);
    assert_eq!(prompts[0], "alpha short");
    assert_eq!(prompts[9].split_whitespace().next().unwrap(), "kappa");
}

#[test]
fn prompt_sampling_deterministic() {
    // Same configuration → byte-identical output across two runs.
    for strategy in [
        SamplingStrategy::FirstN,
        SamplingStrategy::UniformRandom,
        SamplingStrategy::StratifiedByLength,
    ] {
        let src = mk_source(strategy, 5, 42);
        let a = sample_drift_prompts(&src).unwrap();
        let b = sample_drift_prompts(&src).unwrap();
        assert_eq!(a, b, "strategy {strategy:?} non-deterministic");
        assert_eq!(a.len(), 5);
    }
}

#[test]
fn first_n_preserves_trace_order() {
    let src = mk_source(SamplingStrategy::FirstN, 3, 0);
    let s = sample_drift_prompts(&src).unwrap();
    assert_eq!(s.len(), 3);
    assert_eq!(s[0], "alpha short");
    assert_eq!(s[1], "beta a bit longer");
    assert_eq!(s[2], "charlie even more text here for length variation");
}

#[test]
fn uniform_random_changes_with_seed_and_doesnt_match_first_n() {
    let first_n = sample_drift_prompts(&mk_source(SamplingStrategy::FirstN, 5, 0)).unwrap();
    let seed0 = sample_drift_prompts(&mk_source(SamplingStrategy::UniformRandom, 5, 0)).unwrap();
    let seed42 = sample_drift_prompts(&mk_source(SamplingStrategy::UniformRandom, 5, 42)).unwrap();

    // 10-prompt corpus → the chance that a deterministic BLAKE3-based
    // shuffle of seed=0 picks the exact same 5 entries in the exact same
    // order as the file is negligible. If this assertion ever flakes,
    // re-roll the seed.
    assert_ne!(first_n, seed0, "UniformRandom seed=0 collided with FirstN");
    assert_ne!(seed0, seed42, "UniformRandom seed=42 collided with seed=0");
    assert_eq!(seed0.len(), 5);
    assert_eq!(seed42.len(), 5);
    // Sanity: every sampled prompt is in the source.
    let source = read_prompts_jsonl(&fixture_path()).unwrap();
    for p in &seed0 {
        assert!(source.contains(p), "sampled {p:?} not in source");
    }
}

#[test]
fn stratified_by_length_spans_short_to_long() {
    // Pick 3 — should land roughly short / medium / long.
    let src = mk_source(SamplingStrategy::StratifiedByLength, 3, 0);
    let s = sample_drift_prompts(&src).unwrap();
    assert_eq!(s.len(), 3);
    // Lengths should be increasing (or at least not decreasing) since
    // stratified buckets are taken in length-sorted order.
    let lens: Vec<usize> = s.iter().map(|p| p.len()).collect();
    assert!(
        lens[0] <= lens[1] && lens[1] <= lens[2],
        "stratified lengths not monotonic: {lens:?}"
    );
    // The shortest bucket should pick a short prompt; the longest should
    // pick a long one. Concretely, the longest prompt in the fixture has
    // > 80 characters; the shortest is < 10.
    assert!(lens[0] < 30, "shortest bucket too long: {}", lens[0]);
    assert!(lens[2] > 30, "longest bucket too short: {}", lens[2]);
}

#[test]
fn n_larger_than_trace_clamps() {
    let src = mk_source(SamplingStrategy::FirstN, 100, 0);
    let s = sample_drift_prompts(&src).unwrap();
    // Fixture has 10 entries; asking for 100 returns all 10.
    assert_eq!(s.len(), 10);
}

#[test]
fn nonexistent_trace_errors() {
    let src = DriftPromptSource {
        workload_trace_path: PathBuf::from("/tmp/does-not-exist.jsonl"),
        n_drift_prompts: 5,
        sampling_strategy: SamplingStrategy::FirstN,
        sampling_seed: 0,
    };
    assert!(sample_drift_prompts(&src).is_err());
}
