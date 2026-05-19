//! Corpus loading + validation. The corpus no longer carries
//! `drift_prompts` literally — those would be synthetic data. It carries
//! a `DriftPromptSource` pointing to a real workload trace, sampled at
//! calibration time.

mod common;

use skein_calibrate::corpus::{CalibrationCorpus, SamplingStrategy};
use skein_calibrate::error::CalibrationError;

const VALID: &str = r#"
workload_trace_path = "/tmp/nonexistent.jsonl"
n_drift_prompts     = 50
sampling_strategy   = "stratified_by_length"
sampling_seed       = 7

[[kernel_samples]]
op_kind = "gemm"
dtype = "bf16"
shape = [4096, 4096]
repeats = 100

[[kernel_samples]]
op_kind = "attention"
dtype = "fp8_e4m3"
shape = [8, 32, 2048, 128]
repeats = 50
"#;

#[test]
fn corpus_loading_valid() {
    let corpus = CalibrationCorpus::from_toml_str(VALID).unwrap();
    assert_eq!(corpus.kernel_samples.len(), 2);
    assert_eq!(corpus.kernel_samples[0].shape, vec![4096, 4096]);
    assert_eq!(corpus.kernel_samples[0].repeats, 100);
    // Drift source — structural fields only. The trace file does NOT
    // have to exist for the corpus to load.
    assert_eq!(corpus.drift_source.n_drift_prompts, 50);
    assert_eq!(
        corpus.drift_source.sampling_strategy,
        SamplingStrategy::StratifiedByLength
    );
    assert_eq!(corpus.drift_source.sampling_seed, 7);
    assert_eq!(
        corpus.drift_source.workload_trace_path,
        std::path::PathBuf::from("/tmp/nonexistent.jsonl"),
    );
}

#[test]
fn corpus_loading_shipped_fixture_loads() {
    let path = common::repo_root().join("crates/skein_calibrate/corpus/mixtral_8x7b.toml");
    let corpus = CalibrationCorpus::load(&path).expect("load shipped fixture");
    assert!(!corpus.kernel_samples.is_empty());
    // The shipped corpus references a user-provided trace; that file
    // doesn't ship in the repo. We only check that the corpus parses and
    // that the drift-source config is sensibly populated.
    assert!(corpus.drift_source.n_drift_prompts > 0);
    assert!(
        !corpus
            .drift_source
            .workload_trace_path
            .as_os_str()
            .is_empty(),
        "shipped corpus's workload_trace_path is empty",
    );
    // The shipped corpus uses one of the three known strategies; the
    // serde-derived enum guarantees this.
    let _ = corpus.drift_source.sampling_strategy;
}

#[test]
fn corpus_loading_rejects_missing_op_kind() {
    let bad = r#"
workload_trace_path = "/tmp/x.jsonl"
n_drift_prompts     = 5
sampling_strategy   = "first_n"

[[kernel_samples]]
dtype = "bf16"
shape = [4096]
repeats = 1
"#;
    let err = CalibrationCorpus::from_toml_str(bad).unwrap_err();
    assert!(
        matches!(err, CalibrationError::TomlParse { .. }),
        "expected TomlParse for missing op_kind, got {err:?}"
    );
}

#[test]
fn corpus_loading_rejects_empty_shape() {
    let bad = r#"
workload_trace_path = "/tmp/x.jsonl"
n_drift_prompts     = 5
sampling_strategy   = "first_n"

[[kernel_samples]]
op_kind = "gemm"
dtype = "bf16"
shape = []
repeats = 1
"#;
    match CalibrationCorpus::from_toml_str(bad).unwrap_err() {
        CalibrationError::CorpusInvalid { reason } => {
            assert!(reason.contains("shape is empty"), "{reason}");
        }
        other => panic!("expected CorpusInvalid, got {other:?}"),
    }
}

#[test]
fn corpus_loading_rejects_zero_repeats() {
    let bad = r#"
workload_trace_path = "/tmp/x.jsonl"
n_drift_prompts     = 5
sampling_strategy   = "first_n"

[[kernel_samples]]
op_kind = "gemm"
dtype = "bf16"
shape = [4096]
repeats = 0
"#;
    assert!(matches!(
        CalibrationCorpus::from_toml_str(bad).unwrap_err(),
        CalibrationError::CorpusInvalid { reason } if reason.contains("repeats")
    ));
}

#[test]
fn corpus_loading_rejects_unknown_dtype() {
    let bad = r#"
workload_trace_path = "/tmp/x.jsonl"
n_drift_prompts     = 5
sampling_strategy   = "first_n"

[[kernel_samples]]
op_kind = "gemm"
dtype = "fp64"
shape = [4096]
repeats = 1
"#;
    assert!(matches!(
        CalibrationCorpus::from_toml_str(bad).unwrap_err(),
        CalibrationError::TomlParse { .. }
    ));
}

#[test]
fn corpus_loading_rejects_zero_n_drift_prompts() {
    let bad = r#"
workload_trace_path = "/tmp/x.jsonl"
n_drift_prompts     = 0
sampling_strategy   = "first_n"
"#;
    assert!(matches!(
        CalibrationCorpus::from_toml_str(bad).unwrap_err(),
        CalibrationError::CorpusInvalid { reason } if reason.contains("n_drift_prompts")
    ));
}

#[test]
fn corpus_loading_rejects_empty_workload_trace_path() {
    let bad = r#"
workload_trace_path = ""
n_drift_prompts     = 5
sampling_strategy   = "first_n"
"#;
    assert!(matches!(
        CalibrationCorpus::from_toml_str(bad).unwrap_err(),
        CalibrationError::CorpusInvalid { reason } if reason.contains("workload_trace_path")
    ));
}

#[test]
fn corpus_loading_rejects_unknown_strategy() {
    let bad = r#"
workload_trace_path = "/tmp/x.jsonl"
n_drift_prompts     = 5
sampling_strategy   = "round_robin"
"#;
    assert!(matches!(
        CalibrationCorpus::from_toml_str(bad).unwrap_err(),
        CalibrationError::TomlParse { .. }
    ));
}
