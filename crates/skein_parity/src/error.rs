//! `ParityError` — one `thiserror` enum across comparison, reference,
//! drift-update, and Phase-B gating.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ParityError {
    #[error("activation/logit shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error(
        "Plan dtype_map has {actual} per-block entries but the IR has \
         {expected} decoder blocks"
    )]
    DtypeMapIncomplete { expected: usize, actual: usize },

    #[error(
        "cost_constants.toml has no [parity_tolerance_mse] section; this \
         must come from {0} so per-dtype tolerances are calibrated, not \
         hardcoded"
    )]
    MissingTolerance(PathBuf),

    #[error("could not read mock-reference fixture at {path}: {source}")]
    FixtureIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not parse mock-reference fixture at {path}: {source}")]
    FixtureParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("mock-reference fixture has no entry for tokens {tokens:?}")]
    NoFixtureMatch { tokens: Vec<u32> },

    #[error("unsupported reference dtype {dtype:?}; expected one of: bfloat16, float16, float32")]
    InvalidReferenceDtype { dtype: String },

    #[error("python subprocess I/O failed while {action}: {source}")]
    PythonIo {
        action: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("python subprocess timed out after {timeout_ms} ms; stderr: {stderr}")]
    PythonSubprocessTimeout { timeout_ms: u64, stderr: String },

    #[error("python subprocess failed with exit code {exit_code:?}: {stderr}")]
    PythonSubprocessFailed {
        exit_code: Option<i32>,
        stderr: String,
    },

    #[error("python subprocess returned invalid protocol data: {0}")]
    PythonProtocol(String),

    #[error("compile/runtime error: {0}")]
    Compile(#[from] skein_compile::CompileError),

    #[error("could not serialize parity report: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("Plan content hash error: {0}")]
    PlanHash(#[from] skein_ir::error::PlanError),

    #[error("drift table I/O error: {0}")]
    DriftTable(#[from] skein_extract::ExtractError),

    #[error(
        "{what} requires the Phase B CUDA build (rebuild with `--features cuda` \
         on an H100 host); Mac/Phase A cannot run this code path"
    )]
    RequiresCuda { what: &'static str },

    #[error(
        "{what} is a Phase B implementation: the structure is in place but \
         the real implementation lands in Phase B Step 10"
    )]
    PhaseBOnly { what: &'static str },
}
