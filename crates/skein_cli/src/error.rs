//! `CliError` — the CLI boundary error type. Every command runner returns
//! `Result<(), CliError>`; `main` formats and exits non-zero on `Err`.
//!
//! This is the one place in Skein where `anyhow` is allowed — library
//! crates still use `thiserror`, but the CLI bridges them all via the
//! `Other(#[from] anyhow::Error)` variant so a single match in `main`
//! handles every error path.

#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// A CUDA-only subcommand invoked on a CPU (`--no-default-features`)
    /// build. `what` names the subcommand, `reason` explains why CUDA is
    /// needed, and `suggested_fix` gives the exact rebuild + invocation
    /// command.
    #[error(
        "{what} requires a CUDA build.\n\
         reason: {reason}\n\
         suggested fix: {suggested_fix}"
    )]
    RequiresCuda {
        what: &'static str,
        reason: &'static str,
        suggested_fix: &'static str,
    },

    /// A subcommand not available in this build configuration.
    #[error("{what}: {detail}")]
    NotImplemented {
        what: &'static str,
        detail: &'static str,
    },

    /// User-visible argument-level failure.
    #[error("{0}")]
    BadArgument(String),

    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("cost model: {0}")]
    Cost(#[from] skein_cost::CostError),

    #[error("plan search: {0}")]
    Extract(#[from] skein_extract::ExtractError),

    #[error("emit/lowering: {0}")]
    Emit(#[from] skein_emit::EmitError),

    #[error("compile: {0}")]
    Compile(#[from] skein_compile::CompileError),

    #[error("parity: {0}")]
    Parity(#[from] skein_parity::ParityError),

    #[error("calibration: {0}")]
    Calibration(#[from] skein_calibrate::CalibrationError),

    #[error("runtime: {0}")]
    Runtime(#[from] skein_runtime::RuntimeError),

    #[error("plan hash: {0}")]
    PlanHash(#[from] skein_ir::error::PlanError),

    #[error("parity failed for artifact {artifact}")]
    ParityFailed { artifact: String },

    /// Library errors bubble up via anyhow's blanket `From<E>` so the CLI
    /// doesn't need a variant per leaf type. Loaders in `load.rs` return
    /// `anyhow::Result<_>` and attach `.context()` for traceability.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
