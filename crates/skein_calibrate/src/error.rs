//! `CalibrationError` — one `thiserror` enum across corpus, aggregation,
//! writers, and Phase B gating.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not parse TOML at {path}: {source}")]
    TomlParse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("calibration corpus is invalid: {reason}")]
    CorpusInvalid { reason: String },

    #[error("cost-model error: {0}")]
    Cost(#[from] skein_cost::CostError),

    #[error("drift-table error: {0}")]
    DriftTable(#[from] skein_extract::ExtractError),

    #[error("compile/runtime calibration error: {0}")]
    Compile(#[from] skein_compile::CompileError),

    #[error("parity calibration error: {0}")]
    Parity(#[from] skein_parity::ParityError),

    #[error("calibration sample is unsupported: {reason}")]
    UnsupportedSample { reason: String },
}
