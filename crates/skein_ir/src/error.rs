//! Error types for `skein_ir`. Each surface (importer, cluster parser,
//! workload parser, plan) has its own `thiserror` enum so callers can match
//! on the actual failure mode rather than collapsing everything into
//! `anyhow::Error`. `anyhow` is reserved for CLI boundaries.

use std::path::PathBuf;

/// Failure modes for the HF `config.json` → IR importer.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("could not read model config at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("model config is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("model config has no `architectures` array; cannot dispatch")]
    NoArchitecture,

    #[error(
        "architecture {arch} is recognized by Skein but not yet wired through \
         the importer; only MixtralForCausalLM is implemented in Phase A"
    )]
    ArchitectureNotYetImplemented { arch: String },

    #[error("architecture {0} is not supported by Skein")]
    UnsupportedArchitecture(String),

    #[error(
        "model config field {field} = {value}, but {constraint}; this would \
         produce an inconsistent IR"
    )]
    InvalidConfig {
        field: &'static str,
        value: String,
        constraint: &'static str,
    },
}

/// Failure modes for the cluster TOML parser.
#[derive(Debug, thiserror::Error)]
pub enum ClusterError {
    #[error("could not read cluster spec at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cluster spec is not valid TOML: {0}")]
    Toml(#[from] toml::de::Error),

    #[error(
        "cluster spec declares num_devices={declared} but the node list sums \
         to {actual}"
    )]
    DeviceCountMismatch { declared: u32, actual: u32 },

    #[error("link references device {device} which is not present in any node")]
    UnknownDeviceInLink { device: String },

    #[error("link endpoints {a} and {b} are the same device; self-links are not valid")]
    SelfLink { a: String, b: String },

    #[error("device id {0} appears in more than one node")]
    DuplicateDevice(String),

    #[error("node id {0} appears more than once")]
    DuplicateNode(String),
}

/// Failure modes for the workload JSONL parser.
#[derive(Debug, thiserror::Error)]
pub enum WorkloadError {
    #[error("could not read workload trace at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("workload trace is empty; first line must be the SLO header")]
    Empty,

    #[error("line {line}: failed to parse as JSON: {source}")]
    Json {
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error(
        "line 1 of the trace must contain an `slo` object; got something else \
         (the SLO header is mandatory)"
    )]
    MissingSlo,

    #[error(
        "request on line {line} has arrival_ms={arrival} which precedes the \
         previous request's arrival ({previous}); the trace must be \
         monotonically non-decreasing"
    )]
    NonMonotonicArrival {
        line: usize,
        arrival: u64,
        previous: u64,
    },
}

/// Failure modes when hashing or composing a `Plan`.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error(
        "DtypeMap has {actual} per-layer entries but the model has {expected} \
         layers; these must agree"
    )]
    DtypeMapLengthMismatch { expected: usize, actual: usize },

    #[error("failed to serialize Plan for content hashing: {0}")]
    HashSerialize(#[from] serde_json::Error),
}
