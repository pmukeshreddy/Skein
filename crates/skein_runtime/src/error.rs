//! `RuntimeError` — one `thiserror` enum across kv / batcher / hotswap /
//! observability / server. CUDA-specific helper modules keep explicit
//! unavailable-path variants so direct misuse remains visible.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not parse JSON at {path}: {source}")]
    JsonParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("request id {0} not found in the in-flight set")]
    UnknownRequest(u64),

    #[error("KV allocator has no free pages and no evictable pages")]
    KvExhausted,

    #[error(
        "KV allocator capacity {capacity_pages} too small for prompt of \
         {prompt_pages} pages — Plan or workload mis-sized for this device"
    )]
    KvUndersized {
        capacity_pages: u32,
        prompt_pages: u32,
    },

    #[error(
        "candidate artifact at {new} is incompatible with the running \
         artifact at {old}: {reason}"
    )]
    IncompatibleArtifact {
        old: PathBuf,
        new: PathBuf,
        reason: String,
    },

    #[error(
        "drain timed out with {remaining} requests still in-flight after \
         {timeout_seconds} s; caller decides whether to force-kill or extend"
    )]
    DrainTimeout {
        remaining: u32,
        timeout_seconds: u32,
    },

    #[error("prometheus exporter bind failed on port {port}: {source}")]
    PrometheusBind {
        port: u16,
        #[source]
        source: std::io::Error,
    },

    #[error("prometheus encode error: {0}")]
    PrometheusEncode(#[from] prometheus::Error),

    #[error("cost-model error: {0}")]
    Cost(#[from] skein_cost::CostError),

    #[error("compile/runtime executor error: {0}")]
    Compile(#[from] skein_compile::CompileError),

    #[error("dispatch error: {0}")]
    Dispatch(#[from] crate::dispatch::DispatchError),

    #[error("transport error: {0}")]
    Transport(#[from] crate::kv_transport::TransportError),

    #[error("HTTP server bind failed on port {port}: {source}")]
    HttpBind {
        port: u16,
        #[source]
        source: std::io::Error,
    },

    #[error("server forward worker failed to initialize: {0}")]
    ServerInit(String),

    #[error(
        "{what} requires the Phase B CUDA build (rebuild with `--features \
         cuda` on an H100 host); Mac/Phase A cannot run this code path"
    )]
    RequiresCuda { what: &'static str },

    #[error(
        "{what} is a Phase B implementation: the structure is in place but \
         the real implementation lands in Phase B Step 11"
    )]
    PhaseBOnly { what: &'static str },
}
