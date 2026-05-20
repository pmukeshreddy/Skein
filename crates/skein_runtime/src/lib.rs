//! `skein_runtime` — the serving runtime.
//!
//! Five composable subsystems:
//!
//! - [`kv::PagedKVAllocator`] — paged KV blocks + LRU eviction + optional
//!   radix prefix cache.
//! - [`batcher::ContinuousBatcher`] — SLO-aware admission queue, in-flight
//!   set, retire path, chunked-prefill state machine.
//! - [`hotswap::HotSwap`] — drain in-flight, verify shape compatibility,
//!   atomic POSIX-rename symlink swap.
//! - [`observability::ProfileHooks`] — step + request metric ring buffers,
//!   Prometheus exporter, `tracing` spans.
//! - [`server::Server`] — owns the above and exposes a unified `submit /
//!   serve` API. `serve()` runs the HTTP streaming path through the shared
//!   topology executor.
//!
//! The default build targets CUDA (NCCL collectives, CUDA Graphs dispatch,
//! RDMA KV transport) behind `#[cfg(feature = "cuda")]`. Building with
//! `--no-default-features` selects the in-process collective + eager
//! dispatch path used for single-node serving, CI, and runtime-logic
//! development.

pub mod batcher;
pub mod collectives;
pub mod cuda;
pub mod dispatch;
pub mod distributed;
pub mod drift_monitor;
pub mod error;
pub mod hotswap;
pub mod kv;
pub mod kv_cache;
pub mod kv_transport;
pub mod observability;
pub mod server;
pub mod speculative;
pub mod token_stream;
pub mod tokenizer;
pub mod types;

pub use batcher::{AdmissionDecision, ContinuousBatcher, RejectReason, StepBatch};
pub use collectives::{CollectiveBackend, CollectiveError, InProcessCollective};
pub use dispatch::{DispatchError, DispatchOutcome, EagerDispatcher, KernelDispatcher};
pub use drift_monitor::{DriftAssessment, DriftMonitor, WorkloadProfile};
pub use distributed::{
    BarrierCollective, CollectiveError as RankCollectiveError, LocalSegments, RankCollective,
    RankExecutor, SegmentRunner, WorldLayout,
};
pub use error::RuntimeError;
pub use hotswap::HotSwap;
pub use kv::{PagedKVAllocator, RadixPrefixTree};
pub use kv_transport::{KvTransport, LocalKvTransport, TransportError};
pub use observability::{ProfileHooks, RequestMetrics, StepMetrics};
pub use server::Server;
pub use speculative::{SpecError, SpecOutcome, verify as speculative_verify};
pub use token_stream::TokenStreamer;
pub use tokenizer::SkeinTokenizer;
pub use types::{IncomingRequest, RequestId, TokenOutput};
