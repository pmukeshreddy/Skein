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
//!   Prometheus exporter, `tracing` spans (full OTel exporter wiring is
//!   Phase B).
//! - [`server::Server`] — owns the above and exposes a unified `submit /
//!   serve` API. `serve()` runs the Phase B HTTP streaming path through the
//!   shared topology executor.
//!
//! The Mac build stays CPU-only and keeps CUDA imports behind
//! `#[cfg(feature = "cuda")]`; native collectives, dispatch, and KV
//! transport run real single-process semantics.

pub mod batcher;
pub mod collectives;
pub mod cuda;
pub mod dispatch;
pub mod error;
pub mod hotswap;
pub mod kv;
pub mod kv_transport;
pub mod observability;
pub mod server;
pub mod token_stream;
pub mod types;

pub use batcher::{AdmissionDecision, ContinuousBatcher, RejectReason, StepBatch};
pub use collectives::{CollectiveBackend, CollectiveError, MockCollective};
pub use dispatch::{DispatchError, DispatchOutcome, EagerDispatcher, KernelDispatcher};
pub use error::RuntimeError;
pub use hotswap::HotSwap;
pub use kv::{PagedKVAllocator, RadixPrefixTree};
pub use kv_transport::{KvTransport, LocalKvTransport, TransportError};
pub use observability::{ProfileHooks, RequestMetrics, StepMetrics};
pub use server::Server;
pub use token_stream::TokenStreamer;
pub use types::{IncomingRequest, RequestId, TokenOutput};
