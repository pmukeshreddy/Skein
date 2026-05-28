//! Cross-subsystem newtypes.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

impl RequestId {
    /// Monotonic ID source. Tests can construct `RequestId(n)` directly;
    /// production callers should use this so concurrent submitters never
    /// collide.
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        RequestId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// An incoming request as the runtime sees it. The token list has already
/// been tokenized by the front door (the HTTP / gRPC server).
#[derive(Debug, Clone)]
pub struct IncomingRequest {
    pub id: RequestId,
    pub prompt_tokens: Vec<u32>,
    pub max_output_tokens: u32,
    /// Arrival time in milliseconds since the Unix epoch (or any monotonic
    /// reference; tests use 0-anchored values).
    pub arrival_ms: u64,
}

/// One unit of output the runtime streams back to the front door.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenOutput {
    pub token: u32,
    pub is_final: bool,
}
