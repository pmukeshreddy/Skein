//! Trace emission. Phase A uses `tracing::info_span!` so requests / steps
//! show up under any installed `tracing-subscriber`. The full OpenTelemetry
//! exporter (OTLP, batched, with resource attributes) lands in Phase B once
//! the deployment story for the collector is locked in.

use crate::types::RequestId;

/// Emit a span at request entry. Returns a `tracing::Span` the caller
/// `.entered()` for the request's lifetime.
pub fn request_span(id: RequestId, prompt_tokens: u32) -> tracing::Span {
    tracing::info_span!("skein.request", request_id = id.0, prompt_tokens)
}

/// Emit a span at step entry. Same `tracing::Span` semantics.
pub fn step_span(step_idx: u64, batch_size: u32) -> tracing::Span {
    tracing::info_span!("skein.step", step_idx, batch_size)
}
