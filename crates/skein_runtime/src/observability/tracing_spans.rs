//! Structured tracing spans for request and step lifecycle.
//!
//! Emits `tracing::Span` via the `tracing` crate. A `tracing-subscriber`
//! installed by the consumer determines what happens to the spans (stdout,
//! file, OTLP, etc.). This module does not install or own a subscriber.
//!
//! OTLP / OpenTelemetry exporter integration is not in this module; it
//! belongs in deployment-level wiring and is tracked separately.

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
