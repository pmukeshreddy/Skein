//! `Server` — owns the runtime subsystems and exposes the public API.
//!
//! Construction + `submit` + hot-swap + observability all work, and `serve()`
//! now exposes the Phase B axum streaming endpoint backed by the shared
//! topology executor.

pub mod forward;
pub mod lifecycle;

pub use lifecycle::Server;
