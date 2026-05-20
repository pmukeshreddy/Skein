//! `Server` — owns the runtime subsystems and exposes the public API.
//!
//! Construction, `submit`, hot-swap, and observability are all wired, and
//! `serve()` exposes the axum streaming endpoint backed by the shared
//! topology executor.

pub mod forward;
pub mod lifecycle;

pub use lifecycle::Server;
