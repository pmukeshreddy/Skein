//! `skein_cost` — analytic cost model for `Plan` scoring.
//!
//! The cost model runs on Mac. It has zero CUDA-feature-gated code.
//!
//! Five additive terms (all in microseconds, per device):
//!
//! 1. **Compute time** — `flops / (peak_TFLOPS × eff × tp)`. TP splits the
//!    matmul across `tp` devices working in parallel.
//! 2. **Communication time** — topology-graph-aware. Latency = sum along the
//!    critical path; transfer time = `factor × bytes / bottleneck_bw / eff`.
//! 3. **Memory penalty** — zero when the per-device peak memory fits the
//!    device cap; otherwise overshoot × a deliberately huge constant.
//! 4. **Pipeline bubble** — `((pp - 1) / pp) × stage_compute`. Zero for pp=1.
//! 5. **Launch overhead** — `launch_us_per_kernel × kernels × (1 - coverage)`.
//!
//! `total_cost` returns **max over devices**, not sum/average — the slowest
//! device dominates wall-clock. See `docs/cost_model.md` for the full write-up.
//!
//! ## What this crate does *not* do
//!
//! - No accuracy term in the cost. Drift stays a DP constraint in
//!   `skein_extract`.
//! - No hardcoded constants. Every number comes from `cost_constants.toml`.
//! - No fallback comm models. If the topology has no path between two
//!   participants, `total_cost` returns `Err`.

pub mod bubble;
pub mod cluster;
pub mod collectives;
pub mod comm;
pub mod compute;
pub mod constants;
pub mod error;
pub mod launch;
pub mod memory;
pub mod model;
pub mod topology;
pub mod workload_ctx;

pub use cluster::Cluster;
pub use collectives::{Collective, CollectiveKind};
pub use compute::OpKind;
pub use constants::CostConstants;
pub use error::CostError;
pub use model::{Cost, CostModel};
pub use topology::{Hop, Path, Topology};
pub use workload_ctx::WorkloadCtx;
