//! `skein_ir` — typed IR, `Plan`, `ClusterSpec`, `Workload` value types, and
//! the HF `config.json` importer.
//!
//! The IR is hardware-agnostic: it represents *what* a model computes, not
//! *how* it should be sharded or quantized. Those decisions live in `Plan`,
//! which `skein_extract` produces and `skein_emit` consumes.
//!
//! Every type that crosses a process boundary (`Plan`, `ClusterSpec`,
//! `Workload`, the IR itself) is `serde`-serializable. `Plan` additionally
//! exposes a deterministic `content_hash` used for content-addressed artifact
//! directories (`artifacts/<plan_hash>/...`).
//!
//! No floating-point fields appear in equality-critical Plan slots; floats
//! are confined to `ModelMeta` (rope_theta, rms_norm_eps) and `Slo`
//! (drift thresholds) where they are sourced from the input config / trace.

pub mod cluster;
pub mod error;
pub mod ir;
pub mod model;
pub mod plan;
pub mod types;
pub mod workload;
