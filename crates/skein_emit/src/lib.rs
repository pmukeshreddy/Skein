//! `skein_emit` — lowers a `Plan` to per-device `luminal::Graph` instances
//! with sharded tensor shapes baked in, plus a per-device weight shard and
//! a cluster-wide `topology.json` describing the NCCL collectives the
//! runtime must issue between graph invocations.
//!
//! Phase A scope: build the graphs and the slicing/topology metadata. No
//! `cx.build_search_space()` / `cx.search()` — those are Phase B (`skein_compile`).
//! Phase A tests verify graph structure, weight-slice byte ranges, and
//! topology determinism without compiling anything through Luminal.
//!
//! # Public surface
//!
//! - [`lower_per_device`] — the headline entry point. Returns a
//!   [`DeviceArtifact`] with the per-device graph, weight shard, and IO
//!   manifest.
//! - [`emit_topology`] — cluster-wide collective list. Same `(Plan, IR)`
//!   produces byte-identical output (used by `skein` for content-addressable
//!   artifact directories in Phase A Step 6).
//! - [`shard_role`] — pure resolution of a parameter's per-device role
//!   (replicated, TP-sharded, EP-owned, etc.).
//! - [`weights::write_weight_shard`] — writes the device's slice to a
//!   destination safetensors file. Phase A tests exercise this with a
//!   synthetic fixture; the real Mixtral path runs in Phase B.

pub mod error;
pub mod graph_builder;
pub mod handoff;
pub mod io_manifest;
pub mod op_wiring;
pub mod segment;
pub mod shard_role;
pub mod topology;
pub mod weights;

pub use error::EmitError;
pub use graph_builder::{DeclaredTensor, LoweredGraph, build_device_graph};
pub use handoff::{CollectivePoint, device_collective_points};
pub use io_manifest::{IoManifest, IoTensor, IoTensorKind};
pub use segment::{HandoffTensor, Segment, SequenceStep};
pub use shard_role::{ShardRole, shard_role_for_param};
pub use topology::{Topology, TopologyEntry, emit_topology};
pub use weights::{ShardStrategy, WeightShard, WeightSlice, build_weight_shard};

use std::path::Path;

use skein_cost::Cluster;
use skein_ir::ir::Graph;
use skein_ir::plan::Plan;

/// The per-device artifact `skein_emit` produces.
///
/// `graph` carries the uncompiled `luminal::Graph` *and* a name-indexed
/// `declared` map so tests / Phase B's `skein_compile` can look up a
/// parameter's declared shape without walking the graph manually.
/// `weight_shard` lists the source byte ranges this device owns;
/// `io_manifest` is the manifest the runtime uses to validate at load time.
pub struct DeviceArtifact {
    pub device_idx: u32,
    pub graph: LoweredGraph,
    pub weight_shard: WeightShard,
    pub io_manifest: IoManifest,
}

/// Lower a `Plan` to a single device's artifact.
///
/// `source_weights_dir` is the path to the model's source safetensors
/// directory. `lower_per_device` does *not* read from it — it only records
/// pointers (via `WeightSlice::source_key`). Materializing the actual shard
/// bytes is `weights::write_weight_shard`'s job.
pub fn lower_per_device(
    plan: &Plan,
    cluster: &Cluster,
    ir: &Graph,
    device_idx: u32,
    source_weights_dir: &Path,
) -> Result<DeviceArtifact, EmitError> {
    let graph = build_device_graph(plan, cluster, ir, device_idx)?;
    let weight_shard = build_weight_shard(plan, cluster, ir, device_idx, source_weights_dir)?;
    let io_manifest = io_manifest::build_io_manifest(plan, cluster, ir, device_idx)?;
    Ok(DeviceArtifact {
        device_idx,
        graph,
        weight_shard,
        io_manifest,
    })
}
