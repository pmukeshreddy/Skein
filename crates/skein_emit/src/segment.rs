//! Multi-segment lowering types — one [`Segment`] per collective-bracketed
//! region of a device's forward pass.
//!
//! ## Why segments
//!
//! Luminal doesn't model collectives, and its HLIR has no `NoOp` we can
//! use as a marker (see `docs/luminal_integration.md`). Under any TP / EP
//! / PP > 1 plan, the per-device forward pass is interrupted by NCCL
//! collectives that the runtime issues between graph executions. The
//! only viable architecture is **one Luminal graph segment per
//! collective-bracketed region**, with the runtime threading buffers
//! across segments and emitting the right NCCL call in between.
//!
//! For tp=1 ep=1 pp=1 there are zero collective points, so each device
//! has exactly one segment containing the whole forward pass.
//!
//! ## Handoff
//!
//! Adjacent segments share data through [`HandoffTensor`] entries by
//! *logical name*. The runtime owns the buffers between segment
//! executions and:
//!   * Re-feeds carry tensors (logical names like `carry_*`) into the
//!     next segment unchanged.
//!   * Issues the collective named in the [`SequenceStep::Collective`]
//!     entry, transforming the buffer in place; the next segment's
//!     `input_handoff` references the same logical name and reads the
//!     post-collective bytes.
//!
//! ## Sequencing
//!
//! [`LoweredGraph::sequencing`] is the device's execution schedule:
//! `ExecuteSegment(0) → Collective(0) → ExecuteSegment(1) → ... →
//! ExecuteSegment(N)`. For an N-collective device, sequencing has
//! `2N + 1` steps and `N + 1` segments. The runtime walks this list
//! end-to-end per token.

use std::collections::HashMap;

use luminal::prelude::{Graph as LuminalGraph, NodeIndex};
use serde::{Deserialize, Serialize};

use skein_cost::collectives::CollectiveKind;
use skein_ir::types::Dtype;

use crate::graph_builder::DeclaredTensor;

/// One Luminal graph segment — a contiguous run of ops between two
/// collective points on a single device.
///
/// `graph` is a complete `luminal::Graph`: `cx.build_search_space::<R>()`
/// and `cx.search()` run on it independently of any other segment.
/// `input_handoff` / `output_handoff` describe the data flow at the
/// segment's boundaries; `declared` and `op_nodes` carry the same per-tensor
/// metadata the `LoweredGraph` exposes, scoped to this segment.
pub struct Segment {
    pub idx: usize,
    pub graph: LuminalGraph,
    pub declared: HashMap<String, DeclaredTensor>,
    pub op_nodes: HashMap<String, NodeIndex>,
    pub input_handoff: Vec<HandoffTensor>,
    pub output_handoff: Vec<HandoffTensor>,
}

/// A tensor that crosses a segment boundary.
///
/// `logical_name` is the cross-segment identifier (see
/// `crate::handoff` for the naming convention). `luminal_id` is the
/// `NodeIndex` in *this segment's* graph — different segments may have
/// different `NodeIndex` values for the same `logical_name` since they
/// live in different `luminal::Graph` instances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoffTensor {
    pub logical_name: String,
    pub luminal_id: NodeIndex,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

/// One step in the device's execution schedule.
///
/// Serialization is stable: `ExecuteSegment` and `Collective` are
/// distinguished by the externally tagged `kind` field, and all inner
/// fields serialize as plain JSON. Test 4 in `segmentation.rs` pins
/// byte-equality across runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SequenceStep {
    /// Run segment `segment_idx` of device `device_idx`'s [`LoweredGraph`].
    ExecuteSegment { device_idx: u32, segment_idx: usize },
    /// Issue a collective. `tensor` is the [`HandoffTensor::logical_name`]
    /// shared by the upstream segment's output and the downstream
    /// segment's input.
    Collective {
        collective: CollectiveKind,
        participants: Vec<u32>,
        tensor: String,
        shape: Vec<usize>,
        dtype: Dtype,
    },
}
