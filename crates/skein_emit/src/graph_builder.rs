//! Walks the IR for one device, declaring a `luminal::GraphTensor` per
//! parameter the device owns. The declared shapes already reflect the
//! Plan's TP/EP sharding — Luminal's compile pass operates on these shapes
//! and does not need to know about Skein's sharding logic.
//!
//! This module owns the weight-declaration half of lowering (every weight
//! tensor declared with the right sharded shape and dtype). The op edges —
//! matmuls, residual adds, the per-block forward chain — are wired by
//! [`crate::op_wiring`], which `build_device_graph` drives to produce the
//! complete per-segment graphs.

use luminal::prelude::{DType, NodeIndex};

use skein_cost::Cluster;
use skein_ir::ir::{Graph, Param};
use skein_ir::plan::Plan;
use skein_ir::types::{Dim, Dtype};

use crate::error::EmitError;
use crate::shard_role::ShardRole;

/// Lowered per-device graph as a sequence of [`crate::segment::Segment`]s
/// plus a sequencing schedule that interleaves segment execution with
/// the collectives the runtime must issue. See `docs/lowering.md` for
/// the segmentation algorithm.
///
/// For `tp = ep = pp = 1` this collapses to one segment and zero
/// collectives — the whole forward pass in a single Luminal graph.
pub struct LoweredGraph {
    pub segments: Vec<crate::segment::Segment>,
    pub sequencing: Vec<crate::segment::SequenceStep>,
}

/// Metadata about a tensor that was declared on this device. `shape` holds
/// the sharded static dims; `dtype` is what `skein_emit` set on the
/// underlying `Input` op.
#[derive(Debug, Clone)]
pub struct DeclaredTensor {
    pub id: NodeIndex,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub role: ShardRole,
}

/// Build the per-device `LoweredGraph` for `device_idx`. Returns an
/// uncompiled `luminal::Graph` whose tensor declarations reflect the Plan's
/// sharding, plus a `declared` map keyed by HF safetensors name.
pub fn build_device_graph(
    plan: &Plan,
    cluster: &Cluster,
    ir: &Graph,
    device_idx: u32,
) -> Result<LoweredGraph, EmitError> {
    let (segments, sequencing) = crate::op_wiring::wire_segments(plan, cluster, ir, device_idx)?;
    Ok(LoweredGraph {
        segments,
        sequencing,
    })
}

/// Apply the [`ShardRole`] to a parameter's shape, returning the resulting
/// static dims as `Vec<usize>`. Weight tensors are static by construction
/// in `skein_ir`; any symbolic dim here is an importer bug and panics.
pub fn shard_param_dims(param: &Param, role: &ShardRole) -> Result<Vec<usize>, EmitError> {
    let dims: Vec<usize> = param
        .shape
        .0
        .iter()
        .map(|d| match d {
            Dim::Fixed(v) | Dim::Tp(v) | Dim::Ep(v) => *v,
            Dim::Batch | Dim::Seq | Dim::KvLen => panic!(
                "weight tensor {} has symbolic dim {:?}; importer bug",
                param.name, d
            ),
        })
        .collect();

    match role {
        ShardRole::Replicated
        | ShardRole::ExpertOwned { .. }
        | ShardRole::NoParams
        | ShardRole::PipelineStageElsewhere
        | ShardRole::ExpertElsewhere { .. } => Ok(dims),
        ShardRole::TpOutputShard { group_size, .. } => {
            if dims.is_empty() || dims[0] % *group_size as usize != 0 {
                return Err(EmitError::ShapeNotDivisible {
                    param: param.name.clone(),
                    shape: dims,
                    axis: 0,
                    tp: *group_size,
                    role: "TpOutputShard",
                });
            }
            let mut out = dims;
            out[0] /= *group_size as usize;
            Ok(out)
        }
        ShardRole::TpInputShard { group_size, .. } => {
            if dims.len() < 2 || dims[1] % *group_size as usize != 0 {
                return Err(EmitError::ShapeNotDivisible {
                    param: param.name.clone(),
                    shape: dims,
                    axis: 1,
                    tp: *group_size,
                    role: "TpInputShard",
                });
            }
            let mut out = dims;
            out[1] /= *group_size as usize;
            Ok(out)
        }
    }
}

/// Translate `skein_ir::Dtype` to `luminal::DType`. Every Skein dtype maps
/// to exactly one Luminal dtype.
pub fn to_luminal_dtype(d: Dtype) -> DType {
    match d {
        Dtype::F32 => DType::F32,
        Dtype::Bf16 => DType::Bf16,
        Dtype::Fp16 => DType::F16,
        Dtype::Fp8E4m3 => DType::F8E4M3,
        Dtype::Fp8E5m2 => DType::F8E5M2,
        Dtype::Int8 => DType::I8,
        Dtype::Int4 => DType::I4,
    }
}
