//! Per-device weight slicing.
//!
//! `build_weight_shard` walks the IR and emits one [`WeightSlice`] per
//! parameter the device owns. Each slice carries enough metadata to read
//! the right bytes from the source safetensors checkpoint:
//!
//! - `source_key` and `source_shape` — what to look up in the source file's
//!   metadata.
//! - `dest_shape` — the shape after sharding (matches what the graph
//!   builder declared).
//! - `strategy` — the access pattern. `Whole` is a contiguous source range;
//!   `OuterAxisSlice` is contiguous; `InnerAxisSlice` is row-strided;
//!   `ExpertOwned` selects rows of a stacked-experts tensor.
//!
//! `write_weight_shard` materializes the shard to a destination
//! safetensors file. The synthetic-fixture test exercises this; the real
//! Mixtral path requires the source files to be on disk and returns an
//! explicit Phase-B-only error otherwise.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use skein_cost::Cluster;
use skein_ir::ir::Graph;
use skein_ir::plan::Plan;
use skein_ir::types::Dtype;

use crate::error::EmitError;
use crate::graph_builder::shard_param_dims;
use crate::shard_role::{ShardRole, shard_role_for_param};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightShard {
    pub device_idx: u32,
    pub slices: Vec<WeightSlice>,
    /// Total destination bytes across every slice. Useful sanity check —
    /// the sum of all devices' totals should equal the source's decoder
    /// weight bytes (after dtype changes).
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightSlice {
    pub source_key: String,
    pub source_shape: Vec<usize>,
    pub source_dtype: Dtype,
    pub dest_shape: Vec<usize>,
    pub strategy: ShardStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ShardStrategy {
    /// Full tensor — single contiguous source byte range.
    Whole,
    /// Outer-axis slice. Rows `[start, end)` of `[out, in, ...]`.
    /// Bytes are contiguous in row-major layout.
    OuterAxisSlice { start: u64, end: u64 },
    /// Inner-axis slice. Columns `[start, end)` of `[out, in]`.
    /// Bytes are *not* contiguous — every row contributes a stride-spaced
    /// chunk of `(end - start) * dtype_bytes`.
    InnerAxisSlice { start: u64, end: u64 },
    /// MoE expert slice. Treats the source as a stacked
    /// `[num_experts, ...]` tensor — Mixtral stores experts as separate
    /// keys (`experts.0.w1`, `experts.1.w1`, ...), so for Mixtral the
    /// strategy degenerates to `Whole` on each per-expert key. Kept as a
    /// dedicated variant for future architectures that pack experts into
    /// one tensor.
    ExpertOwned { expert_idx: u64 },
}

impl ShardStrategy {
    /// Source-side byte ranges this strategy reads. Single range for
    /// contiguous cases; multiple for `InnerAxisSlice`.
    ///
    /// Clippy's `single_range_in_vec_init` lint would have us collect the
    /// contiguous branches into a `(start..end).collect()` — that returns
    /// a `Vec<u64>`, not a `Vec<Range<u64>>`, which is what callers need.
    /// We *do* want a one-element `Vec<Range<u64>>`.
    #[allow(clippy::single_range_in_vec_init)]
    pub fn source_byte_ranges(
        self,
        source_shape: &[usize],
        source_dtype: Dtype,
    ) -> Vec<Range<u64>> {
        let elem_bytes = (source_dtype.bits() as u64).div_ceil(8);
        match self {
            ShardStrategy::Whole | ShardStrategy::ExpertOwned { .. } => {
                let total: u64 = source_shape.iter().map(|d| *d as u64).product();
                vec![0..total * elem_bytes]
            }
            ShardStrategy::OuterAxisSlice { start, end } => {
                let row_elems: u64 = source_shape.iter().skip(1).map(|d| *d as u64).product();
                vec![(start * row_elems * elem_bytes)..(end * row_elems * elem_bytes)]
            }
            ShardStrategy::InnerAxisSlice { start, end } => {
                // 2D case (Mixtral weight tensors are all 2D). Reject higher
                // ranks loudly — callers shouldn't reach this branch for
                // higher-rank tensors.
                assert!(
                    source_shape.len() == 2,
                    "InnerAxisSlice expects 2D source shape, got {:?}",
                    source_shape
                );
                let row_count = source_shape[0] as u64;
                let row_stride_bytes = source_shape[1] as u64 * elem_bytes;
                let slice_bytes = (end - start) * elem_bytes;
                let start_in_row = start * elem_bytes;
                (0..row_count)
                    .map(|r| {
                        let base = r * row_stride_bytes;
                        (base + start_in_row)..(base + start_in_row + slice_bytes)
                    })
                    .collect()
            }
        }
    }
}

/// Compute the strategy for a parameter at this device, given the resolved
/// `ShardRole` and the parameter's source shape.
pub fn strategy_for_role(role: &ShardRole, source_shape: &[usize]) -> ShardStrategy {
    match role {
        ShardRole::Replicated => ShardStrategy::Whole,
        ShardRole::TpOutputShard {
            group_size,
            group_index,
        } => {
            let g = *group_size as u64;
            let total = source_shape[0] as u64;
            let per = total / g;
            ShardStrategy::OuterAxisSlice {
                start: *group_index as u64 * per,
                end: (*group_index as u64 + 1) * per,
            }
        }
        ShardRole::TpInputShard {
            group_size,
            group_index,
        } => {
            let g = *group_size as u64;
            let total = source_shape[1] as u64;
            let per = total / g;
            ShardStrategy::InnerAxisSlice {
                start: *group_index as u64 * per,
                end: (*group_index as u64 + 1) * per,
            }
        }
        ShardRole::ExpertOwned { expert_idx } => ShardStrategy::ExpertOwned {
            expert_idx: *expert_idx as u64,
        },
        ShardRole::ExpertElsewhere { .. }
        | ShardRole::PipelineStageElsewhere
        | ShardRole::NoParams => {
            // Callers must filter these out before calling here. Reaching
            // this branch is a logic bug.
            panic!("strategy_for_role called on a role this device doesn't own");
        }
    }
}

pub fn build_weight_shard(
    plan: &Plan,
    cluster: &Cluster,
    ir: &Graph,
    device_idx: u32,
    _source_weights_dir: &Path,
) -> Result<WeightShard, EmitError> {
    if device_idx >= cluster.num_devices() {
        return Err(EmitError::DeviceOutOfRange {
            idx: device_idx,
            total: cluster.num_devices(),
        });
    }

    let mut slices: Vec<WeightSlice> = Vec::new();
    let mut total_bytes: u64 = 0;

    for layer in &ir.layers {
        for param in &layer.params {
            let role = shard_role_for_param(plan, cluster, ir, device_idx, layer, param);
            match role {
                ShardRole::PipelineStageElsewhere
                | ShardRole::ExpertElsewhere { .. }
                | ShardRole::NoParams => continue,
                _ => {}
            }
            let source_shape: Vec<usize> = param
                .shape
                .0
                .iter()
                .map(|d| match d {
                    skein_ir::types::Dim::Fixed(v)
                    | skein_ir::types::Dim::Tp(v)
                    | skein_ir::types::Dim::Ep(v) => *v,
                    _ => panic!("weight tensor with symbolic dim: {:?}", param),
                })
                .collect();
            let strategy = strategy_for_role(&role, &source_shape);
            let dest_shape = shard_param_dims(param, &role)?;
            let dest_bytes: u64 = dest_shape.iter().map(|d| *d as u64).product::<u64>()
                * (param.dtype.bits() as u64).div_ceil(8);
            total_bytes = total_bytes.saturating_add(dest_bytes);
            slices.push(WeightSlice {
                source_key: param.name.clone(),
                source_shape,
                source_dtype: param.dtype,
                dest_shape,
                strategy,
            });
        }
    }

    Ok(WeightShard {
        device_idx,
        slices,
        total_bytes,
    })
}

/// Materialize a shard to a destination safetensors file.
///
/// `source_dir` is expected to contain `index.json` and one or more
/// `.safetensors` shards whose metadata lists the tensors referenced by
/// `shard.slices[*].source_key`. Phase A only supports the single-file
/// case (one `<name>.safetensors`); multi-shard checkpoints fall through
/// to a Phase B error.
pub fn write_weight_shard(
    shard: &WeightShard,
    source_dir: &Path,
    dest_path: &Path,
) -> Result<(), EmitError> {
    if !source_dir.exists() {
        return Err(EmitError::SourceMissing {
            path: source_dir.to_path_buf(),
        });
    }

    // Look for a single-file source `weights.safetensors`. Real Mixtral
    // checkpoints are sharded across several files and indexed by
    // `model.safetensors.index.json`; surface a Phase B error rather than
    // silently picking a sub-file.
    let single_file = source_dir.join("weights.safetensors");
    if !single_file.exists() {
        return Err(EmitError::WeightWriteRequiresFixture {
            path: source_dir.to_path_buf(),
        });
    }
    let bytes = std::fs::read(&single_file).map_err(|source| EmitError::SafetensorsIo {
        path: single_file.clone(),
        source,
    })?;
    let st =
        safetensors::SafeTensors::deserialize(&bytes).map_err(|e| EmitError::SafetensorsParse {
            path: single_file.clone(),
            message: format!("{e}"),
        })?;

    // Build the destination buffer. We read each slice's source bytes
    // (concatenating sub-ranges for `InnerAxisSlice`) into an owned
    // `Vec<u8>` per tensor and hand the lot to `safetensors::serialize`.
    let mut owned_buffers: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> =
        Vec::with_capacity(shard.slices.len());
    for slice in &shard.slices {
        let view = st
            .tensor(&slice.source_key)
            .map_err(|e| EmitError::SafetensorsParse {
                path: single_file.clone(),
                message: format!("missing key {}: {e}", slice.source_key),
            })?;
        let src_bytes = view.data();
        let ranges = slice
            .strategy
            .source_byte_ranges(&slice.source_shape, slice.source_dtype);
        let mut buf: Vec<u8> = Vec::new();
        for r in ranges {
            buf.extend_from_slice(&src_bytes[r.start as usize..r.end as usize]);
        }
        let dt = match slice.source_dtype {
            Dtype::Bf16 => safetensors::Dtype::BF16,
            Dtype::Fp16 => safetensors::Dtype::F16,
            // FP8 / Int4 variants don't have stable safetensors codes in
            // 0.4.x; store as raw bytes via the closest match. This is fine
            // for Phase A's synthetic fixture tests — production weights
            // are bf16/fp16 at write time, downcasting happens at compile.
            Dtype::Fp8E4m3 | Dtype::Fp8E5m2 => safetensors::Dtype::F8_E4M3,
            Dtype::Int8 => safetensors::Dtype::I8,
            Dtype::Int4 => safetensors::Dtype::I8,
        };
        owned_buffers.push((slice.source_key.clone(), dt, slice.dest_shape.clone(), buf));
    }

    let tensor_data: HashMap<String, (safetensors::Dtype, Vec<usize>, &[u8])> = owned_buffers
        .iter()
        .map(|(name, dt, shape, buf)| (name.clone(), (*dt, shape.clone(), buf.as_slice())))
        .collect();

    let serialized = safetensors::serialize(
        tensor_data.iter().map(|(k, (dt, shape, buf))| {
            (
                k.as_str().to_string(),
                safetensors::tensor::TensorView::new(*dt, shape.clone(), buf).expect(
                    "tensor view: shape and buffer were just computed by build_weight_shard; \
                     a mismatch is an internal bug",
                ),
            )
        }),
        &None,
    )
    .map_err(|e| EmitError::SafetensorsWrite {
        path: dest_path.to_path_buf(),
        message: format!("{e}"),
    })?;

    std::fs::write(dest_path, &serialized).map_err(|source| EmitError::SafetensorsIo {
        path: dest_path.to_path_buf(),
        source,
    })?;
    let _ = PathBuf::from(dest_path); // silence unused-import in Path
    Ok(())
}
