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
//! safetensors file from a single-file source checkpoint; multi-shard
//! checkpoints return an explicit error (see TODO(multi-shard) there).

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
    // SKEIN_ATTN_FP8: the q/k/v/o attention weights come from an AutoFP8
    // checkpoint as E4M3 + per-tensor `weight_scale`/`input_scale` (bf16 [1])
    // siblings. Slice the weight as fp8 and copy the two scalar scales through.
    let attn_fp8 = std::env::var_os("SKEIN_ATTN_FP8").is_some();

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
            let is_attn_proj = attn_fp8
                && param.name.contains(".self_attn.")
                && param.name.ends_with("_proj.weight");
            let source_dtype = if is_attn_proj {
                Dtype::Fp8E4m3
            } else {
                param.dtype
            };
            let dest_bytes: u64 = dest_shape.iter().map(|d| *d as u64).product::<u64>()
                * (source_dtype.bits() as u64).div_ceil(8);
            total_bytes = total_bytes.saturating_add(dest_bytes);
            slices.push(WeightSlice {
                source_key: param.name.clone(),
                source_shape: source_shape.clone(),
                source_dtype,
                dest_shape: dest_shape.clone(),
                strategy,
            });
            if is_attn_proj {
                // Per-tensor scalar scales, replicated to every shard (the
                // per-tensor scale is identical regardless of TP row split).
                // AutoFP8 names scales `...q_proj.weight_scale` (the `.weight`
                // suffix is replaced, not appended), so strip it off the base.
                let base = param.name.strip_suffix(".weight").unwrap_or(&param.name);
                for suffix in [".weight_scale", ".input_scale"] {
                    slices.push(WeightSlice {
                        source_key: format!("{base}{suffix}"),
                        source_shape: vec![1],
                        source_dtype: Dtype::Bf16,
                        dest_shape: vec![1],
                        strategy: ShardStrategy::Whole,
                    });
                    total_bytes = total_bytes.saturating_add(2);
                }
            }
        }
    }

    Ok(WeightShard {
        device_idx,
        slices,
        total_bytes,
    })
}

/// `model.safetensors.index.json` schema (only the `weight_map` matters):
/// maps each tensor name to the `*.safetensors` shard file that holds it.
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Resolve each `source_key` to the checkpoint file that holds it, supporting
/// both a single-file `weights.safetensors` and a multi-shard checkpoint
/// indexed by `model.safetensors.index.json`.
fn resolve_source_files(
    shard: &WeightShard,
    source_dir: &Path,
) -> Result<HashMap<String, String>, EmitError> {
    if source_dir.join("weights.safetensors").exists() {
        return Ok(shard
            .slices
            .iter()
            .map(|s| (s.source_key.clone(), "weights.safetensors".to_string()))
            .collect());
    }
    let index_path = source_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let bytes = std::fs::read(&index_path).map_err(|source| EmitError::SafetensorsIo {
            path: index_path.clone(),
            source,
        })?;
        let index: SafetensorsIndex =
            serde_json::from_slice(&bytes).map_err(|e| EmitError::SafetensorsParse {
                path: index_path.clone(),
                message: format!("{e}"),
            })?;
        let mut map = HashMap::with_capacity(shard.slices.len());
        for slice in &shard.slices {
            let file = index.weight_map.get(&slice.source_key).ok_or_else(|| {
                EmitError::SafetensorsParse {
                    path: index_path.clone(),
                    message: format!("index has no entry for {}", slice.source_key),
                }
            })?;
            map.insert(slice.source_key.clone(), file.clone());
        }
        return Ok(map);
    }
    Err(EmitError::CheckpointNotFound {
        path: source_dir.to_path_buf(),
    })
}

fn safetensors_dtype(dtype: Dtype) -> safetensors::Dtype {
    match dtype {
        Dtype::F32 => safetensors::Dtype::F32,
        Dtype::Bf16 => safetensors::Dtype::BF16,
        Dtype::Fp16 => safetensors::Dtype::F16,
        // FP8 / Int4 variants don't have stable safetensors codes in 0.4.x;
        // store as raw bytes via the closest match. Source weights are
        // bf16/fp16 at write time — downcasting happens at compile.
        Dtype::Fp8E4m3 | Dtype::Fp8E5m2 => safetensors::Dtype::F8_E4M3,
        Dtype::Int8 => safetensors::Dtype::I8,
        Dtype::Int4 => safetensors::Dtype::I8,
    }
}

/// Materialize a shard to a destination safetensors file.
///
/// `source_dir` must contain either a single-file `weights.safetensors` or a
/// `model.safetensors.index.json` indexing multi-shard `*.safetensors` files
/// (the layout real HF checkpoints ship in). Each `shard.slices[*]` is read
/// from whichever file holds its `source_key`.
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

    let key_to_file = resolve_source_files(shard, source_dir)?;
    // Group slices by source file so each file is read + parsed exactly once.
    let mut slices_by_file: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, slice) in shard.slices.iter().enumerate() {
        slices_by_file
            .entry(key_to_file[&slice.source_key].clone())
            .or_default()
            .push(i);
    }

    // Build the destination buffer. We read each slice's source bytes
    // (concatenating sub-ranges for `InnerAxisSlice`) into an owned
    // `Vec<u8>` per tensor and hand the lot to `safetensors::serialize`.
    let mut owned_buffers: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> =
        Vec::with_capacity(shard.slices.len());
    for (file, slice_indices) in &slices_by_file {
        let path = source_dir.join(file);
        let bytes = std::fs::read(&path).map_err(|source| EmitError::SafetensorsIo {
            path: path.clone(),
            source,
        })?;
        let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| {
            EmitError::SafetensorsParse {
                path: path.clone(),
                message: format!("{e}"),
            }
        })?;
        for &i in slice_indices {
            let slice = &shard.slices[i];
            let view = st
                .tensor(&slice.source_key)
                .map_err(|e| EmitError::SafetensorsParse {
                    path: path.clone(),
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
            owned_buffers.push((
                slice.source_key.clone(),
                safetensors_dtype(slice.source_dtype),
                slice.dest_shape.clone(),
                buf,
            ));
        }
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
