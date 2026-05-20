//! Reloadable Skein artifact schema.
//!
//! Live `luminal::Graph` values are not serialized. Each segment records
//! stable metadata plus an [`OpRecipe`] that can rebuild the live graph by
//! replaying `skein_emit` lowering from the original IR and cluster spec.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use skein_cost::Cluster;
use skein_emit::graph_builder::LoweredGraph;
use skein_emit::io_manifest::IoManifest;
use skein_emit::segment::{Segment, SequenceStep};
use skein_ir::cluster::ClusterSpec;
use skein_ir::ir::Graph;
use skein_ir::plan::Plan;
use skein_ir::types::Dtype;

use crate::CompileError;

/// Live segment type rebuilt from artifact recipes.
pub type LoweredSegment = Segment;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclaredTensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentMetadata {
    pub idx: usize,
    pub declared_tensors: Vec<DeclaredTensorMeta>,
    pub input_handoff_names: Vec<String>,
    pub output_handoff_names: Vec<String>,
    pub op_recipe: OpRecipe,
}

/// High-level recipe for rebuilding graph segments.
///
/// The artifact format intentionally avoids serializing raw Luminal ops. The
/// lowering implementation is deterministic from `(Plan, IR, ClusterSpec,
/// device_idx)`, so the recipe stores that source description and rebuilds
/// through `skein_emit::build_device_graph`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpRecipe {
    WireSegments {
        device_idx: u32,
        cluster_json: Vec<u8>,
        ir_json: Vec<u8>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub skein_commit: String,
    pub luminal_commit: String,
    pub hardware: String,
    pub timestamp: String,
}

impl ArtifactMetadata {
    pub fn new(
        skein_commit: impl Into<String>,
        hardware: impl Into<String>,
        timestamp: impl Into<String>,
    ) -> Self {
        Self {
            skein_commit: skein_commit.into(),
            luminal_commit: "e558ce68498df842107ccb32e316efff4801ccc1".to_string(),
            hardware: hardware.into(),
            timestamp: timestamp.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SkeinArtifact {
    pub root: PathBuf,
    pub plan: Plan,
    pub sequencing: Vec<SequenceStep>,
    pub metadata: ArtifactMetadata,
    pub devices: Vec<DeviceArtifactLoaded>,
}

#[derive(Debug, Clone)]
pub struct DeviceArtifactLoaded {
    pub device_idx: usize,
    pub segments: Vec<SegmentMetadata>,
    pub weights_path: PathBuf,
    pub io_manifest: IoManifest,
    plan: Plan,
}

impl DeviceArtifactLoaded {
    /// Materialize live Luminal graphs from the serialized recipe.
    pub fn rebuild_graphs(&self) -> Result<Vec<LoweredSegment>, CompileError> {
        let recipe = self
            .segments
            .first()
            .ok_or(CompileError::ArtifactHasNoDevices)?
            .op_recipe
            .clone();
        match recipe {
            OpRecipe::WireSegments {
                device_idx,
                cluster_json,
                ir_json,
            } => {
                let cluster: ClusterSpec =
                    serde_json::from_slice(&cluster_json).map_err(|source| {
                        CompileError::RecipeJson {
                            what: "cluster",
                            source,
                        }
                    })?;
                let ir: Graph = serde_json::from_slice(&ir_json)
                    .map_err(|source| CompileError::RecipeJson { what: "ir", source })?;
                let cluster = Cluster::from_spec(cluster);
                let lowered =
                    skein_emit::build_device_graph(&self.plan, &cluster, &ir, device_idx)?;
                Ok(lowered.segments)
            }
        }
    }
}

impl SkeinArtifact {
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        plan: &Plan,
        cluster: &ClusterSpec,
        ir: &Graph,
        lowered_per_device: &[skein_emit::DeviceArtifact],
        weights: &[PathBuf],
        metadata: &ArtifactMetadata,
        out_dir: &Path,
    ) -> Result<Self, CompileError> {
        if lowered_per_device.is_empty() {
            return Err(CompileError::ArtifactHasNoDevices);
        }
        if lowered_per_device.len() != weights.len() {
            return Err(CompileError::ArtifactDeviceCountMismatch {
                devices: lowered_per_device.len(),
                weights: weights.len(),
            });
        }

        fs::create_dir_all(out_dir).map_err(|source| CompileError::Io {
            path: out_dir.to_path_buf(),
            source,
        })?;

        let plan_path = out_dir.join("plan.json");
        write_json(&plan_path, plan)?;

        let sequencing: Vec<SequenceStep> = lowered_per_device
            .iter()
            .flat_map(|d| d.graph.sequencing.iter().cloned())
            .collect();
        let topology_path = out_dir.join("topology.json");
        write_json(&topology_path, &sequencing)?;

        let metadata_path = out_dir.join("metadata.json");
        write_json(&metadata_path, metadata)?;

        for (artifact, weight_path) in lowered_per_device.iter().zip(weights.iter()) {
            if !weight_path.exists() {
                return Err(CompileError::ArtifactMissingFile {
                    path: weight_path.clone(),
                    file: "weights.safetensors",
                });
            }
            let device_dir = out_dir.join(format!("device_{}", artifact.device_idx));
            fs::create_dir_all(&device_dir).map_err(|source| CompileError::Io {
                path: device_dir.clone(),
                source,
            })?;

            let recipe = OpRecipe::WireSegments {
                device_idx: artifact.device_idx,
                cluster_json: serde_json::to_vec(cluster).map_err(|source| {
                    CompileError::RecipeJson {
                        what: "cluster",
                        source,
                    }
                })?,
                ir_json: serde_json::to_vec(ir)
                    .map_err(|source| CompileError::RecipeJson { what: "ir", source })?,
            };
            let segments = segment_metadata(&artifact.graph, recipe);
            let segment_path = device_dir.join("graph_segments.bin");
            let bytes =
                bincode::serialize(&segments).map_err(|source| CompileError::BincodeWrite {
                    path: segment_path.clone(),
                    source,
                })?;
            fs::write(&segment_path, bytes).map_err(|source| CompileError::Io {
                path: segment_path,
                source,
            })?;

            write_json(&device_dir.join("io.json"), &artifact.io_manifest)?;
            fs::copy(weight_path, device_dir.join("weights.safetensors")).map_err(|source| {
                CompileError::Io {
                    path: weight_path.clone(),
                    source,
                }
            })?;
        }

        Self::load(out_dir)
    }

    /// Reconstruct the source IR graph stored in the per-device recipe. The
    /// HF parity gate (`skein verify --hf-reference`) needs it to drive the
    /// `transformers` reference forward against the same architecture.
    pub fn ir(&self) -> Result<Graph, CompileError> {
        let device = self.devices.first().ok_or(CompileError::ArtifactHasNoDevices)?;
        let seg = device
            .segments
            .first()
            .ok_or(CompileError::ArtifactHasNoDevices)?;
        match &seg.op_recipe {
            OpRecipe::WireSegments { ir_json, .. } => serde_json::from_slice(ir_json)
                .map_err(|source| CompileError::RecipeJson { what: "ir", source }),
        }
    }

    pub fn load(dir: &Path) -> Result<Self, CompileError> {
        let plan: Plan = read_json(&dir.join("plan.json"))?;
        let sequencing: Vec<SequenceStep> = read_json(&dir.join("topology.json"))?;
        let metadata: ArtifactMetadata = read_json(&dir.join("metadata.json"))?;

        let mut devices = Vec::new();
        for entry in fs::read_dir(dir).map_err(|source| CompileError::Io {
            path: dir.to_path_buf(),
            source,
        })? {
            let entry = entry.map_err(|source| CompileError::Io {
                path: dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(idx) = name.strip_prefix("device_") else {
                continue;
            };
            let Ok(device_idx) = idx.parse::<usize>() else {
                continue;
            };

            let segment_path = path.join("graph_segments.bin");
            let bytes = fs::read(&segment_path).map_err(|source| CompileError::Io {
                path: segment_path.clone(),
                source,
            })?;
            let segments: Vec<SegmentMetadata> =
                bincode::deserialize(&bytes).map_err(|source| CompileError::BincodeRead {
                    path: segment_path,
                    source,
                })?;
            let io_manifest: IoManifest = read_json(&path.join("io.json"))?;
            let weights_path = path.join("weights.safetensors");
            if !weights_path.exists() {
                return Err(CompileError::ArtifactMissingFile {
                    path,
                    file: "weights.safetensors",
                });
            }
            devices.push(DeviceArtifactLoaded {
                device_idx,
                segments,
                weights_path,
                io_manifest,
                plan: plan.clone(),
            });
        }
        devices.sort_by_key(|d| d.device_idx);
        if devices.is_empty() {
            return Err(CompileError::ArtifactHasNoDevices);
        }

        Ok(Self {
            root: dir.to_path_buf(),
            plan,
            sequencing,
            metadata,
            devices,
        })
    }
}

fn segment_metadata(lowered: &LoweredGraph, recipe: OpRecipe) -> Vec<SegmentMetadata> {
    lowered
        .segments
        .iter()
        .map(|segment| {
            let mut declared_tensors: Vec<DeclaredTensorMeta> = segment
                .declared
                .iter()
                .map(|(name, declared)| DeclaredTensorMeta {
                    name: name.clone(),
                    shape: declared.shape.clone(),
                    dtype: declared.dtype,
                })
                .collect();
            declared_tensors.sort_by(|a, b| a.name.cmp(&b.name));

            SegmentMetadata {
                idx: segment.idx,
                declared_tensors,
                input_handoff_names: segment
                    .input_handoff
                    .iter()
                    .map(|h| h.logical_name.clone())
                    .collect(),
                output_handoff_names: segment
                    .output_handoff
                    .iter()
                    .map(|h| h.logical_name.clone())
                    .collect(),
                op_recipe: recipe.clone(),
            }
        })
        .collect()
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), CompileError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|source| CompileError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    fs::write(path, bytes).map_err(|source| CompileError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, CompileError> {
    let bytes = fs::read(path).map_err(|source| CompileError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| CompileError::Json {
        path: path.to_path_buf(),
        source,
    })
}
