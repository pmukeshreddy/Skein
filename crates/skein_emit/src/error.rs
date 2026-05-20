//! `EmitError` — one `thiserror` enum across the graph builder, weight
//! slicer, and topology emitter.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    #[error("device index {idx} out of range; cluster has {total} devices")]
    DeviceOutOfRange { idx: u32, total: u32 },

    #[error(
        "parameter {param} has shape {shape:?} but TP role {role:?} requires \
         dimension {axis} to be divisible by tp={tp}; sharding would leave \
         a ragged remainder"
    )]
    ShapeNotDivisible {
        param: String,
        shape: Vec<usize>,
        axis: usize,
        tp: u32,
        role: &'static str,
    },

    #[error(
        "expert weight {param} routes to expert index {expert_idx}, which \
         exceeds num_experts={num_experts}"
    )]
    ExpertIndexOutOfRange {
        param: String,
        expert_idx: u32,
        num_experts: u32,
    },

    #[error(
        "param name {name} does not match any sharding pattern Skein \
         recognizes; extend `shard_role::param_kind_for` if this is a new \
         architecture"
    )]
    UnknownParamPattern { name: String },

    #[error("source safetensors directory {path} does not exist")]
    SourceMissing { path: PathBuf },

    #[error(
        "no checkpoint found in {path}: expected either a single-file \
         `weights.safetensors` or a `model.safetensors.index.json` indexing \
         multi-shard `*.safetensors` files"
    )]
    CheckpointNotFound { path: PathBuf },

    #[error("could not read source safetensors at {path}: {source}")]
    SafetensorsIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("safetensors parse error at {path}: {message}")]
    SafetensorsParse { path: PathBuf, message: String },

    #[error("safetensors write error at {path}: {message}")]
    SafetensorsWrite { path: PathBuf, message: String },

    #[error("cost-model error while lowering: {0}")]
    Cost(#[from] skein_cost::CostError),
}
