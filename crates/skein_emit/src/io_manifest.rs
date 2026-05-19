//! Per-device IO manifest. Lists every tensor name + sharded shape + dtype
//! the runtime can expect to load. Written out as `io.json` alongside the
//! safetensors shard so the runtime can validate at load time.

use serde::{Deserialize, Serialize};

use skein_cost::Cluster;
use skein_ir::ir::Graph;
use skein_ir::plan::Plan;
use skein_ir::types::Dtype;

use crate::error::EmitError;
use crate::graph_builder::shard_param_dims;
use crate::shard_role::{ShardRole, shard_role_for_param};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoManifest {
    pub device_idx: u32,
    pub tensors: Vec<IoTensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub kind: IoTensorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IoTensorKind {
    /// Token-id input to the model.
    Input,
    /// Final logits output.
    Output,
    /// Weight tensor loaded from a safetensors shard.
    Weight,
    /// Collective payload entering the device (e.g. all-reduce input).
    CollectiveInput,
    /// Collective payload leaving the device.
    CollectiveOutput,
}

pub fn build_io_manifest(
    plan: &Plan,
    cluster: &Cluster,
    ir: &Graph,
    device_idx: u32,
) -> Result<IoManifest, EmitError> {
    if device_idx >= cluster.num_devices() {
        return Err(EmitError::DeviceOutOfRange {
            idx: device_idx,
            total: cluster.num_devices(),
        });
    }
    let mut tensors: Vec<IoTensor> = Vec::new();
    for layer in &ir.layers {
        for param in &layer.params {
            let role = shard_role_for_param(plan, cluster, ir, device_idx, layer, param);
            match role {
                ShardRole::PipelineStageElsewhere
                | ShardRole::ExpertElsewhere { .. }
                | ShardRole::NoParams => continue,
                _ => {}
            }
            let shape = shard_param_dims(param, &role)?;
            tensors.push(IoTensor {
                name: param.name.clone(),
                shape,
                dtype: param.dtype,
                kind: IoTensorKind::Weight,
            });
        }
    }
    Ok(IoManifest {
        device_idx,
        tensors,
    })
}
