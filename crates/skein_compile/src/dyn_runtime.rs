//! Type-erased runtime wrapper with logical-name tensor access.
//!
//! `ComputeRuntime` stays generic over the backend. Collectives and the
//! Prompt-2 topology executor need trait objects, so this module owns the
//! segment graph plus a stable `name -> NodeIndex` map and exposes a small
//! object-safe API.

use std::collections::{HashMap, HashSet};

use luminal::hlir::Input;
use luminal::prelude::{Graph, NodeIndex};

use crate::ComputeRuntime;

#[derive(Debug, thiserror::Error)]
pub enum DynRuntimeError {
    #[error("runtime tensor {0:?} is not known in this segment")]
    UnknownTensor(String),

    #[error("runtime input {0:?} needs i32 data, but received f32 data")]
    ExpectedI32(String),
}

/// Object-safe runtime API.
///
/// This trait is intentionally single-threaded. At the pinned
/// Luminal rev, `luminal::Graph` owns `dyn HLIROp` / `dyn CustomOp` values
/// that are neither `Send` nor `Sync`, so a wrapper that owns the graph
/// cannot soundly implement those auto traits without unsafe code.
pub trait DynRuntime {
    fn execute_segment(&mut self) -> Result<(), DynRuntimeError>;
    fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError>;
    fn set_tensor_by_name(&mut self, name: &str, data: Vec<f32>) -> Result<(), DynRuntimeError>;
    fn set_tensor_i32_by_name(&mut self, name: &str, data: Vec<i32>)
    -> Result<(), DynRuntimeError>;
}

pub struct DynRuntimeWrapper<R: ComputeRuntime> {
    inner: R,
    graph: Graph,
    name_to_node: HashMap<String, NodeIndex>,
    input_nodes: HashSet<NodeIndex>,
    staged_f32: HashMap<NodeIndex, Vec<f32>>,
    staged_i32: HashMap<NodeIndex, Vec<i32>>,
    external_f32: HashMap<NodeIndex, Vec<f32>>,
}

impl<R: ComputeRuntime> DynRuntimeWrapper<R> {
    pub fn new(inner: R, graph: Graph, name_to_node: HashMap<String, NodeIndex>) -> Self {
        let input_nodes = graph
            .node_indices()
            .filter(|node| {
                graph
                    .node_weight(*node)
                    .is_some_and(|op| op.as_any().is::<Input>())
            })
            .collect();
        Self {
            inner,
            graph,
            name_to_node,
            input_nodes,
            staged_f32: HashMap::new(),
            staged_i32: HashMap::new(),
            external_f32: HashMap::new(),
        }
    }

    pub fn inner(&self) -> &R {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    fn node_for(&self, name: &str) -> Result<NodeIndex, DynRuntimeError> {
        self.name_to_node
            .get(name)
            .copied()
            .ok_or_else(|| DynRuntimeError::UnknownTensor(name.to_string()))
    }
}

impl<R: ComputeRuntime> DynRuntime for DynRuntimeWrapper<R> {
    fn execute_segment(&mut self) -> Result<(), DynRuntimeError> {
        self.external_f32.clear();
        for (node, data) in self.staged_f32.clone() {
            self.inner.set_data_f32(node, data);
        }
        for (node, data) in self.staged_i32.clone() {
            self.inner.set_data_i32(node, data);
        }
        self.inner.execute(&self.graph);
        Ok(())
    }

    fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError> {
        let node = self.node_for(name)?;
        if let Some(data) = self.external_f32.get(&node) {
            return Ok(data.clone());
        }
        Ok(self.inner.get_data_f32(node))
    }

    fn set_tensor_by_name(&mut self, name: &str, data: Vec<f32>) -> Result<(), DynRuntimeError> {
        let node = self.node_for(name)?;
        if self.input_nodes.contains(&node) {
            if name == "input_tokens" {
                return Err(DynRuntimeError::ExpectedI32(name.to_string()));
            }
            self.staged_f32.insert(node, data);
        } else {
            self.external_f32.insert(node, data);
        }
        Ok(())
    }

    fn set_tensor_i32_by_name(
        &mut self,
        name: &str,
        data: Vec<i32>,
    ) -> Result<(), DynRuntimeError> {
        let node = self.node_for(name)?;
        if self.input_nodes.contains(&node) {
            self.staged_i32.insert(node, data);
        } else {
            self.external_f32
                .insert(node, data.into_iter().map(|v| v as f32).collect());
        }
        Ok(())
    }
}
