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

    #[error(
        "weight {tensor:?} byte length {len} is not a multiple of the \
         {dtype:?} element size"
    )]
    WeightByteLength {
        tensor: String,
        dtype: WeightDtype,
        len: usize,
    },
}

/// Source precision of a weight tensor's raw safetensors bytes. These are the
/// dtypes a checkpoint stores weights in; the in-graph compute dtype is the
/// Plan's choice and is handled separately by the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    F32,
    F16,
    Bf16,
}

impl WeightDtype {
    /// Bytes per element of the on-disk representation.
    pub fn byte_width(self) -> usize {
        match self {
            WeightDtype::F32 => 4,
            WeightDtype::F16 | WeightDtype::Bf16 => 2,
        }
    }
}

/// Decode raw little-endian safetensors weight bytes into `f32` values. This
/// is the byte-level loading path shared by the runtime: the CPU
/// `NativeRuntime` works in `f32`, so weights are widened here. (A GPU
/// runtime can override [`DynRuntime::set_tensor_bytes_by_name`] to upload the
/// raw low-precision bytes to device memory instead — GPU-only.)
pub fn decode_weight_bytes(
    tensor: &str,
    bytes: &[u8],
    dtype: WeightDtype,
) -> Result<Vec<f32>, DynRuntimeError> {
    if bytes.len() % dtype.byte_width() != 0 {
        return Err(DynRuntimeError::WeightByteLength {
            tensor: tensor.to_string(),
            dtype,
            len: bytes.len(),
        });
    }
    let values = match dtype {
        WeightDtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        WeightDtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        WeightDtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    };
    Ok(values)
}

/// IEEE-754 half-precision bit pattern → `f32`.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = match exp {
        0 if mant == 0 => sign, // signed zero
        0 => {
            // Subnormal: normalize into f32's exponent range.
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let exp32 = (127 - 15 + 1 + e) as u32;
            sign | (exp32 << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mant << 13), // inf / nan
        _ => sign | ((exp + (127 - 15)) << 23) | (mant << 13),
    };
    f32::from_bits(f32_bits)
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

    /// Load a weight tensor from its raw little-endian safetensors bytes.
    /// The default decodes to `f32` (correct for the CPU runtime); a GPU
    /// runtime can override this to upload the low-precision bytes directly.
    fn set_tensor_bytes_by_name(
        &mut self,
        name: &str,
        bytes: &[u8],
        dtype: WeightDtype,
    ) -> Result<(), DynRuntimeError> {
        let data = decode_weight_bytes(name, bytes, dtype)?;
        self.set_tensor_by_name(name, data)
    }
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
