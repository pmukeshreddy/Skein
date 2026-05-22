//! Type-erased runtime wrapper with logical-name tensor access.
//!
//! `ComputeRuntime` stays generic over the backend. Collectives and the
//! Prompt-2 topology executor need trait objects, so this module owns the
//! segment graph plus a stable `name -> NodeIndex` map and exposes a small
//! object-safe API.

use std::collections::{HashMap, HashSet};

use luminal::hlir::Input;
use luminal::prelude::{DType, Graph, NodeIndex};

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

/// A handoff tensor's index in the rank-local handoff store. Interned once from
/// the tensor's logical name at [`SegmentRunner`](../../skein_runtime) construction,
/// then used to dispatch the per-segment input/output loop by `Vec` index instead
/// of by `HashMap<String, _>` lookup on the decode hot path.
///
/// Defined here (not in `skein_runtime`) because the [`DynRuntime`] trait's
/// `_by_id` methods take it, and `skein_compile` cannot depend on `skein_runtime`
/// (the dependency runs the other way). `skein_runtime::distributed::segment_runner`
/// re-exports it.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct HandoffId(pub u32);

impl HandoffId {
    /// The store index this id selects.
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

/// Object-safe runtime API.
///
/// This trait is intentionally single-threaded. At the pinned
/// Luminal rev, `luminal::Graph` owns `dyn HLIROp` / `dyn CustomOp` values
/// that are neither `Send` nor `Sync`, so a wrapper that owns the graph
/// cannot soundly implement those auto traits without unsafe code.
///
/// ## Name vs id access
///
/// The `_by_name` methods are the setup / parity / CPU-mock path
/// (`materialize_weights`, `TopologyExecutor`, `begin_request`). The `_by_id`
/// methods are the decode hot path: a [`SegmentRunner`](../../skein_runtime)
/// pre-resolves every segment's input/output logical names to [`HandoffId`]s once
/// (via [`DynRuntime::register_handoff_ids`]) and then dispatches with no string
/// hashing per segment per token. Both address the same underlying graph nodes;
/// the id path just skips the name lookup.
pub trait DynRuntime {
    fn execute_segment(&mut self) -> Result<(), DynRuntimeError>;
    fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError>;
    fn set_tensor_by_name(&mut self, name: &str, data: Vec<f32>) -> Result<(), DynRuntimeError>;
    fn set_tensor_i32_by_name(&mut self, name: &str, data: Vec<i32>)
    -> Result<(), DynRuntimeError>;

    /// Build this runtime's `HandoffId -> NodeIndex` table from its own
    /// `name -> node` maps, using the runner's global `name -> id` assignment
    /// (`id_for_name`). Called once per segment after construction so the hot
    /// path resolves ids without string lookups. Names this runtime doesn't know
    /// (other segments' handoffs) are skipped; ids never registered here are
    /// treated as "not in this segment" by the `_by_id` methods. Default: no-op
    /// (name-only runtimes that never see the id hot path).
    fn register_handoff_ids(&mut self, _id_for_name: &dyn Fn(&str) -> Option<HandoffId>) {}

    /// Stage an f32 input by id. Default errors: only runtimes on the decode hot
    /// path (the graph wrapper + the segment-runner mocks) implement id access;
    /// name-only runtimes are never called this way.
    fn set_tensor_by_id(&mut self, id: HandoffId, _data: Vec<f32>) -> Result<(), DynRuntimeError> {
        Err(DynRuntimeError::UnknownTensor(format!("id {}", id.0)))
    }
    /// Stage an i32 input by id. See [`DynRuntime::set_tensor_by_id`].
    fn set_tensor_i32_by_id(
        &mut self,
        id: HandoffId,
        _data: Vec<i32>,
    ) -> Result<(), DynRuntimeError> {
        Err(DynRuntimeError::UnknownTensor(format!("id {}", id.0)))
    }
    /// Read an output by id. See [`DynRuntime::set_tensor_by_id`].
    fn get_tensor_by_id(&self, id: HandoffId) -> Result<Vec<f32>, DynRuntimeError> {
        Err(DynRuntimeError::UnknownTensor(format!("id {}", id.0)))
    }

    /// Device buffer `(raw_ptr, byte_len)` of a named **output**, by id. CUDA
    /// only; `None` by default. See [`DynRuntime::output_device_ptr_by_name`].
    fn output_device_ptr_by_id(&self, _id: HandoffId) -> Option<(u64, usize)> {
        None
    }

    /// Device pointer of a resident weight buffer, by id. CUDA only; `None` by
    /// default. See [`DynRuntime::weight_device_ptr_by_name`].
    fn weight_device_ptr_by_id(&self, _id: HandoffId) -> Option<u64> {
        None
    }

    /// Ensure a named KV-cache **input** is backed by a persistent on-GPU buffer
    /// of `n_bytes`, by id, returning its device pointer. CUDA only; `None`.
    /// See [`DynRuntime::ensure_kv_input_device_by_name`].
    fn ensure_kv_input_device_by_id(&mut self, _id: HandoffId, _n_bytes: usize) -> Option<u64> {
        None
    }

    /// Bind a named input to an external device buffer (a producer segment's
    /// output), by id, transiently. CUDA only; no-op default.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of at least `n_bytes` on this
    /// runtime's device, alive until this segment finishes executing.
    unsafe fn bind_input_device_by_id(&mut self, _id: HandoffId, _ptr: u64, _n_bytes: usize) {}

    /// Bind a named input to an external device buffer **persistently** (bound
    /// once, kept across forwards — the buffer's *contents* may still be
    /// overwritten between forwards). Unlike [`bind_input_device_by_id`] (which
    /// is re-bound each step), this marks the input persistent so it is never
    /// consumed. Used to point the MoE FFN gate-scalar inputs at fixed offsets of
    /// a single resident gate buffer once at bootstrap. CUDA only; no-op default.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of at least `n_bytes` on this
    /// runtime's device, kept alive for the runtime's lifetime.
    unsafe fn set_input_device_persistent_by_id(
        &mut self,
        _id: HandoffId,
        _ptr: u64,
        _n_bytes: usize,
    ) {
    }

    /// Copy a named **output** to `dest_ptr` (device→device), by id. CUDA only.
    ///
    /// # Safety
    /// `dest_ptr` must be a valid device allocation of at least `n_bytes`.
    unsafe fn copy_output_to_device_by_id(&self, _id: HandoffId, _dest_ptr: u64, _n_bytes: usize) {}

    /// Set a dynamic-shape dimension (e.g. the cached-decode `past` length `'p'`)
    /// on the segment's graph before the next [`execute_segment`]. The default
    /// is a no-op (segments with only static shapes ignore it); the graph-owning
    /// wrapper overrides it to update the graph's dyn-dim map.
    fn set_dyn_dim(&mut self, _dim: char, _val: usize) {}

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

    /// Upload any staged weights now (as persistent inputs) instead of lazily on
    /// the first execute. Used so a decode graph's weights are GPU-resident before
    /// a sibling prefill graph shares them by pointer. Default: no-op.
    fn materialize_weights(&mut self) {}

    /// Free the intermediate-buffer arena (re-allocated lazily next execute);
    /// persistent weights untouched. Default: no-op.
    fn clear_intermediates(&mut self) {}

    /// Device pointer of a resident weight buffer, by tensor name (CUDA only).
    fn weight_device_ptr_by_name(&self, _name: &str) -> Option<u64> {
        None
    }

    /// Point a weight input at an external device buffer (shared weights), by
    /// tensor name. `n_bytes` is the buffer size. CUDA only; default no-op.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of `n_bytes` kept alive for this
    /// runtime's lifetime.
    unsafe fn set_weight_device_ptr_by_name(&mut self, _name: &str, _ptr: u64, _n_bytes: usize) {}

    /// Device buffer `(raw_ptr, byte_len)` backing a named **output** tensor, no
    /// host copy — for a device-resident segment-to-segment handoff. CUDA only;
    /// `None` by default (host path) or when the output isn't resident.
    fn output_device_ptr_by_name(&self, _name: &str) -> Option<(u64, usize)> {
        None
    }

    /// Bind a named input to an external device buffer (a producer segment's
    /// output), transiently (re-bound each step — not persistent). CUDA only;
    /// no-op default.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of at least `n_bytes` on this
    /// runtime's device, alive until this segment finishes executing.
    unsafe fn bind_input_device_by_name(&mut self, _name: &str, _ptr: u64, _n_bytes: usize) {}

    /// Ensure a named KV-cache **input** is backed by a persistent on-GPU buffer
    /// of `n_bytes` (allocated once, reused every step), returning its device
    /// pointer — so the cache is never re-uploaded from host. CUDA only; `None`.
    fn ensure_kv_input_device_by_name(&mut self, _name: &str, _n_bytes: usize) -> Option<u64> {
        None
    }

    /// Copy a named **output** (a decode step's new K/V) to `dest_ptr`
    /// (device→device), e.g. into its slot in the resident KV buffer. CUDA only.
    ///
    /// # Safety
    /// `dest_ptr` must be a valid device allocation of at least `n_bytes`.
    unsafe fn copy_output_to_device_by_name(&self, _name: &str, _dest_ptr: u64, _n_bytes: usize) {}
}

pub struct DynRuntimeWrapper<R: ComputeRuntime> {
    inner: R,
    graph: Graph,
    name_to_node: HashMap<String, NodeIndex>,
    input_name_to_node: HashMap<String, NodeIndex>,
    /// `HandoffId -> NodeIndex` for this segment's handoff tensors, built once by
    /// [`DynRuntime::register_handoff_ids`]. Indexed by `HandoffId.idx()`; `None`
    /// for ids belonging to other segments. Lets the decode hot path resolve a
    /// handoff to its graph node with a `Vec` index, no `name_to_node` hashing.
    id_to_node: Vec<Option<NodeIndex>>,
    input_nodes: HashSet<NodeIndex>,
    staged_f32: HashMap<NodeIndex, Vec<f32>>,
    staged_i32: HashMap<NodeIndex, Vec<i32>>,
    external_f32: HashMap<NodeIndex, Vec<f32>>,
    /// Weight tensors, staged once at load (set_tensor_bytes_by_name) and applied
    /// to the runtime as PERSISTENT inputs on the first execute, then dropped.
    /// Unlike `staged_f32` (per-step handoffs re-applied every forward), weights
    /// are uploaded exactly once — avoiding a ~90 GB re-upload per forward.
    staged_weights: HashMap<NodeIndex, Vec<f32>>,
    weights_loaded: bool,
}

impl<R: ComputeRuntime> DynRuntimeWrapper<R> {
    pub fn new(
        inner: R,
        graph: Graph,
        name_to_node: HashMap<String, NodeIndex>,
        input_name_to_node: HashMap<String, NodeIndex>,
    ) -> Self {
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
            input_name_to_node,
            id_to_node: Vec::new(),
            input_nodes,
            staged_f32: HashMap::new(),
            staged_i32: HashMap::new(),
            external_f32: HashMap::new(),
            staged_weights: HashMap::new(),
            weights_loaded: false,
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

    fn input_node_for(&self, name: &str) -> Result<NodeIndex, DynRuntimeError> {
        self.input_name_to_node
            .get(name)
            .copied()
            .or_else(|| self.name_to_node.get(name).copied())
            .ok_or_else(|| DynRuntimeError::UnknownTensor(name.to_string()))
    }

    /// Resolve a [`HandoffId`] to this segment's graph node via the table
    /// [`DynRuntime::register_handoff_ids`] built. `None` if the id is not one of
    /// this segment's handoffs (registration left the slot empty / out of range).
    #[inline]
    fn node_for_id(&self, id: HandoffId) -> Option<NodeIndex> {
        self.id_to_node.get(id.idx()).copied().flatten()
    }
}

impl<R: ComputeRuntime> DynRuntime for DynRuntimeWrapper<R> {
    fn execute_segment(&mut self) -> Result<(), DynRuntimeError> {
        self.external_f32.clear();
        // Weights: upload exactly once, as PERSISTENT inputs (kept across
        // forwards). This is the difference between re-uploading ~90 GB every
        // forward (~78s) and uploading it once.
        self.materialize_weights();
        // drain() (not clone()) moves the staged buffers into the runtime without
        // duplicating them, and keeps the map's capacity for the next forward.
        for (node, data) in self.staged_f32.drain() {
            // Narrow to the input's declared dtype (bf16/f16) so a bf16 input
            // slot receives bf16, not raw f32 bytes. input_meta carries the
            // graph's per-Input dtype; default f32 when absent.
            let dtype = self
                .graph
                .input_meta
                .get(&node)
                .map(|(_, dt)| *dt)
                .unwrap_or(DType::F32);
            self.inner.set_data_f32_as(node, data, dtype);
        }
        for (node, data) in self.staged_i32.drain() {
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
        let node = self.input_node_for(name)?;
        tracing::debug!(
            name,
            node = node.index(),
            is_input = self.input_nodes.contains(&node),
            len = data.len(),
            "set_tensor_by_name"
        );
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
        let node = self.input_node_for(name)?;
        if self.input_nodes.contains(&node) {
            self.staged_i32.insert(node, data);
        } else {
            self.external_f32
                .insert(node, data.into_iter().map(|v| v as f32).collect());
        }
        Ok(())
    }

    fn set_dyn_dim(&mut self, dim: char, val: usize) {
        self.graph.dyn_map.insert(dim, val);
    }

    fn set_tensor_bytes_by_name(
        &mut self,
        name: &str,
        bytes: &[u8],
        dtype: WeightDtype,
    ) -> Result<(), DynRuntimeError> {
        // Only weights are loaded via raw bytes (from safetensors); handoffs and
        // tokens use the f32/i32 setters. Stage weights separately so they are
        // uploaded ONCE as persistent inputs on the first execute, instead of
        // landing in staged_f32 and being re-uploaded every forward.
        let node = self.input_node_for(name)?;
        let data = decode_weight_bytes(name, bytes, dtype)?;
        self.staged_weights.insert(node, data);
        Ok(())
    }

    fn materialize_weights(&mut self) {
        if self.weights_loaded {
            return;
        }
        let weights = std::mem::take(&mut self.staged_weights);
        for (node, data) in weights {
            let dtype = self
                .graph
                .input_meta
                .get(&node)
                .map(|(_, dt)| *dt)
                .unwrap_or(DType::F32);
            self.inner.set_data_persistent_f32_as(node, data, dtype);
        }
        self.weights_loaded = true;
    }

    fn clear_intermediates(&mut self) {
        self.inner.clear_intermediates();
    }

    fn weight_device_ptr_by_name(&self, name: &str) -> Option<u64> {
        // Weights are declared tensors → name_to_node (not input_handoff).
        let node = self.name_to_node.get(name).copied()?;
        self.inner.input_device_ptr(node)
    }

    unsafe fn set_weight_device_ptr_by_name(&mut self, name: &str, ptr: u64, n_bytes: usize) {
        if let Some(node) = self.name_to_node.get(name).copied() {
            unsafe { self.inner.set_input_device_ptr(node, ptr, n_bytes) };
            // It is now resident via the shared pointer; don't also try to load
            // it from staged_weights on first execute.
            self.staged_weights.remove(&node);
            self.weights_loaded = true;
        }
    }

    fn output_device_ptr_by_name(&self, name: &str) -> Option<(u64, usize)> {
        let node = self.node_for(name).ok()?;
        self.inner.output_device_buffer(node)
    }

    unsafe fn bind_input_device_by_name(&mut self, name: &str, ptr: u64, n_bytes: usize) {
        if let Ok(node) = self.input_node_for(name) {
            unsafe { self.inner.bind_input_device_ptr(node, ptr, n_bytes) };
            // A device binding supersedes any host-staged value for this input;
            // drop it so execute() doesn't re-upload stale host bytes over it.
            self.staged_f32.remove(&node);
            self.external_f32.remove(&node);
        }
    }

    fn ensure_kv_input_device_by_name(&mut self, name: &str, n_bytes: usize) -> Option<u64> {
        let node = self.input_node_for(name).ok()?;
        // Never host-upload this input again; it lives on the GPU.
        self.staged_f32.remove(&node);
        self.external_f32.remove(&node);
        self.inner.alloc_persistent_input_zeros(node, n_bytes)
    }

    unsafe fn copy_output_to_device_by_name(&self, name: &str, dest_ptr: u64, n_bytes: usize) {
        if let Ok(node) = self.node_for(name) {
            unsafe {
                self.inner
                    .copy_output_to_device_ptr(node, dest_ptr, n_bytes)
            };
        }
    }

    fn register_handoff_ids(&mut self, id_for_name: &dyn Fn(&str) -> Option<HandoffId>) {
        // Size the table to the largest id any of this segment's names maps to.
        let mut max_idx = 0usize;
        let mut any = false;
        for name in self
            .input_name_to_node
            .keys()
            .chain(self.name_to_node.keys())
        {
            if let Some(id) = id_for_name(name) {
                max_idx = max_idx.max(id.idx());
                any = true;
            }
        }
        self.id_to_node = if any { vec![None; max_idx + 1] } else { Vec::new() };
        // Outputs (producing nodes) first, then inputs override: an Input op is
        // what `set_tensor`/`bind` must target, matching `input_node_for`'s
        // preference, while `get`/`output_device_ptr` read output handoffs whose
        // only entry is in `name_to_node`.
        for (name, &node) in &self.name_to_node {
            if let Some(id) = id_for_name(name) {
                self.id_to_node[id.idx()] = Some(node);
            }
        }
        for (name, &node) in &self.input_name_to_node {
            if let Some(id) = id_for_name(name) {
                self.id_to_node[id.idx()] = Some(node);
            }
        }
    }

    fn set_tensor_by_id(&mut self, id: HandoffId, data: Vec<f32>) -> Result<(), DynRuntimeError> {
        let node = self
            .node_for_id(id)
            .ok_or_else(|| DynRuntimeError::UnknownTensor(format!("id {}", id.0)))?;
        if self.input_nodes.contains(&node) {
            self.staged_f32.insert(node, data);
        } else {
            self.external_f32.insert(node, data);
        }
        Ok(())
    }

    fn set_tensor_i32_by_id(
        &mut self,
        id: HandoffId,
        data: Vec<i32>,
    ) -> Result<(), DynRuntimeError> {
        let node = self
            .node_for_id(id)
            .ok_or_else(|| DynRuntimeError::UnknownTensor(format!("id {}", id.0)))?;
        if self.input_nodes.contains(&node) {
            self.staged_i32.insert(node, data);
        } else {
            self.external_f32
                .insert(node, data.into_iter().map(|v| v as f32).collect());
        }
        Ok(())
    }

    fn get_tensor_by_id(&self, id: HandoffId) -> Result<Vec<f32>, DynRuntimeError> {
        let node = self
            .node_for_id(id)
            .ok_or_else(|| DynRuntimeError::UnknownTensor(format!("id {}", id.0)))?;
        if let Some(data) = self.external_f32.get(&node) {
            return Ok(data.clone());
        }
        Ok(self.inner.get_data_f32(node))
    }

    fn output_device_ptr_by_id(&self, id: HandoffId) -> Option<(u64, usize)> {
        let node = self.node_for_id(id)?;
        self.inner.output_device_buffer(node)
    }

    fn weight_device_ptr_by_id(&self, id: HandoffId) -> Option<u64> {
        let node = self.node_for_id(id)?;
        self.inner.input_device_ptr(node)
    }

    fn ensure_kv_input_device_by_id(&mut self, id: HandoffId, n_bytes: usize) -> Option<u64> {
        let node = self.node_for_id(id)?;
        self.staged_f32.remove(&node);
        self.external_f32.remove(&node);
        self.inner.alloc_persistent_input_zeros(node, n_bytes)
    }

    unsafe fn bind_input_device_by_id(&mut self, id: HandoffId, ptr: u64, n_bytes: usize) {
        if let Some(node) = self.node_for_id(id) {
            unsafe { self.inner.bind_input_device_ptr(node, ptr, n_bytes) };
            self.staged_f32.remove(&node);
            self.external_f32.remove(&node);
        }
    }

    unsafe fn set_input_device_persistent_by_id(&mut self, id: HandoffId, ptr: u64, n_bytes: usize) {
        if let Some(node) = self.node_for_id(id) {
            // Persistent (marks the node so its buffer is never consumed): the
            // gate buffer is bound once and lives for the runtime's lifetime.
            unsafe { self.inner.set_input_device_ptr(node, ptr, n_bytes) };
            self.staged_f32.remove(&node);
            self.external_f32.remove(&node);
        }
    }

    unsafe fn copy_output_to_device_by_id(&self, id: HandoffId, dest_ptr: u64, n_bytes: usize) {
        if let Some(node) = self.node_for_id(id) {
            unsafe {
                self.inner
                    .copy_output_to_device_ptr(node, dest_ptr, n_bytes)
            };
        }
    }
}
