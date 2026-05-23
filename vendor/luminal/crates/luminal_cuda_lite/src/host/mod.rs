use std::{fmt::Debug, sync::Arc};

use crate::cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DriverError, result};
use luminal::{op::EgglogOp, prelude::*};
mod cublas;
mod cublaslt;
pub mod flashinfer;
pub mod moe;

/// True iff SKEIN_CAPTURE is set (full-step CUDA-graph capture). Cached.
pub fn is_capture() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var_os("SKEIN_CAPTURE").is_some())
}

/// Persistent per-`key` scratch device buffer for the SKEIN_CAPTURE full-step
/// graph. A normal per-call `stream.alloc` is freed at scope end, so a captured
/// kernel that scratched into it would, on every graph replay, read/write a
/// freed (and since-reused) address — producing garbage. This keeps one buffer
/// per `key` alive (grown monotonically) and returns its stable device pointer,
/// reused every step. Safe because all captured kernels run serialized on the
/// single shared capture stream, so one buffer per key is never touched by two
/// concurrent kernels. Distinct `key`s are required for buffers that must coexist
/// within one op (e.g. a GEMV's input vs its output).
pub fn capture_scratch(stream: &Arc<CudaStream>, key: u32, bytes: usize) -> u64 {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static POOL: OnceLock<Mutex<HashMap<u32, CudaSlice<u8>>>> = OnceLock::new();
    let pool = POOL.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = pool.lock().unwrap();
    let need = match g.get(&key) {
        Some(b) => b.len() < bytes,
        None => true,
    };
    if need {
        let buf = unsafe { stream.alloc::<u8>(bytes.max(1)).unwrap() };
        g.insert(key, buf);
    }
    let buf = g.get(&key).unwrap();
    let (ptr, _guard) = buf.device_ptr(stream);
    ptr
}

/// Persistent device buffer holding a single `1.0f32` (the unit tensorwide scale),
/// for SKEIN_CAPTURE: a per-call `clone_htod(&[1.0])` is freed at scope end, so a
/// captured fp8 matmul would read a stale scale pointer on replay. Allocated once.
pub fn capture_unit_scale(stream: &Arc<CudaStream>) -> u64 {
    use std::sync::{Mutex, OnceLock};
    static S: OnceLock<Mutex<Option<CudaSlice<u8>>>> = OnceLock::new();
    let m = S.get_or_init(|| Mutex::new(None));
    let mut g = m.lock().unwrap();
    if g.is_none() {
        let bytes = unsafe { std::slice::from_raw_parts([1.0f32].as_ptr() as *const u8, 4) };
        *g = Some(stream.clone_htod(bytes).unwrap());
    }
    let (ptr, _guard) = g.as_ref().unwrap().device_ptr(stream);
    ptr
}

pub type Ops = (
    // cublas::CuBlasSgemmV2,
    cublaslt::CuBlasLt,
    cublaslt::CuBlasLtScaled,
    moe::GLUMoE,
    flashinfer::FlashInferAttention,
);

#[cfg(test)]
pub(crate) type CublasLtTypeTuple = (
    luminal::dtype::DType,
    luminal::dtype::DType,
    luminal::dtype::DType,
    luminal::dtype::DType,
    &'static str,
    luminal::dtype::DType,
);

#[cfg(test)]
pub(crate) fn cublaslt_type_tuple(op: &dyn HostOp) -> Option<CublasLtTypeTuple> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::type_tuple)
}

#[cfg(test)]
pub(crate) type CublasLtScaleValues = (f64, f64);

#[cfg(test)]
pub(crate) fn cublaslt_scale_values(op: &dyn HostOp) -> Option<CublasLtScaleValues> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::scale_values)
}

#[cfg(test)]
pub(crate) fn cublaslt_epilogue(op: &dyn HostOp) -> Option<&'static str> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::epilogue)
}

#[cfg(test)]
pub(crate) type CublasLtMatrixOrders = (&'static str, &'static str, &'static str, &'static str);

#[cfg(test)]
pub(crate) fn cublaslt_matrix_orders(op: &dyn HostOp) -> Option<CublasLtMatrixOrders> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::matrix_orders)
}

#[cfg(test)]
pub(crate) type CublasLtTransposeOps = (&'static str, &'static str);

#[cfg(test)]
pub(crate) fn cublaslt_transpose_ops(op: &dyn HostOp) -> Option<CublasLtTransposeOps> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::transpose_ops)
}

#[cfg(test)]
pub(crate) fn cublaslt_c_d_layouts_match(op: &dyn HostOp) -> Option<bool> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::c_d_layouts_match)
}

#[cfg(test)]
pub(crate) type CublasLtTensorScaleInputs = (bool, bool);

#[cfg(test)]
pub(crate) fn cublaslt_tensor_scale_inputs(op: &dyn HostOp) -> Option<CublasLtTensorScaleInputs> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::tensor_scale_inputs)
}

/// Non-owning device buffer handle used by host operations.
///
/// Runtime-owned intermediates may be a whole `CudaSlice`, a subregion inside
/// the reusable arena, or an external pointer. Host ops only need the pointer
/// and the logical byte length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffer {
    ptr: u64,
    len: usize,
}

impl DeviceBuffer {
    pub fn new(ptr: u64, len: usize) -> Self {
        Self { ptr, len }
    }

    pub fn ptr(self) -> u64 {
        self.ptr
    }

    pub fn len(self) -> usize {
        self.len
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn clone_dtoh(self, stream: &Arc<CudaStream>) -> Result<Vec<u8>, DriverError> {
        let mut host = vec![0u8; self.len];
        unsafe {
            result::memcpy_dtoh_async(&mut host, self.ptr, stream.cu_stream())?;
        }
        stream.synchronize()?;
        Ok(host)
    }
}

/// Host operations that execute on the CPU but orchestrate GPU work.
///
/// This includes operations like cuBLAS calls and CUDA graph executions.
pub trait HostOp: Debug + as_any::AsAny + EgglogOp {
    /// Execute the operation with access to buffers via a map.
    ///
    /// # Arguments
    /// * `stream` - The CUDA stream to execute on
    /// * `self_node` - The NodeIndex of this op in the llir_graph (used as output buffer)
    /// * `inputs` - NodeIndices of input nodes (in edge order from the graph)
    /// * `buffers` - Map from NodeIndex to device buffer for all allocated nodes
    /// * `dyn_map` - Dynamic dimension values
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()>;

    /// Returns the output buffer size in elements.
    /// Return 0 if this op doesn't have a single output buffer (e.g., CudaGraphOp).
    fn output_size(&self) -> Expression;

    /// Returns the output buffer size in bytes (accounts for dtype).
    fn output_bytes(&self) -> Expression;

    /// Dtype of this op's output buffer. The runtime records this in
    /// `buffer_specs` so the host read-back (`get_f32`) widens half-precision
    /// outputs correctly instead of byte-reinterpreting them (which halves the
    /// element count). Default `F32`; ops that emit bf16/f16 (e.g. cublasLt with
    /// a bf16 `d_dtype`) must override this.
    fn output_dtype(&self) -> DType {
        DType::F32
    }

    /// Returns additional nodes (beyond graph edges) that this op needs buffers for.
    ///
    /// For most ops, this returns empty (buffers determined by graph edges).
    /// For CudaGraphOp, this returns all internal kernel nodes.
    fn extra_buffer_nodes(&self) -> Vec<NodeIndex> {
        vec![]
    }

    /// Returns relative lifetimes for extra buffer nodes within this host op.
    ///
    /// The tuple is `(node, first_step, last_step)`, where steps are local to
    /// this host op's execution. Returning `None` tells the runtime to treat
    /// every extra buffer as live for the whole host op.
    fn extra_buffer_lifetimes(&self) -> Option<Vec<(NodeIndex, usize, usize)>> {
        None
    }

    /// Returns buffer size requirements for extra nodes (node -> size in elements).
    ///
    /// Called during buffer allocation to ensure all required buffers exist.
    /// For CudaGraphOp, this returns sizes for all internal kernel output buffers.
    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        FxHashMap::default()
    }

    /// Returns the name of this host op for stats reporting, or None if not reportable.
    fn stats_name(&self) -> Option<&'static str> {
        None
    }
}
