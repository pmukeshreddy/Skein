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

/// SKEIN_GRAPH_KERNEL_TIMING: per-kernel GPU timing INSIDE the captured full-step
/// graph. A fresh `cuEventRecord` issued while the stream is capturing becomes an
/// event-record NODE in the graph (legal — unlike recording on a live stream); on
/// every replay those nodes fire, and `cuEventElapsedTime` between consecutive
/// nodes yields each kernel's true in-graph GPU span. This is the only profile
/// that reflects the production captured path (no launch-latency inflation). Only
/// meaningful under SKEIN_CAPTURE.
pub fn graph_kt_on() -> bool {
    static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *G.get_or_init(|| std::env::var_os("SKEIN_GRAPH_KERNEL_TIMING").is_some() && is_capture())
}

// Event handles stored as usize: the raw `sys::CUevent` (*mut) is !Send, so it
// can't live in a `static Mutex`. Cast back to CUevent at read time.
//
// DAG-aware design: each kernel gets its OWN before+after event pair, so a
// kernel's compute time (elapsed before_i -> after_i) is always read from two
// nodes that fire in the SAME replay — robust to the captured graph's side-stream
// all-reduce waits and to stale/orphaned nodes. The whole-graph span is read once
// (before[0] -> after[last]); STALL = span - sum(compute) is the inter-kernel wait
// (cross-GPU all-reduce / sync), reported as its own bucket instead of being
// (mis)dumped onto whatever kernel happens to precede a wait.
type GraphKtChain = (Vec<usize>, Vec<usize>, Vec<&'static str>); // before, after, name
fn graph_kt_chain() -> &'static std::sync::Mutex<GraphKtChain> {
    static C: std::sync::OnceLock<std::sync::Mutex<GraphKtChain>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new((Vec::new(), Vec::new(), Vec::new())))
}

/// Clear the chain at the start of a (re)capture. Old event handles leak — fine
/// for a diagnostic run.
pub fn graph_kt_reset() {
    if !graph_kt_on() {
        return;
    }
    let mut c = graph_kt_chain().lock().unwrap();
    c.0.clear();
    c.1.clear();
    c.2.clear();
}

/// During capture, just BEFORE launching kernel `name`: record its start event.
pub fn graph_kt_before(stream: &Arc<CudaStream>, name: &'static str) {
    if !graph_kt_on() {
        return;
    }
    let ctx = stream.context();
    if let Ok(ev) = crate::kernel::create_cuda_event(&ctx) {
        let _ = crate::kernel::record_event_on_stream(&ctx, ev, stream);
        let mut c = graph_kt_chain().lock().unwrap();
        c.0.push(ev as usize);
        c.2.push(name);
    }
}

/// During capture, just AFTER launching the current kernel: record its end event.
pub fn graph_kt_after(stream: &Arc<CudaStream>) {
    if !graph_kt_on() {
        return;
    }
    let ctx = stream.context();
    if let Ok(ev) = crate::kernel::create_cuda_event(&ctx) {
        let _ = crate::kernel::record_event_on_stream(&ctx, ev, stream);
        graph_kt_chain().lock().unwrap().1.push(ev as usize);
    }
}

/// After a replay (caller must have synced the stream): per-kernel compute =
/// elapsed(before_i, after_i); whole-graph span = elapsed(before_0, after_last);
/// STALL = span - sum(compute). Values outside [0, 60ms] are treated as bogus
/// (cross-replay/stale) and counted, not summed. Logs every 256 replays.
pub fn graph_kt_read(stream: &Arc<CudaStream>) {
    if !graph_kt_on() {
        return;
    }
    let c = graph_kt_chain().lock().unwrap();
    let n = c.0.len().min(c.1.len()).min(c.2.len());
    if n == 0 {
        return;
    }
    let ctx = stream.context();
    let ev = |x: usize| x as crate::cudarc::driver::sys::CUevent;
    let sane = |us: f64| (0.0..=60_000.0).contains(&us);

    let mut compute_sum = 0.0_f64;
    let mut bogus = 0u64;
    let mut local: std::collections::BTreeMap<&'static str, f64> = std::collections::BTreeMap::new();
    for i in 0..n {
        if let Ok(ms) = crate::kernel::event_elapsed_ms(&ctx, ev(c.0[i]), ev(c.1[i])) {
            let us = ms as f64 * 1000.0;
            if sane(us) {
                *local.entry(c.2[i]).or_insert(0.0) += us;
                compute_sum += us;
            } else {
                bogus += 1;
            }
        }
    }
    // Whole-graph span (first kernel start -> last kernel end): one robust pair.
    let span = crate::kernel::event_elapsed_ms(&ctx, ev(c.0[0]), ev(c.1[n - 1]))
        .map(|ms| ms as f64 * 1000.0)
        .ok()
        .filter(|&us| sane(us))
        .unwrap_or(compute_sum);

    #[allow(clippy::type_complexity)]
    static ACC: std::sync::OnceLock<
        std::sync::Mutex<(std::collections::BTreeMap<String, f64>, f64, f64, u64, u64)>,
    > = std::sync::OnceLock::new();
    // (per-name compute_us, span_us, compute_sum_us, bogus, passes)
    let mut g = ACC
        .get_or_init(|| std::sync::Mutex::new((std::collections::BTreeMap::new(), 0.0, 0.0, 0, 0)))
        .lock()
        .unwrap();
    for (name, us) in local {
        *g.0.entry(name.to_string()).or_insert(0.0) += us;
    }
    g.1 += span;
    g.2 += compute_sum;
    g.3 += bogus;
    g.4 += 1;
    let passes = g.4;
    if passes % 256 == 0 {
        let p = passes as f64;
        let span_us = g.1 / p;
        let compute_us = g.2 / p;
        let stall_us = (span_us - compute_us).max(0.0);
        // Build the whole block as one string + single write so the two ranks'
        // reports interleave at block granularity, not per-line.
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = write!(
            out,
            "=== SKEIN_GRAPH_KERNEL_TIMING passes={passes} | graph_span={span_us:.0}us/tok  compute={compute_us:.0}us/tok ({:.0}%)  STALL/sync={stall_us:.0}us/tok ({:.0}%)  bogus_pairs={:.1}/tok ===",
            compute_us / span_us * 100.0,
            stall_us / span_us * 100.0,
            g.3 as f64 / p,
        );
        let mut rows: Vec<(&String, &f64)> = g.0.iter().collect();
        rows.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (name, us) in rows.iter().take(25) {
            let _ = write!(
                out,
                "\n  {:<34} {:9.1}us/tok  {:5.1}% of compute",
                name,
                *us / p,
                **us / g.2 * 100.0
            );
        }
        eprintln!("{out}");
    }
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
