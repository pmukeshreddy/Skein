use crate::{
    host::{DeviceBuffer, HostOp},
    kernel::{CudaGraphTiming, KernelOp, record_cuda_graph_timings},
};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, result,
};

use fixedbitset::FixedBitSet;
use half::{bf16, f16};
use itertools::Itertools;
use luminal::hlir::*;
use luminal::prelude::{
    petgraph::{
        Directed, Direction,
        algo::{Cycle, toposort},
        prelude::StableGraph,
        visit::{EdgeRef, NodeIndexable},
    },
    *,
};

use luminal_tracing::PerfettoGuard;
use luminal_tracing::prost::Message;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::{
    collections::{VecDeque, hash_map::Entry},
    fmt::Debug,
    fs::File,
    sync::Arc,
    time::Duration,
};
use tracing::{Level, span, trace};
use uuid::Uuid;

const ARENA_ALIGNMENT: usize = 256;

pub enum CudaInput {
    Buffer(CudaSlice<u8>),
    Ptr(u64),
}

/// Executable operation in the runtime graph.
/// All operations (including CUDA graphs) are now HostOps.
pub(crate) struct ExecutableHostOp {
    stream: Arc<CudaStream>,
    inputs: Vec<NodeIndex>,
    output: NodeIndex,
    internal: Arc<Box<dyn HostOp>>,
}

/// Statistics for a single kernel execution
#[derive(Debug, Clone)]
pub struct KernelStats {
    pub name: &'static str,
    pub execution_time_us: f64,
    pub bytes_loaded: usize,
    pub bytes_stored: usize,
    pub flops: usize,
    pub bandwidth_gbps: f64,
    pub tflops: f64,
}

impl Debug for ExecutableHostOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HostOp: ({:?})", self.internal)
    }
}

#[derive(Clone)]
pub(crate) struct BufferSpec {
    bytes: Expression,
    dtype: DType,
}

#[derive(Debug, Clone)]
struct PlannedBuffer {
    node: NodeIndex,
    bytes: usize,
    start: usize,
    end: usize,
}

/// Per-bucket compiled state. Each bucket holds its own executable graph,
/// explicit runtime metadata, intermediate buffers, and node mappings.
/// Weights (hlir_buffers) are shared.
pub(crate) struct CompiledBucket {
    pub(crate) exec_graph: StableGraph<ExecutableHostOp, (), Directed>,
    pub(crate) node_to_exec: FxHashMap<NodeIndex, NodeIndex>,
    /// Single reusable arena for all intermediate buffers in this bucket.
    pub(crate) arena: Option<CudaSlice<u8>>,
    pub(crate) arena_bytes: usize,
    pub(crate) logical_buffer_offsets: FxHashMap<NodeIndex, usize>,
    pub(crate) logical_buffer_bytes: FxHashMap<NodeIndex, usize>,
    pub(crate) cached_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    pub(crate) buffer_specs: FxHashMap<NodeIndex, BufferSpec>,
    /// Dtype of each graph Input (llir node), recorded at build. Lets
    /// `output_dtype` resolve a passed-through Input (an Input wired straight to
    /// an Output, e.g. a residual carry) whose data buffer has no `buffer_specs`
    /// entry — so the bf16->f32 widening read path knows it is reading bf16, not
    /// raw f32 (which would halve it).
    pub(crate) input_dtypes: FxHashMap<NodeIndex, DType>,
    pub(crate) llir_to_hlir: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) hlir_to_llir: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_producers: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_alias_map: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_data_map: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) preserved_hlir_inputs: FxHashSet<NodeIndex>,
    pub(crate) kernel_names: Vec<&'static str>,
    pub(crate) last_dyn_map: FxHashMap<char, usize>,
    pub(crate) intermediate_buffer_dims: FxHashSet<char>,
    /// Which bucket index per dim this compilation targets
    pub(crate) bucket_indices: FxHashMap<char, usize>,
    /// Whether HLIR pointers have been synced into this bucket's cached_buffer_ptrs
    pub(crate) hlir_synced: bool,
    /// Cached topological execution order of `exec_graph`. The graph is fixed
    /// after build, so the order is computed once and reused every execute
    /// instead of re-running `toposort` (which allocates) on the per-token hot
    /// path. Empty = not yet computed.
    pub(crate) exec_order: Vec<NodeIndex>,
}

impl CompiledBucket {
    fn new() -> Self {
        CompiledBucket {
            exec_graph: StableGraph::default(),
            node_to_exec: FxHashMap::default(),
            arena: None,
            arena_bytes: 0,
            logical_buffer_offsets: FxHashMap::default(),
            logical_buffer_bytes: FxHashMap::default(),
            cached_buffer_ptrs: FxHashMap::default(),
            buffer_specs: FxHashMap::default(),
            exec_order: Vec::new(),
            input_dtypes: FxHashMap::default(),
            llir_to_hlir: FxHashMap::default(),
            hlir_to_llir: FxHashMap::default(),
            output_producers: FxHashMap::default(),
            output_alias_map: FxHashMap::default(),
            output_data_map: FxHashMap::default(),
            preserved_hlir_inputs: FxHashSet::default(),
            kernel_names: Vec::new(),
            last_dyn_map: FxHashMap::default(),
            intermediate_buffer_dims: FxHashSet::default(),
            bucket_indices: FxHashMap::default(),
            hlir_synced: false,
        }
    }
}

pub struct CudaRuntime {
    // Shared state across all buckets
    pub hlir_buffers: FxHashMap<NodeIndex, CudaInput>,
    /// HLIR input nodes (e.g. model weights) the caller has marked persistent:
    /// they are uploaded once and must NOT be consumed/freed after each execute,
    /// so they survive across forwards instead of being re-uploaded every step.
    persistent_hlir_inputs: FxHashSet<NodeIndex>,
    cuda_stream: Arc<CudaStream>,
    changed_hlir: FxHashSet<NodeIndex>,
    pub(crate) cuda_graph_timings: Vec<(CudaGraphTiming, Uuid)>,
    pub last_kernel_stats: Vec<KernelStats>,
    pub last_total_time_us: f64,
    kernel_cache: FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    /// When true, execute() skips input buffer consumption (used during search/profile)
    profiling: bool,

    // Per-bucket compiled state
    compiled_buckets: Vec<CompiledBucket>,
    active_bucket: usize,
    /// Bucket definitions per dimension (empty = single-bucket mode)
    dim_buckets: FxHashMap<char, Vec<DimBucket>>,

    /// Non-owning CudaSlice wrappers for external device pointers.
    /// ManuallyDrop prevents cuMemFree — the external allocator (e.g. PyTorch) owns the memory.
    external_buffers: FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,

    /// Pending output pointer registrations: HLIR output id -> (device_ptr, n_bytes)
    /// Set by python before execute(), consumed at start of execute()
    output_ptr_registrations: FxHashMap<NodeIndex, (u64, usize)>,

    /// Non-owning CudaSlice views of external output pointers, keyed by LLIR data node
    /// ManuallyDrop prevents cuMemFree -- Pytorch owns the memory
    external_output_buffers: FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,

    /// Device buffer (1 i32) holding the current decode `position`, updated via
    /// [`set_decode_position`](Self::set_decode_position). Read by the
    /// `kv_slot_write` kernel so the KV-append destination offset is computed on
    /// device (not baked into a host-issued DtoD) — making the append a
    /// graph-capturable node for the full-step CUDA graph.
    decode_position: std::cell::RefCell<Option<CudaSlice<i32>>>,
    /// Lazily-compiled `kv_slot_write` kernel (module + function).
    kv_write_kernel: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    /// Lazily-compiled `kv_slot_write_batched` kernel (batched per-row KV append
    /// for the single-process batched continuous-batch driver).
    kv_write_batched_kernel: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    /// Lazily-compiled `shm_allreduce2` kernel (module + function). Under
    /// SKEIN_CAPTURE the all-reduce must launch from luminal so it lands on
    /// luminal's shared capture stream (skein_runtime's cudarc can't reach the
    /// raw function/stream across the version boundary). skein_runtime owns the
    /// cross-process shm setup and passes the raw shm device pointer.
    shm_allreduce_kernel: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    /// Full-step CUDA graph captured from the shared stream (SKEIN_CAPTURE): the
    /// whole pre-all-gather decode forward recorded once, replayed per token with
    /// one `cuGraphLaunch` instead of ~12k host CUDA calls.
    captured_graph_exec: std::cell::RefCell<Option<cudarc::driver::sys::CUgraphExec>>,
}

/// Process-wide shared **non-default** CUDA stream for the `SKEIN_CAPTURE` path.
/// Stream capture is impossible on the legacy default stream (handle 0), so under
/// capture every segment runtime + KV writes (luminal) AND the cross-process shm
/// all-reduce (skein_runtime, via [`capture_stream_raw`]) share ONE non-default
/// stream — preserving the single-stream ordering the device-resident handoffs
/// rely on. A rank process owns one device, so a single global stream is correct.
static SHARED_CAPTURE_STREAM: std::sync::OnceLock<Arc<CudaStream>> = std::sync::OnceLock::new();

fn capture_stream(ctx: &Arc<cudarc::driver::CudaContext>) -> Arc<CudaStream> {
    SHARED_CAPTURE_STREAM
        .get_or_init(|| ctx.new_stream().expect("create shared SKEIN_CAPTURE stream"))
        .clone()
}

/// Raw `CUstream` handle (as `u64`) of the shared capture stream, or 0 if not yet
/// created. Returned as a plain integer so skein_runtime (a different cudarc
/// version) can launch the shm all-reduce on this exact stream via `cuLaunchKernel`
/// without needing the typed `CudaStream` to cross the version boundary.
pub fn capture_stream_raw() -> u64 {
    SHARED_CAPTURE_STREAM
        .get()
        .map(|s| s.cu_stream() as usize as u64)
        .unwrap_or(0)
}

impl CudaRuntime {
    /// Creates a new CudaRuntime on device 0 (blocking-sync, default stream).
    pub fn new() -> Result<Self, cudarc::driver::DriverError> {
        Self::new_on(0)
    }

    /// Creates a CudaRuntime bound to a specific CUDA device ordinal. Used to
    /// place tensor-parallel shards on distinct GPUs within one process (each
    /// runtime owns its device's context; per-op `bind_to_thread` keeps the
    /// correct context current when several runtimes share a thread).
    pub fn new_on(device: usize) -> Result<Self, cudarc::driver::DriverError> {
        let ctx = cudarc::driver::CudaContext::new(device)?;
        ctx.bind_to_thread()?;
        // SKEIN_CTX_SPIN: busy-wait at syncs instead of sleeping. BLOCKING_SYNC
        // host wake latency was measured at ~56us/sync in this VM, which inflates
        // every NCCL/stream wait; SPIN trades a hot CPU core for low-latency wakes
        // (right for a dedicated latency-critical decode loop).
        let sched = if std::env::var_os("SKEIN_CTX_SPIN").is_some() {
            cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_SPIN
        } else {
            cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC
        };
        ctx.set_flags(sched)?;
        // Single default stream per device (compute kernels, graphs, memcpy, and
        // NCCL all share `ctx.default_stream()`), so cudarc's per-buffer-usage
        // CudaEvent tracking is pure host overhead — nsys measured ~11k
        // cuEventRecord/token (~6ms host) that starves the GPU during decode.
        // Under single-stream execution none of the documented hazards apply, so
        // disabling it is safe. Must be set before any CudaSlice is allocated.
        if std::env::var_os("SKEIN_NO_EVENT_TRACKING").is_some() {
            unsafe { ctx.disable_event_tracking() };
        }
        // SKEIN_CAPTURE: share one non-default stream (capture can't run on the
        // legacy default stream); see `capture_stream`.
        let stream = if std::env::var_os("SKEIN_CAPTURE").is_some() {
            capture_stream(&ctx)
        } else {
            ctx.default_stream()
        };

        Ok(Self::initialize(stream))
    }

    /// Get the active compiled bucket.
    fn active(&self) -> &CompiledBucket {
        &self.compiled_buckets[self.active_bucket]
    }

    /// Get the active compiled bucket mutably.
    fn active_mut(&mut self) -> &mut CompiledBucket {
        &mut self.compiled_buckets[self.active_bucket]
    }

    /// Names of CUDA kernels compiled into the active bucket.
    pub fn kernel_names(&self) -> &[&'static str] {
        &self.active().kernel_names
    }

    /// Host operations in the active executable graph, for diagnostics.
    pub fn host_ops(&self) -> Vec<&dyn HostOp> {
        self.active()
            .exec_graph
            .node_weights()
            .map(|op| op.internal.as_ref().as_ref() as &dyn HostOp)
            .collect()
    }

    fn bucket_buffer(
        bucket: &CompiledBucket,
        stream: &Arc<CudaStream>,
        logical_node: &NodeIndex,
    ) -> Option<DeviceBuffer> {
        let arena = bucket.arena.as_ref()?;
        let offset = *bucket.logical_buffer_offsets.get(logical_node)?;
        let len = *bucket.logical_buffer_bytes.get(logical_node)?;
        let ptr = arena.device_ptr(stream).0.checked_add(offset as u64)?;
        Some(DeviceBuffer::new(ptr, len))
    }

    fn copy_device_buffer_to_new_slice(
        stream: &Arc<CudaStream>,
        src: DeviceBuffer,
    ) -> CudaSlice<u8> {
        let dst = stream.alloc_zeros::<u8>(src.len()).unwrap();
        let dst_ptr = dst.device_ptr(stream).0;
        unsafe {
            result::memcpy_dtod_async(dst_ptr, src.ptr(), src.len(), stream.cu_stream())
                .expect("cuMemcpyDtoDAsync failed");
        }
        stream.synchronize().unwrap();
        dst
    }

    fn resolve_runtime_buffer(
        bucket: &CompiledBucket,
        stream: &Arc<CudaStream>,
        hlir_buffers: &FxHashMap<NodeIndex, CudaInput>,
        external_buffers: &FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,
        external_output_buffers: &FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,
        mut node: NodeIndex,
    ) -> Option<DeviceBuffer> {
        let mut visited = FxHashSet::default();
        loop {
            if !visited.insert(node) {
                return None;
            }

            if let Some(ext) = external_output_buffers.get(&node) {
                return Some(DeviceBuffer::new(ext.device_ptr(stream).0, ext.len()));
            }

            if let Some(buf) = Self::bucket_buffer(bucket, stream, &node) {
                return Some(buf);
            }

            if let Some(hlir_node) = bucket.llir_to_hlir.get(&node) {
                match hlir_buffers.get(hlir_node) {
                    Some(CudaInput::Buffer(buf)) => {
                        return Some(DeviceBuffer::new(buf.device_ptr(stream).0, buf.len()));
                    }
                    Some(CudaInput::Ptr(_)) => {
                        if let Some(ext) = external_buffers.get(hlir_node) {
                            return Some(DeviceBuffer::new(ext.device_ptr(stream).0, ext.len()));
                        }
                    }
                    None => {}
                }
            }

            let alias_target = bucket.output_alias_map.get(&node)?;
            node = *alias_target;
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        let f = File::open(file_path).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&f).unwrap() };
        let st = SafeTensors::deserialize(&mmap).unwrap();
        for node in cx.graph.node_indices() {
            if let Some(Input { label, .. }) = (*cx.graph[node]).as_any().downcast_ref::<Input>()
                && let Ok(tensor) = st.tensor(label)
            {
                self.changed_hlir.insert(node);
                match tensor.dtype() {
                    safetensors::Dtype::F32 => {
                        let bytes = tensor.data();
                        let f32s: &[f32] = bytemuck::cast_slice(bytes);
                        let dev = f32s.to_cuda_input(&self.cuda_stream);
                        self.hlir_buffers.insert(node, dev);
                    }
                    safetensors::Dtype::U8
                    | safetensors::Dtype::BF16
                    | safetensors::Dtype::F16
                    | safetensors::Dtype::F8_E4M3
                    | safetensors::Dtype::F8_E5M2
                    | safetensors::Dtype::F8_E8M0 => {
                        let bytes = tensor.data();
                        let dev = bytes.to_cuda_input(&self.cuda_stream);
                        self.hlir_buffers.insert(node, dev);
                    }
                    dtype => unimplemented!("{dtype} loading not supported yet"),
                }
            }
        }
    }

    pub fn set_data(&mut self, id: impl ToId, data: impl ToCudaInput) {
        let id = id.to_id();
        // A3 (in-place input reuse): if a device buffer of the same byte length is
        // already registered for this id, overwrite its contents in place instead
        // of allocating a fresh one. The device pointer then stays stable across
        // executes, which a captured full-step graph (SKEIN_CAPTURE) requires — it
        // bakes the input pointer in at capture time and reads from it on every
        // replay. The buffer is also marked persistent so execute()'s consume pass
        // keeps it (it would otherwise be freed after the forward and re-allocated
        // at a new address next token).
        let same_size = matches!(
            self.hlir_buffers.get(&id),
            Some(CudaInput::Buffer(buf)) if buf.len() == data.as_host_bytes().len()
        );
        let dbg_small = std::env::var_os("SKEIN_FI_LOG").is_some()
            && data.as_host_bytes().len() <= 8;
        if same_size {
            let stream = self.cuda_stream.clone();
            let mut p = 0u64;
            if let Some(CudaInput::Buffer(buf)) = self.hlir_buffers.get_mut(&id) {
                stream.memcpy_htod(data.as_host_bytes(), buf).unwrap();
                if dbg_small {
                    p = buf.device_ptr(&stream).0;
                }
            }
            self.persistent_hlir_inputs.insert(id);
            self.changed_hlir.insert(id);
            if dbg_small {
                eprintln!("SKEIN_SETDATA id={} INPLACE ptr=0x{:x} bytes={}", id.index(), p, data.as_host_bytes().len());
            }
            return;
        }
        let nbytes = data.as_host_bytes().len();
        let cuda_input = data.to_cuda_input(&self.cuda_stream);
        if dbg_small {
            if let CudaInput::Buffer(buf) = &cuda_input {
                eprintln!("SKEIN_SETDATA id={} ALLOC   ptr=0x{:x} bytes={}", id.index(), buf.device_ptr(&self.cuda_stream).0, nbytes);
            }
        }
        self.hlir_buffers.insert(id, cuda_input);
        self.changed_hlir.insert(id);
        // Under SKEIN_CAPTURE, mark the buffer persistent on its FIRST allocation
        // too. Otherwise execute()'s consume pass frees it after this forward, so
        // next step's set_data re-allocates at a NEW device address — and the
        // captured full-step graph, which baked the buffer's pointer at capture
        // time, would then read a stale/freed address on every replay (the cause
        // of correct warmup but garbage replays). Persisting it keeps the address
        // stable so the in-place branch above handles all subsequent steps and the
        // captured graph always reads the current data.
        static CAP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *CAP.get_or_init(|| std::env::var_os("SKEIN_CAPTURE").is_some()) {
            self.persistent_hlir_inputs.insert(id);
        }
    }

    /// Mark an HLIR input node (e.g. a model weight) as persistent: its buffer
    /// is kept across executes instead of being consumed/freed afterwards, so it
    /// is uploaded once rather than re-fed every forward. Call after `set_data`.
    pub fn mark_hlir_persistent(&mut self, id: impl ToId) {
        self.persistent_hlir_inputs.insert(id.to_id());
    }

    /// Free every bucket's intermediate-buffer arena (re-allocated lazily on the
    /// next execute). Persistent inputs/weights are untouched. Lets a caller
    /// reclaim a graph's arena between forwards so two graphs' arenas aren't
    /// resident at once. Public inherent twin of the `Runtime` trait method.
    pub fn free_intermediate_arenas(&mut self) {
        let _ = self.cuda_stream.synchronize();
        for bucket in &mut self.compiled_buckets {
            bucket.arena = None;
            bucket.cached_buffer_ptrs.clear();
        }
    }

    /// Allocate a zeroed GPU buffer for the given node. This is more efficient than
    /// `set_data` with a host-side zero vector since it avoids the host allocation and H2D copy.
    pub fn set_zeros(&mut self, id: impl ToId, num_bytes: usize) {
        let id = id.to_id();
        let buf = self.cuda_stream.alloc_zeros(num_bytes).unwrap();
        self.hlir_buffers.insert(id, CudaInput::Buffer(buf));
        self.changed_hlir.insert(id);
    }

    /// Device pointer of an already-resident input/weight buffer, if any. Used to
    /// share one graph's resident weights with another graph (e.g. a decode graph
    /// and a seq=N prefill graph) via `set_device_ptr`, instead of loading a
    /// second copy. Returns None if the node has no buffer yet.
    pub fn hlir_device_ptr(&self, id: impl ToId) -> Option<u64> {
        match self.hlir_buffers.get(&id.to_id())? {
            CudaInput::Buffer(buf) => Some(buf.device_ptr(&self.cuda_stream).0),
            CudaInput::Ptr(p) => Some(*p),
        }
    }

    /// Set an external CUDA device pointer as input data. Zero-copy.
    /// The caller must ensure the pointer remains valid for the runtime's lifetime.
    ///
    /// # Safety
    /// The device pointer must point to a valid CUDA allocation on the same device
    /// as this runtime's stream, with at least `n_bytes` bytes available.
    pub unsafe fn set_device_ptr(&mut self, id: impl ToId, device_ptr: u64, n_bytes: usize) {
        debug_assert!(device_ptr != 0, "set_device_ptr called with null pointer");
        let id = id.to_id();
        // Create CudaSlice view via cudarc's upgrade_device_ptr.
        // ManuallyDrop prevents cuMemFree on drop (external allocator owns this memory).
        let slice = unsafe {
            self.cuda_stream
                .upgrade_device_ptr::<u8>(device_ptr, n_bytes)
        };
        self.external_buffers
            .insert(id, std::mem::ManuallyDrop::new(slice));
        self.hlir_buffers.insert(id, CudaInput::Ptr(device_ptr));
        self.changed_hlir.insert(id);
    }

    /// Register an external device pointer for an output tensor (zero-copy output).
    /// The pointer is stored lazily — resolution to LLIR nodes happens in execute().
    ///
    /// # Safety
    /// The device pointer must point to a valid CUDA allocation with at least `n_bytes` bytes,
    /// and must remain valid through the next execute() call.
    pub unsafe fn set_output_device_ptr(&mut self, id: impl ToId, device_ptr: u64, n_bytes: usize) {
        debug_assert!(
            device_ptr != 0,
            "set_output_device_ptr called with null pointer"
        );
        self.output_ptr_registrations
            .insert(id.to_id(), (device_ptr, n_bytes));
    }

    pub fn output_is_zero_copy(&self, id: impl ToId) -> bool {
        let producer = self.find_producer_node(id);
        let data_node = self.follow_aliases(producer);
        self.external_output_buffers.contains_key(&data_node)
    }

    /// Find the LLIR producing node for an output tensor.
    fn find_producer_node(&self, id: impl ToId) -> NodeIndex {
        let id = id.to_id();
        let bucket = self.active();
        *bucket
            .output_producers
            .get(&id)
            .expect("Cannot find output tensor!")
    }

    /// Follow `output_aliases_input` to find the node whose buffer actually contains
    /// the output data. For in-place ops, data lives in the aliased input's buffer.
    fn follow_aliases(&self, mut node: NodeIndex) -> NodeIndex {
        let bucket = self.active();
        while let Some(alias_target) = bucket.output_alias_map.get(&node) {
            node = *alias_target;
        }
        node
    }

    /// Follow `output_data_input` to trace data lineage back to the originating
    /// HLIR input. Used by remove_buffer to find the correct buffer to extract
    /// for the remove_buffer/set_buffer roundtrip pattern.
    ///
    /// For in-place ops (output_aliases_input), this traces to the aliased input.
    /// For copy-then-modify ops (like Scatter), this traces through the copy source
    /// to the HLIR input, so the roundtrip correctly swaps the HLIR buffer.
    fn follow_data_lineage(&self, mut node: NodeIndex) -> NodeIndex {
        let bucket = self.active();
        while let Some(data_target) = bucket.output_data_map.get(&node) {
            node = *data_target;
        }
        node
    }

    #[tracing::instrument(skip_all)]
    /// Resolve the LLIR node that actually holds the data for an output tensor.
    /// For in-place ops, follows output_aliases_input to the aliased input buffer.
    fn resolve_data_node(&self, id: impl ToId) -> NodeIndex {
        let producer = self.find_producer_node(id);
        self.follow_aliases(producer)
    }

    fn get_output_data(&self, id: impl ToId) -> Vec<u8> {
        let data_id = self.resolve_data_node(id);
        let bucket = self.active();

        let truncate_to_logical_bytes = |mut data: Vec<u8>| {
            if let Some(spec) = bucket.buffer_specs.get(&data_id)
                && let Some(logical_bytes) = spec.bytes.exec(&bucket.last_dyn_map)
            {
                data.truncate(logical_bytes.min(data.len()));
            }
            data
        };

        let _span = span!(Level::TRACE, "dtoh").entered();
        // If predecessor is an Input node, data lives in hlir_buffers
        if let Some(hlir_node) = bucket.llir_to_hlir.get(&data_id) {
            match self
                .hlir_buffers
                .get(hlir_node)
                .expect("Cannot find input tensor in runtime!")
            {
                CudaInput::Buffer(buf) => self.cuda_stream.clone_dtoh(buf).unwrap(),
                CudaInput::Ptr(_) => {
                    // External device pointer — use the CudaSlice view from external_buffers
                    if let Some(ext) = self.external_buffers.get(hlir_node) {
                        self.cuda_stream.clone_dtoh(&**ext).unwrap()
                    } else {
                        panic!(
                            "Cannot read raw pointer input — no external_buffers entry for node"
                        );
                    }
                }
            }
        } else {
            if let Some(ext) = self.external_output_buffers.get(&data_id) {
                return truncate_to_logical_bytes(self.cuda_stream.clone_dtoh(&**ext).unwrap());
            }

            // Predecessor is a computation node — data is in the intermediate arena.
            truncate_to_logical_bytes(
                Self::bucket_buffer(bucket, &self.cuda_stream, &data_id)
                    .expect("Cannot find tensor in runtime!")
                    .clone_dtoh(&self.cuda_stream)
                    .unwrap(),
            )
        }
    }

    /// Device buffer `(raw_ptr, byte_len)` backing an output tensor, with **no
    /// host copy**. `None` if the id is not a known output of the active bucket.
    /// Used to hand one segment's output to the next segment by device pointer
    /// (paired with [`set_device_ptr`](Self::set_device_ptr) on the consumer),
    /// eliminating the GPU->host->GPU activation round-trip.
    pub fn output_device_buffer(&self, id: impl ToId) -> Option<(u64, usize)> {
        let id = id.to_id();
        if !self.active().output_producers.contains_key(&id) {
            return None;
        }
        let buf = self.resolve_output_buffer(id);
        Some((buf.ptr(), buf.len()))
    }

    /// Resolve the device-side buffer for an output tensor without copying to host.
    /// Used by copy_output_to_device_ptr for DtoD transfers.
    fn resolve_output_buffer(&self, id: impl ToId) -> DeviceBuffer {
        let data_id = self.resolve_data_node(id);
        let bucket = self.active();
        if let Some(ext) = self.external_output_buffers.get(&data_id) {
            return DeviceBuffer::new(ext.device_ptr(&self.cuda_stream).0, ext.len());
        }
        if let Some(hlir_node) = bucket.llir_to_hlir.get(&data_id) {
            match self
                .hlir_buffers
                .get(hlir_node)
                .expect("Cannot find input tensor in runtime!")
            {
                CudaInput::Buffer(buf) => {
                    DeviceBuffer::new(buf.device_ptr(&self.cuda_stream).0, buf.len())
                }
                CudaInput::Ptr(_) => self
                    .external_buffers
                    .get(hlir_node)
                    .map(|ext| DeviceBuffer::new(ext.device_ptr(&self.cuda_stream).0, ext.len()))
                    .expect("Cannot read raw pointer input — no external_buffers entry for node"),
            }
        } else {
            Self::bucket_buffer(bucket, &self.cuda_stream, &data_id)
                .expect("Cannot find tensor in runtime!")
        }
    }

    /// Copy output tensor data to an external CUDA device pointer (DtoD).
    /// Much faster than get_f32 + HtoD for CUDA-to-CUDA workflows.
    ///
    /// # Safety
    /// The dest_ptr must be a valid CUDA device allocation with at least n_bytes available.
    pub unsafe fn copy_output_to_device_ptr(&self, id: impl ToId, dest_ptr: u64, n_bytes: usize) {
        debug_assert!(
            dest_ptr != 0,
            "copy_output_to_device_ptr called with null pointer"
        );
        let src = self.resolve_output_buffer(id);
        let copy_bytes = n_bytes.min(src.len());
        unsafe {
            result::memcpy_dtod_async(
                dest_ptr,
                src.ptr(),
                copy_bytes,
                self.cuda_stream.cu_stream(),
            )
            .expect("cuMemcpyDtoDAsync failed");
        }
        // The DtoD copy is stream-ordered on the rank's shared default stream
        // with the consumer that later reads it, so no host sync is needed here.
    }

    /// Update the device-resident decode `position` (one i32). Stream-ordered, so
    /// a subsequent [`copy_output_to_device_ptr_kv`](Self::copy_output_to_device_ptr_kv)
    /// on the same stream reads the new value.
    pub fn set_decode_position(&self, pos: usize) {
        let mut slot = self.decode_position.borrow_mut();
        if slot.is_none() {
            *slot = Some(
                self.cuda_stream
                    .alloc_zeros::<i32>(1)
                    .expect("alloc decode_position"),
            );
        }
        let buf = slot.as_mut().unwrap();
        self.cuda_stream
            .memcpy_htod(&[pos as i32], buf)
            .expect("memcpy_htod decode_position");
    }

    fn kv_write_fn(&self) -> &CudaFunction {
        let (_, func) = self.kv_write_kernel.get_or_init(|| {
            // dst = base + position * (n_words*4) bytes; copy n_words u32 words.
            // position is read from the device buffer at launch (kernel) time, so
            // the launch is identical every step and is graph-capturable.
            let src = r#"
extern "C" __global__ void kv_slot_write(
    unsigned long long src, unsigned long long base,
    unsigned long long pos_ptr, int n_words
) {
    long long pos = (long long)(*((const int*)pos_ptr));
    unsigned int* d = (unsigned int*)(base + (unsigned long long)pos * (unsigned long long)n_words * 4ULL);
    const unsigned int* s = (const unsigned int*)src;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_words) d[i] = s[i];
}
"#;
            let ptx = crate::compile_module_image_for_current_device(self.cuda_stream.context(), src)
                .expect("compile kv_slot_write");
            let module = self
                .cuda_stream
                .context()
                .load_module(ptx)
                .expect("load kv_slot_write module");
            let func = module
                .load_function("kv_slot_write")
                .expect("load kv_slot_write fn");
            (module, func)
        });
        func
    }

    /// Device-side KV append: copy this output tensor into `base_ptr` at slot
    /// `position` (read from the device buffer set by [`set_decode_position`]),
    /// i.e. `base_ptr + position * n_bytes`. Replaces the host-issued DtoD whose
    /// destination was baked in per step, so the append can live inside a
    /// replayable full-step CUDA graph.
    ///
    /// # Safety
    /// `base_ptr` must be a valid device allocation; `set_decode_position` must
    /// have been called.
    pub unsafe fn copy_output_to_device_ptr_kv(&self, id: impl ToId, base_ptr: u64, n_bytes: usize) {
        debug_assert!(base_ptr != 0, "copy_output_to_device_ptr_kv null base");
        let src = self.resolve_output_buffer(id);
        let copy_bytes = n_bytes.min(src.len());
        debug_assert!(copy_bytes % 4 == 0, "KV slot bytes must be 4-aligned");
        let n_words = (copy_bytes / 4) as i32;
        if n_words == 0 {
            return;
        }
        let src_ptr = src.ptr();
        let pos_ptr = {
            let slot = self.decode_position.borrow();
            let buf = slot.as_ref().expect("set_decode_position before KV write");
            buf.device_ptr(&self.cuda_stream).0
        };
        let func = self.kv_write_fn().clone();
        let cfg = LaunchConfig {
            grid_dim: ((n_words as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.cuda_stream
                .launch_builder(&func)
                .arg(&src_ptr)
                .arg(&base_ptr)
                .arg(&pos_ptr)
                .arg(&n_words)
                .launch(cfg)
                .expect("launch kv_slot_write");
        }
    }

    /// Batched KV append: write the new token's K/V for every batch row into a
    /// `[batch, cap, row_words]` cache at the shared decode `position`, i.e.
    /// `dst[r, position, :] = src[r, :]`. The source `id` output is `[batch,
    /// row_words]`. Used by the single-process batched continuous-batch driver
    /// (synchronous/lockstep: all rows share `position`). Strided per row, one
    /// kernel launch.
    ///
    /// # Safety
    /// `base_ptr` is a valid `batch*cap*row_words*4`-byte device allocation;
    /// `set_decode_position` must have been called.
    pub unsafe fn copy_output_to_kv_slot_batched(
        &self,
        id: impl ToId,
        base_ptr: u64,
        batch: usize,
        cap: usize,
        row_bytes: usize,
    ) {
        debug_assert!(base_ptr != 0, "copy_output_to_kv_slot_batched null base");
        let src = self.resolve_output_buffer(id);
        debug_assert!(row_bytes % 4 == 0, "KV row bytes must be 4-aligned");
        let row_words = (row_bytes / 4) as i32;
        if row_words == 0 || batch == 0 {
            return;
        }
        let src_ptr = src.ptr();
        let pos_ptr = {
            let slot = self.decode_position.borrow();
            let buf = slot.as_ref().expect("set_decode_position before KV write");
            buf.device_ptr(&self.cuda_stream).0
        };
        let func = self.kv_write_batched_fn().clone();
        let total = (batch as i32) * row_words;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let batch_i = batch as i32;
        let cap_i = cap as i32;
        unsafe {
            self.cuda_stream
                .launch_builder(&func)
                .arg(&src_ptr)
                .arg(&base_ptr)
                .arg(&pos_ptr)
                .arg(&row_words)
                .arg(&batch_i)
                .arg(&cap_i)
                .launch(cfg)
                .expect("launch kv_slot_write_batched");
        }
    }

    fn kv_write_batched_fn(&self) -> &CudaFunction {
        let (_, func) = self.kv_write_batched_kernel.get_or_init(|| {
            let src = r#"
extern "C" __global__ void kv_slot_write_batched(
    unsigned long long src, unsigned long long base,
    unsigned long long pos_ptr, int row_words, int batch, int cap
) {
    long long pos = (long long)(*((const int*)pos_ptr));
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batch * (long long)row_words;
    if ((long long)i >= total) return;
    int r = i / row_words;
    int w = i - r * row_words;
    unsigned int* d = (unsigned int*)base;
    const unsigned int* s = (const unsigned int*)src;
    long long dst_idx = ((long long)r * (long long)cap + pos) * (long long)row_words + (long long)w;
    long long src_idx = (long long)r * (long long)row_words + (long long)w;
    d[dst_idx] = s[src_idx];
}
"#;
            let ptx = crate::compile_module_image_for_current_device(self.cuda_stream.context(), src)
                .expect("compile kv_slot_write_batched");
            let module = self
                .cuda_stream
                .context()
                .load_module(ptx)
                .expect("load kv_slot_write_batched module");
            let func = module
                .load_function("kv_slot_write_batched")
                .expect("load kv_slot_write_batched fn");
            (module, func)
        });
        func
    }

    fn shm_allreduce_fn(&self) -> &CudaFunction {
        let (_, func) = self.shm_allreduce_kernel.get_or_init(|| {
            // 2-rank one-shot all-reduce over mapped shared host memory. flags[]
            // are self-incremented generations (persist across graph replays), so
            // no host seq is baked in -> valid as a static node in a replayed
            // full-step graph. bf16 summed in fp32 (round-to-nearest-even).
            let src = r#"
extern "C" __global__ void shm_allreduce2(
    unsigned long long my_data, unsigned long long shm,
    int my_rank, int elems, int slot_bytes
) {
    volatile unsigned long long* flags = (volatile unsigned long long*)shm;
    char* base = (char*)shm + 64;
    int peer = 1 - my_rank;
    unsigned short* my_slot   = (unsigned short*)(base + (long long)my_rank * slot_bytes);
    unsigned short* peer_slot = (unsigned short*)(base + (long long)peer    * slot_bytes);
    unsigned short* d = (unsigned short*)my_data;
    int tid = threadIdx.x, n = blockDim.x;
    for (int i = tid; i < elems; i += n) my_slot[i] = d[i];
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        unsigned long long g = flags[my_rank] + 1ULL;
        flags[my_rank] = g;
        __threadfence_system();
        while (flags[peer] < g) { }
        __threadfence_system();
    }
    __syncthreads();
    for (int i = tid; i < elems; i += n) {
        unsigned int ua = ((unsigned int)d[i]) << 16;
        unsigned int ub = ((unsigned int)peer_slot[i]) << 16;
        float s = __uint_as_float(ua) + __uint_as_float(ub);
        unsigned int us = __float_as_uint(s);
        unsigned int r = us + 0x7FFFu + ((us >> 16) & 1u);
        d[i] = (unsigned short)(r >> 16);
    }
}
"#;
            let ptx = crate::compile_module_image_for_current_device(self.cuda_stream.context(), src)
                .expect("compile shm_allreduce2");
            let module = self
                .cuda_stream
                .context()
                .load_module(ptx)
                .expect("load shm_allreduce2 module");
            let func = module
                .load_function("shm_allreduce2")
                .expect("load shm_allreduce2 fn");
            (module, func)
        });
        func
    }

    /// In-place 2-rank sum all-reduce of `elems` bf16 at device pointer `data_ptr`,
    /// launched on THIS runtime's stream (so under SKEIN_CAPTURE it lands on the
    /// shared capture stream). `shm_ptr` is the cross-process mapped shared-memory
    /// device pointer (set up by skein_runtime); `slot_bytes` is its per-rank slot
    /// size. Both ranks must call with identical `elems`.
    ///
    /// # Safety
    /// `data_ptr` is a valid device buffer of `elems` bf16; `shm_ptr` is the
    /// registered shared region.
    pub unsafe fn device_shm_all_reduce(
        &self,
        data_ptr: u64,
        shm_ptr: u64,
        rank: i32,
        elems: usize,
        slot_bytes: i32,
    ) {
        let func = self.shm_allreduce_fn().clone();
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let elems_i = elems as i32;
        unsafe {
            self.cuda_stream
                .launch_builder(&func)
                .arg(&data_ptr)
                .arg(&shm_ptr)
                .arg(&rank)
                .arg(&elems_i)
                .arg(&slot_bytes)
                .launch(cfg)
                .expect("launch shm_allreduce2");
        }
    }

    /// Allocate a zeroed device buffer of `n_bytes` and return its raw pointer.
    /// The buffer is intentionally leaked (the `CudaSlice` wrapper is forgotten)
    /// so the caller owns the lifetime — used for per-request KV buffers in the
    /// single-process continuous-batch driver, which need a distinct, stable,
    /// zero-initialized contiguous KV buffer per in-flight request.
    pub fn alloc_device_zeros(&self, n_bytes: usize) -> u64 {
        let buf = self
            .cuda_stream
            .alloc_zeros::<u8>(n_bytes.max(1))
            .expect("alloc_device_zeros");
        let ptr = buf.device_ptr(&self.cuda_stream).0;
        std::mem::forget(buf);
        ptr
    }

    /// Read `elems` bf16 values at external device pointer `ptr` into a host
    /// `Vec<f32>` (widening bf16 -> f32). Used by the single-process
    /// `LocalTopology` to host-stage a cross-GPU all-reduce over device-resident
    /// activation handoffs (no NVLink P2P required). Synchronizes the stream so
    /// the producing kernel's write is complete before the copy is read.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of at least `elems * 2` bytes on
    /// this runtime's device.
    pub unsafe fn read_device_bf16_to_f32(&self, ptr: u64, elems: usize) -> Vec<f32> {
        let nbytes = elems * 2;
        let slice = unsafe { self.cuda_stream.upgrade_device_ptr::<u8>(ptr, nbytes) };
        let host: Vec<u8> = self.cuda_stream.clone_dtoh(&slice).unwrap();
        // The slice is a non-owning view of an externally-owned pointer; forget
        // it so dropping the wrapper does not cuMemFree the caller's buffer.
        std::mem::forget(slice);
        host.chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    }

    /// Write host `data` (f32, narrowed to bf16) to `data.len()` bf16 slots at
    /// external device pointer `ptr`. Counterpart of [`read_device_bf16_to_f32`]
    /// for writing the reduced all-reduce result back into each rank's buffer.
    /// Synchronizes so the host source stays valid through the copy and the
    /// device holds the result before a consumer reads it.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of at least `data.len() * 2` bytes.
    pub unsafe fn write_f32_to_device_bf16(&self, ptr: u64, data: &[f32]) {
        let bf: Vec<u16> = data.iter().map(|&x| bf16::from_f32(x).to_bits()).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&bf);
        let mut slice = unsafe { self.cuda_stream.upgrade_device_ptr::<u8>(ptr, bytes.len()) };
        self.cuda_stream.memcpy_htod(bytes, &mut slice).unwrap();
        let _ = self.cuda_stream.synchronize();
        std::mem::forget(slice);
    }

    // ---- Full-step CUDA graph capture/replay on the shared stream (SKEIN_CAPTURE).
    // Any segment runtime can drive these: they all share the one capture stream,
    // so a capture begun here records every kernel any runtime launches on it.

    /// Begin recording the shared stream into a CUDA graph.
    pub fn begin_stream_capture(&self) -> Result<(), cudarc::driver::DriverError> {
        let s = self.cuda_stream.cu_stream();
        unsafe {
            cudarc::driver::sys::cuStreamBeginCapture_v2(
                s,
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()
        }
    }

    /// End recording, instantiate the executable graph, and store it for replay.
    pub fn end_stream_capture(&self) -> Result<(), cudarc::driver::DriverError> {
        let s = self.cuda_stream.cu_stream();
        let mut graph = std::mem::MaybeUninit::uninit();
        let mut exec = std::mem::MaybeUninit::uninit();
        unsafe {
            cudarc::driver::sys::cuStreamEndCapture(s, graph.as_mut_ptr()).result()?;
            let graph = graph.assume_init();
            cudarc::driver::sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), graph, 0).result()?;
            cudarc::driver::sys::cuGraphDestroy(graph);
            *self.captured_graph_exec.borrow_mut() = Some(exec.assume_init());
        }
        Ok(())
    }

    /// Replay the captured full-step graph on the shared stream. Returns false if
    /// nothing is captured yet.
    pub fn replay_captured(&self) -> bool {
        let exec = *self.captured_graph_exec.borrow();
        match exec {
            Some(exec) => {
                let s = self.cuda_stream.cu_stream();
                unsafe {
                    cudarc::driver::sys::cuGraphLaunch(exec, s)
                        .result()
                        .expect("cuGraphLaunch (captured full-step)");
                }
                true
            }
            None => false,
        }
    }

    pub fn has_captured_graph(&self) -> bool {
        self.captured_graph_exec.borrow().is_some()
    }

    /// Resolve pending output pointer registrations into external_output_buffers.
    /// Called at the start of execute(), after buffer allocation and HLIR sync.
    fn apply_output_ptr_registrations(&mut self) {
        // clear stale external output buffers from previous execution
        self.external_output_buffers.clear();

        if self.output_ptr_registrations.is_empty() {
            return;
        }

        // Collect registrations to avoid borrow conflict (drain borrows self mutably,
        // but find_producer_node/follow_aliases need &self).

        let registrations: Vec<_> = self.output_ptr_registrations.drain().collect();

        for (hlir_id, (device_ptr, n_bytes)) in registrations {
            // Resolve HLIR output id -> LLIR producer -> follow aliases -> data node
            let producer = self.find_producer_node(hlir_id);
            let data_node = self.follow_aliases(producer);

            // If data_node is an HLIR input (aliased output), skip — can't substitute
            if self.compiled_buckets[self.active_bucket]
                .llir_to_hlir
                .contains_key(&data_node)
            {
                continue;
            }

            // Create non-owning CudaSlice view of PyTorch's buffer
            let slice = unsafe {
                self.cuda_stream
                    .upgrade_device_ptr::<u8>(device_ptr, n_bytes)
            };

            self.external_output_buffers
                .insert(data_node, std::mem::ManuallyDrop::new(slice));

            // Update cached_buffer_ptrs so CudaGraphOp picks up the new pointer
            self.compiled_buckets[self.active_bucket]
                .cached_buffer_ptrs
                .insert(data_node, device_ptr);
        }
    }

    /// Dtype of the buffer backing output tensor `id`, if the compiled bucket
    /// recorded a spec for it. Mirrors the spec lookup `get_output_data` uses.
    fn output_dtype(&self, id: impl ToId) -> Option<DType> {
        let data_id = self.resolve_data_node(id);
        let bucket = self.active();
        bucket
            .buffer_specs
            .get(&data_id)
            .map(|spec| spec.dtype)
            // Passed-through Inputs (e.g. residual carries) have no buffer_spec;
            // fall back to the recorded Input dtype so the read widens correctly.
            .or_else(|| bucket.input_dtypes.get(&data_id).copied())
    }

    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let id = id.to_id();
        // bf16/f16 outputs must be *widened* to f32, not byte-reinterpreted: a
        // raw `as *mut f32` cast halves the element count (2 bytes -> 4) and
        // turns pairs of half-precision values into garbage f32s. (This was the
        // root cause of 32000-vocab logits read as 16000 and 4096-hidden
        // activations read as 2048 -> incoherent decode + parity shape mismatch.)
        let dtype = self.output_dtype(id);
        let bytes = self.get_output_data(id);
        match dtype {
            Some(DType::Bf16) => bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_ne_bytes([c[0], c[1]])).to_f32())
                .collect(),
            Some(DType::F16) => bytes
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_ne_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => {
                let n = bytes.len() / 4;
                let cap = bytes.capacity() / 4;
                let ptr = bytes.as_ptr() as *mut f32;
                std::mem::forget(bytes);
                unsafe { Vec::from_raw_parts(ptr, n, cap) }
            }
        }
    }

    /// Take a GPU buffer handle for an output tensor. This removes the buffer from
    /// the runtime, so the caller owns it. Use `set_buffer` to give it back.
    ///
    /// Uses `output_data_input` to trace data lineage back to the originating HLIR
    /// input buffer. This ensures `remove_buffer` always extracts from `hlir_buffers`
    /// (never from intermediate `self.buffers`), keeping intermediate allocations intact.
    ///
    /// For in-place ops (output_aliases_input), the output IS the HLIR buffer — simply
    /// remove and return it. For copy-then-modify ops (like Scatter), the output data
    /// lives in an intermediate buffer while the HLIR buffer has stale data — swap them
    /// so the caller gets the updated data and the intermediate slot stays allocated.
    pub fn remove_buffer(&mut self, id: impl ToId) -> CudaSlice<u8> {
        let producer = self.find_producer_node(id);
        let alias_node = self.follow_aliases(producer);
        let lineage_node = self.follow_data_lineage(producer);
        let bi = self.active_bucket;

        // If aliases and lineage agree, data is in-place — just remove the HLIR buffer.
        // If they differ, data is in an intermediate buffer (copy-then-modify) — swap.
        if alias_node == lineage_node {
            // In-place or direct HLIR: remove and return
            let hlir_node = self.compiled_buckets[bi]
                .llir_to_hlir
                .get(&lineage_node)
                .copied();
            if let Some(hlir_node) = hlir_node {
                match self
                    .hlir_buffers
                    .remove(&hlir_node)
                    .expect("Cannot find input tensor in runtime!")
                {
                    CudaInput::Buffer(buf) => buf,
                    CudaInput::Ptr(p) => panic!("Cannot take raw pointer input (ptr=0x{:x})", p),
                }
            } else {
                let src = Self::bucket_buffer(
                    &self.compiled_buckets[bi],
                    &self.cuda_stream,
                    &lineage_node,
                )
                .expect("Cannot find tensor in runtime!");
                Self::copy_device_buffer_to_new_slice(&self.cuda_stream, src)
            }
        } else {
            // Copy-then-modify: output data is in alias_node's buffer (intermediate),
            // while the lineage HLIR buffer has stale pre-op data. Return an owned
            // copy of the arena output and drop the stale HLIR buffer.
            let hlir_node = *self.compiled_buckets[bi]
                .llir_to_hlir
                .get(&lineage_node)
                .expect("output_data_input lineage must reach an HLIR input node");

            let output =
                Self::bucket_buffer(&self.compiled_buckets[bi], &self.cuda_stream, &alias_node)
                    .expect("Cannot find intermediate output buffer in runtime!");
            let output_buf = Self::copy_device_buffer_to_new_slice(&self.cuda_stream, output);

            match self
                .hlir_buffers
                .remove(&hlir_node)
                .expect("Cannot find HLIR input buffer in runtime!")
            {
                CudaInput::Buffer(_buf) => {}
                CudaInput::Ptr(p) => panic!("Cannot take raw pointer input (ptr=0x{:x})", p),
            }

            // Return the output buffer (has correct data)
            output_buf
        }
    }

    /// Set a GPU buffer handle as input data for a node. This is a zero-copy operation
    /// (just a pointer swap, no GPU memcpy).
    pub fn set_buffer(&mut self, id: impl ToId, buf: CudaSlice<u8>) {
        let id = id.to_id();
        self.hlir_buffers.insert(id, CudaInput::Buffer(buf));
        self.changed_hlir.insert(id);
    }

    pub fn get_bool(&self, id: impl ToId) -> Vec<bool> {
        self.get_output_data(id)
            .into_iter()
            .map(|b| b != 0)
            .collect()
    }

    pub fn get_i32(&self, id: impl ToId) -> Vec<i32> {
        self.get_output_data(id)
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect_vec()
    }

    pub fn get_f16(&self, id: impl ToId) -> Vec<f16> {
        let bytes = self.get_output_data(id);
        let n = bytes.len() / 2;
        let cap = bytes.capacity() / 2;
        let ptr = bytes.as_ptr() as *mut f16;
        std::mem::forget(bytes);
        unsafe { Vec::from_raw_parts(ptr, n, cap) }
    }

    pub fn get_bf16(&self, id: impl ToId) -> Vec<bf16> {
        let bytes = self.get_output_data(id);
        let n = bytes.len() / 2;
        let cap = bytes.capacity() / 2;
        let ptr = bytes.as_ptr() as *mut bf16;
        std::mem::forget(bytes);
        unsafe { Vec::from_raw_parts(ptr, n, cap) }
    }

    /// Swap the GPU buffer of an output tensor into the input slot for another tensor.
    /// This is a zero-copy operation (just pointer swaps, no GPU memcpy).
    /// Useful for feeding back output state (like KV caches) as input for the next step.
    pub fn swap_output_to_input(&mut self, output_id: impl ToId, input_id: impl ToId) {
        let output_id = output_id.to_id();
        let input_id = input_id.to_id();
        let bi = self.active_bucket;

        let bucket = &self.compiled_buckets[bi];
        let data_llir_node = *bucket
            .output_producers
            .get(&output_id)
            .expect("Cannot find output node for swap!");

        // Get the LLIR node for the input
        let input_llir_node = *bucket
            .hlir_to_llir
            .get(&input_id)
            .expect("Cannot find input in LLIR mapping!");

        let src = Self::bucket_buffer(
            &self.compiled_buckets[bi],
            &self.cuda_stream,
            &data_llir_node,
        )
        .expect("Output not in intermediate buffers");
        let input_buf = Self::copy_device_buffer_to_new_slice(&self.cuda_stream, src);
        self.hlir_buffers
            .insert(input_id, CudaInput::Buffer(input_buf));
        self.changed_hlir.insert(input_id);

        // Update cached pointer for the input
        let ptr = match &self.hlir_buffers[&input_id] {
            CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
            CudaInput::Ptr(p) => *p,
        };
        self.compiled_buckets[bi]
            .cached_buffer_ptrs
            .insert(input_llir_node, ptr);
    }

    /// Free all intermediate buffers to reclaim GPU memory.
    /// They will be re-allocated on the next `execute()` call.
    pub fn free_intermediate_buffers(&mut self) {
        for bucket in &mut self.compiled_buckets {
            bucket.arena = None;
            bucket.cached_buffer_ptrs.clear();
        }
    }

    #[tracing::instrument(skip_all)]
    fn allocate_intermediate_buffers(
        bucket: &mut CompiledBucket,
        stream: &Arc<CudaStream>,
        dyn_dims: &FxHashMap<char, usize>,
    ) {
        let needs_new_plan = !Self::buffer_plan_matches(bucket, dyn_dims);
        if needs_new_plan {
            if bucket.arena.is_some() {
                stream.synchronize().unwrap();
            }
            Self::plan_intermediate_buffers(bucket, dyn_dims);
        }

        if bucket.arena_bytes == 0 {
            bucket.arena = None;
            bucket.cached_buffer_ptrs.clear();
            return;
        }

        if bucket
            .arena
            .as_ref()
            .is_none_or(|arena| arena.len() < bucket.arena_bytes)
        {
            bucket.arena = Some(stream.alloc_zeros(bucket.arena_bytes).unwrap());
        }

        let arena_ptr = bucket.arena.as_ref().unwrap().device_ptr(stream).0;
        for (logical_node, &offset) in &bucket.logical_buffer_offsets {
            if let Some(ptr) = arena_ptr.checked_add(offset as u64) {
                bucket.cached_buffer_ptrs.insert(*logical_node, ptr);
            }
        }
    }

    fn buffer_plan_matches(bucket: &CompiledBucket, dyn_dims: &FxHashMap<char, usize>) -> bool {
        if bucket.buffer_specs.is_empty() {
            return true;
        }
        if bucket.logical_buffer_offsets.is_empty() && !bucket.buffer_specs.is_empty() {
            return false;
        }
        bucket
            .intermediate_buffer_dims
            .iter()
            .all(|dim| bucket.last_dyn_map.get(dim) == dyn_dims.get(dim))
    }

    fn plan_intermediate_buffers(bucket: &mut CompiledBucket, dyn_dims: &FxHashMap<char, usize>) {
        bucket.logical_buffer_offsets.clear();
        bucket.logical_buffer_bytes.clear();
        bucket.arena_bytes = 0;
        bucket.intermediate_buffer_dims.clear();
        bucket.cached_buffer_ptrs.clear();
        bucket.last_dyn_map = dyn_dims.clone();

        let mut logical_bytes = FxHashMap::default();
        for (node, spec) in &bucket.buffer_specs {
            bucket
                .intermediate_buffer_dims
                .extend(spec.bytes.dyn_vars());
            let bytes = spec.bytes.exec(dyn_dims).unwrap_or_else(|| {
                panic!(
                    "buffer byte-size {:?} for node {:?} has unresolved dims {:?}; bound dims={:?}",
                    spec.bytes,
                    node,
                    spec.bytes.dyn_vars(),
                    dyn_dims
                )
            });
            if bytes > 0 {
                logical_bytes.insert(*node, bytes);
            }
        }

        if logical_bytes.is_empty() {
            bucket.arena = None;
            return;
        }
        let total_spec_count = logical_bytes.len();
        let total_spec_bytes = logical_bytes.values().copied().sum::<usize>();

        let mut first_use: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut last_use: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let exec_order = toposort(&bucket.exec_graph, None).unwrap_or_default();
        let output_alias_map = bucket.output_alias_map.clone();

        let mut touch = |node: NodeIndex, step: usize| {
            let Some(node) = resolve_logical_buffer_node(node, &logical_bytes, &output_alias_map)
            else {
                return;
            };
            first_use
                .entry(node)
                .and_modify(|first| *first = (*first).min(step))
                .or_insert(step);
            last_use
                .entry(node)
                .and_modify(|last| *last = (*last).max(step))
                .or_insert(step);
        };

        let mut time = 0usize;
        for exec_node in exec_order.iter().copied() {
            let exec_op = &bucket.exec_graph[exec_node];
            let precise_extra_lifetimes = exec_op.internal.extra_buffer_lifetimes();
            let span = precise_extra_lifetimes
                .as_ref()
                .and_then(|lifetimes| lifetimes.iter().map(|(_, _, end)| *end).max())
                .map(|end| end + 1)
                .unwrap_or(1)
                .max(1);
            let start_time = time;
            let end_time = time + span - 1;
            time += span;

            let precise_nodes = precise_extra_lifetimes
                .as_ref()
                .map(|lifetimes| {
                    lifetimes
                        .iter()
                        .filter_map(|(node, _, _)| {
                            resolve_logical_buffer_node(*node, &logical_bytes, &output_alias_map)
                        })
                        .collect::<FxHashSet<_>>()
                })
                .unwrap_or_default();

            let mut touch_if_not_precise = |node: NodeIndex, step: usize| {
                if resolve_logical_buffer_node(node, &logical_bytes, &output_alias_map)
                    .is_some_and(|node| precise_nodes.contains(&node))
                {
                    return;
                }
                touch(node, step);
            };

            touch_if_not_precise(exec_op.output, start_time);
            touch_if_not_precise(exec_op.output, end_time);
            for &input in &exec_op.inputs {
                touch_if_not_precise(input, start_time);
                touch_if_not_precise(input, end_time);
            }

            if let Some(lifetimes) = precise_extra_lifetimes {
                for (node, start, end) in lifetimes {
                    touch(node, start_time + start);
                    touch(node, start_time + end);
                }
            } else {
                for extra_node in exec_op.internal.extra_buffer_nodes() {
                    touch(extra_node, start_time);
                    touch(extra_node, end_time);
                }
            }
        }

        for &producer in bucket.output_producers.values() {
            let mut alias_node = producer;
            while let Some(target) = bucket.output_alias_map.get(&alias_node) {
                alias_node = *target;
            }
            touch(alias_node, time);

            let mut data_node = producer;
            while let Some(target) = bucket.output_data_map.get(&data_node) {
                data_node = *target;
            }
            touch(data_node, time);
            touch(producer, time);
        }

        let mut planned = logical_bytes
            .into_iter()
            .filter(|(node, _)| first_use.contains_key(node) || last_use.contains_key(node))
            .map(|(node, bytes)| PlannedBuffer {
                node,
                bytes,
                start: first_use.get(&node).copied().unwrap_or(0),
                end: last_use.get(&node).copied().unwrap_or(0),
            })
            .collect_vec();
        planned.sort_by_key(|buf| (buf.start, std::cmp::Reverse(buf.bytes), buf.node.index()));
        let planned_logical_count = planned.len();
        let planned_logical_bytes = planned.iter().map(|buf| buf.bytes).sum::<usize>();
        let logical_peak = logical_interval_peak(&planned);

        let mut arena_end = 0usize;
        let mut placed: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(planned.len());
        let mut placement_order = planned.iter().collect_vec();
        placement_order.sort_by_key(|buf| {
            (
                std::cmp::Reverse(buf.bytes),
                std::cmp::Reverse(buf.end.saturating_sub(buf.start)),
                buf.start,
                buf.node.index(),
            )
        });

        for buf in placement_order {
            let allocation_bytes = align_up(buf.bytes, ARENA_ALIGNMENT);
            let mut candidates = vec![0usize];
            for &(placed_start, placed_end, placed_offset, placed_bytes) in &placed {
                if intervals_overlap(buf.start, buf.end, placed_start, placed_end) {
                    candidates.push(align_up(placed_offset + placed_bytes, ARENA_ALIGNMENT));
                }
            }
            candidates.sort_unstable();
            candidates.dedup();

            let offset = candidates
                .into_iter()
                .find(|&candidate| {
                    placed
                        .iter()
                        .all(|&(placed_start, placed_end, placed_offset, placed_bytes)| {
                            !intervals_overlap(buf.start, buf.end, placed_start, placed_end)
                                || !byte_ranges_overlap(
                                    candidate,
                                    allocation_bytes,
                                    placed_offset,
                                    placed_bytes,
                                )
                        })
                })
                .unwrap_or_else(|| {
                    placed
                        .iter()
                        .filter(|(placed_start, placed_end, _, _)| {
                            intervals_overlap(buf.start, buf.end, *placed_start, *placed_end)
                        })
                        .map(|(_, _, offset, bytes)| align_up(offset + bytes, ARENA_ALIGNMENT))
                        .max()
                        .unwrap_or(0)
                });

            bucket.logical_buffer_offsets.insert(buf.node, offset);
            bucket.logical_buffer_bytes.insert(buf.node, buf.bytes);
            placed.push((buf.start, buf.end, offset, allocation_bytes));
            arena_end = arena_end.max(offset + allocation_bytes);
        }
        bucket.arena_bytes = arena_end;

        if std::env::var_os("LUMINAL_CUDA_MEMORY_DEBUG").is_some() {
            eprintln!(
                "   CUDA memory plan specs={total_spec_count} used={planned_logical_count} skipped={} spec_bytes={} used_bytes={} skipped_bytes={} logical_peak={} arena_plan={} allocations={}",
                total_spec_count.saturating_sub(planned_logical_count),
                total_spec_bytes,
                planned_logical_bytes,
                total_spec_bytes.saturating_sub(planned_logical_bytes),
                logical_peak,
                bucket.arena_bytes,
                bucket.logical_buffer_offsets.len(),
            );
        }
    }

    /// Pre-allocate buffers with the given dynamic dimension values.
    /// CUDA graph building is handled internally by CudaGraphOp on first execution.
    #[tracing::instrument(skip_all)]
    pub fn prebuild_graphs(&mut self, dyn_map: &FxHashMap<char, usize>) {
        let bucket = &mut self.compiled_buckets[self.active_bucket];
        // 1. Allocate intermediate buffers (needed for buffer pointers)
        Self::allocate_intermediate_buffers(bucket, &self.cuda_stream, dyn_map);

        // 2. Process changed HLIR inputs to get their buffer pointers
        if !self.changed_hlir.is_empty() || !bucket.hlir_synced {
            let to_process: Vec<(NodeIndex, NodeIndex, u64)> = self
                .changed_hlir
                .iter()
                .chain(
                    // On first sync for this bucket, process ALL hlir keys
                    if !bucket.hlir_synced {
                        self.hlir_buffers.keys().collect::<Vec<_>>()
                    } else {
                        vec![]
                    }
                    .into_iter(),
                )
                .filter_map(|hlir_node| {
                    let llir_node = bucket.hlir_to_llir.get(hlir_node)?;
                    let input = self.hlir_buffers.get(hlir_node)?;
                    let ptr = match input {
                        CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
                        CudaInput::Ptr(p) => *p,
                    };
                    Some((*hlir_node, *llir_node, ptr))
                })
                .collect();

            for (_hlir_node, llir_node, ptr) in to_process {
                bucket.cached_buffer_ptrs.insert(llir_node, ptr);
            }
            bucket.hlir_synced = true;
            // Only clear changed_hlir if there's a single bucket
            // (multi-bucket: other buckets may still need these changes)
            if self.compiled_buckets.len() == 1 {
                self.changed_hlir.clear();
            }
        }

        // CUDA graph building is now handled internally by CudaGraphOp on first execution
    }
}

pub trait ToCudaInput {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput;
    /// Borrow the host bytes backing this input as a raw little-endian `&[u8]`.
    /// Used by `set_data` for in-place buffer reuse (A3): when a same-size device
    /// buffer already exists for an id, its contents are overwritten from these
    /// bytes rather than allocating a fresh buffer, keeping the device pointer
    /// stable across executes (required for captured-graph replay).
    fn as_host_bytes(&self) -> &[u8];
}

impl ToCudaInput for &[f32] {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(self.as_host_bytes()).unwrap())
    }
    fn as_host_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4) }
    }
}

impl ToCudaInput for Vec<i32> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(self.as_host_bytes()).unwrap())
    }
    fn as_host_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4) }
    }
}

impl ToCudaInput for Vec<f32> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(self.as_host_bytes()).unwrap())
    }
    fn as_host_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4) }
    }
}

impl ToCudaInput for Vec<f16> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(self.as_host_bytes()).unwrap())
    }
    fn as_host_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 2) }
    }
}

impl ToCudaInput for Vec<bf16> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(self.as_host_bytes()).unwrap())
    }
    fn as_host_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 2) }
    }
}

impl ToCudaInput for &[u8] {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(self).unwrap())
    }
    fn as_host_bytes(&self) -> &[u8] {
        self
    }
}

impl ToCudaInput for Vec<u8> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(&self).unwrap())
    }
    fn as_host_bytes(&self) -> &[u8] {
        self.as_slice()
    }
}

fn format_duration_precise(d: &std::time::Duration) -> String {
    let us = d.as_micros();
    if us >= 1000 {
        format!("{} ms {} µs", us / 1000, us % 1000)
    } else {
        format!("{} µs", us)
    }
}

fn resolve_logical_buffer_node(
    mut node: NodeIndex,
    logical_bytes: &FxHashMap<NodeIndex, usize>,
    output_alias_map: &FxHashMap<NodeIndex, NodeIndex>,
) -> Option<NodeIndex> {
    let mut visited = FxHashSet::default();
    while !logical_bytes.contains_key(&node) {
        if !visited.insert(node) {
            return None;
        }
        let target = output_alias_map.get(&node)?;
        node = *target;
    }

    Some(node)
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

fn intervals_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start <= b_end && b_start <= a_end
}

fn byte_ranges_overlap(a_offset: usize, a_bytes: usize, b_offset: usize, b_bytes: usize) -> bool {
    a_offset < b_offset + b_bytes && b_offset < a_offset + a_bytes
}

fn is_schedule_only_host_source(llir_graph: &LLIRGraph, source: NodeIndex) -> bool {
    llir_graph[source]
        .to_dialect::<dyn HostOp>()
        .is_some_and(|source_host_op| source_host_op.output_bytes() == 0)
}

fn host_data_inputs(
    llir_graph: &LLIRGraph,
    host_op_node_index: NodeIndex,
    host_op: &dyn HostOp,
) -> Vec<NodeIndex> {
    llir_graph
        .edges_directed(host_op_node_index, Direction::Incoming)
        .sorted_by_key(|e| e.id())
        // CudaGraphOp -> HostOp edges are ordering edges added by kernel_to_host.
        // They must remain in exec_graph, but they are not data pointers.
        .filter(|e| !is_schedule_only_host_source(llir_graph, e.source()))
        .map(|e| e.source())
        .take(host_op.n_inputs())
        .collect_vec()
}

fn logical_interval_peak(planned: &[PlannedBuffer]) -> usize {
    let mut events = Vec::with_capacity(planned.len() * 2);
    for buf in planned {
        events.push((buf.start, buf.bytes as i128));
        events.push((buf.end.saturating_add(1), -(buf.bytes as i128)));
    }
    events.sort_by_key(|(step, delta)| (*step, *delta));

    let mut current = 0i128;
    let mut peak = 0i128;
    for (_, delta) in events {
        current += delta;
        peak = peak.max(current);
    }

    peak.max(0) as usize
}

impl Runtime for CudaRuntime {
    type Ops = (crate::kernel::Ops, crate::host::Ops);
    type CompileArg = Arc<CudaStream>;
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn late_egglog_passes(
        ops: &[Arc<Box<dyn luminal::op::EgglogOp>>],
        options: &luminal::graph::BuildSearchSpaceOptions,
        dyn_map: &FxHashMap<char, usize>,
    ) -> Vec<luminal::egglog_utils::LateEgglogPass> {
        vec![crate::memory_analysis::cuda_memory_analysis_pass(
            ops,
            options.max_memory_bytes,
            dyn_map,
        )]
    }

    fn estimate_graph_memory<'a>(
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        choices: &luminal::egglog_utils::EGraphChoiceSet<'a>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> Option<usize> {
        crate::memory_analysis::estimate_graph_memory_bytes(egraph, choices, dyn_map)
    }

    fn initialize(stream: Self::CompileArg) -> Self {
        Self {
            hlir_buffers: FxHashMap::default(),
            persistent_hlir_inputs: FxHashSet::default(),
            cuda_stream: stream,
            changed_hlir: FxHashSet::default(),
            cuda_graph_timings: vec![],
            last_kernel_stats: vec![],
            last_total_time_us: 0.0,
            kernel_cache: FxHashMap::default(),
            profiling: false,
            compiled_buckets: vec![CompiledBucket::new()],
            active_bucket: 0,
            dim_buckets: FxHashMap::default(),
            output_ptr_registrations: FxHashMap::default(),
            external_output_buffers: FxHashMap::default(),
            external_buffers: FxHashMap::default(),
            decode_position: std::cell::RefCell::new(None),
            kv_write_kernel: std::sync::OnceLock::new(),
            kv_write_batched_kernel: std::sync::OnceLock::new(),
            shm_allreduce_kernel: std::sync::OnceLock::new(),
            captured_graph_exec: std::cell::RefCell::new(None),
        }
    }

    fn aggregate_profile_metrics(metrics: &[Self::ProfileMetric]) -> Self::ProfileMetric {
        metrics.iter().copied().sum()
    }

    #[tracing::instrument(skip_all)]
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        // Sync before clearing old data to ensure all operations complete
        let _ = self.cuda_stream.synchronize();

        // Sync after clearing all buffers to ensure CUDA resources are freed
        if let Err(e) = self.cuda_stream.synchronize() {
            let _ = self.cuda_stream.context().bind_to_thread();
            if self.cuda_stream.synchronize().is_err() {
                panic!("CUDA context unrecoverable after sync error: {e}");
            }
        }

        // Rebind CUDA context to thread after cleanup to ensure valid state
        let _ = self.cuda_stream.context().bind_to_thread();

        let bucket = self.compile_bucket(llir_graph);
        self.compiled_buckets = vec![bucket];
        self.active_bucket = 0;
        self.dim_buckets.clear();

        // Mark all HLIR inputs as changed so their pointers get re-cached in execute
        self.changed_hlir.extend(self.hlir_buffers.keys().copied());

        // Prebuild CUDA graphs if we have a previous dyn_map (e.g., from search/profile)
        let bucket = &self.compiled_buckets[0];
        if !bucket.last_dyn_map.is_empty() {
            let dyn_map = bucket.last_dyn_map.clone();
            self.prebuild_graphs(&dyn_map);
        }
    }

    fn allocate_dummy_input(&mut self, node_index: usize, num_bytes: usize) {
        // Boundary scratch buffers are sized in raw bytes and may represent
        // non-float tensors such as gather/scatter indices. Initialize with zero
        // bytes so integer boundaries stay in-range and the raw allocation size
        // matches the requested tensor storage.
        let host_data = vec![0u8; num_bytes];
        let buf = self.cuda_stream.clone_htod(&host_data).unwrap();
        let id = NodeIndex::new(node_index);
        self.hlir_buffers.insert(id, CudaInput::Buffer(buf));
        self.changed_hlir.insert(id);
    }

    fn has_hlir_buffer(&self, node_index: usize) -> bool {
        self.hlir_buffers.contains_key(&NodeIndex::new(node_index))
    }

    fn clear_intermediate_buffers(&mut self) {
        let _ = self.cuda_stream.synchronize();
        for bucket in &mut self.compiled_buckets {
            bucket.arena = None;
            bucket.cached_buffer_ptrs.clear();
        }
    }

    fn intermediate_buffer_bytes(&self) -> usize {
        self.compiled_buckets
            .iter()
            .map(|b| b.arena.as_ref().map(|arena| arena.len()).unwrap_or(0))
            .sum()
    }

    fn planned_intermediate_buffer_bytes(&self) -> Option<usize> {
        self.compiled_buckets
            .get(self.active_bucket)
            .map(|bucket| bucket.arena_bytes)
    }

    fn allocated_intermediate_buffer_bytes(&self) -> Option<usize> {
        self.compiled_buckets
            .get(self.active_bucket)
            .map(|bucket| bucket.arena.as_ref().map(|arena| arena.len()).unwrap_or(0))
    }

    fn has_nan_outputs(&self, _llir_graph: &LLIRGraph, _dyn_map: &FxHashMap<char, usize>) -> bool {
        let _ = self.cuda_stream.synchronize();
        let bucket = self.active();
        let mut checked = FxHashSet::default();
        for producer in bucket.output_producers.values().copied() {
            let mut node_id = producer;
            while let Some(alias_target) = bucket.output_alias_map.get(&node_id) {
                node_id = *alias_target;
            }
            if !checked.insert(node_id) {
                continue;
            }
            let Some(buf) = Self::bucket_buffer(bucket, &self.cuda_stream, &node_id) else {
                continue;
            };
            let n_bytes = buf.len();
            if n_bytes == 0 || n_bytes % 4 != 0 {
                continue;
            }
            // Determine buffer dtype from the compiled buffer metadata.
            // Only check F32 buffers for NaN; integer/bool buffers have no NaN concept
            // and their bit patterns can produce false positives when reinterpreted as f32.
            let is_float = bucket
                .buffer_specs
                .get(&node_id)
                .map(|spec| matches!(spec.dtype, DType::F32))
                .unwrap_or(true);

            if !is_float {
                continue;
            }

            let host_bytes: Vec<u8> = match buf.clone_dtoh(&self.cuda_stream) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let f32_slice: &[f32] = bytemuck::cast_slice(&host_bytes);
            if f32_slice.iter().any(|x| x.is_nan()) {
                return true;
            }
        }
        false
    }

    #[tracing::instrument(skip_all)]
    fn profile(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
        trials: usize,
        timeout: Option<std::time::Duration>,
    ) -> (Self::ProfileMetric, String) {
        // Clear active bucket's arena before loading new LLIR for profiling.
        if !self.compiled_buckets.is_empty() {
            self.active_mut().arena = None;
        }
        self.load_llir(llir_graph);
        self.profiling = true;
        let profile_start = std::time::Instant::now();
        let mut durations = Vec::with_capacity(trials.max(1));
        for _ in 0..trials.max(1) {
            let start = std::time::Instant::now();
            self.execute(dyn_map);
            durations.push(start.elapsed());
            if timeout.is_some_and(|timeout| profile_start.elapsed() >= timeout) {
                break;
            }
        }
        self.profiling = false;
        let duration = durations.iter().sum::<std::time::Duration>() / durations.len() as u32;

        let total_bytes: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_loaded + s.bytes_stored)
            .sum::<usize>();
        let total_flops: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.flops)
            .sum::<usize>();
        let aggregate_bw = if self.last_total_time_us > 0.0 {
            (total_bytes as f64) / (self.last_total_time_us * 1e-6) / 1e9
        } else {
            0.0
        };
        let aggregate_tf = if self.last_total_time_us > 0.0 {
            (total_flops as f64) / (self.last_total_time_us * 1e-6) / 1e12
        } else {
            0.0
        };

        let peak_bw = crate::cuda_bandwidth_gbps(self.cuda_stream.context());
        let peak_tf = crate::cuda_compute_f32_tflops(self.cuda_stream.context());
        let mbu = peak_bw.map(|p| aggregate_bw / p as f64);
        let mfu = peak_tf.map(|p| aggregate_tf / p as f64);

        let duration_str = format_duration_precise(&duration);
        let mbu_str = mbu.map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0));
        let mfu_str = mfu.map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0));
        let display = format!(
            "{duration_str} | MBU: {mbu_str} | MFU: {mfu_str} [KRN: {} HOST: {}]",
            llir_graph
                .node_weights()
                .filter(|n| n.to_dialect::<dyn KernelOp>().is_some())
                .count(),
            llir_graph
                .node_weights()
                .filter(|n| n.to_dialect::<dyn HostOp>().is_some())
                .count()
        );

        (duration, display)
    }

    /// Execute the compiled graph. Does NOT synchronize the stream on exit.
    /// Callers that read outputs to host (via clone_dtoh / get_data_f32) get
    /// an implicit sync there. Callers that consume outputs device-resident
    /// (zero-copy handoff to the next segment on the same stream) rely on
    /// stream ordering and must not require host-visible completion.
    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        // Dispatch to correct bucket if multi-bucket mode
        if self.compiled_buckets.len() > 1 {
            let idx = self.resolve_bucket(dyn_map);
            if idx != self.active_bucket {
                // Free the old bucket's intermediates to avoid holding 2 full sets in GPU memory
                let old = self.active_bucket;
                self.compiled_buckets[old].arena = None;
                self.compiled_buckets[old].cached_buffer_ptrs.clear();
                self.active_bucket = idx;
                // Mark bucket as needing HLIR sync since it may have missed changes
                self.compiled_buckets[idx].hlir_synced = false;
            }
        }

        let bucket = &mut self.compiled_buckets[self.active_bucket];
        Self::allocate_intermediate_buffers(bucket, &self.cuda_stream, dyn_map);
        // Cache HLIR input pointers
        if !self.changed_hlir.is_empty() || !bucket.hlir_synced {
            let hlir_nodes: Vec<NodeIndex> = if !bucket.hlir_synced {
                // First time this bucket is active since HLIR changed — sync all
                self.hlir_buffers.keys().copied().collect()
            } else {
                self.changed_hlir.iter().copied().collect()
            };
            for hlir_node in hlir_nodes {
                let Some(&llir_node) = bucket.hlir_to_llir.get(&hlir_node) else {
                    continue;
                };
                let Some(input) = self.hlir_buffers.get(&hlir_node) else {
                    continue;
                };
                let ptr = match input {
                    CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
                    CudaInput::Ptr(p) => *p,
                };
                bucket.cached_buffer_ptrs.insert(llir_node, ptr);
            }
            bucket.hlir_synced = true;
            // Only clear changed_hlir if single bucket (multi-bucket: others may need it)
            if self.compiled_buckets.len() == 1 {
                self.changed_hlir.clear();
            }
        }
        // Ensure all CUDA graphs are built (handles first execute and any missing graphs)
        self.prebuild_graphs(dyn_map);

        // Resolve external output pointer registrations (zero-copy output path)
        self.apply_output_ptr_registrations();

        // Cache the toposort order once (exec_graph is fixed after build) so the
        // per-token hot path doesn't re-run `toposort` (which allocates) for
        // every segment, every step.
        if self.compiled_buckets[self.active_bucket].exec_order.is_empty() {
            let order = toposort(&self.compiled_buckets[self.active_bucket].exec_graph, None)
                .expect("exec_graph has a cycle");
            self.compiled_buckets[self.active_bucket].exec_order = order;
        }

        let total_start = std::time::Instant::now();
        let bucket = &self.compiled_buckets[self.active_bucket];

        // SKEIN_OP_PROFILE (non-captured only): attribute the decode step across op
        // types by recording a CUDA timing event between every op and accumulating
        // GPU span per op-label into a process-global table (logged every 64 steps).
        // Disabled under SKEIN_CAPTURE (the step collapses to one fused-graph op, and
        // events are illegal inside a capturing stream anyway).
        static OP_PROF_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let op_prof_on = *OP_PROF_ON.get_or_init(|| {
            std::env::var_os("SKEIN_OP_PROFILE").is_some()
                && std::env::var_os("SKEIN_CAPTURE").is_none()
        });
        let mut prof_ev = Vec::new();
        let mut prof_lbl: Vec<String> = Vec::new();
        if op_prof_on {
            let f = Some(crate::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
            let ev = self.cuda_stream.context().new_event(f).unwrap();
            ev.record(&self.cuda_stream).unwrap();
            prof_ev.push(ev);
        }

        // Reused across nodes to avoid a per-node hashmap allocation on the hot path.
        let mut buffer_map: FxHashMap<NodeIndex, DeviceBuffer> = FxHashMap::default();
        for &exec_node in &bucket.exec_order {
            let exec_op = &bucket.exec_graph[exec_node];
            trace!("Executing: {:?}", exec_op);

            buffer_map.clear();

            if let Some(buf) = Self::resolve_runtime_buffer(
                bucket,
                &self.cuda_stream,
                &self.hlir_buffers,
                &self.external_buffers,
                &self.external_output_buffers,
                exec_op.output,
            ) {
                buffer_map.insert(exec_op.output, buf);
            }

            for &inp in &exec_op.inputs {
                if let Some(buf) = Self::resolve_runtime_buffer(
                    bucket,
                    &self.cuda_stream,
                    &self.hlir_buffers,
                    &self.external_buffers,
                    &self.external_output_buffers,
                    inp,
                ) {
                    buffer_map.insert(inp, buf);
                }
            }

            let extra_nodes = exec_op.internal.extra_buffer_nodes();
            for extra_node in extra_nodes {
                if let Entry::Vacant(e) = buffer_map.entry(extra_node)
                    && let Some(buf) = Self::resolve_runtime_buffer(
                        bucket,
                        &self.cuda_stream,
                        &self.hlir_buffers,
                        &self.external_buffers,
                        &self.external_output_buffers,
                        extra_node,
                    )
                {
                    e.insert(buf);
                }
            }
            let _span = span!(
                Level::TRACE,
                "host_op_execute",
                n_inputs = exec_op.inputs.len()
            )
            .entered();
            exec_op
                .internal
                .execute(
                    &exec_op.stream,
                    exec_op.output,
                    &exec_op.inputs,
                    &buffer_map,
                    dyn_map,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "CUDA execute error in {:?}: {e}",
                        exec_op.internal.stats_name().unwrap_or("unknown")
                    );
                });
            if op_prof_on {
                let lbl = match exec_op.internal.stats_name() {
                    Some(n) => n.to_string(),
                    None => {
                        // First identifier of the op's Debug repr (its type tag).
                        let d = format!("{:?}", exec_op.internal);
                        d.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                            .find(|s| !s.is_empty())
                            .unwrap_or("op")
                            .to_string()
                    }
                };
                let f = Some(crate::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                let ev = self.cuda_stream.context().new_event(f).unwrap();
                ev.record(&self.cuda_stream).unwrap();
                prof_ev.push(ev);
                prof_lbl.push(lbl);
            }
        }
        if op_prof_on && prof_ev.len() >= 2 {
            prof_ev.last().unwrap().synchronize().unwrap();
            static OP_PROF: std::sync::OnceLock<
                std::sync::Mutex<(std::collections::BTreeMap<String, f64>, u64)>,
            > = std::sync::OnceLock::new();
            let mut g = OP_PROF
                .get_or_init(|| std::sync::Mutex::new((std::collections::BTreeMap::new(), 0)))
                .lock()
                .unwrap();
            for i in 0..prof_lbl.len() {
                let us = prof_ev[i].elapsed_ms(&prof_ev[i + 1]).unwrap() as f64 * 1000.0;
                *g.0.entry(prof_lbl[i].clone()).or_insert(0.0) += us;
            }
            g.1 += 1;
            let steps = g.1;
            if steps % 64 == 0 {
                let total: f64 = g.0.values().sum();
                let mut rows: Vec<(&String, &f64)> = g.0.iter().collect();
                rows.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                eprintln!(
                    "=== SKEIN_OP_PROFILE steps={steps} GPU-busy={:.1}us/token (sum of op spans, uncaptured) ===",
                    total / steps as f64
                );
                for (lbl, us) in rows.iter().take(20) {
                    eprintln!(
                        "  {:<30} {:9.2} us/tok  {:5.1}%",
                        lbl,
                        *us / steps as f64,
                        *us / total * 100.0
                    );
                }
            }
        }
        self.last_total_time_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;

        // Populate last_kernel_stats from HostOps that report stats
        self.last_kernel_stats.clear();
        let bucket = &self.compiled_buckets[self.active_bucket];
        for exec_node in bucket.exec_graph.node_indices() {
            let exec_op = &bucket.exec_graph[exec_node];
            if let Some(name) = exec_op.internal.stats_name() {
                self.last_kernel_stats.push(KernelStats {
                    name,
                    execution_time_us: 0.0,
                    bytes_loaded: 0,
                    bytes_stored: 0,
                    flops: 0,
                    bandwidth_gbps: 0.0,
                    tflops: 0.0,
                });
            }
        }

        // Consume input buffers
        if self.profiling {
            return;
        }
        // SKEIN_CAPTURE: never free input buffers. The full-step graph bakes each
        // input's device pointer at capture time; freeing+reallocating an input
        // between steps (the normal consume behavior) would leave the replayed
        // graph reading a stale/reused address — garbage. set_data overwrites the
        // same-size buffer in place, so keeping them resident does NOT accumulate.
        {
            static CAP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *CAP.get_or_init(|| std::env::var_os("SKEIN_CAPTURE").is_some()) {
                return;
            }
        }
        let bucket = &self.compiled_buckets[self.active_bucket];
        let mut inputs_with_outputs = bucket.preserved_hlir_inputs.clone();

        // For multi-bucket: also preserve inputs needed by other buckets
        if self.compiled_buckets.len() > 1 {
            for (i, other_bucket) in self.compiled_buckets.iter().enumerate() {
                if i == self.active_bucket {
                    continue;
                }
                // Preserve all HLIR nodes that other buckets reference
                inputs_with_outputs.extend(other_bucket.hlir_to_llir.keys());
            }
        }

        let to_consume: Vec<NodeIndex> = self
            .hlir_buffers
            .keys()
            .filter(|hlir_node| {
                !inputs_with_outputs.contains(hlir_node)
                    // Weights marked persistent are uploaded once and kept; not
                    // consumed, so they are not re-uploaded every forward.
                    && !self.persistent_hlir_inputs.contains(hlir_node)
            })
            .copied()
            .collect();

        if std::env::var_os("SKEIN_FI_LOG").is_some() && !to_consume.is_empty() {
            eprintln!(
                "SKEIN_CONSUME removing {} bufs (persistent={}, hlir_total={})",
                to_consume.len(),
                self.persistent_hlir_inputs.len(),
                self.hlir_buffers.len()
            );
        }
        for hlir_node in to_consume {
            self.hlir_buffers.remove(&hlir_node);
            self.external_buffers.remove(&hlir_node);
            let bucket = &mut self.compiled_buckets[self.active_bucket];
            if let Some(llir_node) = bucket.hlir_to_llir.get(&hlir_node) {
                bucket.cached_buffer_ptrs.remove(llir_node);
            }
        }
    }

    fn load_llir_buckets(
        &mut self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) {
        // Sync before clearing old data
        let _ = self.cuda_stream.synchronize();
        let _ = self.cuda_stream.context().bind_to_thread();

        self.dim_buckets = dim_buckets.clone();
        self.compiled_buckets.clear();

        for (bucket_indices, representative_dyn_map, llir) in bucket_llirs {
            let mut bucket = self.compile_bucket(llir);
            bucket.bucket_indices = bucket_indices.clone();
            let _ = representative_dyn_map;
            self.compiled_buckets.push(bucket);
        }
        self.active_bucket = 0;

        // Mark all HLIR inputs as changed so their pointers get re-cached
        self.changed_hlir.extend(self.hlir_buffers.keys().copied());
    }
}

impl CudaRuntime {
    /// Compile a single LLIR graph into a CompiledBucket.
    fn compile_bucket(&mut self, llir_graph: &LLIRGraph) -> CompiledBucket {
        let mut bucket = CompiledBucket::new();
        let mut exec_graph = StableGraph::default();
        let mut node_to_exec = FxHashMap::default();

        // Clone llir_graph so we can modify it
        let mut llir_graph = llir_graph.clone();

        // Compile kernel subgraphs into CudaGraphOps (which implement HostOp)
        crate::kernel::kernel_to_host(&mut llir_graph, &self.cuda_stream, &mut self.kernel_cache);

        // Extract all runtime metadata we used to recover from the lowered LLIR
        // at execution time. After this point the LLIR is compile-time only.
        for node in llir_graph.node_indices() {
            if let Some(Input {
                node: hlir_node,
                dtype,
                ..
            }) = llir_graph[node].to_op::<Input>()
            {
                bucket.llir_to_hlir.insert(node, NodeIndex::new(*hlir_node));
                bucket.hlir_to_llir.insert(NodeIndex::new(*hlir_node), node);
                // Record the input's dtype so a passed-through Input read back to
                // host (residual carries) is widened per its real dtype.
                bucket.input_dtypes.insert(node, *dtype);
                continue;
            }

            if let Some(Output { node: hlir_node }) = llir_graph[node].to_op::<Output>() {
                let producer = llir_graph
                    .neighbors_directed(node, Direction::Incoming)
                    .next()
                    .expect("Output node without producer");
                bucket
                    .output_producers
                    .insert(NodeIndex::new(*hlir_node), producer);
                continue;
            }

            let inputs = || {
                llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .collect_vec()
            };

            if let Some(kernel_op) = llir_graph[node].to_dialect::<dyn KernelOp>() {
                let kernel_name = kernel_op.kernel_name();
                bucket.kernel_names.push(kernel_name);

                // Decide if this node needs a real device buffer.
                //
                // The default assumption is "yes" for ordinary kernel ops
                // (Conv outputs, matmul outputs, etc). FusionStart and
                // Cuda*Elementwise are the exceptions — they're synthetic
                // nodes that the fusion rewrites add inside a region; the
                // megakernel computes them in registers and never writes
                // to memory, so allocating a buffer would just be waste.
                //
                // BUT — and this was the cause of the YOLO crash: if such
                // a node has a *consumer in a different region*, that
                // consumer's CudaGraphOp will look up a device pointer for
                // the producer in the runtime's buffer_map and find none,
                // pass NULL into the kernel, and dereference it →
                // `CUDA_ERROR_ILLEGAL_ADDRESS`. Multi-consumer fan-out is
                // the typical trigger: rule R fuses op X into one region
                // (FusionStart-wrapping it as input), but X is also used by
                // an unrelated downstream op that lives in another region.
                //
                // Safe over-approximation: if the node is a FusionStart /
                // Cuda*Elementwise and *any* of its consumers is a FusionStart
                // (which can only happen when that consumer is the leaf
                // of a different region) or a non-marker op (e.g. an
                // unfused Add/Mul reading the value directly), allocate a
                // buffer so cross-region reads have somewhere to land.
                let is_marker = kernel_name == "FusionStart" || kernel_name.starts_with("Cuda");
                let has_external_consumer = is_marker
                    && llir_graph
                        .neighbors_directed(node, Direction::Outgoing)
                        .any(|consumer| {
                            // A consumer that's a non-kernel op (Output, etc.) always
                            // needs a real buffer; otherwise check the kernel name.
                            match llir_graph[consumer].to_dialect::<dyn KernelOp>() {
                                None => true,
                                Some(ck) => {
                                    let cn = ck.kernel_name();
                                    // FusionEnd is the consumer in the SAME region
                                    // (so it's absorbed). Anything else — including
                                    // another FusionStart, which is by definition the
                                    // leaf of a different region — is external.
                                    cn != "FusionEnd"
                                }
                            }
                        });
                let allocated = kernel_op.output_aliases_input().is_none()
                    && (!is_marker || has_external_consumer);
                if allocated {
                    bucket.buffer_specs.insert(
                        node,
                        BufferSpec {
                            bytes: kernel_op.output_bytes(),
                            dtype: kernel_op.output_dtype(),
                        },
                    );
                }

                if let Some(input_idx) = kernel_op.output_aliases_input()
                    && let Some(target) = inputs().get(input_idx).copied()
                {
                    bucket.output_alias_map.insert(node, target);
                }

                if let Some(input_idx) = kernel_op.output_data_input()
                    && let Some(target) = inputs().get(input_idx).copied()
                {
                    bucket.output_data_map.insert(node, target);
                }
            }

            if let Some(host_op) = llir_graph[node].to_dialect::<dyn HostOp>() {
                bucket.buffer_specs.insert(
                    node,
                    BufferSpec {
                        bytes: host_op.output_bytes(),
                        // Real output dtype (e.g. cublasLt writes its `d_dtype`,
                        // typically bf16) so the host read-back widens correctly
                        // instead of byte-reinterpreting bf16 as f32 (halving it).
                        dtype: host_op.output_dtype(),
                    },
                );
            }
        }

        for producer in bucket.output_producers.values().copied() {
            let mut alias_node = producer;
            while let Some(target) = bucket.output_alias_map.get(&alias_node) {
                alias_node = *target;
            }
            if let Some(hlir_node) = bucket.llir_to_hlir.get(&alias_node) {
                bucket.preserved_hlir_inputs.insert(*hlir_node);
            }

            let mut data_node = producer;
            while let Some(target) = bucket.output_data_map.get(&data_node) {
                data_node = *target;
            }
            if let Some(hlir_node) = bucket.llir_to_hlir.get(&data_node) {
                bucket.preserved_hlir_inputs.insert(*hlir_node);
            }

            if let Some(hlir_node) = bucket.llir_to_hlir.get(&producer) {
                bucket.preserved_hlir_inputs.insert(*hlir_node);
            }
        }

        // Add host ops
        {
            let _span = span!(Level::TRACE, "compile_host_ops").entered();
            for host_op_node_index in llir_graph.node_indices() {
                if let Some(host_op) = llir_graph[host_op_node_index].to_dialect::<dyn HostOp>() {
                    let inputs = host_data_inputs(
                        &llir_graph,
                        host_op_node_index,
                        host_op.as_ref().as_ref(),
                    );
                    node_to_exec.insert(
                        host_op_node_index,
                        exec_graph.add_node(ExecutableHostOp {
                            stream: Arc::clone(&self.cuda_stream),
                            inputs,
                            output: host_op_node_index,
                            internal: Arc::clone(host_op),
                        }),
                    );
                }
            }
        }

        // Add edges
        for edge in llir_graph.edge_indices() {
            let (start, end) = llir_graph.edge_endpoints(edge).unwrap();
            if !node_to_exec.contains_key(&start) || !node_to_exec.contains_key(&end) {
                continue;
            }
            let (exec_start, exec_end) = (node_to_exec[&start], node_to_exec[&end]);
            if exec_start != exec_end
                && exec_graph
                    .edges_connecting(exec_start, exec_end)
                    .next()
                    .is_none()
            {
                exec_graph.add_edge(exec_start, exec_end, ());
            }
        }

        bucket.exec_graph = exec_graph;
        bucket.node_to_exec = node_to_exec;
        bucket.hlir_synced = false;
        bucket
    }

    /// Resolve which bucket matches the current dyn_map values.
    fn resolve_bucket(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.compiled_buckets
            .iter()
            .position(|bucket| {
                self.dim_buckets.iter().all(|(dim, buckets)| {
                    let val = dyn_map.get(dim).copied().unwrap_or(0);
                    let bucket_idx = bucket.bucket_indices.get(dim).copied().unwrap_or(0);
                    buckets
                        .get(bucket_idx)
                        .map(|b| b.contains(val))
                        .unwrap_or(true)
                })
            })
            .unwrap_or_else(|| {
                panic!(
                    "No bucket matches dyn_map {:?}. Defined buckets: {:?}",
                    dyn_map, self.dim_buckets
                )
            })
    }

    /// Print execution statistics for the last execution.
    pub fn print_execution_stats(&self) {
        if self.last_kernel_stats.is_empty() {
            println!("No execution stats available.");
            return;
        }

        // Compute aggregates
        let total_bytes_loaded: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_loaded)
            .sum::<usize>();
        let total_bytes_stored: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_stored)
            .sum::<usize>();
        let total_flops: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.flops)
            .sum::<usize>();
        let total_bytes = total_bytes_loaded + total_bytes_stored;
        let aggregate_bw = if self.last_total_time_us > 0.0 {
            (total_bytes as f64) / (self.last_total_time_us * 1e-6) / 1e9
        } else {
            0.0
        };
        let aggregate_tf = if self.last_total_time_us > 0.0 {
            (total_flops as f64) / (self.last_total_time_us * 1e-6) / 1e12
        } else {
            0.0
        };

        let peak_bw = crate::cuda_bandwidth_gbps(self.cuda_stream.context());
        let peak_tf = crate::cuda_compute_f32_tflops(self.cuda_stream.context());

        // Print kernel stats
        if !self.last_kernel_stats.is_empty() {
            println!("\n=== Kernel Execution Statistics ===\n");
            println!(
                "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
                "Kernel",
                "Time (us)",
                "Loaded",
                "Stored",
                "Agg FLOPS",
                "BW (GB/s)",
                "TFLOPS",
                "MBU",
                "MFU"
            );
            println!("{}", "-".repeat(116));
            for s in &self.last_kernel_stats {
                self.print_stat_row(
                    s.name,
                    s.execution_time_us,
                    None,
                    s.bytes_loaded,
                    s.bytes_stored,
                    s.flops,
                    s.bandwidth_gbps,
                    s.tflops,
                    peak_bw,
                    peak_tf,
                );
            }
            println!("{}", "-".repeat(116));
        }

        // Print aggregate stats
        println!("\n=== Aggregate Statistics ===\n");
        println!(
            "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "", "Time (us)", "Loaded", "Stored", "Agg FLOPS", "BW (GB/s)", "TFLOPS", "MBU", "MFU"
        );
        println!("{}", "-".repeat(116));
        let (mbu, mfu) = match (peak_bw, peak_tf) {
            (Some(pb), Some(pt)) => (
                format!("{:.1}%", aggregate_bw / pb as f64 * 100.0),
                format!("{:.1}%", aggregate_tf / pt as f64 * 100.0),
            ),
            _ => ("-".into(), "-".into()),
        };
        println!(
            "{:<20} {:>12.2} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "Total",
            self.last_total_time_us,
            format_size(total_bytes_loaded),
            format_size(total_bytes_stored),
            format_flops(total_flops),
            format!("{:.2}", aggregate_bw),
            format!("{:.4}", aggregate_tf),
            mbu,
            mfu
        );

        if let (Some(pb), Some(pt)) = (peak_bw, peak_tf) {
            println!("\nDevice peak: {} GB/s bandwidth, {} TFLOPS (F32)", pb, pt);
        }
        println!();
    }

    #[allow(clippy::too_many_arguments)]
    fn print_stat_row(
        &self,
        name: &str,
        time_us: f64,
        count: Option<usize>,
        loaded: usize,
        stored: usize,
        flops: usize,
        bw: f64,
        tf: f64,
        peak_bw: Option<usize>,
        peak_tf: Option<usize>,
    ) {
        let total = loaded + stored;
        let ld = if loaded > 0 {
            format_size(loaded)
        } else {
            "-".into()
        };
        let st = if stored > 0 {
            format_size(stored)
        } else {
            "-".into()
        };
        let fl = if flops > 0 {
            format_flops(flops)
        } else {
            "-".into()
        };
        let bw_s = if total > 0 {
            format!("{bw:.2}")
        } else {
            "-".into()
        };
        let tf_s = if flops > 0 {
            format!("{tf:.4}")
        } else {
            "-".into()
        };
        let mbu = peak_bw
            .filter(|_| total > 0)
            .map(|p| format!("{:.1}%", bw / p as f64 * 100.0))
            .unwrap_or("-".into());
        let mfu = peak_tf
            .filter(|_| flops > 0)
            .map(|p| format!("{:.1}%", tf / p as f64 * 100.0))
            .unwrap_or("-".into());

        match count {
            Some(c) => println!(
                "{name:<20} {time_us:>12.2} {c:>8} {ld:>12} {st:>12} {fl:>12} {bw_s:>12} {tf_s:>12} {mbu:>8} {mfu:>8}"
            ),
            None => println!(
                "{name:<20} {time_us:>12.2} {ld:>12} {st:>12} {fl:>12} {bw_s:>12} {tf_s:>12} {mbu:>8} {mfu:>8}"
            ),
        }
    }

    /// Record GPU timings to an existing perfetto trace file.
    pub fn record_cuda_perfetto_trace(&mut self, mut perfetto_guard: PerfettoGuard) {
        perfetto_guard.stop();
        let data = std::fs::read(&perfetto_guard.path).unwrap();
        let mut trace = luminal_tracing::schema::Trace::decode(data.as_slice()).unwrap();
        let extra_packets = record_cuda_graph_timings(&trace, &self.cuda_graph_timings);
        trace.packet.extend(extra_packets);
        // Sort ALL packets by timestamp for proper Perfetto visualization
        trace.packet.sort_by_key(|p| p.timestamp.unwrap_or(0));
        let mut buf = Vec::with_capacity(trace.encoded_len());
        trace.encode(&mut buf).unwrap();
        std::fs::write(perfetto_guard.path, buf).unwrap();
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1e3)
    } else {
        format!("{} B", bytes)
    }
}

fn format_flops(flops: usize) -> String {
    if flops >= 1_000_000_000_000 {
        format!("{:.2} T", flops as f64 / 1e12)
    } else if flops >= 1_000_000_000 {
        format!("{:.2} G", flops as f64 / 1e9)
    } else if flops >= 1_000_000 {
        format!("{:.2} M", flops as f64 / 1e6)
    } else if flops >= 1_000 {
        format!("{:.2} K", flops as f64 / 1e3)
    } else {
        format!("{}", flops)
    }
}

pub(crate) fn partition_marked_convex<T, E>(
    g: &StableGraph<T, E, Directed>,
    marked: &FxHashSet<NodeIndex>,
) -> Result<Vec<FxHashSet<NodeIndex>>, Cycle<NodeIndex>> {
    if marked.is_empty() {
        return Ok(vec![]);
    }

    // --- Global topo order (also validates DAG) ---
    let topo = toposort(g, None)?;
    let topo_len = topo.len();

    // Map NodeIndex <-> topo position
    let mut idx_to_pos: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut pos_to_idx: Vec<NodeIndex> = Vec::with_capacity(topo_len);
    for (pos, &ni) in topo.iter().enumerate() {
        idx_to_pos.insert(ni, pos);
        pos_to_idx.push(ni);
    }

    // --- Full-graph reachability: reach[upos] contains all vpos reachable from u ---
    // (Bitset DP over topo order)
    let mut reach: Vec<FixedBitSet> = (0..topo_len)
        .map(|_| {
            let mut b = FixedBitSet::with_capacity(topo_len);
            b.grow(topo_len);
            b
        })
        .collect();

    for &u in topo.iter().rev() {
        let upos = idx_to_pos[&u];
        for v in g.neighbors_directed(u, Direction::Outgoing) {
            if let Some(&vpos) = idx_to_pos.get(&v) {
                reach[upos].insert(vpos);
                let rv = reach[vpos].clone();
                reach[upos].union_with(&rv);
            }
        }
    }

    // --- 1) Weakly-connected components in the marked-induced subgraph ---
    let components = marked_weak_components(g, marked);

    let mut results: Vec<FxHashSet<NodeIndex>> = Vec::new();

    for comp in components {
        // Component nodes in topo positions (sorted)
        let mut comp_pos: Vec<usize> = comp
            .iter()
            .filter_map(|ni| idx_to_pos.get(ni).copied())
            .collect();
        comp_pos.sort_unstable();

        // Membership: in_comp_pos bitset over topo positions
        let mut in_comp_pos = FixedBitSet::with_capacity(topo_len);
        in_comp_pos.grow(topo_len);
        for &p in &comp_pos {
            in_comp_pos.insert(p);
        }

        // Membership: in_comp_idx vec over NodeIndex::index() for component-relative DP
        let mut in_comp_idx = vec![false; g.node_bound()];
        for &n in &comp {
            in_comp_idx[n.index()] = true;
        }

        // --- Component-relative "between" witnesses (path-wise, correct) ---
        // has_comp_anc[x] == true if x has a component node as an ancestor (or is in comp)
        let mut has_comp_anc = vec![false; g.node_bound()];
        for &u in &topo {
            let mut v = in_comp_idx[u.index()];
            for p in g.neighbors_directed(u, Direction::Incoming) {
                v |= has_comp_anc[p.index()];
                if v {
                    break;
                }
            }
            has_comp_anc[u.index()] = v;
        }

        // has_comp_des[x] == true if x has a component node as a descendant (or is in comp)
        let mut has_comp_des = vec![false; g.node_bound()];
        for &u in topo.iter().rev() {
            let mut v = in_comp_idx[u.index()];
            for s in g.neighbors_directed(u, Direction::Outgoing) {
                v |= has_comp_des[s.index()];
                if v {
                    break;
                }
            }
            has_comp_des[u.index()] = v;
        }

        // --- Build witness constraints Px/Sx only for true witnesses of THIS component ---
        // Witness x is UNMARKED and lies on some path comp_node ->* x ->* comp_node.
        // For each witness x:
        //   Px(x) = {u in comp | u ->* x}
        //   Sx(x) = {v in comp | x ->* v}
        // A valid block cannot contain nodes from both Px(x) and Sx(x).
        let mut px_map: FxHashMap<NodeIndex, FixedBitSet> = FxHashMap::default();
        let mut sx_map: FxHashMap<NodeIndex, FixedBitSet> = FxHashMap::default();
        let mut px_witnesses: FxHashMap<usize, Vec<NodeIndex>> = FxHashMap::default(); // upos -> witnesses where upos ∈ Px
        let mut sx_witnesses: FxHashMap<usize, Vec<NodeIndex>> = FxHashMap::default(); // vpos -> witnesses where vpos ∈ Sx

        for x in g.node_indices() {
            if marked.contains(&x) {
                continue; // must be outside the block (unmarked) to be a witness
            }
            if !(has_comp_anc[x.index()] && has_comp_des[x.index()]) {
                continue; // not between this component's marked nodes
            }

            let Some(&xpos) = idx_to_pos.get(&x) else {
                continue;
            };
            // Sx = reachable-from-x ∩ component
            let mut sx = reach[xpos].clone();
            sx.intersect_with(&in_comp_pos);
            if sx.is_empty() {
                continue;
            }

            // Px = {u in comp | u can reach x}
            let mut px = FixedBitSet::with_capacity(topo_len);
            px.grow(topo_len);
            for &upos in &comp_pos {
                if reach[upos].contains(xpos) {
                    px.insert(upos);
                }
            }
            if px.is_empty() {
                continue;
            }

            px_map.insert(x, px.clone());
            sx_map.insert(x, sx.clone());

            for upos in px.ones() {
                px_witnesses.entry(upos).or_default().push(x);
            }
            for vpos in sx.ones() {
                sx_witnesses.entry(vpos).or_default().push(x);
            }
        }

        // --- 3) Deterministic topo sweep partition within this component ---
        let mut current: FxHashSet<NodeIndex> = FxHashSet::default();
        let mut block_bits = FixedBitSet::with_capacity(topo_len);
        block_bits.grow(topo_len);

        for &p in &comp_pos {
            let violates = would_violate(
                p,
                &block_bits,
                &px_witnesses,
                &sx_witnesses,
                &px_map,
                &sx_map,
            );

            if violates && !current.is_empty() {
                results.push(std::mem::take(&mut current));
                block_bits.clear(); // keeps length
            }

            let ni = pos_to_idx[p];
            current.insert(ni);
            block_bits.insert(p);
        }

        if !current.is_empty() {
            results.push(current);
        }
    }

    Ok(results)
}

/// Deterministic “contiguous marked” components: weakly-connected in the marked-induced subgraph.
fn marked_weak_components<T, E>(
    g: &StableGraph<T, E, Directed>,
    marked: &FxHashSet<NodeIndex>,
) -> Vec<Vec<NodeIndex>> {
    let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut comps: Vec<Vec<NodeIndex>> = Vec::new();

    for start in g.node_indices() {
        if !marked.contains(&start) || seen.contains(&start) {
            continue;
        }

        let mut q = VecDeque::new();
        q.push_back(start);
        seen.insert(start);

        let mut comp = Vec::new();
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for v in g.neighbors_undirected(u) {
                if marked.contains(&v) && seen.insert(v) {
                    q.push_back(v);
                }
            }
        }
        comps.push(comp);
    }

    comps
}

fn would_violate(
    p: usize,
    block_bits: &FixedBitSet,
    px_witnesses: &FxHashMap<usize, Vec<NodeIndex>>,
    sx_witnesses: &FxHashMap<usize, Vec<NodeIndex>>,
    px_map: &FxHashMap<NodeIndex, FixedBitSet>,
    sx_map: &FxHashMap<NodeIndex, FixedBitSet>,
) -> bool {
    // If p ∈ Px(x), block cannot contain any node in Sx(x)
    if let Some(ws) = px_witnesses.get(&p) {
        for &x in ws {
            if let Some(sx) = sx_map.get(&x)
                && intersects(block_bits, sx)
            {
                return true;
            }
        }
    }

    // If p ∈ Sx(x), block cannot contain any node in Px(x)
    if let Some(ws) = sx_witnesses.get(&p) {
        for &x in ws {
            if let Some(px) = px_map.get(&x)
                && intersects(block_bits, px)
            {
                return true;
            }
        }
    }

    false
}

fn intersects(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let mut tmp = a.clone();
    tmp.intersect_with(b);
    // Note: is_empty() checks if length is 0, not if there are no bits set
    // Use count_ones() to check if there are any set bits after intersection
    tmp.count_ones(..) > 0
}
