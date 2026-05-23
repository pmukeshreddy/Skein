//! `skein_compile` — the bridge between Skein's per-device `LoweredGraph`
//! and Luminal's search-based compiler.
//!
//! The crate is intentionally small: one trait (`ComputeRuntime`) abstracts
//! over Luminal's runtime backends, two impls plug in Native and CUDA, and
//! `compile_with_luminal::<R>` invokes `cx.build_search_space::<R::Inner>()`
//! followed by `cx.search(R::Inner::default(), budget)`.
//!
//! ## Backends
//!
//! - `CudaComputeRuntime` is the production backend: `#[cfg(feature =
//!   "cuda")]`-gated, wrapping `luminal_cuda_lite::CudaRuntime`. It is the
//!   default build target and runs on NVIDIA GPUs.
//! - `NativeComputeRuntime` wraps Luminal's CPU `NativeRuntime`. It builds
//!   unconditionally and is the backend used by CI and by every test that
//!   exercises a real Luminal compile-and-execute path without a GPU.

use std::path::Path;

use luminal::op::Runtime;
use luminal::prelude::{DType, Graph, NativeRuntime, NodeIndex};

pub mod artifact;
pub mod dyn_runtime;
pub mod error;
pub mod executor;
pub mod search_cache;

pub use artifact::{
    ArtifactMetadata, DeclaredTensorMeta, DeviceArtifactLoaded, LoweredSegment, OpRecipe,
    SegmentMetadata, SkeinArtifact,
};
pub use dyn_runtime::{
    DynRuntime, DynRuntimeError, DynRuntimeWrapper, HandoffId, WeightDtype, decode_weight_bytes,
};
pub use error::CompileError;
pub use executor::{
    CollectiveExecutor, DEFAULT_SEARCH_BUDGET, RuntimeSegment, StepOutput, TopologyExecutor,
    TopologyStepBatch, load_device_prefill_segments, load_device_runtime_segments,
    load_native_runtime_segments, load_runtime_segments, segment_input_zero_bytes,
};

/// Backend abstraction. Two concrete impls: `CudaComputeRuntime` (Luminal's
/// `CudaRuntime`, `#[cfg(feature = "cuda")]`, the production backend) and
/// `NativeComputeRuntime` (Luminal's CPU `NativeRuntime`, always available).
/// Additional backends (e.g. Metal, ROCm) drop in as new impls.
pub trait ComputeRuntime: Sized {
    /// Run Luminal's search-based compile against `cx` with the given
    /// budget, then return the resulting runtime ready to execute.
    fn build_and_search(cx: &mut Graph, budget: usize) -> Result<Self, CompileError>;

    /// Like [`build_and_search`](Self::build_and_search), but first stage a
    /// zero-filled buffer for every graph `Input` (weights, handoffs, token
    /// ids). Luminal's search *executes* candidate graphs to measure them, and
    /// the CUDA backend hard-errors on an `Input` that has no buffer (the CPU
    /// backend tolerates it). The real weights/activations are loaded after
    /// compile; these zeros exist only so the search can run. `input_zeros` is
    /// `(node, num_bytes)` for each `Input`. The default ignores them (correct
    /// for the CPU backend); the CUDA backend overrides it.
    fn build_and_search_with_input_zeros(
        cx: &mut Graph,
        budget: usize,
        input_zeros: &[(NodeIndex, usize)],
    ) -> Result<Self, CompileError> {
        let _ = input_zeros;
        Self::build_and_search(cx, budget)
    }

    /// Like [`build_and_search_with_input_zeros`](Self::build_and_search_with_input_zeros),
    /// but served from / written to the on-disk compile cache when `cache_dir`
    /// is `Some`. On a cache hit the segment's recorded egglog search result is
    /// replayed — reproducing the same kernels — instead of re-running egglog +
    /// NVRTC. The default ignores the cache (correct, just not accelerated); the
    /// Native and CUDA backends override it. See [`crate::search_cache`].
    fn build_and_search_cached(
        cx: &mut Graph,
        budget: usize,
        input_zeros: &[(NodeIndex, usize)],
        cache_dir: Option<&Path>,
    ) -> Result<Self, CompileError> {
        let _ = cache_dir;
        Self::build_and_search_with_input_zeros(cx, budget, input_zeros)
    }

    /// Stage `data` into the runtime's buffer for the given input tensor.
    fn set_data_f32(&mut self, id: NodeIndex, data: Vec<f32>);

    /// Stage f32 `data` into an input the graph types as `dtype`, narrowing to
    /// bf16/f16 when needed. Host handoffs/KV are carried as `Vec<f32>`, but the
    /// graph's input buffers are bf16; uploading raw f32 bytes into a bf16 slot
    /// (or reading bf16 back as f32 — see `get_f32`) silently halves and
    /// corrupts the data. The default keeps f32 (correct for the all-f32 CPU
    /// backend); the CUDA backend overrides it.
    fn set_data_f32_as(&mut self, id: NodeIndex, data: Vec<f32>, dtype: DType) {
        let _ = dtype;
        self.set_data_f32(id, data);
    }

    /// Like [`set_data_f32_as`](Self::set_data_f32_as) but marks the input
    /// **persistent** — its GPU buffer is uploaded once and kept across forwards
    /// rather than re-fed every step. Used for weights so they aren't re-uploaded
    /// each forward. Default ignores persistence (the CPU runtime never consumes
    /// input buffers); the CUDA backend overrides it to keep the buffer alive.
    fn set_data_persistent_f32_as(&mut self, id: NodeIndex, data: Vec<f32>, dtype: DType) {
        self.set_data_f32_as(id, data, dtype);
    }

    /// Device pointer of an already-resident input buffer (e.g. a loaded weight),
    /// for sharing it with another graph instead of loading a second copy. CUDA
    /// only — returns None on backends without device pointers.
    fn input_device_ptr(&self, id: NodeIndex) -> Option<u64> {
        let _ = id;
        None
    }

    /// Device buffer `(raw_ptr, byte_len)` backing a computed **output** tensor,
    /// without a host copy — so one segment's output can be handed to the next
    /// segment by device pointer (via [`set_input_device_ptr`]) instead of the
    /// GPU->host->GPU round-trip. CUDA only; `None` on backends without device
    /// pointers or if the tensor has no resident output buffer yet.
    fn output_device_buffer(&self, id: NodeIndex) -> Option<(u64, usize)> {
        let _ = id;
        None
    }

    /// Allocate (once) a persistent, zero-initialized **device** buffer of
    /// `n_bytes` for input `id` and return its device pointer; if `id` already
    /// has a resident buffer, return that (idempotent). Used to hold a KV-cache
    /// input on the GPU across decode steps so it is never re-uploaded from host.
    /// CUDA only; `None` default.
    fn alloc_persistent_input_zeros(&mut self, id: NodeIndex, n_bytes: usize) -> Option<u64> {
        let _ = (id, n_bytes);
        None
    }

    /// Copy output tensor `id`'s data to an external device pointer (device→
    /// device, no host). Used to write a decode step's new K/V into its slot in
    /// the resident KV buffer. CUDA only; no-op default.
    ///
    /// # Safety
    /// `dest_ptr` must be a valid device allocation of at least `n_bytes` on this
    /// runtime's device.
    unsafe fn copy_output_to_device_ptr(&self, id: NodeIndex, dest_ptr: u64, n_bytes: usize) {
        let _ = (id, dest_ptr, n_bytes);
    }

    /// Set the device-resident decode `position` for the device-side KV append.
    /// CUDA only; no-op default.
    fn set_decode_position(&self, pos: usize) {
        let _ = pos;
    }

    /// Device-side KV append: copy output `id` into `base_ptr + position*n_bytes`
    /// with `position` read from the device buffer set by [`set_decode_position`],
    /// so the destination is computed on device (graph-capturable). CUDA only;
    /// no-op default.
    ///
    /// # Safety
    /// `base_ptr` must be a valid device allocation; `set_decode_position` set.
    unsafe fn copy_output_to_device_ptr_kv(&self, id: NodeIndex, base_ptr: u64, n_bytes: usize) {
        let _ = (id, base_ptr, n_bytes);
    }

    /// Launch a 2-rank shm all-reduce of `elems` bf16 at `data_ptr` on this
    /// runtime's stream (so it shares the SKEIN_CAPTURE stream). `shm_ptr` is the
    /// cross-process mapped shared region. CUDA only; no-op default.
    ///
    /// # Safety
    /// `data_ptr`/`shm_ptr` valid device pointers; both ranks call identically.
    unsafe fn device_shm_all_reduce(
        &self,
        data_ptr: u64,
        shm_ptr: u64,
        rank: i32,
        elems: usize,
        slot_bytes: i32,
    ) {
        let _ = (data_ptr, shm_ptr, rank, elems, slot_bytes);
    }

    /// Full-step CUDA graph capture/replay on the shared SKEIN_CAPTURE stream.
    /// CUDA only; defaults are no-ops / `false`.
    fn begin_stream_capture(&self) {}
    fn end_stream_capture(&self) {}
    fn replay_captured(&self) -> bool {
        false
    }
    fn has_captured_graph(&self) -> bool {
        false
    }

    /// Free this runtime's intermediate-buffer arena (re-allocated lazily on the
    /// next execute). Persistent inputs/weights are untouched. Called after
    /// load/search and between the prefill and decode graphs so two graphs'
    /// arenas aren't resident at once (they alternate). CUDA only; no-op default.
    fn clear_intermediates(&mut self) {}

    /// Point an input at an external device buffer owned elsewhere (zero-copy
    /// shared weights). CUDA only; a no-op default. `n_bytes` is the buffer size.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of at least `n_bytes` on this
    /// runtime's device, kept alive for the runtime's lifetime.
    unsafe fn set_input_device_ptr(&mut self, id: NodeIndex, ptr: u64, n_bytes: usize) {
        let _ = (id, ptr, n_bytes);
    }

    /// Point an input at an external device buffer **without** marking it
    /// persistent — for a transient segment-to-segment activation handoff that is
    /// re-bound every decode step (the producer overwrites its output buffer each
    /// step). Unlike [`set_input_device_ptr`] (weights, bound once, persistent).
    /// CUDA only; no-op default.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of at least `n_bytes` on this
    /// runtime's device, kept alive until this segment finishes executing.
    unsafe fn bind_input_device_ptr(&mut self, id: NodeIndex, ptr: u64, n_bytes: usize) {
        let _ = (id, ptr, n_bytes);
    }

    /// Upload raw bf16 weight bytes straight into a bf16 input buffer, skipping
    /// the bf16->f32->bf16 round-trip the f32 path forces (which, on ~90 GB of
    /// weights, costs tens of seconds of CPU and 2x host RAM). Default decodes to
    /// f32 then narrows (correct for any backend, e.g. the CPU runtime); the CUDA
    /// backend overrides it to reinterpret the bytes as bf16 and upload directly.
    fn set_data_bf16_bytes(&mut self, id: NodeIndex, bytes: &[u8]) {
        let data: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        self.set_data_f32_as(id, data, DType::Bf16);
    }

    /// Stage integer token/index data into the runtime's buffer.
    fn set_data_i32(&mut self, id: NodeIndex, data: Vec<i32>);

    /// Execute the compiled graph using the graph's dynamic-dim map.
    fn execute(&mut self, cx: &Graph);

    /// Read the output buffer for the given tensor as an owned `Vec<f32>`.
    fn get_data_f32(&self, id: NodeIndex) -> Vec<f32>;

    /// Read `elems` bf16 values at an external device pointer into host f32.
    /// Used by the single-process `LocalTopology` to host-stage a cross-GPU
    /// all-reduce over device-resident activation handoffs. CUDA only; the CPU
    /// backend never holds device pointers, so the default returns empty.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of `elems * 2` bytes on this
    /// runtime's device.
    unsafe fn read_device_bf16(&self, _ptr: u64, _elems: usize) -> Vec<f32> {
        Vec::new()
    }

    /// Write host f32 (narrowed to bf16) back to an external device pointer.
    /// Counterpart of [`read_device_bf16`](Self::read_device_bf16). CUDA only;
    /// no-op default.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of `data.len() * 2` bytes.
    unsafe fn write_device_bf16(&self, _ptr: u64, _data: &[f32]) {}

    /// Allocate a zeroed device buffer of `n_bytes`, returning its raw pointer
    /// (leaked — caller owns the lifetime). Used for per-request KV buffers in
    /// the continuous-batch driver. CUDA only; default returns 0.
    fn alloc_device_zeros(&self, _n_bytes: usize) -> u64 {
        0
    }
}

/// Top-level entry. `R` selects the backend at the call site. Takes a
/// slice of segments — one for each collective-bracketed region the
/// runtime will execute — and returns one runtime per segment in the
/// same order.
///
/// For `tp = ep = pp = 1` the caller passes a single-element slice and
/// gets back a one-element `Vec<R>`; for the multi-segment case
/// (`skein_emit::wire_segments` output) the slice is the full per-device
/// `Vec<Segment>`'s underlying graphs.
///
/// Sequential compilation: segments are compiled one after another.
/// TODO(parallel-compile): compile segments concurrently if compile
/// wall-time on the GPU host becomes a bottleneck.
pub fn compile_with_luminal<R: ComputeRuntime>(
    segments: &mut [Graph],
    search_budget: usize,
) -> Result<Vec<R>, CompileError> {
    let mut runtimes = Vec::with_capacity(segments.len());
    for cx in segments.iter_mut() {
        runtimes.push(R::build_and_search(cx, search_budget)?);
    }
    Ok(runtimes)
}

// ---------------------------------------------------------------------------
// NativeComputeRuntime — Luminal's CPU runtime. Always available.
// ---------------------------------------------------------------------------

/// Wraps Luminal's CPU `NativeRuntime`. Available unconditionally; this is
/// the runtime CI and the `compile_e2e` tests target when no GPU is present.
pub struct NativeComputeRuntime {
    inner: NativeRuntime,
}

impl ComputeRuntime for NativeComputeRuntime {
    fn build_and_search(cx: &mut Graph, budget: usize) -> Result<Self, CompileError> {
        cx.build_search_space::<NativeRuntime>();
        let inner = cx.search(NativeRuntime::default(), budget);
        Ok(Self { inner })
    }

    fn build_and_search_cached(
        cx: &mut Graph,
        budget: usize,
        _input_zeros: &[(NodeIndex, usize)],
        cache_dir: Option<&Path>,
    ) -> Result<Self, CompileError> {
        let inner = crate::search_cache::cached_search::<NativeRuntime>(
            cx,
            budget,
            cache_dir,
            "native",
            || Ok(NativeRuntime::default()),
            |_rt| {},
        )?;
        Ok(Self { inner })
    }

    fn set_data_f32(&mut self, id: NodeIndex, data: Vec<f32>) {
        self.inner.set_data(id, data);
    }

    fn set_data_i32(&mut self, id: NodeIndex, data: Vec<i32>) {
        self.inner.set_data(id, data);
    }

    fn execute(&mut self, cx: &Graph) {
        self.inner.execute(&cx.dyn_map);
    }

    fn get_data_f32(&self, id: NodeIndex) -> Vec<f32> {
        self.inner.get_f32(id).clone()
    }
}

// ---------------------------------------------------------------------------
// CudaComputeRuntime — Luminal's CUDA runtime. Behind `--features cuda`.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub use cuda_impl::CudaComputeRuntime;

/// Re-export of luminal's shared `SKEIN_CAPTURE` stream raw handle (a `u64`, so it
/// crosses the cudarc-version boundary) — lets the skein_runtime shm all-reduce
/// launch on the same stream as the segments.
#[cfg(feature = "cuda")]
pub use luminal_cuda_lite::runtime::capture_stream_raw;

/// Real per-segment CUDA-graph activity from the Luminal `CudaGraphOp` execution
/// path: `(graph_instantiations, graph_launches)` since process start. These come
/// from the actual `cuGraphInstantiate` / `cuGraphLaunch` call sites in
/// `luminal_cuda_lite`, so the runtime can report — and prove — that the model's
/// kernel CUDA graphs really build once and replay on every later forward.
#[cfg(feature = "cuda")]
pub fn cuda_graph_exec_stats() -> (u64, u64) {
    luminal_cuda_lite::kernel::graph_exec_stats()
}

/// The CUDA device ordinal the next `CudaComputeRuntime` built on this thread
/// binds to. `load_runtime_segments` sets this per device so a single process
/// can place tensor-parallel shards on distinct GPUs (d0->GPU0, d1->GPU1),
/// freeing per-GPU memory for concurrent batching. Defaults to 0 (single-GPU /
/// multi-process ranks, where `CUDA_VISIBLE_DEVICES` already maps device 0).
#[cfg(feature = "cuda")]
thread_local! {
    static BUILD_DEVICE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Set the device ordinal for the next CUDA runtime built on this thread.
#[cfg(feature = "cuda")]
pub fn set_build_device(device: usize) {
    BUILD_DEVICE.with(|c| c.set(device));
}

#[cfg(feature = "cuda")]
fn build_device() -> usize {
    BUILD_DEVICE.with(|c| c.get())
}

/// No-op on the CPU build (kept so generic callers compile without `cuda`).
#[cfg(not(feature = "cuda"))]
pub fn set_build_device(_device: usize) {}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use luminal_cuda_lite::runtime::CudaRuntime;

    pub struct CudaComputeRuntime {
        inner: CudaRuntime,
    }

    impl ComputeRuntime for CudaComputeRuntime {
        fn build_and_search(cx: &mut Graph, budget: usize) -> Result<Self, CompileError> {
            Self::build_and_search_with_input_zeros(cx, budget, &[])
        }

        fn build_and_search_with_input_zeros(
            cx: &mut Graph,
            budget: usize,
            input_zeros: &[(NodeIndex, usize)],
        ) -> Result<Self, CompileError> {
            Self::build_and_search_cached(cx, budget, input_zeros, None)
        }

        fn build_and_search_cached(
            cx: &mut Graph,
            budget: usize,
            input_zeros: &[(NodeIndex, usize)],
            cache_dir: Option<&Path>,
        ) -> Result<Self, CompileError> {
            let inner = crate::search_cache::cached_search::<CudaRuntime>(
                cx,
                budget,
                cache_dir,
                "cuda",
                || {
                    CudaRuntime::new_on(build_device()).map_err(|source| {
                        CompileError::CudaRuntimeInit {
                            source: Box::new(source),
                        }
                    })
                },
                // Stage zero buffers for every Input so the search's graph
                // executions have something to read (real data is loaded
                // after compile). Only the miss path profiles, so this runs
                // only when a real search happens.
                |runtime| {
                    for (id, num_bytes) in input_zeros {
                        runtime.set_zeros(*id, *num_bytes);
                    }
                },
            )?;
            Ok(Self { inner })
        }

        fn set_data_f32(&mut self, id: NodeIndex, data: Vec<f32>) {
            self.inner.set_data(id, data);
        }

        fn set_data_f32_as(&mut self, id: NodeIndex, data: Vec<f32>, dtype: DType) {
            // Narrow to the input's real dtype so a bf16 input slot receives
            // bf16 (matching get_f32's bf16->f32 widening on the read side).
            match dtype {
                DType::Bf16 => {
                    let bf: Vec<half::bf16> = data.into_iter().map(half::bf16::from_f32).collect();
                    self.inner.set_data(id, bf);
                }
                DType::F16 => {
                    let h: Vec<half::f16> = data.into_iter().map(half::f16::from_f32).collect();
                    self.inner.set_data(id, h);
                }
                _ => self.inner.set_data(id, data),
            }
        }

        fn set_data_bf16_bytes(&mut self, id: NodeIndex, bytes: &[u8]) {
            // Reinterpret the bf16 bytes directly as bf16 (the safetensors bytes
            // ARE the bf16 bits) and upload — no f32 round-trip, no 2x host RAM.
            let bf: Vec<half::bf16> = bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            self.inner.set_data(id, bf);
        }

        fn set_data_persistent_f32_as(&mut self, id: NodeIndex, data: Vec<f32>, dtype: DType) {
            // Upload (with the bf16/f16 narrow) then mark persistent so luminal
            // keeps the buffer across forwards instead of consuming it — weights
            // are uploaded once, not re-fed every step.
            self.set_data_f32_as(id, data, dtype);
            self.inner.mark_hlir_persistent(id);
        }

        fn input_device_ptr(&self, id: NodeIndex) -> Option<u64> {
            self.inner.hlir_device_ptr(id)
        }

        fn clear_intermediates(&mut self) {
            self.inner.free_intermediate_arenas();
        }

        unsafe fn set_input_device_ptr(&mut self, id: NodeIndex, ptr: u64, n_bytes: usize) {
            // Zero-copy: point this input at a weight buffer owned by another
            // (decode) graph, and mark it persistent so it is never consumed.
            unsafe { self.inner.set_device_ptr(id, ptr, n_bytes) };
            self.inner.mark_hlir_persistent(id);
        }

        unsafe fn bind_input_device_ptr(&mut self, id: NodeIndex, ptr: u64, n_bytes: usize) {
            // Transient activation handoff: bind to the producer segment's output
            // buffer, NOT persistent — re-bound every decode step. `set_device_ptr`
            // marks the node `changed_hlir`, so the new pointer takes effect.
            unsafe { self.inner.set_device_ptr(id, ptr, n_bytes) };
        }

        fn output_device_buffer(&self, id: NodeIndex) -> Option<(u64, usize)> {
            self.inner.output_device_buffer(id)
        }

        fn alloc_persistent_input_zeros(&mut self, id: NodeIndex, n_bytes: usize) -> Option<u64> {
            // Idempotent: only allocate on first sight; later steps reuse the
            // resident buffer (so the KV cache lives on the GPU, not re-uploaded).
            if let Some(p) = self.inner.hlir_device_ptr(id) {
                return Some(p);
            }
            self.inner.set_zeros(id, n_bytes);
            self.inner.mark_hlir_persistent(id);
            self.inner.hlir_device_ptr(id)
        }

        unsafe fn copy_output_to_device_ptr(&self, id: NodeIndex, dest_ptr: u64, n_bytes: usize) {
            unsafe { self.inner.copy_output_to_device_ptr(id, dest_ptr, n_bytes) };
        }

        fn set_decode_position(&self, pos: usize) {
            self.inner.set_decode_position(pos);
        }

        unsafe fn copy_output_to_device_ptr_kv(&self, id: NodeIndex, base_ptr: u64, n_bytes: usize) {
            unsafe { self.inner.copy_output_to_device_ptr_kv(id, base_ptr, n_bytes) };
        }

        unsafe fn device_shm_all_reduce(
            &self,
            data_ptr: u64,
            shm_ptr: u64,
            rank: i32,
            elems: usize,
            slot_bytes: i32,
        ) {
            unsafe {
                self.inner
                    .device_shm_all_reduce(data_ptr, shm_ptr, rank, elems, slot_bytes)
            };
        }

        fn begin_stream_capture(&self) {
            self.inner
                .begin_stream_capture()
                .expect("begin_stream_capture");
        }
        fn end_stream_capture(&self) {
            self.inner.end_stream_capture().expect("end_stream_capture");
        }
        fn replay_captured(&self) -> bool {
            self.inner.replay_captured()
        }
        fn has_captured_graph(&self) -> bool {
            self.inner.has_captured_graph()
        }

        unsafe fn read_device_bf16(&self, ptr: u64, elems: usize) -> Vec<f32> {
            unsafe { self.inner.read_device_bf16_to_f32(ptr, elems) }
        }

        unsafe fn write_device_bf16(&self, ptr: u64, data: &[f32]) {
            unsafe { self.inner.write_f32_to_device_bf16(ptr, data) }
        }

        fn alloc_device_zeros(&self, n_bytes: usize) -> u64 {
            self.inner.alloc_device_zeros(n_bytes)
        }

        fn set_data_i32(&mut self, id: NodeIndex, data: Vec<i32>) {
            self.inner.set_data(id, data);
        }

        fn execute(&mut self, cx: &Graph) {
            self.inner.execute(&cx.dyn_map);
        }

        fn get_data_f32(&self, id: NodeIndex) -> Vec<f32> {
            self.inner.get_f32(id).clone()
        }
    }
}
