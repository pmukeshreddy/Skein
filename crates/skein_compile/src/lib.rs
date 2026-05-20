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

use luminal::op::Runtime;
use luminal::prelude::{Graph, NativeRuntime, NodeIndex};

pub mod artifact;
pub mod dyn_runtime;
pub mod error;
pub mod executor;

pub use artifact::{
    ArtifactMetadata, DeclaredTensorMeta, DeviceArtifactLoaded, LoweredSegment, OpRecipe,
    SegmentMetadata, SkeinArtifact,
};
pub use dyn_runtime::{
    DynRuntime, DynRuntimeError, DynRuntimeWrapper, WeightDtype, decode_weight_bytes,
};
pub use error::CompileError;
pub use executor::{
    CollectiveExecutor, DEFAULT_SEARCH_BUDGET, RuntimeSegment, StepOutput, TopologyExecutor,
    TopologyStepBatch, load_device_runtime_segments, load_native_runtime_segments,
    load_runtime_segments, segment_input_zero_bytes,
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

    /// Stage `data` into the runtime's buffer for the given input tensor.
    fn set_data_f32(&mut self, id: NodeIndex, data: Vec<f32>);

    /// Stage integer token/index data into the runtime's buffer.
    fn set_data_i32(&mut self, id: NodeIndex, data: Vec<i32>);

    /// Execute the compiled graph using the graph's dynamic-dim map.
    fn execute(&mut self, cx: &Graph);

    /// Read the output buffer for the given tensor as an owned `Vec<f32>`.
    fn get_data_f32(&self, id: NodeIndex) -> Vec<f32>;
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
            cx.build_search_space::<CudaRuntime>();
            let mut runtime = CudaRuntime::new().map_err(|source| CompileError::CudaRuntimeInit {
                source: Box::new(source),
            })?;
            // Stage zero buffers for every Input so the search's graph
            // executions have something to read (real data is loaded later).
            for (id, num_bytes) in input_zeros {
                runtime.set_zeros(*id, *num_bytes);
            }
            let inner = cx.search(runtime, budget);
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
}
