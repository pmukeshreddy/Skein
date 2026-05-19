//! `skein_compile` — the bridge between Skein's per-device `LoweredGraph`
//! and Luminal's search-based compiler.
//!
//! The crate is intentionally small: one trait (`ComputeRuntime`) abstracts
//! over Luminal's runtime backends, two impls plug in Native and CUDA, and
//! `compile_with_luminal::<R>` invokes `cx.build_search_space::<R::Inner>()`
//! followed by `cx.search(R::Inner::default(), budget)`.
//!
//! ## Phase 1a scope
//!
//! - `NativeComputeRuntime` is always available; works on Mac via Luminal's
//!   `NativeRuntime`. Used by every Phase A test that exercises a real
//!   Luminal compile-and-execute path.
//! - `CudaComputeRuntime` is `#[cfg(feature = "cuda")]`-gated and pulls in
//!   `luminal_cuda_lite::CudaRuntime`. Compile-time only on Mac (no GPU
//!   needed for the trait impl to type-check), execution requires an H100.
//!
//! ## Deferred to Prompt 1b
//!
//! Multi-segment compile. When the runtime needs collective boundaries
//! (tp > 1, ep > 1), each per-device `LoweredGraph` becomes a sequence
//! of segments and this function compiles each segment independently.
//! Today it compiles a single graph end-to-end.

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
pub use dyn_runtime::{DynRuntime, DynRuntimeError, DynRuntimeWrapper};
pub use error::CompileError;
pub use executor::{
    CollectiveExecutor, DEFAULT_SEARCH_BUDGET, RuntimeSegment, StepOutput, TopologyExecutor,
    TopologyStepBatch, load_native_runtime_segments, load_runtime_segments,
};

/// Backend abstraction. Two concrete impls: `NativeComputeRuntime`
/// (Luminal's `NativeRuntime`, always available) and
/// `CudaComputeRuntime` (Luminal's `CudaRuntime`, `#[cfg(feature = "cuda")]`).
/// Phase B Metal / ROCm backends drop in as new impls.
pub trait ComputeRuntime: Sized {
    /// Run Luminal's search-based compile against `cx` with the given
    /// budget, then return the resulting runtime ready to execute.
    fn build_and_search(cx: &mut Graph, budget: usize) -> Result<Self, CompileError>;

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
/// Parallel compilation is a follow-up if the wall-time becomes the
/// bottleneck on H100; on Mac NativeRuntime the search budget is the
/// dominant cost and parallelism doesn't help much.
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

/// Wraps Luminal's `NativeRuntime`. Available unconditionally; this is
/// the runtime Phase A tests and Phase 1a's `compile_e2e` tests target.
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
            cx.build_search_space::<CudaRuntime>();
            let inner = cx.search(CudaRuntime::default(), budget);
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
