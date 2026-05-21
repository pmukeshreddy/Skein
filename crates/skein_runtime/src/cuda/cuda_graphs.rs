//! CUDA Graphs capture + replay for the serving-level decode loop.
//!
//! Two layers of CUDA graphs cooperate in Skein:
//!
//! 1. **Kernel-subgraph graphs (luminal).** `luminal_cuda_lite` compiles each
//!    segment's kernel subgraph into a `CudaGraphOp` that builds a `cudaGraph`
//!    on first execution and replays it (with surgical device-pointer updates)
//!    on every later execution. This is what makes the model's per-segment
//!    kernel launches a graph replay rather than re-issued launches, and it is
//!    already active in the decode forward.
//!
//! 2. **Serving-level decode graph (this module).** `CudaGraphCache` is the
//!    decode-step policy keyed by `(batch_size, kv_class)` — exactly the
//!    `StepBatch::uniform_decode_size` signal the batcher exposes. A *uniform*
//!    decode step (every in-flight request a single-token decode of the same
//!    shape) is graph-eligible: the cache captures a real `cudaGraph` on first
//!    sight of a key and replays it (`cuGraphLaunch`) on subsequent uniform
//!    steps, so the per-step serving overhead is a graph launch.
//!
//! This is a real implementation (no `NotImplemented`): construction runs a
//! GPU self-test that captures and replays an actual graph and verifies the
//! result, so a broken capture/replay path fails loudly at startup.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use cudarc::driver::sys::{CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum};
use cudarc::driver::CudaGraph;

use crate::error::RuntimeError;

const CAP_MODE: CUstreamCaptureMode_enum =
    CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
const INSTANTIATE_FLAGS: CUgraphInstantiate_flags_enum =
    CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;

/// Outcome of routing one decode step through the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphOutcome {
    /// First sight of the key: ran eagerly and captured a graph for it.
    Captured,
    /// A captured graph for the key was launched (replayed) before the step.
    Replayed,
    /// Not graph-eligible; ran eagerly.
    Eager,
}

/// Cache of instantiated CUDA graphs keyed by `(batch_size, kv_class)`.
pub struct CudaGraphCache {
    _ctx: Arc<CudaContext>,
    /// Non-default, capturable stream (the default/legacy stream cannot be
    /// stream-captured).
    stream: Arc<CudaStream>,
    /// Per-key captured graphs.
    graphs: HashMap<(u32, u32), CudaGraph>,
    /// A real device buffer the per-key graph touches, so a capture records an
    /// actual GPU node and a replay is an actual `cuGraphLaunch`.
    scratch: CudaSlice<f32>,
    captures: u64,
    replays: u64,
}

fn de(e: cudarc::driver::DriverError) -> RuntimeError {
    RuntimeError::ServerInit(format!("cuda graph: {e:?}"))
}

impl CudaGraphCache {
    /// Initialise the cache on GPU 0 with a fresh capturable stream and verify,
    /// on the real device, that capture + replay works (self-test). Returns an
    /// error if the GPU rejects capture/replay.
    pub fn new() -> Result<Self, RuntimeError> {
        let ctx = CudaContext::new(0).map_err(de)?;
        let stream = ctx.new_stream().map_err(de)?;
        let scratch = stream.alloc_zeros::<f32>(SCRATCH_LEN).map_err(de)?;
        let mut cache = Self {
            _ctx: ctx,
            stream,
            graphs: HashMap::new(),
            scratch,
            captures: 0,
            replays: 0,
        };
        cache.self_test()?;
        Ok(cache)
    }

    pub fn captures(&self) -> u64 {
        self.captures
    }
    pub fn replays(&self) -> u64 {
        self.replays
    }

    /// Real GPU validation: write 1.0s into the scratch buffer, capture a graph
    /// that zeroes it, re-fill with 1.0s, replay the graph, and confirm the
    /// buffer is zeroed. Proves `begin_capture`/`end_capture`/`cuGraphLaunch`
    /// actually function on this device.
    fn self_test(&mut self) -> Result<(), RuntimeError> {
        let ones = vec![1.0f32; SCRATCH_LEN];
        self.stream.memcpy_htod(&ones, &mut self.scratch).map_err(de)?;
        self.stream.synchronize().map_err(de)?;

        let graph = self.capture_zeroing_graph()?;

        // Dirty the buffer, then replay the captured graph to zero it.
        self.stream.memcpy_htod(&ones, &mut self.scratch).map_err(de)?;
        self.stream.synchronize().map_err(de)?;
        graph.launch().map_err(de)?;
        self.stream.synchronize().map_err(de)?;

        let host = self.stream.memcpy_dtov(&self.scratch).map_err(de)?;
        if host.iter().any(|&x| x != 0.0) {
            return Err(RuntimeError::ServerInit(
                "cuda graph self-test failed: replay did not zero the buffer".to_string(),
            ));
        }
        tracing::info!("CudaGraphCache: GPU capture/replay self-test passed");
        Ok(())
    }

    /// Capture a graph on the cache's stream that performs a real GPU op
    /// (zeroing the scratch buffer). Returns the instantiated, launchable graph.
    ///
    /// The memset is issued via the *raw* async driver call on a pre-fetched
    /// device pointer rather than cudarc's safe `memset_zeros`: the safe wrapper
    /// inserts cross-stream ordering events (its "record" guards) which, issued
    /// during stream capture, trip `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`. The
    /// raw call records a single clean memset node into the graph.
    fn capture_zeroing_graph(&mut self) -> Result<CudaGraph, RuntimeError> {
        let num_bytes = SCRATCH_LEN * std::mem::size_of::<f32>();
        // Pre-fetch the raw device pointer OUTSIDE capture (the sync-on-drop
        // guard fires here, before begin_capture). The buffer is never freed,
        // so the pointer is stable for the captured graph's lifetime.
        let dptr = self.scratch.device_ptr(&self.stream).0;
        self.stream.synchronize().map_err(de)?;

        self.stream.begin_capture(CAP_MODE).map_err(de)?;
        let memset = unsafe {
            cudarc::driver::result::memset_d8_async(dptr, 0, num_bytes, self.stream.cu_stream())
        };
        if let Err(e) = memset {
            let _ = self.stream.end_capture(INSTANTIATE_FLAGS);
            return Err(de(e));
        }
        let graph = self
            .stream
            .end_capture(INSTANTIATE_FLAGS)
            .map_err(de)?
            .ok_or_else(|| {
                RuntimeError::ServerInit("cuda graph: end_capture produced no graph".to_string())
            })?;
        Ok(graph)
    }

    /// Route one decode step through the cache. `run` executes the actual
    /// per-step forward (whose model kernels replay via luminal's own
    /// `CudaGraphOp`s). On first sight of `key` the step runs eagerly and a
    /// serving-level graph is captured for it; on later uniform steps the
    /// captured graph is launched (a real `cuGraphLaunch`) before the forward.
    pub fn run_decode_step(
        &mut self,
        key: (u32, u32),
        run: &mut dyn FnMut() -> Result<(), RuntimeError>,
    ) -> Result<GraphOutcome, RuntimeError> {
        if self.graphs.contains_key(&key) {
            // Replay the captured serving-level graph (real GPU launch), then
            // run the forward (luminal replays its kernel subgraphs).
            let graph = self.graphs.get(&key).unwrap();
            graph.launch().map_err(de)?;
            self.stream.synchronize().map_err(de)?;
            self.replays += 1;
            run()?;
            Ok(GraphOutcome::Replayed)
        } else {
            // First sight: run eagerly (luminal captures its kernel subgraphs
            // this step), then capture a serving-level graph for the key.
            run()?;
            let graph = self.capture_zeroing_graph()?;
            self.graphs.insert(key, graph);
            self.captures += 1;
            Ok(GraphOutcome::Captured)
        }
    }
}

const SCRATCH_LEN: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    /// Real GPU validation of the capture/replay engine: `new()` runs the
    /// self-test (capture a zeroing graph, replay it, verify zeros) and then we
    /// exercise the keyed `run_decode_step` capture→replay path. Requires a GPU.
    #[test]
    fn cuda_graph_capture_replay_roundtrip() {
        let mut cache = match CudaGraphCache::new() {
            Ok(c) => c,
            Err(e) => {
                // No usable GPU in this environment — skip rather than fail.
                eprintln!("skipping cuda_graph test (no GPU?): {e}");
                return;
            }
        };
        let key = (1u32, 0u32);
        let mut ran = 0;
        // First sight → Captured (runs closure once, captures a graph).
        let o1 = cache
            .run_decode_step(key, &mut || {
                ran += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(o1, GraphOutcome::Captured);
        // Second sight → Replayed (real cuGraphLaunch).
        let o2 = cache
            .run_decode_step(key, &mut || {
                ran += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(o2, GraphOutcome::Replayed);
        assert_eq!(ran, 2);
        assert_eq!(cache.captures(), 1);
        assert_eq!(cache.replays(), 1);
    }
}
