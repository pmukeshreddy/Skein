//! CUDA Graphs capture + dispatch.
//!
//! Captured graphs are keyed by `(batch_size, kv_class)` — the same key the
//! batcher exposes via `StepBatch::uniform_decode_size` — so a uniform decode
//! step can replay a pre-captured graph instead of re-issuing every kernel.
//!
//! TODO(cuda-graphs): implement `capture` (stream-capture the segment's
//! kernels via `cuStreamBeginCapture`/`cuStreamEndCapture` into a
//! `cudaGraphExec_t`) and `replay` (`cuGraphLaunch`). Every method currently
//! returns `RuntimeError::NotImplemented`.

use crate::error::RuntimeError;

/// Cache of instantiated CUDA graphs keyed by `(batch_size, kv_class)`.
pub struct CudaGraphCache;

impl CudaGraphCache {
    pub fn new() -> Result<Self, RuntimeError> {
        Err(RuntimeError::NotImplemented {
            what: "CudaGraphCache::new",
        })
    }

    /// Capture the current step's kernels into a graph for `key`.
    pub fn capture(&mut self, _key: (u32, u32)) -> Result<(), RuntimeError> {
        Err(RuntimeError::NotImplemented {
            what: "CudaGraphCache::capture",
        })
    }

    /// Replay a previously captured graph for `key`. Returns whether a
    /// captured graph was found and launched.
    pub fn replay(&mut self, _key: (u32, u32)) -> Result<bool, RuntimeError> {
        Err(RuntimeError::NotImplemented {
            what: "CudaGraphCache::replay",
        })
    }
}
