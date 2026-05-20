//! CUDA Graphs capture + dispatch.
//!
//! TODO(cuda-graphs): capture per-batch-class CUDA graphs and dispatch them
//! to amortize launch overhead. `new` currently returns
//! `RuntimeError::NotImplemented`.

use crate::error::RuntimeError;

pub struct CudaGraphCache;

impl CudaGraphCache {
    pub fn new() -> Result<Self, RuntimeError> {
        Err(RuntimeError::NotImplemented {
            what: "CudaGraphCache::new",
        })
    }
}
