//! CUDA Graphs capture + dispatch. Phase B Step 11.

use crate::error::RuntimeError;

pub struct CudaGraphCache;

impl CudaGraphCache {
    pub fn new() -> Result<Self, RuntimeError> {
        Err(RuntimeError::PhaseBOnly {
            what: "CudaGraphCache::new",
        })
    }
}
