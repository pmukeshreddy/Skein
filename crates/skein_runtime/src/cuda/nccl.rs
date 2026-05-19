//! NCCL between graph invocations. Phase B Step 11.
//!
//! Placeholder that returns `Err(PhaseBOnly)` for every operation. The
//! file is `#[cfg(feature = "cuda")]`-gated at the `cuda/mod.rs` level,
//! so Phase A non-CUDA builds never see this code.

use crate::error::RuntimeError;

pub struct NcclCommunicator;

impl NcclCommunicator {
    pub fn new() -> Result<Self, RuntimeError> {
        Err(RuntimeError::PhaseBOnly {
            what: "NcclCommunicator::new",
        })
    }
}
