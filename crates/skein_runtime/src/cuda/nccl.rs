//! NCCL collectives between graph invocations.
//!
//! TODO(nccl): wire the NCCL-backed communicator. Every operation currently
//! returns `RuntimeError::NotImplemented`. The module is
//! `#[cfg(feature = "cuda")]`-gated at the `cuda/mod.rs` level, so CPU
//! builds never see this code.

use crate::error::RuntimeError;

pub struct NcclCommunicator;

impl NcclCommunicator {
    pub fn new() -> Result<Self, RuntimeError> {
        Err(RuntimeError::NotImplemented {
            what: "NcclCommunicator::new",
        })
    }
}
