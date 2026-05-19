//! RDMA for P/D KV transfer. Phase B Step 11.

use crate::error::RuntimeError;

pub struct RdmaTransport;

impl RdmaTransport {
    pub fn new() -> Result<Self, RuntimeError> {
        Err(RuntimeError::PhaseBOnly {
            what: "RdmaTransport::new",
        })
    }
}
