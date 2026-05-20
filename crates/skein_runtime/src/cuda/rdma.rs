//! RDMA transport for prefill/decode KV transfer.
//!
//! TODO(rdma): wire RDMA-backed KV-block transfer between prefill and decode
//! pools. `new` currently returns `RuntimeError::NotImplemented`.

use crate::error::RuntimeError;

pub struct RdmaTransport;

impl RdmaTransport {
    pub fn new() -> Result<Self, RuntimeError> {
        Err(RuntimeError::NotImplemented {
            what: "RdmaTransport::new",
        })
    }
}
