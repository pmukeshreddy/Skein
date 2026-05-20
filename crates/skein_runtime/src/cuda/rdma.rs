//! RDMA transport for prefill/decode KV transfer.
//!
//! The production counterpart to [`crate::kv_transport::LocalKvTransport`]:
//! moves KV-cache blocks between a prefill pool and a decode pool over RDMA
//! instead of an in-process copy.
//!
//! TODO(rdma): register the KV-block memory regions and implement
//! `send`/`recv` over an RDMA verbs queue pair (or a library such as NIXL /
//! UCX). Every method currently returns `RuntimeError::NotImplemented`.

use crate::error::RuntimeError;

/// One endpoint of an RDMA connection between two KV pools.
pub struct RdmaTransport;

impl RdmaTransport {
    pub fn new() -> Result<Self, RuntimeError> {
        Err(RuntimeError::NotImplemented {
            what: "RdmaTransport::new",
        })
    }

    /// Send the KV blocks for `request_id` to the remote pool.
    pub fn send(&self, _request_id: u64, _block_ids: &[u32]) -> Result<(), RuntimeError> {
        Err(RuntimeError::NotImplemented {
            what: "RdmaTransport::send",
        })
    }

    /// Receive KV blocks for `request_id` from the remote pool.
    pub fn recv(&self, _request_id: u64, _block_ids: &[u32]) -> Result<(), RuntimeError> {
        Err(RuntimeError::NotImplemented {
            what: "RdmaTransport::recv",
        })
    }
}
