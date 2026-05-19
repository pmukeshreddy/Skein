//! KV page transport abstraction.

use crate::kv::PageId;
use crate::types::RequestId;

pub trait KvTransport: Send + Sync {
    fn transfer(
        &self,
        src_device: usize,
        dst_device: usize,
        pages: &[PageId],
        request_id: RequestId,
    ) -> Result<(), TransportError>;
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("local KV transfer source and destination must differ")]
    SameDevice,
}

pub struct LocalKvTransport;

impl KvTransport for LocalKvTransport {
    fn transfer(
        &self,
        src_device: usize,
        dst_device: usize,
        _pages: &[PageId],
        _request_id: RequestId,
    ) -> Result<(), TransportError> {
        if src_device == dst_device {
            return Err(TransportError::SameDevice);
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct RdmaKvTransport;
