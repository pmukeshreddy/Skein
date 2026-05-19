//! Retire path — release KV pages, close the token streamer.

use std::sync::{Arc, Mutex};

use crate::error::RuntimeError;
use crate::kv::PagedKVAllocator;
use crate::types::RequestId;

use super::inflight::InflightSet;

pub fn retire(
    request_id: RequestId,
    kv: &Arc<Mutex<PagedKVAllocator>>,
    inflight: &mut InflightSet,
) -> Result<(), RuntimeError> {
    let req = inflight
        .remove(request_id)
        .ok_or(RuntimeError::UnknownRequest(request_id.0))?;
    // Closing the sender wakes the front-door's `recv().await` with `None`.
    req.sender.close();
    // Release KV pages.
    if let Ok(mut a) = kv.lock() {
        a.release(request_id)?;
    }
    Ok(())
}
