//! Drain in-flight requests with a deadline.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

use crate::batcher::InflightSet;
use crate::error::RuntimeError;

pub async fn drain_inflight(
    inflight: Arc<RwLock<InflightSet>>,
    timeout: Duration,
) -> Result<(), RuntimeError> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = inflight.read().await.len();
        if remaining == 0 {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(RuntimeError::DrainTimeout {
                remaining: remaining as u32,
                timeout_seconds: timeout.as_secs() as u32,
            });
        }
        // Yield long enough for the batcher to retire something; in tests
        // we drive `retire` from the test thread, so 1 ms keeps the loop
        // responsive without burning CPU.
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}
