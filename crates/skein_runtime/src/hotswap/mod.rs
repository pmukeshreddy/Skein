//! Hot-swap protocol: drain in-flight, verify shape compatibility, swap
//! the `LATEST` symlink atomically.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use crate::batcher::InflightSet;
use crate::error::RuntimeError;

pub mod drain;
pub mod mount;
pub mod symlink;

pub use mount::verify_shape_compatibility;
pub use symlink::atomic_symlink_swap;

pub struct HotSwap {
    pub artifacts_dir: PathBuf,
    pub drain_timeout: Duration,
}

impl HotSwap {
    pub fn new(artifacts_dir: PathBuf, drain_timeout: Duration) -> Self {
        Self {
            artifacts_dir,
            drain_timeout,
        }
    }

    /// Swap the running artifact for the one at `new_artifact_path`:
    ///
    /// 1. Verify shape compatibility (input dtypes must match — output
    ///    dtypes may differ, that's the whole point of recompile).
    /// 2. Drain in-flight requests with timeout.
    /// 3. Atomically update `<artifacts_dir>/LATEST` via POSIX rename.
    ///
    /// On step 1 failure: no drain initiated, no symlink touched, the
    /// running artifact keeps serving.
    pub async fn swap_artifact(
        &self,
        new_artifact_path: &Path,
        current_inflight: Arc<RwLock<InflightSet>>,
    ) -> Result<(), RuntimeError> {
        let latest = self.artifacts_dir.join("LATEST");
        let old_target = symlink::current_target(&latest).unwrap_or_else(|_| latest.clone());
        verify_shape_compatibility(&old_target, new_artifact_path)?;
        drain::drain_inflight(current_inflight, self.drain_timeout).await?;
        atomic_symlink_swap(&latest, new_artifact_path)?;
        Ok(())
    }
}
