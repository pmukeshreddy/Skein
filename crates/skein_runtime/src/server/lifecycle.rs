//! Server construction + shutdown. No forward-pass driver here — that's
//! `server::forward`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::RwLock;

use skein_cost::CostConstants;
use skein_ir::plan::Plan;
use skein_ir::workload::Workload;

use crate::batcher::{AdmissionDecision, ContinuousBatcher, InflightSet};
use crate::error::RuntimeError;
use crate::hotswap::HotSwap;
use crate::kv::PagedKVAllocator;
use crate::observability::ProfileHooks;
use crate::token_stream::TokenStreamer;
use crate::types::IncomingRequest;

pub struct Server {
    pub plan: Plan,
    pub artifact_dir: PathBuf,
    pub kv: Arc<Mutex<PagedKVAllocator>>,
    pub batcher: Arc<RwLock<ContinuousBatcher>>,
    pub inflight: Arc<RwLock<InflightSet>>,
    pub hotswap: Arc<HotSwap>,
    pub profile: Arc<ProfileHooks>,
    pub http_port: u16,
}

pub struct ServerBuildInputs<'a> {
    pub artifact_dir: &'a Path,
    pub plan: Plan,
    pub workload: &'a Workload,
    pub cost_constants: &'a CostConstants,
    pub total_kv_bytes: u64,
    pub bytes_per_token: u64,
}

impl Server {
    /// Build a `Server`. Phase A consumes everything the runtime needs from
    /// the Plan / Workload / `CostConstants` triple. Phase B's CUDA path
    /// additionally loads the per-device compiled artifacts.
    pub fn new(inputs: ServerBuildInputs<'_>) -> Result<Self, RuntimeError> {
        let kv = PagedKVAllocator::new(
            &inputs.plan,
            inputs.total_kv_bytes,
            inputs.bytes_per_token,
            inputs.cost_constants.runtime.radix_max_depth,
        )?;
        let kv = Arc::new(Mutex::new(kv));
        let batcher = ContinuousBatcher::new(
            &inputs.plan,
            inputs.workload,
            inputs.cost_constants,
            kv.clone(),
        );
        let inflight = Arc::new(RwLock::new(InflightSet::new()));
        let hotswap = Arc::new(HotSwap::new(
            inputs.artifact_dir.to_path_buf(),
            Duration::from_secs(inputs.cost_constants.runtime.drain_timeout_seconds as u64),
        ));
        let profile = Arc::new(ProfileHooks::new(
            inputs.cost_constants.runtime.metrics_buffer_capacity as usize,
        )?);

        Ok(Self {
            plan: inputs.plan,
            artifact_dir: inputs.artifact_dir.to_path_buf(),
            kv,
            batcher: Arc::new(RwLock::new(batcher)),
            inflight,
            hotswap,
            profile,
            http_port: 18080,
        })
    }

    pub fn with_http_port(mut self, port: u16) -> Self {
        self.http_port = port;
        self
    }

    /// Submit a request. Returns a token streamer for the caller to read
    /// outputs from. Admission decides; if `Delay` or `Reject`, the stream
    /// is closed before return and the caller sees `None` on `.recv()`.
    pub async fn submit(&self, request: IncomingRequest) -> Result<TokenStreamer, RuntimeError> {
        let now_ms = crate::observability::wall_now_ms();
        let (streamer, sender) = TokenStreamer::paired();
        let decision = self.batcher.write().await.admit(request, sender, now_ms);
        // Phase B: the forward-pass driver picks the request up from the
        // batcher and starts producing tokens. On Phase A the streamer
        // exists but no tokens will arrive — the front door's `recv` will
        // pend forever unless the test closes it. Tests assert on the
        // decision returned by the batcher, not on streamed output.
        let _ = decision;
        Ok(streamer)
    }

    /// Convenience for tests: peek the most recent admission decision
    /// without holding the write lock open.
    pub async fn admit_for_test(
        &self,
        request: IncomingRequest,
    ) -> (AdmissionDecision, TokenStreamer) {
        let now_ms = crate::observability::wall_now_ms();
        let (streamer, sender) = TokenStreamer::paired();
        let decision = self.batcher.write().await.admit(request, sender, now_ms);
        (decision, streamer)
    }

    /// Forward-pass driver. Phase B serves HTTP on Mac with the native runtime.
    pub async fn serve(&self) -> Result<(), RuntimeError> {
        crate::server::forward::serve(self).await
    }
}
