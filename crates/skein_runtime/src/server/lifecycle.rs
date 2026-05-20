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
    /// Build a `Server` from the Plan / Workload / `CostConstants` triple.
    /// The forward-pass driver additionally loads the per-device compiled
    /// artifacts at `serve()` time.
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
        // `submit` only enqueues; the forward-pass driver (`serve`) picks the
        // request up from the batcher and produces tokens. When no driver is
        // running the streamer exists but no tokens arrive — tests assert on
        // the admission decision, not on streamed output.
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

    /// Forward-pass driver: serves the HTTP streaming endpoint backed by the
    /// topology executor.
    pub async fn serve(&self) -> Result<(), RuntimeError> {
        crate::server::forward::serve(self).await
    }

    /// Start the forward driver in the background without binding the HTTP
    /// listener. After this returns, requests submitted via [`Server::submit`]
    /// are picked up by the driver and streamed. Intended for integration
    /// tests; production callers use [`Server::serve`].
    #[doc(hidden)]
    pub async fn start_driver_for_test(&self) -> Result<(), RuntimeError> {
        #[cfg(feature = "cuda")]
        {
            crate::server::forward::start_driver::<skein_compile::CudaComputeRuntime>(self)
        }
        #[cfg(not(feature = "cuda"))]
        {
            crate::server::forward::start_driver::<skein_compile::NativeComputeRuntime>(self)
        }
    }
}
