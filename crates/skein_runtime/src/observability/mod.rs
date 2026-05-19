//! Observability: step + request metric ring buffers, Prometheus exporter,
//! tracing spans. Full OpenTelemetry exporter wiring lands in Phase B; the
//! Phase A skeleton uses `tracing::info_span!` so request/step events
//! show up under any tracing-subscriber installed by the front door.

pub mod metrics;
pub mod otel;
pub mod prometheus;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use ::prometheus::Registry;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use crate::error::RuntimeError;

pub use metrics::{RequestMetrics, StepMetrics};

/// Profile hook bundle. Holds the ring buffers, the Prometheus registry,
/// and the named counters/gauges/histograms the runtime updates per step
/// and per request.
pub struct ProfileHooks {
    step_buffer: Arc<RwLock<VecDeque<StepMetrics>>>,
    request_buffer: Arc<RwLock<VecDeque<RequestMetrics>>>,
    capacity: usize,
    registry: Registry,
    metrics: prometheus::PromMetrics,
}

impl ProfileHooks {
    pub fn new(buffer_capacity: usize) -> Result<Self, RuntimeError> {
        let registry = Registry::new();
        let metrics = prometheus::PromMetrics::register(&registry)?;
        Ok(Self {
            step_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_capacity))),
            request_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_capacity))),
            capacity: buffer_capacity,
            registry,
            metrics,
        })
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    pub async fn record_step(&self, step: StepMetrics) {
        self.metrics.observe_step(&step);
        let mut buf = self.step_buffer.write().await;
        if buf.len() == self.capacity {
            buf.pop_front();
        }
        buf.push_back(step);
    }

    pub async fn record_request(&self, request: RequestMetrics) {
        self.metrics.observe_request(&request);
        let mut buf = self.request_buffer.write().await;
        if buf.len() == self.capacity {
            buf.pop_front();
        }
        buf.push_back(request);
    }

    /// Spawn the Prometheus exporter on `port`. Returns the join handle so
    /// the caller can `abort()` it on shutdown. Tests bind ephemeral ports.
    pub async fn start_prometheus_exporter(
        &self,
        port: u16,
    ) -> Result<JoinHandle<()>, RuntimeError> {
        prometheus::serve_metrics(port, self.registry.clone()).await
    }

    /// Recent request traces within `window`. Used by the Loop-3 workload-
    /// drift detector to compute the KL between live and compile-time
    /// distributions.
    pub async fn recent_traces(&self, window: Duration) -> Vec<RequestMetrics> {
        let now_ms = wall_now_ms();
        let cutoff = now_ms.saturating_sub(window.as_millis() as u64);
        self.request_buffer
            .read()
            .await
            .iter()
            .filter(|m| m.completed_at_ms >= cutoff)
            .cloned()
            .collect()
    }
}

pub(crate) fn wall_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
