//! Prometheus exporter. Tiny tokio TCP HTTP handler — no `hyper`, no
//! `axum`; the metrics endpoint serves Prometheus text format which is
//! line-oriented and trivial to produce. Two endpoints: GET `/metrics`
//! returns the encoded registry; anything else returns 404.

use prometheus::{
    Encoder, Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, Opts, Registry,
    TextEncoder,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use crate::error::RuntimeError;

use super::metrics::{RequestMetrics, StepMetrics};

/// Container of the per-instance Prometheus instruments. Public-by-field
/// at crate level so tests can read them; the public surface is the
/// `ProfileHooks` recorder methods.
pub(crate) struct PromMetrics {
    pub steps_total: IntCounter,
    pub compute_us_total: prometheus::Counter,
    pub comm_us_total: prometheus::Counter,
    pub kv_pages_in_use: IntGauge,
    pub batch_size_hist: Histogram,
    pub requests_total: IntCounterVec,
    pub ttft_ms_hist: Histogram,
    pub tpot_ms_hist: Histogram,
    pub prefix_cache_hit_tokens_total: IntCounter,
}

impl PromMetrics {
    pub fn register(registry: &Registry) -> Result<Self, RuntimeError> {
        let steps_total = IntCounter::new("skein_steps_total", "Forward steps executed")?;
        let compute_us_total = prometheus::Counter::new(
            "skein_step_compute_us_total",
            "Total compute time across all steps, microseconds",
        )?;
        let comm_us_total = prometheus::Counter::new(
            "skein_step_comm_us_total",
            "Total comm time across all steps, microseconds (Phase B)",
        )?;
        let kv_pages_in_use = IntGauge::new("skein_kv_pages_in_use", "Currently active KV pages")?;
        let batch_size_hist = Histogram::with_opts(
            HistogramOpts::new(
                "skein_batch_size_histogram",
                "Per-step batch size distribution",
            )
            .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
        )?;
        let requests_total = IntCounterVec::new(
            Opts::new("skein_requests_total", "Requests completed by status"),
            &["status"],
        )?;
        let ttft_ms_hist = Histogram::with_opts(
            HistogramOpts::new("skein_ttft_ms_histogram", "Time-to-first-token (ms)")
                .buckets(vec![25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0]),
        )?;
        let tpot_ms_hist = Histogram::with_opts(
            HistogramOpts::new("skein_tpot_ms_histogram", "Per-output-token latency (ms)")
                .buckets(vec![5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]),
        )?;
        let prefix_cache_hit_tokens_total = IntCounter::new(
            "skein_prefix_cache_hit_tokens_total",
            "Prompt tokens served from the radix prefix cache",
        )?;
        registry.register(Box::new(steps_total.clone()))?;
        registry.register(Box::new(compute_us_total.clone()))?;
        registry.register(Box::new(comm_us_total.clone()))?;
        registry.register(Box::new(kv_pages_in_use.clone()))?;
        registry.register(Box::new(batch_size_hist.clone()))?;
        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(ttft_ms_hist.clone()))?;
        registry.register(Box::new(tpot_ms_hist.clone()))?;
        registry.register(Box::new(prefix_cache_hit_tokens_total.clone()))?;
        Ok(Self {
            steps_total,
            compute_us_total,
            comm_us_total,
            kv_pages_in_use,
            batch_size_hist,
            requests_total,
            ttft_ms_hist,
            tpot_ms_hist,
            prefix_cache_hit_tokens_total,
        })
    }

    pub fn observe_step(&self, step: &StepMetrics) {
        self.steps_total.inc();
        self.compute_us_total.inc_by(step.compute_us);
        self.comm_us_total.inc_by(step.comm_us);
        self.kv_pages_in_use.set(step.kv_pages_in_use as i64);
        self.batch_size_hist.observe(step.batch_size as f64);
    }

    pub fn observe_request(&self, request: &RequestMetrics) {
        self.requests_total.with_label_values(&["success"]).inc();
        self.ttft_ms_hist.observe(request.ttft_ms);
        for tpot in &request.per_token_latency_ms {
            self.tpot_ms_hist.observe(*tpot);
        }
        self.prefix_cache_hit_tokens_total
            .inc_by(request.prefix_cache_hit_tokens as u64);
    }
}

/// Spawn an HTTP server on `port` serving `/metrics` in Prometheus text
/// format. Returns the join handle so callers can abort on shutdown.
/// Tests pass `port = 0` to get a kernel-assigned ephemeral port.
pub async fn serve_metrics(port: u16, registry: Registry) -> Result<JoinHandle<()>, RuntimeError> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .map_err(|source| RuntimeError::PrometheusBind { port, source })?;
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let registry = registry.clone();
            tokio::spawn(async move {
                let (rd, mut wr) = sock.split();
                let mut reader = BufReader::new(rd);
                // Read the request line.
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).await.is_err() {
                    return;
                }
                // Drain headers (until blank line).
                loop {
                    let mut hdr = String::new();
                    match reader.read_line(&mut hdr).await {
                        Ok(0) => break,
                        Ok(_) => {
                            if hdr.trim().is_empty() {
                                break;
                            }
                        }
                        Err(_) => return,
                    }
                }
                let is_metrics = request_line.split_whitespace().nth(1) == Some("/metrics");
                if !is_metrics {
                    let _ = wr
                        .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                        .await;
                    return;
                }
                let encoder = TextEncoder::new();
                let mut body = Vec::new();
                if encoder.encode(&registry.gather(), &mut body).is_err() {
                    let _ = wr
                        .write_all(b"HTTP/1.1 500 Encode\r\nContent-Length: 0\r\n\r\n")
                        .await;
                    return;
                }
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len(),
                );
                let _ = wr.write_all(header.as_bytes()).await;
                let _ = wr.write_all(&body).await;
                let _ = wr.flush().await;
            });
        }
    });
    Ok(handle)
}

/// Bind a Prometheus exporter to an ephemeral port and return both the
/// task handle and the chosen port. Used by tests that don't want to
/// hard-code a port number.
pub async fn serve_metrics_ephemeral(
    registry: Registry,
) -> Result<(JoinHandle<()>, u16), RuntimeError> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|source| RuntimeError::PrometheusBind { port: 0, source })?;
    let port = listener
        .local_addr()
        .map_err(|source| RuntimeError::PrometheusBind { port: 0, source })?
        .port();
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let registry = registry.clone();
            tokio::spawn(async move {
                let (rd, mut wr) = sock.split();
                let mut reader = BufReader::new(rd);
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).await.is_err() {
                    return;
                }
                loop {
                    let mut hdr = String::new();
                    match reader.read_line(&mut hdr).await {
                        Ok(0) => break,
                        Ok(_) => {
                            if hdr.trim().is_empty() {
                                break;
                            }
                        }
                        Err(_) => return,
                    }
                }
                let is_metrics = request_line.split_whitespace().nth(1) == Some("/metrics");
                if !is_metrics {
                    let _ = wr
                        .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                        .await;
                    return;
                }
                let encoder = TextEncoder::new();
                let mut body = Vec::new();
                if encoder.encode(&registry.gather(), &mut body).is_err() {
                    return;
                }
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len(),
                );
                let _ = wr.write_all(header.as_bytes()).await;
                let _ = wr.write_all(&body).await;
                let _ = wr.flush().await;
            });
        }
    });
    Ok((handle, port))
}
