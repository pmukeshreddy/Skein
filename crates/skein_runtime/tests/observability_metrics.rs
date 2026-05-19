//! Test 10 — Prometheus exporter serves valid text format with the expected
//! metric names.

mod common;

use std::time::Duration;

use skein_runtime::observability::prometheus::serve_metrics_ephemeral;
use skein_runtime::{ProfileHooks, StepMetrics};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[tokio::test]
async fn observability_metrics_endpoint() {
    let hooks = ProfileHooks::new(1024).unwrap();
    // Record 100 step metrics so the histograms have observations.
    for i in 0..100u64 {
        hooks
            .record_step(StepMetrics {
                step_idx: i,
                compute_us: 1_000.0 + i as f64,
                comm_us: 50.0,
                kv_pages_in_use: (i % 64) as u32,
                batch_size: ((i % 8) + 1) as u32,
                uniform_decode_size: Some(8),
            })
            .await;
    }

    let registry = hooks.registry().clone();
    let (_handle, port) = serve_metrics_ephemeral(registry).await.unwrap();

    // Give the server a tick to enter its accept loop.
    tokio::time::sleep(Duration::from_millis(20)).await;

    // Issue a raw HTTP GET /metrics.
    let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    stream
        .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let s = String::from_utf8_lossy(&response);

    assert!(s.starts_with("HTTP/1.1 200 OK"), "headers: {s:?}");
    // Required metric names per the spec.
    for needle in [
        "skein_step_compute_us_total",
        "skein_kv_pages_in_use",
        "skein_batch_size_histogram",
    ] {
        assert!(
            s.contains(needle),
            "metrics body missing {needle:?}; body = {s:?}"
        );
    }
    // Prometheus text format always emits `# TYPE` lines per metric.
    assert!(s.contains("# TYPE"), "missing TYPE lines");
}
