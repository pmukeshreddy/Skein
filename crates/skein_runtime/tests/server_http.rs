//! Ignored end-to-end checks for the Phase B axum serving path.

mod common;

#[allow(dead_code)]
#[path = "../../skein_parity/tests/fixtures/build_tiny_artifact.rs"]
mod tiny_artifact;

use std::io;
use std::net::TcpListener as StdTcpListener;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use common::{load_cost_constants, mk_workload};
use skein_runtime::Server;
use skein_runtime::server::lifecycle::ServerBuildInputs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "native Luminal compile for the tiny artifact is slow"]
async fn server_serve_accepts_request_mac() {
    let fixture = ServerFixture::start().await;

    let response = fixture
        .post_completion(r#"{"prompt":"Hello","max_tokens":3,"stream":true}"#)
        .await;
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert_eq!(response.matches("event: token").count(), 3, "{response}");

    fixture.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "native Luminal compile for the tiny artifact is slow"]
async fn server_metrics_endpoint_exposes_metrics() {
    let fixture = ServerFixture::start().await;

    let response = fixture
        .post_completion(r#"{"prompt":"Hello","max_tokens":3,"stream":true}"#)
        .await;
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");

    let metrics = fixture.get_metrics().await;
    assert!(metrics.contains("skein_requests_total{status=\"success\"} 1"));
    assert!(metrics.contains("skein_ttft_ms_histogram"));
    assert!(metrics.contains("skein_tpot_ms_histogram"));

    fixture.shutdown().await;
}

struct ServerFixture {
    root: PathBuf,
    port: u16,
    handle: tokio::task::JoinHandle<Result<(), skein_runtime::RuntimeError>>,
}

impl ServerFixture {
    async fn start() -> Self {
        let root = unique_temp_dir("skein-runtime-server-http");
        let tiny = tiny_artifact::build_tiny_artifact(&root);
        let port = unused_port();
        let cost_constants = load_cost_constants();
        let workload = mk_workload(5_000, 5_000);
        let server = Server::new(ServerBuildInputs {
            artifact_dir: &tiny.artifact_dir,
            plan: tiny.plan.clone(),
            workload: &workload,
            cost_constants: &cost_constants,
            total_kv_bytes: 1024 * 1024,
            bytes_per_token: 128,
        })
        .expect("server builds from tiny artifact")
        .with_http_port(port);

        let handle = tokio::spawn(async move { server.serve().await });
        wait_for_metrics(port).await;
        Self { root, port, handle }
    }

    async fn post_completion(&self, body: &str) -> String {
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\n\
             Host: 127.0.0.1\r\n\
             Content-Type: application/json\r\n\
             Accept: text/event-stream\r\n\
             Connection: close\r\n\
             Content-Length: {}\r\n\
             \r\n\
             {}",
            body.len(),
            body
        );
        http_exchange(self.port, request)
            .await
            .expect("completion request")
    }

    async fn get_metrics(&self) -> String {
        let request =
            "GET /metrics HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".to_string();
        http_exchange(self.port, request)
            .await
            .expect("metrics request")
    }

    async fn shutdown(self) {
        self.handle.abort();
        let _ = self.handle.await;
        let _ = std::fs::remove_dir_all(self.root);
    }
}

async fn wait_for_metrics(port: u16) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let request =
            "GET /metrics HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".to_string();
        if let Ok(response) = http_exchange(port, request).await
            && response.starts_with("HTTP/1.1 200 OK")
        {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "server did not become ready on port {port}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn http_exchange(port: u16, request: String) -> io::Result<String> {
    let fut = async move {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).await?;
        stream.write_all(request.as_bytes()).await?;
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await?;
        Ok(String::from_utf8_lossy(&response).into_owned())
    };
    tokio::time::timeout(Duration::from_secs(30), fut)
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "HTTP exchange timed out"))?
}

fn unused_port() -> u16 {
    StdTcpListener::bind(("127.0.0.1", 0))
        .expect("bind ephemeral port")
        .local_addr()
        .expect("ephemeral local addr")
        .port()
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
}
