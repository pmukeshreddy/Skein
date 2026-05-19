//! Test 11 — Server surfaces work without CUDA.

mod common;
use common::*;

use std::path::PathBuf;

use skein_ir::types::BatchPolicy;
use skein_runtime::batcher::AdmissionDecision;
use skein_runtime::server::lifecycle::ServerBuildInputs;
use skein_runtime::types::{IncomingRequest, RequestId};
use skein_runtime::{RuntimeError, Server};

fn build_server() -> Server {
    let cost_constants = load_cost_constants();
    let plan = mk_plan(32, true, BatchPolicy::Continuous { max_batch: 4 });
    let workload = mk_workload(500, 50);
    let artifact_dir = PathBuf::from("/tmp/skein_rt_server_test_artifact");
    std::fs::create_dir_all(&artifact_dir).unwrap();
    Server::new(ServerBuildInputs {
        artifact_dir: &artifact_dir,
        plan,
        workload: &workload,
        cost_constants: &cost_constants,
        total_kv_bytes: 1024 * 32 * 32,
        bytes_per_token: 32,
    })
    .expect("Server::new should succeed on Phase A")
}

#[tokio::test]
async fn server_serve_reports_missing_artifact_without_cuda_gate() {
    let server = build_server();
    let err = server.serve().await.unwrap_err();
    assert!(matches!(err, RuntimeError::ServerInit(_)));
}

#[tokio::test]
async fn server_admit_works_without_cuda() {
    let server = build_server();
    let req = IncomingRequest {
        id: RequestId::next(),
        prompt_tokens: (0..32).collect(),
        max_output_tokens: 8,
        arrival_ms: 0,
    };
    let (decision, _streamer) = server.admit_for_test(req).await;
    assert_eq!(decision, AdmissionDecision::Admit);
}
