//! Slow Mac CLI integrations over a tiny Mixtral-shaped model.

#[allow(dead_code)]
#[path = "../../skein_parity/tests/fixtures/build_tiny_artifact.rs"]
mod tiny_artifact;

use std::io;
use std::net::TcpListener as StdTcpListener;
use std::path::{Path, PathBuf};
use std::time::Duration;

use skein_cli::cli::{CompileArgs, OutputFormat, ServeArgs, VerifyArgs};
use skein_cli::cmd;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[test]
#[ignore = "runs Luminal compile for a tiny end-to-end artifact"]
fn skein_compile_verify_tiny_mac() {
    let root = unique_temp_dir("skein-cli-e2e");
    let tiny = tiny_artifact::build_tiny_artifact(&root.join("seed"));
    let inputs = write_inputs(&root, &tiny.root.join("tiny.weights.safetensors"));
    let out = root.join("artifacts");

    cmd::compile::run(
        CompileArgs {
            model: inputs.model,
            cluster: inputs.cluster,
            trace: inputs.trace,
            drift: inputs.drift,
            cost: inputs.cost.clone(),
            weights: inputs.weights,
            out: out.clone(),
            search_budget: 1,
            enforce_parity: false,
            parity_prompts_path: Some(inputs.prompts.clone()),
            n_parity_prompts: 2,
            disaggregated: false,
        },
        OutputFormat::Text,
    )
    .expect("compile tiny artifact");

    let latest = out.join("LATEST");
    assert!(latest.exists());
    let artifact = std::fs::read_link(&latest).expect("LATEST symlink");
    let report_path = artifact.join("parity_report.json");
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).expect("parity report")).unwrap();
    assert!(report.get("per_prompt").is_some());

    cmd::verify::run(
        VerifyArgs {
            artifact,
            reference: None,
            sample_from: Some(inputs.prompts),
            n_prompts: 2,
            cost: inputs.cost,
            enforce: false,
        },
        OutputFormat::Json,
    )
    .expect("verify tiny artifact");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "runs native Luminal compile for the tiny serving artifact"]
async fn skein_serve_cli_tiny_mac() {
    let root = unique_temp_dir("skein-cli-serve");
    let tiny = tiny_artifact::build_tiny_artifact(&root.join("seed"));
    let inputs = write_inputs(&root, &tiny.root.join("tiny.weights.safetensors"));
    let port = unused_port();

    let handle = tokio::spawn({
        let artifact = tiny.artifact_dir.clone();
        let workload = inputs.trace.clone();
        let cost = inputs.cost.clone();
        async move {
            cmd::serve::run(
                ServeArgs {
                    artifact,
                    workload,
                    cost,
                    port,
                    enable_hot_swap: false,
                    total_kv_bytes: 1024 * 1024,
                    bytes_per_token: 128,
                },
                OutputFormat::Text,
            )
            .await
        }
    });

    wait_for_metrics(port).await;
    let response = post_completion(port, r#"{"prompt":"Hello","max_tokens":1,"stream":true}"#)
        .await
        .expect("completion request");
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert!(response.contains("event: token"), "{response}");

    handle.abort();
    let _ = handle.await;
    let _ = std::fs::remove_dir_all(root);
}

struct Inputs {
    model: PathBuf,
    cluster: PathBuf,
    trace: PathBuf,
    drift: PathBuf,
    cost: PathBuf,
    weights: PathBuf,
    prompts: PathBuf,
}

fn write_inputs(root: &Path, source_weights: &Path) -> Inputs {
    std::fs::create_dir_all(root).unwrap();
    let model = root.join("config.json");
    std::fs::write(&model, tiny_config()).unwrap();
    let cluster = root.join("cluster.toml");
    std::fs::write(
        &cluster,
        r#"
num_devices = 1
[[node]]
id = "node0"
devices = ["d0"]
device_kind = "h100_sxm5"
device_memory_gb = 80
"#,
    )
    .unwrap();
    let trace = root.join("trace.jsonl");
    std::fs::write(
        &trace,
        r#"{"slo":{"ttft_p95_ms":500,"tpot_p95_ms":50,"max_accuracy_drift":0.01,"recompile_drift_threshold_kl":0.05}}
{"prompt_tokens":8,"output_tokens":2,"arrival_ms":0}
"#,
    )
    .unwrap();
    let prompts = root.join("prompts.jsonl");
    std::fs::write(
        &prompts,
        r#"{"prompt":"Alice was beginning to get tired."}
{"prompt":"The rabbit hurried down the passage."}
"#,
    )
    .unwrap();
    let drift = root.join("drift.toml");
    std::fs::write(&drift, bf16_only_drift()).unwrap();
    let cost = root.join("cost_constants.toml");
    std::fs::write(&cost, include_str!("../../../cluster/cost_constants.toml")).unwrap();
    let weights = root.join("weights");
    std::fs::create_dir_all(&weights).unwrap();
    std::fs::copy(source_weights, weights.join("weights.safetensors")).unwrap();
    Inputs {
        model,
        cluster,
        trace,
        drift,
        cost,
        weights,
        prompts,
    }
}

fn tiny_config() -> &'static str {
    r#"{
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "vocab_size": 128,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "hidden_act": "silu"
    }"#
}

fn bf16_only_drift() -> &'static str {
    r#"
[default.weight]
bf16 = 0.0
fp16 = 10.0
fp8_e4m3 = 10.0
fp8_e5m2 = 10.0
int8 = 10.0
int4 = 10.0

[default.activation]
bf16 = 0.0
fp16 = 10.0
fp8_e4m3 = 10.0
fp8_e5m2 = 10.0
int8 = 10.0
int4 = 10.0

[default.kv_cache]
bf16 = 0.0
fp16 = 10.0
fp8_e4m3 = 10.0
fp8_e5m2 = 10.0
int8 = 10.0
int4 = 10.0
"#
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
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

async fn post_completion(port: u16, body: &str) -> io::Result<String> {
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
    http_exchange(port, request).await
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
