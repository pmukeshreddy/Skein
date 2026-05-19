//! Axum completion endpoint and native forward worker.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use prometheus::{Encoder, TextEncoder};
use serde::Deserialize;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use super::lifecycle::Server;
use crate::error::RuntimeError;
use crate::observability::metrics::RequestMetrics;
use crate::types::{RequestId, TokenOutput};
use crate::{MockCollective, observability};
use skein_compile::{
    SkeinArtifact, TopologyExecutor, TopologyStepBatch, load_native_runtime_segments,
};

pub async fn serve(server: &Server) -> Result<(), RuntimeError> {
    let state = ServerState::start(server)?;
    let app = Router::new()
        .route("/v1/completions", post(handle_completion))
        .route("/metrics", get(handle_metrics))
        .with_state(state);

    let listener = TcpListener::bind(("0.0.0.0", server.http_port))
        .await
        .map_err(|source| RuntimeError::HttpBind {
            port: server.http_port,
            source,
        })?;
    axum::serve(listener, app)
        .await
        .map_err(|source| RuntimeError::Io {
            path: server.artifact_dir.clone(),
            source,
        })?;
    Ok(())
}

#[derive(Clone)]
struct ServerState {
    tx: std::sync::mpsc::Sender<WorkerCommand>,
    profile: Arc<crate::observability::ProfileHooks>,
    vocab: u32,
}

impl ServerState {
    fn start(server: &Server) -> Result<Arc<Self>, RuntimeError> {
        let artifact_dir = server.artifact_dir.clone();
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let init = (|| -> Result<WorkerState, String> {
                let artifact = SkeinArtifact::load(&artifact_dir).map_err(|e| e.to_string())?;
                let vocab = artifact.plan.model_meta.vocab as u32;
                let sequencing = artifact.sequencing.clone();
                let runtimes =
                    load_native_runtime_segments(&artifact).map_err(|e| e.to_string())?;
                let collectives = MockCollective::new(artifact.devices.len());
                Ok(WorkerState {
                    runtimes,
                    collectives,
                    sequencing,
                    vocab,
                })
            })();

            let mut worker = match init {
                Ok(worker) => {
                    let _ = ready_tx.send(Ok(worker.vocab));
                    worker
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };

            while let Ok(cmd) = cmd_rx.recv() {
                worker.handle(cmd);
            }
        });

        let vocab = ready_rx
            .recv()
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?
            .map_err(RuntimeError::ServerInit)?;
        Ok(Arc::new(Self {
            tx: cmd_tx,
            profile: server.profile.clone(),
            vocab,
        }))
    }
}

struct WorkerState {
    runtimes: Vec<Vec<skein_compile::RuntimeSegment>>,
    collectives: MockCollective,
    sequencing: Vec<skein_emit::SequenceStep>,
    vocab: u32,
}

impl WorkerState {
    fn handle(&mut self, cmd: WorkerCommand) {
        let mut tokens = cmd.prompt_tokens;
        for idx in 0..cmd.max_tokens {
            let batch = TopologyStepBatch {
                request_tokens: vec![tokens.clone()],
            };
            let result = TopologyExecutor::new(
                self.runtimes.as_mut_slice(),
                &self.collectives,
                &self.sequencing,
            )
            .execute_for_step(&batch);
            let token = match result {
                Ok(out) => out.next_tokens.first().copied().unwrap_or(0) % self.vocab,
                Err(err) => {
                    let _ = cmd.out.send(Err(err.to_string()));
                    return;
                }
            };
            tokens.push(token);
            let _ = cmd.out.send(Ok(TokenOutput {
                token,
                is_final: idx + 1 == cmd.max_tokens,
            }));
        }
    }
}

struct WorkerCommand {
    prompt_tokens: Vec<u32>,
    max_tokens: u32,
    out: mpsc::UnboundedSender<Result<TokenOutput, String>>,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    prompt: String,
    max_tokens: Option<u32>,
    stream: Option<bool>,
}

async fn handle_completion(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let request_id = RequestId::next();
    let max_tokens = req.max_tokens.unwrap_or(1).max(1);
    let _stream = req.stream.unwrap_or(true);
    let tokens = tokenize_prompt(&req.prompt, state.vocab);
    let (tx, rx) = mpsc::unbounded_channel();
    let command = WorkerCommand {
        prompt_tokens: tokens,
        max_tokens,
        out: tx,
    };
    if let Err(err) = state.tx.send(command) {
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
        )
            .into_response();
    }

    let started = Instant::now();
    let profile = state.profile.clone();
    let stream = futures_util::stream::unfold(
        (rx, profile, request_id, started, Vec::<f64>::new(), 0_u32),
        |(mut rx, profile, request_id, started, mut latencies, mut count)| async move {
            match rx.recv().await {
                Some(Ok(token)) => {
                    let received_at = Instant::now();
                    count += 1;
                    let elapsed = received_at.duration_since(started).as_secs_f64() * 1000.0;
                    latencies.push(elapsed);
                    if token.is_final {
                        let metric = RequestMetrics {
                            request_id,
                            ttft_ms: latencies.first().copied().unwrap_or(0.0),
                            per_token_latency_ms: latencies.clone(),
                            total_wall_ms: elapsed,
                            prefix_cache_hit_tokens: 0,
                            output_tokens: count,
                            completed_at_ms: observability::wall_now_ms(),
                        };
                        profile.record_request(metric).await;
                    }
                    let data = serde_json::json!({
                        "id": request_id.0,
                        "token": token.token,
                        "is_final": token.is_final,
                    })
                    .to_string();
                    Some((
                        Ok::<_, Infallible>(Event::default().event("token").data(data)),
                        (rx, profile, request_id, started, latencies, count),
                    ))
                }
                Some(Err(err)) => Some((
                    Ok::<_, Infallible>(Event::default().event("error").data(err)),
                    (rx, profile, request_id, started, latencies, count),
                )),
                None => None,
            }
        },
    );
    let keep_alive = axum::response::sse::KeepAlive::new();
    Sse::new(stream).keep_alive(keep_alive).into_response()
}

async fn handle_metrics(State(state): State<Arc<ServerState>>) -> Response {
    let encoder = TextEncoder::new();
    let mut body = Vec::new();
    match encoder.encode(&state.profile.registry().gather(), &mut body) {
        Ok(()) => (
            axum::http::StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4",
            )],
            body,
        )
            .into_response(),
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
        )
            .into_response(),
    }
}

fn tokenize_prompt(prompt: &str, vocab: u32) -> Vec<u32> {
    let modulo = vocab.max(1);
    let mut tokens = prompt
        .as_bytes()
        .iter()
        .map(|b| (*b as u32) % modulo)
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        tokens.push(0);
    }
    tokens
}
