//! Axum completion endpoint + forward-pass driver.
//!
//! The HTTP front door tokenizes the prompt, admits the request through the
//! [`ContinuousBatcher`], and streams tokens back over SSE from the request's
//! [`TokenStreamer`]. A dedicated OS thread (the *driver*) owns the compiled
//! per-device runtimes and steps the batcher: each step composes a batch from
//! the in-flight set, runs one forward step through the
//! [`TopologyExecutor`], records the produced tokens back into the batcher,
//! advances/retires KV state, and emits step metrics.
//!
//! The driver runs on its own thread because the compiled runtimes own
//! `luminal::Graph` values that are neither `Send` nor `Sync`; it reaches the
//! async batcher / streamer / profile hooks through a captured
//! [`tokio::runtime::Handle`].
//!
//! Runtime backend selection mirrors `compile`/`verify`: the CUDA build drives
//! `CudaComputeRuntime`, the default CPU build drives `NativeComputeRuntime`.

use std::collections::HashMap;
use std::convert::Infallible;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use prometheus::{Encoder, TextEncoder};
use serde::Deserialize;
use tokio::net::TcpListener;
use tokio::runtime::Handle;
use tokio::sync::RwLock;

use super::lifecycle::Server;
use crate::batcher::{AdmissionDecision, ContinuousBatcher};
use crate::error::RuntimeError;
use crate::kv::PagedKVAllocator;
use crate::observability::ProfileHooks;
use crate::observability::metrics::{RequestMetrics, StepMetrics};
use crate::token_stream::{TokenSender, TokenStreamer};
use crate::tokenizer::SkeinTokenizer;
use crate::types::{IncomingRequest, RequestId, TokenOutput};
use crate::{InProcessCollective, observability};
use skein_compile::{
    ComputeRuntime, DEFAULT_SEARCH_BUDGET, SkeinArtifact, TopologyExecutor, TopologyStepBatch,
    load_runtime_segments,
};
use std::sync::Mutex as StdMutex;

/// Idle back-off when the in-flight set is empty: how long the driver sleeps
/// before polling the batcher again.
const DRIVER_IDLE_SLEEP: Duration = Duration::from_millis(2);

pub async fn serve(server: &Server) -> Result<(), RuntimeError> {
    #[cfg(feature = "cuda")]
    {
        serve_with::<skein_compile::CudaComputeRuntime>(server).await
    }
    #[cfg(not(feature = "cuda"))]
    {
        serve_with::<skein_compile::NativeComputeRuntime>(server).await
    }
}

/// Start the forward driver in the background without binding the HTTP
/// server. Used by integration tests that drive [`Server::submit`] directly;
/// the spawned driver thread keeps the cloned subsystem handles alive.
pub(crate) fn start_driver<R: ComputeRuntime + 'static>(
    server: &Server,
) -> Result<(), RuntimeError> {
    ServerState::start::<R>(server).map(|_| ())
}

async fn serve_with<R: ComputeRuntime + 'static>(server: &Server) -> Result<(), RuntimeError> {
    let state = ServerState::start::<R>(server)?;
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

/// Per-request prefix-cache hit counts, published by the driver at KV admit
/// time and consumed by the HTTP handler when it records the final request
/// metric. Shared because admission (driver thread) and metric recording
/// (HTTP task) happen on different threads.
type PrefixHits = Arc<StdMutex<HashMap<RequestId, u32>>>;

#[derive(Clone)]
struct ServerState {
    batcher: Arc<RwLock<ContinuousBatcher>>,
    profile: Arc<ProfileHooks>,
    prefix_hits: PrefixHits,
    /// The model's real tokenizer (`tokenizer.json` from the artifact). `None`
    /// falls back to the byte-mod-vocab tokenizer (token-ids only, no text).
    tokenizer: Option<Arc<SkeinTokenizer>>,
    vocab: u32,
}

impl ServerState {
    /// Spawn the driver thread and wait for it to finish loading the
    /// per-device runtimes. Returns once the driver is live (or with the
    /// load error the driver hit).
    fn start<R: ComputeRuntime + 'static>(server: &Server) -> Result<Arc<Self>, RuntimeError> {
        let handle = Handle::current();
        let artifact_dir = server.artifact_dir.clone();
        let batcher = server.batcher.clone();
        let kv = server.kv.clone();
        let profile = server.profile.clone();
        let prefix_hits: PrefixHits = Arc::new(StdMutex::new(HashMap::new()));
        let driver_prefix_hits = prefix_hits.clone();

        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<u32, String>>();

        std::thread::Builder::new()
            .name("skein-forward-driver".to_string())
            .spawn(move || {
                let driver = match Driver::load::<R>(
                    &artifact_dir,
                    batcher,
                    kv,
                    profile,
                    driver_prefix_hits,
                    handle,
                ) {
                    Ok(driver) => driver,
                    Err(err) => {
                        let _ = ready_tx.send(Err(err.to_string()));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(driver.vocab));
                driver.run();
            })
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;

        let vocab = ready_rx
            .recv()
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?
            .map_err(RuntimeError::ServerInit)?;

        // Load the model's real tokenizer if the artifact bundles one; warn and
        // fall back to byte-mod-vocab otherwise so the server still runs.
        let tokenizer = match SkeinTokenizer::from_artifact_dir(&server.artifact_dir) {
            Ok(Some(t)) => Some(Arc::new(t)),
            Ok(None) => {
                tracing::warn!(
                    "no tokenizer.json in artifact; using byte-fallback tokenizer (token ids only)"
                );
                None
            }
            Err(e) => {
                tracing::warn!(%e, "failed to load tokenizer.json; using byte-fallback tokenizer");
                None
            }
        };

        Ok(Arc::new(Self {
            batcher: server.batcher.clone(),
            profile: server.profile.clone(),
            tokenizer,
            prefix_hits,
            vocab,
        }))
    }
}

/// The forward-pass driver. Owns the compiled per-device runtimes (`!Send`,
/// hence the dedicated thread) and steps the shared batcher.
struct Driver {
    runtimes: Vec<Vec<skein_compile::RuntimeSegment>>,
    collectives: InProcessCollective,
    sequencing: Vec<skein_emit::SequenceStep>,
    batcher: Arc<RwLock<ContinuousBatcher>>,
    kv: Arc<StdMutex<PagedKVAllocator>>,
    profile: Arc<ProfileHooks>,
    prefix_hits: PrefixHits,
    handle: Handle,
    vocab: u32,
}

impl Driver {
    fn load<R: ComputeRuntime + 'static>(
        artifact_dir: &std::path::Path,
        batcher: Arc<RwLock<ContinuousBatcher>>,
        kv: Arc<StdMutex<PagedKVAllocator>>,
        profile: Arc<ProfileHooks>,
        prefix_hits: PrefixHits,
        handle: Handle,
    ) -> Result<Self, RuntimeError> {
        let artifact = SkeinArtifact::load(artifact_dir)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
        let vocab = artifact.plan.model_meta.vocab as u32;
        let sequencing = artifact.sequencing.clone();
        let runtimes = load_runtime_segments::<R>(&artifact, DEFAULT_SEARCH_BUDGET)
            .map_err(|e| RuntimeError::ServerInit(e.to_string()))?;
        let collectives = InProcessCollective::new(artifact.devices.len());
        Ok(Self {
            runtimes,
            collectives,
            sequencing,
            batcher,
            kv,
            profile,
            prefix_hits,
            handle,
            vocab,
        })
    }

    /// Drop published prefix-hit entries for requests that will never reach a
    /// final token (failed/panicked steps), so the shared map can't leak.
    fn clear_prefix_hits(&self, ids: &[RequestId]) {
        if let Ok(mut map) = self.prefix_hits.lock() {
            for id in ids {
                map.remove(id);
            }
        }
    }

    fn run(mut self) {
        let mut step_idx: u64 = 0;
        loop {
            let now = observability::wall_now_ms();

            // Promote any delayed requests whose wait has elapsed, then snapshot
            // the next step's batch and each request's running token sequence.
            let snapshot = self.handle.block_on(async {
                let mut b = self.batcher.write().await;
                b.promote_ready(now);
                let step = b.next_step_batch();
                let prefill: Vec<RequestId> = step.prefill_requests.clone();
                let ids: Vec<RequestId> = step
                    .prefill_requests
                    .iter()
                    .chain(step.decode_requests.iter())
                    .copied()
                    .collect();
                let seqs: Vec<Vec<u32>> = ids
                    .iter()
                    .map(|id| b.current_sequence(*id).unwrap_or_default())
                    .collect();
                Snapshot {
                    ids,
                    seqs,
                    prefill,
                    uniform_decode_size: step.uniform_decode_size,
                }
            });

            if snapshot.ids.is_empty() {
                std::thread::sleep(DRIVER_IDLE_SLEEP);
                continue;
            }

            // Structured per-step span: any installed tracing subscriber
            // (incl. an OTLP/OpenTelemetry layer) records the step. Entered on
            // this sync driver thread for the duration of the step.
            let _step_span =
                observability::tracing_spans::step_span(step_idx, snapshot.ids.len() as u32)
                    .entered();

            // Reserve KV pages for prefilling requests (radix prefix reuse
            // happens inside `admit`, which reports how many prompt tokens were
            // served from the cache). A KV failure fails just that request.
            let mut failed: Vec<RequestId> = Vec::new();
            let mut hits: Vec<(RequestId, u32)> = Vec::new();
            if let Ok(mut alloc) = self.kv.lock() {
                let prefill: std::collections::HashSet<RequestId> =
                    snapshot.prefill.iter().copied().collect();
                for (id, seq) in snapshot.ids.iter().zip(snapshot.seqs.iter()) {
                    if prefill.contains(id) {
                        match alloc.admit(*id, seq) {
                            Ok(pt) => hits.push((*id, pt.prefix_hit_tokens)),
                            Err(err) => {
                                tracing::warn!(request = id.0, %err, "kv admit failed");
                                failed.push(*id);
                            }
                        }
                    }
                }
            }
            if !hits.is_empty() {
                if let Ok(mut map) = self.prefix_hits.lock() {
                    for (id, hit) in hits {
                        map.insert(id, hit);
                    }
                }
            }

            // Run one forward step for the whole batch. The executor calls into
            // the compiled backend (luminal); a backend panic must not kill the
            // driver, so it is caught and treated as a failed step — the batch
            // is retired and the loop continues.
            let started = Instant::now();
            let batch = TopologyStepBatch {
                request_tokens: snapshot.seqs.clone(),
            };
            let result = match std::panic::catch_unwind(AssertUnwindSafe(|| {
                TopologyExecutor::new(&mut self.runtimes, &self.collectives, &self.sequencing)
                    .execute_for_step(&batch)
            })) {
                Ok(inner) => inner.map_err(|e| e.to_string()),
                Err(_) => Err("forward step panicked in the compute backend".to_string()),
            };
            let compute_us = started.elapsed().as_secs_f64() * 1_000_000.0;

            match result {
                Ok(out) => {
                    let batch_size = snapshot.ids.len() as u32;
                    let uniform = snapshot.uniform_decode_size;
                    let kv = self.kv.clone();
                    let batcher = self.batcher.clone();
                    let profile = self.profile.clone();
                    let ids = snapshot.ids.clone();
                    self.handle.block_on(async move {
                        // 1. Record each produced token in the batcher.
                        let mut sends: Vec<(TokenSender, TokenOutput, RequestId, bool)> =
                            Vec::with_capacity(ids.len());
                        {
                            let mut b = batcher.write().await;
                            for (id, token) in ids.iter().zip(out.next_tokens.iter()) {
                                if let Some(acc) = b.accept_token(*id, *token) {
                                    sends.push((
                                        acc.sender,
                                        TokenOutput {
                                            token: *token,
                                            is_final: acc.is_final,
                                        },
                                        *id,
                                        acc.is_final,
                                    ));
                                }
                            }
                        }
                        // 2. Grow KV for requests that will continue decoding.
                        let kv_pages = {
                            let mut pages = 0;
                            if let Ok(mut alloc) = kv.lock() {
                                for (_, tok, id, is_final) in &sends {
                                    if !is_final {
                                        let _ = alloc.advance(*id, &[tok.token]);
                                    }
                                }
                                pages = alloc.in_use_pages();
                            }
                            pages
                        };
                        // 3. Stream tokens (async send, lock released).
                        for (sender, tok, _, _) in &sends {
                            let _ = sender.send(*tok).await;
                        }
                        // 4. Retire finished requests (closes stream + frees KV).
                        {
                            let mut b = batcher.write().await;
                            for (_, _, id, is_final) in &sends {
                                if *is_final {
                                    let _ = b.retire(*id);
                                }
                            }
                        }
                        profile
                            .record_step(StepMetrics {
                                step_idx,
                                compute_us,
                                comm_us: 0.0,
                                kv_pages_in_use: kv_pages,
                                batch_size,
                                uniform_decode_size: uniform,
                            })
                            .await;
                    });
                    step_idx += 1;
                }
                Err(err) => {
                    // The whole step failed (error or caught panic): retire every
                    // request in the batch so their streams close rather than
                    // hang, and the driver moves on to the next step.
                    tracing::error!(%err, "forward step failed");
                    self.clear_prefix_hits(&snapshot.ids);
                    let batcher = self.batcher.clone();
                    let ids = snapshot.ids.clone();
                    self.handle.block_on(async move {
                        let mut b = batcher.write().await;
                        for id in &ids {
                            let _ = b.retire(*id);
                        }
                    });
                }
            }

            // Retire requests whose KV admission failed.
            if !failed.is_empty() {
                self.clear_prefix_hits(&failed);
                let batcher = self.batcher.clone();
                self.handle.block_on(async move {
                    let mut b = batcher.write().await;
                    for id in &failed {
                        let _ = b.retire(*id);
                    }
                });
            }
        }
    }
}

struct Snapshot {
    ids: Vec<RequestId>,
    seqs: Vec<Vec<u32>>,
    prefill: Vec<RequestId>,
    uniform_decode_size: Option<u32>,
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
    let max_output_tokens = req.max_tokens.unwrap_or(1).max(1);
    let _stream = req.stream.unwrap_or(true);
    let now_ms = observability::wall_now_ms();
    // Encode with the model's real tokenizer when available; fall back to the
    // byte tokenizer otherwise.
    let prompt_tokens = match &state.tokenizer {
        Some(tok) => match tok.encode(&req.prompt) {
            Ok(ids) => ids,
            Err(e) => return (StatusCode::BAD_REQUEST, format!("tokenize: {e}")).into_response(),
        },
        None => tokenize_prompt(&req.prompt, state.vocab),
    };

    // Structured request span/event for tracing subscribers (incl. OTLP).
    // `in_scope` keeps it on the synchronous portion — no span guard held
    // across an await.
    observability::tracing_spans::request_span(request_id, prompt_tokens.len() as u32)
        .in_scope(|| tracing::info!("skein.request received"));

    // Admit through the batcher; the driver picks the request up on its next
    // step and streams tokens on `streamer`.
    let (streamer, sender) = TokenStreamer::paired();
    let incoming = IncomingRequest {
        id: request_id,
        prompt_tokens,
        max_output_tokens,
        arrival_ms: now_ms,
    };
    let decision = state
        .batcher
        .write()
        .await
        .admit(incoming, sender, now_ms);
    if let AdmissionDecision::Reject { reason } = decision {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            format!("request rejected: {reason:?}"),
        )
            .into_response();
    }

    let started = Instant::now();
    let stream = futures_util::stream::unfold(
        StreamState {
            streamer,
            profile: state.profile.clone(),
            prefix_hits: state.prefix_hits.clone(),
            tokenizer: state.tokenizer.clone(),
            generated: Vec::new(),
            emitted_text: String::new(),
            request_id,
            started,
            latencies: Vec::new(),
            count: 0,
        },
        |mut st| async move {
            let token = st.streamer.recv().await?;
            st.count += 1;
            let elapsed = st.started.elapsed().as_secs_f64() * 1000.0;
            st.latencies.push(elapsed);
            // Incremental detokenization: decode the running output and emit
            // the new text suffix. Empty when there's no real tokenizer.
            let text_delta = if let Some(tok) = &st.tokenizer {
                st.generated.push(token.token);
                match tok.decode(&st.generated) {
                    Ok(full) => {
                        let cut = st.emitted_text.len().min(full.len());
                        let delta = full[cut..].to_string();
                        st.emitted_text = full;
                        delta
                    }
                    Err(_) => String::new(),
                }
            } else {
                String::new()
            };
            if token.is_final {
                // The driver published this request's radix prefix-cache hit
                // count at KV admit time; consume it here (default 0 if the
                // request never prefilled, e.g. prefix cache disabled).
                let prefix_cache_hit_tokens = st
                    .prefix_hits
                    .lock()
                    .ok()
                    .and_then(|mut m| m.remove(&st.request_id))
                    .unwrap_or(0);
                st.profile
                    .record_request(RequestMetrics {
                        request_id: st.request_id,
                        ttft_ms: st.latencies.first().copied().unwrap_or(0.0),
                        per_token_latency_ms: st.latencies.clone(),
                        total_wall_ms: elapsed,
                        prefix_cache_hit_tokens,
                        output_tokens: st.count,
                        completed_at_ms: observability::wall_now_ms(),
                    })
                    .await;
            }
            let data = serde_json::json!({
                "id": st.request_id.0,
                "token": token.token,
                "text": text_delta,
                "is_final": token.is_final,
            })
            .to_string();
            let event = Ok::<_, Infallible>(Event::default().event("token").data(data));
            Some((event, st))
        },
    );
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::new())
        .into_response()
}

/// Per-connection SSE state threaded through `stream::unfold`.
struct StreamState {
    streamer: TokenStreamer,
    profile: Arc<ProfileHooks>,
    prefix_hits: PrefixHits,
    tokenizer: Option<Arc<SkeinTokenizer>>,
    /// Running output token ids, for incremental detokenization.
    generated: Vec<u32>,
    /// Text already emitted, so each event carries only the new suffix.
    emitted_text: String,
    request_id: RequestId,
    started: Instant,
    latencies: Vec<f64>,
    count: u32,
}

async fn handle_metrics(State(state): State<Arc<ServerState>>) -> Response {
    let encoder = TextEncoder::new();
    let mut body = Vec::new();
    match encoder.encode(&state.profile.registry().gather(), &mut body) {
        Ok(()) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4",
            )],
            body,
        )
            .into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

/// Placeholder tokenizer: maps prompt bytes to ids mod vocab. Real tokenizer
/// integration is tracked separately; this keeps the serve path runnable end
/// to end until the artifact carries tokenizer assets.
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
