//! `skein serve` — runtime server.
//!
//! Three modes, dispatched in `run`:
//! 1. **Rank process** (env `SKEIN_RANK` set by the launcher) → join the NCCL
//!    group and run distributed generation on this rank's GPU.
//! 2. **Launcher** (`--gpus 0,1,...`) → spawn one rank process per GPU.
//! 3. **Single process** (default) → the local HTTP serving path.

use skein_compile::SkeinArtifact;
use skein_runtime::Server;
use skein_runtime::distributed::{WorldLayout, launcher};
use skein_runtime::server::lifecycle::ServerBuildInputs;

use crate::cli::{OutputFormat, ServeArgs};
use crate::error::CliError;
use crate::load::{load_cost_model, load_workload};

pub async fn run(args: ServeArgs, output: OutputFormat) -> Result<(), CliError> {
    // Mode 1: launched as a rank (env set by the launcher).
    if let Some(layout) = WorldLayout::from_env() {
        let layout = layout.map_err(|e| CliError::BadArgument(e.to_string()))?;
        return run_rank(args, layout).await;
    }
    // Mode 2: launcher — spawn one process per GPU.
    if let Some(gpus) = args.gpus.clone() {
        return launch_ranks(&args, &gpus);
    }
    // Mode 3a: single-process continuous-batching demo.
    if args.batch_demo {
        return run_batch_demo(args).await;
    }
    // Mode 3b: single-process HTTP serve.
    run_single_process(args, output).await
}

/// Single-process continuous-batching demo: drive the `ContinuousBatcher` over
/// the paged-KV `ContinuousBatchDriver`. Phase A submits several prompts
/// concurrently (admit → mixed prefill/decode batch → retire, with optional
/// CUDA-graph decode replay); Phase B re-submits a prompt that shares Phase A's
/// prefix to exercise cross-request prefix reuse. Prints per-request output and
/// driver metrics.
#[cfg(feature = "cuda")]
async fn run_batch_demo(args: ServeArgs) -> Result<(), CliError> {
    use skein_compile::{CudaComputeRuntime, DEFAULT_SEARCH_BUDGET};
    use skein_runtime::SkeinTokenizer;
    use skein_runtime::batcher::ContinuousBatcher;
    use skein_runtime::distributed::ContinuousBatchDriver;
    use skein_runtime::kv::PagedKVAllocator;
    use std::sync::{Arc, Mutex};

    let artifact = SkeinArtifact::load(&args.artifact)?;
    let workload = load_workload(&args.workload)?;
    let cost_model = load_cost_model(&args.cost)?;
    let cost_constants = cost_model.constants().clone();
    let plan = artifact.plan.clone();
    let artifact_dir = args.artifact.clone();
    let cuda_graphs = args.cuda_graphs;
    let total_kv_bytes = args.total_kv_bytes;
    let bytes_per_token = args.bytes_per_token;

    // Tokenize the demo prompts with the model's real tokenizer when bundled.
    let tokenizer = SkeinTokenizer::from_artifact_dir(&artifact_dir).ok().flatten();
    let vocab = plan.model_meta.vocab as u32;
    // SKEIN_PROMPTS_FILE: a JSON array of prompt strings (handles commas/newlines
    // that the comma-separated `--demo-prompts` cannot) — used to drive the
    // continuous batcher with real ShareGPT conversations.
    let prompts: Vec<String> = if let Some(path) = std::env::var_os("SKEIN_PROMPTS_FILE") {
        let bytes = std::fs::read(&path)
            .map_err(|e| CliError::BadArgument(format!("read SKEIN_PROMPTS_FILE: {e}")))?;
        let v: Vec<String> = serde_json::from_slice(&bytes)
            .map_err(|e| CliError::BadArgument(format!("parse SKEIN_PROMPTS_FILE (expect JSON array of strings): {e}")))?;
        v.into_iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
    } else {
        match args.demo_prompts {
            Some(s) => s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect(),
            None => vec![
                "The capital of France is".to_string(),
                "The capital of France is Paris".to_string(),
            ],
        }
    };
    let encode = |p: &str| -> Vec<u32> {
        match &tokenizer {
            Some(t) => t.encode(p).unwrap_or_default(),
            None => {
                let m = vocab.max(1);
                p.bytes().map(|b| (b as u32) % m).collect()
            }
        }
    };
    let tokenized: Vec<(String, Vec<u32>)> =
        prompts.iter().map(|p| (p.clone(), encode(p))).collect();
    let max_new = args.max_new_tokens.max(1);

    // GPU work is sync + !Send-bound to its thread; run it off the async runtime.
    let report = tokio::task::spawn_blocking(move || -> Result<String, String> {
        let kv = PagedKVAllocator::new(
            &plan,
            total_kv_bytes,
            bytes_per_token,
            cost_constants.runtime.radix_max_depth,
        )
        .map_err(|e| e.to_string())?;
        let batcher = ContinuousBatcher::new(&plan, &workload, &cost_constants, Arc::new(Mutex::new(kv)));
        let mut driver = ContinuousBatchDriver::load::<CudaComputeRuntime>(
            &artifact_dir,
            batcher,
            DEFAULT_SEARCH_BUDGET,
            cuda_graphs,
        )
        .map_err(|e| e.to_string())?;

        // SKEIN_LOCKSTEP: synchronous batched decode of exactly `max_batch` real
        // prompts in lockstep (one batched forward per token). Reports real
        // per-row generated tokens + decode throughput (batch*max_new / decode_s).
        if std::env::var_os("SKEIN_LOCKSTEP").is_some() {
            let gb = plan.batching.max_batch() as usize;
            let mut bp: Vec<Vec<u32>> = tokenized.iter().map(|(_, t)| t.clone()).collect();
            if bp.is_empty() {
                return Err("no prompts".to_string());
            }
            let src_len = bp.len();
            while bp.len() < gb {
                bp.push(bp[bp.len() % src_len].clone());
            }
            bp.truncate(gb);
            let (genr, dt) = driver
                .run_batched_lockstep(&bp, max_new)
                .map_err(|e| e.to_string())?;
            let mut s = format!(
                "=== Synchronous batched decode (real prompts, lockstep) ===\n  \
                 batch={gb} max_new={max_new} decode_s={dt:.3} \
                 decode_tokens_per_s={:.1}\n",
                (gb * max_new) as f64 / dt.max(1e-9),
            );
            for (r, g) in genr.iter().enumerate() {
                let head: Vec<u32> = g.iter().take(8).copied().collect();
                s.push_str(&format!("  row{r} tokens[..8]={head:?}\n"));
            }
            return Ok(s);
        }
        let now_ms = || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        };
        let mut out = String::new();
        // Phase A: submit all prompts concurrently → continuous batching.
        let now = now_ms();
        for (_p, toks) in &tokenized {
            driver
                .submit(toks.clone(), max_new, now)
                .map_err(|e| e.to_string())?;
        }
        let wall = std::time::Instant::now();
        let results = driver.run_to_completion(now).map_err(|e| e.to_string())?;
        let wall_s = wall.elapsed().as_secs_f64();
        out.push_str("=== Phase A: concurrent continuous batching ===\n");
        let mut total_decode_tokens = 0usize;
        for r in &results {
            // Decode tokens = generated tokens beyond the prompt (one per decode step).
            total_decode_tokens += r.tokens.len();
            out.push_str(&format!(
                "  req#{} prompt_len={} prefix_hit={} prefill_steps={} tokens={:?}\n",
                r.order, r.prompt_len, r.prefix_hit_tokens, r.prefill_steps, r.tokens
            ));
        }
        let agg = if wall_s > 0.0 { total_decode_tokens as f64 / wall_s } else { 0.0 };
        out.push_str(&format!(
            "=== Aggregate (Phase A) ===\n  requests={} total_generated_tokens={} wall_s={:.3} \
             aggregate_tokens_per_s={:.2}\n",
            results.len(),
            total_decode_tokens,
            wall_s,
            agg,
        ));

        // Phase B: re-submit the first prompt — its KV pages are still cached,
        // so the shared prefix is reused (prefix_hit > 0) with fewer prefill
        // forward steps.
        if let Some((_p, toks)) = tokenized.first() {
            let now2 = now_ms();
            driver.submit(toks.clone(), max_new, now2).map_err(|e| e.to_string())?;
            let warm = driver.run_to_completion(now2).map_err(|e| e.to_string())?;
            out.push_str("=== Phase B: re-submit prompt #0 (cross-request prefix reuse) ===\n");
            for r in &warm {
                out.push_str(&format!(
                    "  prompt_len={} prefix_hit={} prefill_steps={} tokens={:?}\n",
                    r.prompt_len, r.prefix_hit_tokens, r.prefill_steps, r.tokens
                ));
            }
        }

        let m = driver.metrics();
        out.push_str(&format!(
            "=== Driver metrics ===\n  batch_steps={} forward_steps={} (prefill={} decode={})\n  \
             max_concurrent_inflight={} mixed_batch_steps={}\n  \
             cuda_graph_captures={} cuda_graph_replays={}\n  total_compute_ms={:.1}\n",
            m.batch_steps,
            m.forward_steps,
            m.prefill_forward_steps,
            m.decode_forward_steps,
            m.max_concurrent_inflight,
            m.mixed_batch_steps,
            m.graph_captures,
            m.graph_replays,
            m.total_compute_us / 1000.0,
        ));
        Ok(out)
    })
    .await
    .map_err(|e| CliError::BadArgument(format!("batch demo task panicked: {e}")))?
    .map_err(CliError::BadArgument)?;

    println!("{report}");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
async fn run_batch_demo(_args: ServeArgs) -> Result<(), CliError> {
    Err(CliError::RequiresCuda {
        what: "continuous-batching demo",
        reason: "The continuous-batch driver runs CudaComputeRuntime on the GPU.",
        suggested_fix: "Build with CUDA (default) and run: skein serve --batch-demo ...",
    })
}

async fn run_single_process(args: ServeArgs, _output: OutputFormat) -> Result<(), CliError> {
    tracing::info!(
        "skein serve: loading artifact from {}",
        args.artifact.display()
    );
    let artifact = SkeinArtifact::load(&args.artifact)?;
    let workload = load_workload(&args.workload)?;
    let cost_model = load_cost_model(&args.cost)?;
    let cost_constants = cost_model.constants();

    let inputs = ServerBuildInputs {
        artifact_dir: &args.artifact,
        plan: artifact.plan.clone(),
        workload: &workload,
        cost_constants,
        total_kv_bytes: args.total_kv_bytes,
        bytes_per_token: args.bytes_per_token,
    };

    let server = Server::new(inputs)?.with_http_port(args.port);
    tracing::info!("skein serve: HTTP listening on 0.0.0.0:{}", args.port);
    server.serve().await?;
    Ok(())
}

/// Launcher: spawn one `skein serve` rank process per GPU, each pinned to its
/// GPU via `CUDA_VISIBLE_DEVICES` with the rank/world/rendezvous environment.
fn launch_ranks(args: &ServeArgs, gpus: &str) -> Result<(), CliError> {
    let gpus: Vec<usize> = gpus
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|e| CliError::BadArgument(format!("--gpus must be a comma list of ints: {e}")))?;

    let specs =
        launcher::plan_launch(&gpus, &args.rendezvous).map_err(|e| CliError::BadArgument(e.to_string()))?;
    let exe = std::env::current_exe()?;
    let exe = exe
        .to_str()
        .ok_or_else(|| CliError::BadArgument("executable path is not valid UTF-8".to_string()))?
        .to_string();
    let child_args = child_serve_args(args);

    // Best-effort: remove a stale rendezvous file so this run negotiates fresh.
    let _ = std::fs::remove_file(&args.rendezvous);

    let mut children = Vec::new();
    for spec in &specs {
        let mut cmd = launcher::build_command(&exe, &child_args, spec);
        tracing::info!(rank = spec.rank, gpu = spec.gpu, "launching rank process");
        children.push(cmd.spawn()?);
    }
    let mut failures = 0;
    for (i, mut child) in children.into_iter().enumerate() {
        let status = child.wait()?;
        if !status.success() {
            failures += 1;
            tracing::error!(rank = i, ?status, "rank process exited non-zero");
        }
    }
    if failures > 0 {
        return Err(CliError::BadArgument(format!(
            "{failures} rank process(es) failed"
        )));
    }
    Ok(())
}

/// The `serve` argument vector a launched rank runs — same artifact/prompt
/// settings, **without** `--gpus` (so the child enters rank mode, not the
/// launcher again). Rank/world/rendezvous come from the environment.
fn child_serve_args(args: &ServeArgs) -> Vec<String> {
    let mut v = vec![
        "serve".to_string(),
        "--artifact".to_string(),
        args.artifact.display().to_string(),
        "--workload".to_string(),
        args.workload.display().to_string(),
        "--cost".to_string(),
        args.cost.display().to_string(),
        "--max-new-tokens".to_string(),
        args.max_new_tokens.to_string(),
        "--rendezvous".to_string(),
        args.rendezvous.display().to_string(),
    ];
    if let Some(prompt) = &args.prompt {
        v.push("--prompt".to_string());
        v.push(prompt.clone());
    }
    v
}

#[cfg(feature = "cuda")]
async fn run_rank(args: ServeArgs, layout: WorldLayout) -> Result<(), CliError> {
    use skein_runtime::SkeinTokenizer;
    use skein_runtime::distributed::gpu_rank;

    let prompt = args.prompt.clone().unwrap_or_default();
    let artifact_dir = args.artifact.clone();
    let rendezvous = args.rendezvous.clone();
    let max_new = args.max_new_tokens;
    let tokenizer = SkeinTokenizer::from_artifact_dir(&artifact_dir)
        .ok()
        .flatten();

    // SKEIN_PIPELINE_STREAMS=N (N>=2): drive N concurrent streams with 1F1B
    // pipeline overlap so both PP stages (GPUs) compute at once. Unset/1 = the
    // single-stream path.
    let pipeline_streams = std::env::var("SKEIN_PIPELINE_STREAMS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 2);

    tracing::info!(
        rank = layout.rank,
        world_size = layout.world_size,
        pipeline_streams = pipeline_streams.unwrap_or(1),
        "rank serving: bootstrapping NCCL + loading device segments"
    );
    // NCCL + the compute runtime are blocking/sync — run off the async runtime.
    let out = tokio::task::spawn_blocking(move || match pipeline_streams {
        Some(n) => gpu_rank::run_generation_pipelined(
            &artifact_dir,
            layout,
            &rendezvous,
            &prompt,
            max_new,
            n,
            tokenizer.as_ref(),
        ),
        None => gpu_rank::run_generation(
            &artifact_dir,
            layout,
            &rendezvous,
            &prompt,
            max_new,
            tokenizer.as_ref(),
        ),
    })
    .await
    .map_err(|e| CliError::BadArgument(format!("rank task panicked: {e}")))??;

    if let Some(text) = out {
        println!("{text}");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
async fn run_rank(_args: ServeArgs, _layout: WorldLayout) -> Result<(), CliError> {
    Err(CliError::RequiresCuda {
        what: "distributed rank serving",
        reason: "Multi-GPU generation runs NCCL + CudaComputeRuntime on the GPU.",
        suggested_fix: "Build with CUDA and launch with --gpus:\n\
                        cargo build --release\n\
                        ./target/release/skein serve --artifact artifacts/LATEST \\\n\
                            --workload <trace> --cost <constants> \\\n\
                            --prompt \"...\" --gpus 0,1",
    })
}
