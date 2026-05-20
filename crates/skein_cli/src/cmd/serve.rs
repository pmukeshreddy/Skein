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
    // Mode 3: single-process HTTP serve.
    run_single_process(args, output).await
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

    tracing::info!(
        rank = layout.rank,
        world_size = layout.world_size,
        "rank serving: bootstrapping NCCL + loading device segments"
    );
    // NCCL + the compute runtime are blocking/sync — run off the async runtime.
    let out = tokio::task::spawn_blocking(move || {
        gpu_rank::run_generation(
            &artifact_dir,
            layout,
            &rendezvous,
            &prompt,
            max_new,
            tokenizer.as_ref(),
        )
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
