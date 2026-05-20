//! `skein serve` — Phase B runtime server (Mac native + H100 CUDA).

use skein_compile::SkeinArtifact;
use skein_runtime::Server;
use skein_runtime::server::lifecycle::ServerBuildInputs;

use crate::cli::{OutputFormat, ServeArgs};
use crate::error::CliError;
use crate::load::{load_cost_model, load_workload};

pub async fn run(args: ServeArgs, _output: OutputFormat) -> Result<(), CliError> {
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
