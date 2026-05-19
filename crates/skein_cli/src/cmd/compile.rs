//! `skein compile` — full Mac-capable pipeline.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use skein_calibrate::corpus::{DriftPromptSource, SamplingStrategy};
use skein_calibrate::{default_public_prompt_source, sample_drift_prompts};
use skein_compile::{ArtifactMetadata, ComputeRuntime, NativeComputeRuntime, SkeinArtifact};
use skein_cost::Cluster;
use skein_extract::extract_plan;
use skein_ir::plan::{DtypeMap, Plan};
use skein_ir::types::Dtype;
use skein_parity::drift_update::update_drift_table_on_failure;
use skein_parity::{RealSkeinForward, ToleranceTable, tokenize_prompt_bytes, verify_skein_pair};

use crate::cli::{CompileArgs, OutputFormat};
use crate::error::CliError;
use crate::load::{load_cluster, load_cost_model, load_drift_table, load_ir, load_workload};

pub fn run(args: CompileArgs, output: OutputFormat) -> Result<(), CliError> {
    #[cfg(feature = "cuda")]
    {
        run_compile_inner::<skein_compile::CudaComputeRuntime>(args, output)
    }
    #[cfg(not(feature = "cuda"))]
    {
        run_compile_inner::<NativeComputeRuntime>(args, output)
    }
}

pub fn run_compile_inner<R: ComputeRuntime + 'static>(
    args: CompileArgs,
    _output: OutputFormat,
) -> Result<(), CliError> {
    if args.disaggregated {
        return Err(CliError::BadArgument(
            "skein compile --disaggregated is not wired in this pipeline yet".into(),
        ));
    }

    tracing::info!("skein compile: loading inputs");
    let ir = load_ir(&args.model)?;
    let cluster_spec = load_cluster(&args.cluster)?;
    let workload = load_workload(&args.trace)?;
    let drift_table = load_drift_table(&args.drift)?;
    let cost_model = load_cost_model(&args.cost)?;
    let cluster = Cluster::from_spec(cluster_spec.clone());

    tracing::info!("skein compile: running plan search");
    let plan = extract_plan(&ir, &cluster, &workload, &drift_table, &cost_model)?;
    let artifact_dir = args.out.join(plan.content_hash()?.to_hex().to_string());

    let candidate = build_artifact::<R>(
        &plan,
        &cluster_spec,
        &cluster,
        &ir,
        &args.weights,
        &artifact_dir,
        args.search_budget,
        "candidate",
    )?;

    let mut reference_plan = plan.clone();
    reference_plan.dtype_map = DtypeMap::uniform(ir.meta.num_layers, Dtype::Bf16);
    let reference_dir = artifact_dir.join("reference_bf16");
    let reference = build_artifact::<R>(
        &reference_plan,
        &cluster_spec,
        &cluster,
        &ir,
        &args.weights,
        &reference_dir,
        args.search_budget,
        "reference_bf16",
    )?;

    tracing::info!("skein compile: running advisory Skein-vs-Skein parity");
    let mut reference_forward =
        RealSkeinForward::load_with_runtime::<R>(&reference.root, args.search_budget)?;
    let mut candidate_forward =
        RealSkeinForward::load_with_runtime::<R>(&candidate.root, args.search_budget)?;
    let prompt_source =
        compile_prompt_source(args.parity_prompts_path.as_deref(), args.n_parity_prompts);
    let prompts = sample_drift_prompts(&prompt_source)?;
    let tokenized = prompts
        .iter()
        .map(|p| tokenize_prompt_bytes(p, plan.model_meta.vocab as u32))
        .collect::<Vec<_>>();
    let tolerances = ToleranceTable::from_cost_constants(cost_model.constants());
    let parity_report = verify_skein_pair(
        &mut reference_forward,
        &mut candidate_forward,
        &tokenized,
        &plan,
        &workload,
        &tolerances,
    )?;
    write_json(&artifact_dir.join("parity_report.json"), &parity_report)?;

    if !parity_report.passed {
        if let Some(failing) = &parity_report.failing_layer {
            let _ = update_drift_table_on_failure(&args.drift, failing)?;
        }
        if args.enforce_parity {
            return Err(CliError::ParityFailed {
                artifact: artifact_dir.display().to_string(),
            });
        }
    }

    update_latest_symlink(&args.out, &artifact_dir)?;
    tracing::info!("skein compile complete: {}", artifact_dir.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_artifact<R: ComputeRuntime>(
    plan: &Plan,
    cluster_spec: &skein_ir::cluster::ClusterSpec,
    cluster: &Cluster,
    ir: &skein_ir::ir::Graph,
    weights_dir: &Path,
    artifact_dir: &Path,
    search_budget: usize,
    label: &str,
) -> Result<SkeinArtifact, CliError> {
    if artifact_dir.exists() {
        return SkeinArtifact::load(artifact_dir).map_err(CliError::from);
    }
    std::fs::create_dir_all(artifact_dir)?;
    let staging = artifact_dir.join(format!(
        ".staging-{label}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&staging)?;

    let mut lowered = Vec::new();
    let mut shard_paths = Vec::new();
    for device_idx in 0..cluster.num_devices() {
        let mut artifact =
            skein_emit::lower_per_device(plan, cluster, ir, device_idx, weights_dir)?;
        for segment in &mut artifact.graph.segments {
            let _ = skein_compile::compile_with_luminal::<R>(
                std::slice::from_mut(&mut segment.graph),
                search_budget,
            )?;
        }
        let shard_path = staging.join(format!("device_{device_idx}.weights.safetensors"));
        skein_emit::weights::write_weight_shard(&artifact.weight_shard, weights_dir, &shard_path)?;
        lowered.push(artifact);
        shard_paths.push(shard_path);
    }

    let metadata = ArtifactMetadata::new("working-tree", cluster_spec.device_kind_summary(), "now");
    let artifact = SkeinArtifact::write(
        plan,
        cluster_spec,
        ir,
        &lowered,
        &shard_paths,
        &metadata,
        artifact_dir,
    )?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(artifact)
}

fn compile_prompt_source(path: Option<&Path>, n: usize) -> DriftPromptSource {
    match path {
        Some(path) => DriftPromptSource {
            workload_trace_path: path.to_path_buf(),
            n_drift_prompts: n.max(1),
            sampling_strategy: SamplingStrategy::FirstN,
            sampling_seed: 0,
        },
        None => {
            let mut source = default_public_prompt_source();
            source.n_drift_prompts = n.max(1);
            source
        }
    }
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), CliError> {
    let bytes = serde_json::to_vec_pretty(value)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn update_latest_symlink(out_dir: &Path, artifact_dir: &Path) -> Result<(), CliError> {
    std::fs::create_dir_all(out_dir)?;
    let latest = out_dir.join("LATEST");
    let tmp = out_dir.join(format!(
        "LATEST.new.{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    if tmp.exists() {
        std::fs::remove_file(&tmp)?;
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(artifact_dir, &tmp)?;
    #[cfg(not(unix))]
    std::fs::write(&tmp, artifact_dir.display().to_string())?;
    std::fs::rename(tmp, latest)?;
    Ok(())
}

trait ClusterSummary {
    fn device_kind_summary(&self) -> String;
}

impl ClusterSummary for skein_ir::cluster::ClusterSpec {
    fn device_kind_summary(&self) -> String {
        let mut kinds = self
            .nodes
            .iter()
            .map(|node| node.device_kind.clone())
            .collect::<Vec<_>>();
        kinds.sort();
        kinds.dedup();
        kinds.join(",")
    }
}
