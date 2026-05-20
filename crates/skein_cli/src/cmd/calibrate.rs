//! `skein calibrate` — real kernel and Skein-vs-Skein drift calibration.

use std::path::{Path, PathBuf};

use skein_calibrate::corpus::CalibrationCorpus;
use skein_calibrate::hardware::{HardwareSpec, ModelSpec};
use skein_calibrate::{calibrate, default_public_prompt_source};
use skein_cost::CostConstants;
use skein_extract::DriftTable;

use crate::cli::{CalibrateArgs, OutputFormat};
use crate::error::CliError;

#[cfg(not(feature = "cuda"))]
use skein_compile::NativeComputeRuntime;

pub fn run(args: CalibrateArgs, _output: OutputFormat) -> Result<(), CliError> {
    #[cfg(not(feature = "cuda"))]
    tracing::warn!(
        "non-CUDA calibration produces non-production constants; use --features cuda on target hardware"
    );

    #[cfg(feature = "cuda")]
    {
        run_calibrate_inner::<skein_compile::CudaComputeRuntime>(args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        run_calibrate_inner::<NativeComputeRuntime>(args)
    }
}

fn run_calibrate_inner<R: skein_compile::ComputeRuntime + 'static>(
    args: CalibrateArgs,
) -> Result<(), CliError> {
    let mut corpus = CalibrationCorpus::load(&args.corpus)?;
    if let Some(path) = args.drift_prompts_path {
        corpus.drift_source.workload_trace_path = path;
    } else if corpus
        .drift_source
        .workload_trace_path
        .to_string_lossy()
        .contains("sharegpt_2000")
    {
        corpus.drift_source = default_public_prompt_source();
    }

    let base_cost = CostConstants::load(&args.out_cost)?;
    let hardware = HardwareSpec::new(args.hardware).with_cost_constants(&base_cost);
    let out_drift = drift_output_path(&args.out_drift_dir, &args.model);
    let base_drift = if out_drift.exists() {
        Some(DriftTable::load(&out_drift)?)
    } else {
        None
    };
    if let Some(parent) = out_drift.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let model_name = args
        .model
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let mut model = ModelSpec::with_config(model_name, args.model);
    if let (Some(reference), Some(candidate)) = (args.reference_artifact, args.candidate_artifact) {
        model = model.with_artifacts(reference, candidate);
    }

    let report = calibrate::<R>(
        hardware,
        model,
        &corpus,
        &base_cost,
        base_drift.as_ref(),
        "now",
        &args.out_cost,
        &out_drift,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn drift_output_path(out_drift_dir: &Path, model: &Path) -> PathBuf {
    if out_drift_dir.extension().is_some() {
        return out_drift_dir.to_path_buf();
    }
    let stem = model
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    out_drift_dir.join(format!("{stem}_drift.toml"))
}
