//! `skein_calibrate` — offline calibration tool.
//!
//! Phase A scope: data plumbing.
//!
//! - `CalibrationCorpus` loader + validator.
//! - `KernelMeasurement` / `DriftMeasurement` structs + statistical
//!   aggregation (median for efficiency, p95 for drift).
//! - Deterministic, byte-stable TOML writers that preserve unmeasured
//!   fields. The cost-constants writer renders the same canonical
//!   structure cluster/cost_constants.toml ships with; the drift writer
//!   produces a canonical drift TOML.
//!
//! Phase B wires real samplers. Mac runs them through `NativeComputeRuntime`
//! for pipeline-valid but non-production measurements; CUDA builds select the
//! production runtime on target hardware.

pub mod corpus;
pub mod cost_fit;
pub mod cost_writer;
pub mod drift_fit;
pub mod drift_writer;
pub mod error;
pub mod hardware;
pub mod measurement;
pub mod prompt_sampling;
pub mod sampler;

use std::path::Path;

use serde::{Deserialize, Serialize};

use skein_compile::{ComputeRuntime, SkeinArtifact};
use skein_cost::CostConstants;

pub use corpus::{CalibrationCorpus, DriftPromptSource, KernelSample, SamplingStrategy};
pub use error::CalibrationError;
pub use hardware::{HardwareSpec, ModelSpec};
pub use measurement::{
    DriftMeasurement, KernelMeasurement, aggregate_drift_measurements,
    aggregate_kernel_measurements,
};
pub use prompt_sampling::{default_public_prompt_source, sample_drift_prompts};

/// What a calibration run produced. The raw measurements are retained
/// alongside the fitted summaries so a future Skein version can re-fit
/// with different statistics without re-running the GPU work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub hardware: HardwareSpec,
    pub model: String,
    pub timestamp: String,
    pub kernel_measurements: Vec<KernelMeasurement>,
    pub drift_measurements: Vec<DriftMeasurement>,
    pub fitted_efficiency: Vec<FittedEfficiencyEntry>,
    pub fitted_drift: Vec<FittedDriftEntry>,
}

/// One row of the fitted efficiency table. Kept as a `Vec` of named rows
/// (not a `HashMap`) so the report serializes deterministically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedEfficiencyEntry {
    pub op: skein_cost::OpKind,
    pub dtype: skein_ir::types::Dtype,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedDriftEntry {
    pub layer_idx: usize,
    pub component: skein_ir::types::Component,
    pub dtype: skein_ir::types::Dtype,
    pub value: f64,
}

/// Top-level calibration entry point. Calls the GPU samplers (Phase B),
/// aggregates measurements, and writes both TOML files.
///
/// Phase A: the samplers refuse, so this function refuses with the same
/// error. Tests bypass via the lower-level aggregation + writer functions.
#[allow(clippy::too_many_arguments)] // CLI-style entry point — every arg is a real configuration knob
pub fn calibrate<R: ComputeRuntime + 'static>(
    hardware: HardwareSpec,
    model: ModelSpec,
    corpus: &CalibrationCorpus,
    base_cost: &CostConstants,
    base_drift: Option<&skein_extract::DriftTable>,
    timestamp: &str,
    out_cost: &Path,
    out_drift: &Path,
) -> Result<CalibrationReport, CalibrationError> {
    let kernel_measurements = sampler::sample_kernel_runtimes::<R>(corpus, &hardware)?;
    // Sample drift prompts from the real workload trace first (Phase A
    // capable on Mac — it's just JSONL + selection logic), then hand the
    // sampled prompt list to the Phase-B drift sampler.
    let drift_prompts = sample_drift_prompts(&corpus.drift_source)?;
    let reference_path =
        model
            .reference_artifact
            .as_deref()
            .ok_or_else(|| CalibrationError::UnsupportedSample {
                reason: "Skein-vs-Skein drift calibration requires reference_artifact".into(),
            })?;
    let candidate_path =
        model
            .candidate_artifact
            .as_deref()
            .ok_or_else(|| CalibrationError::UnsupportedSample {
                reason: "Skein-vs-Skein drift calibration requires candidate_artifact".into(),
            })?;
    let reference_artifact = SkeinArtifact::load(reference_path)?;
    let candidate_artifact = SkeinArtifact::load(candidate_path)?;
    let drift_measurements =
        sampler::sample_drift::<R>(&reference_artifact, &candidate_artifact, &drift_prompts)?;

    let fitted_eff = aggregate_kernel_measurements(&kernel_measurements);
    let fitted_drift = aggregate_drift_measurements(&drift_measurements);

    cost_writer::write_cost_constants(out_cost, base_cost, &fitted_eff, &hardware, timestamp)?;
    drift_writer::write_drift_table(out_drift, base_drift, &fitted_drift, timestamp)?;

    Ok(CalibrationReport {
        hardware,
        model: model.name,
        timestamp: timestamp.to_string(),
        kernel_measurements,
        drift_measurements,
        fitted_efficiency: fitted_eff
            .into_iter()
            .map(|((op, dtype), value)| FittedEfficiencyEntry { op, dtype, value })
            .collect(),
        fitted_drift: fitted_drift
            .into_iter()
            .map(|((layer_idx, component, dtype), value)| FittedDriftEntry {
                layer_idx,
                component,
                dtype,
                value,
            })
            .collect(),
    })
}
