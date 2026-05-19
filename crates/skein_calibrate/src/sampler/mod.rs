//! Real calibration samplers.

pub mod drift_sample;
pub mod kernel_sample;

use crate::corpus::CalibrationCorpus;
use crate::error::CalibrationError;
use crate::hardware::HardwareSpec;
use crate::measurement::{DriftMeasurement, KernelMeasurement};
use skein_compile::ComputeRuntime;
use skein_compile::SkeinArtifact;

pub fn sample_kernel_runtimes<R: ComputeRuntime>(
    corpus: &CalibrationCorpus,
    hardware: &HardwareSpec,
) -> Result<Vec<KernelMeasurement>, CalibrationError> {
    kernel_sample::sample::<R>(corpus, hardware)
}

/// Measure drift for each prompt by comparing a bf16 Skein artifact against
/// a candidate Skein artifact.
pub fn sample_drift<R: ComputeRuntime + 'static>(
    reference_artifact: &SkeinArtifact,
    candidate_artifact: &SkeinArtifact,
    prompts: &[String],
) -> Result<Vec<DriftMeasurement>, CalibrationError> {
    drift_sample::sample::<R>(reference_artifact, candidate_artifact, prompts)
}
