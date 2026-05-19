//! Skein-vs-Skein drift sampler.

use skein_compile::{ComputeRuntime, DEFAULT_SEARCH_BUDGET, SkeinArtifact};
use skein_ir::types::Component;
use skein_parity::comparison::mse;
use skein_parity::{RealSkeinForward, SkeinForward, tokenize_prompt_bytes};

use crate::error::CalibrationError;
use crate::measurement::DriftMeasurement;

pub fn sample<R: ComputeRuntime + 'static>(
    reference_artifact: &SkeinArtifact,
    candidate_artifact: &SkeinArtifact,
    prompts: &[String],
) -> Result<Vec<DriftMeasurement>, CalibrationError> {
    let mut reference =
        RealSkeinForward::load_with_runtime::<R>(&reference_artifact.root, DEFAULT_SEARCH_BUDGET)?;
    let mut candidate =
        RealSkeinForward::load_with_runtime::<R>(&candidate_artifact.root, DEFAULT_SEARCH_BUDGET)?;
    let vocab = candidate_artifact.plan.model_meta.vocab as u32;
    let mut measurements = Vec::new();

    for (prompt_idx, prompt) in prompts.iter().enumerate() {
        let tokens = tokenize_prompt_bytes(prompt, vocab);
        let ref_out = reference.forward_with_hooks(&tokens)?;
        let cand_out = candidate.forward_with_hooks(&tokens)?;
        for (layer_idx, (ref_acts, cand_acts)) in ref_out
            .per_layer_activations
            .iter()
            .zip(cand_out.per_layer_activations.iter())
            .enumerate()
        {
            let measured = mse(ref_acts, cand_acts)?;
            let dtype_entry = candidate_artifact
                .plan
                .dtype_map
                .per_layer
                .get(layer_idx)
                .copied()
                .ok_or_else(|| CalibrationError::UnsupportedSample {
                    reason: format!("candidate plan has no dtype entry for layer {layer_idx}"),
                })?;
            for (component, dtype) in [
                (Component::Weight, dtype_entry.weight),
                (Component::Activation, dtype_entry.activation),
                (Component::KvCache, dtype_entry.kv_cache),
            ] {
                if dtype == skein_ir::types::Dtype::Bf16 {
                    continue;
                }
                measurements.push(DriftMeasurement {
                    layer_idx,
                    component,
                    dtype,
                    prompt_idx,
                    measured_kl: measured,
                });
            }
        }
    }
    Ok(measurements)
}
