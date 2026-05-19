//! `verify_plan` — the top-level orchestrator.

use std::collections::HashMap;

use skein_ir::ir::Graph;
use skein_ir::plan::Plan;
use skein_ir::types::{Component, Dtype};
use skein_ir::workload::Workload;

use crate::comparison::{kl_divergence, mse};
use crate::error::ParityError;
use crate::reference::HFReference;
use crate::report::{FailingLayerReport, ParityReport, PerPromptReport};
use crate::skein_forward::SkeinForward;
use crate::tolerance::{ToleranceTable, dominant_dtype_at_layer, tolerance_for_layer};

/// Tokenize text for the current Skein-native runtime path. Prompt 2's
/// server uses the same byte-mod-vocab tokenizer because artifacts do not
/// yet carry tokenizer assets.
pub fn tokenize_prompt_bytes(prompt: &str, vocab: u32) -> Vec<u32> {
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

/// Compare the Skein artifact against the HF reference for every prompt in
/// `sample_prompts`. Returns a `ParityReport` whose `passed` field is the
/// conjunction of:
///
/// - Every per-prompt, per-layer MSE is within the layer's dtype-derived
///   tolerance.
/// - Average final-logit KL divergence across all prompts is at most
///   `workload.slo.max_accuracy_drift`.
///
/// The function takes the reference + Skein implementations as trait
/// objects so Phase A's mock + Phase B's real subprocess share the same
/// orchestration code.
#[allow(clippy::too_many_arguments)]
pub fn verify_plan(
    reference: &dyn HFReference,
    skein: &mut dyn SkeinForward,
    ir: &Graph,
    plan: &Plan,
    workload: &Workload,
    tolerances: &ToleranceTable,
    sample_prompts: &[String],
) -> Result<ParityReport, ParityError> {
    if plan.dtype_map.per_layer.len() != ir.meta.num_layers {
        return Err(ParityError::DtypeMapIncomplete {
            expected: ir.meta.num_layers,
            actual: plan.dtype_map.per_layer.len(),
        });
    }
    if sample_prompts.is_empty() {
        return Err(ParityError::ShapeMismatch {
            expected: 1,
            got: 0,
        });
    }

    let num_blocks = ir.meta.num_layers;
    let mut per_prompt: Vec<PerPromptReport> = Vec::with_capacity(sample_prompts.len());
    // Track which prompts violated each `(layer, component, dtype)` triple.
    let mut violations: HashMap<(usize, Component, Dtype), (Vec<usize>, f64)> = HashMap::new();
    let mut all_layers_pass = true;

    let tokenized = sample_prompts
        .iter()
        .map(|prompt| reference.tokenize(prompt))
        .collect::<Result<Vec<_>, _>>()?;
    let reference_outputs = reference.forward_batch_with_hooks(&tokenized)?;

    for (prompt_idx, tokens) in tokenized.iter().enumerate() {
        let ref_out =
            reference_outputs
                .get(prompt_idx)
                .cloned()
                .ok_or(ParityError::ShapeMismatch {
                    expected: tokenized.len(),
                    got: reference_outputs.len(),
                })?;
        let skein_out = skein.forward_with_hooks(tokens)?;
        if ref_out.per_layer_activations.len() != num_blocks {
            return Err(ParityError::ShapeMismatch {
                expected: num_blocks,
                got: ref_out.per_layer_activations.len(),
            });
        }
        if skein_out.per_layer_activations.len() != num_blocks {
            return Err(ParityError::ShapeMismatch {
                expected: num_blocks,
                got: skein_out.per_layer_activations.len(),
            });
        }

        let per_layer_mse: Vec<f64> = ref_out
            .per_layer_activations
            .iter()
            .zip(skein_out.per_layer_activations.iter())
            .map(|(r, s)| mse(r, s))
            .collect::<Result<_, _>>()?;
        let final_kl = kl_divergence(&ref_out.final_logits, &skein_out.final_logits)?;

        for (layer_idx, &measured) in per_layer_mse.iter().enumerate() {
            let tol = tolerance_for_layer(layer_idx, plan, tolerances);
            if measured > tol {
                all_layers_pass = false;
                let (component, dtype) = dominant_dtype_at_layer(plan, layer_idx);
                let entry = violations
                    .entry((layer_idx, component, dtype))
                    .or_insert_with(|| (Vec::new(), measured));
                entry.0.push(prompt_idx);
                if measured > entry.1 {
                    entry.1 = measured;
                }
            }
        }

        per_prompt.push(PerPromptReport {
            prompt_idx,
            per_layer_mse,
            final_kl,
        });
    }

    let n = per_prompt.len() as f64;
    let avg_final_kl = per_prompt.iter().map(|p| p.final_kl).sum::<f64>() / n;
    let max_final_kl = per_prompt
        .iter()
        .map(|p| p.final_kl)
        .fold(0.0_f64, f64::max);

    // Pick the worst violation (most prompts affected). Ties broken by
    // higher measured MSE.
    let failing_layer = violations.into_iter().max_by(|a, b| {
        a.1.0.len().cmp(&b.1.0.len()).then(
            a.1.1
                .partial_cmp(&b.1.1)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    let failing_layer =
        failing_layer.map(|((layer_idx, component, dtype), (prompts, measured))| {
            FailingLayerReport {
                layer_idx,
                component,
                dtype,
                measured_mse: measured,
                tolerance: tolerance_for_layer(layer_idx, plan, tolerances),
                prompts_violating: prompts,
            }
        });

    let kl_passed = avg_final_kl <= workload.slo.max_accuracy_drift;
    let passed = kl_passed && all_layers_pass;

    let plan_hash_hex = plan.content_hash()?.to_hex().to_string();

    Ok(ParityReport {
        passed,
        plan_hash: plan_hash_hex,
        model: ir.meta.architecture.clone(),
        num_prompts: sample_prompts.len(),
        per_prompt,
        avg_final_kl,
        max_final_kl,
        slo_max_drift: workload.slo.max_accuracy_drift,
        failing_layer,
    })
}

pub fn verify_skein_pair(
    reference: &mut dyn SkeinForward,
    skein: &mut dyn SkeinForward,
    tokenized_prompts: &[Vec<u32>],
    plan: &Plan,
    workload: &Workload,
    tolerances: &ToleranceTable,
) -> Result<ParityReport, ParityError> {
    if plan.dtype_map.per_layer.len() != plan.model_meta.num_layers {
        return Err(ParityError::DtypeMapIncomplete {
            expected: plan.model_meta.num_layers,
            actual: plan.dtype_map.per_layer.len(),
        });
    }
    if tokenized_prompts.is_empty() {
        return Err(ParityError::ShapeMismatch {
            expected: 1,
            got: 0,
        });
    }

    let num_blocks = plan.model_meta.num_layers;
    let mut per_prompt: Vec<PerPromptReport> = Vec::with_capacity(tokenized_prompts.len());
    let mut violations: HashMap<(usize, Component, Dtype), (Vec<usize>, f64)> = HashMap::new();
    let mut all_layers_pass = true;

    for (prompt_idx, tokens) in tokenized_prompts.iter().enumerate() {
        let ref_out = reference.forward_with_hooks(tokens)?;
        let skein_out = skein.forward_with_hooks(tokens)?;
        if ref_out.per_layer_activations.len() != num_blocks {
            return Err(ParityError::ShapeMismatch {
                expected: num_blocks,
                got: ref_out.per_layer_activations.len(),
            });
        }
        if skein_out.per_layer_activations.len() != num_blocks {
            return Err(ParityError::ShapeMismatch {
                expected: num_blocks,
                got: skein_out.per_layer_activations.len(),
            });
        }

        let per_layer_mse: Vec<f64> = ref_out
            .per_layer_activations
            .iter()
            .zip(skein_out.per_layer_activations.iter())
            .map(|(r, s)| mse(r, s))
            .collect::<Result<_, _>>()?;
        let final_kl = kl_divergence(&ref_out.final_logits, &skein_out.final_logits)?;

        for (layer_idx, &measured) in per_layer_mse.iter().enumerate() {
            let tol = tolerance_for_layer(layer_idx, plan, tolerances);
            if measured > tol {
                all_layers_pass = false;
                let (component, dtype) = dominant_dtype_at_layer(plan, layer_idx);
                let entry = violations
                    .entry((layer_idx, component, dtype))
                    .or_insert_with(|| (Vec::new(), measured));
                entry.0.push(prompt_idx);
                if measured > entry.1 {
                    entry.1 = measured;
                }
            }
        }

        per_prompt.push(PerPromptReport {
            prompt_idx,
            per_layer_mse,
            final_kl,
        });
    }

    let n = per_prompt.len() as f64;
    let avg_final_kl = per_prompt.iter().map(|p| p.final_kl).sum::<f64>() / n;
    let max_final_kl = per_prompt
        .iter()
        .map(|p| p.final_kl)
        .fold(0.0_f64, f64::max);
    let failing_layer = violations.into_iter().max_by(|a, b| {
        a.1.0.len().cmp(&b.1.0.len()).then(
            a.1.1
                .partial_cmp(&b.1.1)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    let failing_layer =
        failing_layer.map(|((layer_idx, component, dtype), (prompts, measured))| {
            FailingLayerReport {
                layer_idx,
                component,
                dtype,
                measured_mse: measured,
                tolerance: tolerance_for_layer(layer_idx, plan, tolerances),
                prompts_violating: prompts,
            }
        });

    let kl_passed = avg_final_kl <= workload.slo.max_accuracy_drift;
    let passed = kl_passed && all_layers_pass;
    let plan_hash_hex = plan.content_hash()?.to_hex().to_string();

    Ok(ParityReport {
        passed,
        plan_hash: plan_hash_hex,
        model: plan.model_meta.architecture.clone(),
        num_prompts: tokenized_prompts.len(),
        per_prompt,
        avg_final_kl,
        max_final_kl,
        slo_max_drift: workload.slo.max_accuracy_drift,
        failing_layer,
    })
}
