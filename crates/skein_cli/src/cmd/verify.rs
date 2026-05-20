//! `skein verify` — Skein-vs-Skein parity on an existing artifact.

use std::path::Path;

use skein_calibrate::corpus::{DriftPromptSource, SamplingStrategy};
use skein_calibrate::{default_public_prompt_source, sample_drift_prompts};
use skein_compile::{ComputeRuntime, SkeinArtifact};
use skein_ir::workload::{Slo, Workload};
use skein_parity::{RealSkeinForward, ToleranceTable, tokenize_prompt_bytes, verify_skein_pair};

use crate::cli::{OutputFormat, VerifyArgs};
use crate::error::CliError;
use crate::load::load_cost_model;

#[cfg(not(feature = "cuda"))]
use skein_compile::NativeComputeRuntime;

pub fn run(args: VerifyArgs, output: OutputFormat) -> Result<(), CliError> {
    #[cfg(feature = "cuda")]
    {
        run_verify_inner::<skein_compile::CudaComputeRuntime>(args, output)
    }
    #[cfg(not(feature = "cuda"))]
    {
        run_verify_inner::<NativeComputeRuntime>(args, output)
    }
}

pub fn run_verify_inner<R: ComputeRuntime + 'static>(
    args: VerifyArgs,
    _output: OutputFormat,
) -> Result<(), CliError> {
    let artifact = SkeinArtifact::load(&args.artifact)?;
    let reference_path = match args.reference {
        Some(path) => path,
        None => {
            let p = args.artifact.join("reference_bf16");
            if !p.exists() {
                return Err(CliError::BadArgument(format!(
                    "no --reference supplied and {} does not exist",
                    p.display()
                )));
            }
            p
        }
    };
    let reference = SkeinArtifact::load(&reference_path)?;
    let prompt_source = verify_prompt_source(args.sample_from.as_deref(), args.n_prompts);
    let prompts = sample_drift_prompts(&prompt_source)?;
    let tokenized = prompts
        .iter()
        .take(args.n_prompts)
        .map(|p| tokenize_prompt_bytes(p, artifact.plan.model_meta.vocab as u32))
        .collect::<Vec<_>>();

    let cost_model = load_cost_model(&args.cost)?;
    let tolerances = ToleranceTable::from_cost_constants(cost_model.constants());
    let workload = Workload {
        slo: Slo {
            ttft_p95_ms: 500,
            tpot_p95_ms: 50,
            max_accuracy_drift: 0.01,
            recompile_drift_threshold_kl: 0.05,
        },
        requests: vec![],
    };

    let mut reference_forward = RealSkeinForward::load_with_runtime::<R>(
        &reference.root,
        skein_compile::DEFAULT_SEARCH_BUDGET,
    )?;
    let mut candidate_forward = RealSkeinForward::load_with_runtime::<R>(
        &artifact.root,
        skein_compile::DEFAULT_SEARCH_BUDGET,
    )?;
    let report = verify_skein_pair(
        &mut reference_forward,
        &mut candidate_forward,
        &tokenized,
        &artifact.plan,
        &workload,
        &tolerances,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if args.enforce && !report.passed {
        return Err(CliError::ParityFailed {
            artifact: args.artifact.display().to_string(),
        });
    }
    Ok(())
}

fn verify_prompt_source(path: Option<&Path>, n: usize) -> DriftPromptSource {
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
