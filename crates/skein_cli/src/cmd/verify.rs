//! `skein verify` — Skein-vs-Skein parity on an existing artifact.

use std::path::Path;

use skein_calibrate::corpus::{DriftPromptSource, SamplingStrategy};
use skein_calibrate::{default_public_prompt_source, sample_drift_prompts};
use skein_compile::{ComputeRuntime, SkeinArtifact};
use skein_ir::workload::{Slo, Workload};
use skein_parity::{
    PythonSubprocessReference, RealSkeinForward, ToleranceTable, tokenize_prompt_bytes, verify_plan,
    verify_skein_pair,
};

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
    // ── Debug-only dump path: run one Skein forward over explicit tokens and
    // dump layer-0 op taps (with SKEIN_DEBUG_TAPS + SKEIN_DUMP_DIR). No HF, no
    // reference artifact, so it never co-resides another 90 GB model on GPU. ──
    if args.dump_only {
        use skein_parity::SkeinForward;
        let tokens_path = args.tokens_file.as_ref().ok_or_else(|| {
            CliError::BadArgument("--dump-only requires --tokens-file".to_string())
        })?;
        let raw = std::fs::read_to_string(tokens_path)?;
        let tokens: Vec<u32> = raw
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>())
            .collect::<Result<_, _>>()
            .map_err(|e| CliError::BadArgument(format!("bad token id in {tokens_path:?}: {e}")))?;
        eprintln!("dump-only: {} tokens = {:?}", tokens.len(), tokens);
        // SKEIN_DUMP_NATIVE forces the CPU NativeComputeRuntime (no NVRTC
        // codegen) so the SAME tp=2 lowering can be dumped on CPU and diffed
        // against the CUDA dump: if they match, a divergence from HF is a
        // wiring/lowering bug; if they differ, it is CUDA codegen of the
        // composed graph.
        let timing = std::env::var_os("SKEIN_TIMING").is_some();
        let out = if std::env::var_os("SKEIN_DUMP_NATIVE").is_some() {
            eprintln!("dump-only: using NativeComputeRuntime (CPU)");
            let mut candidate = RealSkeinForward::load_native(&args.artifact)?;
            candidate.forward_with_hooks(&tokens)?
        } else {
            let t_load = std::time::Instant::now();
            let mut candidate = RealSkeinForward::load_with_runtime::<R>(
                &args.artifact,
                skein_compile::DEFAULT_SEARCH_BUDGET,
            )?;
            let load_s = t_load.elapsed().as_secs_f64();
            let t_fwd = std::time::Instant::now();
            let out = candidate.forward_with_hooks(&tokens)?;
            let fwd_s = t_fwd.elapsed().as_secs_f64();
            if timing {
                eprintln!(
                    "SKEIN_TIMING TOTAL: load(graph+weights) {load_s:.1}s | forward {fwd_s:.1}s",
                );
            }
            out
        };
        let argmax = out
            .final_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        eprintln!(
            "dump-only: final_logits len={} argmax={}",
            out.final_logits.len(),
            argmax
        );
        if let Some(dir) = std::env::var_os("SKEIN_DUMP_DIR") {
            let path = std::path::Path::new(&dir).join("final_logits.f32");
            let bytes: Vec<u8> = out
                .final_logits
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let _ = std::fs::write(&path, bytes);
            eprintln!("dump-only: wrote {}", path.display());
        }
        return Ok(());
    }

    let artifact = SkeinArtifact::load(&args.artifact)?;
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
    let prompt_source = verify_prompt_source(args.sample_from.as_deref(), args.n_prompts);
    let prompts = sample_drift_prompts(&prompt_source)?;

    // ── HF reference path: the README accuracy gate vs `transformers`. Drives
    // verify_reference.py over the *real* tokenizer + bf16 model, compares the
    // candidate Skein artifact's per-layer activations + final-logit KL. ──
    if let Some(hf_model) = args.hf_reference.as_ref() {
        let reference =
            PythonSubprocessReference::new(hf_model.clone(), args.reference_dtype.clone())?;
        let ir = artifact.ir()?;
        let mut candidate = RealSkeinForward::load_with_runtime::<R>(
            &artifact.root,
            skein_compile::DEFAULT_SEARCH_BUDGET,
        )?;
        let sample: Vec<String> = prompts.iter().take(args.n_prompts).cloned().collect();
        let report = verify_plan(
            &reference,
            &mut candidate,
            &ir,
            &artifact.plan,
            &workload,
            &tolerances,
            &sample,
        )?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        if args.enforce && !report.passed {
            return Err(CliError::ParityFailed {
                artifact: args.artifact.display().to_string(),
            });
        }
        return Ok(());
    }

    // ── Skein-vs-Skein path (default): compare against the bf16 reference. ──
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
    let tokenized = prompts
        .iter()
        .take(args.n_prompts)
        .map(|p| tokenize_prompt_bytes(p, artifact.plan.model_meta.vocab as u32))
        .collect::<Vec<_>>();

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
