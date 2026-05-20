//! Remaining CUDA-gated subcommands return `CliError::RequiresCuda` with a
//! message that names the subcommand and suggests the rebuild fix.

use std::path::PathBuf;

use skein_cli::CliError;
use skein_cli::cli::{BenchArgs, BenchMetric, OutputFormat};
use skein_cli::cmd;

fn assert_requires_cuda_for(err: CliError, expected_what: &str) {
    match err {
        CliError::RequiresCuda {
            what,
            reason,
            suggested_fix,
        } => {
            assert_eq!(what, expected_what);
            // Reason mentions CUDA / GPU somewhere.
            let r = reason.to_ascii_lowercase();
            assert!(
                r.contains("cuda") || r.contains("gpu"),
                "reason should mention CUDA/GPU: {reason}"
            );
            // The suggested fix names `--features cuda` and the subcommand.
            assert!(
                suggested_fix.contains("--features cuda"),
                "suggested fix should mention --features cuda: {suggested_fix}"
            );
            assert!(
                suggested_fix.contains(expected_what.trim_start_matches("skein ")),
                "suggested fix should reference the subcommand: {suggested_fix}"
            );
        }
        other => panic!("expected RequiresCuda, got {other:?}"),
    }
}

#[test]
fn bench_returns_requires_cuda_on_phase_a() {
    let err = cmd::bench::run(
        BenchArgs {
            artifact: PathBuf::from("/tmp/a"),
            workload: PathBuf::from("/tmp/w"),
            baseline: "http://localhost:8000".to_string(),
            metrics: vec![BenchMetric::Throughput],
        },
        OutputFormat::Text,
    )
    .unwrap_err();
    assert_requires_cuda_for(err, "skein bench");
}
