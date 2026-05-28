//! CUDA-gated subcommands. On a CPU (`--no-default-features`) build, `bench`
//! returns `CliError::RequiresCuda` with a message that names the subcommand
//! and suggests the rebuild fix. On a CUDA build it returns `Ok(())`.

use std::path::PathBuf;

use skein_cli::CliError;
use skein_cli::cli::{BenchArgs, BenchMetric, OutputFormat};
use skein_cli::cmd;

fn bench_args() -> BenchArgs {
    BenchArgs {
        artifact: PathBuf::from("/tmp/a"),
        workload: PathBuf::from("/tmp/w"),
        baseline: "http://localhost:8000".to_string(),
        metrics: vec![BenchMetric::Throughput],
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn bench_returns_requires_cuda_on_cpu_build() {
    let err = cmd::bench::run(bench_args(), OutputFormat::Text).unwrap_err();
    match err {
        CliError::RequiresCuda {
            what,
            reason,
            suggested_fix,
        } => {
            assert_eq!(what, "skein bench");
            let r = reason.to_ascii_lowercase();
            assert!(
                r.contains("cuda") || r.contains("gpu"),
                "reason should mention CUDA/GPU: {reason}"
            );
            assert!(
                suggested_fix.contains("cargo build"),
                "suggested fix should give a rebuild command: {suggested_fix}"
            );
            assert!(
                suggested_fix.contains("bench"),
                "suggested fix should reference the subcommand: {suggested_fix}"
            );
        }
        other => panic!("expected RequiresCuda, got {other:?}"),
    }
}

#[cfg(feature = "cuda")]
#[test]
fn bench_returns_ok_on_cuda_build() {
    cmd::bench::run(bench_args(), OutputFormat::Text).unwrap();
}
