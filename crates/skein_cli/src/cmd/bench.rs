//! `skein bench` — validation harness against a vLLM baseline.

use crate::cli::{BenchArgs, OutputFormat};
use crate::error::CliError;

pub fn run(_args: BenchArgs, _output: OutputFormat) -> Result<(), CliError> {
    #[cfg(not(feature = "cuda"))]
    {
        Err(CliError::RequiresCuda {
            what: "skein bench",
            reason: "Benchmarking drives a real Skein artifact + vLLM endpoint, both of which \
                     run on the GPU.",
            suggested_fix: "On an NVIDIA GPU host (the default build enables CUDA):\n\
                            cargo build --release\n\
                            ./target/release/skein bench --artifact artifacts/LATEST \\\n\
                                --workload <trace.jsonl> --baseline <vllm-endpoint> \\\n\
                                --metrics throughput,goodput,drift_compliance",
        })
    }

    // TODO(bench): drive the artifact + vLLM baseline and emit the
    // throughput / goodput / drift-compliance comparison.
    #[cfg(feature = "cuda")]
    {
        Err(CliError::NotImplemented {
            what: "skein bench",
            detail: "the vLLM comparison harness is not yet wired",
        })
    }
}
