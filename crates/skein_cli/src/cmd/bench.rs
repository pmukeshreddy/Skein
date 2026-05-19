//! `skein bench` — Phase B validation harness against vLLM.

use crate::cli::{BenchArgs, OutputFormat};
use crate::error::CliError;

pub fn run(_args: BenchArgs, _output: OutputFormat) -> Result<(), CliError> {
    #[cfg(not(feature = "cuda"))]
    {
        Err(CliError::RequiresCuda {
            what: "skein bench",
            reason: "Benchmarking drives a real Skein artifact + vLLM endpoint, both of which \
                     run on the GPU.",
            suggested_fix: "On an H100 host:\n\
                            cargo build --release --features cuda\n\
                            ./target/release/skein bench --artifact artifacts/LATEST \\\n\
                                --workload <trace.jsonl> --baseline <vllm-endpoint> \\\n\
                                --metrics throughput,goodput,drift_compliance",
        })
    }

    #[cfg(feature = "cuda")]
    {
        Err(CliError::PhaseBOnly {
            what: "skein bench",
            tracking: "Phase B Step 14",
        })
    }
}
