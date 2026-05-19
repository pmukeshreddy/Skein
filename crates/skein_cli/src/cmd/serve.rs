//! `skein serve` — Phase B runtime server.

use crate::cli::{OutputFormat, ServeArgs};
use crate::error::CliError;

pub async fn run(_args: ServeArgs, _output: OutputFormat) -> Result<(), CliError> {
    #[cfg(not(feature = "cuda"))]
    {
        Err(CliError::RequiresCuda {
            what: "skein serve",
            reason: "The forward-pass driver issues NCCL collectives between Luminal-compiled \
                     graphs and dispatches CUDA Graphs at warmup; both require a CUDA toolchain.",
            suggested_fix: "On an H100 host:\n\
                            cargo build --release --features cuda\n\
                            ./target/release/skein serve --artifact artifacts/LATEST --port 8080",
        })
    }

    #[cfg(feature = "cuda")]
    {
        Err(CliError::PhaseBOnly {
            what: "skein serve",
            tracking: "Phase B Step 11",
        })
    }
}
