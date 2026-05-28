//! Clap structures for `skein`. Each subcommand maps to a separate `Args`
//! struct so command runners can be unit-tested independently.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "skein",
    version,
    about = "Distributed LLM inference plan compiler",
    arg_required_else_help = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Logging filter passed to `tracing-subscriber`. Accepts the standard
    /// `RUST_LOG` syntax (`info`, `debug`, `skein_extract=debug,info`, …).
    #[arg(long, global = true, default_value = "info")]
    pub log_level: String,

    /// Output format for the final status report. `text` is human-friendly;
    /// `json` is a single object on stdout (suitable for piping to `jq`).
    #[arg(long, global = true, default_value = "text", value_enum)]
    pub output: OutputFormat,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Plan search only — outputs `plan.json`. No GPU required.
    Extract(ExtractArgs),

    /// Full compile: search + lower + Luminal compile + parity + artifact write.
    Compile(CompileArgs),

    /// Parity check on an existing artifact against a bf16 Skein reference.
    Verify(VerifyArgs),

    /// Launch the runtime server.
    Serve(ServeArgs),

    /// Calibrate cost constants + drift table from runtime measurements.
    Calibrate(CalibrateArgs),

}

#[derive(Args, Debug, Clone)]
pub struct ExtractArgs {
    /// Path to the HuggingFace `config.json` for the target model.
    #[arg(long)]
    pub model: PathBuf,

    /// Path to the cluster topology TOML (see `cluster/<name>.toml`).
    #[arg(long)]
    pub cluster: PathBuf,

    /// Path to the workload trace JSONL (see `cluster/sample_trace.jsonl`).
    #[arg(long)]
    pub trace: PathBuf,

    /// Path to the model's drift TOML (`models/<model>_drift.toml`).
    #[arg(long)]
    pub drift: PathBuf,

    /// Path to `cluster/cost_constants.toml`.
    #[arg(long)]
    pub cost: PathBuf,

    /// Where to write the chosen `Plan` as JSON.
    #[arg(long, default_value = "plan.json")]
    pub out: PathBuf,
}

#[derive(Args, Debug, Clone)]
pub struct CompileArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub cluster: PathBuf,
    #[arg(long)]
    pub trace: PathBuf,
    #[arg(long)]
    pub drift: PathBuf,
    #[arg(long)]
    pub cost: PathBuf,
    /// Source safetensors directory.
    #[arg(long)]
    pub weights: PathBuf,
    #[arg(long, default_value = "artifacts/")]
    pub out: PathBuf,
    /// Luminal search budget per segment.
    #[arg(long, default_value_t = default_search_budget())]
    pub search_budget: usize,
    /// Treat advisory parity failure as a hard compile failure.
    #[arg(long)]
    pub enforce_parity: bool,
    /// Prompt JSONL used by advisory parity. Defaults to Skein's public dev corpus.
    #[arg(long)]
    pub parity_prompts_path: Option<PathBuf>,
    #[arg(long, default_value_t = 8)]
    pub n_parity_prompts: usize,
}

#[derive(Args, Debug, Clone)]
pub struct VerifyArgs {
    #[arg(long)]
    pub artifact: PathBuf,
    /// Path to the bf16 reference Skein artifact. Defaults to
    /// `<artifact>/reference_bf16` when compile wrote it.
    #[arg(long)]
    pub reference: Option<PathBuf>,
    /// Prompt JSONL to draw sample prompts from.
    #[arg(long)]
    pub sample_from: Option<PathBuf>,
    #[arg(long, default_value_t = 50)]
    pub n_prompts: usize,
    #[arg(long, default_value = "cluster/cost_constants.toml")]
    pub cost: PathBuf,
    #[arg(long)]
    pub enforce: bool,
    /// Verify against the real HuggingFace `transformers` reference at this
    /// model path (runs `verify_reference.py`), instead of the Skein-bf16
    /// reference. This is the true accuracy gate from the README.
    #[arg(long)]
    pub hf_reference: Option<PathBuf>,
    /// Reference dtype for the HF model: bfloat16 | float16 | float32.
    #[arg(long, default_value = "bfloat16")]
    pub reference_dtype: String,
    /// Debug-only: load just the candidate artifact, run one forward over the
    /// tokens in `--tokens-file`, and exit. Dumps final logits to
    /// `SKEIN_DUMP_DIR` when set. No HF subprocess, no reference artifact.
    #[arg(long)]
    pub dump_only: bool,
    /// Whitespace/comma-separated token ids fed to the `--dump-only` forward.
    #[arg(long)]
    pub tokens_file: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
    /// Path to the Skein artifact directory, or the LATEST symlink.
    #[arg(long)]
    pub artifact: PathBuf,

    /// Workload trace JSONL for batcher tuning.
    #[arg(long)]
    pub workload: PathBuf,

    /// Cost constants TOML, usually the same file passed to `skein compile`.
    #[arg(long)]
    pub cost: PathBuf,

    /// HTTP port to bind.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    #[arg(long)]
    pub enable_hot_swap: bool,

    /// Total bytes available for KV cache across all devices.
    #[arg(long, default_value_t = 268_435_456)]
    pub total_kv_bytes: u64,

    /// Bytes consumed by each cached token.
    #[arg(long, default_value_t = 524_288)]
    pub bytes_per_token: u64,

    // ── Multi-GPU distributed generation ──────────────────────────────────
    /// Launch one process per listed GPU (e.g. `--gpus 0,1,2,3`) and run
    /// distributed generation across them. Without this, runs single-process.
    #[arg(long)]
    pub gpus: Option<String>,

    /// Prompt for distributed generation (multi-GPU path).
    #[arg(long)]
    pub prompt: Option<String>,

    /// Tokens to generate in the distributed path.
    #[arg(long, default_value_t = 32)]
    pub max_new_tokens: usize,

    /// Shared file path for the NCCL rendezvous id (multi-GPU path).
    #[arg(long, default_value = "/tmp/skein_rendezvous")]
    pub rendezvous: PathBuf,

    /// Single-process continuous-batching demo: drive the `ContinuousBatcher`
    /// over the paged-KV `ContinuousBatchDriver` with several prompts (admit →
    /// mixed prefill/decode batch → retire), exercising prefix reuse and CUDA
    /// graphs, then print per-request output + driver metrics. Runs instead of
    /// the HTTP server.
    #[arg(long)]
    pub batch_demo: bool,

    /// Comma-separated prompts for `--batch-demo` (defaults to a built-in set
    /// that shares a prefix to exercise cross-request reuse).
    #[arg(long)]
    pub demo_prompts: Option<String>,

    /// Enable CUDA-graph capture/replay for uniform decode steps in the demo.
    #[arg(long)]
    pub cuda_graphs: bool,
}

#[derive(Args, Debug, Clone)]
pub struct CalibrateArgs {
    /// Hardware kind (matches `[peak_tflops.<kind>]` in `cost_constants.toml`).
    #[arg(long)]
    pub hardware: String,
    #[arg(long)]
    pub model: PathBuf,
    /// Calibration corpus TOML.
    #[arg(long)]
    pub corpus: PathBuf,
    /// Override the public drift-prompt JSONL carried by the corpus.
    #[arg(long)]
    pub drift_prompts_path: Option<PathBuf>,
    /// Reference bf16 Skein artifact for Skein-vs-Skein drift calibration.
    #[arg(long)]
    pub reference_artifact: Option<PathBuf>,
    /// Candidate Skein artifact for Skein-vs-Skein drift calibration.
    #[arg(long)]
    pub candidate_artifact: Option<PathBuf>,
    #[arg(long, default_value = "cluster/cost_constants.toml")]
    pub out_cost: PathBuf,
    #[arg(long, default_value = "models/")]
    pub out_drift_dir: PathBuf,
}

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

const fn default_search_budget() -> usize {
    if cfg!(feature = "cuda") { 100 } else { 10 }
}
