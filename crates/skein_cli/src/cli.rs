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
    /// Plan search only — outputs `plan.json`. Runs on Mac (Phase A).
    Extract(ExtractArgs),

    /// Full compile: search + lower + Luminal compile + parity + artifact write.
    /// Phase B (requires `--features cuda` on an NVIDIA host).
    Compile(CompileArgs),

    /// Parity check on an existing artifact against the HF bf16 reference.
    /// Phase B.
    Verify(VerifyArgs),

    /// Launch the runtime server. Phase B.
    Serve(ServeArgs),

    /// Calibrate cost constants + drift table from GPU measurements. Phase B.
    Calibrate(CalibrateArgs),

    /// Three validation metrics vs vLLM. Phase B.
    Bench(BenchArgs),
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

    /// Enumerate P/D-disaggregated plans. Phase B only; Phase A rejects.
    #[arg(long)]
    pub disaggregated: bool,
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
    #[arg(long)]
    pub disaggregated: bool,
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
}

#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
    #[arg(long)]
    pub artifact: PathBuf,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    #[arg(long)]
    pub enable_hot_swap: bool,
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

#[derive(Args, Debug, Clone)]
pub struct BenchArgs {
    #[arg(long)]
    pub artifact: PathBuf,
    #[arg(long)]
    pub workload: PathBuf,
    /// vLLM endpoint URL (e.g. `http://localhost:8000`).
    #[arg(long)]
    pub baseline: String,
    /// Which metrics to run. Comma-separated.
    #[arg(long, value_delimiter = ',', value_enum)]
    pub metrics: Vec<BenchMetric>,
}

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq, Eq)]
pub enum BenchMetric {
    Throughput,
    Goodput,
    // Pin the snake_case spelling on the CLI so docs and `--help` agree.
    // Clap's default `ValueEnum` derive emits kebab-case (`drift-compliance`);
    // this attribute forces `drift_compliance` to match the spec example.
    #[value(name = "drift_compliance")]
    DriftCompliance,
}

const fn default_search_budget() -> usize {
    if cfg!(feature = "cuda") { 100 } else { 10 }
}
