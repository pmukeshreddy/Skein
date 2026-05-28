//! Output formatting — `tracing` setup + extract report rendering.
//!
//! Two output modes:
//!
//! - **text** — human-readable. Progress logs flow through `tracing` to
//!   stderr; the final summary is a tidy block on stdout.
//! - **json** — a single JSON object on stdout, with `tracing` redirected
//!   to a no-op so the JSON line is the only thing on stdout. Suitable
//!   for piping to `jq` or downstream tools.
//!
//! Both formats are *byte-stable* on identical inputs — the JSON path
//! sorts map keys (via `BTreeMap`) so tests can compare bytes directly.

use std::collections::BTreeMap;
use std::sync::Once;

use serde::Serialize;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::cli::OutputFormat;
use crate::error::CliError;

/// Per-process subscriber installation guard. Tests construct multiple
/// `Cli` runs in one process; we only set the subscriber once.
static INSTALL_ONCE: Once = Once::new();

/// Env var: when set to an OTLP gRPC endpoint (e.g. `http://localhost:4317`),
/// the request/step spans are exported via OpenTelemetry in addition to the
/// stderr logs.
const OTLP_ENV: &str = "SKEIN_OTLP_ENDPOINT";

pub fn install_tracing(log_level: &str, output: OutputFormat) {
    INSTALL_ONCE.call_once(|| {
        // Honor `RUST_LOG` if set; otherwise fall back to `--log-level`.
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
        // JSON mode: keep stdout clean for the final JSON object by
        // routing every span/event to stderr without ANSI colors.
        let ansi = matches!(output, OutputFormat::Text);
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_ansi(ansi);

        // Optional OpenTelemetry OTLP export of the `tracing` spans
        // (`skein.request` / `skein.step`). `Option<Layer>` is itself a
        // `Layer`, so a single composed subscriber covers both cases.
        let otel_layer = std::env::var(OTLP_ENV)
            .ok()
            .and_then(|endpoint| match build_otlp_layer(&endpoint) {
                Ok(layer) => Some(layer),
                Err(e) => {
                    eprintln!("skein: OTLP export disabled ({OTLP_ENV}={endpoint}): {e}");
                    None
                }
            });

        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(otel_layer)
            .init();
    });
}

/// Build an OpenTelemetry tracing layer that exports spans to `endpoint` over
/// OTLP/gRPC. Requires a tokio runtime to be active (we install from inside
/// `#[tokio::main]`).
fn build_otlp_layer<S>(
    endpoint: &str,
) -> Result<tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>, String>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    use opentelemetry_otlp::WithExportConfig;
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(endpoint.to_string());
    // `install_batch` returns the `Tracer` directly in this otlp version.
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .install_batch(opentelemetry_sdk::runtime::Tokio)
        .map_err(|e| e.to_string())?;
    Ok(tracing_opentelemetry::layer().with_tracer(tracer))
}

/// Per-plan summary block emitted at the end of `skein extract`.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractReport {
    /// Hex-encoded `Plan::content_hash`. Stable across runs for the same Plan.
    pub plan_hash: String,
    /// Path the plan was written to.
    pub plan_path: String,
    /// Wall-clock time the search took, milliseconds.
    pub search_wall_ms: u64,
    /// Predicted per-step cost in microseconds.
    pub cost_us: f64,
    pub parallelism: ReportParallelism,
    /// Count of decoder blocks per weight dtype. `BTreeMap` so iteration
    /// (and therefore JSON serialization) is sorted by key.
    pub dtype_summary: BTreeMap<String, usize>,
    /// Architecture string from `ModelMeta`.
    pub model: String,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ReportParallelism {
    pub tp: u32,
    pub pp: u32,
    pub ep: u32,
}

/// Print the report in the requested format. JSON goes to stdout as a
/// single line; text formats as a human-readable block.
pub fn print_report(report: &ExtractReport, output: OutputFormat) -> Result<(), CliError> {
    match output {
        OutputFormat::Json => {
            // Serialize via the struct's Serialize impl — struct fields
            // emit in declaration order, and `dtype_summary` is a BTreeMap
            // so its inner keys are sorted. Result: byte-stable JSON.
            let s = serde_json::to_string(report)?;
            println!("{s}");
        }
        OutputFormat::Text => {
            println!("Plan summary:");
            println!("  hash         {}", report.plan_hash);
            println!(
                "  parallelism  tp={} pp={} ep={}",
                report.parallelism.tp, report.parallelism.pp, report.parallelism.ep,
            );
            println!("  model        {}", report.model);
            let dtype_str = report
                .dtype_summary
                .iter()
                .map(|(k, v)| format!("{v} layers @ {k}"))
                .collect::<Vec<_>>()
                .join(", ");
            println!("  dtypes       {dtype_str}");
            println!("  cost         {:.2} µs / step (predicted)", report.cost_us);
            println!("  search       {} ms wall", report.search_wall_ms);
            println!("  written to   {}", report.plan_path);
        }
    }
    Ok(())
}

/// Render a `CliError` to stderr. JSON mode emits a structured error
/// object on stderr (kept off stdout so consumers of the success path
/// see a clean output stream); text mode prints the chained message.
pub fn print_error(err: &CliError, output: OutputFormat) {
    match output {
        OutputFormat::Text => {
            eprintln!("error: {err}");
        }
        OutputFormat::Json => {
            let payload = ErrorPayload {
                error: format!("{err}"),
                kind: err_kind(err),
            };
            // Best-effort — if even serializing fails, fall back to text.
            match serde_json::to_string(&payload) {
                Ok(s) => eprintln!("{s}"),
                Err(_) => eprintln!("error: {err}"),
            }
        }
    }
}

#[derive(Serialize)]
struct ErrorPayload {
    error: String,
    kind: &'static str,
}

fn err_kind(e: &CliError) -> &'static str {
    match e {
        CliError::RequiresCuda { .. } => "requires_cuda",
        CliError::BadArgument(_) => "bad_argument",
        CliError::Io(_) => "io",
        CliError::Serde(_) => "serde",
        CliError::Cost(_) => "cost",
        CliError::Extract(_) => "extract",
        CliError::Emit(_) => "emit",
        CliError::Compile(_) => "compile",
        CliError::Parity(_) => "parity",
        CliError::Calibration(_) => "calibration",
        CliError::Runtime(_) => "runtime",
        CliError::PlanHash(_) => "plan_hash",
        CliError::ParityFailed { .. } => "parity_failed",
        CliError::Other(_) => "other",
    }
}
