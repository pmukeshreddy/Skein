//! `skein` CLI internals. Exposed as a library so integration tests can
//! drive the command runners directly without re-building the binary and
//! shelling out.

pub mod cli;
pub mod cmd;
pub mod error;
pub mod load;
pub mod output;

pub use cli::{BenchMetric, Cli, Command, OutputFormat};
pub use error::CliError;
pub use output::{ExtractReport, ReportParallelism};
