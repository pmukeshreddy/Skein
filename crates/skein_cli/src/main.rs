//! `skein` binary entry point. Configures tracing, parses CLI, dispatches
//! to the per-subcommand runner.

use std::process::ExitCode;

use clap::Parser;

use skein_cli::{Cli, Command, output};

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    output::install_tracing(&cli.log_level, cli.output);

    let result: Result<(), skein_cli::CliError> = match cli.command {
        Command::Extract(args) => skein_cli::cmd::extract::run(args, cli.output),
        Command::Compile(args) => skein_cli::cmd::compile::run(args, cli.output),
        Command::Verify(args) => skein_cli::cmd::verify::run(args, cli.output),
        Command::Serve(args) => skein_cli::cmd::serve::run(args, cli.output).await,
        Command::Calibrate(args) => skein_cli::cmd::calibrate::run(args, cli.output),
        Command::Bench(args) => skein_cli::cmd::bench::run(args, cli.output),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            // Errors go to stderr; success output is on stdout (jq-friendly).
            output::print_error(&err, cli.output);
            ExitCode::from(1)
        }
    }
}
