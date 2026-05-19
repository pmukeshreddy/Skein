//! Test 4 — clap argument-parsing completeness for every subcommand.

use clap::Parser;
use clap::error::ErrorKind;

use skein_cli::Cli;
use skein_cli::cli::{BenchMetric, Command, OutputFormat};

#[test]
fn extract_parses_required_args() {
    let cli = Cli::try_parse_from([
        "skein",
        "extract",
        "--model",
        "/a.json",
        "--cluster",
        "/b.toml",
        "--trace",
        "/c.jsonl",
        "--drift",
        "/d.toml",
        "--cost",
        "/e.toml",
    ])
    .expect("extract should parse with the required flags");
    match cli.command {
        Command::Extract(args) => {
            assert_eq!(args.model.to_str().unwrap(), "/a.json");
            assert_eq!(args.cluster.to_str().unwrap(), "/b.toml");
            assert_eq!(args.trace.to_str().unwrap(), "/c.jsonl");
            assert_eq!(args.drift.to_str().unwrap(), "/d.toml");
            assert_eq!(args.cost.to_str().unwrap(), "/e.toml");
            // Default `out`.
            assert_eq!(args.out.to_str().unwrap(), "plan.json");
            assert!(!args.disaggregated);
        }
        other => panic!("unexpected subcommand: {other:?}"),
    }
}

#[test]
fn extract_missing_required_arg_errors() {
    let err = Cli::try_parse_from([
        "skein", "extract", "--model", "/a.json",
        // missing --cluster, --trace, ...
    ])
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
    assert!(err.to_string().contains("--cluster"));
}

#[test]
fn compile_parses_required_args() {
    let cli = Cli::try_parse_from([
        "skein",
        "compile",
        "--model",
        "/m.json",
        "--cluster",
        "/c.toml",
        "--trace",
        "/t.jsonl",
        "--drift",
        "/d.toml",
        "--cost",
        "/k.toml",
        "--weights",
        "/w/",
    ])
    .expect("compile parse");
    if let Command::Compile(args) = cli.command {
        assert_eq!(args.search_budget, 10);
        assert!(!args.enforce_parity);
        assert_eq!(args.n_parity_prompts, 8);
    } else {
        panic!("expected Compile");
    }
}

#[test]
fn verify_parses_required_args() {
    let cli = Cli::try_parse_from(["skein", "verify", "--artifact", "/a"]).expect("verify parse");
    if let Command::Verify(args) = cli.command {
        assert_eq!(args.n_prompts, 50, "default n_prompts");
        assert!(args.reference.is_none());
        assert!(args.sample_from.is_none());
        assert!(!args.enforce);
    } else {
        panic!("expected Verify");
    }
}

#[test]
fn serve_parses_required_args() {
    let cli = Cli::try_parse_from(["skein", "serve", "--artifact", "/a"]).expect("serve parse");
    if let Command::Serve(args) = cli.command {
        assert_eq!(args.port, 8080);
        assert!(!args.enable_hot_swap);
    } else {
        panic!("expected Serve");
    }
}

#[test]
fn calibrate_parses_required_args() {
    let cli = Cli::try_parse_from([
        "skein",
        "calibrate",
        "--hardware",
        "h100_sxm5",
        "--model",
        "/m.json",
        "--corpus",
        "/c.toml",
    ])
    .expect("calibrate parse");
    if let Command::Calibrate(args) = cli.command {
        assert!(args.reference_artifact.is_none());
        assert!(args.candidate_artifact.is_none());
    } else {
        panic!("expected Calibrate");
    }
}

#[test]
fn bench_parses_required_args() {
    let cli = Cli::try_parse_from([
        "skein",
        "bench",
        "--artifact",
        "/a",
        "--workload",
        "/w.jsonl",
        "--baseline",
        "http://localhost:8000",
        "--metrics",
        "throughput,goodput,drift_compliance",
    ])
    .expect("bench parse");
    if let Command::Bench(args) = cli.command {
        assert_eq!(
            args.metrics,
            vec![
                BenchMetric::Throughput,
                BenchMetric::Goodput,
                BenchMetric::DriftCompliance,
            ]
        );
    } else {
        panic!("expected Bench");
    }
}

#[test]
fn top_level_help_is_help_kind() {
    // `--help` triggers DisplayHelp; assert that's what clap produces so we
    // never accidentally start swallowing help in a future refactor.
    let err = Cli::try_parse_from(["skein", "--help"]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::DisplayHelp);
}

#[test]
fn extract_help_is_help_kind() {
    let err = Cli::try_parse_from(["skein", "extract", "--help"]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::DisplayHelp);
}

#[test]
fn no_subcommand_prints_help() {
    // `arg_required_else_help = true` on the top-level `Cli` causes clap to
    // exit with `DisplayHelpOnMissingArgumentOrSubcommand` when no subcommand
    // is provided. Older clap revisions returned `MissingSubcommand`; this
    // matches both so a future clap bump doesn't silently regress.
    let err = Cli::try_parse_from(["skein"]).unwrap_err();
    assert!(
        matches!(
            err.kind(),
            ErrorKind::DisplayHelp
                | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
                | ErrorKind::MissingSubcommand,
        ),
        "got {:?}",
        err.kind()
    );
}

#[test]
fn global_output_flag_overrides_default() {
    let cli = Cli::try_parse_from([
        "skein",
        "--output",
        "json",
        "extract",
        "--model",
        "/a.json",
        "--cluster",
        "/b.toml",
        "--trace",
        "/c.jsonl",
        "--drift",
        "/d.toml",
        "--cost",
        "/e.toml",
    ])
    .unwrap();
    assert_eq!(cli.output, OutputFormat::Json);
}
