//! Shared fixtures + tempdir helpers for `skein_cli` integration tests.

#![allow(dead_code)]

use std::path::PathBuf;

use skein_cli::cli::ExtractArgs;

pub fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

pub fn tempdir(prefix: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "{prefix}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// `ExtractArgs` pointing to the bundled Mixtral + 2× H100 + drift +
/// cost-constants fixtures, with `out` redirected into a fresh tempdir.
pub fn mixtral_extract_args(prefix: &str) -> (ExtractArgs, PathBuf) {
    let dir = tempdir(prefix);
    let args = ExtractArgs {
        model: repo_root().join("configs/mixtral_8x7b_config.json"),
        cluster: repo_root().join("cluster/h100_2x.toml"),
        trace: repo_root().join("cluster/sample_trace.jsonl"),
        drift: repo_root().join("models/mixtral_8x7b_drift.toml"),
        cost: repo_root().join("cluster/cost_constants.toml"),
        out: dir.join("plan.json"),
        disaggregated: false,
    };
    (args, dir)
}
