//! Shared fixtures for `skein_calibrate` integration tests.

#![allow(dead_code)]

use std::path::PathBuf;

use skein_cost::{CostConstants, CostModel};

pub fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

pub fn load_cost_constants() -> CostConstants {
    CostModel::load(&repo_root().join("cluster/cost_constants.toml"))
        .expect("load cost constants")
        .constants()
        .clone()
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
