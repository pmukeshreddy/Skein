//! Shared fixtures for `skein_extract` integration tests.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use skein_cost::{Cluster, CostModel};
use skein_extract::DriftTable;
use skein_ir::cluster::ClusterSpec;
use skein_ir::ir::Graph;
use skein_ir::types::{Component, Dtype};
use skein_ir::workload::Workload;

pub fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

pub fn load_mixtral_ir() -> Graph {
    let path = repo_root().join("configs/mixtral_8x7b_config.json");
    let s = std::fs::read_to_string(&path).expect("read mixtral config");
    skein_ir::model::import_from_str(&s).expect("import mixtral")
}

pub fn load_2x_h100_cluster() -> Cluster {
    let spec = ClusterSpec::from_toml_file(&repo_root().join("cluster/h100_2x.toml"))
        .expect("parse cluster");
    Cluster::from_spec(spec)
}

pub fn load_cost_model() -> CostModel {
    CostModel::load(&repo_root().join("cluster/cost_constants.toml")).expect("load cost model")
}

pub fn load_drift_table() -> DriftTable {
    DriftTable::load(&repo_root().join("models/mixtral_8x7b_drift.toml")).expect("load drift table")
}

pub fn load_workload() -> Workload {
    Workload::from_jsonl_file(&repo_root().join("cluster/sample_trace.jsonl"))
        .expect("load workload trace")
}

/// Build a DriftTable directly from per-(component, dtype) defaults. Tests
/// use this to dial drift up/down without touching the on-disk file.
pub fn make_drift_table(
    weights: [(Dtype, f64); 6],
    acts: [(Dtype, f64); 6],
    kvs: [(Dtype, f64); 6],
) -> DriftTable {
    let mut defaults: HashMap<(Component, Dtype), f64> = HashMap::new();
    for (d, v) in weights {
        defaults.insert((Component::Weight, d), v);
    }
    for (d, v) in acts {
        defaults.insert((Component::Activation, d), v);
    }
    for (d, v) in kvs {
        defaults.insert((Component::KvCache, d), v);
    }
    DriftTable::with_defaults(defaults).expect("build drift table")
}

/// Default flat drift table — useful when the test only cares about memory.
pub fn flat_drift_table() -> DriftTable {
    let w = [
        (Dtype::Bf16, 0.0),
        (Dtype::Fp16, 0.0),
        (Dtype::Fp8E4m3, 0.0),
        (Dtype::Fp8E5m2, 0.0),
        (Dtype::Int8, 0.0),
        (Dtype::Int4, 0.0),
    ];
    make_drift_table(w, w, w)
}
