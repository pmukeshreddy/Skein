//! Loader helpers shared by the command runners. All return
//! `anyhow::Result<_>` and attach `.context()` so the user sees *which*
//! file failed, not just the underlying syntax error.

use std::path::Path;

use anyhow::Context;

use skein_cost::CostModel;
use skein_extract::DriftTable;
use skein_ir::cluster::ClusterSpec;
use skein_ir::ir::Graph;
use skein_ir::workload::Workload;

pub fn load_ir(path: &Path) -> anyhow::Result<Graph> {
    skein_ir::model::import_from_file(path)
        .with_context(|| format!("loading model IR from {}", path.display()))
}

pub fn load_cluster(path: &Path) -> anyhow::Result<ClusterSpec> {
    ClusterSpec::from_toml_file(path)
        .with_context(|| format!("loading cluster topology from {}", path.display()))
}

pub fn load_workload(path: &Path) -> anyhow::Result<Workload> {
    Workload::from_jsonl_file(path)
        .with_context(|| format!("loading workload trace from {}", path.display()))
}

pub fn load_drift_table(path: &Path) -> anyhow::Result<DriftTable> {
    DriftTable::load(path).with_context(|| format!("loading drift table from {}", path.display()))
}

pub fn load_cost_model(path: &Path) -> anyhow::Result<CostModel> {
    CostModel::load(path).with_context(|| format!("loading cost constants from {}", path.display()))
}
