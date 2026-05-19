//! End-to-end fixture tests: every shipped sample file (configs/, cluster/)
//! must parse successfully. If any of these regress, downstream demos break.

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points to crates/skein_ir; the repo root is two up.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

#[test]
fn mixtral_8x7b_config_imports_from_disk() {
    let path = repo_root().join("configs/mixtral_8x7b_config.json");
    let g = skein_ir::model::import_from_file(&path).expect("import from disk");
    assert_eq!(g.meta.architecture, "MixtralForCausalLM");
    assert_eq!(g.meta.num_layers, 32);
    assert_eq!(g.num_decoder_blocks(), 32);
}

#[test]
fn h100_2x_cluster_parses_from_disk() {
    let path = repo_root().join("cluster/h100_2x.toml");
    let c = skein_ir::cluster::ClusterSpec::from_toml_file(&path).expect("parse cluster");
    assert_eq!(c.num_devices, 2);
    assert_eq!(c.nodes.len(), 1);
    assert_eq!(c.links.len(), 1);
}

#[test]
fn sample_trace_parses_from_disk() {
    let path = repo_root().join("cluster/sample_trace.jsonl");
    let w = skein_ir::workload::Workload::from_jsonl_file(&path).expect("parse trace");
    assert_eq!(w.slo.ttft_p95_ms, 500);
    assert_eq!(w.slo.tpot_p95_ms, 50);
    assert_eq!(w.requests.len(), 5);
    // Arrival times must be monotonically non-decreasing — the parser would
    // have rejected otherwise, but assert anyway for clarity.
    let mut prev = 0;
    for r in &w.requests {
        assert!(r.arrival_ms >= prev);
        prev = r.arrival_ms;
    }
}
