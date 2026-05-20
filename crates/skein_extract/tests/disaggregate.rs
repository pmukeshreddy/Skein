//! P/D disaggregation planning: split the cluster into prefill + decode pools
//! and compose a `Disaggregation`.

mod common;
use common::*;

use skein_cost::Cluster;
use skein_extract::{ExtractError, extract_disaggregated_plan};
use skein_ir::cluster::ClusterSpec;
use skein_ir::plan::KvTransferMode;

/// 4× H100, fully enough linked that each 2-device pool can run tp=2.
fn four_h100_cluster() -> Cluster {
    let spec = ClusterSpec::from_toml_str(
        r#"
num_devices = 4
[[node]]
id = "n0"
devices = ["d0","d1","d2","d3"]
device_kind = "h100_sxm5"
device_memory_gb = 80
[[link]]
endpoints = ["d0","d1"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d2","d3"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d1","d2"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
"#,
    )
    .expect("4x cluster parses");
    Cluster::from_spec(spec)
}

#[test]
fn disaggregated_plan_splits_into_prefill_and_decode_pools() {
    let ir = load_mixtral_ir();
    let cluster = four_h100_cluster();
    let workload = load_workload();
    let drift = load_drift_table();
    let cost = load_cost_model();

    // 4-device cluster → 2 decode devices, 2 prefill devices (each runs tp=2,
    // which is what Mixtral 8x7B needs to fit per pool).
    let plan =
        extract_disaggregated_plan(&ir, &cluster, &workload, &drift, &cost, 2).expect("disagg plan");

    let d = plan
        .disaggregation
        .as_ref()
        .expect("disaggregation is populated");
    // Each pool has two devices → tp·pp·ep == 2.
    let prod = |p: &skein_ir::plan::ParallelismPlacement| p.tp * p.pp * p.ep;
    assert_eq!(prod(&d.prefill.parallelism), 2, "prefill pool is 2 devices");
    assert_eq!(prod(&d.decode.parallelism), 2, "decode pool is 2 devices");
    assert_eq!(d.transfer.mode, KvTransferMode::NcclByLayer);
    // The composed plan (decode primary + disaggregation) is structurally valid.
    plan.validate().expect("disaggregated plan validates");
    // Both sub-plans cover every layer.
    assert_eq!(d.prefill.dtype_map.len(), ir.meta.num_layers);
    assert_eq!(d.decode.dtype_map.len(), ir.meta.num_layers);
}

#[test]
fn invalid_split_is_rejected() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let workload = load_workload();
    let drift = load_drift_table();
    let cost = load_cost_model();

    // 0 (empty decode pool), 2 (empty prefill pool), 3 (> total) are all invalid.
    for bad in [0usize, 2, 3] {
        let err = extract_disaggregated_plan(&ir, &cluster, &workload, &drift, &cost, bad);
        assert!(
            matches!(err, Err(ExtractError::InvalidDisaggregation { .. })),
            "decode_devices={bad} must be rejected, got {err:?}"
        );
    }
}
