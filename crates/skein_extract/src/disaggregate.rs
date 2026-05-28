//! Prefill/decode (P/D) disaggregation planning.
//!
//! Splits the cluster into a **prefill pool** and a **decode pool**, runs the
//! normal plan search independently over each pool's sub-cluster, and composes
//! a [`Disaggregation`] (prefill plan + decode plan + KV-transfer topology).
//! The KV computed by the prefill pool is shipped to the decode pool at serve
//! time over [`KvTransferMode::NcclByLayer`] (the `NcclKvTransport` in
//! `skein_runtime`).
//!
//! This is the planner side of P/D — composing a valid disaggregated `Plan`.
//! Lowering that plan to per-pool device graphs (`skein_emit`) and the
//! two-pool serve orchestration are the remaining, GPU-validated pieces.

use std::collections::HashSet;

use skein_cost::{Cluster, CostModel};
use skein_ir::cluster::{ClusterSpec, Node};
use skein_ir::ir::Graph;
use skein_ir::plan::{Disaggregation, KvTransferMode, Plan, TransferTopology};
use skein_ir::workload::Workload;

use crate::drift_table::DriftTable;
use crate::error::ExtractError;
use crate::extract_plan;

/// Every device id in the spec, in node-then-device order.
pub fn all_devices(spec: &ClusterSpec) -> Vec<String> {
    spec.nodes
        .iter()
        .flat_map(|n| n.devices.iter().cloned())
        .collect()
}

/// A sub-cluster spec restricted to `keep`: each node keeps only its devices in
/// `keep` (dropping empty nodes), links keep only edges with both endpoints in
/// `keep`, and `num_devices` is recomputed.
pub fn sub_cluster_spec(spec: &ClusterSpec, keep: &HashSet<String>) -> ClusterSpec {
    let nodes = spec
        .nodes
        .iter()
        .filter_map(|n| {
            let devices: Vec<String> =
                n.devices.iter().filter(|d| keep.contains(*d)).cloned().collect();
            if devices.is_empty() {
                None
            } else {
                Some(Node {
                    id: n.id.clone(),
                    devices,
                    device_kind: n.device_kind.clone(),
                    device_memory_gb: n.device_memory_gb,
                })
            }
        })
        .collect();
    let links = spec
        .links
        .iter()
        .filter(|l| keep.contains(&l.endpoints[0]) && keep.contains(&l.endpoints[1]))
        .cloned()
        .collect();
    ClusterSpec {
        num_devices: keep.len() as u32,
        nodes,
        links,
    }
}

/// Search a disaggregated plan. The first `decode_devices` devices form the
/// decode pool; the rest form the prefill pool. Each pool runs the full search
/// over its own sub-cluster. Returns the decode `Plan` (the serve-time primary)
/// carrying the [`Disaggregation`].
pub fn extract_disaggregated_plan(
    ir: &Graph,
    cluster: &Cluster,
    workload: &Workload,
    drift_table: &DriftTable,
    cost_model: &CostModel,
    decode_devices: usize,
) -> Result<Plan, ExtractError> {
    let spec = cluster.spec();
    let devices = all_devices(spec);
    let total = devices.len();
    if decode_devices == 0 || decode_devices >= total {
        return Err(ExtractError::InvalidDisaggregation {
            decode_devices,
            total,
        });
    }

    let decode_set: HashSet<String> = devices[..decode_devices].iter().cloned().collect();
    let prefill_set: HashSet<String> = devices[decode_devices..].iter().cloned().collect();

    let decode_cluster = Cluster::from_spec(sub_cluster_spec(spec, &decode_set));
    let prefill_cluster = Cluster::from_spec(sub_cluster_spec(spec, &prefill_set));

    // Each pool searches independently over the same workload + drift + cost inputs.
    let decode_plan = extract_plan(ir, &decode_cluster, workload, drift_table, cost_model)?;
    let prefill_plan = extract_plan(ir, &prefill_cluster, workload, drift_table, cost_model)?;

    let mut plan = decode_plan.clone();
    plan.disaggregation = Some(Disaggregation {
        prefill: Box::new(prefill_plan),
        decode: Box::new(decode_plan),
        transfer: TransferTopology {
            mode: KvTransferMode::NcclByLayer,
            layer_overlap: true,
        },
    });
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn four_device_spec() -> ClusterSpec {
        ClusterSpec::from_toml_str(
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
        .expect("spec parses")
    }

    #[test]
    fn sub_cluster_keeps_only_subset_devices_and_internal_links() {
        let spec = four_device_spec();
        let keep: HashSet<String> = ["d0", "d1"].iter().map(|s| s.to_string()).collect();
        let sub = sub_cluster_spec(&spec, &keep);
        assert_eq!(sub.num_devices, 2);
        assert_eq!(all_devices(&sub), vec!["d0", "d1"]);
        // Only the d0-d1 link survives; d2-d3 and d1-d2 are dropped.
        assert_eq!(sub.links.len(), 1);
        assert_eq!(sub.links[0].endpoints, ["d0".to_string(), "d1".to_string()]);
    }

    #[test]
    fn all_devices_is_node_then_device_order() {
        let spec = four_device_spec();
        assert_eq!(all_devices(&spec), vec!["d0", "d1", "d2", "d3"]);
    }
}
