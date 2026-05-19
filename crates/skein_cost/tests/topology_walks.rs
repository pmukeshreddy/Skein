//! Test 2 — multi-hop bandwidth bottleneck.

mod common;
use common::*;

use skein_cost::Cluster;
use skein_cost::collectives::{Collective, CollectiveKind};
use skein_ir::cluster::ClusterSpec;

const TWO_NODE_4GPU: &str = r#"
num_devices = 4
[[node]]
id               = "node0"
devices          = ["d0", "d1"]
device_kind      = "h100_sxm5"
device_memory_gb = 80
[[node]]
id               = "node1"
devices          = ["d2", "d3"]
device_kind      = "h100_sxm5"
device_memory_gb = 80
[[link]]
endpoints = ["d0", "d1"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d2", "d3"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d1", "d2"]
kind = "infiniband400g"
bandwidth_gbps = 400.0
latency_us = 5.0
"#;

#[test]
fn multi_hop_bandwidth_bottleneck() {
    let model = load_cost_model();
    let spec = ClusterSpec::from_toml_str(TWO_NODE_4GPU).unwrap();
    let cluster = Cluster::from_spec(spec);

    // AllReduce across all 4 participants. The critical path is d0 ↔ d3,
    // which traverses NVLink — IB — NVLink. The IB hop's 400 Gbps must be
    // the bottleneck.
    let all4 = Collective {
        kind: CollectiveKind::RingAllReduce,
        participants: vec![0, 1, 2, 3],
        bytes: 64 * 1024 * 1024,
    };
    let intra_node = Collective {
        kind: CollectiveKind::RingAllReduce,
        participants: vec![0, 1],
        bytes: 64 * 1024 * 1024,
    };

    let inter = model.comm_time_one(&all4, &cluster).unwrap();
    let intra = model.comm_time_one(&intra_node, &cluster).unwrap();

    // Inter-node (IB-bottlenecked at 400 Gbps) must be slower than intra-node
    // (NVLink at 900 Gbps).
    assert!(
        inter > intra,
        "inter-node {inter} should exceed intra-node {intra}"
    );

    // The IB bottleneck implies inter / intra ≥ 900/400 × (factor(4)/factor(2))
    //                                      = 2.25 × (1.5 / 1.0)
    //                                      = 3.375
    // — with some slack for shared latency overhead.
    assert!(
        inter / intra > 3.0,
        "ratio {} should reflect 900→400 bandwidth drop plus larger factor",
        inter / intra
    );

    // And there must be a critical path with min_bandwidth = 400.
    let path = cluster.topology().collective_path(&[0, 1, 2, 3]).unwrap();
    assert_eq!(path.min_bandwidth_gbps(), 400.0);
}
