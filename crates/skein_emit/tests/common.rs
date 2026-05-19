//! Shared fixtures for `skein_emit` integration tests.

#![allow(dead_code)]

use std::path::PathBuf;

use skein_cost::Cluster;
use skein_ir::cluster::ClusterSpec;
use skein_ir::ir::{Graph, ModelMeta};
use skein_ir::plan::*;
use skein_ir::types::*;

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

/// Build a 4-device single-node cluster — needed for `pp=2 × tp=2` tests
/// and for the EP-routing test.
pub fn build_4x_h100_cluster() -> Cluster {
    let toml = r#"
num_devices = 4
[[node]]
id               = "node0"
devices          = ["d0", "d1", "d2", "d3"]
device_kind      = "h100_sxm5"
device_memory_gb = 80
[[link]]
endpoints      = ["d0", "d1"]
kind           = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us     = 1.0
[[link]]
endpoints      = ["d2", "d3"]
kind           = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us     = 1.0
[[link]]
endpoints      = ["d0", "d2"]
kind           = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us     = 1.0
[[link]]
endpoints      = ["d1", "d3"]
kind           = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us     = 1.0
"#;
    let spec = ClusterSpec::from_toml_str(toml).expect("parse 4× cluster");
    Cluster::from_spec(spec)
}

pub fn mk_plan(meta: ModelMeta, tp: u32, pp: u32, ep: u32) -> Plan {
    Plan {
        parallelism: ParallelismPlacement { tp, pp, ep },
        kv: KVCacheSpec {
            layout: KVLayout::Paged { page_size: 32 },
            kv_sharded: false,
        },
        batching: BatchPolicy::Continuous { max_batch: 8 },
        dtype_map: DtypeMap::uniform(meta.num_layers, Dtype::Bf16),
        execution: ExecutionConfig {
            cuda_graphs: CudaGraphsConfig {
                enable: false,
                capture_classes: vec![],
            },
            spec_decode: SpecDecodeConfig {
                enable: false,
                draft: None,
            },
            prefix_cache: PrefixCacheConfig {
                enable: false,
                reuse_policy: RadixReusePolicy::LruByLastAccess,
            },
        },
        disaggregation: None,
        model_meta: meta,
    }
}
