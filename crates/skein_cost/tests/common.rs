//! Shared fixtures for the integration tests.
//!
//! Loads the real `cost_constants.toml` from disk so the tests exercise the
//! same TOML the production search loop uses. Tests live in
//! `crates/skein_cost/tests/` and reach back to the repo root via
//! `CARGO_MANIFEST_DIR`.

#![allow(dead_code)] // each test file uses some-but-not-all helpers

use std::path::PathBuf;

use skein_cost::{Cluster, CostModel};
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

pub fn load_cost_model() -> CostModel {
    let path = repo_root().join("cluster/cost_constants.toml");
    CostModel::load(&path).expect("load cluster/cost_constants.toml")
}

pub fn load_mixtral_ir() -> Graph {
    let path = repo_root().join("configs/mixtral_8x7b_config.json");
    let s = std::fs::read_to_string(&path).expect("read mixtral config");
    skein_ir::model::import_from_str(&s).expect("import mixtral")
}

pub fn load_2x_h100_cluster() -> Cluster {
    let path = repo_root().join("cluster/h100_2x.toml");
    let spec = ClusterSpec::from_toml_file(&path).expect("parse 2x H100 cluster");
    Cluster::from_spec(spec)
}

pub fn dummy_meta(num_layers: usize) -> ModelMeta {
    ModelMeta {
        architecture: "test".into(),
        num_layers,
        hidden: 4096,
        vocab: 32000,
        max_position: 32768,
        num_attention_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        num_experts: Some(8),
        top_k: Some(2),
        intermediate: 14336,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-5,
        sliding_window: None,
        tied_embeddings: false,
    }
}

#[allow(clippy::too_many_arguments)] // test-fixture builder: keep all axes explicit
pub fn plan_with(
    meta: ModelMeta,
    tp: u32,
    pp: u32,
    ep: u32,
    weight: Dtype,
    activation: Dtype,
    kv: Dtype,
    batching: BatchPolicy,
    cuda_graphs: CudaGraphsConfig,
) -> Plan {
    let num_blocks = meta.num_layers;
    Plan {
        parallelism: ParallelismPlacement { tp, pp, ep },
        kv: KVCacheSpec {
            layout: KVLayout::Paged { page_size: 32 },
            kv_sharded: false,
        },
        batching,
        dtype_map: DtypeMap {
            per_layer: vec![
                PerLayerDtype {
                    weight,
                    activation,
                    kv_cache: kv
                };
                num_blocks
            ],
        },
        execution: ExecutionConfig {
            cuda_graphs,
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
