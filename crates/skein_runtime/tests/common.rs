//! Shared fixtures for `skein_runtime` integration tests.

#![allow(dead_code)]

use std::path::PathBuf;

use skein_cost::{CostConstants, CostModel};
use skein_ir::ir::ModelMeta;
use skein_ir::plan::*;
use skein_ir::types::*;
use skein_ir::workload::{Slo, Workload};

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

pub fn mk_meta(num_layers: usize) -> ModelMeta {
    ModelMeta {
        architecture: "MixtralForCausalLM".into(),
        num_layers,
        hidden: 4096,
        vocab: 32_000,
        max_position: 32_768,
        num_attention_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        num_experts: Some(8),
        top_k: Some(2),
        intermediate: 14_336,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-5,
        sliding_window: None,
        tied_embeddings: false,
    }
}

pub fn mk_plan(page_size: u32, prefix_cache_enabled: bool, batching: BatchPolicy) -> Plan {
    Plan {
        parallelism: ParallelismPlacement {
            tp: 2,
            pp: 1,
            ep: 1,
        },
        kv: KVCacheSpec {
            layout: KVLayout::Paged { page_size },
            kv_sharded: false,
        },
        batching,
        dtype_map: DtypeMap::uniform(32, Dtype::Bf16),
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
                enable: prefix_cache_enabled,
                reuse_policy: RadixReusePolicy::LruByLastAccess,
            },
        },
        disaggregation: None,
        model_meta: mk_meta(32),
    }
}

pub fn mk_workload(ttft_p95_ms: u32, tpot_p95_ms: u32) -> Workload {
    Workload {
        slo: Slo {
            ttft_p95_ms,
            tpot_p95_ms,
            max_accuracy_drift: 0.01,
            recompile_drift_threshold_kl: 0.05,
        },
        requests: vec![],
    }
}
