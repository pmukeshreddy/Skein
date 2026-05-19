//! Shared fixtures for `skein_parity` integration tests. Builds a tiny
//! in-memory `MockReference` whose activations match the real Mixtral
//! `num_layers = 32` so the per-layer loop in `verify_plan` has work to do.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use skein_cost::{CostConstants, CostModel};
use skein_ir::ir::ModelMeta;
use skein_ir::plan::*;
use skein_ir::types::*;
use skein_ir::workload::{Slo, Workload};
use skein_parity::{
    HFReference, MockReference, PhaseAStub, ReferenceOutput, SkeinForward, ToleranceTable,
};

pub const NUM_BLOCKS: usize = 32;
pub const HIDDEN: usize = 4;
pub const VOCAB: usize = 6;

pub fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

pub fn mixtral_meta() -> ModelMeta {
    ModelMeta {
        architecture: "MixtralForCausalLM".into(),
        num_layers: NUM_BLOCKS,
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

pub fn mk_plan(weight: Dtype, activation: Dtype, kv: Dtype) -> Plan {
    Plan {
        parallelism: ParallelismPlacement {
            tp: 2,
            pp: 1,
            ep: 1,
        },
        kv: KVCacheSpec {
            layout: KVLayout::Paged { page_size: 32 },
            kv_sharded: false,
        },
        batching: BatchPolicy::Continuous { max_batch: 8 },
        dtype_map: DtypeMap {
            per_layer: vec![
                PerLayerDtype {
                    weight,
                    activation,
                    kv_cache: kv,
                };
                NUM_BLOCKS
            ],
        },
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
        model_meta: mixtral_meta(),
    }
}

pub fn loose_slo() -> Workload {
    Workload {
        slo: Slo {
            ttft_p95_ms: 500,
            tpot_p95_ms: 50,
            max_accuracy_drift: 1.0,
            recompile_drift_threshold_kl: 0.05,
        },
        requests: vec![],
    }
}

pub fn tight_slo() -> Workload {
    Workload {
        slo: Slo {
            ttft_p95_ms: 500,
            tpot_p95_ms: 50,
            max_accuracy_drift: 1.0e-6,
            recompile_drift_threshold_kl: 0.05,
        },
        requests: vec![],
    }
}

pub fn load_cost_constants() -> CostConstants {
    let path = repo_root().join("cluster/cost_constants.toml");
    CostModel::load(&path)
        .expect("load cost model")
        .constants()
        .clone()
}

pub fn load_tolerances() -> ToleranceTable {
    ToleranceTable::from_cost_constants(&load_cost_constants())
}

/// Deterministic mock-reference activations for a list of test prompts.
/// Tokenization uses `MockReference::tokenize`'s BLAKE3 scheme, so each
/// prompt maps to a unique token vector. The activation values are
/// per-block constant `0.1 * (block_idx + 1)` and logits are
/// `[0.0..VOCAB]` — these are arbitrary but small enough that drift
/// injection makes the MSE math easy to reason about by hand.
pub fn build_mock_reference(prompts: &[String]) -> MockReference {
    let mut entries: HashMap<Vec<u32>, ReferenceOutput> = HashMap::new();
    let stub_for_tokenize = MockReference::from_entries(HashMap::new());
    for prompt in prompts {
        let tokens = stub_for_tokenize.tokenize(prompt).unwrap();
        let per_layer_activations: Vec<Vec<f32>> = (0..NUM_BLOCKS)
            .map(|b| vec![0.1_f32 * (b as f32 + 1.0); HIDDEN])
            .collect();
        let final_logits: Vec<f32> = (0..VOCAB).map(|i| i as f32).collect();
        entries.insert(
            tokens,
            ReferenceOutput {
                per_layer_activations,
                final_logits,
            },
        );
    }
    MockReference::from_entries(entries)
}

pub fn sample_prompts() -> Vec<String> {
    vec![
        "alpha".to_string(),
        "bravo".to_string(),
        "charlie".to_string(),
        "delta".to_string(),
    ]
}

/// Sanity helper for tests that don't actually exercise the stub.
pub fn passthrough_stub(reference: MockReference) -> impl SkeinForward {
    PhaseAStub::new(reference)
}
