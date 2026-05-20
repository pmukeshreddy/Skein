//! Shared fixtures for `skein_parity` integration tests.
//!
//! These tests exercise the real `verify_plan` / `verify_skein_pair`
//! orchestration logic (tolerance gating, KL SLO gating, failing-layer
//! selection) using small test-local doubles for the reference and
//! artifact-side forward passes. The doubles live here, in the test crate —
//! they are not part of the shipped library surface.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use skein_cost::{CostConstants, CostModel};
use skein_ir::ir::ModelMeta;
use skein_ir::plan::*;
use skein_ir::types::*;
use skein_ir::workload::{Slo, Workload};
use skein_parity::{
    HFReference, ParityError, ReferenceOutput, SkeinForward, SkeinOutput, ToleranceTable,
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

pub fn sample_prompts() -> Vec<String> {
    vec![
        "alpha".to_string(),
        "bravo".to_string(),
        "charlie".to_string(),
        "delta".to_string(),
    ]
}

/// Deterministic, model-free tokenizer for tests: maps a prompt to a fixed
/// four-token sequence derived from its bytes. Distinct prompts map to
/// distinct sequences, which is all the orchestration tests need.
fn tokenize_for_test(prompt: &str) -> Vec<u32> {
    let bytes = prompt.as_bytes();
    (0..4)
        .map(|i| {
            let mut acc = 0u32;
            for (j, b) in bytes.iter().enumerate() {
                if j % 4 == i {
                    acc = acc.wrapping_mul(31).wrapping_add(*b as u32);
                }
            }
            acc % 32_000
        })
        .collect()
}

/// In-memory `HFReference`: a fixed token→activations map plus the test
/// tokenizer. Per-block activations are the constant `0.1 * (block + 1)`
/// and logits are `[0, 1, .., VOCAB)` — small values that make the MSE / KL
/// arithmetic easy to reason about by hand.
#[derive(Debug, Clone)]
pub struct InMemoryReference {
    entries: HashMap<Vec<u32>, ReferenceOutput>,
}

impl InMemoryReference {
    pub fn for_prompts(prompts: &[String]) -> Self {
        let mut entries = HashMap::new();
        for prompt in prompts {
            let per_layer_activations = (0..NUM_BLOCKS)
                .map(|b| vec![0.1_f32 * (b as f32 + 1.0); HIDDEN])
                .collect();
            let final_logits = (0..VOCAB).map(|i| i as f32).collect();
            entries.insert(
                tokenize_for_test(prompt),
                ReferenceOutput {
                    per_layer_activations,
                    final_logits,
                },
            );
        }
        Self { entries }
    }

    pub fn lookup(&self, tokens: &[u32]) -> ReferenceOutput {
        self.entries
            .get(tokens)
            .cloned()
            .expect("reference has no entry for tokens")
    }
}

impl HFReference for InMemoryReference {
    fn forward_with_hooks(&self, tokens: &[u32]) -> Result<ReferenceOutput, ParityError> {
        Ok(self.lookup(tokens))
    }

    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, ParityError> {
        Ok(tokenize_for_test(prompt))
    }
}

/// Test-local `SkeinForward` that derives its output from an
/// [`InMemoryReference`] plus a configurable per-layer additive offset and a
/// uniform final-logit offset. The per-layer MSE between reference and Skein
/// is then exactly `offset²`, which lets a test dial drift to a known
/// magnitude.
pub struct OffsetForward {
    reference: InMemoryReference,
    layer_offsets: Vec<f32>,
    final_logit_offset: f32,
}

impl OffsetForward {
    pub fn new(reference: InMemoryReference) -> Self {
        Self {
            reference,
            layer_offsets: Vec::new(),
            final_logit_offset: 0.0,
        }
    }

    pub fn with_layer_offsets(mut self, offsets: Vec<f32>) -> Self {
        self.layer_offsets = offsets;
        self
    }

    pub fn with_final_logit_offset(mut self, offset: f32) -> Self {
        self.final_logit_offset = offset;
        self
    }
}

impl SkeinForward for OffsetForward {
    fn forward_with_hooks(&mut self, tokens: &[u32]) -> Result<SkeinOutput, ParityError> {
        let ReferenceOutput {
            mut per_layer_activations,
            mut final_logits,
        } = self.reference.lookup(tokens);
        for (i, layer) in per_layer_activations.iter_mut().enumerate() {
            let offset = self.layer_offsets.get(i).copied().unwrap_or(0.0);
            if offset != 0.0 {
                layer.iter_mut().for_each(|v| *v += offset);
            }
        }
        if self.final_logit_offset != 0.0 {
            final_logits
                .iter_mut()
                .for_each(|v| *v += self.final_logit_offset);
        }
        Ok(SkeinOutput {
            per_layer_activations,
            final_logits,
        })
    }
}
