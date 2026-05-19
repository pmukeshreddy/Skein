//! Typed IR `Graph` — the hardware-agnostic, plan-agnostic representation
//! produced by `skein_ir::model::import_from_file`.
//!
//! The IR is intentionally a thin layer above the HF config: enough structure
//! for `skein_cost` to score and `skein_emit` to lower, but no decisions
//! about sharding, quantization, or batching. Those live in `Plan`.
//!
//! Layer order matters — `layers` is in execution order from token embedding
//! through final lm_head.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::types::{Dtype, Shape};

/// Stable, monotonically-assigned tensor identifier.
pub type TensorId = u32;

/// A named tensor in the graph (input, output, or intermediate). Parameters
/// (weights) have a `Param` instead.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Shape,
    pub dtype: Dtype,
}

/// A weight tensor that will be loaded from safetensors. The `name` matches
/// the HF safetensors key (e.g. `model.layers.0.self_attn.q_proj.weight`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub shape: Shape,
    pub dtype: Dtype,
}

/// Activation used inside MLP / MoE blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    /// `x * sigmoid(x)` — SwiGLU's nonlinearity (Llama, Mistral, Mixtral).
    Silu,
    /// GELU (BERT-style). Not used by the current Mixtral importer but
    /// reserved for future architectures.
    Gelu,
}

/// Positional encoding strategy for attention.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttentionPosition {
    Rope { theta: f32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingCfg {
    pub vocab: usize,
    pub hidden: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormCfg {
    pub eps: f32,
    pub hidden: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionCfg {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub position: AttentionPosition,
    /// `Some(window)` for Mistral-style sliding window attention; `None` for
    /// full attention.
    pub sliding_window: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MlpCfg {
    pub hidden: usize,
    pub intermediate: usize,
    pub activation: Activation,
}

/// Mixture-of-Experts block. `top_k` of `num_experts` are routed per token.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoeCfg {
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub activation: Activation,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LmheadCfg {
    pub vocab: usize,
    pub hidden: usize,
    /// When `true`, the lm_head shares weights with the embedding table.
    /// `skein_emit` skips loading a separate lm_head weight in this case.
    pub tied: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayerKind {
    Embedding(EmbeddingCfg),
    RmsNorm(NormCfg),
    Attention(AttentionCfg),
    Mlp(MlpCfg),
    Moe(MoeCfg),
    Lmhead(LmheadCfg),
}

/// A single layer in execution order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// Position in `Graph::layers`. Equal to the index — stored explicitly so
    /// downstream code (cost model, DP) doesn't need a separate enumerate().
    pub idx: usize,
    /// Decoder-block index (0-based) for the transformer body; `None` for
    /// pre/post layers like the embedding, the final norm, and lm_head. The
    /// DP in `skein_extract` keys per-layer dtype off this value.
    pub block_idx: Option<usize>,
    pub kind: LayerKind,
    pub params: Vec<Param>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

/// Top-level model metadata. Derived from the HF `config.json` during import.
/// Floats (`rope_theta`, `rms_norm_eps`) are sourced from config and do not
/// participate in equality-sensitive Plan comparisons.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMeta {
    pub architecture: String,
    pub num_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub max_position: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// `Some(num_experts)` iff this is an MoE model. `top_k` is the
    /// per-token routing count.
    pub num_experts: Option<usize>,
    pub top_k: Option<usize>,
    pub intermediate: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub sliding_window: Option<usize>,
    pub tied_embeddings: bool,
}

/// The complete typed IR for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub meta: ModelMeta,
    pub layers: Vec<Layer>,
    /// Tensor catalog. `BTreeMap` so iteration order is deterministic — which
    /// matters for `Plan::content_hash`.
    pub tensors: BTreeMap<TensorId, Tensor>,
}

impl Graph {
    /// Count distinct decoder blocks (unique `block_idx` values). Each block
    /// contains several IR `Layer`s — pre-norm, attention, post-norm, MoE —
    /// that share a `block_idx`. The DP in `skein_extract` walks exactly
    /// this many transitions.
    pub fn num_decoder_blocks(&self) -> usize {
        let mut max_seen: Option<usize> = None;
        for l in &self.layers {
            if let Some(b) = l.block_idx {
                max_seen = Some(max_seen.map_or(b, |m| m.max(b)));
            }
        }
        max_seen.map_or(0, |m| m + 1)
    }

    /// Total parameter count (using each param's static numel). Symbolic
    /// dimensions in a `Param` shape are a hard error: weights must be
    /// statically sized.
    pub fn param_count(&self) -> u64 {
        let mut total: u64 = 0;
        for layer in &self.layers {
            for p in &layer.params {
                // Weights are always statically sized. If this ever returns
                // `None` it's an importer bug, not a runtime condition —
                // hence the panic with explanatory message rather than
                // returning Result.
                let n = p.shape.static_numel().expect(
                    "weight param has symbolic dim — this is an importer bug, \
                     weight shapes must be fully static",
                );
                total = total.saturating_add(n);
            }
        }
        total
    }
}

/// Helper for importers: hand out monotonic `TensorId`s without bookkeeping.
#[derive(Debug, Default)]
pub struct TensorIdGen {
    next: TensorId,
}

impl TensorIdGen {
    pub fn fresh(&mut self) -> TensorId {
        let id = self.next;
        self.next += 1;
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Dim;

    fn mk_param(name: &str, dims: Vec<Dim>) -> Param {
        Param {
            name: name.to_string(),
            shape: Shape(dims),
            dtype: Dtype::Bf16,
        }
    }

    #[test]
    fn param_count_sums_static_numel() {
        let layer = Layer {
            idx: 0,
            block_idx: Some(0),
            kind: LayerKind::Mlp(MlpCfg {
                hidden: 4,
                intermediate: 8,
                activation: Activation::Silu,
            }),
            params: vec![
                mk_param("a", vec![Dim::Fixed(4), Dim::Fixed(8)]),
                mk_param("b", vec![Dim::Fixed(8), Dim::Fixed(4)]),
            ],
            inputs: vec![],
            outputs: vec![],
        };
        let g = Graph {
            meta: ModelMeta {
                architecture: "test".into(),
                num_layers: 1,
                hidden: 4,
                vocab: 32,
                max_position: 16,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                num_experts: None,
                top_k: None,
                intermediate: 8,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-5,
                sliding_window: None,
                tied_embeddings: false,
            },
            layers: vec![layer],
            tensors: BTreeMap::new(),
        };
        assert_eq!(g.param_count(), 4 * 8 + 8 * 4);
        // One layer with block_idx=Some(0) → one distinct decoder block.
        assert_eq!(g.num_decoder_blocks(), 1);
    }

    #[test]
    fn tensor_id_gen_monotonic() {
        let mut g = TensorIdGen::default();
        assert_eq!(g.fresh(), 0);
        assert_eq!(g.fresh(), 1);
        assert_eq!(g.fresh(), 2);
    }
}
