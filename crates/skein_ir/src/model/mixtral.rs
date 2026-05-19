//! Mixtral importer.
//!
//! Produces an IR `Graph` for `MixtralForCausalLM` (Mixtral 8x7B, 8x22B, etc.)
//! from the HF `config.json`. Parameter names match the HF safetensors keys
//! exactly — `skein_emit` will use these names to slice safetensors files at
//! load time.
//!
//! Layer order (one IR `Layer` per logical op):
//!
//! ```text
//!   0:       Embedding              (block_idx = None)
//!   1..4*N:  for each block i in 0..num_hidden_layers:
//!              4i+1: RmsNorm    "input_layernorm"          block_idx = Some(i)
//!              4i+2: Attention  "self_attn"                block_idx = Some(i)
//!              4i+3: RmsNorm    "post_attention_layernorm" block_idx = Some(i)
//!              4i+4: Moe        "block_sparse_moe"         block_idx = Some(i)
//!   4N+1:    RmsNorm "model.norm" (final)                  block_idx = None
//!   4N+2:    Lmhead                                        block_idx = None
//! ```
//!
//! Activation tensors (hidden states between layers + final logits) are
//! catalogued in `Graph::tensors` with the bf16 *reference* dtype. The Plan
//! later assigns a per-block activation dtype which may override this on a
//! per-block basis; the bf16 reference is what `skein_parity` compares
//! against.

use serde::Deserialize;
use std::collections::BTreeMap;

use crate::error::ImportError;
use crate::ir::{
    Activation, AttentionCfg, AttentionPosition, EmbeddingCfg, Graph, Layer, LayerKind, LmheadCfg,
    ModelMeta, MoeCfg, NormCfg, Param, Tensor, TensorIdGen,
};
use crate::types::{Dim, Dtype, Shape};

/// The raw HF `config.json` fields the Mixtral importer reads. Anything not
/// listed here is intentionally ignored — Skein consults only fields whose
/// values change the IR.
#[derive(Debug, Deserialize)]
struct MixtralConfig {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    num_local_experts: usize,
    num_experts_per_tok: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    vocab_size: usize,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    tie_word_embeddings: bool,
    /// HF defaults `head_dim` to `hidden_size / num_attention_heads` when
    /// absent. We replicate that here.
    #[serde(default)]
    head_dim: Option<usize>,
    /// HF defaults `hidden_act` to `silu` for Mixtral. Other values would
    /// require a different IR `Activation` — we reject anything unrecognized
    /// rather than silently downgrading.
    #[serde(default)]
    hidden_act: Option<String>,
}

pub(crate) fn build_from_str(s: &str) -> Result<Graph, ImportError> {
    let cfg: MixtralConfig = serde_json::from_str(s)?;
    build(cfg)
}

fn build(cfg: MixtralConfig) -> Result<Graph, ImportError> {
    // --- Validation: every constraint that would silently produce an
    // inconsistent IR. No fallbacks — fail loudly.
    let head_dim = match cfg.head_dim {
        Some(d) => d,
        None => {
            if cfg.hidden_size % cfg.num_attention_heads != 0 {
                return Err(ImportError::InvalidConfig {
                    field: "hidden_size / num_attention_heads",
                    value: format!("{} / {}", cfg.hidden_size, cfg.num_attention_heads),
                    constraint: "hidden_size must be divisible by num_attention_heads when head_dim is absent",
                });
            }
            cfg.hidden_size / cfg.num_attention_heads
        }
    };
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
        return Err(ImportError::InvalidConfig {
            field: "num_attention_heads / num_key_value_heads",
            value: format!("{} / {}", cfg.num_attention_heads, cfg.num_key_value_heads),
            constraint: "num_attention_heads must be divisible by num_key_value_heads (GQA)",
        });
    }
    if cfg.num_experts_per_tok > cfg.num_local_experts {
        return Err(ImportError::InvalidConfig {
            field: "num_experts_per_tok",
            value: cfg.num_experts_per_tok.to_string(),
            constraint: "top-k must not exceed num_local_experts",
        });
    }
    if let Some(act) = cfg.hidden_act.as_deref() {
        if act != "silu" {
            return Err(ImportError::InvalidConfig {
                field: "hidden_act",
                value: act.into(),
                constraint: "Mixtral importer supports hidden_act = silu only",
            });
        }
    }
    let arch = cfg
        .architectures
        .first()
        .cloned()
        .ok_or(ImportError::NoArchitecture)?;

    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let intermediate = cfg.intermediate_size;
    let num_heads = cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;
    let num_experts = cfg.num_local_experts;
    let top_k = cfg.num_experts_per_tok;
    let num_blocks = cfg.num_hidden_layers;

    let meta = ModelMeta {
        architecture: arch,
        num_layers: num_blocks,
        hidden,
        vocab,
        max_position: cfg.max_position_embeddings,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        num_experts: Some(num_experts),
        top_k: Some(top_k),
        intermediate,
        rope_theta: cfg.rope_theta,
        rms_norm_eps: cfg.rms_norm_eps,
        sliding_window: cfg.sliding_window,
        tied_embeddings: cfg.tie_word_embeddings,
    };

    // --- Build tensors + layers ---
    let mut tids = TensorIdGen::default();
    let mut tensors: BTreeMap<u32, Tensor> = BTreeMap::new();

    // Hidden-state shape between layers and final logit shape, in *reference*
    // (bf16) dtype. The Plan may override per-block activation dtype.
    let hidden_shape = Shape(vec![Dim::Batch, Dim::Seq, Dim::Fixed(hidden)]);
    let logits_shape = Shape(vec![Dim::Batch, Dim::Seq, Dim::Fixed(vocab)]);

    let fresh_hidden = |tids: &mut TensorIdGen, tensors: &mut BTreeMap<u32, Tensor>| -> u32 {
        let id = tids.fresh();
        tensors.insert(
            id,
            Tensor {
                id,
                shape: hidden_shape.clone(),
                dtype: Dtype::Bf16,
            },
        );
        id
    };

    let mut layers: Vec<Layer> = Vec::with_capacity(2 + 4 * num_blocks + 2);
    let mut next_idx = 0_usize;
    let mut alloc_idx = || {
        let i = next_idx;
        next_idx += 1;
        i
    };

    // ---- Embedding ----
    let post_embed = fresh_hidden(&mut tids, &mut tensors);
    layers.push(Layer {
        idx: alloc_idx(),
        block_idx: None,
        kind: LayerKind::Embedding(EmbeddingCfg { vocab, hidden }),
        params: vec![Param {
            name: "model.embed_tokens.weight".into(),
            shape: Shape(vec![Dim::Fixed(vocab), Dim::Fixed(hidden)]),
            dtype: Dtype::Bf16,
        }],
        inputs: vec![],
        outputs: vec![post_embed],
    });

    // ---- Decoder blocks ----
    let mut residual = post_embed;
    for block in 0..num_blocks {
        // input_layernorm
        let post_in_norm = fresh_hidden(&mut tids, &mut tensors);
        layers.push(Layer {
            idx: alloc_idx(),
            block_idx: Some(block),
            kind: LayerKind::RmsNorm(NormCfg {
                eps: cfg.rms_norm_eps,
                hidden,
            }),
            params: vec![Param {
                name: format!("model.layers.{block}.input_layernorm.weight"),
                shape: Shape(vec![Dim::Fixed(hidden)]),
                dtype: Dtype::Bf16,
            }],
            inputs: vec![residual],
            outputs: vec![post_in_norm],
        });

        // self_attn (q,k,v,o)
        let post_attn = fresh_hidden(&mut tids, &mut tensors);
        let q_out = num_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;
        layers.push(Layer {
            idx: alloc_idx(),
            block_idx: Some(block),
            kind: LayerKind::Attention(AttentionCfg {
                num_heads,
                num_kv_heads,
                head_dim,
                position: AttentionPosition::Rope {
                    theta: cfg.rope_theta,
                },
                sliding_window: cfg.sliding_window,
            }),
            params: vec![
                Param {
                    name: format!("model.layers.{block}.self_attn.q_proj.weight"),
                    shape: Shape(vec![Dim::Fixed(q_out), Dim::Fixed(hidden)]),
                    dtype: Dtype::Bf16,
                },
                Param {
                    name: format!("model.layers.{block}.self_attn.k_proj.weight"),
                    shape: Shape(vec![Dim::Fixed(kv_out), Dim::Fixed(hidden)]),
                    dtype: Dtype::Bf16,
                },
                Param {
                    name: format!("model.layers.{block}.self_attn.v_proj.weight"),
                    shape: Shape(vec![Dim::Fixed(kv_out), Dim::Fixed(hidden)]),
                    dtype: Dtype::Bf16,
                },
                Param {
                    name: format!("model.layers.{block}.self_attn.o_proj.weight"),
                    shape: Shape(vec![Dim::Fixed(hidden), Dim::Fixed(q_out)]),
                    dtype: Dtype::Bf16,
                },
            ],
            inputs: vec![post_in_norm],
            outputs: vec![post_attn],
        });

        // post_attention_layernorm
        let post_post_norm = fresh_hidden(&mut tids, &mut tensors);
        layers.push(Layer {
            idx: alloc_idx(),
            block_idx: Some(block),
            kind: LayerKind::RmsNorm(NormCfg {
                eps: cfg.rms_norm_eps,
                hidden,
            }),
            params: vec![Param {
                name: format!("model.layers.{block}.post_attention_layernorm.weight"),
                shape: Shape(vec![Dim::Fixed(hidden)]),
                dtype: Dtype::Bf16,
            }],
            inputs: vec![post_attn],
            outputs: vec![post_post_norm],
        });

        // block_sparse_moe: gate + per-expert (w1, w2, w3)
        let post_moe = fresh_hidden(&mut tids, &mut tensors);
        let mut moe_params: Vec<Param> = Vec::with_capacity(1 + 3 * num_experts);
        moe_params.push(Param {
            name: format!("model.layers.{block}.block_sparse_moe.gate.weight"),
            shape: Shape(vec![Dim::Fixed(num_experts), Dim::Fixed(hidden)]),
            dtype: Dtype::Bf16,
        });
        for e in 0..num_experts {
            moe_params.push(Param {
                name: format!("model.layers.{block}.block_sparse_moe.experts.{e}.w1.weight"),
                shape: Shape(vec![Dim::Fixed(intermediate), Dim::Fixed(hidden)]),
                dtype: Dtype::Bf16,
            });
            moe_params.push(Param {
                name: format!("model.layers.{block}.block_sparse_moe.experts.{e}.w2.weight"),
                shape: Shape(vec![Dim::Fixed(hidden), Dim::Fixed(intermediate)]),
                dtype: Dtype::Bf16,
            });
            moe_params.push(Param {
                name: format!("model.layers.{block}.block_sparse_moe.experts.{e}.w3.weight"),
                shape: Shape(vec![Dim::Fixed(intermediate), Dim::Fixed(hidden)]),
                dtype: Dtype::Bf16,
            });
        }
        layers.push(Layer {
            idx: alloc_idx(),
            block_idx: Some(block),
            kind: LayerKind::Moe(MoeCfg {
                num_experts,
                top_k,
                hidden,
                intermediate,
                activation: Activation::Silu,
            }),
            params: moe_params,
            inputs: vec![post_post_norm],
            outputs: vec![post_moe],
        });

        residual = post_moe;
    }

    // ---- Final norm ----
    let post_final_norm = fresh_hidden(&mut tids, &mut tensors);
    layers.push(Layer {
        idx: alloc_idx(),
        block_idx: None,
        kind: LayerKind::RmsNorm(NormCfg {
            eps: cfg.rms_norm_eps,
            hidden,
        }),
        params: vec![Param {
            name: "model.norm.weight".into(),
            shape: Shape(vec![Dim::Fixed(hidden)]),
            dtype: Dtype::Bf16,
        }],
        inputs: vec![residual],
        outputs: vec![post_final_norm],
    });

    // ---- LM head ----
    let logits_id = tids.fresh();
    tensors.insert(
        logits_id,
        Tensor {
            id: logits_id,
            shape: logits_shape,
            dtype: Dtype::Bf16,
        },
    );
    let lmhead_params: Vec<Param> = if cfg.tie_word_embeddings {
        // Tied: reuses model.embed_tokens.weight at runtime; no separate
        // safetensors slice. We deliberately do not duplicate the param.
        vec![]
    } else {
        vec![Param {
            name: "lm_head.weight".into(),
            shape: Shape(vec![Dim::Fixed(vocab), Dim::Fixed(hidden)]),
            dtype: Dtype::Bf16,
        }]
    };
    layers.push(Layer {
        idx: alloc_idx(),
        block_idx: None,
        kind: LayerKind::Lmhead(LmheadCfg {
            vocab,
            hidden,
            tied: cfg.tie_word_embeddings,
        }),
        params: lmhead_params,
        inputs: vec![post_final_norm],
        outputs: vec![logits_id],
    });

    Ok(Graph {
        meta,
        layers,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::LayerKind;

    // Real Mixtral 8x7B HF config values. Sourced from
    //   https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
    // Kept inline so the test runs with no network access.
    const MIXTRAL_8X7B_CONFIG: &str = r#"{
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32000,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "hidden_act": "silu"
    }"#;

    #[test]
    fn imports_mixtral_8x7b_shape() {
        let g = build_from_str(MIXTRAL_8X7B_CONFIG).unwrap();
        assert_eq!(g.meta.num_layers, 32);
        assert_eq!(g.meta.hidden, 4096);
        assert_eq!(g.meta.head_dim, 128);
        assert_eq!(g.meta.num_attention_heads, 32);
        assert_eq!(g.meta.num_kv_heads, 8);
        assert_eq!(g.meta.num_experts, Some(8));
        assert_eq!(g.meta.top_k, Some(2));
        assert_eq!(g.meta.intermediate, 14336);
        assert_eq!(g.meta.vocab, 32000);
        assert!(!g.meta.tied_embeddings);

        // Layer count: 1 (embed) + 4 * 32 (blocks) + 1 (final norm) + 1 (lm_head)
        assert_eq!(g.layers.len(), 1 + 4 * 32 + 1 + 1);
        assert_eq!(g.num_decoder_blocks(), 32);

        // First layer is Embedding.
        match &g.layers[0].kind {
            LayerKind::Embedding(e) => {
                assert_eq!(e.vocab, 32000);
                assert_eq!(e.hidden, 4096);
            }
            other => panic!("expected Embedding, got {other:?}"),
        }
        // Last layer is Lmhead.
        match &g.layers.last().unwrap().kind {
            LayerKind::Lmhead(l) => {
                assert_eq!(l.vocab, 32000);
                assert!(!l.tied);
            }
            other => panic!("expected Lmhead, got {other:?}"),
        }
        // Pick block 5 and assert its 4-layer structure.
        let block5: Vec<&Layer> = g.layers.iter().filter(|l| l.block_idx == Some(5)).collect();
        assert_eq!(block5.len(), 4);
        assert!(matches!(block5[0].kind, LayerKind::RmsNorm(_)));
        assert!(matches!(block5[1].kind, LayerKind::Attention(_)));
        assert!(matches!(block5[2].kind, LayerKind::RmsNorm(_)));
        assert!(matches!(block5[3].kind, LayerKind::Moe(_)));

        // MoE block 5 has 1 (gate) + 3*8 (experts) = 25 params.
        assert_eq!(block5[3].params.len(), 25);
    }

    #[test]
    fn param_count_matches_published_mixtral_8x7b() {
        // Mixtral 8x7B has ~46.7 B total parameters. We compute the exact
        // sum of static numel and assert it falls in a tight bracket — exact
        // values:
        //   embed:    32000 * 4096                            =     131_072_000
        //   per block:
        //     2 * rmsnorm weights = 2*4096                    =           8_192
        //     attn q,k,v,o: 4096*4096 + 1024*4096 + 1024*4096 + 4096*4096
        //                 = 16_777_216 + 4_194_304 + 4_194_304 + 16_777_216
        //                 = 41_943_040
        //     moe gate:    8 * 4096                           =          32_768
        //     moe experts: 8 * (3 * 14336 * 4096)             =   1_409_286_144
        //                  total block                        =   1_451_270_144
        //   per-block * 32                                    =  46_440_644_608
        //   final norm:  4096                                 =           4_096
        //   lm_head:     32000 * 4096                         =     131_072_000
        //   GRAND TOTAL                                       =  46_702_792_704
        let g = build_from_str(MIXTRAL_8X7B_CONFIG).unwrap();
        assert_eq!(g.param_count(), 46_702_792_704);
    }

    #[test]
    fn rejects_unsupported_activation() {
        let cfg = MIXTRAL_8X7B_CONFIG.replace(r#""hidden_act": "silu""#, r#""hidden_act": "gelu""#);
        let err = build_from_str(&cfg).unwrap_err();
        match err {
            ImportError::InvalidConfig { field, .. } => assert_eq!(field, "hidden_act"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_gqa_indivisibility() {
        // num_attention_heads = 32, num_key_value_heads = 7 → invalid.
        let cfg = MIXTRAL_8X7B_CONFIG
            .replace(r#""num_key_value_heads": 8"#, r#""num_key_value_heads": 7"#);
        let err = build_from_str(&cfg).unwrap_err();
        assert!(matches!(err, ImportError::InvalidConfig { .. }));
    }

    #[test]
    fn honours_tied_embeddings() {
        let cfg = MIXTRAL_8X7B_CONFIG.replace(
            r#""tie_word_embeddings": false"#,
            r#""tie_word_embeddings": true"#,
        );
        let g = build_from_str(&cfg).unwrap();
        let lmhead = g.layers.last().unwrap();
        assert!(matches!(
            lmhead.kind,
            LayerKind::Lmhead(LmheadCfg { tied: true, .. })
        ));
        assert!(
            lmhead.params.is_empty(),
            "tied lm_head must not duplicate the embedding param"
        );
    }
}
