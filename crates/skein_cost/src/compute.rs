//! Per-layer FLOP counting + `compute_time`.
//!
//! Decode-step model: `seq_len = 1`, `kv_len = decode_kv_tokens`. Attention
//! FLOPs scale with `seq_len × kv_len`; everything else with `seq_len`.
//!
//! TP > 1 divides compute time by `tp`: row/column-parallel projections split
//! the matmul work across the TP group. EP > 1 divides MoE expert compute by
//! `ep`. Bubble/launch/comm are handled separately.

use skein_ir::ir::{Graph, Layer, LayerKind, ModelMeta};
use skein_ir::plan::Plan;
use skein_ir::types::Dtype;

use crate::cluster::{Cluster, DeviceIdx, Placement};
use crate::constants::CostConstants;
use crate::error::CostError;
use crate::workload_ctx::WorkloadCtx;

/// Three op-kind categories the efficiency table is indexed by. Each `Layer`
/// maps to exactly one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    Gemm,
    Attention,
    Elementwise,
}

impl OpKind {
    pub fn from_layer(kind: &LayerKind) -> Self {
        match kind {
            LayerKind::Embedding(_) | LayerKind::RmsNorm(_) => OpKind::Elementwise,
            LayerKind::Attention(_) => OpKind::Attention,
            LayerKind::Mlp(_) | LayerKind::Moe(_) | LayerKind::Lmhead(_) => OpKind::Gemm,
        }
    }

    pub fn efficiency_key(self) -> &'static str {
        match self {
            OpKind::Gemm => "gemm",
            OpKind::Attention => "attention",
            OpKind::Elementwise => "elementwise",
        }
    }
}

/// Weight dtype for a given decoder block. Non-block layers (embedding,
/// final norm, lm_head) are not in the search axis and use the bf16
/// reference dtype.
pub fn weight_dtype(plan: &Plan, block_idx: Option<usize>) -> Dtype {
    match block_idx {
        Some(b) if b < plan.dtype_map.per_layer.len() => plan.dtype_map.per_layer[b].weight,
        _ => Dtype::Bf16,
    }
}

/// Activation dtype — same rule as `weight_dtype`.
pub fn activation_dtype(plan: &Plan, block_idx: Option<usize>) -> Dtype {
    match block_idx {
        Some(b) if b < plan.dtype_map.per_layer.len() => plan.dtype_map.per_layer[b].activation,
        _ => Dtype::Bf16,
    }
}

/// KV cache dtype — same rule.
pub fn kv_dtype(plan: &Plan, block_idx: Option<usize>) -> Dtype {
    match block_idx {
        Some(b) if b < plan.dtype_map.per_layer.len() => plan.dtype_map.per_layer[b].kv_cache,
        _ => Dtype::Bf16,
    }
}

/// FLOPs for one forward pass of `layer` at workload `wl`. Multiply-add is
/// counted as 2 FLOPs. The numbers are intentionally analytic — they match
/// the textbook arithmetic intensity for each op, not measured kernel work.
pub fn layer_flops(layer: &Layer, meta: &ModelMeta, wl: &WorkloadCtx) -> u64 {
    let b = wl.batch as u64;
    let s = wl.seq_len as u64;
    let k = wl.kv_len as u64;
    match &layer.kind {
        LayerKind::Embedding(e) => {
            // Token embedding is a gather, modelled as a hidden-state touch.
            b * s * e.hidden as u64
        }
        LayerKind::RmsNorm(n) => {
            // Mean, rsqrt, multiply, add — roughly 5 ops per element.
            5 * b * s * n.hidden as u64
        }
        LayerKind::Attention(a) => {
            let hidden = meta.hidden as u64;
            let q_out = (a.num_heads * a.head_dim) as u64;
            let kv_out = (a.num_kv_heads * a.head_dim) as u64;
            // Four linear projections: q, k, v, o.
            let qkvo = 2 * b * s * (q_out + kv_out + kv_out + q_out) * hidden;
            // Q · K^T over kv_len.
            let scores = 2 * b * (a.num_heads as u64) * s * k * (a.head_dim as u64);
            // Softmax (max + exp + normalize) — ~3 ops per (head, q, kv) entry.
            let softmax = 3 * b * (a.num_heads as u64) * s * k;
            // Attn · V.
            let att_v = 2 * b * (a.num_heads as u64) * s * k * (a.head_dim as u64);
            qkvo + scores + softmax + att_v
        }
        LayerKind::Mlp(m) => {
            let h = m.hidden as u64;
            let i = m.intermediate as u64;
            // SwiGLU: w1 + w3 + w2 = three matmuls. SiLU+mul is dominated by
            // the matmuls and folded into them at the constant level.
            6 * b * s * h * i
        }
        LayerKind::Moe(m) => {
            let h = m.hidden as u64;
            let i = m.intermediate as u64;
            let ne = m.num_experts as u64;
            let tk = m.top_k as u64;
            // Gate: tiny matmul, hidden -> num_experts.
            let gate = 2 * b * s * h * ne;
            // Each token visits `top_k` experts; each expert performs the same
            // SwiGLU FFN as Mlp. Total expert FLOPs = b·s·top_k · 6·h·i.
            let experts = 6 * b * s * tk * h * i;
            gate + experts
        }
        LayerKind::Lmhead(l) => 2 * b * s * (l.vocab as u64) * (l.hidden as u64),
    }
}

/// Sharding divisor that splits a layer's compute across the TP/EP group.
///
/// Per-layer rules:
/// - Attention / MLP / LMHead / Embedding: divide by `tp` (column- or
///   row-parallel projections).
/// - MoE: divide by `tp × ep` (experts split by EP, the inner FFN by TP if
///   present).
/// - RmsNorm: 1 (replicated; the work is per-token elementwise).
pub fn compute_divisor(kind: &LayerKind, placement: Placement) -> u64 {
    let tp = placement.tp as u64;
    let ep = placement.ep as u64;
    match kind {
        LayerKind::Attention(_)
        | LayerKind::Mlp(_)
        | LayerKind::Lmhead(_)
        | LayerKind::Embedding(_) => tp,
        LayerKind::Moe(_) => tp * ep,
        LayerKind::RmsNorm(_) => 1,
    }
}

/// Microseconds for `layer` on this device, given an explicit weight dtype
/// and parallelism placement. This is the primitive form — `compute_time`
/// derives `(weight_dtype, placement)` from a Plan and delegates here.
/// `skein_extract`'s inner DP uses this form to score one block at a
/// candidate dtype without building a Plan per try.
#[allow(clippy::too_many_arguments)]
pub fn compute_time_with_dtype(
    layer: &Layer,
    meta: &ModelMeta,
    weight_dtype: Dtype,
    placement: Placement,
    cluster: &Cluster,
    constants: &CostConstants,
    wl: &WorkloadCtx,
    device: DeviceIdx,
) -> Result<f64, CostError> {
    let flops = layer_flops(layer, meta, wl) as f64;
    let kind = cluster.device_kind(device)?;
    let peak_tflops = constants.peak_tflops_for(kind, weight_dtype)?;
    let op = OpKind::from_layer(&layer.kind);
    let eff = constants.efficiency_for(op, weight_dtype)?;
    let div = compute_divisor(&layer.kind, placement).max(1) as f64;
    // peak_TFLOPS × 1e12 = peak FLOPS. Time in seconds, ×1e6 → µs.
    Ok(flops / (peak_tflops * 1e12 * eff * div) * 1e6)
}

/// Microseconds for `layer` on this device, given the chosen dtype and
/// efficiency.
#[allow(clippy::too_many_arguments)]
pub fn compute_time(
    layer: &Layer,
    meta: &ModelMeta,
    plan: &Plan,
    cluster: &Cluster,
    constants: &CostConstants,
    wl: &WorkloadCtx,
    device: DeviceIdx,
) -> Result<f64, CostError> {
    let dtype = weight_dtype(plan, layer.block_idx);
    let placement = Placement::from_plan(plan);
    compute_time_with_dtype(
        layer, meta, dtype, placement, cluster, constants, wl, device,
    )
}

/// Microseconds for an entire decoder block (all IR layers with the given
/// `block_idx`) at a chosen weight dtype.
///
/// `skein_extract::dp::layer_dtype_dp` uses this to score one block's compute
/// contribution at each candidate weight dtype. Activation/KV dtypes don't
/// change compute time (the matmuls are pinned by the weight dtype path) so
/// the DP only needs to vary `weight_dtype` here; activation/KV are scored
/// by memory + drift, not compute.
#[allow(clippy::too_many_arguments)]
pub fn block_compute_time(
    block_idx: usize,
    ir: &Graph,
    weight_dtype: Dtype,
    placement: Placement,
    cluster: &Cluster,
    constants: &CostConstants,
    wl: &WorkloadCtx,
    device: DeviceIdx,
) -> Result<f64, CostError> {
    let mut total = 0.0_f64;
    for layer in &ir.layers {
        if layer.block_idx != Some(block_idx) {
            continue;
        }
        total += compute_time_with_dtype(
            layer,
            &ir.meta,
            weight_dtype,
            placement,
            cluster,
            constants,
            wl,
            device,
        )?;
    }
    Ok(total)
}

/// Number of CUDA kernels Skein expects per forward step of `layer`. Used by
/// the launch-overhead term.
pub fn kernels_per_layer(kind: &LayerKind) -> u32 {
    match kind {
        LayerKind::Embedding(_) | LayerKind::RmsNorm(_) | LayerKind::Lmhead(_) => 1,
        // q,k,v,o GEMMs + RoPE + masked softmax + attention matmul ≈ 7.
        LayerKind::Attention(_) => 7,
        // w1 fused with SiLU, w3, elementwise mul, w2 ≈ 3.
        LayerKind::Mlp(_) => 3,
        // gate + scatter + (w1+silu+w3+mul+w2 fused per top_k expert) + combine.
        LayerKind::Moe(m) => 3 + m.top_k as u32,
    }
}

/// Total kernels per forward step for the layers running on `device`. With
/// `pp > 1` only the layers assigned to this device's stage contribute.
pub fn kernels_per_step_on_device(ir: &Graph, plan: &Plan, device: DeviceIdx) -> u32 {
    let placement = Placement::from_plan(plan);
    let Some(stage) = placement.stage_of(device) else {
        return 0;
    };
    let num_blocks = ir.meta.num_layers;
    let mut total: u32 = 0;
    for layer in &ir.layers {
        let on_stage = match layer.block_idx {
            Some(b) => crate::cluster::block_to_stage(b, num_blocks, placement.pp) == stage,
            // Embedding / final-norm / lm_head: on stage 0 / last stage by
            // convention.
            None => match &layer.kind {
                LayerKind::Embedding(_) => stage == 0,
                LayerKind::Lmhead(_) => stage == placement.pp - 1,
                // Other non-block layers (e.g. an extra norm) are placed on
                // stage 0 by default.
                _ => stage == 0,
            },
        };
        if on_stage {
            total += kernels_per_layer(&layer.kind);
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use skein_ir::ir::*;
    use skein_ir::plan::*;
    use skein_ir::types::*;

    fn mk_attn_layer() -> Layer {
        Layer {
            idx: 0,
            block_idx: Some(0),
            kind: LayerKind::Attention(AttentionCfg {
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
                position: AttentionPosition::Rope { theta: 1_000_000.0 },
                sliding_window: None,
            }),
            params: vec![],
            inputs: vec![],
            outputs: vec![],
        }
    }

    fn mk_meta() -> ModelMeta {
        ModelMeta {
            architecture: "test".into(),
            num_layers: 32,
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

    #[test]
    fn attention_flops_scale_with_kv_len() {
        let layer = mk_attn_layer();
        let meta = mk_meta();
        let wl_short = WorkloadCtx {
            batch: 1,
            seq_len: 1,
            kv_len: 128,
        };
        let wl_long = WorkloadCtx {
            batch: 1,
            seq_len: 1,
            kv_len: 4096,
        };
        let short = layer_flops(&layer, &meta, &wl_short);
        let long = layer_flops(&layer, &meta, &wl_long);
        // QKVO projections (~84 MFLOPs at Mixtral shape × seq=1) dominate at
        // short kv_len; the scores / Attn·V terms grow linearly with kv_len
        // and start to dominate as the cache grows. Going from kv=128 to
        // kv=4096 should at least 1.5× the total — anything tighter would be
        // sensitive to numerator constants and not worth pinning here.
        assert!(long > short);
        assert!(long as f64 > short as f64 * 1.5);
    }

    #[test]
    fn op_kind_maps_layers() {
        let attn = mk_attn_layer();
        assert_eq!(OpKind::from_layer(&attn.kind), OpKind::Attention);

        let mlp = LayerKind::Mlp(MlpCfg {
            hidden: 4,
            intermediate: 8,
            activation: Activation::Silu,
        });
        assert_eq!(OpKind::from_layer(&mlp), OpKind::Gemm);

        let norm = LayerKind::RmsNorm(NormCfg {
            eps: 1e-5,
            hidden: 4,
        });
        assert_eq!(OpKind::from_layer(&norm), OpKind::Elementwise);
    }

    #[test]
    fn weight_dtype_defaults_to_bf16_for_non_block_layers() {
        let plan = Plan {
            parallelism: ParallelismPlacement {
                tp: 1,
                pp: 1,
                ep: 1,
            },
            kv: KVCacheSpec {
                layout: KVLayout::Contiguous,
                kv_sharded: false,
            },
            batching: BatchPolicy::Continuous { max_batch: 1 },
            dtype_map: DtypeMap::uniform(4, Dtype::Fp8E4m3),
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
            model_meta: mk_meta(),
        };
        assert_eq!(weight_dtype(&plan, None), Dtype::Bf16);
        assert_eq!(weight_dtype(&plan, Some(0)), Dtype::Fp8E4m3);
        assert_eq!(weight_dtype(&plan, Some(99)), Dtype::Bf16); // out of range
    }

    #[test]
    fn compute_divisor_per_kind() {
        let placement = Placement {
            tp: 4,
            pp: 1,
            ep: 2,
        };
        assert_eq!(
            compute_divisor(
                &LayerKind::Attention(AttentionCfg {
                    num_heads: 32,
                    num_kv_heads: 8,
                    head_dim: 128,
                    position: AttentionPosition::Rope { theta: 1.0 },
                    sliding_window: None
                }),
                placement
            ),
            4
        );
        assert_eq!(
            compute_divisor(
                &LayerKind::Moe(MoeCfg {
                    num_experts: 8,
                    top_k: 2,
                    hidden: 4096,
                    intermediate: 14336,
                    activation: Activation::Silu
                }),
                placement
            ),
            8
        );
        assert_eq!(
            compute_divisor(
                &LayerKind::RmsNorm(NormCfg {
                    eps: 1e-5,
                    hidden: 4096
                }),
                placement
            ),
            1
        );
    }
}
