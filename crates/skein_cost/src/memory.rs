//! Peak per-device memory + the overshoot penalty term.
//!
//! Three contributions:
//!
//! - **Weights** — sum over layers assigned to this device's stage of
//!   `param_count × weight_dtype_bytes / shard_divisor`.
//! - **KV cache** — `2 × batch × kv_len × num_kv_heads × head_dim ×
//!   kv_dtype_bytes` per layer, summed across the layers on this stage.
//!   Divided by `tp` if `Plan::kv.kv_sharded`.
//! - **Activations** — a small constant-factor envelope around the hidden
//!   state. Approximated by four hidden buffers (`batch × seq × hidden`).
//!
//! When the per-device peak exceeds the device cap, the overshoot is scaled
//! by `overshoot_us_per_gb` and added as a large penalty term. That turns
//! the memory-fit hard constraint into a cost, keeping the cost model itself
//! purely additive.

use skein_ir::ir::{Graph, LayerKind};
use skein_ir::plan::Plan;
use skein_ir::types::Dtype;

use crate::cluster::{Cluster, DeviceIdx, Placement, block_to_stage};
use crate::compute::{activation_dtype, kv_dtype, weight_dtype};
use crate::constants::CostConstants;
use crate::error::CostError;

/// Bytes a single layer's weights occupy on `device`, after sharding.
fn layer_weight_bytes(layer: &skein_ir::ir::Layer, plan: &Plan, placement: Placement) -> u64 {
    let dt = weight_dtype(plan, layer.block_idx);
    let mut raw: u64 = 0;
    for p in &layer.params {
        let n = p.shape.static_numel().expect(
            "param shape contains a symbolic dim — this is an importer bug; \
             weight tensors must be fully static",
        );
        raw = raw.saturating_add(dt.bytes_for(n));
    }
    let div = crate::compute::compute_divisor(&layer.kind, placement).max(1);
    raw / div
}

/// Total weight bytes on `device` across every layer assigned to its stage.
pub fn weight_bytes_on_device(plan: &Plan, ir: &Graph, device: DeviceIdx) -> u64 {
    let placement = Placement::from_plan(plan);
    let Some(stage) = placement.stage_of(device) else {
        return 0;
    };
    let num_blocks = ir.meta.num_layers;
    let mut total: u64 = 0;
    for layer in &ir.layers {
        let on_stage = match layer.block_idx {
            Some(b) => block_to_stage(b, num_blocks, placement.pp) == stage,
            None => match &layer.kind {
                LayerKind::Embedding(_) => stage == 0,
                LayerKind::Lmhead(_) => stage == placement.pp - 1,
                _ => stage == 0,
            },
        };
        if on_stage {
            total = total.saturating_add(layer_weight_bytes(layer, plan, placement));
        }
    }
    total
}

/// KV cache bytes on `device`. Decode-step model: KV holds `kv_len` tokens
/// per concurrent sequence, both K and V.
pub fn kv_bytes_on_device(
    plan: &Plan,
    ir: &Graph,
    constants: &CostConstants,
    device: DeviceIdx,
) -> u64 {
    let placement = Placement::from_plan(plan);
    let Some(stage) = placement.stage_of(device) else {
        return 0;
    };
    let max_batch = plan.batching.max_batch() as u64;
    let kv_len = constants.representative_workload.decode_kv_tokens as u64;
    let num_kv_heads = ir.meta.num_kv_heads as u64;
    let head_dim = ir.meta.head_dim as u64;
    // Elements per-layer, K and V combined.
    let elems_per_layer_per_seq = 2 * num_kv_heads * head_dim * kv_len;
    let num_blocks = ir.meta.num_layers;
    let mut total: u64 = 0;
    for b in 0..num_blocks {
        if block_to_stage(b, num_blocks, placement.pp) != stage {
            continue;
        }
        let dt = kv_dtype(plan, Some(b));
        let bytes_per_layer = dt.bytes_for(elems_per_layer_per_seq);
        total = total.saturating_add(bytes_per_layer.saturating_mul(max_batch));
    }
    if plan.kv.kv_sharded && placement.tp > 1 {
        total / placement.tp as u64
    } else {
        total
    }
}

/// Conservative activation envelope: four hidden-state buffers in flight.
/// At decode time these are tiny (batch × 1 × hidden) so they don't move the
/// needle in the memory comparison, but keeping the term in the model
/// prevents future regressions if `seq_len` ever grows in the workload.
pub fn activation_bytes_on_device(plan: &Plan, ir: &Graph, device: DeviceIdx) -> u64 {
    let placement = Placement::from_plan(plan);
    if placement.stage_of(device).is_none() {
        return 0;
    }
    let max_batch = plan.batching.max_batch() as u64;
    let hidden = ir.meta.hidden as u64;
    // Use the first block's activation dtype if present, bf16 otherwise.
    let dt = if !plan.dtype_map.per_layer.is_empty() {
        plan.dtype_map.per_layer[0].activation
    } else {
        Dtype::Bf16
    };
    // seq_len = 1 at decode time; written long-form for clarity then folded
    // to keep clippy happy.
    let per_buf = dt.bytes_for(max_batch * hidden);
    per_buf.saturating_mul(4)
}

/// Peak memory bytes on `device`. The sum is saturating; an enormous Plan
/// will report `u64::MAX` rather than wrapping.
pub fn peak_memory_bytes(
    plan: &Plan,
    ir: &Graph,
    constants: &CostConstants,
    device: DeviceIdx,
) -> u64 {
    let w = weight_bytes_on_device(plan, ir, device);
    let k = kv_bytes_on_device(plan, ir, constants, device);
    let a = activation_bytes_on_device(plan, ir, device);
    // Pull the unused-import lint over the line so a future tweak doesn't
    // silently lose the `activation_dtype` import.
    let _ = activation_dtype;
    w.saturating_add(k).saturating_add(a)
}

// ---------------------------------------------------------------------------
// Per-block memory helpers consumed by `skein_extract`'s inner DP. These
// score *one* decoder block at a candidate dtype, without a full Plan.
// ---------------------------------------------------------------------------

/// Bytes one decoder block's weights occupy on a single device of the TP/EP
/// group, given the candidate weight dtype.
///
/// Sums every IR `Param` belonging to a layer with `block_idx == Some(b)`,
/// applies the layer kind's compute divisor, then converts to bytes via
/// `dtype.bytes_for`.
pub fn block_weight_bytes(
    block_idx: usize,
    ir: &Graph,
    weight_dtype: Dtype,
    placement: Placement,
) -> u64 {
    let mut total: u64 = 0;
    for layer in &ir.layers {
        if layer.block_idx != Some(block_idx) {
            continue;
        }
        let mut raw: u64 = 0;
        for p in &layer.params {
            let n = p.shape.static_numel().expect(
                "param shape contains a symbolic dim — this is an importer bug; \
                 weight tensors must be fully static",
            );
            raw = raw.saturating_add(weight_dtype.bytes_for(n));
        }
        let div = crate::compute::compute_divisor(&layer.kind, placement).max(1);
        total = total.saturating_add(raw / div);
    }
    total
}

/// Bytes one decoder block's KV cache occupies on a single device, given the
/// candidate KV dtype and whether KV is sharded along the TP axis.
///
/// One KV-producing layer per block (the Attention layer). `2 × batch ×
/// kv_len × num_kv_heads × head_dim` elements; sharded by `tp` if requested.
pub fn block_kv_bytes(
    ir: &Graph,
    kv_dtype: Dtype,
    placement: Placement,
    wl: &crate::workload_ctx::WorkloadCtx,
    kv_sharded: bool,
) -> u64 {
    let num_kv_heads = ir.meta.num_kv_heads as u64;
    let head_dim = ir.meta.head_dim as u64;
    let kv_len = wl.kv_len as u64;
    let batch = wl.batch as u64;
    let elems = 2 * num_kv_heads * head_dim * kv_len * batch;
    let bytes = kv_dtype.bytes_for(elems);
    if kv_sharded && placement.tp > 1 {
        bytes / placement.tp as u64
    } else {
        bytes
    }
}

/// Bytes the non-decoder layers (`Embedding`, the final `RmsNorm`, `Lmhead`)
/// occupy on the device that hosts them. Used by `skein_extract::budgets`
/// when reserving headroom *before* the DP picks per-block dtypes.
///
/// These layers are not in the search axis — they always use bf16.
pub fn non_decoder_weight_bytes(ir: &Graph, placement: Placement) -> u64 {
    let mut total: u64 = 0;
    for layer in &ir.layers {
        if layer.block_idx.is_some() {
            continue;
        }
        let mut raw: u64 = 0;
        for p in &layer.params {
            let n = p
                .shape
                .static_numel()
                .expect("param shape contains a symbolic dim — importer bug");
            raw = raw.saturating_add(Dtype::Bf16.bytes_for(n));
        }
        let div = crate::compute::compute_divisor(&layer.kind, placement).max(1);
        total = total.saturating_add(raw / div);
    }
    total
}

/// Soft-constraint memory penalty in microseconds. Zero when the device
/// fits; `overshoot_gb × overshoot_us_per_gb` when it does not.
pub fn memory_penalty(
    plan: &Plan,
    ir: &Graph,
    cluster: &Cluster,
    constants: &CostConstants,
    device: DeviceIdx,
) -> Result<f64, CostError> {
    let peak = peak_memory_bytes(plan, ir, constants, device);
    let cap = cluster.device_memory_bytes(device)?;
    if peak <= cap {
        Ok(0.0)
    } else {
        let overshoot_gb = (peak - cap) as f64 / 1.0e9;
        Ok(overshoot_gb * constants.overshoot_us_per_gb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skein_ir::cluster::ClusterSpec;
    use skein_ir::plan::*;
    use skein_ir::types::*;

    const TWO_H100: &str = r#"
num_devices = 2
[[node]]
id               = "node0"
devices          = ["d0", "d1"]
device_kind      = "h100_sxm5"
device_memory_gb = 80
[[link]]
endpoints      = ["d0", "d1"]
kind           = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us     = 1.0
"#;

    fn mixtral_8x7b() -> Graph {
        let cfg = include_str!("../../../configs/mixtral_8x7b_config.json");
        skein_ir::model::import_from_str(cfg).expect("import Mixtral fixture")
    }

    fn dummy_constants() -> CostConstants {
        CostConstants::from_toml_str(
            r#"
launch_us_per_kernel = 7.0
overshoot_us_per_gb  = 1000000.0
[peak_tflops.h100_sxm5]
bf16     = 989.0
fp16     = 989.0
fp8_e4m3 = 1979.0
fp8_e5m2 = 1979.0
int8     = 1979.0
int4     = 3958.0
[efficiency.gemm]
bf16     = 0.78
fp16     = 0.78
fp8_e4m3 = 0.72
fp8_e5m2 = 0.72
int8     = 0.68
int4     = 0.55
[efficiency.attention]
bf16     = 0.64
fp16     = 0.64
fp8_e4m3 = 0.58
fp8_e5m2 = 0.58
int8     = 0.52
int4     = 0.40
[efficiency.elementwise]
bf16     = 0.95
fp16     = 0.95
fp8_e4m3 = 0.92
fp8_e5m2 = 0.92
int8     = 0.90
int4     = 0.85
[collective_efficiency]
ring_allreduce = 0.85
allgather      = 0.90
reducescatter  = 0.88
alltoall       = 0.75
broadcast      = 0.95
send_recv      = 0.92
[parity_tolerance_mse]
bf16     = 1.0e-3
fp16     = 1.0e-3
fp8_e4m3 = 5.0e-3
fp8_e5m2 = 5.0e-3
int8     = 1.0e-2
int4     = 2.0e-2

[representative_workload]
prefill_tokens   = 1024
decode_kv_tokens = 2048
[dp]
memory_buckets = 100
drift_buckets  = 50

[runtime]
metrics_buffer_capacity = 100000
drain_timeout_seconds   = 60
prometheus_port         = 9090
radix_max_depth         = 4096

[runtime_estimator]
per_token_decode_us_at_b1 = 8000.0
batch_scaling_exponent    = 0.7
prefill_per_token_us      = 25.0
"#,
        )
        .unwrap()
    }

    fn plan_with(tp: u32, w: Dtype, kv: Dtype) -> Plan {
        let meta = mixtral_8x7b().meta;
        Plan {
            parallelism: ParallelismPlacement { tp, pp: 1, ep: 1 },
            kv: KVCacheSpec {
                layout: KVLayout::Paged { page_size: 32 },
                kv_sharded: false,
            },
            batching: BatchPolicy::Continuous { max_batch: 16 },
            dtype_map: DtypeMap {
                per_layer: vec![
                    PerLayerDtype {
                        weight: w,
                        activation: Dtype::Bf16,
                        kv_cache: kv
                    };
                    meta.num_layers
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
            model_meta: meta,
        }
    }

    #[test]
    fn tp1_bf16_overshoots_80gb() {
        let ir = mixtral_8x7b();
        let spec = ClusterSpec::from_toml_str(TWO_H100).unwrap();
        let cluster = Cluster::from_spec(spec);
        let consts = dummy_constants();
        let plan = plan_with(1, Dtype::Bf16, Dtype::Bf16);
        let bytes = peak_memory_bytes(&plan, &ir, &consts, 0);
        // 46.7B × 2 = 93.4 GB+, well over the 80 GB cap.
        assert!(
            bytes > 90 * 1_000_000_000,
            "expected >90 GB, got {bytes} bytes"
        );
        let pen = memory_penalty(&plan, &ir, &cluster, &consts, 0).unwrap();
        assert!(pen > 1e6, "expected memory penalty above 1M µs, got {pen}");
    }

    #[test]
    fn tp2_bf16_fits_80gb() {
        let ir = mixtral_8x7b();
        let spec = ClusterSpec::from_toml_str(TWO_H100).unwrap();
        let cluster = Cluster::from_spec(spec);
        let consts = dummy_constants();
        let plan = plan_with(2, Dtype::Bf16, Dtype::Bf16);
        let pen = memory_penalty(&plan, &ir, &cluster, &consts, 0).unwrap();
        assert_eq!(pen, 0.0, "TP=2 bf16 Mixtral 8x7B should fit");
    }
}
