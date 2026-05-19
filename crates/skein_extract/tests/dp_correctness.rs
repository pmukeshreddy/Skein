//! Tests 4–7 — inner DP correctness on the canonical Mixtral 2× H100 setup.

mod common;
use common::*;

use skein_cost::cluster::Placement;
use skein_extract::budgets::Budgets;
use skein_extract::candidate::GlobalConfig;
use skein_extract::dp::layer_dtype_dp;
use skein_extract::error::ExtractError;
use skein_ir::plan::ParallelismPlacement;
use skein_ir::types::{
    BatchPolicy, CudaGraphsConfig, Dtype, KVLayout, PrefixCacheConfig, RadixReusePolicy,
    SpecDecodeConfig,
};

fn mk_global() -> GlobalConfig {
    GlobalConfig {
        parallelism: ParallelismPlacement {
            tp: 2,
            pp: 1,
            ep: 1,
        },
        kv_layout: KVLayout::Paged { page_size: 32 },
        kv_shard: false,
        batch: BatchPolicy::Continuous { max_batch: 8 },
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
    }
}

// Test 4 — tight drift budget forces the DP to pick bf16 everywhere.
//
// Note: the spec originally said "abundant memory AND abundant drift" forces
// bf16. That's backwards — with abundant drift any combo is admissible, and
// the DP would pick the *fastest*-compute dtype (int4 weights, since H100
// int4 peak × eff > bf16's). The bf16 choice is forced by a *tight* drift
// budget that excludes everything but bf16.
#[test]
fn dp_picks_bf16_when_drift_tight() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let cost_model = load_cost_model();
    // Use the production drift table — bf16 is the only zero-drift dtype.
    let drift_table = load_drift_table();

    // Abundant memory (80 GB cap times two devices, plenty headroom) but a
    // drift budget below even one block's fp8_e4m3 contribution. Any
    // non-bf16 weight/activation/kv pick exceeds the budget at block 0.
    let budgets = Budgets {
        memory_bytes: 70 * 1_000_000_000,
        drift: 0.001, // strictly less than fp8_e4m3 weight drift (0.008)
    };
    let global = mk_global();
    let result =
        layer_dtype_dp(&ir, &cost_model, &cluster, &global, budgets, &drift_table).unwrap();

    for entry in &result.dtype_map.per_layer {
        assert_eq!(entry.weight, Dtype::Bf16);
        assert_eq!(entry.activation, Dtype::Bf16);
        assert_eq!(entry.kv_cache, Dtype::Bf16);
    }
    assert_eq!(result.dtype_map.per_layer.len(), ir.meta.num_layers);
}

// Test 5 — tight memory forces aggressive quantization.
#[test]
fn dp_picks_aggressive_quant_under_tight_memory() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let cost_model = load_cost_model();
    // Flat-zero drift so memory is the only binding axis.
    let drift_table = flat_drift_table();

    // ~50% of bf16's decoder-weight requirement on a tp=2 device. Mixtral
    // bf16 at tp=2 needs ~46 GB just for the decoder; this budget excludes
    // bf16 but admits fp8 / int8 / int4. With 100 memory buckets the
    // per-block ceil-bucketing adds ~30% overhead, so the budget number is
    // sized to leave headroom for that rounding.
    let budgets = Budgets {
        memory_bytes: 25 * 1_000_000_000,
        drift: 1.0,
    };
    let global = mk_global();
    let result =
        layer_dtype_dp(&ir, &cost_model, &cluster, &global, budgets, &drift_table).unwrap();

    let aggressive = result
        .dtype_map
        .per_layer
        .iter()
        .filter(|e| matches!(e.weight, Dtype::Int8 | Dtype::Int4))
        .count();
    let n = result.dtype_map.per_layer.len();
    assert!(
        aggressive as f64 / n as f64 >= 0.8,
        "expected ≥ 80% int8/int4 weights to fit; got {aggressive}/{n}"
    );
}

// Test 6 — pathologically small memory budget → DpInfeasible.
#[test]
fn dp_infeasible_when_no_combo_fits() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let cost_model = load_cost_model();
    let drift_table = flat_drift_table();

    let budgets = Budgets {
        memory_bytes: 1_000_000, // 1 MB — smaller than any single block's weights at any dtype.
        drift: 1.0,
    };
    let global = mk_global();
    let err =
        layer_dtype_dp(&ir, &cost_model, &cluster, &global, budgets, &drift_table).unwrap_err();
    assert!(matches!(err, ExtractError::DpInfeasible { .. }));
}

// Test 7 — the DP's reported compute time matches `cost_model.block_compute_time`
// summed externally over the chosen dtype map.
#[test]
fn dp_matches_cost_model() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let cost_model = load_cost_model();
    let drift_table = flat_drift_table();

    let budgets = Budgets {
        memory_bytes: 70 * 1_000_000_000,
        drift: 1.0,
    };
    let global = mk_global();
    let result =
        layer_dtype_dp(&ir, &cost_model, &cluster, &global, budgets, &drift_table).unwrap();

    let placement = Placement {
        tp: global.parallelism.tp,
        pp: global.parallelism.pp,
        ep: global.parallelism.ep,
    };
    let wl = skein_cost::WorkloadCtx {
        batch: global.batch.max_batch(),
        seq_len: 1,
        kv_len: cost_model
            .constants()
            .representative_workload
            .decode_kv_tokens,
    };

    let mut external_sum = 0.0_f64;
    for (b, entry) in result.dtype_map.per_layer.iter().enumerate() {
        external_sum += cost_model
            .block_compute_time(b, &ir, entry.weight, placement, &cluster, &wl, 0)
            .unwrap();
    }

    // Tight tolerance: the DP must agree with the cost model to a sub-µs
    // floor — anything larger means a stale cost path is being summed.
    let diff = (external_sum - result.cost_us).abs();
    assert!(
        diff < 1.0,
        "DP cost {} differs from cost-model sum {} by {} µs",
        result.cost_us,
        external_sum,
        diff
    );
}
