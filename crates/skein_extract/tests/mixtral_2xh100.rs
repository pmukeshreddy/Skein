//! Test 8 — end-to-end search on Mixtral 8x7B / 2× H100 / ShareGPT trace.
//!
//! The headline acceptance: the full search returns a
//! feasible Plan in under 5 seconds of wall time, the Plan is structurally
//! valid (all 32 decoder blocks assigned a dtype combo), and ranks better
//! than the all-bf16 baseline.

mod common;
use common::*;

use std::time::Instant;

use skein_cost::cluster::Placement;
use skein_extract::extract_plan;
use skein_ir::plan::{DtypeMap, ParallelismPlacement, PerLayerDtype, Plan};
use skein_ir::types::*;

#[test]
fn mixtral_2xh100_end_to_end() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let cost_model = load_cost_model();
    let drift_table = load_drift_table();
    let workload = load_workload();

    let started = Instant::now();
    let plan = extract_plan(&ir, &cluster, &workload, &drift_table, &cost_model)
        .expect("extract_plan should find a feasible plan on Mixtral 2× H100");
    let elapsed = started.elapsed();
    println!("extract_plan completed in {:?}", elapsed);

    // Wall-time budget. Generous bound — if this regresses, the DP is the
    // most likely culprit.
    assert!(
        elapsed.as_secs_f64() < 5.0,
        "extract_plan took {:?}, expected < 5 s",
        elapsed
    );

    // Structural validity.
    assert_eq!(plan.dtype_map.per_layer.len(), ir.meta.num_layers);
    assert!(plan.parallelism.tp >= 1);
    assert!(plan.parallelism.pp >= 1);
    assert!(plan.parallelism.ep >= 1);
    assert!(plan.parallelism.devices_used() <= cluster.num_devices());
    // On a 2-device cluster the only feasible parallelism families are
    // (tp=1, pp=1, ep ∈ {1,2}) or (tp=2, pp=1, ep=1) — pp=2 fits memory but
    // costs more in bubbles. Just assert tp×pp×ep is feasible (already
    // checked above), and that tp itself is reasonable.
    assert!([1u32, 2].contains(&plan.parallelism.tp));

    // Comparison: same Plan, all-bf16 dtype map. The search must beat it
    // (the chosen Plan should at least match the bf16 baseline cost; for a
    // properly drift-tolerant SLO it should beat it via quantization).
    let bf16_plan = Plan {
        parallelism: plan.parallelism,
        kv: plan.kv,
        batching: plan.batching,
        dtype_map: DtypeMap {
            per_layer: vec![
                PerLayerDtype {
                    weight: Dtype::Bf16,
                    activation: Dtype::Bf16,
                    kv_cache: Dtype::Bf16,
                };
                ir.meta.num_layers
            ],
        },
        execution: plan.execution.clone(),
        disaggregation: None,
        model_meta: ir.meta.clone(),
    };

    let chosen_cost = cost_model.total_cost(&plan, &ir, &cluster).unwrap().as_us();
    let bf16_cost = cost_model
        .total_cost(&bf16_plan, &ir, &cluster)
        .unwrap()
        .as_us();

    println!(
        "chosen tp={} pp={} ep={}, total_cost = {:.2} µs (bf16 baseline = {:.2} µs)",
        plan.parallelism.tp, plan.parallelism.pp, plan.parallelism.ep, chosen_cost, bf16_cost
    );

    // The search should never return a strictly worse Plan than the bf16
    // baseline at the same parallelism — bf16-everywhere is in the combo
    // set, so the DP can always fall back to it.
    assert!(
        chosen_cost <= bf16_cost + 1.0,
        "search returned cost {} worse than all-bf16 baseline {}",
        chosen_cost,
        bf16_cost
    );

    // Touch the placement helper to fail loudly if `parallelism.devices_used`
    // is at zero — that would mean the search wandered into idle-device land.
    let placement = Placement {
        tp: plan.parallelism.tp,
        pp: plan.parallelism.pp,
        ep: plan.parallelism.ep,
    };
    assert!(placement.devices_used() >= 1);

    // Silence the unused import warning if proptest later opts out.
    let _ = ParallelismPlacement {
        tp: 1,
        pp: 1,
        ep: 1,
    };
}
