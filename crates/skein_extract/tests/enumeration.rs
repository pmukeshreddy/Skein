//! Test 1 — outer enumeration cardinality.

mod common;
use common::*;

use skein_extract::constraints;
use skein_extract::enumerate::{
    enumerate_cuda_graphs_configs, enumerate_global_configs, enumerate_spec_decode_configs,
};
use skein_extract::extract_plan;

#[test]
fn enumeration_size_bounded() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let cost_model = load_cost_model();

    let configs = enumerate_global_configs(&cluster, &ir);
    let raw = configs.len();
    let survivors = configs
        .iter()
        .filter(|g| constraints::satisfies_hard_constraints(g, &cluster, &ir, &cost_model))
        .count();

    println!("raw = {raw}, survivors = {survivors}");

    // Sanity bound: the outer space stays under 10k entries on the
    // canonical 2× H100 Mixtral setup after the Phase A enumeration prune
    // (see docs/search_algorithms.md). A regression past this would
    // immediately push extract_plan past its 5 s wall-time budget.
    assert!(raw <= 10_000, "raw enumeration too large: {raw}");

    // Survivor count must be non-zero (the cluster supports *some* viable
    // global) and not exceed the raw total. Tight constraint filtering on
    // a 2-device cluster is rare — all four parallelism families fit, so
    // we expect survivors ≈ raw on this setup.
    assert!(survivors > 0, "no global config survived constraints");
    assert!(survivors <= raw);

    // Every reported survivor must actually satisfy the constraints — a
    // mismatch would mean `reject` and `satisfies_hard_constraints`
    // disagree.
    for g in &configs {
        let passes = constraints::satisfies_hard_constraints(g, &cluster, &ir, &cost_model);
        let reason = constraints::reject(g, &cluster, &ir, &cost_model);
        assert_eq!(passes, reason.is_none());
    }
}

#[test]
fn spec_decode_and_cuda_graphs_are_not_enumerated_until_runtime_wires_them() {
    let spec = enumerate_spec_decode_configs();
    assert_eq!(spec.len(), 1);
    assert!(!spec[0].enable);
    assert!(spec[0].draft.is_none());

    let cuda_graphs = enumerate_cuda_graphs_configs();
    assert_eq!(cuda_graphs.len(), 1);
    assert!(!cuda_graphs[0].enable);
    assert!(cuda_graphs[0].capture_classes.is_empty());
}

#[test]
fn extracted_plan_never_enables_unwired_runtime_features() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let cost_model = load_cost_model();
    let drift_table = load_drift_table();
    let workload = load_workload();

    let plan = extract_plan(&ir, &cluster, &workload, &drift_table, &cost_model)
        .expect("representative search produces a plan");
    assert!(!plan.execution.spec_decode.enable);
    assert!(plan.execution.spec_decode.draft.is_none());
    assert!(!plan.execution.cuda_graphs.enable);
    assert!(plan.execution.cuda_graphs.capture_classes.is_empty());
}
