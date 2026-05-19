//! Test 1 — outer enumeration cardinality.

mod common;
use common::*;

use skein_extract::constraints;
use skein_extract::enumerate::enumerate_global_configs;

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
