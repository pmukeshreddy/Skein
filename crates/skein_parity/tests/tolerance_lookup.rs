//! Test 3 — tolerance lookup.

mod common;
use common::*;

use skein_ir::types::Dtype;
use skein_parity::tolerance::tolerance_for_layer;

#[test]
fn tolerance_picks_loosest_component() {
    let tolerances = load_tolerances();
    // weight=bf16 (1e-3), activation=fp8 (5e-3), kv=int8 (1e-2).
    let plan = mk_plan(Dtype::Bf16, Dtype::Fp8E4m3, Dtype::Int8);
    let tol = tolerance_for_layer(5, &plan, &tolerances);
    // The loosest (max) is int8's 1e-2.
    assert!((tol - 1.0e-2).abs() < 1e-12, "got {tol}");
}

#[test]
fn tolerance_all_bf16_is_tight() {
    let tolerances = load_tolerances();
    let plan = mk_plan(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16);
    let tol = tolerance_for_layer(0, &plan, &tolerances);
    assert!((tol - 1.0e-3).abs() < 1e-12);
}

#[test]
fn tolerance_layer_out_of_range_defaults_bf16() {
    let tolerances = load_tolerances();
    let plan = mk_plan(Dtype::Int4, Dtype::Int4, Dtype::Int4);
    // Out-of-range layer falls back to bf16 (tight) — appropriate for
    // non-decoder layers (embed, lm_head, final norm) that are not in the
    // search axis.
    let tol = tolerance_for_layer(9_999, &plan, &tolerances);
    assert!((tol - 1.0e-3).abs() < 1e-12);
}
