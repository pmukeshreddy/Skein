//! Test 7 — drift writer is byte-stable across runs.

mod common;
use common::*;

use std::collections::HashMap;

use skein_calibrate::drift_writer::{empty_drift_table, write_drift_table};
use skein_extract::DriftTable;
use skein_ir::types::{Component, Dtype};

#[test]
fn drift_writer_byte_stable() {
    let mut fitted: HashMap<(usize, Component, Dtype), f64> = HashMap::new();
    fitted.insert((0, Component::Weight, Dtype::Bf16), 0.0);
    fitted.insert((7, Component::Weight, Dtype::Int4), 0.092);
    fitted.insert((7, Component::Activation, Dtype::Fp8E4m3), 0.004);
    fitted.insert((31, Component::KvCache, Dtype::Int8), 0.021);

    let base = empty_drift_table();

    let dir = tempdir("skein_cal_drift_stable");
    let path_a = dir.join("a.toml");
    let path_b = dir.join("b.toml");

    write_drift_table(&path_a, Some(&base), &fitted, "2026-05-19T00:00:00Z").unwrap();
    write_drift_table(&path_b, Some(&base), &fitted, "2026-05-19T00:00:00Z").unwrap();

    let a = std::fs::read(&path_a).unwrap();
    let b = std::fs::read(&path_b).unwrap();
    assert_eq!(a, b, "drift writer is non-deterministic");

    // Re-parse and verify the per-layer overrides land at the expected
    // values (i.e. the writer didn't drop or reorder them).
    let reloaded = DriftTable::from_toml_str(&String::from_utf8_lossy(&a)).unwrap();
    let v = reloaded
        .lookup_raw(7, Component::Weight, Dtype::Int4)
        .expect("override present after round-trip");
    assert!((v - 0.092).abs() < 1e-12);
}
