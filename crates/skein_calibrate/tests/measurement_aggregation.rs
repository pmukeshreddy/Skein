//! Tests 3 + 4 — aggregation statistics.

use skein_calibrate::measurement::{
    DriftMeasurement, KernelMeasurement, aggregate_drift_measurements,
    aggregate_kernel_measurements, median, p95,
};
use skein_cost::OpKind;
use skein_ir::types::{Component, Dtype};

#[test]
fn median_basic() {
    let v = [0.6, 0.7, 0.75, 0.8, 0.95];
    assert!((median(&v) - 0.75).abs() < 1e-12);
}

#[test]
fn p95_linear_interp_picks_outliers() {
    let mut v: Vec<f64> = (0..80).map(|_| 0.01).collect();
    v.extend((0..20).map(|_| 0.05));
    // Linearly-interpolated p95 of [0.01]*80 + [0.05]*20 sorted ascending:
    // pos = 0.95 × 99 = 94.05; sorted[94] = 0.05, sorted[95] = 0.05 → 0.05.
    let p = p95(&v);
    assert!((p - 0.05).abs() < 1e-12, "p95 = {p}");
}

#[test]
fn aggregation_kernel_uses_median() {
    // Build 5 KernelMeasurements whose efficiencies are [0.6, 0.7, 0.75, 0.8, 0.95].
    // efficiency = theoretical / measured; pick theoretical = 1.0 and measured
    // = 1.0 / efficiency_target.
    let targets = [0.6_f64, 0.7, 0.75, 0.8, 0.95];
    let measurements: Vec<KernelMeasurement> = targets
        .iter()
        .map(|&eff| KernelMeasurement {
            op_kind: OpKind::Gemm,
            dtype: Dtype::Bf16,
            shape: vec![4096, 4096],
            measured_us: 1.0 / eff,
            theoretical_peak_us: 1.0,
        })
        .collect();
    let agg = aggregate_kernel_measurements(&measurements);
    let val = agg
        .get(&(OpKind::Gemm, Dtype::Bf16))
        .copied()
        .expect("expected (Gemm, Bf16) entry");
    assert!((val - 0.75).abs() < 1e-6, "expected median 0.75, got {val}");
}

#[test]
fn aggregation_drift_uses_p95() {
    // Build 100 DriftMeasurements: 80 around 0.01, 20 at 0.05. P95 ≈ 0.05.
    let mut measurements = Vec::new();
    for i in 0..80 {
        measurements.push(DriftMeasurement {
            layer_idx: 7,
            component: Component::Weight,
            dtype: Dtype::Int4,
            prompt_idx: i,
            measured_kl: 0.01,
        });
    }
    for i in 0..20 {
        measurements.push(DriftMeasurement {
            layer_idx: 7,
            component: Component::Weight,
            dtype: Dtype::Int4,
            prompt_idx: 80 + i,
            measured_kl: 0.05,
        });
    }
    let agg = aggregate_drift_measurements(&measurements);
    let v = agg
        .get(&(7, Component::Weight, Dtype::Int4))
        .copied()
        .expect("entry");
    assert!((v - 0.05).abs() < 1e-12, "expected p95 ≈ 0.05, got {v}");

    // Pin that the result is *not* the median (which is 0.01). Test 4 from
    // the spec is "uses p95 not median".
    assert!(
        (v - 0.01).abs() > 0.01,
        "drift aggregation collapsed to median ({v}); want p95",
    );
}
