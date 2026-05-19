//! Tests 4–7 + 9 — verify_plan flow, drift-table update, Phase-B gating.

mod common;
use common::*;

use skein_ir::types::{Component, Dtype};
use skein_parity::drift_update::update_drift_table_on_failure;
use skein_parity::report::FailingLayerReport;
use skein_parity::{HFReference, ParityError, PhaseAStub, PythonSubprocessReference, verify_plan};

// Test 4 — passes when the stub returns identical activations.
#[test]
fn verify_plan_passes_under_mock_reference_no_drift() {
    let prompts = sample_prompts();
    let reference = build_mock_reference(&prompts);
    let mut skein = PhaseAStub::new(reference.clone());
    let tolerances = load_tolerances();
    let plan = mk_plan(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16);
    let workload = loose_slo();
    let meta = mk_plan(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16).model_meta;
    let ir = make_ir_from_meta(meta);

    let report = verify_plan(
        &reference,
        &mut skein,
        &ir,
        &plan,
        &workload,
        &tolerances,
        &prompts,
    )
    .expect("verify_plan should succeed under no-drift mock");
    assert!(report.passed, "report should pass: {report:?}");
    assert!(report.failing_layer.is_none());
    for p in &report.per_prompt {
        for &m in &p.per_layer_mse {
            assert_eq!(m, 0.0);
        }
        assert!(p.final_kl.abs() < 1e-10);
    }
}

// Test 5 — fails when one layer exceeds its tolerance.
#[test]
fn verify_plan_fails_when_layer_exceeds_tolerance() {
    let prompts = sample_prompts();
    let reference = build_mock_reference(&prompts);
    // bf16 tolerance is 1e-3. Injecting an offset of 0.5 at layer 10 makes
    // the MSE = 0.25 (since each element is offset by 0.5 → squared = 0.25,
    // mean across the small hidden state is 0.25). 0.25 >> 1e-3.
    let mut offsets = vec![0.0_f32; NUM_BLOCKS];
    offsets[10] = 0.5;
    let mut skein = PhaseAStub::new(reference.clone()).with_layer_offsets(offsets);
    let tolerances = load_tolerances();
    let plan = mk_plan(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16);
    let workload = loose_slo();
    let ir = make_ir_from_meta(plan.model_meta.clone());

    let report = verify_plan(
        &reference,
        &mut skein,
        &ir,
        &plan,
        &workload,
        &tolerances,
        &prompts,
    )
    .expect("verify_plan should run");
    assert!(!report.passed);
    let failing = report.failing_layer.expect("expected failing_layer");
    assert_eq!(failing.layer_idx, 10);
    // Bf16 plan + dominant_dtype is bf16 (weight component, ties broken by
    // first triple).
    assert_eq!(failing.dtype, Dtype::Bf16);
    assert_eq!(failing.component, Component::Weight);
    assert_eq!(failing.prompts_violating.len(), prompts.len());
    assert!(failing.measured_mse > failing.tolerance);
}

// Test 6 — fails when per-layer MSE is within tolerance but final KL
// exceeds the SLO. We accomplish this by leaving the per-layer offsets at
// zero (MSE = 0) and giving the final logits a small uniform offset (which
// keeps log-softmax unchanged — KL still 0). To actually drive KL up we
// inject a non-uniform per-block offset so the final-logit hooks diverge.
//
// Easier approach: inject a tiny per-layer offset whose MSE is below
// every tolerance, but make the final logits *non-uniform-different* by
// also varying per-layer (which the stub propagates to logits is not what
// we have — logits and per-layer activations are separate fields in
// ReferenceOutput). So instead: set per-layer offsets to zero (MSE 0,
// passes the tolerance gate) AND override `final_logits` directly via a
// custom stub.
#[test]
fn verify_plan_fails_when_kl_exceeds_slo() {
    let prompts = sample_prompts();
    let reference = build_mock_reference(&prompts);

    // Build a "tilted logits" stub by wrapping the reference output and
    // perturbing only the final logits with a non-uniform shift. The
    // per-layer activations remain identical.
    struct TiltedStub {
        reference: skein_parity::MockReference,
    }
    impl skein_parity::SkeinForward for TiltedStub {
        fn forward_with_hooks(
            &mut self,
            tokens: &[u32],
        ) -> Result<skein_parity::SkeinOutput, skein_parity::ParityError> {
            let mut out = self.reference.forward_with_hooks(tokens)?;
            // Tilt the logits non-uniformly so KL > 0.
            for (i, v) in out.final_logits.iter_mut().enumerate() {
                *v += (i as f32) * 2.0;
            }
            Ok(skein_parity::SkeinOutput {
                per_layer_activations: out.per_layer_activations,
                final_logits: out.final_logits,
            })
        }
    }
    let mut skein = TiltedStub {
        reference: reference.clone(),
    };

    let tolerances = load_tolerances();
    let plan = mk_plan(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16);
    // A very tight KL SLO that the tilted logits will exceed.
    let workload = tight_slo();
    let ir = make_ir_from_meta(plan.model_meta.clone());

    let report = verify_plan(
        &reference,
        &mut skein,
        &ir,
        &plan,
        &workload,
        &tolerances,
        &prompts,
    )
    .expect("verify_plan should run");
    assert!(!report.passed, "tilted logits should fail KL SLO");
    assert!(
        report.failing_layer.is_none(),
        "per-layer MSE is zero, so failing_layer must be None"
    );
    assert!(report.avg_final_kl > report.slo_max_drift);
}

// Test 7 — drift-table monotonic update.
#[test]
fn drift_table_update_monotonic() {
    let tmpdir = std::env::temp_dir().join(format!(
        "skein_parity_drift_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmpdir).unwrap();
    let drift_path = tmpdir.join("drift.toml");

    // Seed with full defaults (every (component, dtype) pair) so DriftTable
    // loads validate.
    let seed_toml = r#"
[default.weight]
bf16 = 0.0
fp16 = 0.0
fp8_e4m3 = 0.0
fp8_e5m2 = 0.0
int8 = 0.0
int4 = 0.0

[default.activation]
bf16 = 0.0
fp16 = 0.0
fp8_e4m3 = 0.0
fp8_e5m2 = 0.0
int8 = 0.0
int4 = 0.0

[default.kv_cache]
bf16 = 0.0
fp16 = 0.0
fp8_e4m3 = 0.0
fp8_e5m2 = 0.0
int8 = 0.0
int4 = 0.0
"#;
    std::fs::write(&drift_path, seed_toml).unwrap();

    // Apply a measured drift of 0.05 at layer 7 / weight / int4.
    let failing = FailingLayerReport {
        layer_idx: 7,
        component: Component::Weight,
        dtype: Dtype::Int4,
        measured_mse: 0.05,
        tolerance: 0.02,
        prompts_violating: vec![0, 1],
    };
    let after_high = update_drift_table_on_failure(&drift_path, &failing).unwrap();
    assert!((after_high - 0.05).abs() < 1e-12);

    // Now apply a *smaller* measurement. The recorded value must NOT
    // decrease.
    let failing_smaller = FailingLayerReport {
        measured_mse: 0.01,
        ..failing.clone()
    };
    let after_smaller = update_drift_table_on_failure(&drift_path, &failing_smaller).unwrap();
    assert!(
        (after_smaller - 0.05).abs() < 1e-12,
        "monotonic-up violated: shrank to {after_smaller}"
    );

    // Apply a *larger* measurement. The recorded value updates upward.
    let failing_larger = FailingLayerReport {
        measured_mse: 0.15,
        ..failing
    };
    let after_larger = update_drift_table_on_failure(&drift_path, &failing_larger).unwrap();
    assert!((after_larger - 0.15).abs() < 1e-12);

    // Cleanup.
    let _ = std::fs::remove_dir_all(&tmpdir);
}

// Test 9 — PythonSubprocessReference validates the independent reference dtype.
#[test]
fn python_subprocess_reference_validates_reference_dtype() {
    let r = PythonSubprocessReference::with_paths(
        std::path::PathBuf::from("/tmp/model"),
        std::path::PathBuf::from("/usr/bin/python3"),
        std::path::PathBuf::from("/tmp/script.py"),
        "float32",
    );
    let r = r.expect("valid dtype should construct the reference config");
    assert_eq!(r.reference_dtype.as_str(), "float32");

    let bad = PythonSubprocessReference::with_paths(
        std::path::PathBuf::from("/tmp/model"),
        std::path::PathBuf::from("/usr/bin/python3"),
        std::path::PathBuf::from("/tmp/script.py"),
        "fp32",
    );
    assert!(matches!(
        bad,
        Err(ParityError::InvalidReferenceDtype { dtype }) if dtype == "fp32"
    ));
}

/// Build the smallest possible `Graph` whose `meta` matches `mk_plan`'s
/// `model_meta`. We don't exercise the IR layers — `verify_plan` only
/// reads `ir.meta.num_layers` and `ir.meta.architecture`.
fn make_ir_from_meta(meta: skein_ir::ir::ModelMeta) -> skein_ir::ir::Graph {
    skein_ir::ir::Graph {
        meta,
        layers: Vec::new(),
        tensors: std::collections::BTreeMap::new(),
    }
}
