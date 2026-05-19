//! Test 8 — report serialization round-trips byte-equal.

use skein_ir::types::{Component, Dtype};
use skein_parity::report::{FailingLayerReport, ParityReport, PerPromptReport};

#[test]
fn report_serialization_roundtrip() {
    let report = ParityReport {
        passed: false,
        plan_hash: "deadbeef".into(),
        model: "MixtralForCausalLM".into(),
        num_prompts: 2,
        per_prompt: vec![
            PerPromptReport {
                prompt_idx: 0,
                per_layer_mse: vec![0.0, 0.0, 0.0],
                final_kl: 0.0,
            },
            PerPromptReport {
                prompt_idx: 1,
                per_layer_mse: vec![0.5, 0.25, 0.125],
                final_kl: 0.0625,
            },
        ],
        avg_final_kl: 0.03125,
        max_final_kl: 0.0625,
        slo_max_drift: 0.01,
        failing_layer: Some(FailingLayerReport {
            layer_idx: 7,
            component: Component::Weight,
            dtype: Dtype::Int4,
            measured_mse: 0.125,
            tolerance: 0.02,
            prompts_violating: vec![1],
        }),
    };

    // Round-trip via JSON.
    let json = serde_json::to_string(&report).unwrap();
    let back: ParityReport = serde_json::from_str(&json).unwrap();
    assert_eq!(report, back);

    // Byte-equal serialization on the *same* in-memory report — this is
    // what the content-addressable artifact path in Phase A Step 6 relies
    // on. Two serializations of the same Rust value must produce the same
    // bytes.
    let json2 = serde_json::to_string(&report).unwrap();
    assert_eq!(json.as_bytes(), json2.as_bytes());
}
