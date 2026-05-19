//! Test 5 — `--output json` produces byte-stable output across two runs
//! with the same inputs.

use std::collections::BTreeMap;

use skein_cli::output::{ExtractReport, ReportParallelism};

/// We exercise the *serializer* directly here. The full `cmd::extract::run`
/// path is covered by extract_e2e.rs; this test pins the serialization
/// contract that the integration test depends on (BTreeMap → sorted keys
/// → byte-identical JSON across runs).
#[test]
fn output_json_stable() {
    let report = ExtractReport {
        plan_hash: "3a7f9e1b8c2d".to_string(),
        plan_path: "plan.json".to_string(),
        search_wall_ms: 624,
        cost_us: 12_700.0,
        parallelism: ReportParallelism {
            tp: 2,
            pp: 1,
            ep: 1,
        },
        dtype_summary: {
            let mut m = BTreeMap::new();
            m.insert("fp8_e4m3".to_string(), 29);
            m.insert("bf16".to_string(), 3);
            m
        },
        model: "MixtralForCausalLM".to_string(),
    };

    let a = serde_json::to_string(&report).unwrap();
    let b = serde_json::to_string(&report).unwrap();
    assert_eq!(
        a.as_bytes(),
        b.as_bytes(),
        "JSON output is non-deterministic"
    );

    // The BTreeMap emits keys in sorted order — `bf16` before `fp8_e4m3`.
    let bf_pos = a.find("bf16").expect("bf16 in JSON");
    let fp8_pos = a.find("fp8_e4m3").expect("fp8_e4m3 in JSON");
    assert!(bf_pos < fp8_pos, "dtype_summary keys should be alphabetic");

    // Field order is the struct's declaration order (serde default), not
    // insertion-order of any map. Spot-check that the report stays a
    // single object and contains every named field.
    for needle in [
        "\"plan_hash\":",
        "\"plan_path\":",
        "\"search_wall_ms\":",
        "\"cost_us\":",
        "\"parallelism\":",
        "\"dtype_summary\":",
        "\"model\":",
    ] {
        assert!(a.contains(needle), "missing key {needle} in {a}");
    }
}
