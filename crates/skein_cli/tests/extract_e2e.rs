//! Extract CLI end-to-end and determinism tests.

mod common;
use common::*;

use skein_cli::cli::OutputFormat;
use skein_cli::cmd;

// Test 1 — `skein extract` against the bundled Mixtral fixture.
#[test]
fn extract_e2e_mixtral_2xh100() {
    let (args, _dir) = mixtral_extract_args("skein_cli_e2e");
    let out = args.out.clone();
    cmd::extract::run(args, OutputFormat::Text).expect("extract should succeed");

    // plan.json was written.
    assert!(out.exists(), "plan.json missing at {}", out.display());

    // The file round-trips back into a `Plan`.
    let bytes = std::fs::read(&out).unwrap();
    let plan: skein_ir::plan::Plan =
        serde_json::from_slice(&bytes).expect("plan.json should deserialize");

    // The chosen Plan parallelism fits the 2-device cluster.
    assert!(plan.parallelism.devices_used() <= 2);
    assert!(plan.parallelism.tp >= 1);
    // Mixtral has 32 decoder blocks; the dtype_map must cover all of them.
    assert_eq!(plan.dtype_map.per_layer.len(), 32);
    assert_eq!(plan.model_meta.architecture, "MixtralForCausalLM");
}

// Test 2 — running extract twice with the same inputs produces
// byte-identical plan.json files.
#[test]
fn extract_deterministic() {
    let (a_args, _a_dir) = mixtral_extract_args("skein_cli_det_a");
    let a_out = a_args.out.clone();
    cmd::extract::run(a_args, OutputFormat::Text).unwrap();
    let a_bytes = std::fs::read(&a_out).unwrap();

    let (b_args, _b_dir) = mixtral_extract_args("skein_cli_det_b");
    let b_out = b_args.out.clone();
    cmd::extract::run(b_args, OutputFormat::Text).unwrap();
    let b_bytes = std::fs::read(&b_out).unwrap();

    assert_eq!(
        a_bytes, b_bytes,
        "plan.json should be byte-identical across runs"
    );
}
