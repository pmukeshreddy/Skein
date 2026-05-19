//! Tests 5 + 6 — cost-constants writer preserves unmeasured fields and
//! produces byte-stable output across runs.

mod common;
use common::*;

use std::collections::HashMap;

use skein_calibrate::HardwareSpec;
use skein_calibrate::cost_writer::{render, write_cost_constants};
use skein_cost::{CostConstants, CostModel, OpKind};
use skein_ir::types::Dtype;

#[test]
fn cost_writer_preserves_unmeasured() {
    let base = load_cost_constants();
    // Only fit one cell — (Gemm, Bf16). Every other (op, dtype) in
    // [efficiency.*] must come straight from `base`.
    let mut fitted: HashMap<(OpKind, Dtype), f64> = HashMap::new();
    fitted.insert((OpKind::Gemm, Dtype::Bf16), 0.999);

    let hardware = HardwareSpec::new("h100_sxm5");
    let dir = tempdir("skein_cal_preserve");
    let out = dir.join("cost_constants.toml");
    write_cost_constants(&out, &base, &fitted, &hardware, "2026-05-19T00:00:00Z").unwrap();

    // Reload via the production parser; that's the contract the runtime
    // counts on.
    let reloaded = CostModel::from_toml_str(&std::fs::read_to_string(&out).unwrap())
        .expect("reloaded cost constants")
        .constants()
        .clone();

    let base_attn = base.efficiency.get("attention").copied().unwrap();
    let new_attn = reloaded.efficiency.get("attention").copied().unwrap();
    assert_eq!(base_attn.bf16.to_bits(), new_attn.bf16.to_bits());
    assert_eq!(base_attn.fp16.to_bits(), new_attn.fp16.to_bits());
    assert_eq!(base_attn.fp8_e4m3.to_bits(), new_attn.fp8_e4m3.to_bits());
    assert_eq!(base_attn.fp8_e5m2.to_bits(), new_attn.fp8_e5m2.to_bits());
    assert_eq!(base_attn.int8.to_bits(), new_attn.int8.to_bits());
    assert_eq!(base_attn.int4.to_bits(), new_attn.int4.to_bits());

    let base_elem = base.efficiency.get("elementwise").copied().unwrap();
    let new_elem = reloaded.efficiency.get("elementwise").copied().unwrap();
    assert_eq!(base_elem.bf16.to_bits(), new_elem.bf16.to_bits());
    assert_eq!(base_elem.int4.to_bits(), new_elem.int4.to_bits());

    // The fitted cell took effect.
    let new_gemm = reloaded.efficiency.get("gemm").copied().unwrap();
    assert!((new_gemm.bf16 - 0.999).abs() < 1e-12);
    // Other gemm dtypes preserved.
    assert_eq!(
        base.efficiency
            .get("gemm")
            .copied()
            .unwrap()
            .fp8_e4m3
            .to_bits(),
        new_gemm.fp8_e4m3.to_bits()
    );

    // Other top-level scalars survive (random spot-checks).
    assert_eq!(reloaded.launch_us_per_kernel, base.launch_us_per_kernel);
    assert_eq!(reloaded.overshoot_us_per_gb, base.overshoot_us_per_gb);
    assert_eq!(
        reloaded.representative_workload.prefill_tokens,
        base.representative_workload.prefill_tokens
    );
    assert_eq!(reloaded.dp.memory_buckets, base.dp.memory_buckets);
    assert_eq!(
        reloaded.runtime.prometheus_port,
        base.runtime.prometheus_port
    );
}

#[test]
fn cost_writer_byte_stable() {
    let base = load_cost_constants();
    let mut fitted: HashMap<(OpKind, Dtype), f64> = HashMap::new();
    fitted.insert((OpKind::Gemm, Dtype::Bf16), 0.8123);
    fitted.insert((OpKind::Attention, Dtype::Fp8E4m3), 0.61);
    let hardware = HardwareSpec::new("h100_sxm5");

    let a = render(&base, &fitted, &hardware, "2026-05-19T00:00:00Z");
    let b = render(&base, &fitted, &hardware, "2026-05-19T00:00:00Z");
    assert_eq!(a.as_bytes(), b.as_bytes(), "writer is non-deterministic");

    // Also confirm the byte form parses back via the production loader.
    let _ = CostConstants::from_toml_str(&a).expect("written TOML reparses cleanly");
}
