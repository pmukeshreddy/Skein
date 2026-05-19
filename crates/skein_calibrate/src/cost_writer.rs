//! Deterministic `cost_constants.toml` writer.
//!
//! Renders the same canonical structure `cluster/cost_constants.toml`
//! ships with — sections in a fixed order, floats via `{:?}` so the
//! shortest-round-trip form is bit-stable, dtype keys in `Dtype::ALL`
//! order. Fitted efficiency values overlay the corresponding `[efficiency.*]`
//! cells; *unmeasured* cells are preserved verbatim from `base`. Section
//! ordering, float formatting, and integer field formatting are
//! independent of the OS / hash seed.

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;

use skein_cost::OpKind;
use skein_cost::constants::{CostConstants, DtypeTable};
use skein_ir::types::Dtype;

use crate::error::CalibrationError;
use crate::hardware::HardwareSpec;

/// Write a fully-canonical `cost_constants.toml`.
///
/// `base` carries every value Skein knows about; `fitted` overlays
/// efficiency for the `(op, dtype)` pairs the calibration corpus
/// actually measured. Sections + dtypes not present in `fitted` are
/// emitted with `base`'s value.
pub fn write_cost_constants(
    out_path: &Path,
    base: &CostConstants,
    fitted: &HashMap<(OpKind, Dtype), f64>,
    hardware: &HardwareSpec,
    timestamp: &str,
) -> Result<(), CalibrationError> {
    let body = render(base, fitted, hardware, timestamp);
    std::fs::write(out_path, body).map_err(|source| CalibrationError::Io {
        path: out_path.to_path_buf(),
        source,
    })
}

/// Render-only path — used by tests to compare against bytes without
/// touching the filesystem.
pub fn render(
    base: &CostConstants,
    fitted: &HashMap<(OpKind, Dtype), f64>,
    hardware: &HardwareSpec,
    timestamp: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# Calibrated {timestamp} on {hardware_kind}",
        hardware_kind = hardware.kind,
    );
    let _ = writeln!(out);

    // Top-level scalars first (must come before any [table] header so
    // they land at the document root and not nested inside a table).
    let _ = writeln!(
        out,
        "launch_us_per_kernel = {}",
        fmt_f64(base.launch_us_per_kernel)
    );
    let _ = writeln!(
        out,
        "overshoot_us_per_gb = {}",
        fmt_f64(base.overshoot_us_per_gb)
    );
    let _ = writeln!(out);

    // [peak_tflops.<hw>] — sorted by hardware kind.
    let mut hw_keys: Vec<&String> = base.peak_tflops.keys().collect();
    hw_keys.sort();
    for hw in hw_keys {
        let table = base
            .peak_tflops
            .get(hw)
            .copied()
            .unwrap_or(zero_dtype_table());
        let _ = writeln!(out, "[peak_tflops.{hw}]");
        write_dtype_table(&mut out, &table);
        let _ = writeln!(out);
    }

    // [efficiency.<op>] — fixed op order. fitted overrides apply here.
    for op in [OpKind::Gemm, OpKind::Attention, OpKind::Elementwise] {
        let op_key = op_to_key(op);
        let base_table = base
            .efficiency
            .get(op_key)
            .copied()
            .unwrap_or(zero_dtype_table());
        let merged = merge_efficiency_table(op, &base_table, fitted);
        let _ = writeln!(out, "[efficiency.{op_key}]");
        write_dtype_table(&mut out, &merged);
        let _ = writeln!(out);
    }

    // [collective_efficiency] — fixed field order.
    let _ = writeln!(out, "[collective_efficiency]");
    let c = base.collective_efficiency;
    let _ = writeln!(out, "ring_allreduce = {}", fmt_f64(c.ring_allreduce));
    let _ = writeln!(out, "allgather = {}", fmt_f64(c.allgather));
    let _ = writeln!(out, "reducescatter = {}", fmt_f64(c.reducescatter));
    let _ = writeln!(out, "alltoall = {}", fmt_f64(c.alltoall));
    let _ = writeln!(out, "broadcast = {}", fmt_f64(c.broadcast));
    let _ = writeln!(out, "send_recv = {}", fmt_f64(c.send_recv));
    let _ = writeln!(out);

    // [parity_tolerance_mse]
    let _ = writeln!(out, "[parity_tolerance_mse]");
    write_dtype_table(&mut out, &base.parity_tolerance_mse);
    let _ = writeln!(out);

    // [representative_workload]
    let _ = writeln!(out, "[representative_workload]");
    let _ = writeln!(
        out,
        "prefill_tokens = {}",
        base.representative_workload.prefill_tokens
    );
    let _ = writeln!(
        out,
        "decode_kv_tokens = {}",
        base.representative_workload.decode_kv_tokens
    );
    let _ = writeln!(out);

    // [dp]
    let _ = writeln!(out, "[dp]");
    let _ = writeln!(out, "memory_buckets = {}", base.dp.memory_buckets);
    let _ = writeln!(out, "drift_buckets = {}", base.dp.drift_buckets);
    let _ = writeln!(out);

    // [runtime]
    let _ = writeln!(out, "[runtime]");
    let r = base.runtime;
    let _ = writeln!(
        out,
        "metrics_buffer_capacity = {}",
        r.metrics_buffer_capacity
    );
    let _ = writeln!(out, "drain_timeout_seconds = {}", r.drain_timeout_seconds);
    let _ = writeln!(out, "prometheus_port = {}", r.prometheus_port);
    let _ = writeln!(out, "radix_max_depth = {}", r.radix_max_depth);
    let _ = writeln!(out);

    // [runtime_estimator]
    let _ = writeln!(out, "[runtime_estimator]");
    let e = base.runtime_estimator;
    let _ = writeln!(
        out,
        "per_token_decode_us_at_b1 = {}",
        fmt_f64(e.per_token_decode_us_at_b1)
    );
    let _ = writeln!(
        out,
        "batch_scaling_exponent = {}",
        fmt_f64(e.batch_scaling_exponent)
    );
    let _ = writeln!(
        out,
        "prefill_per_token_us = {}",
        fmt_f64(e.prefill_per_token_us)
    );

    out
}

fn merge_efficiency_table(
    op: OpKind,
    base: &DtypeTable,
    fitted: &HashMap<(OpKind, Dtype), f64>,
) -> DtypeTable {
    DtypeTable {
        bf16: fitted.get(&(op, Dtype::Bf16)).copied().unwrap_or(base.bf16),
        fp16: fitted.get(&(op, Dtype::Fp16)).copied().unwrap_or(base.fp16),
        fp8_e4m3: fitted
            .get(&(op, Dtype::Fp8E4m3))
            .copied()
            .unwrap_or(base.fp8_e4m3),
        fp8_e5m2: fitted
            .get(&(op, Dtype::Fp8E5m2))
            .copied()
            .unwrap_or(base.fp8_e5m2),
        int8: fitted.get(&(op, Dtype::Int8)).copied().unwrap_or(base.int8),
        int4: fitted.get(&(op, Dtype::Int4)).copied().unwrap_or(base.int4),
    }
}

fn write_dtype_table(out: &mut String, t: &DtypeTable) {
    let _ = writeln!(out, "bf16 = {}", fmt_f64(t.bf16));
    let _ = writeln!(out, "fp16 = {}", fmt_f64(t.fp16));
    let _ = writeln!(out, "fp8_e4m3 = {}", fmt_f64(t.fp8_e4m3));
    let _ = writeln!(out, "fp8_e5m2 = {}", fmt_f64(t.fp8_e5m2));
    let _ = writeln!(out, "int8 = {}", fmt_f64(t.int8));
    let _ = writeln!(out, "int4 = {}", fmt_f64(t.int4));
}

fn zero_dtype_table() -> DtypeTable {
    DtypeTable {
        bf16: 0.0,
        fp16: 0.0,
        fp8_e4m3: 0.0,
        fp8_e5m2: 0.0,
        int8: 0.0,
        int4: 0.0,
    }
}

fn op_to_key(op: OpKind) -> &'static str {
    match op {
        OpKind::Gemm => "gemm",
        OpKind::Attention => "attention",
        OpKind::Elementwise => "elementwise",
    }
}

/// `{:?}` produces Rust's shortest round-trip representation for f64,
/// which `toml::from_str` parses back bit-exact. This matches the format
/// the `skein_extract::DriftTable` writer uses, keeping the two crates'
/// output stylistically consistent.
fn fmt_f64(v: f64) -> String {
    format!("{v:?}")
}
