//! Measurement structs + statistical aggregation.
//!
//! Median for efficiency: we want the typical sustained per-kernel
//! efficiency — outliers (warmup runs, page faults) don't represent
//! steady-state behaviour.
//!
//! P95 for drift, linearly interpolated: the drift table should
//! over-predict, not under-predict. Matching `skein_parity`'s monotonic-up
//! protocol — predictions should bound the worst-case prompt behaviour so
//! the DP excludes Plans before the parity gate has to reject them.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use skein_cost::OpKind;
use skein_ir::types::{Component, Dtype};

/// One kernel-runtime measurement. The kernel sampler produces these and the
/// aggregation/fit code derives cost constants from them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMeasurement {
    pub op_kind: OpKind,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub measured_us: f64,
    pub theoretical_peak_us: f64,
}

impl KernelMeasurement {
    /// Sustained efficiency relative to peak: `theoretical / measured`,
    /// always in `(0.0, 1.0]` for healthy kernels. Returns 0.0 for a
    /// degenerate `measured_us == 0` to avoid division by zero — callers
    /// can filter, but the aggregation pipeline handles 0.0 cleanly.
    pub fn efficiency(&self) -> f64 {
        if self.measured_us <= 0.0 {
            0.0
        } else {
            self.theoretical_peak_us / self.measured_us
        }
    }
}

/// One drift measurement against the bf16 reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMeasurement {
    pub layer_idx: usize,
    pub component: Component,
    pub dtype: Dtype,
    pub prompt_idx: usize,
    pub measured_kl: f64,
}

/// Median efficiency per `(op_kind, dtype)`.
pub fn aggregate_kernel_measurements(
    measurements: &[KernelMeasurement],
) -> HashMap<(OpKind, Dtype), f64> {
    let mut by_key: HashMap<(OpKind, Dtype), Vec<f64>> = HashMap::new();
    for m in measurements {
        by_key
            .entry((m.op_kind, m.dtype))
            .or_default()
            .push(m.efficiency());
    }
    by_key
        .into_iter()
        .map(|(k, effs)| (k, median(&effs)))
        .collect()
}

/// P95 drift per `(layer_idx, component, dtype)`.
pub fn aggregate_drift_measurements(
    measurements: &[DriftMeasurement],
) -> HashMap<(usize, Component, Dtype), f64> {
    let mut by_key: HashMap<(usize, Component, Dtype), Vec<f64>> = HashMap::new();
    for m in measurements {
        by_key
            .entry((m.layer_idx, m.component, m.dtype))
            .or_default()
            .push(m.measured_kl);
    }
    by_key.into_iter().map(|(k, kls)| (k, p95(&kls))).collect()
}

/// Median by sorting + indexing. Even-length arrays use the lower
/// midpoint (no interpolation) — efficiency measurements have enough
/// samples that this rounding effect is negligible.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// P95 with linear interpolation between adjacent sorted indices. Standard
/// "Type 7" definition (NumPy default).
pub fn p95(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    let pos = 0.95 * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    v[lo] * (1.0 - frac) + v[hi] * frac
}
