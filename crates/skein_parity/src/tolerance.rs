//! `ToleranceTable` — per-dtype activation-MSE thresholds used by the
//! parity gate. Loaded from `[parity_tolerance_mse]` in
//! `cluster/cost_constants.toml` (the same TOML the cost model uses, so
//! both crates stay in lockstep).

use skein_cost::CostConstants;
use skein_cost::constants::DtypeTable;
use skein_ir::plan::Plan;
use skein_ir::types::Dtype;

/// Per-dtype MSE tolerance. Wrapping the `DtypeTable` rather than re-using
/// it directly keeps the parity-gate intent named at every call site.
#[derive(Debug, Clone, Copy)]
pub struct ToleranceTable {
    inner: DtypeTable,
}

impl ToleranceTable {
    /// Build from a previously-loaded `CostConstants`. There is no
    /// constructor that reads from a file — callers should already have
    /// loaded `cost_constants.toml` once via `CostModel::load`. That
    /// keeps the TOML the single source of truth.
    pub fn from_cost_constants(c: &CostConstants) -> Self {
        ToleranceTable {
            inner: c.parity_tolerance_mse,
        }
    }

    pub fn for_dtype(&self, d: Dtype) -> f64 {
        self.inner.get(d)
    }
}

/// Tolerance the parity gate will apply at `layer_idx`. The layer's
/// per-component dtypes from the Plan are looked up and the *max* across
/// `(weight, activation, kv_cache)` is returned — a layer can drift at
/// least as much as its loosest component allows.
///
/// Out-of-range `layer_idx` falls back to the bf16 tolerance (the
/// tightest threshold), which makes the gate conservative for non-decoder
/// layers (embed, final norm, lm_head).
pub fn tolerance_for_layer(layer_idx: usize, plan: &Plan, tolerances: &ToleranceTable) -> f64 {
    let entry = plan.dtype_map.per_layer.get(layer_idx);
    let dtypes = match entry {
        Some(e) => [e.weight, e.activation, e.kv_cache],
        None => [Dtype::Bf16, Dtype::Bf16, Dtype::Bf16],
    };
    dtypes
        .into_iter()
        .map(|d| tolerances.for_dtype(d))
        .fold(0.0_f64, f64::max)
}

/// The "dominant" dtype at `layer_idx` — the most aggressively quantized
/// among `(weight, activation, kv_cache)`. Used by the parity gate to
/// attribute a failure to a specific `(layer, component, dtype)` triple.
pub fn dominant_dtype_at_layer(
    plan: &Plan,
    layer_idx: usize,
) -> (skein_ir::types::Component, Dtype) {
    use skein_ir::types::Component;
    let entry = plan
        .dtype_map
        .per_layer
        .get(layer_idx)
        .copied()
        .unwrap_or(skein_ir::plan::PerLayerDtype::uniform(Dtype::Bf16));
    let triples = [
        (Component::Weight, entry.weight),
        (Component::Activation, entry.activation),
        (Component::KvCache, entry.kv_cache),
    ];
    // Pick whichever component uses the dtype with the *smallest* bit-width
    // (most aggressive quantization). Ties broken by component order.
    triples
        .into_iter()
        .min_by_key(|(_, d)| (d.bits(), component_order(*d)))
        .expect("triples is non-empty")
}

/// Stable secondary key for tie-breaking inside `dominant_dtype_at_layer`.
fn component_order(_d: Dtype) -> u8 {
    0
}
