//! `DriftTable` — KL-divergence contributions per `(layer_idx, component,
//! dtype)` triple. Loaded from `models/<model>_drift.toml`.
//!
//! Schema: a `[default.<component>]` table mapping every dtype to a non-
//! negative float; optional `[layer.<idx>.<component>]` overrides on
//! per-layer specifics. Missing `(component, dtype)` pairs in `[default.*]`
//! are a hard error — the DP would silently treat the combo as "zero drift"
//! otherwise.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use skein_ir::types::{Component, Dtype};

use crate::error::ExtractError;

#[derive(Debug, Clone)]
pub struct DriftTable {
    /// `(layer_idx, component, dtype) → drift`. Populated from
    /// `[layer.N.<component>]` sections, if present.
    per_layer: HashMap<(usize, Component, Dtype), f64>,
    /// `(component, dtype) → drift`. Required to cover every dtype the DP
    /// might consider; `validate_defaults_present` enforces this at load.
    defaults: HashMap<(Component, Dtype), f64>,
}

impl DriftTable {
    pub fn load(path: &Path) -> Result<Self, ExtractError> {
        let s = std::fs::read_to_string(path).map_err(|source| ExtractError::DriftIo {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml_str(&s)
    }

    pub fn from_toml_str(s: &str) -> Result<Self, ExtractError> {
        let raw: RawDriftFile = toml::from_str(s)?;
        let defaults = raw.default.flatten();
        let mut per_layer: HashMap<(usize, Component, Dtype), f64> = HashMap::new();
        if let Some(map) = raw.layer {
            for (idx_str, sections) in map {
                let idx: usize = idx_str
                    .parse()
                    .map_err(|_| ExtractError::DriftBadLayerKey {
                        key: idx_str.clone(),
                    })?;
                for (component, dtype_map) in sections.flatten_by_component() {
                    for (dtype, drift) in dtype_map {
                        per_layer.insert((idx, component, dtype), drift);
                    }
                }
            }
        }
        let t = DriftTable {
            per_layer,
            defaults,
        };
        t.validate_defaults_present()?;
        Ok(t)
    }

    /// Drift contribution for a single `(layer, component, dtype)` triple.
    /// Falls back to the `default` table when no per-layer override exists.
    /// Returns `f64::INFINITY` if the dtype is absent from both — that
    /// excludes the combo from the DP without a special-case branch.
    pub fn lookup(&self, layer_idx: usize, component: Component, dtype: Dtype) -> f64 {
        if let Some(v) = self.per_layer.get(&(layer_idx, component, dtype)) {
            return *v;
        }
        self.defaults
            .get(&(component, dtype))
            .copied()
            .unwrap_or(f64::INFINITY)
    }

    /// Additive drift contribution for a full `(weight, activation, kv)`
    /// combo at one decoder block. The DP accumulates these.
    pub fn block_contribution(
        &self,
        layer_idx: usize,
        weight: Dtype,
        activation: Dtype,
        kv_cache: Dtype,
    ) -> f64 {
        self.lookup(layer_idx, Component::Weight, weight)
            + self.lookup(layer_idx, Component::Activation, activation)
            + self.lookup(layer_idx, Component::KvCache, kv_cache)
    }

    /// Build a table directly from default values. Used by tests that don't
    /// need per-layer overrides.
    pub fn with_defaults(defaults: HashMap<(Component, Dtype), f64>) -> Result<Self, ExtractError> {
        let t = DriftTable {
            per_layer: HashMap::new(),
            defaults,
        };
        t.validate_defaults_present()?;
        Ok(t)
    }

    /// Look up the raw stored drift without falling back to the default
    /// table. Returns `None` if no per-layer override exists for this
    /// `(layer, component, dtype)` triple. `skein_parity::drift_update` uses
    /// this to read the current override before deciding whether to refine
    /// upward.
    pub fn lookup_raw(&self, layer_idx: usize, component: Component, dtype: Dtype) -> Option<f64> {
        self.per_layer.get(&(layer_idx, component, dtype)).copied()
    }

    /// Insert or overwrite a per-layer override. Note that this does *not*
    /// enforce monotonicity — callers (`skein_parity::drift_update`) own
    /// the never-decrease protocol.
    pub fn set_layer_override(
        &mut self,
        layer_idx: usize,
        component: Component,
        dtype: Dtype,
        drift: f64,
    ) {
        self.per_layer.insert((layer_idx, component, dtype), drift);
    }

    /// Serialize back to canonical TOML. Per-layer overrides are sorted by
    /// `(layer_idx, component, dtype)` so the same in-memory `DriftTable`
    /// produces byte-identical output across runs.
    pub fn save_to_toml_file(&self, path: &std::path::Path) -> Result<(), ExtractError> {
        let toml_str = self.serialize_to_toml();
        std::fs::write(path, toml_str).map_err(|source| ExtractError::DriftIo {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Canonical TOML representation. Public for tests; production code uses
    /// `save_to_toml_file`.
    pub fn serialize_to_toml(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "# Drift table written by `skein_parity::drift_update`. \
             Per-layer overrides are recorded after a measured parity\n# \
             failure and never decrease over time.\n\n",
        );
        // Defaults — emit in (component, dtype) canonical order.
        for component in Component::ALL {
            out.push_str(&format!("[default.{}]\n", component_key(component)));
            for dtype in Dtype::ALL {
                let v = self
                    .defaults
                    .get(&(component, dtype))
                    .copied()
                    .unwrap_or(0.0);
                out.push_str(&format!("{} = {}\n", dtype_key(dtype), float_canonical(v)));
            }
            out.push('\n');
        }
        // Per-layer overrides — sorted for determinism.
        let mut keys: Vec<(usize, Component, Dtype)> = self.per_layer.keys().copied().collect();
        keys.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(component_order(a.1).cmp(&component_order(b.1)))
                .then(dtype_order(a.2).cmp(&dtype_order(b.2)))
        });
        // Group by (layer, component) so each TOML table holds the dtype map.
        let mut current: Option<(usize, Component)> = None;
        for key in &keys {
            let (layer, comp, dt) = *key;
            if current != Some((layer, comp)) {
                if current.is_some() {
                    out.push('\n');
                }
                out.push_str(&format!("[layer.{layer}.{}]\n", component_key(comp)));
                current = Some((layer, comp));
            }
            let v = self.per_layer[key];
            out.push_str(&format!("{} = {}\n", dtype_key(dt), float_canonical(v)));
        }
        out
    }

    fn validate_defaults_present(&self) -> Result<(), ExtractError> {
        for component in Component::ALL {
            for dtype in Dtype::ALL {
                if !self.defaults.contains_key(&(component, dtype)) {
                    return Err(ExtractError::DriftMissingDefault { component, dtype });
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Raw TOML shape. We keep this as a private deserializer that flattens into
// the public `DriftTable` form so the file layout can evolve without leaking
// into the lookup surface.
// ---------------------------------------------------------------------------

// --- Canonical ordering helpers for `serialize_to_toml`. ---

fn component_key(c: Component) -> &'static str {
    match c {
        Component::Weight => "weight",
        Component::Activation => "activation",
        Component::KvCache => "kv_cache",
    }
}

fn dtype_key(d: Dtype) -> &'static str {
    match d {
        Dtype::Bf16 => "bf16",
        Dtype::Fp16 => "fp16",
        Dtype::Fp8E4m3 => "fp8_e4m3",
        Dtype::Fp8E5m2 => "fp8_e5m2",
        Dtype::Int8 => "int8",
        Dtype::Int4 => "int4",
    }
}

fn component_order(c: Component) -> u8 {
    match c {
        Component::Weight => 0,
        Component::Activation => 1,
        Component::KvCache => 2,
    }
}

fn dtype_order(d: Dtype) -> u8 {
    match d {
        Dtype::Bf16 => 0,
        Dtype::Fp16 => 1,
        Dtype::Fp8E4m3 => 2,
        Dtype::Fp8E5m2 => 3,
        Dtype::Int8 => 4,
        Dtype::Int4 => 5,
    }
}

/// Float formatting that round-trips cleanly through `toml::from_str`. Uses
/// `{:?}` (Rust's shortest representation) which preserves f64 bit patterns
/// for any finite value.
fn float_canonical(v: f64) -> String {
    format!("{v:?}")
}

#[derive(Debug, Deserialize)]
struct RawDriftFile {
    default: ComponentSections,
    #[serde(default)]
    layer: Option<HashMap<String, ComponentSections>>,
}

#[derive(Debug, Deserialize)]
struct ComponentSections {
    weight: Option<DtypeSection>,
    activation: Option<DtypeSection>,
    kv_cache: Option<DtypeSection>,
}

#[derive(Debug, Deserialize)]
struct DtypeSection {
    bf16: Option<f64>,
    fp16: Option<f64>,
    fp8_e4m3: Option<f64>,
    fp8_e5m2: Option<f64>,
    int8: Option<f64>,
    int4: Option<f64>,
}

impl ComponentSections {
    /// Flatten to `(component, dtype) → drift` entries, skipping any dtype
    /// the section doesn't mention. Used to assemble the `defaults` table.
    fn flatten(self) -> HashMap<(Component, Dtype), f64> {
        let mut out = HashMap::new();
        for (component, section) in self.flatten_by_component() {
            for (dtype, drift) in section {
                out.insert((component, dtype), drift);
            }
        }
        out
    }

    fn flatten_by_component(self) -> Vec<(Component, Vec<(Dtype, f64)>)> {
        let mut out: Vec<(Component, Vec<(Dtype, f64)>)> = Vec::new();
        if let Some(w) = self.weight {
            out.push((Component::Weight, w.into_pairs()));
        }
        if let Some(a) = self.activation {
            out.push((Component::Activation, a.into_pairs()));
        }
        if let Some(k) = self.kv_cache {
            out.push((Component::KvCache, k.into_pairs()));
        }
        out
    }
}

impl DtypeSection {
    fn into_pairs(self) -> Vec<(Dtype, f64)> {
        let mut v = Vec::new();
        if let Some(x) = self.bf16 {
            v.push((Dtype::Bf16, x));
        }
        if let Some(x) = self.fp16 {
            v.push((Dtype::Fp16, x));
        }
        if let Some(x) = self.fp8_e4m3 {
            v.push((Dtype::Fp8E4m3, x));
        }
        if let Some(x) = self.fp8_e5m2 {
            v.push((Dtype::Fp8E5m2, x));
        }
        if let Some(x) = self.int8 {
            v.push((Dtype::Int8, x));
        }
        if let Some(x) = self.int4 {
            v.push((Dtype::Int4, x));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_DEFAULTS: &str = r#"
[default.weight]
bf16 = 0.0
fp16 = 0.0001
fp8_e4m3 = 0.008
fp8_e5m2 = 0.012
int8 = 0.025
int4 = 0.094

[default.activation]
bf16 = 0.0
fp16 = 0.0001
fp8_e4m3 = 0.005
fp8_e5m2 = 0.008
int8 = 0.020
int4 = 0.080

[default.kv_cache]
bf16 = 0.0
fp16 = 0.0001
fp8_e4m3 = 0.003
fp8_e5m2 = 0.005
int8 = 0.018
int4 = 0.075
"#;

    #[test]
    fn parses_defaults_and_looks_up() {
        let t = DriftTable::from_toml_str(FULL_DEFAULTS).unwrap();
        assert_eq!(t.lookup(0, Component::Weight, Dtype::Bf16), 0.0);
        assert!((t.lookup(0, Component::Weight, Dtype::Fp8E4m3) - 0.008).abs() < 1e-12);
        assert!((t.lookup(99, Component::KvCache, Dtype::Int8) - 0.018).abs() < 1e-12);
    }

    #[test]
    fn block_contribution_sums_three_components() {
        let t = DriftTable::from_toml_str(FULL_DEFAULTS).unwrap();
        let s = t.block_contribution(0, Dtype::Fp8E4m3, Dtype::Bf16, Dtype::Int8);
        // 0.008 + 0.0 + 0.018 = 0.026
        assert!((s - 0.026).abs() < 1e-12);
    }

    #[test]
    fn missing_default_dtype_fails_load() {
        // Drop the bf16 line from weight: must error.
        let bad = FULL_DEFAULTS.replacen("bf16 = 0.0\n", "", 1);
        let err = DriftTable::from_toml_str(&bad).unwrap_err();
        match err {
            ExtractError::DriftMissingDefault {
                component: Component::Weight,
                dtype: Dtype::Bf16,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn per_layer_override_wins() {
        let with_override = format!("{FULL_DEFAULTS}\n[layer.7.weight]\nfp8_e4m3 = 0.5\n");
        let t = DriftTable::from_toml_str(&with_override).unwrap();
        // Layer 7 weight fp8_e4m3 is overridden.
        assert!((t.lookup(7, Component::Weight, Dtype::Fp8E4m3) - 0.5).abs() < 1e-12);
        // Other layers still use the default.
        assert!((t.lookup(8, Component::Weight, Dtype::Fp8E4m3) - 0.008).abs() < 1e-12);
    }
}
