//! Deterministic drift-table writer.
//!
//! Re-uses `skein_extract::DriftTable`'s canonical TOML format for the
//! body (sorted keys, `{:?}` floats), prepended with a calibration
//! header. Unmeasured per-layer overrides from `base` are preserved;
//! fitted measurements overwrite the corresponding `(layer, component,
//! dtype)` entries.

use std::collections::HashMap;
use std::path::Path;

use skein_extract::DriftTable;
use skein_ir::types::{Component, Dtype};

use crate::error::CalibrationError;

/// Write a canonical drift TOML to `out_path`. If `base` is `Some`, its
/// per-layer overrides are carried forward except where `fitted`
/// supersedes them. Defaults always come from `base` when present;
/// otherwise zeros across every `(component, dtype)` pair.
pub fn write_drift_table(
    out_path: &Path,
    base: Option<&DriftTable>,
    fitted: &HashMap<(usize, Component, Dtype), f64>,
    timestamp: &str,
) -> Result<(), CalibrationError> {
    // Start from base (or empty), apply fitted overrides, ask the
    // existing canonical writer to render the body.
    let mut table = base.cloned().unwrap_or_else(empty_drift_table);
    for (&(layer_idx, component, dtype), &drift) in fitted {
        table.set_layer_override(layer_idx, component, dtype, drift);
    }
    let body = format!("# Calibrated {timestamp}\n\n{}", table.serialize_to_toml());
    std::fs::write(out_path, body).map_err(|source| CalibrationError::Io {
        path: out_path.to_path_buf(),
        source,
    })
}

/// Drift table with zero defaults across every `(component, dtype)`.
/// Used when no base table exists yet (first calibration run).
pub fn empty_drift_table() -> DriftTable {
    let mut defaults: HashMap<(Component, Dtype), f64> = HashMap::new();
    for c in Component::ALL {
        for d in Dtype::ALL {
            defaults.insert((c, d), 0.0);
        }
    }
    DriftTable::with_defaults(defaults).expect(
        "empty_drift_table fills every (component, dtype) pair so \
         validate_defaults_present can't fail",
    )
}
