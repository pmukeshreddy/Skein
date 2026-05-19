//! Drift-table update on a parity failure.
//!
//! The protocol is **monotonic-up**: when a `(layer, component, dtype)`
//! triple is measured at drift `m`, we set the recorded value to
//! `max(existing, m)`. Subsequent passes that measure a smaller drift do
//! not lower the record; subsequent passes that measure a larger drift
//! refine it upward.
//!
//! This is the right contract for the re-search trigger:
//!
//! - `skein_extract` predicted that some Plan would meet the SLO at the
//!   recorded drift values.
//! - `skein_parity` measured worse drift than predicted.
//! - We update the recorded value to the measured one (or higher) so the
//!   next `extract_plan` excludes the Plan that just failed.
//! - If we ever decreased the recorded drift, a future search could pick
//!   the same losing Plan back. Hence: never decrease.

use std::path::Path;

use skein_extract::DriftTable;

use crate::error::ParityError;
use crate::report::FailingLayerReport;

/// Apply the failing measurement to `drift_table_path` in place. Reads the
/// current TOML, refines the relevant entry upward (or inserts if absent),
/// and writes back deterministically. Returns the recorded value after the
/// update so callers can log it.
pub fn update_drift_table_on_failure(
    drift_table_path: &Path,
    failing: &FailingLayerReport,
) -> Result<f64, ParityError> {
    let mut table = DriftTable::load(drift_table_path)?;
    let existing = table
        .lookup_raw(failing.layer_idx, failing.component, failing.dtype)
        .unwrap_or(0.0);
    let new_value = existing.max(failing.measured_mse);
    table.set_layer_override(
        failing.layer_idx,
        failing.component,
        failing.dtype,
        new_value,
    );
    table.save_to_toml_file(drift_table_path)?;
    Ok(new_value)
}
