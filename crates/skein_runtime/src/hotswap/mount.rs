//! Shape-compatibility check against `io.json` manifests.
//!
//! The new artifact must accept the same *input* shapes the old one
//! accepts; otherwise in-flight requests cannot transfer. Output shapes
//! may differ (the new Plan might quantize differently).

use std::path::Path;

use skein_emit::IoManifest;

use crate::error::RuntimeError;

pub fn verify_shape_compatibility(
    old_artifact: &Path,
    new_artifact: &Path,
) -> Result<(), RuntimeError> {
    // `io.json` is conventionally placed at <artifact>/device_0/io.json (the
    // first device's manifest); for swap-compatibility we compare device 0
    // on both sides. Different `num_devices` is *not* a swap break — the
    // input shapes are identical across the TP group.
    let old_path = old_artifact.join("device_0/io.json");
    let new_path = new_artifact.join("device_0/io.json");
    let old = load_manifest(&old_path)?;
    let new = load_manifest(&new_path)?;
    let mismatch = describe_mismatch(&old, &new);
    if let Some(reason) = mismatch {
        return Err(RuntimeError::IncompatibleArtifact {
            old: old_path,
            new: new_path,
            reason,
        });
    }
    Ok(())
}

fn load_manifest(path: &Path) -> Result<IoManifest, RuntimeError> {
    let bytes = std::fs::read(path).map_err(|source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| RuntimeError::JsonParse {
        path: path.to_path_buf(),
        source,
    })
}

/// Returns `Some(reason)` when the new manifest is incompatible, `None`
/// when it's a valid swap target.
fn describe_mismatch(old: &IoManifest, new: &IoManifest) -> Option<String> {
    // Input tensors define what the runtime feeds at request entry. They
    // must match in name + shape + dtype across artifacts.
    let old_inputs: Vec<&skein_emit::IoTensor> = old
        .tensors
        .iter()
        .filter(|t| matches!(t.kind, skein_emit::IoTensorKind::Input))
        .collect();
    let new_inputs: Vec<&skein_emit::IoTensor> = new
        .tensors
        .iter()
        .filter(|t| matches!(t.kind, skein_emit::IoTensorKind::Input))
        .collect();

    if old_inputs.len() != new_inputs.len() {
        return Some(format!(
            "input tensor count differs: old has {}, new has {}",
            old_inputs.len(),
            new_inputs.len(),
        ));
    }
    for old_t in &old_inputs {
        let Some(new_t) = new_inputs.iter().find(|t| t.name == old_t.name) else {
            return Some(format!(
                "input tensor {:?} missing in new manifest",
                old_t.name
            ));
        };
        if new_t.shape != old_t.shape {
            return Some(format!(
                "input tensor {:?} shape changed: {:?} → {:?}",
                old_t.name, old_t.shape, new_t.shape,
            ));
        }
        if new_t.dtype != old_t.dtype {
            return Some(format!(
                "input tensor {:?} dtype changed: {:?} → {:?}",
                old_t.name, old_t.dtype, new_t.dtype,
            ));
        }
    }
    None
}
