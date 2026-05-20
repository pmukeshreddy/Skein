//! Hardware + model identifiers.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use skein_cost::CostConstants;
use skein_ir::types::Dtype;

/// Identifier matching the `[peak_tflops.<kind>]` key in
/// `cluster/cost_constants.toml`. The cost-constants writer uses this to
/// pick the table to render.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub kind: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub peak_tflops: HashMap<Dtype, f64>,
}

impl HardwareSpec {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            peak_tflops: HashMap::new(),
        }
    }

    pub fn with_cost_constants(mut self, constants: &CostConstants) -> Self {
        if let Some(table) = constants.peak_tflops.get(&self.kind) {
            for dtype in Dtype::ALL {
                self.peak_tflops.insert(dtype, table.get(dtype));
            }
        }
        self
    }

    pub fn peak_tflops(&self, dtype: Dtype) -> Option<f64> {
        self.peak_tflops
            .get(&dtype)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
    }
}

/// Model identifier — a name + optional path to the HF `config.json`. The
/// path is consumed only by the drift sampler; the cost-constants writer
/// doesn't need it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub config_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_artifact: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_artifact: Option<PathBuf>,
}

impl ModelSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config_path: None,
            reference_artifact: None,
            candidate_artifact: None,
        }
    }

    pub fn with_config(name: impl Into<String>, config_path: PathBuf) -> Self {
        Self {
            name: name.into(),
            config_path: Some(config_path),
            reference_artifact: None,
            candidate_artifact: None,
        }
    }

    pub fn with_artifacts(mut self, reference: PathBuf, candidate: PathBuf) -> Self {
        self.reference_artifact = Some(reference);
        self.candidate_artifact = Some(candidate);
        self
    }
}
