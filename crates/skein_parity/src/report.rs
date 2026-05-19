//! `ParityReport` — the JSON the parity gate emits. Round-trips cleanly via
//! `serde_json` so the artifact directory's `parity_report.json` is
//! reproducible across runs.

use serde::{Deserialize, Serialize};

use skein_ir::types::{Component, Dtype};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityReport {
    pub passed: bool,
    /// Hex of `Plan::content_hash()`. Traceability — the artifact directory
    /// is keyed on this hash.
    pub plan_hash: String,
    /// Architecture string from `ModelMeta` (e.g. "MixtralForCausalLM").
    pub model: String,
    pub num_prompts: usize,
    pub per_prompt: Vec<PerPromptReport>,
    pub avg_final_kl: f64,
    pub max_final_kl: f64,
    pub slo_max_drift: f64,
    /// `None` iff `passed`. When `Some`, callers feed this into
    /// `drift_update::update_drift_table_on_failure`.
    pub failing_layer: Option<FailingLayerReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerPromptReport {
    pub prompt_idx: usize,
    /// One entry per decoder block (`ModelMeta::num_layers`).
    pub per_layer_mse: Vec<f64>,
    pub final_kl: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailingLayerReport {
    pub layer_idx: usize,
    pub component: Component,
    pub dtype: Dtype,
    pub measured_mse: f64,
    pub tolerance: f64,
    /// Prompt indices (into `ParityReport::per_prompt`) that triggered the
    /// failure for this `(layer, component, dtype)` triple.
    pub prompts_violating: Vec<usize>,
}
