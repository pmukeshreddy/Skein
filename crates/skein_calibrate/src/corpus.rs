//! `CalibrationCorpus` — what the calibration tool measures.
//!
//! Loaded from a TOML manifest at `corpus/<model>.toml`. Two parts:
//!
//! - **Kernel samples** — architecture facts (Mixtral GEMMs are 4096-wide
//!   regardless of workload). Enumerated literally in TOML.
//! - **Drift prompt source** — a reference to a JSONL trace file plus a
//!   sampling configuration. Drift prompts are *not* enumerated in the
//!   corpus (that would be synthetic data baked into a config file); the
//!   Phase B drift sampler reads the trace and samples deterministically
//!   via `prompt_sampling::sample_drift_prompts`.
//!
//! The loader validates structure only — the trace file does not need to
//! exist on the host parsing the corpus (it's user-provided at calibration
//! time on the GPU box).

use std::path::{Path, PathBuf};

use serde::Deserialize;

use skein_cost::OpKind;
use skein_ir::types::Dtype;

use crate::error::CalibrationError;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CalibrationCorpus {
    /// Where to read drift-measurement prompts from. Flattened into the
    /// top level of the TOML for readability — `workload_trace_path`,
    /// `n_drift_prompts`, `sampling_strategy`, `sampling_seed` appear as
    /// flat keys.
    #[serde(flatten)]
    pub drift_source: DriftPromptSource,
    #[serde(default)]
    pub kernel_samples: Vec<KernelSample>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DriftPromptSource {
    pub workload_trace_path: PathBuf,
    pub n_drift_prompts: usize,
    pub sampling_strategy: SamplingStrategy,
    /// Seed for the deterministic-shuffle sampling strategies. Same seed +
    /// same trace + same strategy → byte-identical sampled prompt list.
    #[serde(default)]
    pub sampling_seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategy {
    UniformRandom,
    FirstN,
    StratifiedByLength,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct KernelSample {
    pub op_kind: OpKind,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub repeats: u32,
}

impl CalibrationCorpus {
    pub fn load(path: &Path) -> Result<Self, CalibrationError> {
        let s = std::fs::read_to_string(path).map_err(|source| CalibrationError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml_str(&s).map_err(|e| match e {
            CalibrationError::CorpusInvalid { .. } => e,
            CalibrationError::TomlParse { source, .. } => CalibrationError::TomlParse {
                path: path.to_path_buf(),
                source,
            },
            other => other,
        })
    }

    pub fn from_toml_str(s: &str) -> Result<Self, CalibrationError> {
        let corpus: CalibrationCorpus =
            toml::from_str(s).map_err(|source| CalibrationError::TomlParse {
                path: Default::default(),
                source,
            })?;
        corpus.validate()?;
        Ok(corpus)
    }

    fn validate(&self) -> Result<(), CalibrationError> {
        // Drift source — structural checks only. Existence of the trace
        // file is *not* enforced at load time so the corpus stays parseable
        // on hosts that don't have the trace handy.
        let trace = self.drift_source.workload_trace_path.as_os_str();
        if trace.is_empty() {
            return Err(CalibrationError::CorpusInvalid {
                reason: "workload_trace_path is empty".into(),
            });
        }
        if self.drift_source.n_drift_prompts == 0 {
            return Err(CalibrationError::CorpusInvalid {
                reason: "n_drift_prompts must be > 0".into(),
            });
        }

        for (i, k) in self.kernel_samples.iter().enumerate() {
            if k.shape.is_empty() {
                return Err(CalibrationError::CorpusInvalid {
                    reason: format!("kernel_samples[{i}].shape is empty"),
                });
            }
            if k.repeats == 0 {
                return Err(CalibrationError::CorpusInvalid {
                    reason: format!("kernel_samples[{i}].repeats must be > 0"),
                });
            }
        }
        Ok(())
    }
}
