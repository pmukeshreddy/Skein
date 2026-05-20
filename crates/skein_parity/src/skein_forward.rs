//! `SkeinForward` — the *artifact-side* forward pass. [`RealSkeinForward`]
//! loads a compiled [`SkeinArtifact`], rebuilds its per-device runtime
//! segments, and drives them — together with the collective backend — through
//! one forward step, capturing per-layer activations and final logits.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ParityError;
use skein_compile::CollectiveExecutor;
use skein_compile::{
    ComputeRuntime, DEFAULT_SEARCH_BUDGET, NativeComputeRuntime, SkeinArtifact, TopologyExecutor,
    load_runtime_segments,
};
use skein_runtime::InProcessCollective;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkeinOutput {
    pub per_layer_activations: Vec<Vec<f32>>,
    pub final_logits: Vec<f32>,
}

pub trait SkeinForward {
    fn forward_with_hooks(&mut self, tokens: &[u32]) -> Result<SkeinOutput, ParityError>;
}

pub struct RealSkeinForward {
    artifact: SkeinArtifact,
    runtimes: Vec<Vec<skein_compile::RuntimeSegment>>,
    collectives: Box<dyn CollectiveExecutor>,
}

impl RealSkeinForward {
    pub fn load_native(artifact_dir: &Path) -> Result<Self, ParityError> {
        Self::load_with_runtime::<NativeComputeRuntime>(artifact_dir, DEFAULT_SEARCH_BUDGET)
    }

    pub fn load_with_runtime<R: ComputeRuntime + 'static>(
        artifact_dir: &Path,
        search_budget: usize,
    ) -> Result<Self, ParityError> {
        let artifact = SkeinArtifact::load(artifact_dir)?;
        let runtimes = load_runtime_segments::<R>(&artifact, search_budget)?;
        let collectives: Box<dyn CollectiveExecutor> =
            Box::new(InProcessCollective::new(artifact.devices.len()));
        Ok(Self {
            artifact,
            runtimes,
            collectives,
        })
    }

    pub fn artifact(&self) -> &SkeinArtifact {
        &self.artifact
    }
}

impl SkeinForward for RealSkeinForward {
    fn forward_with_hooks(&mut self, tokens: &[u32]) -> Result<SkeinOutput, ParityError> {
        let (per_layer_activations, final_logits) = TopologyExecutor::new(
            self.runtimes.as_mut_slice(),
            self.collectives.as_ref(),
            &self.artifact.sequencing,
        )
        .execute_with_hooks(tokens)?;
        Ok(SkeinOutput {
            per_layer_activations,
            final_logits,
        })
    }
}
