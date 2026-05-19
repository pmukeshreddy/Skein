//! `SkeinForward` — the *artifact-side* forward pass. Phase A ships
//! `PhaseAStub` which derives its output from a `MockReference` plus a
//! configurable per-layer additive drift. Phase B will provide the real
//! implementation that drives the compiled `luminal::Graph` runtimes
//! through one forward step.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ParityError;
use crate::reference::{MockReference, ReferenceOutput};
use skein_compile::CollectiveExecutor;
use skein_compile::{
    ComputeRuntime, DEFAULT_SEARCH_BUDGET, NativeComputeRuntime, SkeinArtifact, TopologyExecutor,
    load_runtime_segments,
};
use skein_runtime::MockCollective;

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
            Box::new(MockCollective::new(artifact.devices.len()));
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

/// Phase A stub. Reads the reference activations for `tokens` (via a
/// `MockReference`) and adds a deterministic per-layer offset — so the
/// per-layer MSE between reference and Skein output is exactly the square
/// of the offset, times the activation length. That lets tests dial drift
/// up and down with predictable magnitudes.
///
/// Final-logit drift is a separate uniform offset so KL-divergence tests
/// can be tuned independently of the per-layer MSE.
pub struct PhaseAStub {
    reference: MockReference,
    layer_offsets: Vec<f32>,
    final_logit_offset: f32,
}

impl PhaseAStub {
    pub fn new(reference: MockReference) -> Self {
        Self {
            reference,
            layer_offsets: Vec::new(),
            final_logit_offset: 0.0,
        }
    }

    /// Set the per-layer additive offset. The vector's length should match
    /// the model's number of decoder blocks; shorter vectors are
    /// zero-padded at the tail.
    pub fn with_layer_offsets(mut self, offsets: Vec<f32>) -> Self {
        self.layer_offsets = offsets;
        self
    }

    /// Set a uniform additive offset on the final logits. Used to dial
    /// final-KL drift independently of per-layer MSE.
    pub fn with_final_logit_offset(mut self, offset: f32) -> Self {
        self.final_logit_offset = offset;
        self
    }
}

impl SkeinForward for PhaseAStub {
    fn forward_with_hooks(&mut self, tokens: &[u32]) -> Result<SkeinOutput, ParityError> {
        let ReferenceOutput {
            mut per_layer_activations,
            mut final_logits,
        } = self.reference.lookup(tokens)?;
        for (i, layer) in per_layer_activations.iter_mut().enumerate() {
            let offset = self.layer_offsets.get(i).copied().unwrap_or(0.0);
            if offset == 0.0 {
                continue;
            }
            for v in layer.iter_mut() {
                *v += offset;
            }
        }
        if self.final_logit_offset != 0.0 {
            for v in final_logits.iter_mut() {
                *v += self.final_logit_offset;
            }
        }
        Ok(SkeinOutput {
            per_layer_activations,
            final_logits,
        })
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
