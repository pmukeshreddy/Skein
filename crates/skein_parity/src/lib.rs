//! `skein_parity` — the post-compile parity gate.
//!
//! Compares a candidate Skein artifact's per-layer activations + final
//! logits against a Skein-bf16 reference artifact and decides pass/fail per
//! the workload SLO. On failure, records the offending `(layer, component, dtype,
//! measured_mse)` to the model's drift table and signals the caller (the
//! compile orchestrator) to re-invoke `skein_extract::extract_plan` — the
//! Plan that just lost parity is now predicted as drift-violating and
//! excluded from the next search.
//!
//! ## Phase A scope
//!
//! - Comparison math: `mse`, `kl_divergence` (numerically stable via
//!   log-softmax with max-shift), tolerance lookup.
//! - Report structures: `ParityReport`, `PerPromptReport`,
//!   `FailingLayerReport`. Round-trip cleanly through `serde_json`.
//! - Failure handling: `drift_update::update_drift_table_on_failure` with
//!   monotonic (never-decrease) refinement protocol.
//! - Stubbed I/O: `MockReference` reads pre-recorded activations from JSON;
//!   `PhaseAStub` wraps it and injects deterministic per-layer drift to
//!   exercise the verification flow end-to-end on Mac.
//!
//! The external `PythonSubprocessReference` path remains available for
//! verify-only architecture checks against `transformers`; the production
//! parity gate uses real `SkeinForward` implementations against reloadable
//! `SkeinArtifact`s. The Mac path uses
//! `NativeComputeRuntime` and in-process collectives; CUDA-specific pieces
//! remain feature-gated.

pub mod comparison;
pub mod drift_update;
pub mod error;
pub mod reference;
pub mod report;
pub mod skein_forward;
pub mod tolerance;
pub mod verify;

pub use error::ParityError;
pub use reference::{
    HFReference, MockReference, PythonSubprocessReference, ReferenceDtype, ReferenceOutput,
};
pub use report::{FailingLayerReport, ParityReport, PerPromptReport};
pub use skein_forward::{PhaseAStub, RealSkeinForward, SkeinForward, SkeinOutput};
pub use tolerance::{ToleranceTable, tolerance_for_layer};
pub use verify::{tokenize_prompt_bytes, verify_plan, verify_skein_pair};
