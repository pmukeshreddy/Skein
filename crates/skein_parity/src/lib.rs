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
//! ## Components
//!
//! - Comparison math: `mse`, `kl_divergence` (numerically stable via
//!   log-softmax with max-shift), tolerance lookup.
//! - Report structures: `ParityReport`, `PerPromptReport`,
//!   `FailingLayerReport`. Round-trip cleanly through `serde_json`.
//! - Failure handling: `drift_update::update_drift_table_on_failure` with
//!   the monotonic (never-decrease) refinement protocol.
//! - References: [`PythonSubprocessReference`] runs a `transformers`
//!   subprocess for architecture-level verify checks; the production parity
//!   gate compares two compiled [`RealSkeinForward`] artifacts via
//!   [`verify_skein_pair`].

pub mod comparison;
pub mod drift_update;
pub mod error;
pub mod reference;
pub mod report;
pub mod skein_forward;
pub mod tolerance;
pub mod verify;

pub use error::ParityError;
pub use reference::{HFReference, PythonSubprocessReference, ReferenceDtype, ReferenceOutput};
pub use report::{FailingLayerReport, ParityReport, PerPromptReport};
pub use skein_forward::{RealSkeinForward, SkeinForward, SkeinOutput};
pub use tolerance::{ToleranceTable, tolerance_for_layer};
pub use verify::{tokenize_prompt_bytes, verify_plan, verify_skein_pair};
