//! SLO-aware admission decision.

use skein_cost::CostConstants;
use skein_ir::workload::Slo;

use crate::types::IncomingRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// The request's prompt is so long that even alone it would exceed
    /// the TTFT SLO at the estimator's prefill rate.
    PromptTooLong,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AdmissionDecision {
    Admit,
    /// Caller should re-enqueue and reconsider at or after `until_ms`.
    /// The runtime usually pulls the queued request back on the next step.
    Delay {
        until_ms: u64,
    },
    Reject {
        reason: RejectReason,
    },
}

/// Closed-form estimator using the `[runtime_estimator]` constants.
#[derive(Debug, Clone, Copy)]
pub struct LatencyEstimator {
    pub per_token_decode_us_at_b1: f64,
    pub batch_scaling_exponent: f64,
    pub prefill_per_token_us: f64,
}

impl LatencyEstimator {
    pub fn from_cost_constants(c: &CostConstants) -> Self {
        Self {
            per_token_decode_us_at_b1: c.runtime_estimator.per_token_decode_us_at_b1,
            batch_scaling_exponent: c.runtime_estimator.batch_scaling_exponent,
            prefill_per_token_us: c.runtime_estimator.prefill_per_token_us,
        }
    }

    /// Predicted prefill latency in ms for a `prompt_tokens`-long prompt.
    pub fn predict_prefill_ms(&self, prompt_tokens: u32) -> f64 {
        self.prefill_per_token_us * prompt_tokens as f64 / 1000.0
    }

    /// Predicted per-token decode latency in ms at a batch size of
    /// `concurrent_decodes`. `concurrent_decodes` of zero is treated as 1 —
    /// at least the new request itself runs.
    pub fn predict_tpot_ms(&self, concurrent_decodes: u32) -> f64 {
        let b = (concurrent_decodes.max(1)) as f64;
        self.per_token_decode_us_at_b1 * b.powf(self.batch_scaling_exponent) / 1000.0
    }
}

/// Pure admission predicate. Made pure so tests can drive it without a
/// full `ContinuousBatcher`.
pub fn decide(
    request: &IncomingRequest,
    now_ms: u64,
    max_batch: u32,
    current_inflight: u32,
    slo: &Slo,
    estimator: &LatencyEstimator,
) -> AdmissionDecision {
    // Hard reject: the prompt is so long that ever finishing prefill
    // within TTFT SLO is impossible. No queue position helps.
    let prefill_ms = estimator.predict_prefill_ms(request.prompt_tokens.len() as u32);
    if prefill_ms > slo.ttft_p95_ms as f64 {
        return AdmissionDecision::Reject {
            reason: RejectReason::PromptTooLong,
        };
    }

    // Capacity gate: too many in-flight → delay.
    if current_inflight >= max_batch {
        // Wait at minimum one decode step for any current request to
        // produce a token and free a slot. Tests assert the delay times
        // are physically meaningful (non-zero, finite).
        let one_step_ms = estimator.predict_tpot_ms(current_inflight) as u64;
        return AdmissionDecision::Delay {
            until_ms: now_ms + one_step_ms.max(1),
        };
    }

    // Latency-prediction gate: would admitting now push TPOT past the SLO?
    let projected_inflight = current_inflight + 1;
    let predicted_tpot = estimator.predict_tpot_ms(projected_inflight);
    if predicted_tpot > slo.tpot_p95_ms as f64 {
        let one_step_ms = estimator.predict_tpot_ms(current_inflight) as u64;
        return AdmissionDecision::Delay {
            until_ms: now_ms + one_step_ms.max(1),
        };
    }

    AdmissionDecision::Admit
}
