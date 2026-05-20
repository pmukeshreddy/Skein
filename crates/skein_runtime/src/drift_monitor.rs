//! Workload-drift detection — the decision half of "drift → re-search".
//!
//! The compiler picks a plan for the *compile-time* workload (the trace given
//! to `skein extract`). If the live request mix drifts far enough from that
//! trace, the plan may no longer be SLO-optimal and the system should re-search
//! + recompile + hot-swap (Loop 3 in the README). The hot-swap mechanism
//! already exists; what was missing is the *automated decision* of when to fire
//! it. This module is that decision: build a length distribution from the live
//! requests, compare it to the compile-time distribution via KL divergence, and
//! flag drift when KL exceeds [`Slo::recompile_drift_threshold_kl`].
//!
//! It is pure + deterministic (no GPU): a control loop samples
//! `ProfileHooks::recent_traces`, maps to output lengths, and calls
//! [`DriftMonitor::assess`]; on `drifted` it kicks the (GPU-side) re-extract +
//! recompile + hot-swap.

use skein_ir::workload::Workload;

/// Upper edges of the sequence-length histogram buckets. A length `l` lands in
/// the first bucket whose edge is `>= l`; lengths past the last edge share the
/// overflow bucket — so there are `EDGES.len() + 1` buckets.
const EDGES: [u32; 9] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
/// Laplace smoothing added to every bucket so the distribution has no zeros
/// (keeps KL finite when a bucket is empty in one distribution but not the
/// other).
const SMOOTHING: f64 = 1.0;
/// Below this many live samples, drift is not assessed (too noisy).
const MIN_SAMPLES: usize = 32;

fn bucket(len: u32) -> usize {
    EDGES.iter().position(|&e| len <= e).unwrap_or(EDGES.len())
}

/// A normalized sequence-length distribution over the fixed buckets.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkloadProfile {
    /// Probabilities, summing to 1, length `EDGES.len() + 1`.
    buckets: Vec<f64>,
}

impl WorkloadProfile {
    /// Smoothed, normalized histogram of `lengths`.
    pub fn from_lengths(lengths: &[u32]) -> Self {
        let mut counts = vec![SMOOTHING; EDGES.len() + 1];
        for &l in lengths {
            counts[bucket(l)] += 1.0;
        }
        let total: f64 = counts.iter().sum();
        for c in &mut counts {
            *c /= total;
        }
        Self { buckets: counts }
    }

    /// KL(self ‖ other) = Σ pᵢ·ln(pᵢ/qᵢ). Both are smoothed, so it is finite.
    pub fn kl_to(&self, other: &WorkloadProfile) -> f64 {
        self.buckets
            .iter()
            .zip(other.buckets.iter())
            .map(|(&p, &q)| if p > 0.0 { p * (p / q).ln() } else { 0.0 })
            .sum()
    }
}

/// Outcome of a drift check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftAssessment {
    /// KL divergence of the live distribution from the compile-time baseline.
    pub kl: f64,
    /// `true` when `kl` exceeds the threshold and enough samples were seen —
    /// the signal to re-search + recompile + hot-swap.
    pub drifted: bool,
    /// Number of live samples assessed.
    pub samples: usize,
}

/// Compares the live workload against the compile-time baseline.
#[derive(Debug, Clone)]
pub struct DriftMonitor {
    baseline: WorkloadProfile,
    threshold_kl: f64,
}

impl DriftMonitor {
    pub fn new(baseline: WorkloadProfile, threshold_kl: f64) -> Self {
        Self {
            baseline,
            threshold_kl,
        }
    }

    /// Baseline from the compile-time trace (output-token lengths), threshold
    /// from its SLO (`recompile_drift_threshold_kl`).
    pub fn from_workload(workload: &Workload) -> Self {
        let lengths: Vec<u32> = workload.requests.iter().map(|r| r.output_tokens).collect();
        Self::new(
            WorkloadProfile::from_lengths(&lengths),
            workload.slo.recompile_drift_threshold_kl,
        )
    }

    pub fn threshold(&self) -> f64 {
        self.threshold_kl
    }

    /// Assess live output-token `lengths` (e.g. from `ProfileHooks::recent_traces`).
    /// Below [`MIN_SAMPLES`] the result is never `drifted` (too noisy to act on).
    pub fn assess(&self, lengths: &[u32]) -> DriftAssessment {
        let live = WorkloadProfile::from_lengths(lengths);
        let kl = live.kl_to(&self.baseline);
        DriftAssessment {
            kl,
            drifted: lengths.len() >= MIN_SAMPLES && kl > self.threshold_kl,
            samples: lengths.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rep(value: u32, n: usize) -> Vec<u32> {
        vec![value; n]
    }

    #[test]
    fn identical_distribution_has_zero_kl_and_no_drift() {
        let base = WorkloadProfile::from_lengths(&rep(128, 200));
        let monitor = DriftMonitor::new(base, 0.05);
        let a = monitor.assess(&rep(128, 200));
        assert!(a.kl < 1e-9, "kl={}", a.kl);
        assert!(!a.drifted);
    }

    #[test]
    fn large_length_shift_flags_drift() {
        // Baseline: short outputs (~64). Live: long outputs (~4096) → big KL.
        let base = WorkloadProfile::from_lengths(&rep(64, 500));
        let monitor = DriftMonitor::new(base, 0.05);
        let a = monitor.assess(&rep(4096, 200));
        assert!(a.kl > 0.05, "kl={}", a.kl);
        assert!(a.drifted);
    }

    #[test]
    fn too_few_samples_never_drifts() {
        let base = WorkloadProfile::from_lengths(&rep(64, 500));
        let monitor = DriftMonitor::new(base, 0.05);
        // Only 4 samples, even though the shape is very different.
        let a = monitor.assess(&rep(4096, 4));
        assert!(!a.drifted, "must not act on {} samples", a.samples);
    }

    #[test]
    fn from_workload_uses_slo_threshold() {
        let wl = Workload {
            slo: skein_ir::workload::Slo {
                ttft_p95_ms: 500,
                tpot_p95_ms: 50,
                max_accuracy_drift: 0.01,
                recompile_drift_threshold_kl: 0.123,
            },
            requests: vec![skein_ir::workload::RequestRecord {
                prompt_tokens: 10,
                output_tokens: 64,
                arrival_ms: 0,
            }],
        };
        let monitor = DriftMonitor::from_workload(&wl);
        assert_eq!(monitor.threshold(), 0.123);
    }

    #[test]
    fn bucketing_edges() {
        assert_eq!(bucket(1), 0);
        assert_eq!(bucket(16), 0);
        assert_eq!(bucket(17), 1);
        assert_eq!(bucket(5000), EDGES.len()); // overflow bucket
    }
}
