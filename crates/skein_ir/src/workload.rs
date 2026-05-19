//! Workload trace parser. The trace is a JSONL stream:
//!
//! * Line 1 — an object containing an `slo` field, the binding SLO header.
//! * Lines 2..N — per-request records `{prompt_tokens, output_tokens, arrival_ms}`.
//!
//! Real traces (ShareGPT, anonymized production captures) feed this parser.
//! The spec forbids synthetic traces; the tests use a tiny ShareGPT-derived
//! fixture, not RNG output.
//!
//! Note on percentiles: the SLO field names use `_p95_` because that is what
//! the binding admission-control thresholds are gated on. The validation
//! benchmark (`skein bench`) reports P99 as a separate metric on top of the
//! same trace; the two coexist.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::WorkloadError;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Slo {
    pub ttft_p95_ms: u32,
    pub tpot_p95_ms: u32,
    /// KL divergence ceiling vs the bf16 HF reference, on a per-request
    /// basis. The drift-compliance metric measures the fraction of requests
    /// that meet this.
    pub max_accuracy_drift: f64,
    /// When workload-distribution KL vs the compile-time trace exceeds this,
    /// the runtime triggers a background recompile + hot-swap. Distinct from
    /// `max_accuracy_drift` — that gates per-request quality, this gates
    /// recompile cadence.
    pub recompile_drift_threshold_kl: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestRecord {
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub arrival_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Workload {
    pub slo: Slo,
    pub requests: Vec<RequestRecord>,
}

/// Just the SLO header — the first line of the trace.
#[derive(Debug, Clone, Copy, Deserialize)]
struct SloHeader {
    slo: Slo,
}

impl Workload {
    pub fn from_jsonl_str(s: &str) -> Result<Self, WorkloadError> {
        // Skip blank lines / lines that are only whitespace. Real trace
        // captures occasionally have trailing newlines.
        let mut iter = s.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());

        let (header_lineno, header_line) = iter.next().ok_or(WorkloadError::Empty)?;
        let header: SloHeader = serde_json::from_str(header_line).map_err(|source| {
            // If the first non-blank line is JSON but has no `slo` field, we
            // still want a clear message — serde_json will surface a
            // "missing field `slo`" error and we wrap it. If it's not even
            // JSON, we report a parse error at that line.
            if header_line.trim_start().starts_with('{') {
                WorkloadError::MissingSlo
            } else {
                WorkloadError::Json {
                    line: header_lineno + 1,
                    source,
                }
            }
        })?;

        let mut requests = Vec::new();
        let mut prev_arrival: u64 = 0;
        for (lineno, line) in iter {
            let r: RequestRecord =
                serde_json::from_str(line).map_err(|source| WorkloadError::Json {
                    line: lineno + 1,
                    source,
                })?;
            if r.arrival_ms < prev_arrival {
                return Err(WorkloadError::NonMonotonicArrival {
                    line: lineno + 1,
                    arrival: r.arrival_ms,
                    previous: prev_arrival,
                });
            }
            prev_arrival = r.arrival_ms;
            requests.push(r);
        }

        Ok(Workload {
            slo: header.slo,
            requests,
        })
    }

    pub fn from_jsonl_file(path: &Path) -> Result<Self, WorkloadError> {
        let s = std::fs::read_to_string(path).map_err(|source| WorkloadError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_jsonl_str(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"slo": {"ttft_p95_ms": 500, "tpot_p95_ms": 50, "max_accuracy_drift": 0.01, "recompile_drift_threshold_kl": 0.05}}
{"prompt_tokens": 47, "output_tokens": 132, "arrival_ms": 0}
{"prompt_tokens": 312, "output_tokens": 89, "arrival_ms": 234}
"#;

    #[test]
    fn parses_canonical_trace() {
        let w = Workload::from_jsonl_str(SAMPLE).unwrap();
        assert_eq!(w.slo.ttft_p95_ms, 500);
        assert_eq!(w.slo.tpot_p95_ms, 50);
        assert!((w.slo.max_accuracy_drift - 0.01).abs() < 1e-9);
        assert_eq!(w.requests.len(), 2);
        assert_eq!(w.requests[0].prompt_tokens, 47);
        assert_eq!(w.requests[1].arrival_ms, 234);
    }

    #[test]
    fn rejects_empty_trace() {
        assert!(matches!(
            Workload::from_jsonl_str(""),
            Err(WorkloadError::Empty)
        ));
        assert!(matches!(
            Workload::from_jsonl_str("\n\n  \n"),
            Err(WorkloadError::Empty)
        ));
    }

    #[test]
    fn rejects_missing_slo() {
        let bad = r#"{"oops": true}
{"prompt_tokens": 1, "output_tokens": 1, "arrival_ms": 0}
"#;
        assert!(matches!(
            Workload::from_jsonl_str(bad),
            Err(WorkloadError::MissingSlo)
        ));
    }

    #[test]
    fn rejects_non_monotonic_arrival() {
        let bad = r#"{"slo": {"ttft_p95_ms": 500, "tpot_p95_ms": 50, "max_accuracy_drift": 0.01, "recompile_drift_threshold_kl": 0.05}}
{"prompt_tokens": 1, "output_tokens": 1, "arrival_ms": 100}
{"prompt_tokens": 1, "output_tokens": 1, "arrival_ms": 50}
"#;
        let err = Workload::from_jsonl_str(bad).unwrap_err();
        match err {
            WorkloadError::NonMonotonicArrival {
                line,
                arrival,
                previous,
            } => {
                assert_eq!(line, 3);
                assert_eq!(arrival, 50);
                assert_eq!(previous, 100);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_garbage_line() {
        let bad = "this is not JSON\n";
        assert!(matches!(
            Workload::from_jsonl_str(bad),
            Err(WorkloadError::Json { line: 1, .. })
        ));
    }
}
