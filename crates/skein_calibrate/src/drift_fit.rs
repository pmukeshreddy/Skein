//! Thin re-export so callers can reach the drift aggregation under a
//! drift-focused name. The aggregation logic itself lives in
//! `measurement::aggregate_drift_measurements`.

pub use crate::measurement::aggregate_drift_measurements;
