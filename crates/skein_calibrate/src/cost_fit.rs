//! Thin re-export so callers can reach the kernel aggregation under a
//! cost-focused name. The aggregation logic itself lives in
//! `measurement::aggregate_kernel_measurements`.

pub use crate::measurement::aggregate_kernel_measurements;
