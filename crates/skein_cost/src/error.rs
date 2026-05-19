//! Error type for the cost model. One `thiserror` enum; callers can match on
//! the exact failure (missing constant, no topology path, etc.) rather than
//! collapsing into `anyhow`.

use std::path::PathBuf;

use skein_ir::types::Dtype;

use crate::collectives::CollectiveKind;
use crate::compute::OpKind;

#[derive(Debug, thiserror::Error)]
pub enum CostError {
    #[error("could not read cost constants at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cost constants TOML is invalid: {0}")]
    Toml(#[from] toml::de::Error),

    #[error(
        "cost constants have no peak_tflops table for device kind {kind:?}; \
         add a [peak_tflops.{kind}] section"
    )]
    UnknownDeviceKind { kind: String },

    #[error(
        "cost constants have no efficiency table for op kind {op:?}; \
         add an [efficiency.{}] section",
        op.efficiency_key()
    )]
    UnknownEfficiencyOp { op: OpKind },

    #[error("cost constants have no efficiency entry for ({op:?}, {dtype:?})")]
    MissingEfficiencyDtype { op: OpKind, dtype: Dtype },

    #[error("cost constants have no peak_tflops entry for ({kind}, {dtype:?})")]
    MissingPeakTflopsDtype { kind: String, dtype: Dtype },

    #[error("topology has no path between device {from} and device {to}")]
    NoPath { from: u32, to: u32 },

    #[error("collective {kind:?} has {n} participants; needs at least 2")]
    DegenerateCollective { kind: CollectiveKind, n: usize },

    #[error("device index {idx} out of range; cluster has {total} devices")]
    DeviceOutOfRange { idx: u32, total: u32 },

    #[error(
        "Plan dtype_map has {actual} per-block entries but the IR has \
         {expected} decoder blocks"
    )]
    DtypeMapShape { expected: usize, actual: usize },
}
