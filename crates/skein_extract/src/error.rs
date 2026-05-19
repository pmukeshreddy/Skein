//! Error type for `skein_extract`.

use std::path::PathBuf;

use crate::constraints::RejectReason;

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("could not read drift table at {path}: {source}")]
    DriftIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("drift table TOML is invalid: {0}")]
    DriftToml(#[from] toml::de::Error),

    #[error(
        "drift table has no default entry for ({component:?}, {dtype:?}); \
         every component/dtype pair must appear under [default.<component>]"
    )]
    DriftMissingDefault {
        component: skein_ir::types::Component,
        dtype: skein_ir::types::Dtype,
    },

    #[error("drift table [layer.{key}.…] key is not a non-negative integer")]
    DriftBadLayerKey { key: String },

    #[error("cost model error: {0}")]
    Cost(#[from] skein_cost::CostError),

    #[error(
        "inner DP is infeasible: no per-layer dtype map satisfies the \
         memory ({memory_bytes_budget} B) and drift ({drift_budget}) budgets \
         the outer global config left"
    )]
    DpInfeasible {
        memory_bytes_budget: u64,
        drift_budget: f64,
    },

    #[error(
        "no feasible plan found after evaluating {raw_candidates} raw \
         outer candidates ({dp_infeasible} reached DP-infeasible); the \
         dominant rejection was {dominant_rejection:?}"
    )]
    NoFeasiblePlan {
        raw_candidates: u64,
        dp_infeasible: u64,
        dominant_rejection: Option<RejectReason>,
    },
}
