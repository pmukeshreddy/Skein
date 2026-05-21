//! `skein_extract` — search for the lowest-cost feasible `Plan`.
//!
//! Two-level search:
//!
//! 1. **Outer**: enumerate all global axis configurations (parallelism
//!    placement, KV layout, batching, CUDA Graphs, spec decode, prefix
//!    cache). Discard candidates violating hard constraints (per-dim
//!    divisibility, device count, optimistic memory fit).
//! 2. **Inner DP**: for each surviving global, run a knapsack-style DP over
//!    per-block `(weight, activation, kv_cache)` dtype combos. State is
//!    `(block_idx, memory_bucket, drift_bucket)`. Backtrack to recover the
//!    optimal `DtypeMap`.
//!
//! For each `(global, dtype_map)` pair we compose a `Plan` and score it with
//! `cost_model.total_cost`. The best across all globals wins.
//!
//! See `docs/search_algorithms.md` for the cardinality estimate and the
//! "why enumeration + DP, not ILP" justification.

pub mod budgets;
pub mod candidate;
pub mod constraints;
pub mod disaggregate;
pub mod dp;
pub mod drift_table;
pub mod enumerate;
pub mod error;

pub use candidate::GlobalConfig;
pub use disaggregate::extract_disaggregated_plan;
pub use drift_table::DriftTable;
pub use error::ExtractError;

use std::collections::HashMap;

use skein_cost::{Cluster, Cost, CostModel};
use skein_ir::ir::Graph;
use skein_ir::plan::{DtypeMap, ParallelismPlacement, Plan};
use skein_ir::workload::Workload;

/// The crate's only entry point. Returns the lowest-cost feasible `Plan`.
///
/// On infeasibility (every global was rejected, every DP was infeasible),
/// returns `ExtractError::NoFeasiblePlan` annotated with the constraint
/// counter so callers can diagnose which constraint was the binding one.
/// Debug parallelism pin: returns `true` if `p` should be skipped because it
/// does not match a `SKEIN_FORCE_{TP,PP,EP}` environment override (each unset
/// var matches anything). Lets a reproducer compile a specific placement.
fn force_parallelism_skip(p: &ParallelismPlacement) -> bool {
    let want = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok());
    if let Some(tp) = want("SKEIN_FORCE_TP") {
        if p.tp != tp {
            return true;
        }
    }
    if let Some(pp) = want("SKEIN_FORCE_PP") {
        if p.pp != pp {
            return true;
        }
    }
    if let Some(ep) = want("SKEIN_FORCE_EP") {
        if p.ep != ep {
            return true;
        }
    }
    false
}

pub fn extract_plan(
    ir: &Graph,
    cluster: &Cluster,
    workload: &Workload,
    drift_table: &DriftTable,
    cost_model: &CostModel,
) -> Result<Plan, ExtractError> {
    let mut best: Option<(Cost, Plan)> = None;
    let mut counters = SearchCounters::default();

    // DP-result cache. The inner DP depends on `(placement, max_batch,
    // kv_shard)` — every other field of `GlobalConfig` only affects the
    // outer's `total_cost` ranking, not the DP itself. Caching collapses
    // ~1000 outer candidates into ~50 distinct DP calls on the canonical
    // Mixtral 2× H100 setup. `Option<DtypeMap>` makes infeasible memos
    // explicit (re-attempting them would just re-fail).
    let mut dp_cache: HashMap<DpKey, Option<DtypeMap>> = HashMap::new();

    for global in enumerate::enumerate_global_configs(cluster, ir) {
        counters.raw += 1;
        // Debug override: SKEIN_FORCE_TP/PP/EP pin the parallelism so a specific
        // placement (e.g. tp=2) can be compiled regardless of the cost ranking —
        // used to reproduce a placement-specific bug on a small model.
        if force_parallelism_skip(&global.parallelism) {
            continue;
        }
        if let Some(reason) = constraints::reject(&global, cluster, ir, cost_model) {
            counters.reject(reason);
            continue;
        }
        counters.survivors += 1;

        let key = DpKey {
            placement: global.parallelism,
            max_batch: global.batch.max_batch(),
            kv_shard: global.kv_shard,
        };
        let dtype_map = if let Some(memo) = dp_cache.get(&key) {
            match memo {
                Some(m) => m.clone(),
                None => {
                    counters.dp_infeasible += 1;
                    continue;
                }
            }
        } else {
            let budgets = budgets::budgets_after_global(&global, cluster, workload, ir, cost_model);
            let result = dp::layer_dtype_dp(ir, cost_model, cluster, &global, budgets, drift_table);
            match result {
                Ok(r) => {
                    dp_cache.insert(key, Some(r.dtype_map.clone()));
                    r.dtype_map
                }
                Err(ExtractError::DpInfeasible { .. }) => {
                    dp_cache.insert(key, None);
                    counters.dp_infeasible += 1;
                    continue;
                }
                Err(e) => return Err(e),
            }
        };

        let plan = candidate::compose_plan(global, dtype_map, ir.meta.clone());
        let total = cost_model
            .total_cost(&plan, ir, cluster)
            .map_err(ExtractError::Cost)?;
        counters.scored += 1;

        match &best {
            None => best = Some((total, plan)),
            Some((cur, _)) if total < *cur => best = Some((total, plan)),
            _ => {}
        }
    }

    tracing::info!(
        raw = counters.raw,
        survivors = counters.survivors,
        dp_infeasible = counters.dp_infeasible,
        scored = counters.scored,
        rejected_tp = counters.rejected_tp_divisibility,
        rejected_ep = counters.rejected_ep_divisibility,
        rejected_devcount = counters.rejected_device_count,
        rejected_memory = counters.rejected_global_memory,
        "skein_extract: search complete"
    );

    match best {
        Some((_, plan)) => Ok(plan),
        None => Err(ExtractError::NoFeasiblePlan {
            raw_candidates: counters.raw,
            dp_infeasible: counters.dp_infeasible,
            dominant_rejection: counters.dominant_rejection(),
        }),
    }
}

/// Key into the inner-DP memoization cache. Two outer candidates with the
/// same key produce the same DP result regardless of their other fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DpKey {
    placement: ParallelismPlacement,
    max_batch: u32,
    kv_shard: bool,
}

/// Diagnostic counters surfaced when the search finishes. `Default` zero.
#[derive(Debug, Default, Clone, Copy)]
struct SearchCounters {
    raw: u64,
    survivors: u64,
    dp_infeasible: u64,
    scored: u64,
    rejected_tp_divisibility: u64,
    rejected_ep_divisibility: u64,
    rejected_device_count: u64,
    rejected_global_memory: u64,
}

impl SearchCounters {
    fn reject(&mut self, reason: constraints::RejectReason) {
        match reason {
            constraints::RejectReason::TpDivisibility => self.rejected_tp_divisibility += 1,
            constraints::RejectReason::EpDivisibility => self.rejected_ep_divisibility += 1,
            constraints::RejectReason::DeviceCount => self.rejected_device_count += 1,
            constraints::RejectReason::GlobalMemory => self.rejected_global_memory += 1,
        }
    }

    fn dominant_rejection(self) -> Option<constraints::RejectReason> {
        let opts = [
            (
                self.rejected_tp_divisibility,
                constraints::RejectReason::TpDivisibility,
            ),
            (
                self.rejected_ep_divisibility,
                constraints::RejectReason::EpDivisibility,
            ),
            (
                self.rejected_device_count,
                constraints::RejectReason::DeviceCount,
            ),
            (
                self.rejected_global_memory,
                constraints::RejectReason::GlobalMemory,
            ),
        ];
        opts.iter()
            .max_by_key(|(n, _)| *n)
            .and_then(|(n, r)| if *n > 0 { Some(*r) } else { None })
    }
}
