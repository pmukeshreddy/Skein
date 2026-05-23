//! Inner DP — for one global config, find the optimal per-block dtype map.
//!
//! State: `(block_idx, memory_bucket, drift_bucket)`.
//!
//! Transitions: for each block, try every `(weight, activation, kv_cache)`
//! combo. Memory cost = decoder-block weight + KV bytes at this combo,
//! quantized into the budget's bucket grid. Drift cost = the combo's drift
//! contribution from the `DriftTable`, quantized similarly. Compute cost (a
//! continuous `f64`) is summed in `dp[b][m][d]`.
//!
//! Backtrack from the lowest-cost finishing cell to recover the dtype map.
//!
//! ## Combo prune
//!
//! The full 6 × 6 × 6 combo set (216 per block) would make the DP slow.
//! Skein restricts the search to dtypes that actually move the cost/memory
//! frontier:
//!
//! - weights: bf16, fp8_e4m3, int8, int4 (4)
//! - activations: bf16, fp8_e4m3 (2)
//! - kv_cache: bf16, fp8_e4m3, int8 (3)
//!
//! ⇒ 24 combos per block. The dropped dtypes either duplicate another
//! choice's behaviour (`fp16` ≈ `bf16` on H100 tensor cores) or are not
//! supported in the activation/KV path of the runtime (int4 activations).
//! Documented in `docs/search_algorithms.md`.

use skein_cost::cluster::Placement;
use skein_cost::memory::{block_kv_bytes, block_weight_bytes};
use skein_cost::{Cluster, CostModel, WorkloadCtx};
use skein_ir::ir::Graph;
use skein_ir::plan::{DtypeMap, PerLayerDtype};
use skein_ir::types::Dtype;

use crate::budgets::Budgets;
use crate::candidate::GlobalConfig;
use crate::drift_table::DriftTable;
use crate::error::ExtractError;

/// Dtype combo the DP enumerates per block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DtypeCombo {
    pub weight: Dtype,
    pub activation: Dtype,
    pub kv_cache: Dtype,
}

impl From<DtypeCombo> for PerLayerDtype {
    fn from(c: DtypeCombo) -> Self {
        PerLayerDtype {
            weight: c.weight,
            activation: c.activation,
            kv_cache: c.kv_cache,
        }
    }
}

/// Restricted dtype sets the DP considers — see module docs for why.
pub const WEIGHT_DTYPES: &[Dtype] = &[Dtype::Bf16, Dtype::Fp8E4m3, Dtype::Int8, Dtype::Int4];
pub const ACTIVATION_DTYPES: &[Dtype] = &[Dtype::Bf16, Dtype::Fp8E4m3];
pub const KV_DTYPES: &[Dtype] = &[Dtype::Bf16, Dtype::Fp8E4m3, Dtype::Int8];

/// All `(weight × activation × kv)` combos in a stable order. 24 entries.
pub fn all_combos() -> Vec<DtypeCombo> {
    let mut out =
        Vec::with_capacity(WEIGHT_DTYPES.len() * ACTIVATION_DTYPES.len() * KV_DTYPES.len());
    for &w in WEIGHT_DTYPES {
        for &a in ACTIVATION_DTYPES {
            for &k in KV_DTYPES {
                out.push(DtypeCombo {
                    weight: w,
                    activation: a,
                    kv_cache: k,
                });
            }
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct InnerResult {
    /// Sum of per-block compute time at the chosen dtypes, in microseconds.
    pub cost_us: f64,
    pub dtype_map: DtypeMap,
}

/// Run the inner DP. Returns `DpInfeasible` if no combo sequence fits both
/// budgets.
pub fn layer_dtype_dp(
    ir: &Graph,
    cost_model: &CostModel,
    cluster: &Cluster,
    global: &GlobalConfig,
    budgets: Budgets,
    drift_table: &DriftTable,
) -> Result<InnerResult, ExtractError> {
    let constants = cost_model.constants();
    let mem_buckets = constants.dp.memory_buckets as usize;
    let drift_buckets = constants.dp.drift_buckets as usize;
    let num_blocks = ir.meta.num_layers;
    let placement = Placement {
        tp: global.parallelism.tp,
        pp: global.parallelism.pp,
        ep: global.parallelism.ep,
    };
    let wl = WorkloadCtx {
        batch: global.batch.max_batch(),
        seq_len: 1,
        kv_len: constants.representative_workload.decode_kv_tokens,
    };

    // Bytes/drift per bucket. If budget is zero either way, the global is
    // infeasible — every combo has positive delta.
    if budgets.memory_bytes == 0 {
        return Err(ExtractError::DpInfeasible {
            memory_bytes_budget: 0,
            drift_budget: budgets.drift,
        });
    }
    let bytes_per_mem_bucket = budgets.memory_bytes.div_ceil(mem_buckets as u64).max(1);
    // Drift may legitimately be zero across the row (all-bf16 picks); guard
    // against div-by-zero by treating a zero drift budget as "no drift
    // allowed" — only zero-drift transitions are allowed.
    let drift_per_bucket = (budgets.drift / drift_buckets as f64).max(f64::MIN_POSITIVE);

    // Combo set + per-block precomputed deltas. We compute weight/KV bytes
    // and drift contributions once per (block, combo) pair; only the
    // compute cost is queried per cell.
    let combos = all_combos();
    let mut combo_mem: Vec<Vec<u64>> = vec![vec![0; combos.len()]; num_blocks];
    let mut combo_drift: Vec<Vec<f64>> = vec![vec![0.0; combos.len()]; num_blocks];
    let mut combo_cost: Vec<Vec<f64>> = vec![vec![0.0; combos.len()]; num_blocks];
    // Under pipeline parallelism the decoder blocks are *distributed* across
    // `pp` stages (each device hosts ~num_blocks/pp whole blocks), not split
    // within a block like tp/ep. `block_weight_bytes`/`block_kv_bytes` report a
    // whole block's footprint on its host device; dividing by `pp` here turns
    // the all-blocks sum into the per-device total. pp=1 is a no-op. This is an
    // even-split approximation for the feasibility pre-filter; the authoritative
    // per-device check is `peak_memory_bytes` (PP-aware via `block_to_stage`).
    let pp = placement.pp.max(1) as u64;
    for b in 0..num_blocks {
        for (ci, c) in combos.iter().enumerate() {
            let w_bytes = block_weight_bytes(b, ir, c.weight, placement) / pp;
            let k_bytes = block_kv_bytes(ir, c.kv_cache, placement, &wl, global.kv_shard) / pp;
            combo_mem[b][ci] = w_bytes.saturating_add(k_bytes);
            combo_drift[b][ci] =
                drift_table.block_contribution(b, c.weight, c.activation, c.kv_cache);
            // Compute time depends only on weight dtype + placement.
            combo_cost[b][ci] = cost_model
                .block_compute_time(b, ir, c.weight, placement, cluster, &wl, /*device=*/ 0)?;
        }
    }

    // dp[b][m][d] = min cumulative compute time to reach this state.
    // back[b][m][d] = (chosen combo index at block b-1, prev m, prev d).
    let m_dim = mem_buckets + 1;
    let d_dim = drift_buckets + 1;
    let mut dp: Vec<f64> = vec![f64::INFINITY; (num_blocks + 1) * m_dim * d_dim];
    let mut back: Vec<Option<Backptr>> = vec![None; (num_blocks + 1) * m_dim * d_dim];
    let idx = |b: usize, m: usize, d: usize| -> usize { ((b * m_dim) + m) * d_dim + d };
    dp[idx(0, 0, 0)] = 0.0;

    for b in 0..num_blocks {
        for m in 0..m_dim {
            for d in 0..d_dim {
                let base = dp[idx(b, m, d)];
                if !base.is_finite() {
                    continue;
                }
                for ci in 0..combos.len() {
                    let mem_delta = combo_mem[b][ci];
                    let drift_delta = combo_drift[b][ci];
                    // Infinite drift = combo excluded by the drift table.
                    if !drift_delta.is_finite() {
                        continue;
                    }
                    let mem_step = (mem_delta.div_ceil(bytes_per_mem_bucket)) as usize;
                    let drift_step =
                        ((drift_delta / drift_per_bucket).ceil() as i64).max(0) as usize;
                    let m2 = m + mem_step;
                    let d2 = d + drift_step;
                    if m2 > mem_buckets || d2 > drift_buckets {
                        continue;
                    }
                    let new_cost = base + combo_cost[b][ci];
                    let i2 = idx(b + 1, m2, d2);
                    if new_cost < dp[i2] {
                        dp[i2] = new_cost;
                        back[i2] = Some(Backptr {
                            combo_idx: ci as u16,
                            prev_m: m as u16,
                            prev_d: d as u16,
                        });
                    }
                }
            }
        }
    }

    // Find the minimum-cost finishing cell.
    let mut best: Option<(f64, usize, usize)> = None;
    for m in 0..m_dim {
        for d in 0..d_dim {
            let c = dp[idx(num_blocks, m, d)];
            if c.is_finite() && best.as_ref().is_none_or(|(bc, _, _)| c < *bc) {
                best = Some((c, m, d));
            }
        }
    }

    let (cost_us, end_m, end_d) = match best {
        Some(t) => t,
        None => {
            return Err(ExtractError::DpInfeasible {
                memory_bytes_budget: budgets.memory_bytes,
                drift_budget: budgets.drift,
            });
        }
    };

    // Backtrack to recover the per-block combo sequence.
    let mut chosen: Vec<DtypeCombo> = vec![
        DtypeCombo {
            weight: Dtype::Bf16,
            activation: Dtype::Bf16,
            kv_cache: Dtype::Bf16,
        };
        num_blocks
    ];
    let mut cur_m = end_m;
    let mut cur_d = end_d;
    for b in (0..num_blocks).rev() {
        let bp = back[idx(b + 1, cur_m, cur_d)].expect(
            "backtrack reached a state with finite cost but no predecessor; \
             this can only happen if the dp table was mutated mid-flight",
        );
        chosen[b] = combos[bp.combo_idx as usize];
        cur_m = bp.prev_m as usize;
        cur_d = bp.prev_d as usize;
    }
    debug_assert_eq!(cur_m, 0);
    debug_assert_eq!(cur_d, 0);

    Ok(InnerResult {
        cost_us,
        dtype_map: DtypeMap {
            per_layer: chosen.into_iter().map(PerLayerDtype::from).collect(),
        },
    })
}

#[derive(Debug, Clone, Copy)]
struct Backptr {
    combo_idx: u16,
    prev_m: u16,
    prev_d: u16,
}
