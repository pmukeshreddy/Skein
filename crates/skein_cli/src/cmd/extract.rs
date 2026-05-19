//! `skein extract` — Phase A search-only entry point.
//!
//! Loads IR / cluster / workload / drift / cost-model, rejects
//! `--disaggregated` (Phase B), runs `skein_extract::extract_plan`,
//! serializes the chosen `Plan` to JSON, and emits an `ExtractReport`.

use std::collections::BTreeMap;
use std::time::Instant;

use skein_cost::Cost;
use skein_extract::extract_plan;
use skein_ir::types::Dtype;

use crate::cli::{ExtractArgs, OutputFormat};
use crate::error::CliError;
use crate::load::{load_cluster, load_cost_model, load_drift_table, load_ir, load_workload};
use crate::output::{ExtractReport, ReportParallelism, print_report};

pub fn run(args: ExtractArgs, output: OutputFormat) -> Result<(), CliError> {
    if args.disaggregated {
        return Err(CliError::PhaseBOnly {
            what: "P/D disaggregated mode",
            tracking: "Phase B Step 13 (skein compile --disaggregated). \
                       Phase A extract supports only non-disaggregated plans.",
        });
    }

    tracing::info!("skein extract: loading inputs");
    let ir = load_ir(&args.model)?;
    let cluster_spec = load_cluster(&args.cluster)?;
    let workload = load_workload(&args.trace)?;
    let drift_table = load_drift_table(&args.drift)?;
    let cost_model = load_cost_model(&args.cost)?;
    let cluster = skein_cost::Cluster::from_spec(cluster_spec);

    tracing::info!(
        model = %ir.meta.architecture,
        layers = ir.meta.num_layers,
        devices = cluster.num_devices(),
        "skein extract: searching plan space"
    );

    let start = Instant::now();
    let plan = extract_plan(&ir, &cluster, &workload, &drift_table, &cost_model)?;
    let elapsed = start.elapsed();

    tracing::info!(
        wall_ms = elapsed.as_millis() as u64,
        "skein extract: search complete"
    );

    let plan_json = serde_json::to_string_pretty(&plan)?;
    std::fs::write(&args.out, &plan_json)?;

    // Build the report. `plan.content_hash()` returns `Result<_, PlanError>`
    // and is bridged to `CliError::Other` via `anyhow`.
    let plan_hash = plan
        .content_hash()
        .map_err(|e| anyhow::anyhow!("computing plan content hash: {e}"))?;
    let total_cost: Cost = cost_model.total_cost(&plan, &ir, &cluster)?;

    let report = ExtractReport {
        plan_hash: plan_hash.to_hex().to_string(),
        plan_path: args.out.display().to_string(),
        search_wall_ms: elapsed.as_millis() as u64,
        cost_us: total_cost.as_us(),
        parallelism: ReportParallelism {
            tp: plan.parallelism.tp,
            pp: plan.parallelism.pp,
            ep: plan.parallelism.ep,
        },
        dtype_summary: dtype_summary(&plan),
        model: plan.model_meta.architecture.clone(),
    };

    print_report(&report, output)?;
    Ok(())
}

/// Count decoder blocks per weight dtype. `BTreeMap` so iteration is
/// sorted by key, matching the JSON serialization contract.
fn dtype_summary(plan: &skein_ir::plan::Plan) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for entry in &plan.dtype_map.per_layer {
        *counts
            .entry(dtype_key(entry.weight).to_string())
            .or_insert(0) += 1;
    }
    counts
}

fn dtype_key(d: Dtype) -> &'static str {
    match d {
        Dtype::Bf16 => "bf16",
        Dtype::Fp16 => "fp16",
        Dtype::Fp8E4m3 => "fp8_e4m3",
        Dtype::Fp8E5m2 => "fp8_e5m2",
        Dtype::Int8 => "int8",
        Dtype::Int4 => "int4",
    }
}
