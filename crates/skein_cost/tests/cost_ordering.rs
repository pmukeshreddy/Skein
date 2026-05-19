//! Phase-A acceptance tests for `skein_cost`: the cost-function ordering
//! invariants the search algorithm depends on.

mod common;
use common::*;

use skein_cost::bubble::bubble_time;
use skein_cost::cluster::Placement;
use skein_cost::collectives::{Collective, CollectiveKind};
use skein_cost::comm::collectives_on_device;
use skein_cost::compute::kernels_per_step_on_device;
use skein_cost::launch::launch_overhead;
use skein_ir::cluster::ClusterSpec;
use skein_ir::plan::*;
use skein_ir::types::*;

// ---------------------------------------------------------------------------
// Test 1 — compute time roughly halves when TP doubles.
// ---------------------------------------------------------------------------

#[test]
fn compute_time_scales_with_tp() {
    let model = load_cost_model();
    let ir = load_mixtral_ir();

    // Build a 4-device single-node cluster with NVLink between every pair so
    // comm cost is small compared to compute. Each pair gets one NVLink Gen4
    // hop. This factors out comm from the comparison.
    let toml = r#"
num_devices = 4
[[node]]
id               = "node0"
devices          = ["d0", "d1", "d2", "d3"]
device_kind      = "h100_sxm5"
device_memory_gb = 80
[[link]]
endpoints = ["d0", "d1"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d0", "d2"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d0", "d3"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d1", "d2"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d1", "d3"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d2", "d3"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
"#;
    let spec = ClusterSpec::from_toml_str(toml).unwrap();
    let cluster = skein_cost::Cluster::from_spec(spec);
    let wl = skein_cost::WorkloadCtx::decode_step(
        &plan_with(
            ir.meta.clone(),
            1,
            1,
            1,
            Dtype::Bf16,
            Dtype::Bf16,
            Dtype::Bf16,
            BatchPolicy::Continuous { max_batch: 1 },
            CudaGraphsConfig {
                enable: false,
                capture_classes: vec![],
            },
        ),
        model.constants(),
    );

    let compute_for = |tp: u32| -> f64 {
        let plan = plan_with(
            ir.meta.clone(),
            tp,
            1,
            1,
            Dtype::Bf16,
            Dtype::Bf16,
            Dtype::Bf16,
            BatchPolicy::Continuous { max_batch: 1 },
            CudaGraphsConfig {
                enable: false,
                capture_classes: vec![],
            },
        );
        model
            .compute_on_device(&plan, &ir, &cluster, &wl, 0)
            .unwrap()
    };

    let c1 = compute_for(1);
    let c2 = compute_for(2);
    let c4 = compute_for(4);

    // c2 ≈ c1 / 2 within 10% (the small elementwise RmsNorm work doesn't
    // divide by tp, so the ratio is not exact).
    let ratio_12 = c1 / c2;
    assert!(
        (1.8..=2.2).contains(&ratio_12),
        "expected c1/c2 ≈ 2, got {ratio_12}"
    );
    let ratio_24 = c2 / c4;
    assert!(
        (1.8..=2.2).contains(&ratio_24),
        "expected c2/c4 ≈ 2, got {ratio_24}"
    );
}

// ---------------------------------------------------------------------------
// Test 3 — memory overshoot dominates total cost.
// ---------------------------------------------------------------------------

#[test]
fn memory_overshoot_dominates() {
    let model = load_cost_model();
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();

    let oom_plan = plan_with(
        ir.meta.clone(),
        1,
        1,
        1,
        Dtype::Bf16,
        Dtype::Bf16,
        Dtype::Bf16, // TP=1: weights replicated
        BatchPolicy::Continuous { max_batch: 8 },
        CudaGraphsConfig {
            enable: false,
            capture_classes: vec![],
        },
    );
    let fits_plan = plan_with(
        ir.meta.clone(),
        2,
        1,
        1,
        Dtype::Bf16,
        Dtype::Bf16,
        Dtype::Bf16,
        BatchPolicy::Continuous { max_batch: 8 },
        CudaGraphsConfig {
            enable: false,
            capture_classes: vec![],
        },
    );

    let oom = model.total_cost(&oom_plan, &ir, &cluster).unwrap().as_us();
    let fits = model.total_cost(&fits_plan, &ir, &cluster).unwrap().as_us();
    // OOM plan should be at least 1e6 µs worse than any fitting plan, because
    // overshoot_us_per_gb = 1e6 and the overshoot is ≥ 1 GB.
    assert!(
        oom - fits >= 1_000_000.0,
        "OOM plan ({oom} µs) should dwarf fitting plan ({fits} µs)"
    );
}

// ---------------------------------------------------------------------------
// Test 4 — pipeline bubble formula (also covered in unit tests; pinned here
// against `Plan` types to make the formula's contract explicit).
// ---------------------------------------------------------------------------

#[test]
fn pipeline_bubble_formula() {
    let stage = 1234.5_f64;
    let mk = |pp: u32| Plan {
        parallelism: ParallelismPlacement { tp: 1, pp, ep: 1 },
        kv: KVCacheSpec {
            layout: KVLayout::Contiguous,
            kv_sharded: false,
        },
        batching: BatchPolicy::Continuous { max_batch: 1 },
        dtype_map: DtypeMap::uniform(32, Dtype::Bf16),
        execution: ExecutionConfig {
            cuda_graphs: CudaGraphsConfig {
                enable: false,
                capture_classes: vec![],
            },
            spec_decode: SpecDecodeConfig {
                enable: false,
                draft: None,
            },
            prefix_cache: PrefixCacheConfig {
                enable: false,
                reuse_policy: RadixReusePolicy::LruByLastAccess,
            },
        },
        disaggregation: None,
        model_meta: dummy_meta(32),
    };
    assert_eq!(bubble_time(&mk(1), stage), 0.0);
    assert!((bubble_time(&mk(2), stage) - 0.5 * stage).abs() < 1e-9);
    assert!((bubble_time(&mk(4), stage) - 0.75 * stage).abs() < 1e-9);
    assert!((bubble_time(&mk(8), stage) - (7.0 / 8.0) * stage).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Test 5 — CUDA Graphs reduce total cost.
// ---------------------------------------------------------------------------

#[test]
fn cuda_graphs_reduces_total() {
    let model = load_cost_model();
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();

    // Same base plan, vary only the CUDA Graphs toggle. Provide ≥ 4 capture
    // classes to hit the high-coverage row of the table.
    let classes: Vec<CaptureClass> = (1..=4)
        .map(|i| CaptureClass {
            batch_size: 1 << i,
            kv_class: 1,
        })
        .collect();

    let off = plan_with(
        ir.meta.clone(),
        2,
        1,
        1,
        Dtype::Bf16,
        Dtype::Bf16,
        Dtype::Bf16,
        BatchPolicy::Continuous { max_batch: 8 },
        CudaGraphsConfig {
            enable: false,
            capture_classes: vec![],
        },
    );
    let on = plan_with(
        ir.meta.clone(),
        2,
        1,
        1,
        Dtype::Bf16,
        Dtype::Bf16,
        Dtype::Bf16,
        BatchPolicy::Continuous { max_batch: 8 },
        CudaGraphsConfig {
            enable: true,
            capture_classes: classes,
        },
    );

    let off_cost = model.total_cost(&off, &ir, &cluster).unwrap().as_us();
    let on_cost = model.total_cost(&on, &ir, &cluster).unwrap().as_us();
    assert!(
        on_cost < off_cost,
        "CUDA Graphs on ({on_cost}) should beat off ({off_cost})"
    );

    // Approximate magnitude: 0.95 × launch_us_per_kernel × kernels_per_step.
    let kernels = kernels_per_step_on_device(&ir, &off, 0);
    let expected_diff = 0.95 * model.constants().launch_us_per_kernel * kernels as f64;
    let diff = off_cost - on_cost;
    assert!(
        (diff - expected_diff).abs() < expected_diff * 0.05,
        "expected CG savings ≈ {expected_diff:.2} µs, got {diff:.2} µs"
    );

    // And sanity-check launch_overhead directly.
    let launch_off = launch_overhead(&off, model.constants(), kernels);
    let launch_on = launch_overhead(&on, model.constants(), kernels);
    assert!(launch_on < launch_off);
}

// ---------------------------------------------------------------------------
// Test 6 — per-device max dominates (asymmetric per-stage dtype).
// ---------------------------------------------------------------------------

#[test]
fn per_device_max_dominates() {
    // pp = 2 split evenly: stage 0 gets blocks 0..16, stage 1 gets 16..32.
    // Stage 0 uses bf16 weights; stage 1 uses int4. Stage 0 is the slower
    // stage, so `total_cost ≈ per_device_cost(0)`, not the average.
    let model = load_cost_model();
    let ir = load_mixtral_ir();
    let num_blocks = ir.meta.num_layers;

    // Build a 4-device, 1-node, all-NVLink cluster so neither stage is comm-
    // bound on its TP allreduces.
    let toml = r#"
num_devices = 4
[[node]]
id               = "node0"
devices          = ["d0", "d1", "d2", "d3"]
device_kind      = "h100_sxm5"
device_memory_gb = 80
[[link]]
endpoints = ["d0", "d1"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d2", "d3"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d0", "d2"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
[[link]]
endpoints = ["d1", "d3"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
"#;
    let spec = ClusterSpec::from_toml_str(toml).unwrap();
    let cluster = skein_cost::Cluster::from_spec(spec);

    let mut per_layer: Vec<PerLayerDtype> = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        // Stage 0 = bf16, stage 1 = int4.
        let stage = skein_cost::cluster::block_to_stage(b, num_blocks, 2);
        let dt = if stage == 0 { Dtype::Bf16 } else { Dtype::Int4 };
        per_layer.push(PerLayerDtype {
            weight: dt,
            activation: Dtype::Bf16,
            kv_cache: Dtype::Bf16,
        });
    }
    let plan = Plan {
        parallelism: ParallelismPlacement {
            tp: 2,
            pp: 2,
            ep: 1,
        },
        kv: KVCacheSpec {
            layout: KVLayout::Paged { page_size: 32 },
            kv_sharded: false,
        },
        batching: BatchPolicy::Continuous { max_batch: 8 },
        dtype_map: DtypeMap { per_layer },
        execution: ExecutionConfig {
            cuda_graphs: CudaGraphsConfig {
                enable: false,
                capture_classes: vec![],
            },
            spec_decode: SpecDecodeConfig {
                enable: false,
                draft: None,
            },
            prefix_cache: PrefixCacheConfig {
                enable: false,
                reuse_policy: RadixReusePolicy::LruByLastAccess,
            },
        },
        disaggregation: None,
        model_meta: ir.meta.clone(),
    };

    let wl = skein_cost::WorkloadCtx::decode_step(&plan, model.constants());
    let d0 = model.per_device_cost(&plan, &ir, &cluster, &wl, 0).unwrap();
    let d2 = model.per_device_cost(&plan, &ir, &cluster, &wl, 2).unwrap();
    let total = model.total_cost(&plan, &ir, &cluster).unwrap().as_us();

    assert!(
        d0 > d2,
        "stage-0 (bf16) should be slower than stage-1 (int4); got d0={d0}, d2={d2}"
    );
    // total_cost is the max over devices — d0 in this setup. Allow a tiny
    // float jitter envelope.
    assert!(
        (total - d0).abs() < 1.0,
        "total_cost ({total}) should equal d0 ({d0}), not average"
    );
}

// ---------------------------------------------------------------------------
// Test 7 — fused all-reduce beats unfused pair.
// ---------------------------------------------------------------------------

#[test]
fn fused_allreduce_beats_unfused_pair() {
    let model = load_cost_model();
    let cluster = load_2x_h100_cluster();
    const N: u64 = 16 * 1024 * 1024; // 16 MB

    let fused = Collective {
        kind: CollectiveKind::RingAllReduce,
        participants: vec![0, 1],
        bytes: 2 * N,
    };
    let unfused = (
        Collective {
            kind: CollectiveKind::RingAllReduce,
            participants: vec![0, 1],
            bytes: N,
        },
        Collective {
            kind: CollectiveKind::RingAllReduce,
            participants: vec![0, 1],
            bytes: N,
        },
    );

    let fused_us = model.comm_time_one(&fused, &cluster).unwrap();
    let unfused_us = model.comm_time_one(&unfused.0, &cluster).unwrap()
        + model.comm_time_one(&unfused.1, &cluster).unwrap();

    // Transfer terms are equal (linear in total bytes); unfused pays the
    // path latency twice. With latency = 1 µs the gap is 1 µs, small but
    // strictly positive.
    assert!(
        fused_us < unfused_us,
        "fused {fused_us} should be < unfused pair {unfused_us}"
    );
    assert!(
        (unfused_us - fused_us - 1.0).abs() < 1e-6,
        "expected exactly one extra latency hop ({} µs), got {}",
        1.0,
        unfused_us - fused_us
    );
}

// ---------------------------------------------------------------------------
// Auxiliary regression: tp=1 issues no TP collectives; tp=2 issues two per
// block. Guards `collectives_on_device` from silently regressing.
// ---------------------------------------------------------------------------

#[test]
fn tp_collective_count_matches_blocks() {
    let ir = load_mixtral_ir();
    let plan_tp1 = plan_with(
        ir.meta.clone(),
        1,
        1,
        1,
        Dtype::Bf16,
        Dtype::Bf16,
        Dtype::Bf16,
        BatchPolicy::Continuous { max_batch: 1 },
        CudaGraphsConfig {
            enable: false,
            capture_classes: vec![],
        },
    );
    let plan_tp2 = plan_with(
        ir.meta.clone(),
        2,
        1,
        1,
        Dtype::Bf16,
        Dtype::Bf16,
        Dtype::Bf16,
        BatchPolicy::Continuous { max_batch: 1 },
        CudaGraphsConfig {
            enable: false,
            capture_classes: vec![],
        },
    );
    assert!(collectives_on_device(&plan_tp1, &ir, 0).is_empty());
    let tp2 = collectives_on_device(&plan_tp2, &ir, 0);
    // 32 blocks × 2 AllReduces each (attention + MoE) = 64.
    assert_eq!(
        tp2.iter()
            .filter(|c| c.kind == CollectiveKind::RingAllReduce)
            .count(),
        ir.meta.num_layers * 2
    );
    // Placement validity sanity check.
    let p = Placement::from_plan(&plan_tp2);
    assert_eq!(p.tp_group(0), vec![0, 1]);
}
