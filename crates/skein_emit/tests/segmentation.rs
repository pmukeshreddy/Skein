//! Multi-segment LoweredGraph tests.
//!
//! Test 1: segmentation count exact for tp=2 ep=1 and tp=2 ep=2.
//! Test 2: handoff name alignment across each collective boundary.
//! Test 3: each segment compiles via `cx.build_search_space::<NativeRuntime>()`.
//! Test 4: sequencing serialization is byte-stable.
//! Test 5: tp=1 still produces one segment (regression).

mod common;
use common::*;

use luminal::prelude::NativeRuntime;
use skein_cost::Cluster;
use skein_cost::collectives::CollectiveKind;
use skein_emit::build_device_graph;
use skein_emit::segment::SequenceStep;
use skein_ir::cluster::ClusterSpec;
use skein_ir::ir::Graph;
use skein_ir::plan::Plan;
use skein_ir::types::Dtype;

fn load_4x_h100_cluster_for_tp2ep2() -> Cluster {
    // tp=2 ep=2 needs 4 devices in one stage. Use `common::build_4x_h100_cluster`
    // which is already shaped right.
    build_4x_h100_cluster()
}

fn segments_per_device(plan: &Plan, cluster: &Cluster, ir: &Graph, device_idx: u32) -> usize {
    let lowered = build_device_graph(plan, cluster, ir, device_idx).expect("build device graph");
    lowered.segments.len()
}

// ── Test 1 — segmentation count exact ──────────────────────────────────────

#[test]
fn segmentation_count_tp2_ep1() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 1);

    // tp=2 ep=1: 2 collectives/block × 32 blocks = 64, plus the two
    // vocab-parallel collectives (embedding AllReduce + logits AllGather) = 66
    // collectives → 67 segments.
    for device_idx in 0..2 {
        let count = segments_per_device(&plan, &cluster, &ir, device_idx);
        assert_eq!(
            count, 67,
            "device {device_idx} at tp=2 ep=1: expected 67 segments, got {count}",
        );
    }
}

#[test]
fn segmentation_count_tp2_ep2() {
    let ir = load_mixtral_ir();
    let cluster = load_4x_h100_cluster_for_tp2ep2();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 2);

    // tp=2 ep=2: 4 collectives/block × 32 blocks = 128, plus the two
    // vocab-parallel collectives (embedding AllReduce + logits AllGather) = 130
    // collectives → 131 segments.
    for device_idx in 0..4 {
        let count = segments_per_device(&plan, &cluster, &ir, device_idx);
        assert_eq!(
            count, 131,
            "device {device_idx} at tp=2 ep=2: expected 131 segments, got {count}",
        );
    }
}

// ── Test 2 — handoff names align across each collective boundary ──────────

#[test]
fn handoff_names_match_across_segments() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 1);
    let lowered = build_device_graph(&plan, &cluster, &ir, 0).expect("build d0 graph");

    // Walk sequencing: every `Collective` step's `tensor` should appear
    // in BOTH the preceding segment's `output_handoff` AND the
    // following segment's `input_handoff`. The "same logical name"
    // invariant from the prompt.
    let mut prev_segment_idx: Option<usize> = None;
    let mut steps = lowered.sequencing.iter().peekable();
    while let Some(step) = steps.next() {
        match step {
            SequenceStep::ExecuteSegment { segment_idx, .. } => {
                prev_segment_idx = Some(*segment_idx);
            }
            SequenceStep::Collective { tensor, .. } => {
                let upstream_idx = prev_segment_idx.expect(
                    "Collective must follow at least one ExecuteSegment in the device schedule",
                );
                let upstream = &lowered.segments[upstream_idx];
                let in_upstream = upstream
                    .output_handoff
                    .iter()
                    .any(|h| h.logical_name == *tensor);
                assert!(
                    in_upstream,
                    "collective on '{tensor}' but upstream segment {upstream_idx} doesn't list it",
                );

                // The next step should be ExecuteSegment.
                let next_step = steps.peek().expect("collective followed by ExecuteSegment");
                let SequenceStep::ExecuteSegment {
                    segment_idx: downstream_idx,
                    ..
                } = **next_step
                else {
                    panic!("expected ExecuteSegment after Collective, got {next_step:?}");
                };
                let downstream = &lowered.segments[downstream_idx];
                let in_downstream = downstream
                    .input_handoff
                    .iter()
                    .any(|h| h.logical_name == *tensor);
                assert!(
                    in_downstream,
                    "collective on '{tensor}' but downstream segment {downstream_idx} doesn't list it",
                );
            }
        }
    }
}

// ── Test 3 — every segment compiles independently via NativeRuntime ───────

#[test]
fn each_segment_compiles_independently_native() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 1);
    let mut lowered = build_device_graph(&plan, &cluster, &ir, 0).expect("build d0 graph");

    let started = std::time::Instant::now();
    for seg in &mut lowered.segments {
        seg.graph.build_search_space::<NativeRuntime>();
    }
    let elapsed = started.elapsed();
    eprintln!(
        "compiled {} segments in {:?}",
        lowered.segments.len(),
        elapsed
    );
    // Guard against pathological (e.g. super-linear) blowup in
    // `build_search_space` across all 65 segments of the full decoder graph
    // — which now carries the complete per-op math (RoPE, causal mask, top-k
    // routing). This is a coarse upper bound, not a latency SLA: native
    // compile is single-threaded and sensitive to host load.
    assert!(
        elapsed.as_secs() < 300,
        "per-segment compile budget exceeded ({} segments in {:?})",
        lowered.segments.len(),
        elapsed,
    );
}

// ── Test 4 — sequencing serialization is byte-stable ──────────────────────

#[test]
fn sequencing_serialization_byte_stable() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 1);

    let a = build_device_graph(&plan, &cluster, &ir, 0).expect("build d0 once");
    let b = build_device_graph(&plan, &cluster, &ir, 0).expect("build d0 twice");

    let ja = serde_json::to_string(&a.sequencing).expect("serialize a");
    let jb = serde_json::to_string(&b.sequencing).expect("serialize b");
    assert_eq!(ja.as_bytes(), jb.as_bytes(), "sequencing JSON not stable");

    // Spot-check: the count of Collective entries equals the segment count − 1.
    let coll_count = a
        .sequencing
        .iter()
        .filter(|s| matches!(s, SequenceStep::Collective { .. }))
        .count();
    assert_eq!(coll_count, a.segments.len() - 1);

    // Collectives at tp=2 ep=1 with vocab-parallel: an embedding AllReduce,
    // two TP RingAllReduces per block, and a final logits AllGather — all bf16.
    let kinds: Vec<CollectiveKind> = a
        .sequencing
        .iter()
        .filter_map(|s| match s {
            SequenceStep::Collective {
                collective, dtype, ..
            } => {
                assert_eq!(*dtype, Dtype::Bf16);
                Some(*collective)
            }
            _ => None,
        })
        .collect();
    let (last, rest) = kinds.split_last().expect("at least one collective");
    assert_eq!(
        *last,
        CollectiveKind::AllGather,
        "final collective is the vocab-parallel logits AllGather"
    );
    for k in rest {
        assert_eq!(*k, CollectiveKind::RingAllReduce);
    }
}

// ── Test 5 — tp=1 regression: one segment per device ──────────────────────

#[test]
fn tp1_still_works() {
    let ir = load_mixtral_ir();
    // Use a 1-device cluster so device_idx=0 is the whole world.
    let cluster_toml = r#"
num_devices = 1
[[node]]
id               = "node0"
devices          = ["d0"]
device_kind      = "h100_sxm5"
device_memory_gb = 80
"#;
    let spec = ClusterSpec::from_toml_str(cluster_toml).expect("parse 1× cluster");
    let cluster = Cluster::from_spec(spec);
    let plan = mk_plan(ir.meta.clone(), 1, 1, 1);

    let lowered = build_device_graph(&plan, &cluster, &ir, 0).expect("build d0 graph");
    assert_eq!(lowered.segments.len(), 1, "tp=1 → exactly one segment");
    // No collectives; just one ExecuteSegment step.
    assert_eq!(lowered.sequencing.len(), 1);
    assert!(matches!(
        lowered.sequencing[0],
        SequenceStep::ExecuteSegment { .. }
    ));
}
