//! Test 8 — `emit_topology` byte-determinism + counts under tp / ep.

mod common;
use common::*;

use skein_cost::collectives::CollectiveKind;
use skein_emit::emit_topology;

#[test]
fn topology_deterministic_for_mixtral() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 1);

    let a = emit_topology(&plan, &cluster, &ir);
    let b = emit_topology(&plan, &cluster, &ir);
    let a_bytes = serde_json::to_vec(&a).unwrap();
    let b_bytes = serde_json::to_vec(&b).unwrap();
    assert_eq!(a_bytes, b_bytes, "emit_topology must be deterministic");

    // tp=2, ep=1: 32 blocks × 2 AllReduces (after attn + after MoE down),
    // plus 1 vocab-parallel embedding AllReduce before block 0.
    let ar_count = a
        .collectives
        .iter()
        .filter(|c| c.kind == CollectiveKind::RingAllReduce)
        .count();
    assert_eq!(ar_count, ir.meta.num_layers * 2 + 1);

    // Exactly one vocab-parallel logits AllGather after the LM head.
    let ag_count = a
        .collectives
        .iter()
        .filter(|c| c.kind == CollectiveKind::AllGather)
        .count();
    assert_eq!(ag_count, 1);

    // No EP collectives, no SendRecv (single PP stage).
    assert!(
        !a.collectives
            .iter()
            .any(|c| c.kind == CollectiveKind::AllToAll)
    );
    assert!(
        !a.collectives
            .iter()
            .any(|c| c.kind == CollectiveKind::SendRecv)
    );

    // Sequence indices are monotonic and dense (0..N).
    for (i, entry) in a.collectives.iter().enumerate() {
        assert_eq!(entry.sequence_idx, i as u64);
    }
}

#[test]
fn topology_adds_ep_combine_allreduce_under_ep() {
    let ir = load_mixtral_ir();
    let cluster = build_4x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 2);

    let t = emit_topology(&plan, &cluster, &ir);
    let ar_count = t
        .collectives
        .iter()
        .filter(|c| c.kind == CollectiveKind::RingAllReduce)
        .count();
    let a2a_count = t
        .collectives
        .iter()
        .filter(|c| c.kind == CollectiveKind::AllToAll)
        .count();
    // Dense expert parallel: per block there are 3 all-reduces — TP after attn,
    // EP combine after MoE, TP after the MoE down-proj — plus 1 vocab-parallel
    // embedding all-reduce before block 0. No AllToAll (no token dispatch).
    assert_eq!(ar_count, ir.meta.num_layers * 3 + 1);
    assert_eq!(a2a_count, 0);
}

#[test]
fn topology_emits_send_recv_under_pp() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 1, 2, 1);

    let t = emit_topology(&plan, &cluster, &ir);
    let sr_count = t
        .collectives
        .iter()
        .filter(|c| c.kind == CollectiveKind::SendRecv)
        .count();
    // pp=2 → exactly one stage boundary, one SendRecv per forward step.
    assert_eq!(sr_count, 1);
    // No TP allreduces, no EP all-to-alls.
    assert!(
        !t.collectives
            .iter()
            .any(|c| c.kind == CollectiveKind::RingAllReduce || c.kind == CollectiveKind::AllToAll)
    );
}
