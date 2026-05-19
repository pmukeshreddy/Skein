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

    // tp=2, ep=1: 32 blocks × 2 AllReduces (after attn + after MoE down).
    let ar_count = a
        .collectives
        .iter()
        .filter(|c| c.kind == CollectiveKind::RingAllReduce)
        .count();
    assert_eq!(ar_count, ir.meta.num_layers * 2);

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
fn topology_adds_alltoall_under_ep() {
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
    // 32 blocks × 2 AllReduces + 32 blocks × 2 AllToAll (dispatch + combine).
    assert_eq!(ar_count, ir.meta.num_layers * 2);
    assert_eq!(a2a_count, ir.meta.num_layers * 2);
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
