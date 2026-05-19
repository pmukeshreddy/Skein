//! Test 6 — declared luminal::Graph tensor shapes reflect the Plan's sharding.
//!
//! Under Phase 1b's segmented LoweredGraph, `declared` lives per-segment;
//! we look up each weight by walking the segments and asserting the
//! first place it appears matches the expected sharded shape.

mod common;
use common::*;

use skein_emit::build_device_graph;
use skein_emit::graph_builder::DeclaredTensor;
use skein_ir::types::Dtype;

/// Find a declared weight across all segments of a lowered device.
/// Each weight is declared in exactly one segment (the one that uses
/// it); this helper hides that detail.
fn find_declared<'a>(
    lowered: &'a skein_emit::LoweredGraph,
    name: &str,
) -> Option<&'a DeclaredTensor> {
    for seg in &lowered.segments {
        if let Some(d) = seg.declared.get(name) {
            return Some(d);
        }
    }
    None
}

#[test]
fn graph_shapes_match_plan() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 1);

    let lowered_d0 = build_device_graph(&plan, &cluster, &ir, 0).expect("build d0 graph");
    let lowered_d1 = build_device_graph(&plan, &cluster, &ir, 1).expect("build d1 graph");

    // Q-proj weight: source shape [4096, 4096]; under TpOutputShard with
    // tp=2, device 0 + device 1 each see [2048, 4096].
    let q0 = find_declared(&lowered_d0, "model.layers.0.self_attn.q_proj.weight")
        .expect("d0 declared q_proj");
    let q1 = find_declared(&lowered_d1, "model.layers.0.self_attn.q_proj.weight")
        .expect("d1 declared q_proj");
    assert_eq!(q0.shape, vec![2048, 4096]);
    assert_eq!(q1.shape, vec![2048, 4096]);
    // Dtype propagated from the Plan (bf16 here).
    assert_eq!(q0.dtype, Dtype::Bf16);

    // O-proj: source [4096, 4096]; row-parallel ⇒ [4096, 2048] on each.
    let o0 = find_declared(&lowered_d0, "model.layers.0.self_attn.o_proj.weight")
        .expect("d0 declared o_proj");
    assert_eq!(o0.shape, vec![4096, 2048]);

    // K-proj: source [1024, 4096] (GQA → 8 KV heads × 128 head_dim).
    // tp=2 TpOutputShard → [512, 4096].
    let k0 = find_declared(&lowered_d0, "model.layers.0.self_attn.k_proj.weight")
        .expect("d0 declared k_proj");
    assert_eq!(k0.shape, vec![512, 4096]);

    // Norm weights stay replicated → unchanged shape [4096].
    let n0 = find_declared(&lowered_d0, "model.layers.0.input_layernorm.weight")
        .expect("d0 declared input_layernorm");
    assert_eq!(n0.shape, vec![4096]);

    // Embedding replicated → unchanged [32000, 4096].
    let e0 = find_declared(&lowered_d0, "model.embed_tokens.weight").expect("d0 declared embed");
    assert_eq!(e0.shape, vec![32000, 4096]);
}
