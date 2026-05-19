//! Test 3 — `wire_segments` produces a structurally valid `LoweredGraph`
//! for the full Mixtral 8x7B IR at tp=1.
//!
//! Adapted for Phase 1b's segmented LoweredGraph: op_nodes and declared
//! are per-segment, so we walk all segments to find the expected
//! tensors. At tp=1 there's exactly one segment per device.
//!
//! "Structurally valid" means:
//!   1. `op_nodes` somewhere contains the semantic outputs we expect
//!      (`input_tokens`, `hidden_after_embed`, `hidden_after_block_31`,
//!      `logits`).
//!   2. The headline parameters are reachable through some segment's
//!      `declared`.
//!   3. Every segment's `cx.build_search_space::<NativeRuntime>()`
//!      succeeds — i.e. every op `wire_segments` added is representable
//!      in Luminal's compile pipeline.

mod common;
use common::*;

use luminal::prelude::NativeRuntime;
use skein_emit::build_device_graph;

#[test]
fn wire_segments_produces_valid_graph_tp1() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 1, 1, 1);

    let mut lowered = build_device_graph(&plan, &cluster, &ir, 0).expect("build d0 graph");

    // tp=ep=pp=1 collapses to a single segment.
    assert_eq!(lowered.segments.len(), 1, "tp=1 should produce 1 segment");

    let expected_ops = [
        "input_tokens",
        "hidden_after_embed",
        "hidden_after_block_0",
        "hidden_after_block_31",
        "logits",
    ];
    for key in expected_ops {
        let found = lowered
            .segments
            .iter()
            .any(|s| s.op_nodes.contains_key(key));
        assert!(found, "op_nodes should contain '{key}' in some segment");
    }

    let expected_weights = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.7.w3.weight",
        "model.layers.31.self_attn.o_proj.weight",
    ];
    for key in expected_weights {
        let found = lowered
            .segments
            .iter()
            .any(|s| s.declared.contains_key(key));
        assert!(found, "declared should contain '{key}' in some segment");
    }

    for seg in &mut lowered.segments {
        seg.graph.build_search_space::<NativeRuntime>();
    }
}
