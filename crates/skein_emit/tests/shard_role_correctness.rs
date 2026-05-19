//! Tests 1–5 — `ShardRole` resolution.

mod common;
use common::*;

use skein_emit::{ShardRole, shard_role_for_param};
use skein_ir::ir::LayerKind;

// Test 1 — layernorm + final norm + (Phase A) embedding/lm_head are
// Replicated regardless of parallelism.
#[test]
fn shard_role_replicated_for_norms() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 1);

    for device_idx in 0..2 {
        for layer in &ir.layers {
            for param in &layer.params {
                let role = shard_role_for_param(&plan, &cluster, &ir, device_idx, layer, param);
                if param.name.ends_with("_layernorm.weight") || param.name == "model.norm.weight" {
                    assert_eq!(
                        role,
                        ShardRole::Replicated,
                        "norm {} on device {} should be Replicated, got {:?}",
                        param.name,
                        device_idx,
                        role,
                    );
                }
            }
        }
    }
}

// Test 2 — Q/K/V projections under TP are TpOutputShard.
#[test]
fn shard_role_tp_column_for_qkv() {
    let ir = load_mixtral_ir();
    let cluster = build_4x_h100_cluster();

    for &tp in &[2u32, 4u32] {
        let plan = mk_plan(ir.meta.clone(), tp, 1, 1);
        for device_idx in 0..tp {
            for layer in &ir.layers {
                if !matches!(layer.kind, LayerKind::Attention(_)) {
                    continue;
                }
                for param in &layer.params {
                    if !(param.name.ends_with(".q_proj.weight")
                        || param.name.ends_with(".k_proj.weight")
                        || param.name.ends_with(".v_proj.weight"))
                    {
                        continue;
                    }
                    let role = shard_role_for_param(&plan, &cluster, &ir, device_idx, layer, param);
                    match role {
                        ShardRole::TpOutputShard {
                            group_size,
                            group_index,
                        } => {
                            assert_eq!(group_size, tp);
                            assert_eq!(group_index, device_idx);
                        }
                        other => panic!(
                            "{} on device {} (tp={}) should be TpOutputShard, got {:?}",
                            param.name, device_idx, tp, other
                        ),
                    }
                }
            }
        }
    }
}

// Test 3 — O projection and (MoE expert) w2 are TpInputShard at tp>1.
#[test]
fn shard_role_tp_row_for_o_and_down() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 1);

    let mut saw_o = false;
    let mut saw_w2 = false;

    for device_idx in 0..2 {
        for layer in &ir.layers {
            for param in &layer.params {
                let role = shard_role_for_param(&plan, &cluster, &ir, device_idx, layer, param);
                if param.name.ends_with(".o_proj.weight") {
                    saw_o = true;
                    match role {
                        ShardRole::TpInputShard {
                            group_size,
                            group_index,
                        } => {
                            assert_eq!(group_size, 2);
                            assert_eq!(group_index, device_idx);
                        }
                        other => panic!("o_proj should be TpInputShard, got {:?}", other),
                    }
                }
                if param.name.contains(".block_sparse_moe.experts.")
                    && param.name.ends_with(".w2.weight")
                {
                    saw_w2 = true;
                    // With ep=1, expert weights fall through to TP rules —
                    // w2 row-parallel.
                    match role {
                        ShardRole::TpInputShard {
                            group_size,
                            group_index,
                        } => {
                            assert_eq!(group_size, 2);
                            assert_eq!(group_index, device_idx);
                        }
                        other => panic!("expert w2 should be TpInputShard, got {:?}", other),
                    }
                }
            }
        }
    }
    assert!(saw_o, "no o_proj weights found in IR");
    assert!(saw_w2, "no expert w2 weights found in IR");
}

// Test 4 — Expert routing under EP=2 on Mixtral's 8 experts.
#[test]
fn shard_role_expert_routing_modulo() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 1, 1, 2);

    // For each expert index 0..8, exactly one device should report
    // ExpertOwned and the other ExpertElsewhere.
    for expert_idx in 0..8u32 {
        let target_param =
            format!("model.layers.0.block_sparse_moe.experts.{expert_idx}.w1.weight");
        let layer = ir
            .layers
            .iter()
            .find(|l| l.params.iter().any(|p| p.name == target_param))
            .expect("find moe layer for block 0");
        let param = layer
            .params
            .iter()
            .find(|p| p.name == target_param)
            .unwrap();

        let mut owners: Vec<u32> = Vec::new();
        for device_idx in 0..2u32 {
            let role = shard_role_for_param(&plan, &cluster, &ir, device_idx, layer, param);
            match role {
                ShardRole::ExpertOwned { expert_idx: idx } => {
                    assert_eq!(idx, expert_idx);
                    owners.push(device_idx);
                }
                ShardRole::ExpertElsewhere { expert_idx: idx } => {
                    assert_eq!(idx, expert_idx);
                }
                other => panic!(
                    "expert {expert_idx} role on device {device_idx}: {:?}",
                    other
                ),
            }
        }
        assert_eq!(owners.len(), 1, "expert {expert_idx} owned by {owners:?}");
        // Device 0 should own even-indexed experts (mod 2).
        let expected_owner = expert_idx % 2;
        assert_eq!(owners[0], expected_owner);
    }
}

// Test 5 — PP stages disjoint. Mixtral 32 blocks under pp=2 → blocks 0–15 on
// stage 0, 16–31 on stage 1.
#[test]
fn shard_role_pp_stages_disjoint() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 1, 2, 1);

    let mut blocks_seen_on_stage: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for device_idx in 0..2u32 {
        for layer in &ir.layers {
            let Some(b) = layer.block_idx else { continue };
            // Pick an arbitrary param from the block.
            let Some(param) = layer.params.first() else {
                continue;
            };
            let role = shard_role_for_param(&plan, &cluster, &ir, device_idx, layer, param);
            if !matches!(role, ShardRole::PipelineStageElsewhere)
                && !blocks_seen_on_stage[device_idx as usize].contains(&b)
            {
                blocks_seen_on_stage[device_idx as usize].push(b);
            }
        }
    }
    blocks_seen_on_stage[0].sort();
    blocks_seen_on_stage[1].sort();
    let stage0_expected: Vec<usize> = (0..16).collect();
    let stage1_expected: Vec<usize> = (16..32).collect();
    assert_eq!(blocks_seen_on_stage[0], stage0_expected);
    assert_eq!(blocks_seen_on_stage[1], stage1_expected);

    // Lm_head: stage pp-1 (= 1). Embedding: stage 0.
    let embed_layer = ir
        .layers
        .iter()
        .find(|l| matches!(l.kind, LayerKind::Embedding(_)))
        .unwrap();
    let embed_param = &embed_layer.params[0];
    let embed_d0 = shard_role_for_param(&plan, &cluster, &ir, 0, embed_layer, embed_param);
    let embed_d1 = shard_role_for_param(&plan, &cluster, &ir, 1, embed_layer, embed_param);
    assert_eq!(embed_d0, ShardRole::Replicated);
    assert_eq!(embed_d1, ShardRole::PipelineStageElsewhere);

    let lm_layer = ir
        .layers
        .iter()
        .find(|l| matches!(l.kind, LayerKind::Lmhead(_)))
        .unwrap();
    let lm_param = &lm_layer.params[0];
    let lm_d0 = shard_role_for_param(&plan, &cluster, &ir, 0, lm_layer, lm_param);
    let lm_d1 = shard_role_for_param(&plan, &cluster, &ir, 1, lm_layer, lm_param);
    assert_eq!(lm_d0, ShardRole::PipelineStageElsewhere);
    assert_eq!(lm_d1, ShardRole::Replicated);
}
