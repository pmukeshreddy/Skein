//! RoPE — hand-computed reference for a single rotation.
//!
//! Verifies `rope_tables` + `apply_rope` against a value computed by hand:
//! for head_dim=4, theta=10000, RoPE rotates the dimension pairs (0,2) and
//! (1,3). Rotating the unit vector [1,0,0,0] at position 1 by angle
//! theta_0 = 1 rad gives [cos 1, 0, sin 1, 0]. Position 0 is the identity.

use luminal::op::Runtime;
use luminal::prelude::*;
use skein_emit::op_wiring::{apply_rope, rope_tables};

#[test]
fn apply_rope_matches_hand_computed_rotation() {
    let mut cx = Graph::new();
    // x: [batch=1, seq=2, heads=1, head_dim=4].
    let x = cx.named_tensor("x", (1usize, 2usize, 1usize, 4usize));
    let (cos_t, sin_t) = rope_tables(&mut cx, 2usize, 4, 10_000.0, 0);
    let out = apply_rope(x, cos_t, sin_t, 4).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    // Both sequence positions carry the unit vector [1, 0, 0, 0].
    rt.set_data(x.id, vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out.id).clone();
    assert_eq!(got.len(), 8);

    // Position 0: identity rotation.
    let pos0 = &got[0..4];
    for (i, (g, w)) in pos0.iter().zip([1.0_f32, 0.0, 0.0, 0.0]).enumerate() {
        assert!((g - w).abs() < 1e-5, "pos0 dim {i}: got {g}, want {w}");
    }

    // Position 1: rotate [1,0,0,0] by 1 rad in the (0,2) plane.
    let pos1 = &got[4..8];
    let expected1 = [1.0_f32.cos(), 0.0, 1.0_f32.sin(), 0.0];
    for (i, (g, w)) in pos1.iter().zip(expected1).enumerate() {
        assert!((g - w).abs() < 1e-5, "pos1 dim {i}: got {g}, want {w}");
    }
}

/// Decode-position: a single token (`seq = 1`) at `position_offset = 1` must
/// be rotated as absolute position 1 — identical to the position-1 row of a
/// prefill that started at 0. This is the offset that a decode step feeds as
/// the current KV-cache length.
#[test]
fn rope_position_offset_places_decode_token_correctly() {
    let mut cx = Graph::new();
    let x = cx.named_tensor("x", (1usize, 1usize, 1usize, 4usize)); // one decode token
    let (cos_t, sin_t) = rope_tables(&mut cx, 1usize, 4, 10_000.0, 1); // offset = KV length 1
    let out = apply_rope(x, cos_t, sin_t, 4).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    rt.set_data(x.id, vec![1.0_f32, 0.0, 0.0, 0.0]);
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out.id).clone();

    let expected = [1.0_f32.cos(), 0.0, 1.0_f32.sin(), 0.0]; // rotation by 1 rad
    for (i, (g, w)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "decode token at offset 1, dim {i}: got {g}, want {w}"
        );
    }
}
