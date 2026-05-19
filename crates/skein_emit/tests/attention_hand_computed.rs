//! Test 2 — hand-computed tiny attention block.
//!
//! Drives `op_wiring::wire_attention_math` directly with F32 weights and a
//! controlled F32 input, compiles via Luminal's `NativeRuntime`, executes,
//! and compares to a hand-computed reference. Math is inlined in Rust below
//! (no synthetic fixture, no PyTorch import — see Rule 3).
//!
//! Config:
//!   batch = 1, seq = 2, hidden = 4, num_attention_heads = 1,
//!   num_kv_heads = 1, head_dim = 4.
//! Weights: Q = K = V = O = identity(4).
//! Input X = [[1,0,0,0], [0,1,0,0]] (one-hot rows).
//!
//! Reference math (all f32, no rounding):
//!   Q = K = V = X
//!   scores = (Q @ K^T) / sqrt(head_dim) = [[0.5, 0], [0, 0.5]]
//!   row_softmax([0.5, 0]) = [e^0.5/(e^0.5+1), 1/(e^0.5+1)]
//!                       = [0.62245933..., 0.37754066...]
//!   weights = [[0.6224..., 0.3775...],
//!              [0.3775..., 0.6224...]]
//!   weights @ V = [[0.6224..., 0.3775..., 0, 0],
//!                  [0.3775..., 0.6224..., 0, 0]]
//!   attn = (weights @ V) @ O^T = same (O = I).
//!
//! Tolerance 1e-5 is appropriate because every op runs in F32 throughout:
//! the test goes through Luminal's *exact* op graph but with `cx.tensor`
//! handles (default F32) instead of the production Bf16 path. The Bf16
//! variant of this test belongs on the GPU calibrate path.

use luminal::op::Runtime;
use luminal::prelude::*;
use skein_emit::op_wiring::wire_attention_math;
use skein_ir::ir::ModelMeta;

/// Build a `ModelMeta` shaped for the tiny attention block under test. The
/// non-attention fields are minimal placeholders — `wire_attention_math`
/// reads only the heads / head_dim / hidden trio.
fn meta_for_tiny_attention() -> ModelMeta {
    ModelMeta {
        architecture: "TinyAttention".to_string(),
        num_layers: 1,
        hidden: 4,
        vocab: 1,
        max_position: 8,
        num_attention_heads: 1,
        num_kv_heads: 1,
        head_dim: 4,
        num_experts: None,
        top_k: None,
        intermediate: 4,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        sliding_window: None,
        tied_embeddings: false,
    }
}

#[test]
fn wire_attention_matches_hand_computed_reference() {
    let mut cx = Graph::new();

    // Static-shape input: [batch=1, seq=2, hidden=4].
    let input = cx.named_tensor("input", (1usize, 2usize, 4usize));

    // Identity 4x4 weights for Q, K, V, O.
    let q_w = cx.named_tensor("q_w", (4usize, 4usize));
    let k_w = cx.named_tensor("k_w", (4usize, 4usize));
    let v_w = cx.named_tensor("v_w", (4usize, 4usize));
    let o_w = cx.named_tensor("o_w", (4usize, 4usize));

    let meta = meta_for_tiny_attention();
    let out = wire_attention_math(
        meta.num_attention_heads,
        meta.num_kv_heads,
        meta.head_dim,
        input,
        q_w,
        k_w,
        v_w,
        o_w,
    )
    .output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let identity = vec![
        1.0_f32, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, // row 2
        0.0, 0.0, 0.0, 1.0, // row 3
    ];
    let input_data = vec![
        1.0_f32, 0.0, 0.0, 0.0, // [b=0, s=0, :]
        0.0, 1.0, 0.0, 0.0, // [b=0, s=1, :]
    ];

    rt.set_data(input.id, input_data);
    rt.set_data(q_w.id, identity.clone());
    rt.set_data(k_w.id, identity.clone());
    rt.set_data(v_w.id, identity.clone());
    rt.set_data(o_w.id, identity);
    rt.execute(&cx.dyn_map);

    let got = rt.get_f32(out.id).clone();
    // [batch=1, seq=2, hidden=4] = 8 elements
    assert_eq!(got.len(), 8, "output shape [batch, seq, hidden]");

    // Hand-computed reference.
    let s0 = 0.5_f32.exp(); // e^0.5
    let denom = s0 + 1.0;
    let w_hi = s0 / denom; // ≈ 0.62245933
    let w_lo = 1.0 / denom; // ≈ 0.37754067
    let expected = [
        w_hi, w_lo, 0.0, 0.0, // [b=0, s=0, :]
        w_lo, w_hi, 0.0, 0.0, // [b=0, s=1, :]
    ];

    for (i, (g, w)) in got.iter().zip(expected.iter()).enumerate() {
        let d = (g - w).abs();
        assert!(d < 1e-5, "pos {i}: got {g}, want {w} (diff={d:.3e})");
    }
}
