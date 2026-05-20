//! Hand-checked tiny attention block, now exercising RoPE + the causal mask.
//!
//! Drives `op_wiring::wire_attention_math` directly with F32 weights and a
//! controlled F32 input, compiles via Luminal's `NativeRuntime`, executes,
//! and checks invariants that are robustly verifiable by hand.
//!
//! Config:
//!   batch = 1, seq = 2, hidden = 4, num_attention_heads = 1,
//!   num_kv_heads = 1, head_dim = 4.
//! Weights: Q = K = V = O = identity(4).
//! Input X = [[1,0,0,0], [0,1,0,0]] (one-hot rows).
//!
//! Checked invariants:
//!   * Causal masking + RoPE@position-0 identity: query at position 0 may
//!     attend only to key 0, and RoPE at position 0 is the identity rotation
//!     (cos 0 = 1, sin 0 = 0). With identity weights the output row 0 is
//!     therefore exactly V[0] @ O = X[0] = [1, 0, 0, 0].
//!   * Query at position 1 attends to keys 0 and 1: the output row is a
//!     convex combination of V[0]=[1,0,0,0] and V[1]=[0,1,0,0] (V is not
//!     rotated), so out[1] = [w0, w1, 0, 0] with w0 + w1 = 1 and both in
//!     (0, 1).
//!
//! Numeric parity of the *rotated* row 1 against a bf16 HF reference belongs
//! on the GPU verify/calibrate path; here we pin the structural guarantees.

use luminal::op::Runtime;
use luminal::prelude::*;
use skein_emit::op_wiring::wire_attention_math;
use skein_ir::ir::ModelMeta;

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
fn wire_attention_respects_causal_mask_and_rope() {
    let mut cx = Graph::new();

    // Static-shape input: [batch=1, seq=2, hidden=4].
    let input = cx.named_tensor("input", (1usize, 2usize, 4usize));
    let q_w = cx.named_tensor("q_w", (4usize, 4usize));
    let k_w = cx.named_tensor("k_w", (4usize, 4usize));
    let v_w = cx.named_tensor("v_w", (4usize, 4usize));
    let o_w = cx.named_tensor("o_w", (4usize, 4usize));

    let meta = meta_for_tiny_attention();
    let out = wire_attention_math(
        meta.num_attention_heads,
        meta.num_kv_heads,
        meta.head_dim,
        meta.rope_theta,
        0, // prefill: positions 0..seq
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
        1.0_f32, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
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
    assert_eq!(got.len(), 8, "output shape [batch, seq, hidden]");
    assert!(got.iter().all(|x| x.is_finite()), "outputs must be finite");

    // Row 0: causal mask blocks key 1, RoPE@pos0 is identity → exactly V[0].
    let row0 = &got[0..4];
    let expected0 = [1.0_f32, 0.0, 0.0, 0.0];
    for (i, (g, w)) in row0.iter().zip(expected0).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "row0 pos {i}: got {g}, want {w} (causal mask + RoPE@0 identity)"
        );
    }

    // Row 1: convex combination of V[0] and V[1] (V is not rotated).
    let row1 = &got[4..8];
    let (w0, w1) = (row1[0], row1[1]);
    assert!(
        (w0 + w1 - 1.0).abs() < 1e-5,
        "row1 weights sum to 1: {row1:?}"
    );
    assert!(w0 > 0.0 && w0 < 1.0, "row1 attends to key 0: w0={w0}");
    assert!(w1 > 0.0 && w1 < 1.0, "row1 attends to key 1: w1={w1}");
    assert!(
        row1[2].abs() < 1e-6 && row1[3].abs() < 1e-6,
        "row1 tail zero: {row1:?}"
    );
}
