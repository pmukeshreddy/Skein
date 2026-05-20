//! Expert-parallel routing — GShard dispatch/combine round-trip.
//!
//! With identity experts and renormalized top-k gates, combining the
//! dispatched tokens must reconstruct the input exactly: `dispatch` scatters
//! each token into its experts' capacity slots and `combine` gathers them
//! back weighted by gate probabilities that sum to 1. This validates the
//! scatter/gather/capacity math that the AllToAll-based EP path relies on.

use luminal::op::Runtime;
use luminal::prelude::*;
use skein_emit::op_wiring::moe_dispatch_combine;

#[test]
fn dispatch_combine_round_trips_with_identity_experts() {
    let tokens = 4usize;
    let hidden = 3usize;
    let n_experts = 4usize;
    let top_k = 2usize;
    let capacity = 4usize;

    let mut cx = Graph::new();
    let gate_logits = cx.named_tensor("gate", (tokens, n_experts));
    let h = cx.named_tensor("hidden", (tokens, hidden));

    let (dispatch, combine) = moe_dispatch_combine(gate_logits, top_k, n_experts, capacity);
    // dispatched[e*C + c] = the token scattered into expert e's slot c.
    let dispatched = dispatch.permute((1, 0)).matmul(h); // [E*C, hidden]
    // Identity experts: expert_out == dispatched. Combine back to [T, hidden].
    let out = combine.matmul(dispatched).output(); // [tokens, hidden]

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    // Distinct top-2 experts per token; capacity 4 never overflows.
    rt.set_data(
        gate_logits.id,
        vec![
            3.0, 1.0, 0.0, 0.0, // token 0 -> experts 0,1
            0.0, 3.0, 1.0, 0.0, // token 1 -> experts 1,2
            0.0, 0.0, 3.0, 1.0, // token 2 -> experts 2,3
            1.0, 0.0, 0.0, 3.0, // token 3 -> experts 3,0
        ],
    );
    let hidden_data = vec![
        1.0_f32, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, //
        10.0, 11.0, 12.0, //
    ];
    rt.set_data(h.id, hidden_data.clone());
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out.id).clone();

    assert_eq!(got.len(), tokens * hidden);
    for (i, (g, w)) in got.iter().zip(&hidden_data).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "dispatch∘combine must reconstruct input at {i}: got {g}, want {w}",
        );
    }
}

/// Full MoE through the dispatch/combine path with real (batched) SwiGLU
/// experts must equal applying each token's selected expert directly. This
/// exercises the complete expert-parallel compute: scatter → per-expert FFN →
/// gather, the same math the multi-device AllToAll path runs after
/// distributing the dispatched buffer across ranks.
#[test]
fn moe_via_dispatch_equals_direct_expert() {
    let tokens = 2usize;
    let hidden = 2usize;
    let ff = 3usize;
    let n_experts = 2usize;
    let top_k = 1usize; // each token routes to one expert → gate renormalizes to 1
    let capacity = 2usize;

    let mut cx = Graph::new();
    let gate_logits = cx.named_tensor("gate", (tokens, n_experts));
    let h = cx.named_tensor("hidden", (tokens, hidden));
    // Batched expert weights: [n_experts, ...].
    let w1 = cx.named_tensor("w1", (n_experts, hidden, ff));
    let w3 = cx.named_tensor("w3", (n_experts, hidden, ff));
    let w2 = cx.named_tensor("w2", (n_experts, ff, hidden));

    let (dispatch, combine) = moe_dispatch_combine(gate_logits, top_k, n_experts, capacity);
    // Scatter: [n_experts*capacity, hidden] -> [n_experts, capacity, hidden].
    let dispatched = dispatch.permute((1, 0)).matmul(h).split_dims(0, capacity); // [n_experts, capacity, hidden]
    // Batched SwiGLU over experts.
    let gated = dispatched.matmul(w1).silu(); // [E, C, ff]
    let up = dispatched.matmul(w3); // [E, C, ff]
    let down = (gated * up).matmul(w2); // [E, C, hidden]
    let expert_out = down.merge_dims(0, 1); // [E*C, hidden]
    let out = combine.matmul(expert_out).output(); // [tokens, hidden]

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    // token 0 -> expert 0, token 1 -> expert 1.
    let gate_data = vec![3.0_f32, 0.0, 0.0, 3.0];
    let h_data = vec![0.5_f32, -0.5, 1.0, 2.0];
    let w1_data = vec![
        0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6,
    ];
    let w3_data = vec![
        0.2_f32, -0.1, 0.4, 0.0, 0.1, 0.3, 0.5, -0.2, 0.1, 0.3, -0.4, 0.2,
    ];
    let w2_data = vec![
        0.3_f32, -0.2, 0.1, 0.4, -0.5, 0.2, 0.1, 0.0, -0.3, 0.2, 0.5, -0.1,
    ];
    rt.set_data(gate_logits.id, gate_data.clone());
    rt.set_data(h.id, h_data.clone());
    rt.set_data(w1.id, w1_data.clone());
    rt.set_data(w3.id, w3_data.clone());
    rt.set_data(w2.id, w2_data.clone());
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out.id).clone();

    // Reference: each token through its selected expert's SwiGLU (top_k=1).
    let silu = |z: f32| z / (1.0 + (-z).exp());
    let expert_fwd = |e: usize, x: &[f32]| -> Vec<f32> {
        // gate = silu(x @ w1_e), up = x @ w3_e, both [ff]; out = (gate*up) @ w2_e [hidden].
        let mut g = vec![0.0f32; ff];
        let mut u = vec![0.0f32; ff];
        for j in 0..ff {
            for i in 0..hidden {
                g[j] += x[i] * w1_data[e * hidden * ff + i * ff + j];
                u[j] += x[i] * w3_data[e * hidden * ff + i * ff + j];
            }
            g[j] = silu(g[j]);
        }
        let mut o = vec![0.0f32; hidden];
        for hh in 0..hidden {
            for j in 0..ff {
                o[hh] += g[j] * u[j] * w2_data[e * ff * hidden + j * hidden + hh];
            }
        }
        o
    };
    let ref0 = expert_fwd(0, &h_data[0..2]);
    let ref1 = expert_fwd(1, &h_data[2..4]);
    let expected: Vec<f32> = ref0.iter().chain(&ref1).copied().collect();

    assert_eq!(got.len(), tokens * hidden);
    for (i, (g, w)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "MoE-via-dispatch must match direct expert at {i}: got {g}, want {w}",
        );
    }
}
