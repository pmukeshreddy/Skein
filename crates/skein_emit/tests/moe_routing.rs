//! Top-k MoE routing — hand-computed reference.
//!
//! Drives `op_wiring::top_k_route` through Luminal's `NativeRuntime` and
//! checks that only the top-k experts receive weight and that those weights
//! renormalize to 1 (i.e. `softmax` over just the selected logits).

use luminal::op::Runtime;
use luminal::prelude::*;
use skein_emit::op_wiring::top_k_route;

#[test]
fn top_k_route_selects_and_renormalizes_top2() {
    let mut cx = Graph::new();
    // One token, four experts. Logits chosen so the top-2 are experts 1 and 2.
    let logits = cx.named_tensor("logits", (1usize, 4usize));
    let probs = top_k_route(logits, 2, 4, 1).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    rt.set_data(logits.id, vec![1.0_f32, 3.0, 2.0, 0.5]);
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(probs.id).clone();

    assert_eq!(got.len(), 4);
    // Top-2 are experts 1 (logit 3.0) and 2 (logit 2.0); softmax over {3, 2}
    // is [e/(e+1), 1/(e+1)]. Experts 0 and 3 must be exactly zero-weight.
    let e = std::f32::consts::E;
    let expected = [0.0, e / (e + 1.0), 1.0 / (e + 1.0), 0.0];
    for (i, (g, w)) in got.iter().zip(expected).enumerate() {
        assert!((g - w).abs() < 1e-5, "expert {i}: got {g}, want {w}");
    }
    let sum: f32 = got.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "top-k weights must renormalize to 1, got {sum}"
    );
}

#[test]
fn top_k_route_k_at_or_above_experts_is_plain_softmax() {
    let mut cx = Graph::new();
    let logits = cx.named_tensor("logits", (1usize, 3usize));
    let probs = top_k_route(logits, 3, 3, 1).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    rt.set_data(logits.id, vec![0.0_f32, 0.0, 0.0]);
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(probs.id).clone();

    for (i, g) in got.iter().enumerate() {
        assert!(
            (g - 1.0 / 3.0).abs() < 1e-5,
            "uniform logits → uniform softmax; expert {i} got {g}"
        );
    }
}
