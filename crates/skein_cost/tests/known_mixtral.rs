//! Test 8 — known-ordering on Mixtral 8x7B / 2x H100. Three candidate Plans:
//!
//! - **A:** TP=1 BF16 — replicates weights, overshoots 80 GB.
//! - **B:** TP=2 BF16 — fits memory.
//! - **C:** TP=2 FP8 weights, BF16 KV — fits memory, faster compute than B.

mod common;
use common::*;

use skein_ir::types::*;

#[test]
fn known_optimal_mixtral_2xh100() {
    let model = load_cost_model();
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();

    // Enable CUDA Graphs with a realistic capture-class set so the launch
    // overhead doesn't drown out the FP8 win in the additive sum.
    let classes: Vec<CaptureClass> = (1..=4)
        .map(|i| CaptureClass {
            batch_size: 1 << i,
            kv_class: 1,
        })
        .collect();
    let cg = CudaGraphsConfig {
        enable: true,
        capture_classes: classes,
    };

    let a = plan_with(
        ir.meta.clone(),
        1,
        1,
        1,
        Dtype::Bf16,
        Dtype::Bf16,
        Dtype::Bf16,
        BatchPolicy::Continuous { max_batch: 8 },
        cg.clone(),
    );
    let b = plan_with(
        ir.meta.clone(),
        2,
        1,
        1,
        Dtype::Bf16,
        Dtype::Bf16,
        Dtype::Bf16,
        BatchPolicy::Continuous { max_batch: 8 },
        cg.clone(),
    );
    let c = plan_with(
        ir.meta.clone(),
        2,
        1,
        1,
        Dtype::Fp8E4m3,
        Dtype::Bf16,
        Dtype::Bf16,
        BatchPolicy::Continuous { max_batch: 8 },
        cg,
    );

    let ca = model.total_cost(&a, &ir, &cluster).unwrap().as_us();
    let cb = model.total_cost(&b, &ir, &cluster).unwrap().as_us();
    let cc = model.total_cost(&c, &ir, &cluster).unwrap().as_us();

    // A overshoots → cost dominated by the memory penalty term.
    assert!(ca > 1.0e6, "A (OOM) should be > 1e6 µs, got {ca}");
    // B and C fit (no overshoot term).
    assert!(
        cb < 1.0e5,
        "B (TP=2 BF16) should fit and cost well under 1e5 µs, got {cb}"
    );
    assert!(
        cc < 1.0e5,
        "C (TP=2 FP8 weights) should fit and cost well under 1e5 µs, got {cc}"
    );
    // FP8 weights are faster than BF16 weights at the same TP/KV layout.
    assert!(
        cc < cb,
        "C ({cc}) should beat B ({cb}) — FP8 doubles peak throughput"
    );

    // Magnitude: at decode-step the cost is split across compute (which FP8
    // halves), comm and launch (which FP8 weights don't touch), and bubble
    // (zero here). So the all-up speedup is the *compute-fraction* of the
    // FP8 win. Assert it is non-trivial (at least 5%) and isn't reporting a
    // suspiciously huge improvement (under 2.5×, the dense-peak ceiling).
    let speedup = cb / cc;
    assert!(
        (1.05..=2.5).contains(&speedup),
        "expected modest decode-step FP8 win (1.05–2.5×), got {speedup}"
    );
}
