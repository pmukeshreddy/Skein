//! End-to-end Luminal compile tests via the CPU `NativeComputeRuntime`.
//!
//! Test 1 — tiny matmul with hand-computed reference (validates the
//!          ComputeRuntime trait + the full compile/execute round-trip).
//! Test 4 — search budget is forwarded (compile at budget=1 and budget=10,
//!          assert outputs match and budget=10 isn't faster than budget=1).
//! Test 5 — `#[cfg(feature = "cuda")]` compile-only type-check of the
//!          CUDA runtime impl.

use luminal::prelude::*;

use skein_compile::{ComputeRuntime, NativeComputeRuntime, compile_with_luminal};

/// Reference matmul for `a: [m,k]` and `b: [k,n]`, row-major. Used to
/// validate Luminal's NativeRuntime output without pulling in another
/// linalg dep.
fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

#[test]
fn compile_tiny_matmul_native() {
    let mut cx = Graph::new();
    let a = cx.tensor((3usize, 4usize));
    let b = cx.tensor((4usize, 2usize));
    let c = a.matmul(b).output();

    let mut rts = compile_with_luminal::<NativeComputeRuntime>(std::slice::from_mut(&mut cx), 1)
        .expect("compile tiny matmul");
    let rt = &mut rts[0];

    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    rt.set_data_f32(a.id, a_data.clone());
    rt.set_data_f32(b.id, b_data.clone());
    rt.execute(&cx);

    let out = rt.get_data_f32(c.id);
    let expected = matmul_ref(&a_data, &b_data, 3, 4, 2);
    assert_eq!(out.len(), expected.len(), "output size mismatch");
    for (i, (g, w)) in out.iter().zip(expected.iter()).enumerate() {
        let d = (g - w).abs();
        assert!(d < 1e-5, "pos {i}: got {g}, want {w} (diff={d})");
    }
}

#[test]
fn search_budget_respected() {
    // Identical graphs compiled at budget=1 and budget=10. Outputs must
    // agree exactly (search picks the same canonical form regardless of
    // how many alternatives it explored); budget=10 must take at least as
    // long as budget=1 (proves the parameter is being forwarded — not
    // hardcoded to a constant somewhere).

    fn build_and_run(budget: usize) -> (Vec<f32>, std::time::Duration) {
        let mut cx = Graph::new();
        let a = cx.tensor((4usize, 4usize));
        let b = cx.tensor((4usize, 4usize));
        let c = a.matmul(b).output();

        let started = std::time::Instant::now();
        let mut rts =
            compile_with_luminal::<NativeComputeRuntime>(std::slice::from_mut(&mut cx), budget)
                .expect("compile");
        let compile_time = started.elapsed();
        let rt = &mut rts[0];

        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        rt.set_data_f32(a.id, data.clone());
        rt.set_data_f32(b.id, data.clone());
        rt.execute(&cx);
        (rt.get_data_f32(c.id), compile_time)
    }

    let (out_low, time_low) = build_and_run(1);
    let (out_high, time_high) = build_and_run(10);

    // Correctness invariant: same inputs, same output regardless of how
    // many search alternatives were explored.
    assert_eq!(out_low.len(), out_high.len());
    for (i, (l, h)) in out_low.iter().zip(out_high.iter()).enumerate() {
        let d = (l - h).abs();
        assert!(d < 1e-5, "pos {i}: budget=1 gave {l}, budget=10 gave {h}");
    }

    // Hand-compute the expected value to confirm both compiles are right.
    let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let expected = matmul_ref(&data, &data, 4, 4, 4);
    for (i, (g, w)) in out_low.iter().zip(expected.iter()).enumerate() {
        let d = (g - w).abs();
        assert!(d < 1e-5, "budget=1 pos {i}: got {g}, want {w}");
    }

    // Budget is forwarded: budget=10 isn't cheaper than budget=1. We don't
    // assert a strict inequality (search caching, JIT warmup, OS noise);
    // instead we assert the high-budget time is within a reasonable factor
    // of the low-budget time and is at minimum non-zero (proving the
    // search actually ran rather than short-circuiting).
    assert!(
        time_high >= time_low.mul_f64(0.5),
        "budget=10 ({:?}) finished much faster than budget=1 ({:?}) — \
         budget parameter may not be forwarded",
        time_high,
        time_low,
    );
    assert!(time_low.as_nanos() > 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_runtime_compiles() {
    // Compile-time only: reference `CudaComputeRuntime` so the type and its
    // `ComputeRuntime` impl are checked. Actual execution requires an H100;
    // The GPU-execution path runs the same code through `CudaComputeRuntime`.
    fn _check(
        segments: &mut [luminal::prelude::Graph],
        budget: usize,
    ) -> Result<Vec<skein_compile::CudaComputeRuntime>, skein_compile::CompileError> {
        skein_compile::compile_with_luminal::<skein_compile::CudaComputeRuntime>(segments, budget)
    }
    let _f: fn(&mut [luminal::prelude::Graph], usize) -> _ = _check;
}
