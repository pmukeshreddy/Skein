//! CUDA-vs-CPU op isolation at real Mixtral sizes.
//!
//! `op_numerics` (in skein_emit) proved the op *wiring* math matches the HF
//! reference on the CPU `NativeRuntime`. The remaining Mixtral parity failure
//! is therefore in the **CUDA codegen/execution** of that (correct) graph. This
//! test runs the *identical* luminal graph on both `NativeComputeRuntime` (CPU,
//! the proven-correct reference) and `CudaComputeRuntime` (GPU) and reports the
//! MSE between them per op. The first op whose CUDA output diverges from the CPU
//! output far above the bf16 noise floor is the buggy codegen.
//!
//! Requires a GPU (`--features cuda`).
#![cfg(feature = "cuda")]

use luminal::prelude::*;
use skein_compile::{ComputeRuntime, CudaComputeRuntime, NativeComputeRuntime};
use skein_emit::op_wiring::{attention_fixed_cache, embedding_lookup, rms_norm};

fn mse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>()
        / n as f64
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Build a graph via `build`, then run it on backend `R`, returning the output
/// `Vec<f32>`. `build` returns `(out_node, f32_inputs, i32_inputs)`.
struct Inputs {
    f32: Vec<(NodeIndex, Vec<f32>)>,
    i32: Vec<(NodeIndex, Vec<i32>)>,
}
impl Inputs {
    fn f(v: Vec<(NodeIndex, Vec<f32>)>) -> Self {
        Self { f32: v, i32: vec![] }
    }
}

fn run<R: ComputeRuntime>(build: impl Fn(&mut Graph) -> (NodeIndex, Inputs)) -> Vec<f32> {
    let mut cx = Graph::new();
    let (out, ins) = build(&mut cx);
    // Zero-buffer every Input (4 bytes/elem) so the CUDA search can execute
    // candidate graphs to measure them.
    let input_zeros: Vec<(NodeIndex, usize)> = ins
        .f32
        .iter()
        .map(|(id, d)| (*id, d.len() * 4))
        .chain(ins.i32.iter().map(|(id, d)| (*id, d.len() * 4)))
        .collect();
    let mut rt = R::build_and_search_with_input_zeros(&mut cx, 1, &input_zeros)
        .expect("build_and_search");
    for (id, d) in &ins.f32 {
        rt.set_data_f32(*id, d.clone());
    }
    for (id, d) in &ins.i32 {
        rt.set_data_i32(*id, d.clone());
    }
    rt.execute(&cx);
    rt.get_data_f32(out)
}

/// Create an integer input tensor (matching how the runtime feeds `input_tokens`).
fn int_input(cx: &mut Graph, shape: impl luminal::shape::ToShape) -> GraphTensor {
    let t = cx.tensor(shape);
    cx.get_op_mut::<luminal::hlir::Input>(t.id).dtype = DType::Int;
    t.as_dtype(DType::Int)
}

const HIDDEN: usize = 4096;

/// Embedding gather at real hidden width. The gather indexes a flattened
/// `[vocab, hidden]` table by `token*hidden + col`. A wrong CUDA gather returns
/// a different/garbled row → the whole forward decorrelates from token 0.
#[test]
fn embedding_gather_cuda_matches_cpu() {
    let vocab = 64usize;
    let token = 37i32;
    // Deterministic table: row r, col c = (r*hidden + c) scaled small.
    let table: Vec<f32> = (0..vocab * HIDDEN).map(|i| (i % 101) as f32 * 0.01 - 0.5).collect();
    let build = |cx: &mut Graph| {
        let w = cx.tensor((vocab, HIDDEN));
        let toks = int_input(cx, (1, 1));
        let out = embedding_lookup(toks, w, 1, 1, HIDDEN).output();
        (
            out.id,
            Inputs {
                f32: vec![(w.id, table.clone())],
                i32: vec![(toks.id, vec![token])],
            },
        )
    };
    let cpu = run::<NativeComputeRuntime>(build);
    let gpu = run::<CudaComputeRuntime>(build);
    let want = &table[token as usize * HIDDEN..(token as usize + 1) * HIDDEN];
    eprintln!(
        "[cuda-vs-cpu] embedding: len cpu={} gpu={} | cpu-vs-want MSE={:.3e} | gpu-vs-cpu MSE={:.3e} cos={:.5}",
        cpu.len(),
        gpu.len(),
        mse(&cpu, want),
        mse(&gpu, &cpu),
        cos(&gpu, &cpu),
    );
    assert_eq!(cpu.len(), HIDDEN, "cpu embed wrong length");
    assert_eq!(gpu.len(), HIDDEN, "gpu embed wrong length");
    assert!(mse(&cpu, want) < 1e-8, "CPU embedding itself is wrong");
    assert!(
        mse(&gpu, &cpu) < 1e-3,
        "CUDA embedding diverges from CPU: MSE={:.3e} cos={:.5}",
        mse(&gpu, &cpu),
        cos(&gpu, &cpu)
    );
}

/// RMSNorm at real hidden width, CUDA vs CPU.
#[test]
fn rms_norm_cuda_matches_cpu() {
    let x: Vec<f32> = (0..HIDDEN).map(|i| ((i * 31 % 97) as f32) * 0.02 - 0.9).collect();
    let w: Vec<f32> = (0..HIDDEN).map(|i| 0.5 + (i % 13) as f32 * 0.03).collect();
    let build = |cx: &mut Graph| {
        let xt = cx.tensor((1, 1, HIDDEN));
        let wt = cx.tensor((HIDDEN,));
        let out = rms_norm(xt, wt, 1e-5).output();
        (out.id, Inputs::f(vec![(xt.id, x.clone()), (wt.id, w.clone())]))
    };
    let cpu = run::<NativeComputeRuntime>(build);
    let gpu = run::<CudaComputeRuntime>(build);
    eprintln!(
        "[cuda-vs-cpu] rms_norm: gpu-vs-cpu MSE={:.3e} cos={:.5}",
        mse(&gpu, &cpu),
        cos(&gpu, &cpu)
    );
    assert!(
        mse(&gpu, &cpu) < 1e-3,
        "CUDA rms_norm diverges from CPU: MSE={:.3e}",
        mse(&gpu, &cpu)
    );
}

/// Linear projection (matmul) at real width: [1,1,4096] @ [4096,4096]^T.
#[test]
fn matmul_cuda_matches_cpu() {
    let din = HIDDEN;
    let dout = 512usize; // like a sharded k_proj
    let x: Vec<f32> = (0..din).map(|i| ((i * 17 % 89) as f32) * 0.01 - 0.4).collect();
    let w: Vec<f32> = (0..dout * din).map(|i| ((i * 53 % 71) as f32) * 0.003 - 0.1).collect();
    let build = |cx: &mut Graph| {
        let xt = cx.tensor((1, 1, din));
        let wt = cx.tensor((dout, din));
        let out = xt.matmul(wt.permute((1, 0))).output();
        (out.id, Inputs::f(vec![(xt.id, x.clone()), (wt.id, w.clone())]))
    };
    let cpu = run::<NativeComputeRuntime>(build);
    let gpu = run::<CudaComputeRuntime>(build);
    eprintln!(
        "[cuda-vs-cpu] matmul: len cpu={} gpu={} gpu-vs-cpu MSE={:.3e} cos={:.5}",
        cpu.len(),
        gpu.len(),
        mse(&gpu, &cpu),
        cos(&gpu, &cpu)
    );
    assert!(
        mse(&gpu, &cpu) < 1e-2,
        "CUDA matmul diverges from CPU: MSE={:.3e}",
        mse(&gpu, &cpu)
    );
}

/// Cached-decode attention at real head dims (16 q-heads, 4 kv-heads, d=128).
/// Exercises the fused KV-cache select (stride-0 broadcast [b,1,kv]->[b,C,kv]),
/// RoPE at the runtime position, GQA expansion, masked softmax, and the
/// permute/merge — the ops `op_numerics` validated on CPU. If CUDA diverges
/// here, the parity bug is in this region's codegen (task hypotheses 1-3).
#[test]
fn attention_fixed_cache_cuda_matches_cpu() {
    let (n_heads, n_kv, d, cap) = (16usize, 4usize, 128usize, 16usize);
    let qd = n_heads * d; // 2048
    let kvd = n_kv * d; // 512
    let pos = 5i64;
    let rng = |seed: usize, n: usize, s: f32| {
        (0..n).map(move |i| (((i + seed) * 2654435761 % 1009) as f32 / 1009.0 - 0.5) * s).collect::<Vec<f32>>()
    };
    let q = rng(1, qd, 1.0);
    let k_new = rng(2, kvd, 1.0);
    let v_new = rng(3, kvd, 1.0);
    let k_cache = rng(4, cap * kvd, 1.0);
    let v_cache = rng(5, cap * kvd, 1.0);
    let build = |cx: &mut Graph| {
        let qg = cx.tensor((1, 1, qd));
        let kn = cx.tensor((1, 1, kvd));
        let vn = cx.tensor((1, 1, kvd));
        let kc = cx.tensor((1, cap, kvd));
        let vc = cx.tensor((1, cap, kvd));
        let position = cx.tensor((1,));
        let (attn, _ks, _vs) =
            attention_fixed_cache(qg, kn, vn, kc, vc, position, n_heads, n_kv, d, cap, 1.0e6);
        let out = attn.output();
        (
            out.id,
            Inputs::f(vec![
                (qg.id, q.clone()),
                (kn.id, k_new.clone()),
                (vn.id, v_new.clone()),
                (kc.id, k_cache.clone()),
                (vc.id, v_cache.clone()),
                (position.id, vec![pos as f32]),
            ]),
        )
    };
    let cpu = run::<NativeComputeRuntime>(build);
    let gpu = run::<CudaComputeRuntime>(build);
    eprintln!(
        "[cuda-vs-cpu] attention: len cpu={} gpu={} gpu-vs-cpu MSE={:.3e} cos={:.5}",
        cpu.len(),
        gpu.len(),
        mse(&gpu, &cpu),
        cos(&gpu, &cpu)
    );
    assert!(
        mse(&gpu, &cpu) < 1e-2,
        "CUDA attention diverges from CPU: MSE={:.3e} cos={:.5}",
        mse(&gpu, &cpu),
        cos(&gpu, &cpu)
    );
}

/// bf16 matmul (the production projection pattern: bf16 inputs, fp32 accumulate,
/// bf16 output). The real forward casts activations + weights to bf16; an f32-
/// only test misses a bf16-specific codegen bug (task hypothesis 3).
#[test]
fn matmul_bf16_cuda_matches_cpu() {
    let din = HIDDEN;
    let dout = 512usize;
    let x: Vec<f32> = (0..din).map(|i| ((i * 17 % 89) as f32) * 0.01 - 0.4).collect();
    let w: Vec<f32> = (0..dout * din).map(|i| ((i * 53 % 71) as f32) * 0.003 - 0.1).collect();
    let build = |cx: &mut Graph| {
        let xt = cx.tensor((1, 1, din)).cast(DType::Bf16);
        let wt = cx.tensor((dout, din)).cast(DType::Bf16);
        let out = xt.matmul(wt.permute((1, 0))).cast(DType::F32).output();
        (out.id, Inputs::f(vec![(xt_src(&xt), x.clone()), (wt_src(&wt), w.clone())]))
    };
    // The .cast() makes xt/wt derived nodes; we need the *input* node ids. Rebuild
    // capturing input ids explicitly.
    let _ = build;
    let build = |cx: &mut Graph| {
        let xin = cx.tensor((1, 1, din));
        let win = cx.tensor((dout, din));
        let out = xin.cast(DType::Bf16).matmul(win.cast(DType::Bf16).permute((1, 0))).cast(DType::F32).output();
        (out.id, Inputs::f(vec![(xin.id, x.clone()), (win.id, w.clone())]))
    };
    let cpu = run::<NativeComputeRuntime>(build);
    let gpu = run::<CudaComputeRuntime>(build);
    eprintln!(
        "[cuda-vs-cpu] matmul_bf16: len cpu={} gpu={} gpu-vs-cpu MSE={:.3e} cos={:.5}",
        cpu.len(), gpu.len(), mse(&gpu, &cpu), cos(&gpu, &cpu)
    );
    assert!(mse(&gpu, &cpu) < 1e-2, "CUDA bf16 matmul diverges from CPU: MSE={:.3e} cos={:.5}", mse(&gpu, &cpu), cos(&gpu, &cpu));
}

fn xt_src(t: &GraphTensor) -> NodeIndex { t.id }
fn wt_src(t: &GraphTensor) -> NodeIndex { t.id }

/// bf16 rms_norm -> bf16 matmul chain (post_input_layernorm -> q_proj), CUDA vs
/// CPU, mirroring the real activation dtype flow.
#[test]
fn rmsnorm_then_matmul_bf16_cuda_matches_cpu() {
    let x: Vec<f32> = (0..HIDDEN).map(|i| ((i * 31 % 97) as f32) * 0.02 - 0.9).collect();
    let nw: Vec<f32> = (0..HIDDEN).map(|i| 0.5 + (i % 13) as f32 * 0.03).collect();
    let dout = 512usize;
    let qw: Vec<f32> = (0..dout * HIDDEN).map(|i| ((i * 53 % 71) as f32) * 0.003 - 0.1).collect();
    let build = |cx: &mut Graph| {
        let xin = cx.tensor((1, 1, HIDDEN)).cast(DType::Bf16);
        let nin = cx.tensor((HIDDEN,)).cast(DType::Bf16);
        let win = cx.tensor((dout, HIDDEN)).cast(DType::Bf16);
        let normed = rms_norm(xin, nin, 1e-5);
        let out = normed.matmul(win.permute((1, 0))).cast(DType::F32).output();
        // capture input ids: the .cast nodes' producers
        (out.id, Inputs::f(vec![]))
    };
    let _ = build;
    let build = |cx: &mut Graph| {
        let xin = cx.tensor((1, 1, HIDDEN));
        let nin = cx.tensor((HIDDEN,));
        let win = cx.tensor((dout, HIDDEN));
        let normed = rms_norm(xin.cast(DType::Bf16), nin.cast(DType::Bf16), 1e-5);
        let out = normed.matmul(win.cast(DType::Bf16).permute((1, 0))).cast(DType::F32).output();
        (out.id, Inputs::f(vec![(xin.id, x.clone()), (nin.id, nw.clone()), (win.id, qw.clone())]))
    };
    let cpu = run::<NativeComputeRuntime>(build);
    let gpu = run::<CudaComputeRuntime>(build);
    eprintln!(
        "[cuda-vs-cpu] rmsnorm+matmul bf16: len cpu={} gpu={} gpu-vs-cpu MSE={:.3e} cos={:.5}",
        cpu.len(), gpu.len(), mse(&gpu, &cpu), cos(&gpu, &cpu)
    );
    assert!(mse(&gpu, &cpu) < 5e-2, "CUDA bf16 rmsnorm+matmul diverges from CPU: MSE={:.3e} cos={:.5}", mse(&gpu, &cpu), cos(&gpu, &cpu));
}

/// MoE expert + top-k routing combine (bf16), CUDA vs CPU. Replicates
/// `wire_block_moe` for 2 experts: per expert `down = (silu(x@w1) * (x@w3)) @ w2`
/// (a multi-input elementwise fusion of two matmul results), weighted by the
/// renormalized router prob (a stride-0 [.,.,1]->[.,.,hidden] broadcast in a
/// fused region) and summed. Exercises task hypotheses 1 (stride-0 broadcast in
/// a fused region) and 2 (multi-input elementwise region assembly).
#[test]
fn moe_expert_combine_bf16_cuda_matches_cpu() {
    use skein_emit::op_wiring::top_k_route;
    let hidden = HIDDEN;
    let inter = 14336usize;
    let n_exp = 4usize;
    let top_k = 2usize;
    let g = |seed: usize, n: usize, s: f32| {
        (0..n).map(move |i| ((((i + seed) * 2654435761) % 2003) as f32 / 2003.0 - 0.5) * s).collect::<Vec<f32>>()
    };
    let x = g(1, hidden, 1.0);
    let gate_w = g(2, n_exp * hidden, 0.2);
    let w1: Vec<Vec<f32>> = (0..n_exp).map(|e| g(10 + e, inter * hidden, 0.05)).collect();
    let w3: Vec<Vec<f32>> = (0..n_exp).map(|e| g(20 + e, inter * hidden, 0.05)).collect();
    let w2: Vec<Vec<f32>> = (0..n_exp).map(|e| g(30 + e, hidden * inter, 0.05)).collect();
    let build = |cx: &mut Graph| {
        let xin = cx.tensor((1, 1, hidden));
        let gin = cx.tensor((n_exp, hidden));
        let xb = xin.cast(DType::Bf16);
        let routing_logits = xb.matmul(gin.cast(DType::Bf16).permute((1, 0)));
        let probs = top_k_route(routing_logits, top_k, n_exp, 2);
        let mut inputs = vec![(xin.id, x.clone()), (gin.id, gate_w.clone())];
        let mut acc: Option<GraphTensor> = None;
        for e in 0..n_exp {
            let w1t = cx.tensor((inter, hidden));
            let w3t = cx.tensor((inter, hidden));
            let w2t = cx.tensor((hidden, inter));
            inputs.push((w1t.id, w1[e].clone()));
            inputs.push((w3t.id, w3[e].clone()));
            inputs.push((w2t.id, w2[e].clone()));
            let gate_val = xb.matmul(w1t.cast(DType::Bf16).permute((1, 0))).silu();
            let up_val = xb.matmul(w3t.cast(DType::Bf16).permute((1, 0)));
            let down = (gate_val * up_val).matmul(w2t.cast(DType::Bf16).permute((1, 0)));
            let mut prob_e = probs.slice((.., .., e..e + 1));
            prob_e.shape.expand(down.dims());
            let weighted = down * prob_e;
            acc = Some(match acc {
                None => weighted,
                Some(a) => a + weighted,
            });
        }
        let out = acc.unwrap().cast(DType::F32).output();
        (out.id, Inputs::f(inputs))
    };
    let cpu = run::<NativeComputeRuntime>(build);
    let gpu = run::<CudaComputeRuntime>(build);
    eprintln!(
        "[cuda-vs-cpu] moe_expert_combine bf16: len cpu={} gpu={} gpu-vs-cpu MSE={:.3e} cos={:.5}",
        cpu.len(), gpu.len(), mse(&gpu, &cpu), cos(&gpu, &cpu)
    );
    assert!(mse(&gpu, &cpu) < 5e-2, "CUDA MoE diverges from CPU: MSE={:.3e} cos={:.5}", mse(&gpu, &cpu), cos(&gpu, &cpu));
}

/// Vocab-parallel embedding on CUDA vs CPU (real hidden). Unlike the plain
/// gather test, this exercises clamp + gather + the range-mask BROADCAST
/// multiply (`embeds * mask`, mask [.,.,1]->[.,.,hidden] stride-0) — a fused
/// multi-input region (task hypotheses 1+2). This is the real layer-0 input op.
#[test]
fn vocab_parallel_embed_cuda_matches_cpu() {
    use skein_emit::op_wiring::vocab_parallel_embed;
    let vocab_local = 64usize;
    let vocab_start = 64usize; // device 1's range [64,128); test in-range + out-of-range
    for &token in &[80i32, 10i32] {
        let table: Vec<f32> = (0..vocab_local * HIDDEN).map(|i| (i % 97) as f32 * 0.01 - 0.5).collect();
        let build = |cx: &mut Graph| {
            let w = cx.tensor((vocab_local, HIDDEN));
            let toks = int_input(cx, (1, 1));
            let out = vocab_parallel_embed(toks, w, vocab_start, vocab_local, 1, 1, HIDDEN).output();
            (out.id, Inputs { f32: vec![(w.id, table.clone())], i32: vec![(toks.id, vec![token])] })
        };
        let cpu = run::<NativeComputeRuntime>(build);
        let gpu = run::<CudaComputeRuntime>(build);
        eprintln!(
            "[cuda-vs-cpu] vocab_parallel_embed token={token} (start={vocab_start}): len cpu={} gpu={} MSE={:.3e} cos={:.5} cpu_norm={:.3} gpu_norm={:.3}",
            cpu.len(), gpu.len(), mse(&gpu,&cpu), cos(&gpu,&cpu),
            (cpu.iter().map(|x|(*x as f64).powi(2)).sum::<f64>()).sqrt(),
            (gpu.iter().map(|x|(*x as f64).powi(2)).sum::<f64>()).sqrt(),
        );
        assert!(mse(&gpu,&cpu) < 1e-3, "CUDA vocab_parallel_embed diverges from CPU token={token}: MSE={:.3e} cos={:.5}", mse(&gpu,&cpu), cos(&gpu,&cpu));
    }
}

/// Embedding gather of a *bf16* table (the real model's weight dtype) — the
/// earlier gather test used an f32 table and passed. If CUDA diverges here, the
/// gather codegen mishandles bf16 element size (the Mixtral embedding feeds a
/// bf16 weight).
#[test]
fn embedding_gather_bf16_cuda_matches_cpu() {
    let vocab = 64usize;
    let token = 37i32;
    let table: Vec<f32> = (0..vocab * HIDDEN).map(|i| (i % 101) as f32 * 0.01 - 0.5).collect();
    let build = |cx: &mut Graph| {
        let w = cx.tensor((vocab, HIDDEN));
        let wb = w.cast(DType::Bf16);
        let toks = int_input(cx, (1, 1));
        let out = embedding_lookup(toks, wb, 1, 1, HIDDEN).cast(DType::F32).output();
        (out.id, Inputs { f32: vec![(w.id, table.clone())], i32: vec![(toks.id, vec![token])] })
    };
    let cpu = run::<NativeComputeRuntime>(build);
    let gpu = run::<CudaComputeRuntime>(build);
    let even_zeros = gpu.iter().step_by(2).filter(|&&x| x == 0.0).count();
    eprintln!(
        "[cuda-vs-cpu] embedding_bf16: len cpu={} gpu={} gpu-vs-cpu MSE={:.3e} cos={:.5} | gpu even-index zeros={}/{}",
        cpu.len(), gpu.len(), mse(&gpu,&cpu), cos(&gpu,&cpu), even_zeros, gpu.len()/2
    );
    eprintln!("  cpu[:6]={:?}", &cpu[..6.min(cpu.len())]);
    eprintln!("  gpu[:6]={:?}", &gpu[..6.min(gpu.len())]);
    assert!(mse(&gpu,&cpu) < 1e-3, "CUDA bf16 embedding gather diverges: MSE={:.3e} cos={:.5}", mse(&gpu,&cpu), cos(&gpu,&cpu));
}
