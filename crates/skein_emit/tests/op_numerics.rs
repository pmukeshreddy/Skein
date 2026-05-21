//! Numeric isolation of the layer-0 attention/MoE math against a pure-Rust
//! reference (which is exactly the HF reference math), on the CPU
//! `NativeRuntime`.
//!
//! These ops are wired in `op_wiring.rs` and are **backend-independent** — the
//! native runtime executes the same op graph the CUDA backend lowers. So if an
//! op's math is wrong in the wiring, it is wrong here too and this test catches
//! it without a GPU or HF. If every op matches the reference to ~f32 noise, the
//! wiring math is proven correct and a remaining Mixtral divergence must live in
//! the CUDA codegen, not the wiring.
//!
//! Each test prints the MSE so the divergence (or lack of it) is a real number,
//! per the debugging brief.

use luminal::prelude::*;
use skein_emit::op_wiring::{apply_rope, attention_fixed_cache, rope_tables, top_k_route};

fn mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32
}

fn run_native(cx: &mut Graph, out: NodeIndex, inputs: &[(NodeIndex, Vec<f32>)]) -> Vec<f32> {
    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    for (id, data) in inputs {
        rt.set_data(*id, data.clone());
    }
    rt.execute(&cx.dyn_map);
    rt.get_f32(out).clone()
}

/// Reference: Mixtral router = softmax over ALL experts, take top-k, renormalize.
/// Equivalent to softmax over the top-k logits (the Z cancels).
fn ref_top_k_route(logits: &[f32], t: usize, e: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t * e];
    for row in 0..t {
        let r = &logits[row * e..(row + 1) * e];
        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_by(|&i, &j| r[j].partial_cmp(&r[i]).unwrap());
        let kept = &idx[..k];
        let max = kept.iter().map(|&i| r[i]).fold(f32::MIN, f32::max);
        let denom: f32 = kept.iter().map(|&i| (r[i] - max).exp()).sum();
        for &i in kept {
            out[row * e + i] = (r[i] - max).exp() / denom;
        }
    }
    out
}

#[test]
fn top_k_route_matches_reference() {
    let (t, e, k) = (4usize, 8usize, 2usize);
    let logits: Vec<f32> = (0..t * e).map(|x| ((x * 37 % 23) as f32) * 0.1 - 1.1).collect();

    let mut cx = Graph::new();
    let l = cx.tensor((t, e));
    let probs = top_k_route(l, k, e, 1).output();
    let got = run_native(&mut cx, probs.id, &[(l.id, logits.clone())]);

    let want = ref_top_k_route(&logits, t, e, k);
    let m = mse(&got, &want);
    eprintln!("[op-numerics] top_k_route MSE = {m:e}");
    // Exactly k nonzeros per row (the selected experts).
    for row in 0..t {
        let nz = (0..e).filter(|&i| got[row * e + i] > 1e-6).count();
        assert_eq!(nz, k, "row {row} selected {nz} experts, expected {k}");
    }
    assert!(m < 1e-6, "top_k_route diverges from Mixtral router: MSE {m:e}");
}

/// Reference rotate-half (NeoX) RoPE on x[b,seq,heads,hd] with theta/offset.
fn ref_rope(
    x: &[f32],
    b: usize,
    seq: usize,
    heads: usize,
    hd: usize,
    theta: f32,
    offset: usize,
) -> Vec<f32> {
    let half = hd / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (-(2.0 * i as f32) / hd as f32 * theta.ln()).exp())
        .collect();
    let mut out = vec![0.0f32; x.len()];
    for bi in 0..b {
        for s in 0..seq {
            let pos = (offset + s) as f32;
            for h in 0..heads {
                let base = ((bi * seq + s) * heads + h) * hd;
                for i in 0..hd {
                    let ang = pos * inv_freq[i % half];
                    let (c, sn) = (ang.cos(), ang.sin());
                    // rotate_half: [-x2, x1] → for i<half pair is -x[i+half]; else +x[i-half]
                    let pair = if i < half {
                        -x[base + i + half]
                    } else {
                        x[base + i - half]
                    };
                    out[base + i] = x[base + i] * c + pair * sn;
                }
            }
        }
    }
    out
}

#[test]
fn rope_matches_reference() {
    let (b, seq, heads, hd) = (1usize, 3usize, 2usize, 8usize);
    let theta = 1_000_000.0f32; // Mixtral rope_theta
    let offset = 5usize; // decode-style absolute position offset
    let x: Vec<f32> = (0..b * seq * heads * hd)
        .map(|n| ((n * 13 % 17) as f32) * 0.05 - 0.4)
        .collect();

    let mut cx = Graph::new();
    let xt = cx.tensor((b, seq, heads, hd));
    let (cos_t, sin_t) = rope_tables(xt.graph(), seq, hd, theta, offset);
    let y = apply_rope(xt, cos_t, sin_t, hd).output();
    let got = run_native(&mut cx, y.id, &[(xt.id, x.clone())]);

    let want = ref_rope(&x, b, seq, heads, hd, theta, offset);
    let m = mse(&got, &want);
    eprintln!("[op-numerics] rope (rotate-half) MSE = {m:e}");
    assert!(m < 1e-6, "RoPE diverges from rotate-half reference: MSE {m:e}");
}

/// Rotate-half RoPE for one [heads, hd] tensor at absolute `pos`.
fn rope_rows(x: &[f32], heads: usize, hd: usize, theta: f32, pos: usize) -> Vec<f32> {
    let half = hd / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (-(2.0 * i as f32) / hd as f32 * theta.ln()).exp())
        .collect();
    let mut out = vec![0.0f32; x.len()];
    for h in 0..heads {
        for i in 0..hd {
            let ang = pos as f32 * inv_freq[i % half];
            let pair = if i < half {
                -x[h * hd + i + half]
            } else {
                x[h * hd + i - half]
            };
            out[h * hd + i] = x[h * hd + i] * ang.cos() + pair * ang.sin();
        }
    }
    out
}

/// Reference for `attention_fixed_cache` (batch=1): rotate q/k_new at `pos`,
/// write the new k/v into cache slot `pos`, mask slots > pos, GQA-attend.
#[allow(clippy::too_many_arguments)]
fn ref_attention_fixed_cache(
    q: &[f32],
    k_new: &[f32],
    v_new: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    pos: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    cap: usize,
    theta: f32,
) -> Vec<f32> {
    let groups = n_heads / n_kv;
    let scale = 1.0 / (hd as f32).sqrt();
    let q_rot = rope_rows(q, n_heads, hd, theta, pos); // [n_heads, hd]
    let k_new_rot = rope_rows(k_new, n_kv, hd, theta, pos); // [n_kv, hd]

    // Build per-slot k/v full views: slot==pos -> new (rotated k / raw v), else cache.
    let kv_dim = n_kv * hd;
    let kf = |s: usize, kv: usize, d: usize| -> f32 {
        if s == pos {
            k_new_rot[kv * hd + d]
        } else {
            k_cache[s * kv_dim + kv * hd + d]
        }
    };
    let vf = |s: usize, kv: usize, d: usize| -> f32 {
        if s == pos {
            v_new[kv * hd + d]
        } else {
            v_cache[s * kv_dim + kv * hd + d]
        }
    };

    let mut out = vec![0.0f32; n_heads * hd];
    for h in 0..n_heads {
        let kv = h / groups;
        let mut scores = vec![0.0f32; cap];
        for s in 0..cap {
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += q_rot[h * hd + d] * kf(s, kv, d);
            }
            scores[s] = dot * scale + if s <= pos { 0.0 } else { -1.0e9 };
        }
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let denom: f32 = exps.iter().sum();
        for d in 0..hd {
            let mut acc = 0.0f32;
            for s in 0..cap {
                acc += (exps[s] / denom) * vf(s, kv, d);
            }
            out[h * hd + d] = acc;
        }
    }
    out
}

#[test]
fn attention_fixed_cache_matches_reference() {
    let (n_heads, n_kv, hd, cap, pos) = (2usize, 1usize, 4usize, 4usize, 2usize);
    let theta = 1_000_000.0f32;
    let kv_dim = n_kv * hd;
    let qd = n_heads * hd;
    let mk = |n: usize, salt: usize| -> Vec<f32> {
        (0..n).map(|x| (((x + salt) * 11 % 13) as f32) * 0.07 - 0.4).collect()
    };
    let q = mk(qd, 1);
    let k_new = mk(kv_dim, 2);
    let v_new = mk(kv_dim, 3);
    let k_cache = mk(cap * kv_dim, 4);
    let v_cache = mk(cap * kv_dim, 5);

    let mut cx = Graph::new();
    let qt = cx.tensor((1usize, 1usize, qd));
    let knt = cx.tensor((1usize, 1usize, kv_dim));
    let vnt = cx.tensor((1usize, 1usize, kv_dim));
    let kct = cx.tensor((1usize, cap, kv_dim));
    let vct = cx.tensor((1usize, cap, kv_dim));
    let post = cx.tensor((1usize,));
    let (attn, _k, _v) = attention_fixed_cache(
        qt, knt, vnt, kct, vct, post, n_heads, n_kv, hd, cap, theta,
    );
    let attn = attn.output();
    let got = run_native(
        &mut cx,
        attn.id,
        &[
            (qt.id, q.clone()),
            (knt.id, k_new.clone()),
            (vnt.id, v_new.clone()),
            (kct.id, k_cache.clone()),
            (vct.id, v_cache.clone()),
            (post.id, vec![pos as f32]),
        ],
    );

    let want = ref_attention_fixed_cache(
        &q, &k_new, &v_new, &k_cache, &v_cache, pos, n_heads, n_kv, hd, cap, theta,
    );
    let m = mse(&got, &want);
    eprintln!("[op-numerics] attention_fixed_cache MSE = {m:e}");
    assert!(m < 1e-5, "attention_fixed_cache diverges from reference: MSE {m:e}");
}

/// Reference RMSNorm: x / sqrt(mean(x^2) + eps), no weight.
fn ref_rmsnorm(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..cols {
            out[r * cols + c] = row[c] * inv;
        }
    }
    out
}

#[test]
fn rmsnorm_matches_reference() {
    let (rows, cols, eps) = (4usize, 16usize, 1e-5f32);
    let x: Vec<f32> = (0..rows * cols).map(|n| ((n * 7 % 19) as f32) * 0.1 - 0.9).collect();

    let mut cx = Graph::new();
    let xt = cx.tensor((rows, cols));
    let normed = xt.std_norm(1, eps).output();
    let got = run_native(&mut cx, normed.id, &[(xt.id, x.clone())]);

    let want = ref_rmsnorm(&x, rows, cols, eps);
    let m = mse(&got, &want);
    eprintln!("[op-numerics] rmsnorm MSE = {m:e}");
    assert!(m < 1e-6, "RMSNorm diverges from reference: MSE {m:e}");
}
