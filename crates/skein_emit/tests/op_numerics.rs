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

/// tp=2 attention sharding decomposition (CPU). The full attention over
/// n_heads/n_kv heads must equal the concatenation of the two per-device shards
/// (device 0 = first half of heads + its kv heads, device 1 = second half),
/// since attention is independent per (kv-head) group. `op_numerics` previously
/// only tested full-width attention; this pins the tp=2 head/kv split that the
/// real 2-GPU forward relies on (q/k/v column-parallel by head, o_proj
/// row-parallel, AllReduce-summed).
#[test]
fn attention_fixed_cache_tp2_shard_matches_full() {
    let (n_heads, n_kv, d, cap, pos) = (8usize, 2usize, 8usize, 4usize, 2i64);
    let kv_groups = n_heads / n_kv; // 4
    let qd = n_heads * d; // 64
    let kvd = n_kv * d; // 16
    let theta = 1.0e6f32;
    let g = |seed: usize, n: usize| {
        (0..n).map(move |i| (((i + seed) * 1103515245 % 997) as f32 / 997.0 - 0.5)).collect::<Vec<f32>>()
    };
    let q = g(1, qd);
    let kn = g(2, kvd);
    let vn = g(3, kvd);
    let kc = g(4, cap * kvd);
    let vc = g(5, cap * kvd);

    // Run attention_fixed_cache for given local head counts + inputs.
    let run_attn = |nh: usize, nkv: usize, q: &[f32], kn: &[f32], vn: &[f32], kc: &[f32], vc: &[f32]| -> Vec<f32> {
        let mut cx = Graph::new();
        let qg = cx.tensor((1, 1, nh * d));
        let kng = cx.tensor((1, 1, nkv * d));
        let vng = cx.tensor((1, 1, nkv * d));
        let kcg = cx.tensor((1, cap, nkv * d));
        let vcg = cx.tensor((1, cap, nkv * d));
        let position = cx.tensor((1,));
        let (attn, _ks, _vs) = attention_fixed_cache(qg, kng, vng, kcg, vcg, position, nh, nkv, d, cap, theta);
        let out = attn.output();
        run_native(&mut cx, out.id, &[
            (qg.id, q.to_vec()), (kng.id, kn.to_vec()), (vng.id, vn.to_vec()),
            (kcg.id, kc.to_vec()), (vcg.id, vc.to_vec()), (position.id, vec![pos as f32]),
        ])
    };

    let full = run_attn(n_heads, n_kv, &q, &kn, &vn, &kc, &vc);

    // Build device d's shard: heads [d*nh_loc, (d+1)*nh_loc), kv [d*nkv_loc, ...).
    let nh_loc = n_heads / 2;
    let nkv_loc = n_kv / 2;
    let _ = kv_groups;
    let mut sharded = Vec::new();
    for dev in 0..2 {
        let qh0 = dev * nh_loc * d;
        let q_dev = q[qh0..qh0 + nh_loc * d].to_vec();
        let kv0 = dev * nkv_loc * d;
        let kn_dev = kn[kv0..kv0 + nkv_loc * d].to_vec();
        let vn_dev = vn[kv0..kv0 + nkv_loc * d].to_vec();
        // cache: per slot, take this device's kv-head columns.
        let slice_cache = |c: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(cap * nkv_loc * d);
            for s in 0..cap {
                let base = s * kvd + dev * nkv_loc * d;
                out.extend_from_slice(&c[base..base + nkv_loc * d]);
            }
            out
        };
        let kc_dev = slice_cache(&kc);
        let vc_dev = slice_cache(&vc);
        let out_dev = run_attn(nh_loc, nkv_loc, &q_dev, &kn_dev, &vn_dev, &kc_dev, &vc_dev);
        sharded.extend_from_slice(&out_dev);
    }

    let m = mse(&full, &sharded);
    eprintln!("[op-numerics] tp2 attention shard-vs-full MSE = {m:e} (full_len={}, sharded_len={})", full.len(), sharded.len());
    assert_eq!(full.len(), sharded.len(), "shard concat length mismatch");
    assert!(m < 1e-6, "tp=2 attention sharding diverges from full: MSE {m:e}");
}

/// Vocab-parallel embedding (tp=2) decomposition on CPU. The full single-table
/// embedding of a token must equal the AllReduce-sum of the two per-device
/// vocab-parallel lookups (each masks out-of-range tokens to zero). This pins
/// the masking/clamp the real 2-GPU embedding relies on; `op_numerics`
/// previously only tested the full single-device gather.
#[test]
fn vocab_parallel_embed_tp2_matches_full() {
    use skein_emit::op_wiring::{embedding_lookup, vocab_parallel_embed};
    let vocab = 64usize;
    let hidden = 8usize;
    let table: Vec<f32> = (0..vocab * hidden).map(|i| (i % 31) as f32 * 0.1 - 1.5).collect();
    for &token in &[20i32, 37i32, 0i32, 63i32] {
        // Full single-device lookup.
        let mut cx = Graph::new();
        let w = cx.tensor((vocab, hidden));
        let toks = cx.tensor((1, 1));
        cx.get_op_mut::<luminal::hlir::Input>(toks.id).dtype = luminal::prelude::DType::Int;
        let out = embedding_lookup(toks.as_dtype(luminal::prelude::DType::Int), w, 1, 1, hidden).output();
        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);
        rt.set_data(w.id, table.clone());
        rt.set_data(toks.id, vec![token]);
        rt.execute(&cx.dyn_map);
        let full = rt.get_f32(out.id).clone();

        // Two vocab-parallel shards, summed (= the AllReduce the runtime does).
        let half = vocab / 2;
        let run_shard = |start: usize| -> Vec<f32> {
            let mut cx = Graph::new();
            let w = cx.tensor((half, hidden));
            let toks = cx.tensor((1, 1));
            cx.get_op_mut::<luminal::hlir::Input>(toks.id).dtype = luminal::prelude::DType::Int;
            let out = vocab_parallel_embed(
                toks.as_dtype(luminal::prelude::DType::Int), w, start, half, 1, 1, hidden,
            ).output();
            cx.build_search_space::<NativeRuntime>();
            let mut rt = cx.search(NativeRuntime::default(), 1);
            rt.set_data(w.id, table[start * hidden..(start + half) * hidden].to_vec());
            rt.set_data(toks.id, vec![token]);
            rt.execute(&cx.dyn_map);
            rt.get_f32(out.id).clone()
        };
        let d0 = run_shard(0);
        let d1 = run_shard(half);
        let summed: Vec<f32> = d0.iter().zip(&d1).map(|(a, b)| a + b).collect();
        let m = mse(&full, &summed);
        eprintln!("[op-numerics] vocab-parallel embed token={token} MSE={m:e}");
        assert!(m < 1e-8, "vocab-parallel embed diverges from full for token {token}: MSE {m:e}");
    }
}
