//! Paged KV cache — incremental decode equals full prefill.
//!
//! The defining correctness property of a KV cache: decoding token-by-token
//! while accumulating the cache must produce exactly the same per-token
//! attention output as running the whole sequence at once with a causal mask.
//! We build both in one graph (the decode steps chain their `k_full`/`v_full`
//! outputs as the next step's cache) and compare.

use luminal::op::Runtime;
use luminal::prelude::*;
use skein_emit::op_wiring::attention_with_kv_cache;

#[test]
fn decode_with_cache_matches_full_prefill() {
    let n_heads = 2usize;
    let n_kv_heads = 2usize;
    let head_dim = 2usize;
    let theta = 10_000.0_f32;
    let qd = n_heads * head_dim; // 4
    let kvd = n_kv_heads * head_dim; // 4
    let seq = 3usize;

    let mut cx = Graph::new();
    let q = cx.named_tensor("q", (1usize, seq, qd));
    let k = cx.named_tensor("k", (1usize, seq, kvd));
    let v = cx.named_tensor("v", (1usize, seq, kvd));

    // Full prefill over all `seq` tokens at once (past = 0).
    let (out_full, _, _) =
        attention_with_kv_cache(q, k, v, q, v, n_heads, n_kv_heads, head_dim, 0, theta);
    let out_full = out_full.output();

    // Incremental decode: one token per step, chaining the cache.
    let qt = |i: usize| q.slice((.., i..i + 1, ..));
    let kt = |i: usize| k.slice((.., i..i + 1, ..));
    let vt = |i: usize| v.slice((.., i..i + 1, ..));

    let (o0, k0, v0) = attention_with_kv_cache(
        qt(0),
        kt(0),
        vt(0),
        kt(0),
        vt(0),
        n_heads,
        n_kv_heads,
        head_dim,
        0,
        theta,
    );
    let (o1, k1, v1) = attention_with_kv_cache(
        qt(1),
        kt(1),
        vt(1),
        k0,
        v0,
        n_heads,
        n_kv_heads,
        head_dim,
        1,
        theta,
    );
    let (o2, _k2, _v2) = attention_with_kv_cache(
        qt(2),
        kt(2),
        vt(2),
        k1,
        v1,
        n_heads,
        n_kv_heads,
        head_dim,
        2,
        theta,
    );
    let o0 = o0.output();
    let o1 = o1.output();
    let o2 = o2.output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    // Deterministic, distinct per-(token, channel) values.
    let fill = |base: f32| {
        (0..seq * qd)
            .map(|i| base + i as f32 * 0.1)
            .collect::<Vec<f32>>()
    };
    rt.set_data(q.id, fill(0.0));
    rt.set_data(k.id, fill(1.0));
    rt.set_data(v.id, fill(2.0));
    rt.execute(&cx.dyn_map);

    let full = rt.get_f32(out_full.id).clone();
    let dec0 = rt.get_f32(o0.id).clone();
    let dec1 = rt.get_f32(o1.id).clone();
    let dec2 = rt.get_f32(o2.id).clone();
    assert_eq!(full.len(), seq * qd);

    let check = |tok: usize, dec: &[f32]| {
        for c in 0..qd {
            let f = full[tok * qd + c];
            let d = dec[c];
            assert!(
                (f - d).abs() < 1e-4,
                "token {tok} chan {c}: prefill {f} vs decode {d}",
            );
        }
    };
    check(0, &dec0);
    check(1, &dec1);
    check(2, &dec2);
}
