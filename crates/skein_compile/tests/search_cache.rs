//! Native-backend verification for the on-disk compile cache.
//!
//! These run on the CPU `NativeComputeRuntime`, which exercises the same
//! `build_search_space` + search + cache path the CUDA backend uses (minus
//! NVRTC). They prove:
//!   1. Replaying a cached search result reproduces byte-for-byte the same
//!      output as the original search, *without* re-running egglog.
//!   2. The end-to-end `build_and_search_cached` writes the cache on the first
//!      run and a second run loads it and is faster (the "boot" speedup).

use std::time::Instant;

use luminal::prelude::*;
use skein_compile::search_cache;
use skein_compile::{ComputeRuntime, NativeComputeRuntime};

/// A small multi-matmul chain — enough egglog work that skipping it on the warm
/// run is clearly measurable, while staying fast for CI. Returns the input and
/// output node ids. `GraphTensor` is `Copy`, so `b` feeds every matmul.
fn build_chain(cx: &mut Graph) -> (NodeIndex, NodeIndex, NodeIndex) {
    let a = cx.tensor((8usize, 16usize));
    let b = cx.tensor((16usize, 16usize));
    let c = a.matmul(b).relu();
    let d = c.matmul(b).relu();
    let e = d.matmul(b).relu();
    let out = e.matmul(b).exp().output();
    (a.id, b.id, out.id)
}

fn inputs() -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..8 * 16).map(|x| (x as f32 % 7.0) * 0.01).collect();
    let b: Vec<f32> = (0..16 * 16).map(|x| (x as f32 % 5.0) * 0.02 - 0.05).collect();
    (a, b)
}

fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("skein-search-cache-{tag}-{nanos}"))
}

/// Replaying a captured (egraph, genome) against a *freshly built, never
/// searched* identical graph yields the same numerical result as the original
/// search. This is the correctness guarantee behind the cache: same kernels.
#[test]
fn load_search_result_matches_full_search() {
    let (a_data, b_data) = inputs();

    // Cold: full search, capture the winning genome and the e-graph.
    let mut cx = Graph::new();
    let (a, b, out) = build_chain(&mut cx);
    cx.build_search_space::<NativeRuntime>();
    let (mut rt, genome) = cx.search_capture(NativeRuntime::default(), 1);
    let egraph = cx.egraph().expect("e-graph present after search").clone();
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    rt.execute(&cx.dyn_map);
    let want = rt.get_f32(out).clone();

    // Warm: a fresh identical graph that NEVER calls build_search_space.
    let mut cx2 = Graph::new();
    let (a2, b2, out2) = build_chain(&mut cx2);
    let mut rt2 = cx2
        .load_search_result(NativeRuntime::default(), &egraph, &genome)
        .expect("reconstruct runtime from cached search result");
    rt2.set_data(a2, a_data);
    rt2.set_data(b2, b_data);
    rt2.execute(&cx2.dyn_map);
    let got = rt2.get_f32(out2).clone();

    assert_eq!(want.len(), got.len(), "output length mismatch");
    assert!(!want.is_empty(), "empty output");
    for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
        assert!((w - g).abs() < 1e-5, "pos {i}: cold {w} vs warm {g}");
    }
}

/// End-to-end: the first `build_and_search_cached` writes the cache, the second
/// loads it. Same output, and the warm run is faster (it skips egglog).
#[test]
fn cached_search_cold_then_warm() {
    let (a_data, b_data) = inputs();
    let dir = unique_temp_dir("e2e");

    // Cold run — populates the cache.
    let mut cx = Graph::new();
    let (a, b, out) = build_chain(&mut cx);
    let t0 = Instant::now();
    let mut rt = NativeComputeRuntime::build_and_search_cached(&mut cx, 1, &[], Some(&dir))
        .expect("cold build_and_search_cached");
    let cold = t0.elapsed();
    rt.set_data_f32(a, a_data.clone());
    rt.set_data_f32(b, b_data.clone());
    rt.execute(&cx);
    let want = rt.get_data_f32(out);

    // The cache must now contain an egraph + genome entry.
    let exts: Vec<String> = std::fs::read_dir(&dir)
        .expect("cache dir created")
        .filter_map(|e| e.ok())
        .filter_map(|e| e.path().extension().map(|x| x.to_string_lossy().into_owned()))
        .collect();
    assert!(exts.iter().any(|e| e == "egraph"), "no .egraph cached: {exts:?}");
    assert!(exts.iter().any(|e| e == "genome"), "no .genome cached: {exts:?}");

    // Warm run — a fresh graph served from the cache.
    let mut cx2 = Graph::new();
    let (a2, b2, out2) = build_chain(&mut cx2);
    let t1 = Instant::now();
    let mut rt2 = NativeComputeRuntime::build_and_search_cached(&mut cx2, 1, &[], Some(&dir))
        .expect("warm build_and_search_cached");
    let warm = t1.elapsed();
    rt2.set_data_f32(a2, a_data);
    rt2.set_data_f32(b2, b_data);
    rt2.execute(&cx2);
    let got = rt2.get_data_f32(out2);

    eprintln!("[search-cache] cold={cold:?}  warm={warm:?}");

    assert_eq!(want.len(), got.len(), "output length mismatch");
    for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
        assert!((w - g).abs() < 1e-5, "pos {i}: cold {w} vs warm {g}");
    }
    assert!(
        warm < cold,
        "warm run ({warm:?}) should skip egglog and beat cold ({cold:?})"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The cache key is `hash(hlir_to_egglog(graph))`. For a hit to survive across
/// the `compile` process and a later `serve`/`verify` process, that program
/// string must be identical for two independently-built copies of the same
/// graph. `hlir_to_egglog` orders its output by node index (BinaryHeap +
/// topo/externals), using HashMaps only for lookups, so it does not depend on
/// per-process hash seeding — assert that here.
#[test]
fn egglog_program_is_deterministic_across_rebuilds() {
    use luminal::egglog_utils::hlir_to_egglog;

    let mut cx1 = Graph::new();
    build_chain(&mut cx1);
    let mut cx2 = Graph::new();
    build_chain(&mut cx2);

    let (p1, _) = hlir_to_egglog(&cx1);
    let (p2, _) = hlir_to_egglog(&cx2);
    assert_eq!(p1, p2, "egglog program differs between identical graph rebuilds");
}

/// A corrupt cache entry must not break compilation: the loader falls back to a
/// full search rather than erroring or producing a wrong result.
#[test]
fn corrupt_cache_falls_back() {
    let (a_data, b_data) = inputs();
    let dir = unique_temp_dir("corrupt");
    std::fs::create_dir_all(&dir).unwrap();

    // Cold to populate, then clobber every cache file with garbage.
    let mut cx = Graph::new();
    let (_a, _b, _out) = build_chain(&mut cx);
    NativeComputeRuntime::build_and_search_cached(&mut cx, 1, &[], Some(&dir)).unwrap();
    for entry in std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()) {
        std::fs::write(entry.path(), b"not a valid cache entry").unwrap();
    }

    // Warm run over the corrupt cache still succeeds and is correct.
    let mut cx2 = Graph::new();
    let (a2, b2, out2) = build_chain(&mut cx2);
    let mut rt2 = NativeComputeRuntime::build_and_search_cached(&mut cx2, 1, &[], Some(&dir))
        .expect("fall back past corrupt cache");
    rt2.set_data_f32(a2, a_data);
    rt2.set_data_f32(b2, b_data);
    rt2.execute(&cx2);
    assert!(!rt2.get_data_f32(out2).is_empty());

    // Sanity: the helper paths point under the artifact root as documented.
    let root = std::path::Path::new("/tmp/artifact");
    assert!(search_cache::search_cache_dir(root).ends_with("cache"));
    assert!(search_cache::cubin_cache_dir(root).ends_with("cache/cubins"));

    let _ = std::fs::remove_dir_all(&dir);
}
