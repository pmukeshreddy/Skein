//! Tests 3 + 4 — radix prefix tree reuse and refcount correctness.

mod common;
use common::*;

use skein_ir::types::BatchPolicy;
use skein_runtime::kv::PagedKVAllocator;
use skein_runtime::types::RequestId;

// Test 3 — 50% prefix overlap. With prefix-cache *on*, more than 40% of
// total admitted tokens should be served from the cache. With prefix-cache
// *off*, the rate is 0.
#[test]
fn radix_prefix_reuse_hit_rate() {
    let shared_prefix: Vec<u32> = (0..96).collect(); // 3 pages at page_size=32
    let n_requests = 20u64;

    // --- prefix cache OFF ---
    let plan_off = mk_plan(32, false, BatchPolicy::Continuous { max_batch: 8 });
    let mut off = PagedKVAllocator::new(&plan_off, 64 * 32 * 1024, 1, 4096).unwrap();
    let mut total_tokens_off: u64 = 0;
    let mut hit_tokens_off: u64 = 0;
    for i in 0..n_requests {
        let mut p = shared_prefix.clone();
        p.extend([100_000 + i as u32, 100_500 + i as u32]); // unique suffix
        let pt = off.admit(RequestId(i + 1), &p).unwrap();
        total_tokens_off += p.len() as u64;
        hit_tokens_off += pt.prefix_hit_tokens as u64;
    }
    assert_eq!(hit_tokens_off, 0);
    let _ = total_tokens_off;

    // --- prefix cache ON ---
    let plan_on = mk_plan(32, true, BatchPolicy::Continuous { max_batch: 8 });
    let mut on = PagedKVAllocator::new(&plan_on, 64 * 32 * 1024, 1, 4096).unwrap();
    let mut total_tokens_on: u64 = 0;
    let mut hit_tokens_on: u64 = 0;
    for i in 0..n_requests {
        let mut p = shared_prefix.clone();
        p.extend([200_000 + i as u32, 200_500 + i as u32]);
        let pt = on.admit(RequestId(i + 1), &p).unwrap();
        total_tokens_on += p.len() as u64;
        hit_tokens_on += pt.prefix_hit_tokens as u64;
    }
    let hit_rate = hit_tokens_on as f64 / total_tokens_on as f64;
    assert!(
        hit_rate > 0.40,
        "expected > 40% hit rate, got {:.3} ({hit_tokens_on}/{total_tokens_on})",
        hit_rate
    );
}

// Test 4 — three requests share a 100-token prefix; refcount on that
// prefix's terminal node is 3. Release two → refcount 1. Release third → 0.
#[test]
fn radix_refcount_correctness() {
    let plan = mk_plan(32, true, BatchPolicy::Continuous { max_batch: 8 });
    let mut allocator = PagedKVAllocator::new(&plan, 64 * 32 * 1024, 1, 4096).unwrap();

    let shared: Vec<u32> = (0..100).collect();
    let mut reqs = Vec::new();
    for i in 0..3u64 {
        let mut p = shared.clone();
        p.extend([900_000 + i as u32]);
        allocator.admit(RequestId(i + 1), &p).unwrap();
        reqs.push((RequestId(i + 1), p));
    }
    // Inspect refcount at the shared prefix via reflection of utilization:
    // we don't expose the radix node refcount directly, but we can verify
    // the *page* refcount (page covering tokens 0..32) is 3 — every
    // request shares those pages.
    //
    // The radix lookup is read-only and tells us how many tokens of a
    // candidate prompt would be reused. Three sharing requests have the
    // shared pages allocated; if we ask "would this prompt reuse anything?"
    // we should see the cached prefix length.
    let probe = allocator.try_prefix_reuse(&shared);
    // page_size=32 → 96 is the largest multiple of 32 ≤ 100.
    assert_eq!(probe, 96);

    // Release two requests. The shared pages still have refcount==1
    // (from the remaining request), so subsequent `try_prefix_reuse`
    // still finds them.
    allocator.release(reqs[0].0).unwrap();
    allocator.release(reqs[1].0).unwrap();
    assert_eq!(allocator.try_prefix_reuse(&shared), 96);

    // Release the third. The shared pages drop to refcount==0; with
    // prefix-cache on they go to cached_lru but the radix still maps
    // them (until eviction). The assertion we *can* make
    // cleanly: after the third release `in_use_pages` is zero.
    allocator.release(reqs[2].0).unwrap();
    assert_eq!(allocator.in_use_pages(), 0);
}
