//! Test 1 + 2 — admit/advance/release and LRU eviction.

mod common;
use common::*;

use skein_ir::types::BatchPolicy;
use skein_runtime::kv::PagedKVAllocator;
use skein_runtime::types::RequestId;

// Test 1 — admit a 100-token prompt at page_size=32 → 4 pages
// (ceil(100/32) = 4). Advance with 50 more tokens → 1-2 more pages.
// Release → all pages back in the free list.
#[test]
fn kv_allocator_admit_advance_release() {
    let plan = mk_plan(
        32,
        /*prefix_cache=*/ false,
        BatchPolicy::Continuous { max_batch: 8 },
    );
    // 64 pages worth of memory; bytes_per_token=1 to keep the arithmetic
    // trivial.
    let mut allocator = PagedKVAllocator::new(&plan, /*total_kv_bytes=*/ 64 * 32, 1, 4096).unwrap();
    assert_eq!(allocator.total_pages(), 64);

    let prompt: Vec<u32> = (0..100).collect();
    let pt = allocator.admit(RequestId(1), &prompt).unwrap();
    assert_eq!(pt.pages.len(), 4);
    assert_eq!(allocator.in_use_pages(), 4);

    // Advance 50 tokens — last_token_idx 100 → 150 → pages_needed 5.
    allocator
        .advance(RequestId(1), &(100..150).collect::<Vec<u32>>())
        .unwrap();
    assert!(allocator.in_use_pages() >= 5);

    allocator.release(RequestId(1)).unwrap();
    assert_eq!(allocator.in_use_pages(), 0);
}

// Test 2 — fill to 95% capacity with many short requests that release
// (so their pages enter the cached LRU), then admit a new request that
// requires fresh pages. Eviction kicks in and generations bump.
#[test]
fn kv_allocator_lru_eviction() {
    let plan = mk_plan(
        32,
        /*prefix_cache=*/ true,
        BatchPolicy::Continuous { max_batch: 8 },
    );
    let mut allocator = PagedKVAllocator::new(&plan, 64 * 32, 1, 4096).unwrap();
    let total = allocator.total_pages();
    assert_eq!(total, 64);

    // Admit 60 short requests (one page each, 32 tokens) and release each
    // immediately so its page enters the cached LRU. With prefix_cache on
    // the released pages stay in `cached_lru` (still mapped by the radix)
    // instead of going straight to `free`.
    let prompt: Vec<u32> = (0..32).collect();
    for i in 0..60u64 {
        // Vary the prompt slightly so each request's radix path is unique;
        // otherwise the radix would short-circuit subsequent admits via
        // prefix reuse and we wouldn't allocate fresh pages.
        let mut p = prompt.clone();
        p[0] = i as u32 + 10_000;
        let _ = allocator.admit(RequestId(i + 1), &p).unwrap();
        allocator.release(RequestId(i + 1)).unwrap();
    }
    assert_eq!(allocator.in_use_pages(), 0);

    // Pick one cached page id and remember its generation.
    let probe_page_idx = 0u32;
    let gen_before = allocator
        .generation_of(skein_runtime::kv::PageId(probe_page_idx))
        .unwrap();

    // Now admit a fresh request that requires more pages than the free
    // list holds (all 60 admitted pages went to cached_lru, only 4 pages
    // truly free). Asking for 8 pages forces eviction of 4 cached pages.
    let new_prompt: Vec<u32> = (0..8 * 32).map(|i| (i as u32) + 100_000).collect();
    let pt = allocator.admit(RequestId(999), &new_prompt).unwrap();
    assert_eq!(pt.pages.len(), 8);
    assert_eq!(allocator.in_use_pages(), 8);

    // The probed page's generation either bumped (it was evicted to satisfy
    // the new request) OR it stayed (still in cached_lru). Either way the
    // generation never *decreases*. The point of test 2 is that *some*
    // cached page got bumped — verify by scanning.
    let gen_after_on_probe = allocator
        .generation_of(skein_runtime::kv::PageId(probe_page_idx))
        .unwrap();
    assert!(gen_after_on_probe >= gen_before);

    let mut any_bumped = false;
    for pid in 0..total {
        let g = allocator
            .generation_of(skein_runtime::kv::PageId(pid))
            .unwrap();
        if g > 0 {
            any_bumped = true;
            break;
        }
    }
    assert!(
        any_bumped,
        "expected at least one page generation to bump after eviction"
    );
}
