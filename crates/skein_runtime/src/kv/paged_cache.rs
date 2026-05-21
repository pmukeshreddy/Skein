//! `PagedKvCache` — page-backed per-layer KV storage that actually holds the
//! key/value bytes the attention kernels read each decode step.
//!
//! Where [`crate::kv_cache::KvCache`] kept one flat, fixed-capacity buffer per
//! `(layer, kind)` and wiped it between requests, this cache stores KV bytes in
//! **physical pages** managed by [`PagedKVAllocator`]. A request's logical token
//! positions map, page by page, onto physical `PageId`s; the bytes for a page
//! survive a request's release so a later request sharing a prompt prefix can
//! *reuse* them (the radix prefix tree in the allocator picks the longest match
//! and the reused pages already hold the correct KV — identical tokens at
//! identical absolute positions produce identical K/V, RoPE included).
//!
//! The graph still consumes a contiguous fixed-capacity `[batch, CAP, kv_dim]`
//! buffer per step; [`buffer`](PagedKvCache::buffer) *assembles* that view by
//! gathering the active request's pages into slots `0..position`. New K/V is
//! written into the page covering its slot via [`write_slot`]. So paging is real
//! (allocation, eviction, cross-request reuse) and actually backs the forward —
//! the contiguous view is only the per-step staging the kernels expect.

use crate::error::RuntimeError;
use crate::kv::PagedKVAllocator;
use crate::kv::pages::PageId;
use crate::kv_cache::KvKind;
use crate::types::RequestId;

/// The implicit request id used when the caller never calls
/// [`begin_request`](PagedKvCache::begin_request) (e.g. the single-stream serve
/// path before prefix caching is driven, and unit tests). It owns raw-allocated
/// pages outside the radix prefix tree.
const IMPLICIT_REQUEST: RequestId = RequestId(u64::MAX);

#[derive(Debug)]
struct Active {
    id: RequestId,
    /// Physical pages in logical token-block order: page `i` covers slots
    /// `[i*page_size, (i+1)*page_size)`.
    pages: Vec<PageId>,
    /// Number of valid (written) slots — what [`buffer`] gathers and the
    /// "past" length attention reads.
    position: usize,
    /// Tokens served from the prefix cache on admit (metrics + prefill skip).
    matched_prefix: usize,
    /// `true` for the [`IMPLICIT_REQUEST`] (raw pages, no radix).
    implicit: bool,
}

#[derive(Debug)]
pub struct PagedKvCache {
    allocator: PagedKVAllocator,
    page_size: usize,
    /// Page-indexed KV bytes, one flat `Vec` per layer of length
    /// `total_pages * page_size * kv_width[layer]`. Lazily sized once the
    /// per-token width is learned from the first write.
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// Per-token K (== V) element width for each layer (`n_kv_local*head_dim`),
    /// learned from the first [`write_slot`]; `0` until then.
    kv_width: Vec<usize>,
    total_pages: usize,
    active: Option<Active>,
}

impl PagedKvCache {
    /// Build a cache with an explicit page geometry. `page_size` tokens per
    /// page; `total_pages` device page budget; `prefix_enable` turns on the
    /// radix prefix tree (cross-request reuse); `layers` pre-sizes the per-layer
    /// width table.
    pub fn new(
        page_size: u32,
        total_pages: u32,
        prefix_enable: bool,
        radix_max_depth: u32,
        layers: usize,
    ) -> Result<Self, RuntimeError> {
        let allocator =
            PagedKVAllocator::with_capacity(page_size, total_pages, prefix_enable, radix_max_depth)?;
        Ok(Self {
            allocator,
            page_size: page_size as usize,
            keys: vec![Vec::new(); layers],
            values: vec![Vec::new(); layers],
            kv_width: vec![0; layers],
            total_pages: total_pages as usize,
            active: None,
        })
    }

    pub fn allocator(&self) -> &PagedKVAllocator {
        &self.allocator
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Tokens the active request reused from the prefix cache (0 if none /
    /// no active request).
    pub fn prefix_hit_tokens(&self) -> usize {
        self.active.as_ref().map(|a| a.matched_prefix).unwrap_or(0)
    }

    fn store_mut(&mut self, kind: KvKind) -> &mut Vec<Vec<f32>> {
        match kind {
            KvKind::Key => &mut self.keys,
            KvKind::Value => &mut self.values,
        }
    }

    fn store(&self, kind: KvKind) -> &Vec<Vec<f32>> {
        match kind {
            KvKind::Key => &self.keys,
            KvKind::Value => &self.values,
        }
    }

    fn ensure_layer(&mut self, layer: usize) {
        for s in [&mut self.keys, &mut self.values] {
            while s.len() <= layer {
                s.push(Vec::new());
            }
        }
        while self.kv_width.len() <= layer {
            self.kv_width.push(0);
        }
    }

    /// Begin a real request: admit it through the allocator (radix prefix match
    /// + page allocation) and make it the active request. Returns the number of
    /// prompt tokens served from the prefix cache — the caller skips
    /// recomputing those tokens' KV (they are already in the reused pages).
    pub fn begin_request(
        &mut self,
        id: RequestId,
        tokens: &[u32],
    ) -> Result<usize, RuntimeError> {
        // Release any leftover active request first (defensive).
        self.end_request();
        let matched = self.admit_request(id, tokens)?;
        self.set_active(id, matched)?;
        if let Some(a) = self.active.as_mut() {
            a.matched_prefix = matched;
        }
        Ok(matched)
    }

    /// Admit a request through the allocator (prefix match + page allocation)
    /// WITHOUT changing the active request. For the continuous-batching driver,
    /// which keeps several requests in-flight and switches between them with
    /// [`set_active`](Self::set_active). Returns the prefix-cache hit length.
    pub fn admit_request(&mut self, id: RequestId, tokens: &[u32]) -> Result<usize, RuntimeError> {
        let pt = self.allocator.admit(id, tokens)?;
        Ok(pt.prefix_hit_tokens as usize)
    }

    /// Make an already-admitted request the active one, restoring its page
    /// table and setting its valid-slot count to `position`. Used by the
    /// continuous-batching driver to interleave several in-flight requests:
    /// before each request's step it switches the active request, so each
    /// request's KV is read/written through its own pages. Returns
    /// [`RuntimeError::UnknownRequest`] if the id was never admitted.
    pub fn set_active(&mut self, id: RequestId, position: usize) -> Result<(), RuntimeError> {
        let pages = self
            .allocator
            .pages_for(id)
            .ok_or(RuntimeError::UnknownRequest(id.0))?;
        self.active = Some(Active {
            id,
            pages,
            position,
            matched_prefix: 0,
            implicit: false,
        });
        Ok(())
    }

    /// Release a specific request's pages (whether or not it is active).
    pub fn release(&mut self, id: RequestId) -> Result<(), RuntimeError> {
        if self.active.as_ref().map(|a| a.id) == Some(id) {
            self.active = None;
        }
        self.allocator.release(id)
    }

    /// Extend the active request by `n` decode tokens, allocating pages as the
    /// sequence crosses page boundaries. Must be called before the step that
    /// writes those tokens' KV so the covering page exists.
    pub fn advance(&mut self, n: usize) -> Result<(), RuntimeError> {
        let Some(active) = self.active.as_ref() else {
            return Ok(());
        };
        if active.implicit {
            // Implicit request grows its own raw pages in `set_position`.
            return Ok(());
        }
        let id = active.id;
        // A single dummy token per step is enough for the allocator to extend
        // `last_token_idx` and allocate covering pages; the real token bytes are
        // written via `write_slot`.
        for _ in 0..n {
            self.allocator.advance(id, &[0])?;
        }
        let pages = self
            .allocator
            .pages_for(id)
            .ok_or(RuntimeError::UnknownRequest(id.0))?;
        if let Some(active) = self.active.as_mut() {
            active.pages = pages;
        }
        Ok(())
    }

    /// Release the active request's pages (returns them to the cache/free list)
    /// and clear the active slot. Safe to call with no active request.
    pub fn end_request(&mut self) {
        if let Some(active) = self.active.take() {
            if active.implicit {
                for &pid in &active.pages {
                    self.allocator.free_page(pid);
                }
            } else {
                let _ = self.allocator.release(active.id);
            }
        }
    }

    /// Compatibility shim for the single-stream path: release the active
    /// request. (Unlike the old contiguous cache, this does NOT zero stored
    /// bytes — released pages stay populated for prefix reuse.)
    pub fn reset(&mut self) {
        self.end_request();
    }

    /// Set the current decode position (valid-slot count). For the implicit
    /// request, grows raw pages to cover `position`.
    pub fn set_position(&mut self, position: usize) -> Result<(), RuntimeError> {
        self.ensure_implicit_pages(position)?;
        if let Some(active) = self.active.as_mut() {
            active.position = position;
        }
        Ok(())
    }

    /// Ensure an implicit active request exists with pages covering `slot`.
    fn ensure_implicit_pages(&mut self, slot: usize) -> Result<(), RuntimeError> {
        let need_implicit = match &self.active {
            None => true,
            Some(a) => a.implicit,
        };
        if !need_implicit {
            return Ok(());
        }
        if self.active.is_none() {
            self.active = Some(Active {
                id: IMPLICIT_REQUEST,
                pages: Vec::new(),
                position: 0,
                matched_prefix: 0,
                implicit: true,
            });
        }
        let needed_pages = (slot + 1).div_ceil(self.page_size);
        let have = self.active.as_ref().map(|a| a.pages.len()).unwrap_or(0);
        for _ in have..needed_pages {
            let pid = self.allocator.allocate_page()?;
            if let Some(a) = self.active.as_mut() {
                a.pages.push(pid);
            }
        }
        Ok(())
    }

    /// Physical byte offset of `slot` within layer `layer`'s page-indexed store.
    /// `None` if the active request has no page covering the slot.
    fn slot_offset(&self, slot: usize, width: usize) -> Option<usize> {
        let active = self.active.as_ref()?;
        let page_idx = slot / self.page_size;
        let within = slot % self.page_size;
        let pid = *active.pages.get(page_idx)?;
        Some((pid.0 as usize * self.page_size + within) * width)
    }

    /// Assemble the contiguous fixed-capacity buffer the graph reads this step:
    /// `full_len` zeros with slots `0..position` filled from the active
    /// request's pages. Slots beyond `position` are masked by the graph.
    pub fn buffer(&mut self, kind: KvKind, layer: usize, full_len: usize) -> Vec<f32> {
        self.ensure_layer(layer);
        let width = self.kv_width[layer];
        let position = self.active.as_ref().map(|a| a.position).unwrap_or(0);
        let mut out = vec![0.0f32; full_len];
        if width == 0 || position == 0 {
            return out; // nothing cached yet
        }
        let store = self.store(kind);
        let layer_bytes = &store[layer];
        for slot in 0..position {
            let Some(src) = self.slot_offset(slot, width) else {
                continue;
            };
            let dst = slot * width;
            if src + width <= layer_bytes.len() && dst + width <= out.len() {
                out[dst..dst + width].copy_from_slice(&layer_bytes[src..src + width]);
            }
        }
        out
    }

    /// Write this step's new K/V `data` into `slot` of `(kind, layer)`, routed
    /// to the physical page covering the slot. Learns the per-token width and
    /// lazily sizes the layer's page-indexed store on first use.
    pub fn write_slot(&mut self, kind: KvKind, layer: usize, slot: usize, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        self.ensure_layer(layer);
        let width = data.len();
        if self.kv_width[layer] == 0 {
            self.kv_width[layer] = width;
        }
        if self.kv_width[layer] != width {
            // Width must be stable per layer; a mismatch means a wiring bug.
            tracing::warn!(
                layer,
                expected = self.kv_width[layer],
                got = width,
                "PagedKvCache: kv width changed mid-stream"
            );
            return;
        }
        // Implicit single-stream path grows pages on demand.
        if self.active.as_ref().map(|a| a.implicit).unwrap_or(true) {
            if self.ensure_implicit_pages(slot).is_err() {
                return;
            }
        }
        let Some(off) = self.slot_offset(slot, width) else {
            return;
        };
        let total = self.total_pages * self.page_size * width;
        let store = self.store_mut(kind);
        if store[layer].len() < total {
            store[layer].resize(total, 0.0);
        }
        if off + width <= store[layer].len() {
            store[layer][off..off + width].copy_from_slice(data);
        }
        // Advance the implicit valid-slot watermark so `buffer` includes it.
        if let Some(a) = self.active.as_mut() {
            if a.implicit && slot + 1 > a.position {
                a.position = slot + 1;
            }
        }
    }

    /// Pages currently in use across all requests (allocator accounting).
    pub fn in_use_pages(&self) -> u32 {
        self.allocator.in_use_pages()
    }

    pub fn utilization(&self) -> f64 {
        self.allocator.utilization()
    }

    pub fn prefix_enabled(&self) -> bool {
        self.allocator.prefix_enabled()
    }

    /// Test/inspection helper: the active request's assembled buffer for
    /// `(kind, layer)` over its valid slots (empty if nothing written).
    #[cfg(test)]
    pub fn peek_active(&self, kind: KvKind, layer: usize) -> Vec<f32> {
        let width = self.kv_width.get(layer).copied().unwrap_or(0);
        let position = self.active.as_ref().map(|a| a.position).unwrap_or(0);
        if width == 0 || position == 0 {
            return Vec::new();
        }
        let store = self.store(kind);
        let layer_bytes = &store[layer];
        let mut out = vec![0.0f32; position * width];
        for slot in 0..position {
            if let Some(src) = self.slot_offset(slot, width) {
                if src + width <= layer_bytes.len() {
                    out[slot * width..(slot + 1) * width]
                        .copy_from_slice(&layer_bytes[src..src + width]);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> PagedKvCache {
        // page_size=2, 64 pages, prefix on, depth 4096, 2 layers.
        PagedKvCache::new(2, 64, true, 4096, 2).unwrap()
    }

    #[test]
    fn implicit_request_writes_and_assembles_buffer() {
        let mut c = cache();
        // width-2 tokens, write slots 0,1,2.
        c.set_position(0).unwrap();
        c.write_slot(KvKind::Key, 0, 0, &[1.0, 2.0]);
        c.write_slot(KvKind::Key, 0, 1, &[3.0, 4.0]);
        c.write_slot(KvKind::Key, 0, 2, &[5.0, 6.0]);
        c.set_position(3).unwrap();
        // Fixed-cap buffer of 4 slots (full_len 8): first 3 filled, slot 3 zero.
        let buf = c.buffer(KvKind::Key, 0, 8);
        assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn prefix_reuse_serves_shared_prefix_from_cache() {
        let mut c = cache();
        // Request A: tokens [10,11,12,13], width-2 KV per token.
        let a = RequestId(1);
        let matched_a = c.begin_request(a, &[10, 11, 12, 13]).unwrap();
        assert_eq!(matched_a, 0, "first request has no prefix to reuse");
        for (slot, base) in [(0usize, 100.0f32), (1, 110.0), (2, 120.0), (3, 130.0)] {
            c.set_position(slot).unwrap();
            c.write_slot(KvKind::Key, 0, slot, &[base, base + 1.0]);
            c.write_slot(KvKind::Value, 0, slot, &[base + 0.5, base + 1.5]);
        }
        c.set_position(4).unwrap();
        c.end_request();

        // Request B shares the first 4-token prefix [10,11,12,13] then diverges.
        let b = RequestId(2);
        let matched_b = c.begin_request(b, &[10, 11, 12, 13, 99]).unwrap();
        // page_size=2 → prefix match is page-aligned; 4 shared tokens = 2 pages.
        assert_eq!(matched_b, 4, "B reuses the 4-token shared prefix");
        // The reused pages already hold A's KV: B's assembled buffer over the
        // matched prefix equals what A wrote, with no recompute.
        c.set_position(4).unwrap();
        let buf = c.buffer(KvKind::Key, 0, 5 * 2);
        assert_eq!(
            &buf[0..8],
            &[100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0],
            "matched-prefix KV reused byte-for-byte from the cached pages"
        );
    }

    #[test]
    fn release_returns_pages_for_reuse_not_zeroed() {
        let mut c = cache();
        let a = RequestId(1);
        c.begin_request(a, &[7, 8]).unwrap();
        c.write_slot(KvKind::Key, 0, 0, &[1.0, 1.0]);
        c.write_slot(KvKind::Key, 0, 1, &[2.0, 2.0]);
        let used = c.in_use_pages();
        assert!(used >= 1);
        c.end_request();
        // After release the pages are reusable (in_use drops) but bytes persist
        // for a prefix match — verified by the reuse test above.
        assert_eq!(c.in_use_pages(), 0);
    }
}
