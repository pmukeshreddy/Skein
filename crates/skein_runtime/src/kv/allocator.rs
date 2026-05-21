//! Paged KV allocator. Owns the page array, the free + cached-LRU lists,
//! and (optionally) the radix prefix tree. Page accounting is host-side; the
//! page bytes live in device memory on the CUDA build.

use std::collections::{HashMap, VecDeque};

use skein_ir::plan::Plan;
use skein_ir::types::KVLayout;

use crate::error::RuntimeError;
use crate::types::RequestId;

use super::pages::{Page, PageId, PageTable};
use super::radix::RadixPrefixTree;

#[derive(Debug)]
pub struct PagedKVAllocator {
    page_size: u32,
    pages: Vec<Page>,
    /// Pages with `refcount == 0` and not in the cache. First to be used
    /// on the next allocation.
    free: VecDeque<PageId>,
    /// Pages with `refcount == 0` that are still referenced by the radix
    /// tree (i.e., are usable for prefix reuse). Eviction pops from the
    /// front (LRU) to reclaim.
    cached_lru: VecDeque<PageId>,
    page_tables: HashMap<RequestId, PageTable>,
    radix: Option<RadixPrefixTree>,
}

impl PagedKVAllocator {
    /// Build an allocator from a `Plan` and a per-device KV memory budget.
    /// `bytes_per_token` is the cost of one token's K+V at the chosen
    /// dtype × num_kv_heads × head_dim (`skein_cost::memory::block_kv_bytes`
    /// computes this from a Plan; tests pass it directly).
    pub fn new(
        plan: &Plan,
        total_kv_bytes: u64,
        bytes_per_token: u64,
        radix_max_depth: u32,
    ) -> Result<Self, RuntimeError> {
        let page_size = match plan.kv.layout {
            KVLayout::Paged { page_size } => page_size,
            // Contiguous = one giant page per request. We model it as
            // page_size=1 so the arithmetic stays uniform.
            // TODO(contiguous-kv): a dedicated contiguous allocation path.
            KVLayout::Contiguous => 1,
        };
        if bytes_per_token == 0 || page_size == 0 {
            return Err(RuntimeError::KvUndersized {
                capacity_pages: 0,
                prompt_pages: 0,
            });
        }
        let bytes_per_page = bytes_per_token * page_size as u64;
        let total_pages_u64 = total_kv_bytes / bytes_per_page;
        if total_pages_u64 == 0 {
            return Err(RuntimeError::KvUndersized {
                capacity_pages: 0,
                prompt_pages: 1,
            });
        }
        let total_pages = total_pages_u64.min(u32::MAX as u64) as u32;

        let mut pages = Vec::with_capacity(total_pages as usize);
        let mut free = VecDeque::with_capacity(total_pages as usize);
        for i in 0..total_pages {
            pages.push(Page::new(PageId(i), page_size));
            free.push_back(PageId(i));
        }
        let radix = plan
            .execution
            .prefix_cache
            .enable
            .then(|| RadixPrefixTree::new(page_size, radix_max_depth));

        Ok(Self {
            page_size,
            pages,
            free,
            cached_lru: VecDeque::new(),
            page_tables: HashMap::new(),
            radix,
        })
    }

    /// Build an allocator with an explicit page geometry, bypassing the Plan's
    /// `KVLayout`. Used by the runtime KV cache ([`crate::kv::PagedKvCache`]),
    /// which manages KV bytes itself and picks a multi-token `page_size` for
    /// real paging even when the Plan chose a contiguous layout. `total_pages`
    /// is the device's page budget; `prefix_enable` turns on the radix tree.
    pub fn with_capacity(
        page_size: u32,
        total_pages: u32,
        prefix_enable: bool,
        radix_max_depth: u32,
    ) -> Result<Self, RuntimeError> {
        if page_size == 0 || total_pages == 0 {
            return Err(RuntimeError::KvUndersized {
                capacity_pages: total_pages,
                prompt_pages: 1,
            });
        }
        let mut pages = Vec::with_capacity(total_pages as usize);
        let mut free = VecDeque::with_capacity(total_pages as usize);
        for i in 0..total_pages {
            pages.push(Page::new(PageId(i), page_size));
            free.push_back(PageId(i));
        }
        let radix = prefix_enable.then(|| RadixPrefixTree::new(page_size, radix_max_depth));
        Ok(Self {
            page_size,
            pages,
            free,
            cached_lru: VecDeque::new(),
            page_tables: HashMap::new(),
            radix,
        })
    }

    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    pub fn total_pages(&self) -> u32 {
        self.pages.len() as u32
    }

    /// Whether prefix caching (the radix tree) is enabled.
    pub fn prefix_enabled(&self) -> bool {
        self.radix.is_some()
    }

    /// The physical pages currently mapped for `request_id`, in logical token
    /// order (page `i` covers tokens `[i*page_size, (i+1)*page_size)`).
    pub fn pages_for(&self, request_id: RequestId) -> Option<Vec<PageId>> {
        self.page_tables.get(&request_id).map(|pt| pt.pages.clone())
    }

    /// Number of free pages plus cached (reusable) pages — the headroom an
    /// admission can draw on before hitting [`RuntimeError::KvExhausted`].
    pub fn available_pages(&self) -> u32 {
        (self.free.len() + self.cached_lru.len()) as u32
    }

    pub fn in_use_pages(&self) -> u32 {
        self.pages.iter().filter(|p| p.refcount > 0).count() as u32
    }

    pub fn utilization(&self) -> f64 {
        if self.pages.is_empty() {
            return 0.0;
        }
        self.in_use_pages() as f64 / self.pages.len() as f64
    }

    /// Query-only prefix reuse estimate. Returns the number of prompt
    /// tokens that *would* be served from the cache if the request was
    /// admitted right now. Does not mutate state.
    pub fn try_prefix_reuse(&self, prompt_tokens: &[u32]) -> u32 {
        let Some(radix) = &self.radix else { return 0 };
        let gens: Vec<u64> = self.pages.iter().map(|p| p.generation).collect();
        let (matched, _) = radix.lookup_longest_match(prompt_tokens, &gens);
        matched as u32
    }

    pub fn admit(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
    ) -> Result<PageTable, RuntimeError> {
        // 1. Radix lookup (if enabled). Filter stale generations.
        let (matched_n, matched_pages) = if let Some(radix) = &self.radix {
            let gens: Vec<u64> = self.pages.iter().map(|p| p.generation).collect();
            radix.lookup_longest_match(prompt_tokens, &gens)
        } else {
            (0, Vec::new())
        };

        // 2. Bump per-page refcount on matched pages.
        for &pid in &matched_pages {
            let page = &mut self.pages[pid.0 as usize];
            page.refcount = page.refcount.saturating_add(1);
            // Page may have been in cached_lru — pull it out.
            self.cached_lru.retain(|p| *p != pid);
        }

        // 3. Allocate fresh pages for the remaining tokens.
        let remaining = prompt_tokens.len().saturating_sub(matched_n);
        let need = remaining.div_ceil(self.page_size as usize);
        if (need as u32) > self.total_pages() {
            return Err(RuntimeError::KvUndersized {
                capacity_pages: self.total_pages(),
                prompt_pages: need as u32,
            });
        }
        let mut new_pages = Vec::with_capacity(need);
        for _ in 0..need {
            let pid = self.allocate_or_evict()?;
            self.pages[pid.0 as usize].refcount =
                self.pages[pid.0 as usize].refcount.saturating_add(1);
            new_pages.push(pid);
        }

        let mut all_pages = matched_pages;
        all_pages.extend(&new_pages);

        // 4. Insert into the radix so future requests with this prefix can
        //    reuse this request's pages.
        if let Some(radix) = &mut self.radix {
            let gens: Vec<u64> = all_pages
                .iter()
                .map(|p| self.pages[p.0 as usize].generation)
                .collect();
            radix.insert(prompt_tokens, &all_pages, &gens);
            radix.refcount_inc(prompt_tokens);
        }

        let pt = PageTable {
            request_id,
            pages: all_pages,
            last_token_idx: prompt_tokens.len() as u32,
            prefix_hit_tokens: matched_n as u32,
            prompt_tokens: prompt_tokens.to_vec(),
        };
        self.page_tables.insert(request_id, pt.clone());
        Ok(pt)
    }

    pub fn advance(
        &mut self,
        request_id: RequestId,
        new_tokens: &[u32],
    ) -> Result<(), RuntimeError> {
        let pt = self
            .page_tables
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id.0))?;
        let next_idx = pt.last_token_idx + new_tokens.len() as u32;
        let pages_needed = next_idx.div_ceil(self.page_size) as usize;
        // Track which extra pages we still need to allocate beyond what
        // the PageTable already covers.
        let extra = pages_needed.saturating_sub(pt.pages.len());
        if extra > 0 {
            // Allocate without holding `pt` across the borrow.
            let mut allocated: Vec<PageId> = Vec::with_capacity(extra);
            for _ in 0..extra {
                let pid = self.allocate_or_evict()?;
                self.pages[pid.0 as usize].refcount =
                    self.pages[pid.0 as usize].refcount.saturating_add(1);
                allocated.push(pid);
            }
            // Re-borrow `pt` to extend.
            let pt = self
                .page_tables
                .get_mut(&request_id)
                .expect("just looked up");
            pt.pages.extend(&allocated);
        }
        let pt = self
            .page_tables
            .get_mut(&request_id)
            .expect("just looked up");
        pt.last_token_idx = next_idx;
        Ok(())
    }

    pub fn release(&mut self, request_id: RequestId) -> Result<(), RuntimeError> {
        let pt = self
            .page_tables
            .remove(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id.0))?;
        if let Some(radix) = &mut self.radix {
            radix.refcount_dec(&pt.prompt_tokens);
        }
        for &pid in &pt.pages {
            let page = &mut self.pages[pid.0 as usize];
            page.refcount = page.refcount.saturating_sub(1);
            if page.refcount == 0 {
                if self.radix.is_some() {
                    // Held by the cache for potential reuse.
                    self.cached_lru.push_back(pid);
                } else {
                    self.free.push_back(pid);
                }
            }
        }
        Ok(())
    }

    /// Generation of a page. Tests verify this bumps after eviction.
    pub fn generation_of(&self, pid: PageId) -> Option<u64> {
        self.pages.get(pid.0 as usize).map(|p| p.generation)
    }

    /// Allocate a single page (refcount = 1), drawing from the free list or
    /// evicting the LRU cached page. Used by the runtime KV cache for the
    /// implicit single-stream request and for raw growth not tied to a radix
    /// path. Pair with [`free_page`](Self::free_page).
    pub fn allocate_page(&mut self) -> Result<PageId, RuntimeError> {
        let pid = self.allocate_or_evict()?;
        self.pages[pid.0 as usize].refcount = self.pages[pid.0 as usize].refcount.saturating_add(1);
        Ok(pid)
    }

    /// Return a raw-allocated page. Decrements its refcount; at zero it goes
    /// back to the free list (no radix involvement — the inverse of
    /// [`allocate_page`](Self::allocate_page)).
    pub fn free_page(&mut self, id: PageId) {
        if let Some(page) = self.pages.get_mut(id.0 as usize) {
            page.refcount = page.refcount.saturating_sub(1);
            if page.refcount == 0 {
                self.free.push_back(id);
            }
        }
    }

    fn allocate_or_evict(&mut self) -> Result<PageId, RuntimeError> {
        if let Some(pid) = self.free.pop_front() {
            return Ok(pid);
        }
        // No free pages — try evicting from the cache (oldest first).
        while let Some(pid) = self.cached_lru.pop_front() {
            let page = &mut self.pages[pid.0 as usize];
            if page.refcount > 0 {
                // Should not happen: cached_lru is the refcount-0 set. If
                // it does, the page got revived by a concurrent admit's
                // `retain` step but stayed in the deque. Skip and keep
                // looking.
                continue;
            }
            page.generation = page.generation.saturating_add(1);
            return Ok(pid);
        }
        Err(RuntimeError::KvExhausted)
    }
}
