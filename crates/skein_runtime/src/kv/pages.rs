//! Page types — the units the KV allocator manages.

use crate::types::RequestId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct Page {
    pub id: PageId,
    /// Tokens this page can hold. Matches the Plan's `page_size`.
    pub capacity_tokens: u32,
    /// Number of active request `PageTable`s referencing this page. A page
    /// with `refcount == 0` is either in the free list or in the
    /// cached-LRU (held only by the radix tree).
    pub refcount: u32,
    /// Bumped on eviction. Any cached reference recorded with an earlier
    /// generation is stale and rejected at lookup.
    pub generation: u64,
}

impl Page {
    pub(crate) fn new(id: PageId, capacity_tokens: u32) -> Self {
        Self {
            id,
            capacity_tokens,
            refcount: 0,
            generation: 0,
        }
    }
}

/// Per-request KV bookkeeping. Each in-flight request has exactly one
/// `PageTable` mapping its logical token positions to physical `PageId`s.
#[derive(Debug, Clone)]
pub struct PageTable {
    pub request_id: RequestId,
    pub pages: Vec<PageId>,
    /// One-past-the-last token currently stored (i.e. the next `advance`
    /// writes at this index).
    pub last_token_idx: u32,
    /// Tokens served from the prefix cache. Useful for metrics.
    pub prefix_hit_tokens: u32,
    /// The full prompt token sequence, retained so `release` can walk the
    /// radix tree and decrement refcounts along the exact path this
    /// request inserted.
    pub(crate) prompt_tokens: Vec<u32>,
}
