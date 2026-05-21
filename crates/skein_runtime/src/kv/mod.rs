//! KV cache subsystem: pages + paged allocator + radix prefix tree.

pub mod allocator;
pub mod paged_cache;
pub mod pages;
pub mod radix;

pub use allocator::PagedKVAllocator;
pub use paged_cache::PagedKvCache;
pub use pages::{Page, PageId, PageTable};
pub use radix::RadixPrefixTree;
