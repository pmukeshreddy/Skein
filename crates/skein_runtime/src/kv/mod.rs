//! KV cache subsystem: pages + paged allocator + radix prefix tree.

pub mod allocator;
pub mod pages;
pub mod radix;

pub use allocator::PagedKVAllocator;
pub use pages::{Page, PageId, PageTable};
pub use radix::RadixPrefixTree;
