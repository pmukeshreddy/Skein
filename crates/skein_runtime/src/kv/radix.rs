//! `RadixPrefixTree` — cross-request KV reuse.
//!
//! A per-token trie. Each node represents one token transition; the node
//! at depth `d` carries an `Option<(PageId, u64)>` slot that is set when
//! `d` is a page boundary. The `u64` is the page's generation at insert
//! time; eviction bumps the generation, and lookup filters stale entries.
//!
//! Refcount is tracked per node alongside the page slot. The runtime
//! increments / decrements along a request's full token path on admit /
//! release so the radix tree's reuse decisions are aware of how many
//! active requests cover each prefix.

use std::collections::HashMap;

use super::pages::PageId;

#[derive(Debug, Default)]
pub struct RadixPrefixTree {
    page_size: u32,
    max_depth: u32,
    root: RadixNode,
}

#[derive(Debug, Default)]
struct RadixNode {
    children: HashMap<u32, Box<RadixNode>>,
    /// `Some` iff this node's depth is a multiple of `page_size`: the
    /// PageId covering tokens `[depth - page_size, depth)` plus the
    /// page's generation at insert time.
    page_here: Option<(PageId, u64)>,
    refcount: u32,
}

impl RadixPrefixTree {
    pub fn new(page_size: u32, max_depth: u32) -> Self {
        Self {
            page_size,
            max_depth,
            root: RadixNode::default(),
        }
    }

    /// Insert a complete `(tokens, pages, generations)` triple. The slices
    /// `pages` and `generations` are aligned: `pages[i]` covers
    /// `tokens[i*page_size .. (i+1)*page_size)`.
    pub fn insert(&mut self, tokens: &[u32], pages: &[PageId], generations: &[u64]) {
        let mut node = &mut self.root;
        let take = (tokens.len() as u32).min(self.max_depth) as usize;
        for (i, &t) in tokens.iter().take(take).enumerate() {
            node = node.children.entry(t).or_default();
            let pos = i + 1;
            if pos % self.page_size as usize == 0 {
                let page_idx = pos / self.page_size as usize - 1;
                if page_idx < pages.len() {
                    node.page_here = Some((pages[page_idx], generations[page_idx]));
                }
            }
        }
    }

    /// Longest prefix match. Returns `(tokens_matched, pages)` where
    /// `tokens_matched` is the number of prompt tokens served from the
    /// cache (always a multiple of `page_size`) and `pages` are the
    /// `PageId`s covering those tokens. `current_generations[pid.0 as usize]`
    /// must hold the current generation for each page — a mismatch evicts
    /// the candidate and stops the match.
    pub fn lookup_longest_match(
        &self,
        tokens: &[u32],
        current_generations: &[u64],
    ) -> (usize, Vec<PageId>) {
        let mut node = &self.root;
        let mut matched_tokens = 0usize;
        let mut matched_pages: Vec<PageId> = Vec::new();
        let take = (tokens.len() as u32).min(self.max_depth) as usize;
        for (i, &t) in tokens.iter().take(take).enumerate() {
            let Some(child) = node.children.get(&t) else {
                break;
            };
            node = child;
            let pos = i + 1;
            if pos % self.page_size as usize == 0 {
                if let Some((pid, stored_gen)) = node.page_here {
                    let cur_gen = current_generations
                        .get(pid.0 as usize)
                        .copied()
                        .unwrap_or(0);
                    if cur_gen == stored_gen {
                        matched_pages.push(pid);
                        matched_tokens = pos;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        (matched_tokens, matched_pages)
    }

    pub fn refcount_inc(&mut self, tokens: &[u32]) {
        let mut node = &mut self.root;
        node.refcount = node.refcount.saturating_add(1);
        let take = (tokens.len() as u32).min(self.max_depth) as usize;
        for &t in tokens.iter().take(take) {
            let Some(child) = node.children.get_mut(&t) else {
                break;
            };
            child.refcount = child.refcount.saturating_add(1);
            node = child;
        }
    }

    pub fn refcount_dec(&mut self, tokens: &[u32]) {
        let mut node = &mut self.root;
        node.refcount = node.refcount.saturating_sub(1);
        let take = (tokens.len() as u32).min(self.max_depth) as usize;
        for &t in tokens.iter().take(take) {
            let Some(child) = node.children.get_mut(&t) else {
                break;
            };
            child.refcount = child.refcount.saturating_sub(1);
            node = child;
        }
    }

    /// Refcount at the deepest node reachable along `tokens`. Used by
    /// tests to verify shared-prefix behaviour.
    pub fn refcount_at_prefix(&self, tokens: &[u32]) -> u32 {
        let mut node = &self.root;
        let take = (tokens.len() as u32).min(self.max_depth) as usize;
        for &t in tokens.iter().take(take) {
            let Some(child) = node.children.get(&t) else {
                return 0;
            };
            node = child;
        }
        node.refcount
    }
}
