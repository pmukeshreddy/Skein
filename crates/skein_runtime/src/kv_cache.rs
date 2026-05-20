//! Per-request KV cache for incremental (cached) decoding.
//!
//! In cached decode the model attends to *all* prior tokens without
//! recomputing them: each step computes only the new token's key/value, and the
//! attention reads the accumulated past. Skein keeps the accumulation here, in
//! the runtime, so each compiled decode segment stays fixed-shape per step
//! (current token's K/V) and reads the growing past as a dynamic-length input.
//!
//! Naming convention tying the graph to this cache: an attention layer `L`
//! reads its past keys/values from handoffs `kvcache_k_{L}` / `kvcache_v_{L}`
//! and writes the *new* token's keys/values back under the same names; the
//! runner ([`crate::distributed::SegmentRunner`]) feeds the accumulated cache in
//! and appends the new step's output here. See [`parse_kvcache_name`].

/// Which half of the cache a handoff carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvKind {
    Key,
    Value,
}

/// Per-layer accumulated keys and values (flat `f32`, row-major over the
/// cached tokens). Grows by one token's worth each decode step.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl KvCache {
    /// A cache for `layers` attention layers, all initially empty (prefill
    /// seeds them on the first step).
    pub fn new(layers: usize) -> Self {
        Self {
            keys: vec![Vec::new(); layers],
            values: vec![Vec::new(); layers],
        }
    }

    pub fn layers(&self) -> usize {
        self.keys.len()
    }

    /// Accumulated past for `(kind, layer)`. Empty before the first step.
    pub fn past(&self, kind: KvKind, layer: usize) -> &[f32] {
        let store = match kind {
            KvKind::Key => &self.keys,
            KvKind::Value => &self.values,
        };
        store.get(layer).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Append this step's new keys/values for `(kind, layer)`.
    pub fn append(&mut self, kind: KvKind, layer: usize, data: &[f32]) {
        let store = match kind {
            KvKind::Key => &mut self.keys,
            KvKind::Value => &mut self.values,
        };
        if layer >= store.len() {
            store.resize(layer + 1, Vec::new());
        }
        store[layer].extend_from_slice(data);
    }

    /// Total cached elements for `(kind, layer)` — proportional to the number of
    /// cached tokens.
    pub fn len(&self, kind: KvKind, layer: usize) -> usize {
        self.past(kind, layer).len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.iter().all(|k| k.is_empty()) && self.values.iter().all(|v| v.is_empty())
    }

    /// Clear all layers (new request reusing the same cache allocation).
    pub fn reset(&mut self) {
        for k in &mut self.keys {
            k.clear();
        }
        for v in &mut self.values {
            v.clear();
        }
    }
}

/// Parse a KV-cache handoff name `kvcache_{k|v}_{layer}` into `(kind, layer)`.
/// Returns `None` for any other handoff (the runner treats those normally).
pub fn parse_kvcache_name(name: &str) -> Option<(KvKind, usize)> {
    let rest = name.strip_prefix("kvcache_")?;
    let (tag, layer) = rest.split_once('_')?;
    let kind = match tag {
        "k" => KvKind::Key,
        "v" => KvKind::Value,
        _ => return None,
    };
    Some((kind, layer.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_grows_and_past_concatenates() {
        let mut c = KvCache::new(2);
        assert!(c.is_empty());
        c.append(KvKind::Key, 0, &[1.0, 2.0]);
        c.append(KvKind::Key, 0, &[3.0, 4.0]);
        c.append(KvKind::Value, 0, &[9.0]);
        assert_eq!(c.past(KvKind::Key, 0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.past(KvKind::Value, 0), &[9.0]);
        assert_eq!(c.past(KvKind::Key, 1), &[] as &[f32]);
        assert!(!c.is_empty());
    }

    #[test]
    fn reset_clears_but_keeps_layers() {
        let mut c = KvCache::new(1);
        c.append(KvKind::Key, 0, &[1.0]);
        c.reset();
        assert!(c.is_empty());
        assert_eq!(c.layers(), 1);
    }

    #[test]
    fn parses_cache_names() {
        assert_eq!(parse_kvcache_name("kvcache_k_0"), Some((KvKind::Key, 0)));
        assert_eq!(parse_kvcache_name("kvcache_v_12"), Some((KvKind::Value, 12)));
        assert_eq!(parse_kvcache_name("block_3_attn_out"), None);
        assert_eq!(parse_kvcache_name("kvcache_x_1"), None);
        assert_eq!(parse_kvcache_name("logits"), None);
    }
}
