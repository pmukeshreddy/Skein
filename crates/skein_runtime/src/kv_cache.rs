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

/// Per-layer **fixed-capacity** keys and values (flat `f32`, row-major over a
/// fixed `KV_CACHE_CAP` slots × `n_kv*head_dim`). Each layer's buffer is
/// allocated zero on first use and the new token's K/V is written into slot
/// `position` each decode step; the graph reads the whole buffer and masks slots
/// `> position`. (The previous growing/append design used a dynamic `past` dim,
/// which luminal's CUDA backend can't represent at `past == 0`.)
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl KvCache {
    /// A cache for `layers` attention layers; each layer's fixed buffer is
    /// allocated lazily on first [`buffer`](Self::buffer) call.
    pub fn new(layers: usize) -> Self {
        Self {
            keys: vec![Vec::new(); layers],
            values: vec![Vec::new(); layers],
        }
    }

    pub fn layers(&self) -> usize {
        self.keys.len()
    }

    fn store_mut(&mut self, kind: KvKind) -> &mut Vec<Vec<f32>> {
        match kind {
            KvKind::Key => &mut self.keys,
            KvKind::Value => &mut self.values,
        }
    }

    /// The whole fixed-capacity buffer for `(kind, layer)`, allocated to
    /// `full_len` zeros on first use. Fed to the graph each step.
    pub fn buffer(&mut self, kind: KvKind, layer: usize, full_len: usize) -> &[f32] {
        let store = self.store_mut(kind);
        while store.len() <= layer {
            store.push(Vec::new());
        }
        if full_len > 0 && store[layer].len() < full_len {
            store[layer].resize(full_len, 0.0);
        }
        &store[layer]
    }

    /// Write this step's new token K/V `data` into `slot` of `(kind, layer)`
    /// (offset `slot * data.len()`). No-op if the buffer isn't allocated yet or
    /// the slot is out of range.
    pub fn write_slot(&mut self, kind: KvKind, layer: usize, slot: usize, data: &[f32]) {
        let store = self.store_mut(kind);
        let Some(buf) = store.get_mut(layer) else {
            return;
        };
        if data.is_empty() {
            return;
        }
        let off = slot * data.len();
        if off + data.len() <= buf.len() {
            buf[off..off + data.len()].copy_from_slice(data);
        }
    }

    /// Read-only view of `(kind, layer)`'s fixed buffer (empty if unallocated).
    pub fn peek(&self, kind: KvKind, layer: usize) -> &[f32] {
        let store = match kind {
            KvKind::Key => &self.keys,
            KvKind::Value => &self.values,
        };
        store.get(layer).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn is_empty(&self) -> bool {
        self.keys.iter().all(|k| k.is_empty()) && self.values.iter().all(|v| v.is_empty())
    }

    /// Zero all layers for a new request, keeping the allocations.
    pub fn reset(&mut self) {
        for k in &mut self.keys {
            k.iter_mut().for_each(|x| *x = 0.0);
        }
        for v in &mut self.values {
            v.iter_mut().for_each(|x| *x = 0.0);
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
    fn fixed_buffer_writes_into_slots() {
        // 2 layers, per-token width 2, capacity 3 slots -> full_len = 6.
        let mut c = KvCache::new(2);
        assert!(c.is_empty());
        assert_eq!(c.buffer(KvKind::Key, 0, 6), &[0.0; 6]); // lazily zero-allocated
        // Write token 0 into slot 0 and token 1 into slot 1.
        c.write_slot(KvKind::Key, 0, 0, &[1.0, 2.0]);
        c.write_slot(KvKind::Key, 0, 1, &[3.0, 4.0]);
        let _ = c.buffer(KvKind::Value, 0, 6);
        c.write_slot(KvKind::Value, 0, 0, &[9.0, 8.0]);
        assert_eq!(c.buffer(KvKind::Key, 0, 6), &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
        assert_eq!(c.buffer(KvKind::Value, 0, 6), &[9.0, 8.0, 0.0, 0.0, 0.0, 0.0]);
        // Out-of-range slot is a no-op (slot 3 would start at offset 6).
        c.write_slot(KvKind::Key, 0, 3, &[7.0, 7.0]);
        assert_eq!(c.buffer(KvKind::Key, 0, 6), &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
        assert!(!c.is_empty());
    }

    #[test]
    fn reset_zeros_but_keeps_layers() {
        let mut c = KvCache::new(1);
        let _ = c.buffer(KvKind::Key, 0, 4);
        c.write_slot(KvKind::Key, 0, 0, &[1.0, 2.0]);
        c.reset();
        assert_eq!(c.buffer(KvKind::Key, 0, 4), &[0.0; 4]);
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
