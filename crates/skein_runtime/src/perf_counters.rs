//! Process-global decode perf counters that quantify the CPU round-trips in the
//! current segment-staging path — the traffic the device-resident handoff work
//! is meant to remove.
//!
//! Bytes are counted as **host-side f32 bytes** (`len * 4`): the size of the
//! `Vec<f32>` that is materialized on the host and then uploaded (H2D) or that
//! is read back from the device (D2H). `host_materializations` is the exact
//! count of such host tensors created per the staging path. These are honest
//! measurements of host traffic; they are not a substitute for a real fix.

use std::sync::atomic::{AtomicU64, Ordering};

static H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static D2H_BYTES: AtomicU64 = AtomicU64::new(0);
static HOST_MATERIALIZATIONS: AtomicU64 = AtomicU64::new(0);
static SEGMENT_LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Record a host->device staging of `bytes` host bytes (one host materialization).
#[inline]
pub fn record_h2d(bytes: usize) {
    H2D_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    HOST_MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
}

/// Record a device->host read of `bytes` host bytes (one host materialization).
#[inline]
pub fn record_d2h(bytes: usize) {
    D2H_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    HOST_MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
}

/// Record one segment execution (a graph launch on the device).
#[inline]
pub fn record_segment_launch() {
    SEGMENT_LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

/// A point-in-time snapshot of the process-global counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerfSnapshot {
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub host_materializations: u64,
    pub segment_launches: u64,
}

/// Read the current counters.
pub fn snapshot() -> PerfSnapshot {
    PerfSnapshot {
        h2d_bytes: H2D_BYTES.load(Ordering::Relaxed),
        d2h_bytes: D2H_BYTES.load(Ordering::Relaxed),
        host_materializations: HOST_MATERIALIZATIONS.load(Ordering::Relaxed),
        segment_launches: SEGMENT_LAUNCHES.load(Ordering::Relaxed),
    }
}

impl PerfSnapshot {
    /// Per-token deltas: `(self - earlier) / tokens`, saturating. `tokens` is
    /// clamped to at least 1.
    pub fn per_token(self, earlier: PerfSnapshot, tokens: u64) -> PerfSnapshot {
        let t = tokens.max(1);
        PerfSnapshot {
            h2d_bytes: self.h2d_bytes.saturating_sub(earlier.h2d_bytes) / t,
            d2h_bytes: self.d2h_bytes.saturating_sub(earlier.d2h_bytes) / t,
            host_materializations: self
                .host_materializations
                .saturating_sub(earlier.host_materializations)
                / t,
            segment_launches: self
                .segment_launches
                .saturating_sub(earlier.segment_launches)
                / t,
        }
    }
}
