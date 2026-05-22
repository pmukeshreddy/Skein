//! Compile-time gated decode-hot-loop timing.
//!
//! The per-segment input-stage / GPU-launch / output-capture split and the
//! per-forward segment-vs-collective split are useful when profiling decode, but
//! a release build should not pay for `Instant::now()` calls (two per segment,
//! several per forward — hundreds per token).
//!
//! So the timing is gated by the `perf-trace` Cargo feature, not a runtime env
//! var. With the feature ON, [`SegTimer`] / [`StepTimer`] measure and emit
//! `tracing::debug!` records. With it OFF (the production default) they are
//! zero-sized no-op shims whose `#[inline(always)]` methods compile to nothing —
//! there are no `Instant::now()` calls in the hot loop at all, and `tracing`
//! macros (whose arguments would otherwise be evaluated even when filtered) are
//! not present.

// ---------------------------------------------------------------------------
// SegTimer — per-segment timing in `SegmentRunner::run_segment`.
// ---------------------------------------------------------------------------

/// Times one segment: host-input staging, GPU launch, host-output capture.
#[cfg(feature = "perf-trace")]
pub struct SegTimer {
    in_start: std::time::Instant,
    host_in_us: u128,
    gpu_us: u128,
    gpu_start: Option<std::time::Instant>,
    host_out_start: Option<std::time::Instant>,
}

#[cfg(feature = "perf-trace")]
impl SegTimer {
    /// Start timing input staging.
    #[inline]
    pub fn start_in() -> Self {
        Self {
            in_start: std::time::Instant::now(),
            host_in_us: 0,
            gpu_us: 0,
            gpu_start: None,
            host_out_start: None,
        }
    }
    /// Inputs staged; record host-in time and start the GPU-launch timer.
    #[inline]
    pub fn mark_gpu(&mut self) {
        self.host_in_us = self.in_start.elapsed().as_micros();
        self.gpu_start = Some(std::time::Instant::now());
    }
    /// Execute launched; record GPU-launch time and start the host-out timer.
    #[inline]
    pub fn mark_out(&mut self) {
        self.gpu_us = self
            .gpu_start
            .expect("mark_gpu before mark_out")
            .elapsed()
            .as_micros();
        self.host_out_start = Some(std::time::Instant::now());
    }
    /// Outputs captured; emit the per-segment record.
    #[inline]
    pub fn finish(self, segment_idx: usize) {
        let host_out_us = self
            .host_out_start
            .expect("mark_out before finish")
            .elapsed()
            .as_micros();
        tracing::debug!(
            segment_idx,
            host_in_us = self.host_in_us,
            gpu_launch_us = self.gpu_us,
            host_out_us,
            "SKEIN_SEG"
        );
    }
}

/// Zero-sized no-op when `perf-trace` is off: no `Instant`, no `tracing`.
#[cfg(not(feature = "perf-trace"))]
pub struct SegTimer;

#[cfg(not(feature = "perf-trace"))]
impl SegTimer {
    #[inline(always)]
    pub fn start_in() -> Self {
        Self
    }
    #[inline(always)]
    pub fn mark_gpu(&mut self) {}
    #[inline(always)]
    pub fn mark_out(&mut self) {}
    #[inline(always)]
    pub fn finish(self, _segment_idx: usize) {}
}

// ---------------------------------------------------------------------------
// StepTimer — per-forward segment-exec vs collective accumulators in
// `RankExecutor::run`.
// ---------------------------------------------------------------------------

/// Accumulates per-forward time spent in local segment execution vs in
/// collectives / MoE routing.
#[cfg(feature = "perf-trace")]
pub struct StepTimer {
    seg_us: u128,
    comm_us: u128,
}

#[cfg(feature = "perf-trace")]
impl StepTimer {
    #[inline]
    pub fn new() -> Self {
        Self {
            seg_us: 0,
            comm_us: 0,
        }
    }
    /// Run `f`, adding its wall time to the segment-exec accumulator.
    #[inline]
    pub fn time_seg<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let t = std::time::Instant::now();
        let r = f();
        self.seg_us += t.elapsed().as_micros();
        r
    }
    /// Run `f`, adding its wall time to the collective/route accumulator.
    #[inline]
    pub fn time_comm<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let t = std::time::Instant::now();
        let r = f();
        self.comm_us += t.elapsed().as_micros();
        r
    }
    /// Emit the per-forward record.
    #[inline]
    pub fn finish(self) {
        tracing::debug!(
            seg_us = self.seg_us,
            comm_us = self.comm_us,
            "SKEIN_PERF_STEP: segment-exec vs collective time for one forward pass"
        );
    }
}

#[cfg(feature = "perf-trace")]
impl Default for StepTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-sized no-op when `perf-trace` is off: `time_seg`/`time_comm` just run the
/// closure with no timing.
#[cfg(not(feature = "perf-trace"))]
pub struct StepTimer;

#[cfg(not(feature = "perf-trace"))]
impl StepTimer {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }
    #[inline(always)]
    pub fn time_seg<T>(&mut self, f: impl FnOnce() -> T) -> T {
        f()
    }
    #[inline(always)]
    pub fn time_comm<T>(&mut self, f: impl FnOnce() -> T) -> T {
        f()
    }
    #[inline(always)]
    pub fn finish(self) {}
}

#[cfg(not(feature = "perf-trace"))]
impl Default for StepTimer {
    fn default() -> Self {
        Self::new()
    }
}
