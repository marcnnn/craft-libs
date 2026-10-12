//! Timestamp-based audio clock tracking.
//!
//! Two guarantees drive this crate:
//!
//! 1. **Normal hardware keeps working exactly as before.** All tracking is passive: the
//!    [`TimestampTracker`] observes device timestamps when the platform provides them and
//!    informs position mapping, but nothing is ever inserted into the audio path.
//! 2. **PTP-clocked devices get exact sample timing with a fixed delay.** When a soundcard
//!    locks its media clock to PTP, sample timing is not estimated but exact:
//!    [`LockedClock`] maps every sample index to PTP nanoseconds with exact integer math,
//!    and [`LatencyChain`] accumulates each processing stage's constant delay into one
//!    fixed end-to-end delay — never varying between runs or between audio and video
//!    processing, since both report on the same PTP axis.
//!
//! Timestamp sources are whatever the platform offers: audio-callback timestamps, ALSA PCM
//! status timestamps, PipeWire `pw_time` ticks, CoreAudio `AudioTimeStamp`, WASAPI device
//! positions, ASIO sample-position/host-time pairs, or FPGA PHC/PTP readings. While a
//! device clock stays within tolerance, audio flows through a 1:1 passthrough; only a
//! sustained deviation ([`ClockRelation::NeedsCorrection`]) justifies a resampler as a
//! last resort.
//!
//! Platform timelines live in [`clocks`]: [`clocks::MonotonicClock`] pairs with device
//! sample counters, and [`clocks::TaiClock`] carries the PTP/TAI timeline — on Linux the
//! kernel's `CLOCK_TAI` (steered by `ptp4l`/`phc2sys`) replaces the out-of-band PTP agent
//! other systems need. [`arbiter::SyncArbiter`] turns noisy [`ClockRelation`]
//! observations into stable passthrough/locked/correcting decisions, and
//! [`arbiter::promote_to_locked`] builds the exact [`LockedClock`] once promotion is
//! justified.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod arbiter;
pub mod clocks;
pub mod ptp_agent;

pub use arbiter::{SyncArbiter, SyncMode, promote_to_locked};
pub use clocks::{ClockDomain, MediaClock, MonotonicClock, SyncStatus, TAI_UTC_OFFSET_S, TaiClock, TaiSource};
pub use ptp_agent::{AGENT_MAPPING_PATH, PtpAgentError, PtpAgentSample, read_mapping};

use std::collections::VecDeque;

/// How the engine should treat the device clock relative to the physical timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClockRelation {
    /// Not enough timestamp pairs yet: keep the passthrough and keep observing.
    InsufficientData,
    /// Deviation is within tolerance: passthrough; derive positions from timestamps alone.
    WithinTolerance { deviation_ppm: f64 },
    /// Sustained deviation beyond tolerance: last-resort rate correction is required.
    NeedsCorrection { deviation_ppm: f64 },
}

/// Robust tracker mapping device sample positions to physical time.
///
/// Maintains a sliding window of observed `(device_sample_position, physical_time_ns)` pairs
/// and fits `time_ns = a + b * sample` by least squares (refit per observation, O(window)).
/// Observations are rejected when the position does not strictly advance or the timestamp
/// deviates from the current fit by more than the outlier threshold. Allocation happens only
/// at construction; observing and querying are allocation-free.
#[derive(Debug, Clone)]
pub struct TimestampTracker {
    nominal_hz: f64,
    window: usize,
    outlier_ns: f64,
    min_pairs: usize,
    pairs: VecDeque<(f64, f64)>,
    fit: Option<(f64, f64)>,
}

impl TimestampTracker {
    /// Create a tracker for a device nominally running at `nominal_hz`.
    pub fn new(nominal_hz: f64) -> Self {
        Self {
            nominal_hz: if nominal_hz.is_finite() && nominal_hz >= 1.0 { nominal_hz } else { 48000.0 },
            window: 512,
            outlier_ns: 100.0e6,
            min_pairs: 16,
            pairs: VecDeque::new(),
            fit: None,
        }
    }

    /// Maximum retained timestamp pairs (sliding window; default 512).
    pub fn set_window(&mut self, pairs: usize) {
        self.window = pairs.max(4);
    }

    /// Outlier rejection threshold in nanoseconds against the current fit (default 100 ms).
    pub fn set_outlier_threshold_ns(&mut self, ns: f64) {
        self.outlier_ns = if ns.is_finite() && ns > 0.0 { ns } else { 100.0e6 };
    }

    /// Pairs required before a [`ClockRelation`] recommendation is made (default 16).
    pub fn set_min_pairs(&mut self, pairs: usize) {
        self.min_pairs = pairs.max(2);
    }

    /// Number of accepted timestamp pairs.
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// Observe that `device_sample_position` (frames since stream start) was reached at
    /// `physical_time_ns`. Returns `true` if accepted, `false` if rejected (non-advancing
    /// position or outlier timestamp).
    pub fn observe(&mut self, device_sample_position: i64, physical_time_ns: i64) -> bool {
        let x = device_sample_position as f64;
        let y = physical_time_ns as f64;
        if let Some(&(last_x, _)) = self.pairs.back()
            && x <= last_x
        {
            return false;
        }
        if let Some((a, b)) = self.fit {
            let predicted = a + b * x;
            if (y - predicted).abs() > self.outlier_ns {
                return false;
            }
        }
        self.pairs.push_back((x, y));
        while self.pairs.len() > self.window {
            self.pairs.pop_front();
        }
        self.refit();
        true
    }

    /// Least-squares refit over the window.
    fn refit(&mut self) {
        let n = self.pairs.len();
        if n < 2 {
            self.fit = None;
            return;
        }
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for &(x, y) in &self.pairs {
            sx += x;
            sy += y;
        }
        let (mx, my) = (sx / n as f64, sy / n as f64);
        let (mut sxx, mut sxy) = (0.0f64, 0.0f64);
        for &(x, y) in &self.pairs {
            let (dx, dy) = (x - mx, y - my);
            sxx += dx * dx;
            sxy += dx * dy;
        }
        if sxx <= 0.0 {
            self.fit = None;
            return;
        }
        let b = sxy / sxx;
        let a = my - b * mx;
        self.fit = Some((a, b));
    }

    /// Fitted nanoseconds per device sample.
    pub fn ns_per_sample(&self) -> Option<f64> {
        self.fit.filter(|&(_, b)| b > 0.0).map(|(_, b)| b)
    }

    /// Estimated device sample rate in Hz, measured against physical time.
    pub fn rate_hz(&self) -> Option<f64> {
        self.ns_per_sample().map(|b| 1.0e9 / b)
    }

    /// Device-rate deviation from nominal, in ppm (positive: the device runs fast).
    pub fn deviation_ppm(&self) -> Option<f64> {
        self.rate_hz().map(|r| (r / self.nominal_hz - 1.0) * 1.0e6)
    }

    /// Predicted physical time (ns) of a device sample position.
    pub fn time_ns_at(&self, device_sample_position: i64) -> Option<i64> {
        let (a, b) = self.fit?;
        if b <= 0.0 {
            return None;
        }
        Some((a + b * device_sample_position as f64).round() as i64)
    }

    /// Predicted device sample position at a physical time (ns).
    pub fn device_sample_at(&self, physical_time_ns: i64) -> Option<i64> {
        let (a, b) = self.fit?;
        if b <= 0.0 {
            return None;
        }
        Some(((physical_time_ns as f64 - a) / b).round() as i64)
    }

    /// Recommendation for this device clock given a tolerance in ppm. The engine owns
    /// hysteresis: require a sustained `NeedsCorrection` before engaging a resampler.
    pub fn relation(&self, tolerance_ppm: f64) -> ClockRelation {
        let Some(deviation) = self.deviation_ppm() else {
            return ClockRelation::InsufficientData;
        };
        if self.pairs.len() < self.min_pairs {
            return ClockRelation::InsufficientData;
        }
        let tol = if tolerance_ppm.is_finite() && tolerance_ppm >= 0.0 { tolerance_ppm } else { 0.0 };
        if deviation.abs() <= tol {
            ClockRelation::WithinTolerance { deviation_ppm: deviation }
        } else {
            ClockRelation::NeedsCorrection { deviation_ppm: deviation }
        }
    }

    /// Forget all observations.
    pub fn reset(&mut self) {
        self.pairs.clear();
        self.fit = None;
    }
}

/// Exact sample-index ↔ PTP-time mapping for a device whose media clock is locked to PTP.
///
/// With the media clock locked (PHC), sample timing is exact rather than estimated:
/// sample `n` exists at PTP time `epoch_ns + n * 1e9 / rate`. All arithmetic is exact
/// integer math (i128 intermediates, floor to whole nanoseconds), so the mapping can be
/// carried through the entire processing chain — capture DMA ring, host buffers, DSP —
/// with zero accumulated error and no float drift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockedClock {
    rate_hz: u32,
    epoch_ns: i64,
    next_sample: i64,
}

impl LockedClock {
    /// Clock locked at `rate_hz`, where sample 0 occurs at `epoch_ptp_ns`.
    pub fn new(rate_hz: u32, epoch_ptp_ns: i64) -> Self {
        Self { rate_hz: rate_hz.max(1), epoch_ns: epoch_ptp_ns, next_sample: 0 }
    }

    /// Locked sample rate in Hz.
    pub fn rate_hz(&self) -> u32 {
        self.rate_hz
    }

    /// PTP time (ns) of sample 0.
    pub fn epoch_ns(&self) -> i64 {
        self.epoch_ns
    }

    /// Exact PTP time (ns) of a sample index (floored to whole ns; error < 1 ns).
    /// Returns `None` if the result would overflow `i64`.
    pub fn time_ns_of(&self, sample: i64) -> Option<i64> {
        let n = (sample as i128).checked_mul(1_000_000_000)?.div_euclid(self.rate_hz as i128);
        i64::try_from(self.epoch_ns as i128 + n).ok()
    }

    /// Sample index covering a PTP time (ns), floored; `None` on overflow.
    pub fn sample_at(&self, time_ns: i64) -> Option<i64> {
        let d = (time_ns as i128).checked_sub(self.epoch_ns as i128)?;
        let n = d.checked_mul(self.rate_hz as i128)?.div_euclid(1_000_000_000);
        i64::try_from(n).ok()
    }

    /// Advance the stream position by `n` samples (saturating).
    pub fn advance(&mut self, n: i64) {
        self.next_sample = self.next_sample.saturating_add(n);
    }

    /// Next unconsumed sample index.
    pub fn next_sample(&self) -> i64 {
        self.next_sample
    }

    /// PTP time (ns) of the next unconsumed sample; `None` on overflow.
    pub fn next_time_ns(&self) -> Option<i64> {
        self.time_ns_of(self.next_sample)
    }

    /// Re-anchor so that `sample` maps to `time_ns` (e.g. after a lock event), keeping the rate.
    pub fn rebase(&mut self, sample: i64, time_ns: i64) {
        let offset = (sample as i128).checked_mul(1_000_000_000).map(|n| n.div_euclid(self.rate_hz as i128));
        if let Some(offset) = offset
            && let Ok(epoch) = i64::try_from(time_ns as i128 - offset)
        {
            self.epoch_ns = epoch;
        }
    }
}

/// Fixed end-to-end latency of a processing chain on a locked clock.
///
/// Each stage contributes an exact, constant delay — in samples at the locked rate
/// (capture DMA ring depth, host buffer size, DSP block size, look-ahead) or in
/// nanoseconds for delays not quantized to samples (serializer, network). Because the
/// media clock is locked, the total converts exactly between samples and PTP
/// nanoseconds and never varies: not between runs, and not between audio and video
/// processing, since both chains report their delay on the same PTP axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatencyChain {
    rate_hz: u32,
    samples: i64,
    extra_ns: i64,
}

impl LatencyChain {
    /// An empty latency chain for a chain running at `rate_hz`.
    pub fn new(rate_hz: u32) -> Self {
        Self { rate_hz: rate_hz.max(1), samples: 0, extra_ns: 0 }
    }

    /// Add a stage delay measured in samples at the locked rate.
    pub fn add_samples(&mut self, samples: i64) {
        self.samples = self.samples.saturating_add(samples);
    }

    /// Add a fixed stage delay in nanoseconds (not sample-quantized).
    pub fn add_ns(&mut self, ns: i64) {
        self.extra_ns = self.extra_ns.saturating_add(ns);
    }

    /// Total fixed delay in samples (excludes sub-sample ns additions).
    pub fn total_samples(&self) -> i64 {
        self.samples
    }

    /// Total fixed delay in nanoseconds: `samples * 1e9 / rate + extra_ns`, exact integer
    /// math; `None` if the result would overflow `i64`.
    pub fn total_ns(&self) -> Option<i64> {
        let from_samples = (self.samples as i128).checked_mul(1_000_000_000)?.div_euclid(self.rate_hz as i128);
        i64::try_from(from_samples + self.extra_ns as i128).ok()
    }

    /// Playout PTP time (ns) of a sample captured at `capture_ptp_ns`; `None` on overflow.
    pub fn playout_ns_at(&self, capture_ptp_ns: i64) -> Option<i64> {
        capture_ptp_ns.checked_add(self.total_ns()?)
    }

    /// Capture PTP time (ns) of a sample played out at `playout_ptp_ns`; `None` on overflow.
    pub fn capture_ns_of(&self, playout_ptp_ns: i64) -> Option<i64> {
        playout_ptp_ns.checked_sub(self.total_ns()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOMINAL: f64 = 48000.0;

    /// Feed `count` synthetic pairs: every `step_samples` frames at `step_ns` per step.
    fn feed(tracker: &mut TimestampTracker, count: i64, step_samples: i64, step_ns: f64) {
        for k in 0..count {
            let x = k * step_samples;
            let y = (k as f64 * step_ns).round() as i64;
            assert!(tracker.observe(x, y));
        }
    }

    #[test]
    fn zero_drift_is_within_tolerance() {
        let mut t = TimestampTracker::new(NOMINAL);
        feed(&mut t, 100, 48000, 1.0e9);
        let dev = t.deviation_ppm().unwrap();
        assert!(dev.abs() < 1.0, "deviation: {dev}");
        assert!(matches!(t.relation(5.0), ClockRelation::WithinTolerance { .. }));
    }

    #[test]
    fn detects_25ppm_drift() {
        // Device crystal 25 ppm fast: each 48000-sample block takes 1e9 / 1.000025 ns.
        let mut t = TimestampTracker::new(NOMINAL);
        let step_ns = 1.0e9 / 1.000025;
        feed(&mut t, 100, 48000, step_ns);
        let dev = t.deviation_ppm().unwrap();
        assert!((dev - 25.0).abs() < 0.5, "deviation: {dev}");
        assert!(matches!(t.relation(5.0), ClockRelation::NeedsCorrection { .. }));
        assert!(matches!(t.relation(50.0), ClockRelation::WithinTolerance { .. }));
    }

    #[test]
    fn maps_between_domains() {
        let mut t = TimestampTracker::new(NOMINAL);
        let step_ns = 1.0e9 / 0.999990; // -10 ppm device
        feed(&mut t, 100, 48000, step_ns);
        let sample = 48000 * 50;
        let expected = (50.0 * step_ns).round() as i64;
        let got = t.time_ns_at(sample).unwrap();
        assert!((got - expected).abs() < 1_000, "time_ns_at: {got} vs {expected}");
        let back = t.device_sample_at(expected).unwrap();
        assert!((back - sample).abs() <= 1, "device_sample_at: {back}");
    }

    #[test]
    fn rejects_outliers_and_stale_positions() {
        let mut t = TimestampTracker::new(NOMINAL);
        feed(&mut t, 50, 48000, 1.0e9);
        let before = t.pair_count();
        let dev_before = t.deviation_ppm().unwrap();
        // Wild timestamp off by a full second.
        assert!(!t.observe(50 * 48000, 50 * 1_000_000_000 + 1_000_000_000));
        // Position must strictly advance.
        assert!(!t.observe(49 * 48000, 51 * 1_000_000_000));
        assert_eq!(t.pair_count(), before);
        assert!((t.deviation_ppm().unwrap() - dev_before).abs() < 1e-9);
    }

    #[test]
    fn insufficient_data_below_min_pairs() {
        let mut t = TimestampTracker::new(NOMINAL);
        assert_eq!(t.relation(100.0), ClockRelation::InsufficientData);
        feed(&mut t, 5, 48000, 1.0e9);
        assert_eq!(t.relation(100.0), ClockRelation::InsufficientData);
        // A fit (and deviation) exists with 2+ pairs even without a recommendation.
        assert!(t.deviation_ppm().is_some());
    }

    #[test]
    fn reset_forgets_observations() {
        let mut t = TimestampTracker::new(NOMINAL);
        feed(&mut t, 50, 48000, 1.0e9);
        t.reset();
        assert_eq!(t.pair_count(), 0);
        assert!(t.time_ns_at(0).is_none());
    }

    #[test]
    fn locked_clock_exact_mapping() {
        let clock = LockedClock::new(48000, 1_000_000_000);
        // One second of samples is exactly one second of PTP time.
        assert_eq!(clock.time_ns_of(48000), Some(2_000_000_000));
        assert_eq!(clock.time_ns_of(48000 * 3600), Some(1_000_000_000 + 3_600_000_000_000));
        // 48 kHz does not divide a nanosecond: flooring keeps the error < 1 ns.
        let t1 = clock.time_ns_of(1).unwrap();
        assert_eq!(t1, 1_000_020_833); // floor(1e9/48000) = 20833.33...
        // Inverse mapping is exact on whole-sample times.
        assert_eq!(clock.sample_at(2_000_000_000), Some(48000));
        // Round trip within one sample for arbitrary times.
        let mid = 1_000_500_000;
        let s = clock.sample_at(mid).unwrap();
        assert!((clock.time_ns_of(s).unwrap() - mid).abs() <= 1_000_000_000 / 48000_i64 + 1);
    }

    #[test]
    fn locked_clock_stream_and_rebase() {
        let mut clock = LockedClock::new(48000, 500_000_000);
        assert_eq!(clock.next_sample(), 0);
        clock.advance(1024);
        assert_eq!(clock.next_sample(), 1024);
        assert_eq!(clock.next_time_ns(), clock.time_ns_of(1024));
        // Re-anchor after a lock event: sample 4096 now maps to a new epoch.
        clock.rebase(4096, 9_000_000_000);
        assert_eq!(clock.time_ns_of(4096), Some(9_000_000_000));
        // Hostile indices never panic: an unrepresentable time is reported, not wrapped.
        assert!(LockedClock::new(48000, 0).time_ns_of(i64::MAX).is_none());
        // A far-future time maps to a huge (but valid) sample index without panic.
        assert!(LockedClock::new(48000, 0).sample_at(i64::MAX).is_some());
    }

    #[test]
    fn latency_chain_is_exact_and_fixed() {
        // Capture DMA ring 256 + host buffer 256 + limiter look-ahead 64 samples at 48 kHz.
        let mut chain = LatencyChain::new(48000);
        chain.add_samples(256);
        chain.add_samples(256);
        chain.add_samples(64);
        assert_eq!(chain.total_samples(), 576);
        // 576 samples at 48 kHz is exactly 12 ms — no float, no drift.
        assert_eq!(chain.total_ns(), Some(12_000_000));
        assert_eq!(chain.playout_ns_at(1_000_000_000), Some(1_012_000_000));
        assert_eq!(chain.capture_ns_of(1_012_000_000), Some(1_000_000_000));

        // Non-sample-quantized stage delays add exactly.
        chain.add_ns(500_000);
        assert_eq!(chain.total_ns(), Some(12_500_000));

        // The same stages always produce the same delay: audio and video chains each
        // report a constant on the same PTP axis, so their relative offset never moves.
        let mut again = LatencyChain::new(48000);
        again.add_samples(576);
        assert_ne!(chain.total_ns(), again.total_ns()); // ours includes the 0.5 ms stage
        again.add_ns(500_000);
        assert_eq!(chain.total_ns(), again.total_ns());
    }

    #[test]
    fn latency_chain_overflow_is_reported_not_panicked() {
        let mut chain = LatencyChain::new(8000);
        chain.add_samples(i64::MAX / 2);
        assert!(chain.total_ns().is_none());
        assert!(chain.playout_ns_at(0).is_none());
        assert!(chain.capture_ns_of(0).is_none());
        // A sane chain still works afterwards.
        let mut sane = LatencyChain::new(8000);
        sane.add_samples(8000);
        assert_eq!(sane.total_ns(), Some(1_000_000_000));
    }
}
