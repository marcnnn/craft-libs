//! Physical clock reading and the TAI timeline.
//!
//! On Linux the "PTP agent" is the kernel itself: `ptp4l` disciplines the NIC's PHC and
//! `phc2sys` steers `CLOCK_TAI`, so every process reads the shared PTP mapping with a plain
//! `clock_gettime` — no IPC file, no ports (see HANDOFF.md §2G). This module provides the
//! dependency-free in-process halves: a monotonic anchor ([`MonotonicClock`]) and the TAI
//! timeline built on top of it ([`TaiClock`]).
//!
//! The precise `CLOCK_TAI` reading needs a single libc call, which this crate — no
//! dependencies, no `unsafe` — cannot make itself: the app's platform layer reads
//! `clock_gettime(CLOCK_TAI)` and feeds [`TaiClock::calibrate_now`] /
//! [`TaiClock::recalibrate_now`] periodically. Until then,
//! [`TaiClock::from_unix_plus_leap`] offers a UTC + leap-seconds fallback.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// TAI − UTC offset in seconds. Valid since 2017-01-01; update at the next leap second.
pub const TAI_UTC_OFFSET_S: i64 = 37;

/// Clock timeline a reading belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockDomain {
    /// Process-lifetime monotonic time (`CLOCK_MONOTONIC`-anchored).
    Monotonic,
    /// Temps Atomique International, the PTP timescale (`CLOCK_TAI` when steered by
    /// `phc2sys`, or UTC + leap seconds as a fallback).
    Tai,
}

/// Synchronisation state of a clock source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyncStatus {
    /// Free-running, no reference.
    Free,
    /// Following a reference, not yet trustworthy for exact mapping.
    Tracking,
    /// Locked to the domain reference.
    Locked,
}

/// A physical clock readable in nanoseconds.
pub trait MediaClock {
    /// Current reading in nanoseconds on the clock's domain.
    fn now_ns(&self) -> i64;

    /// Which timeline [`now_ns`](MediaClock::now_ns) reads.
    fn domain(&self) -> ClockDomain;

    /// How well the clock is synchronised to its domain's reference.
    fn sync_status(&self) -> SyncStatus {
        SyncStatus::Free
    }
}

/// Monotonic clock anchored at construction (`CLOCK_MONOTONIC`-based via [`Instant`]).
///
/// Readings are only comparable within one process lifetime. Pair them with a device's
/// sample positions (e.g. PipeWire `pw_time` ticks) as the `(sample, time_ns)` input to
/// [`crate::TimestampTracker`].
#[derive(Debug, Clone)]
pub struct MonotonicClock {
    start: Instant,
    offset_ns: i64,
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock {
    /// Anchor at the current instant, reading 0 ns.
    pub fn new() -> Self {
        Self { start: Instant::now(), offset_ns: 0 }
    }

    /// Anchor at the current instant, reading `offset_ns` ns (e.g. to align several
    /// clocks on one timeline).
    pub fn with_offset_ns(offset_ns: i64) -> Self {
        Self { start: Instant::now(), offset_ns }
    }

    /// Nanoseconds elapsed since anchoring, plus the initial offset. Monotonic and
    /// saturating; never panics.
    pub fn now_ns(&self) -> i64 {
        let elapsed = self.start.elapsed().as_nanos();
        self.offset_ns.saturating_add(elapsed.min(i64::MAX as u128) as i64)
    }
}

impl MediaClock for MonotonicClock {
    fn now_ns(&self) -> i64 {
        self.now_ns()
    }

    fn domain(&self) -> ClockDomain {
        ClockDomain::Monotonic
    }

    fn sync_status(&self) -> SyncStatus {
        SyncStatus::Tracking
    }
}

/// How the TAI offset of a [`TaiClock`] was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaiSource {
    /// UTC (`CLOCK_REALTIME`) plus [`TAI_UTC_OFFSET_S`]; fallback until a real
    /// `CLOCK_TAI` reading is supplied. Accurate only to system-clock correctness.
    UnixPlusLeap,
    /// Calibrated against an externally read `CLOCK_TAI` (or an equivalent PTP reading).
    Calibrated,
}

/// TAI timeline on top of a monotonic anchor: `tai = mono + offset`.
///
/// Build one with [`from_unix_plus_leap`](TaiClock::from_unix_plus_leap) at startup and
/// recalibrate from the platform's `CLOCK_TAI` reading once `ptp4l`/`phc2sys` are up;
/// after that [`now_tai_ns`](TaiClock::now_tai_ns) is PTP time in-process, with the
/// monotonic base keeping reads cheap, monotonic and jitter-free between calibrations.
#[derive(Debug, Clone)]
pub struct TaiClock {
    mono: MonotonicClock,
    tai_minus_mono_ns: i64,
    source: TaiSource,
}

impl TaiClock {
    /// Fallback construction: TAI estimated as UTC + leap seconds from the system clock.
    pub fn from_unix_plus_leap() -> Self {
        let mono = MonotonicClock::new();
        let unix_ns = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos().min(i64::MAX as u128) as i64).unwrap_or(0);
        let tai_ns = unix_ns.saturating_add(TAI_UTC_OFFSET_S.saturating_mul(1_000_000_000));
        Self { tai_minus_mono_ns: tai_ns.saturating_sub(mono.now_ns()), mono, source: TaiSource::UnixPlusLeap }
    }

    /// Construct calibrated with a `CLOCK_TAI` reading taken *now* (read it immediately
    /// before calling; the smaller the gap, the better the calibration).
    pub fn calibrate_now(tai_ns: i64) -> Self {
        let mono = MonotonicClock::new();
        Self { tai_minus_mono_ns: tai_ns.saturating_sub(mono.now_ns()), mono, source: TaiSource::Calibrated }
    }

    /// Re-calibrate with a fresh `CLOCK_TAI` reading taken *now*. Feed periodically so
    /// the offset tracks `phc2sys` steering.
    pub fn recalibrate_now(&mut self, tai_ns: i64) {
        self.tai_minus_mono_ns = tai_ns.saturating_sub(self.mono.now_ns());
        self.source = TaiSource::Calibrated;
    }

    /// Construct with an explicit `tai − mono` offset (synthetic timelines, tests).
    pub fn with_tai_minus_mono_ns(tai_minus_mono_ns: i64) -> Self {
        Self { mono: MonotonicClock::new(), tai_minus_mono_ns, source: TaiSource::Calibrated }
    }

    /// The current monotonic-timeline reading (same timeline as
    /// [`tai_from_mono_ns`](TaiClock::tai_from_mono_ns)).
    pub fn now_mono_ns(&self) -> i64 {
        self.mono.now_ns()
    }

    /// The current TAI reading.
    pub fn now_tai_ns(&self) -> i64 {
        self.mono.now_ns().saturating_add(self.tai_minus_mono_ns)
    }

    /// Map a monotonic-timeline nanosecond value to TAI; `None` on overflow.
    pub fn tai_from_mono_ns(&self, mono_ns: i64) -> Option<i64> {
        mono_ns.checked_add(self.tai_minus_mono_ns)
    }

    /// Current `tai − mono` offset in nanoseconds.
    pub fn offset_ns(&self) -> i64 {
        self.tai_minus_mono_ns
    }

    /// How the offset was obtained.
    pub fn source(&self) -> TaiSource {
        self.source
    }
}

impl MediaClock for TaiClock {
    fn now_ns(&self) -> i64 {
        self.now_tai_ns()
    }

    fn domain(&self) -> ClockDomain {
        ClockDomain::Tai
    }

    fn sync_status(&self) -> SyncStatus {
        match self.source {
            TaiSource::Calibrated => SyncStatus::Locked,
            TaiSource::UnixPlusLeap => SyncStatus::Tracking,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_clock_advances_and_saturates() {
        let clock = MonotonicClock::new();
        let a = clock.now_ns();
        std::thread::sleep(std::time::Duration::from_micros(200));
        let b = clock.now_ns();
        assert!(b > a, "monotonic clock must advance: {a} -> {b}");
        // Hostile offset: never panics, saturates.
        let pinned = MonotonicClock::with_offset_ns(i64::MAX - 10);
        let _ = pinned.now_ns();
    }

    #[test]
    fn tai_clock_offset_math() {
        let tai = TaiClock::with_tai_minus_mono_ns(37_000_000_000);
        assert_eq!(tai.offset_ns(), 37_000_000_000);
        assert_eq!(tai.tai_from_mono_ns(1_000).unwrap(), 37_000_001_000);
        assert!((tai.now_tai_ns() - tai.now_mono_ns() - 37_000_000_000).abs() < 1_000);
        assert_eq!(tai.tai_from_mono_ns(i64::MAX), None);
        assert_eq!(tai.sync_status(), SyncStatus::Locked);
        assert_eq!(tai.domain(), ClockDomain::Tai);
    }

    #[test]
    fn tai_clock_calibration_moves_the_offset() {
        let mut tai = TaiClock::from_unix_plus_leap();
        assert_eq!(tai.source(), TaiSource::UnixPlusLeap);
        assert_eq!(tai.sync_status(), SyncStatus::Tracking);
        let before = tai.now_tai_ns();
        tai.recalibrate_now(before + 1_000_000);
        assert_eq!(tai.source(), TaiSource::Calibrated);
        assert!((tai.now_tai_ns() - (before + 1_000_000)).abs() < 1_000_000);
    }
}
