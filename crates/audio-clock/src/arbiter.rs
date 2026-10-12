//! Hysteresis arbiter: stable mode decisions from noisy [`ClockRelation`] observations.
//!
//! Borderline clocks flap; exact-lock promotion and resampler engagement must not. The
//! arbiter requires a sustained streak of a relation before switching modes
//! (see HANDOFF.md §2E/§2G):
//!
//! - [`SyncMode::Passthrough`] is the default and the safe neutral state.
//! - [`SyncMode::Locked`] once the measured deviation stays within tolerance long enough
//!   to trust the exact [`crate::LockedClock`] mapping (`promote_to_locked`).
//! - [`SyncMode::Correcting`] only on a sustained [`ClockRelation::NeedsCorrection`]:
//!   engage the last-resort [`crate::StreamResampler`] at the measured drift ratio.
//!
//! An absent tracker ([`ClockRelation::InsufficientData`]) resets drift streaks and, if it
//! persists, returns the chain to passthrough rather than acting on stale data.

use crate::ClockRelation;
use crate::LockedClock;
use crate::TimestampTracker;
use crate::clocks::TaiClock;

/// Engine mode decided by [`SyncArbiter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyncMode {
    /// 1:1 passthrough; positions from the fitted tracker.
    Passthrough,
    /// Device clock trusted as PTP-locked; positions from the exact `LockedClock`.
    Locked,
    /// Sustained drift beyond tolerance; last-resort resampler engaged.
    Correcting,
}

/// Hysteresis state machine over [`ClockRelation`] observations.
#[derive(Debug, Clone)]
pub struct SyncArbiter {
    mode: SyncMode,
    lock_after: u32,
    correct_after: u32,
    release_after: u32,
    lock_streak: u32,
    correct_streak: u32,
    release_streak: u32,
    absent_streak: u32,
}

impl SyncArbiter {
    /// Create with dwell counts: `lock_after` consecutive within-tolerance observations
    /// before promoting to [`SyncMode::Locked`], `correct_after` consecutive
    /// `NeedsCorrection` observations before engaging correction, `release_after`
    /// consecutive good (or absent) observations before releasing it. All clamped to ≥ 1.
    pub fn new(lock_after: u32, correct_after: u32, release_after: u32) -> Self {
        Self {
            mode: SyncMode::Passthrough,
            lock_after: lock_after.max(1),
            correct_after: correct_after.max(1),
            release_after: release_after.max(1),
            lock_streak: 0,
            correct_streak: 0,
            release_streak: 0,
            absent_streak: 0,
        }
    }

    /// Current mode without feeding an observation.
    pub fn mode(&self) -> SyncMode {
        self.mode
    }

    /// Feed one observation and return the (possibly changed) mode.
    pub fn update(&mut self, relation: ClockRelation) -> SyncMode {
        match relation {
            ClockRelation::WithinTolerance { .. } => {
                self.lock_streak = self.lock_streak.saturating_add(1);
                self.correct_streak = 0;
                self.release_streak = self.release_streak.saturating_add(1);
                self.absent_streak = 0;
            }
            ClockRelation::NeedsCorrection { .. } => {
                self.lock_streak = 0;
                self.correct_streak = self.correct_streak.saturating_add(1);
                self.release_streak = 0;
                self.absent_streak = 0;
            }
            ClockRelation::InsufficientData => {
                self.lock_streak = 0;
                self.correct_streak = 0;
                self.release_streak = 0;
                self.absent_streak = self.absent_streak.saturating_add(1);
            }
        }

        self.mode = match self.mode {
            SyncMode::Passthrough => {
                if self.lock_streak >= self.lock_after {
                    SyncMode::Locked
                } else if self.correct_streak >= self.correct_after {
                    SyncMode::Correcting
                } else {
                    SyncMode::Passthrough
                }
            }
            SyncMode::Locked => {
                if self.correct_streak >= self.correct_after {
                    SyncMode::Correcting
                } else if self.absent_streak >= self.release_after {
                    SyncMode::Passthrough
                } else {
                    SyncMode::Locked
                }
            }
            SyncMode::Correcting => {
                if self.release_streak >= self.release_after || self.absent_streak >= self.release_after {
                    SyncMode::Passthrough
                } else {
                    SyncMode::Correcting
                }
            }
        };
        self.mode
    }
}

/// Promote a sustained tracker fit into an exact [`LockedClock`] on the TAI timeline.
///
/// Derives the epoch (PTP/TAI time of device sample 0) from the tracker's monotonic fit
/// plus the [`TaiClock`] offset, and returns a `LockedClock` rebased to it. Call only
/// once the [`SyncArbiter`] reports [`SyncMode::Locked`]; the epoch is a point estimate,
/// exact (not estimated) from then on. `None` while the tracker has no fit or the
/// conversion overflows.
pub fn promote_to_locked(tracker: &TimestampTracker, tai: &TaiClock, rate_hz: u32) -> Option<LockedClock> {
    let mono_at_zero = tracker.time_ns_at(0)?;
    let tai_at_zero = tai.tai_from_mono_ns(mono_at_zero)?;
    let mut clock = LockedClock::new(rate_hz, 0);
    clock.rebase(0, tai_at_zero);
    Some(clock)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn within() -> ClockRelation {
        ClockRelation::WithinTolerance { deviation_ppm: 0.5 }
    }

    fn needs() -> ClockRelation {
        ClockRelation::NeedsCorrection { deviation_ppm: 25.0 }
    }

    #[test]
    fn promotes_to_locked_only_after_sustained_tolerance() {
        let mut arbiter = SyncArbiter::new(5, 3, 4);
        assert_eq!(arbiter.mode(), SyncMode::Passthrough);
        for _ in 0..4 {
            assert_eq!(arbiter.update(within()), SyncMode::Passthrough);
        }
        assert_eq!(arbiter.update(within()), SyncMode::Locked);
        // Stays locked while good; a single bad observation does not demote.
        assert_eq!(arbiter.update(needs()), SyncMode::Locked);
        assert_eq!(arbiter.update(within()), SyncMode::Locked);
    }

    #[test]
    fn flapping_relation_never_switches_modes() {
        let mut arbiter = SyncArbiter::new(5, 3, 4);
        for _ in 0..50 {
            assert_eq!(arbiter.update(within()), SyncMode::Passthrough);
            assert_eq!(arbiter.update(needs()), SyncMode::Passthrough);
        }
    }

    #[test]
    fn engages_and_releases_correction_with_hysteresis() {
        let mut arbiter = SyncArbiter::new(5, 3, 4);
        for _ in 0..3 {
            arbiter.update(needs());
        }
        assert_eq!(arbiter.mode(), SyncMode::Correcting);
        for _ in 0..3 {
            assert_eq!(arbiter.update(within()), SyncMode::Correcting);
        }
        assert_eq!(arbiter.update(within()), SyncMode::Passthrough);
    }

    #[test]
    fn locked_demotes_to_correcting_on_sustained_drift() {
        let mut arbiter = SyncArbiter::new(2, 2, 2);
        arbiter.update(within());
        arbiter.update(within());
        assert_eq!(arbiter.mode(), SyncMode::Locked);
        arbiter.update(needs());
        assert_eq!(arbiter.mode(), SyncMode::Locked);
        assert_eq!(arbiter.update(needs()), SyncMode::Correcting);
    }

    #[test]
    fn absent_tracker_returns_chain_to_passthrough() {
        let mut arbiter = SyncArbiter::new(2, 2, 3);
        arbiter.update(within());
        arbiter.update(within());
        assert_eq!(arbiter.mode(), SyncMode::Locked);
        arbiter.update(ClockRelation::InsufficientData);
        arbiter.update(ClockRelation::InsufficientData);
        assert_eq!(arbiter.mode(), SyncMode::Locked); // brief absence tolerated
        assert_eq!(arbiter.update(ClockRelation::InsufficientData), SyncMode::Passthrough);
    }

    #[test]
    fn promotion_derives_the_exact_epoch() {
        // Device at exactly 48 kHz on the mono timeline, mono origin at 5 s.
        let mut tracker = TimestampTracker::new(48000.0);
        for k in 0i64..100 {
            let sample = k * 4800;
            let t_mono = 5_000_000_000i64 + k * 100_000_000;
            assert!(tracker.observe(sample, t_mono));
        }
        let tai = TaiClock::with_tai_minus_mono_ns(37_000_000_000);
        let clock = promote_to_locked(&tracker, &tai, 48000).unwrap();
        // Sample 0 sits at mono 5 s -> TAI 42 s; one second of samples is exactly a second.
        assert_eq!(clock.time_ns_of(0), Some(42_000_000_000));
        assert_eq!(clock.time_ns_of(48000), Some(43_000_000_000));
        // None before a fit exists.
        let empty = TimestampTracker::new(48000.0);
        assert!(promote_to_locked(&empty, &tai, 48000).is_none());
    }
}
