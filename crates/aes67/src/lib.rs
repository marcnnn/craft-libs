//! AES67 / SMPTE ST 2110-30 essentials for the Craft Apps.
//!
//! Pure, dependency-free logic shared by the host engine and tooling
//! (media clock is 48 kHz throughout, per the 48-frame block grid in HANDOFF §2E.5):
//!
//! - [`ptime`]: the five AES67 packet times and the per-tier policy (1 ms interop,
//!   125 µs FPGA tier, 250 µs optional host fast tier)
//! - [`rtp`]: RTP timestamp ↔ sample-index mapping with 2³² wraparound handling
//!   (ST 2110-30: RTP timestamp = PTP samples since epoch, mod 2³²)
//! - [`schedule`]: media-clock-paced transmit deadlines and block ↔ packet slicing
//! - [`jitter`]: an allocation-free de-jitter buffer with loss/reorder/late statistics
//!
//! All arithmetic is exact integer math; nothing allocates in steady state.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod jitter;
pub mod ptime;
pub mod rtp;
pub mod schedule;
