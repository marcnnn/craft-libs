//! Reader for the `AES67PTPAgent` shared clock mapping (macOS AES67 driver, HANDOFF §2F).
//!
//! The driver's PTP agent publishes grandmaster time through a fixed-size, sequence-locked
//! mapping (default `/tmp/aes67_ptp_clock`). The layout below mirrors the driver's
//! `SharedPTPClock.h` — it is append-only there, and every field here is read defensively:
//! all values are untrusted input (a malformed or stale mapping degrades to an error or
//! "unaligned", never to a wrong time and never to a panic).

use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

/// Mapping magic: `"AES7"`.
pub const AGENT_MAGIC: u32 = 0x4145_5337;
/// Mapping layout version this reader understands.
pub const AGENT_VERSION: u32 = 1;
/// Default path the agent publishes to.
pub const AGENT_MAPPING_PATH: &str = "/tmp/aes67_ptp_clock";
/// A sample older than this is treated as "no clock" (matches the driver's own policy).
pub const AGENT_STALE_AFTER_NS: u64 = 5_000_000_000;
/// Size of the published mapping in bytes.
pub const MAPPING_BYTES: usize = 48;

/// A consistent snapshot of the agent's published clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PtpAgentSample {
    locked: bool,
    master_time_ns: u64,
    local_monotonic_ns: u64,
    frequency_ppb: f64,
    update_count: u64,
}

/// Failures reading or validating the mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtpAgentError {
    /// The mapping file could not be opened (agent not running).
    Open,
    /// An I/O error while reading.
    Read,
    /// Fewer bytes than the fixed mapping size.
    Short,
    /// Wrong magic: not an agent mapping.
    BadMagic,
    /// Layout version newer than this reader.
    UnsupportedVersion,
    /// The sequence lock never settled (writer active or corrupt).
    Torn,
    /// Values a sane agent can never publish (non-finite or absurd rate error).
    Implausible,
}

/// Parse one raw mapping image. Returns the sequence number and the decoded sample.
/// Byte order is little-endian (x86-64 / arm64 macOS).
pub fn parse_mapping(buf: &[u8]) -> Result<(u32, PtpAgentSample), PtpAgentError> {
    if buf.len() < MAPPING_BYTES {
        return Err(PtpAgentError::Short);
    }
    let u32le = |off: usize| u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    let u64le = |off: usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&buf[off..off + 8]);
        u64::from_le_bytes(b)
    };
    if u32le(0) != AGENT_MAGIC {
        return Err(PtpAgentError::BadMagic);
    }
    let version = u32le(4);
    if version != AGENT_VERSION {
        return Err(PtpAgentError::UnsupportedVersion);
    }
    let sequence = u32le(8);
    let locked = u32le(12) != 0;
    let master_time_ns = u64le(16);
    let local_monotonic_ns = u64le(24);
    let frequency_ppb = f64::from_bits(u64le(32));
    let update_count = u64le(40);
    if !frequency_ppb.is_finite() || frequency_ppb.abs() > 1_000_000.0 {
        return Err(PtpAgentError::Implausible);
    }
    Ok((sequence, PtpAgentSample { locked, master_time_ns, local_monotonic_ns, frequency_ppb, update_count }))
}

/// Read a consistent snapshot from the mapping file, retrying the seqlock a bounded
/// number of times. `Torn` means the writer never settled within the retries.
pub fn read_mapping(path: &Path) -> Result<PtpAgentSample, PtpAgentError> {
    let mut file = File::open(path).map_err(|_| PtpAgentError::Open)?;
    let mut buf = [0u8; MAPPING_BYTES];
    for _ in 0..4 {
        file.rewind().map_err(|_| PtpAgentError::Read)?;
        file.read_exact(&mut buf).map_err(|_| PtpAgentError::Read)?;
        let (seq, sample) = parse_mapping(&buf)?;
        if seq % 2 == 1 {
            continue; // write in progress
        }
        // Confirm the sequence has not changed: the snapshot is then consistent.
        file.rewind().map_err(|_| PtpAgentError::Read)?;
        let mut confirm = [0u8; MAPPING_BYTES];
        file.read_exact(&mut confirm).map_err(|_| PtpAgentError::Read)?;
        let (seq2, _) = parse_mapping(&confirm)?;
        if seq2 == seq {
            return Ok(sample);
        }
    }
    Err(PtpAgentError::Torn)
}

impl PtpAgentSample {
    /// Whether the agent's PTP slave has locked to a grandmaster.
    pub fn locked(&self) -> bool {
        self.locked
    }

    /// Grandmaster time (ns) at the moment of the sample.
    pub fn master_time_ns(&self) -> u64 {
        self.master_time_ns
    }

    /// Local `CLOCK_MONOTONIC` (ns) at which [`master_time_ns`](PtpAgentSample::master_time_ns)
    /// was measured.
    pub fn local_monotonic_ns(&self) -> u64 {
        self.local_monotonic_ns
    }

    /// Local clock rate error against the master, in parts per billion.
    pub fn frequency_ppb(&self) -> f64 {
        self.frequency_ppb
    }

    /// Agent update counter (diagnostics; increments with every publish).
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Grandmaster time (ns) projected to `now_monotonic_ns`, using the measured rate
    /// error rather than assuming the local clock ticks at the master's rate.
    /// Saturating; never panics.
    pub fn master_time_at(&self, now_monotonic_ns: u64) -> u64 {
        if now_monotonic_ns <= self.local_monotonic_ns {
            return self.master_time_ns;
        }
        let elapsed_ns = (now_monotonic_ns - self.local_monotonic_ns) as f64;
        let scaled = elapsed_ns * (1.0 + self.frequency_ppb * 1.0e-9);
        self.master_time_ns.saturating_add(scaled.round().max(0.0) as u64)
    }

    /// `tai − mono` offset for calibrating a
    /// [`TaiClock`](crate::clocks::TaiClock) at `now_monotonic_ns`.
    pub fn tai_minus_mono_ns(&self, now_monotonic_ns: u64) -> i64 {
        self.master_time_at(now_monotonic_ns).saturating_sub(now_monotonic_ns) as i64
    }

    /// Whether this sample may be used as a clock reference at `now_monotonic_ns`:
    /// locked, and not older than [`AGENT_STALE_AFTER_NS`].
    pub fn usable_at(&self, now_monotonic_ns: u64) -> bool {
        self.locked && now_monotonic_ns.saturating_sub(self.local_monotonic_ns) <= AGENT_STALE_AFTER_NS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn encode(sequence: u32, sample: &PtpAgentSample) -> [u8; MAPPING_BYTES] {
        let mut b = [0u8; MAPPING_BYTES];
        b[0..4].copy_from_slice(&AGENT_MAGIC.to_le_bytes());
        b[4..8].copy_from_slice(&AGENT_VERSION.to_le_bytes());
        b[8..12].copy_from_slice(&sequence.to_le_bytes());
        b[12..16].copy_from_slice(&u32::from(sample.locked).to_le_bytes());
        b[16..24].copy_from_slice(&sample.master_time_ns.to_le_bytes());
        b[24..32].copy_from_slice(&sample.local_monotonic_ns.to_le_bytes());
        b[32..40].copy_from_slice(&sample.frequency_ppb.to_bits().to_le_bytes());
        b[40..48].copy_from_slice(&sample.update_count.to_le_bytes());
        b
    }

    fn sample() -> PtpAgentSample {
        PtpAgentSample { locked: true, master_time_ns: 1_000_000_000, local_monotonic_ns: 900_000_000, frequency_ppb: 0.0, update_count: 7 }
    }

    #[test]
    fn parse_roundtrip() {
        let buf = encode(2, &sample());
        let (seq, got) = parse_mapping(&buf).unwrap();
        assert_eq!(seq, 2);
        assert_eq!(got, sample());
        assert!(got.locked());
        assert_eq!(got.master_time_ns(), 1_000_000_000);
        assert_eq!(got.update_count(), 7);
    }

    #[test]
    fn parse_rejects_hostile_input() {
        assert_eq!(parse_mapping(&[0u8; 16]), Err(PtpAgentError::Short));
        let mut buf = encode(2, &sample());
        buf[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        assert_eq!(parse_mapping(&buf), Err(PtpAgentError::BadMagic));
        let mut buf = encode(2, &sample());
        buf[4..8].copy_from_slice(&9u32.to_le_bytes());
        assert_eq!(parse_mapping(&buf), Err(PtpAgentError::UnsupportedVersion));
        let mut s = sample();
        s.frequency_ppb = f64::NAN;
        assert_eq!(parse_mapping(&encode(2, &s)), Err(PtpAgentError::Implausible));
        let mut s = sample();
        s.frequency_ppb = 9.0e9;
        assert_eq!(parse_mapping(&encode(2, &s)), Err(PtpAgentError::Implausible));
    }

    #[test]
    fn projection_and_staleness() {
        let s = sample();
        // 100 ms after the sample point: master advanced by exactly 100 ms at 0 ppb.
        assert_eq!(s.master_time_at(1_000_000_000), 1_100_000_000);
        // At +1000 ppb the same interval gains 100 ns.
        let fast = PtpAgentSample { frequency_ppb: 1000.0, ..sample() };
        assert_eq!(fast.master_time_at(1_000_000_000), 1_100_000_100);
        // Before the sample point: the measurement itself.
        assert_eq!(s.master_time_at(0), 1_000_000_000);
        // Offset for TaiClock calibration.
        assert_eq!(s.tai_minus_mono_ns(1_000_000_000), 100_000_000);
        // Usable while fresh and locked; stale or unlocked is not.
        assert!(s.usable_at(1_000_000_000));
        assert!(!s.usable_at(s.local_monotonic_ns() + AGENT_STALE_AFTER_NS + 1));
        let unlocked = PtpAgentSample { locked: false, ..sample() };
        assert!(!unlocked.usable_at(1_000_000_000));
    }

    #[test]
    fn read_mapping_from_file() {
        let path = std::env::temp_dir().join(format!("craft_ptp_agent_test_{}", std::process::id()));
        // Stable mapping reads cleanly.
        std::fs::write(&path, encode(4, &sample())).unwrap();
        assert_eq!(read_mapping(&path).unwrap(), sample());
        // Odd sequence (write in progress) that never settles: Torn, not a hang.
        std::fs::write(&path, encode(3, &sample())).unwrap();
        assert_eq!(read_mapping(&path), Err(PtpAgentError::Torn));
        // Missing file: Open.
        let missing = std::env::temp_dir().join("craft_ptp_agent_missing_9f");
        assert_eq!(read_mapping(&missing), Err(PtpAgentError::Open));
        std::fs::remove_file(&path).ok();

        // Read back through the TaiClock calibration path end to end.
        let mut w = std::fs::File::create(&path).unwrap();
        w.write_all(&encode(6, &sample())).unwrap();
        drop(w);
        let s = read_mapping(&path).unwrap();
        let tai = crate::clocks::TaiClock::with_tai_minus_mono_ns(s.tai_minus_mono_ns(1_000_000_000));
        assert!((tai.now_tai_ns() - tai.now_mono_ns() - 100_000_000).abs() < 1_000);
        std::fs::remove_file(&path).ok();
    }
}
