//! AES67 packet times on the 48-frame block grid.
//!
//! AES67 defines five packet times; at 48 kHz every one of them divides the 48-frame
//! engine block exactly (or is a whole multiple of it), so one block is always a whole
//! number of packets and vice versa. The per-tier policy (HANDOFF §2E.5):
//! 1 ms for interop (required of every device, the only ptime Dante speaks in AES67
//! mode), 125 µs on the FPGA tier, 250 µs for ours-to-ours host fast paths, 4 ms only
//! for non-realtime side audio.

/// RTP payload size must not exceed this to avoid IP fragmentation on a 1500-byte MTU
/// (1500 − 20 IP − 8 UDP − 12 RTP).
pub const RTP_PAYLOAD_LIMIT: u32 = 1460;

/// Media sample rate this crate is pinned to (48 kHz).
pub const SAMPLE_RATE: u32 = 48_000;

/// Engine base block in frames (1 ms at 48 kHz).
pub const BLOCK_FRAMES: u32 = 48;

/// The five AES67 packet times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PacketTime {
    /// 125 µs — 6 frames. FPGA tier (ours-to-ours); AVB Class A interval.
    P125us,
    /// 250 µs — 12 frames. Optional host fast tier when both ends are ours or
    /// RAVENNA-class.
    P250us,
    /// 333⅓ µs — 16 frames. "Efficient low-latency" per the standard.
    P333us,
    /// 1 ms — 48 frames. **Default**: required of every AES67 device; the interop tier.
    #[default]
    P1ms,
    /// 4 ms — 192 frames. WAN / weak QoS / non-realtime side audio only.
    P4ms,
}

impl PacketTime {
    /// All defined packet times.
    pub const ALL: [PacketTime; 5] = [PacketTime::P125us, PacketTime::P250us, PacketTime::P333us, PacketTime::P1ms, PacketTime::P4ms];

    /// Packet duration in nanoseconds.
    #[must_use]
    pub const fn nanoseconds(self) -> u64 {
        match self {
            PacketTime::P125us => 125_000,
            PacketTime::P250us => 250_000,
            PacketTime::P333us => 333_333,
            PacketTime::P1ms => 1_000_000,
            PacketTime::P4ms => 4_000_000,
        }
    }

    /// Frames per packet at 48 kHz.
    #[must_use]
    pub const fn frames(self) -> u32 {
        match self {
            PacketTime::P125us => 6,
            PacketTime::P250us => 12,
            PacketTime::P333us => 16,
            PacketTime::P1ms => 48,
            PacketTime::P4ms => 192,
        }
    }

    /// Relationship to the 48-frame block: `(packets_per_block, blocks_per_packet)`;
    /// exactly one of the two is 1 for every defined packet time.
    #[must_use]
    pub const fn block_ratio(self) -> (u32, u32) {
        match self {
            PacketTime::P125us => (8, 1),
            PacketTime::P250us => (4, 1),
            PacketTime::P333us => (3, 1),
            PacketTime::P1ms => (1, 1),
            PacketTime::P4ms => (1, 4),
        }
    }

    /// Maximum channels per packet for a sample width in bytes, so the RTP payload stays
    /// within [`RTP_PAYLOAD_LIMIT`] (no IP fragmentation).
    #[must_use]
    pub fn max_channels(self, bytes_per_sample: u32) -> u32 {
        let frame_bytes = self.frames().saturating_mul(bytes_per_sample.max(1));
        RTP_PAYLOAD_LIMIT / frame_bytes
    }

    /// Minimum receiver de-jitter buffer in nanoseconds (AES67 requires ≥3 packet times).
    #[must_use]
    pub const fn min_receiver_buffer_ns(self) -> u64 {
        self.nanoseconds() * 3
    }

    /// Absolute sample index where packet `packet_index` begins.
    #[must_use]
    pub fn packet_start_sample(self, packet_index: i64) -> i64 {
        packet_index.saturating_mul(self.frames() as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_packet_times_align_with_the_48_frame_grid() {
        for p in PacketTime::ALL {
            let frames = p.frames();
            assert!(frames.is_multiple_of(BLOCK_FRAMES) || BLOCK_FRAMES.is_multiple_of(frames), "{p:?} breaks the grid");
            let (ppb, bpp) = p.block_ratio();
            assert_eq!(ppb * frames, bpp * BLOCK_FRAMES, "{p:?} ratio mismatch");
            // Duration and frame count agree to within the ⅓ ms rounding.
            let ns_from_frames = frames as f64 * 1.0e9 / SAMPLE_RATE as f64;
            assert!((ns_from_frames - p.nanoseconds() as f64).abs() <= 1.0, "{p:?} ns/frames mismatch");
        }
    }

    #[test]
    fn channel_limits_match_the_standard_table() {
        // 48 kHz / 24-bit: the AES67 table lists 80 for 125 µs (its own conservative
        // rounding); the exact 1460-byte payload bound is 81. Every other row matches
        // the published table exactly. Same for the 16-bit 125 µs row (121 exact,
        // 120 published).
        assert_eq!(PacketTime::P125us.max_channels(3), 81);
        assert_eq!(PacketTime::P250us.max_channels(3), 40);
        assert_eq!(PacketTime::P333us.max_channels(3), 30);
        assert_eq!(PacketTime::P1ms.max_channels(3), 10);
        assert_eq!(PacketTime::P4ms.max_channels(3), 2);
        // 48 kHz / 16-bit row.
        assert_eq!(PacketTime::P1ms.max_channels(2), 15);
        assert_eq!(PacketTime::P125us.max_channels(2), 121);
    }

    #[test]
    fn interop_defaults() {
        assert_eq!(PacketTime::default(), PacketTime::P1ms);
        assert_eq!(PacketTime::P1ms.nanoseconds(), 1_000_000);
        assert_eq!(PacketTime::P1ms.min_receiver_buffer_ns(), 3_000_000);
        assert_eq!(PacketTime::P125us.min_receiver_buffer_ns(), 375_000);
        assert_eq!(PacketTime::P1ms.packet_start_sample(7), 336);
        assert_eq!(PacketTime::P125us.packet_start_sample(8), 48);
    }
}
