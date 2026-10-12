//! Media-clock-paced transmit scheduling and block ↔ packet slicing.
//!
//! Senders pace packets on the media clock, not the local clock: packet `k` is
//! transmitted when the media clock reaches its first sample (TX jitter < 1 packet
//! recommended, < 17 required by AES67 — HANDOFF §2E.5). Block slicing maps the
//! engine's 48-frame blocks onto packets for any packet time, including the 4 ms
//! case where one packet spans four blocks.

use crate::ptime::{BLOCK_FRAMES, PacketTime};
use craft_audio_clock::LockedClock;

/// Transmit deadline (media/PTP ns) of packet `packet_index`: the exact time its first
/// sample exists on the media clock. `None` if the sample index overflows the clock.
#[must_use]
pub fn tx_deadline_ns(clock: &LockedClock, ptime: PacketTime, packet_index: i64) -> Option<i64> {
    let sample = packet_index.checked_mul(i64::from(ptime.frames()))?;
    clock.time_ns_of(sample)
}

/// Overlap of one packet with a block: where the packet's bytes land in the block and
/// vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketSlice {
    /// Absolute index of the packet.
    pub packet_index: i64,
    /// Offset of the overlap inside the packet, in frames.
    pub packet_offset: u32,
    /// Offset of the overlap inside the block, in frames.
    pub block_offset: u32,
    /// Overlap length in frames.
    pub frames: u32,
}

/// Fill `out` with the packets overlapping the 48-frame block `block_index`, cleared
/// first (reused allocation). Packet times ≤ 1 ms yield 1–8 slices fully inside the
/// block; 4 ms yields one slice covering this block's part of a 4-block packet.
pub fn packets_overlapping_into(block_index: i64, ptime: PacketTime, out: &mut Vec<PacketSlice>) {
    out.clear();
    let Some(block_start) = block_index.checked_mul(i64::from(BLOCK_FRAMES)) else { return };
    let Some(block_end) = block_start.checked_add(i64::from(BLOCK_FRAMES)) else { return };
    let f = i64::from(ptime.frames());
    let first = block_start.div_euclid(f);
    let last = (block_end - 1).div_euclid(f);
    let mut k = first;
    while k <= last {
        let Some(pkt_start) = k.checked_mul(f) else { break };
        let overlap_start = pkt_start.max(block_start);
        let overlap_end = (pkt_start + f).min(block_end);
        if overlap_end > overlap_start {
            out.push(PacketSlice {
                packet_index: k,
                packet_offset: (overlap_start - pkt_start) as u32,
                block_offset: (overlap_start - block_start) as u32,
                frames: (overlap_end - overlap_start) as u32,
            });
        }
        match k.checked_add(1) {
            Some(next) => k = next,
            None => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptime::PacketTime::*;

    fn slices(block: i64, p: PacketTime) -> Vec<PacketSlice> {
        let mut v = Vec::new();
        packets_overlapping_into(block, p, &mut v);
        v
    }

    #[test]
    fn one_ms_is_one_packet_per_block() {
        let s = slices(5, P1ms);
        assert_eq!(s, vec![PacketSlice { packet_index: 5, packet_offset: 0, block_offset: 0, frames: 48 }]);
    }

    #[test]
    fn sub_block_times_tile_the_block() {
        let s = slices(1, P125us);
        assert_eq!(s.len(), 8);
        assert_eq!(s[0], PacketSlice { packet_index: 8, packet_offset: 0, block_offset: 0, frames: 6 });
        assert_eq!(s[7], PacketSlice { packet_index: 15, packet_offset: 0, block_offset: 42, frames: 6 });
        let total: u32 = s.iter().map(|x| x.frames).sum();
        assert_eq!(total, 48);

        let s = slices(2, P250us);
        assert_eq!(s.len(), 4);
        assert_eq!(s[0].packet_index, 8);
        assert_eq!(s[3].block_offset, 36);

        let s = slices(3, P333us);
        assert_eq!(s.len(), 3);
        assert_eq!(s.iter().map(|x| x.frames).sum::<u32>(), 48);
    }

    #[test]
    fn four_ms_spans_four_blocks() {
        // Block 0: first quarter of packet 0.
        assert_eq!(slices(0, P4ms), vec![PacketSlice { packet_index: 0, packet_offset: 0, block_offset: 0, frames: 48 }]);
        // Block 5: second quarter of packet 1 (samples 192..384).
        assert_eq!(slices(5, P4ms), vec![PacketSlice { packet_index: 1, packet_offset: 48, block_offset: 0, frames: 48 }]);
        // Block 7: last quarter of packet 1.
        assert_eq!(slices(7, P4ms), vec![PacketSlice { packet_index: 1, packet_offset: 144, block_offset: 0, frames: 48 }]);
    }

    #[test]
    fn tx_deadlines_follow_the_media_clock_exactly() {
        let clock = LockedClock::new(48_000, 1_000_000_000);
        // Packet 10 at 1 ms starts at sample 480 — 10 ms after the epoch.
        assert_eq!(tx_deadline_ns(&clock, P1ms, 10), Some(1_010_000_000));
        // 125 µs packets advance 125 µs each: packet 8 starts at sample 48 (+1 ms),
        // packet 16 at sample 96 (+2 ms).
        assert_eq!(tx_deadline_ns(&clock, P125us, 8), Some(1_001_000_000));
        assert_eq!(tx_deadline_ns(&clock, P125us, 16), Some(1_002_000_000));
        // 4 ms packets: packet 2 starts at sample 384 (8 ms).
        assert_eq!(tx_deadline_ns(&clock, P4ms, 2), Some(1_008_000_000));
        // Hostile index: reported, not wrapped.
        assert_eq!(tx_deadline_ns(&clock, P1ms, i64::MAX), None);
    }
}
