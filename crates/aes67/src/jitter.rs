//! Allocation-free de-jitter buffer for fixed-packet-time AES67 receive flows.
//!
//! Packets enter with their absolute (unwrapped) packet index as soon as they arrive;
//! the playout side pops one packet per packet time, paced by the media clock. The
//! buffer implements the AES67 receiver requirement (≥ 3 packet times of buffering)
//! as a configurable priming depth, and accounts every abnormal event: losses, late
//! packets, reordering, duplicates, and capacity overflows.
//!
//! Storage is preallocated at construction: pushes copy into reused slots and pops
//! copy into the caller's buffer — no allocation in steady state.

use crate::rtp;

/// Marker for a free slot.
const EMPTY_SLOT: i64 = i64::MIN;

/// Lifetime counters of everything abnormal (or normal) the buffer has seen.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JitterStats {
    /// Packets accepted into the buffer.
    pub received: u64,
    /// Packets handed to playout.
    pub delivered: u64,
    /// Packets that never arrived by their playout time (or were evicted).
    pub lost: u64,
    /// Packets arriving behind the playout cursor (discarded).
    pub late: u64,
    /// Packets arriving out of order but still ahead of the cursor.
    pub reordered: u64,
    /// Repeated packet indices.
    pub duplicates: u64,
    /// Packets too far ahead of the cursor for this capacity (caller should reset).
    pub overflow: u64,
    /// Payloads larger than the configured maximum.
    pub oversize: u64,
}

/// Result of pushing one packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    /// Stored; will be available to playout.
    Accepted,
    /// Same index pushed twice; the first copy was kept.
    Duplicate,
    /// Behind the playout cursor; discarded and counted.
    Late,
    /// Further ahead than `capacity` packets; not stored (caller should reset).
    Overflow,
    /// Payload exceeds the configured maximum size; not stored.
    Oversize,
}

/// Result of popping one packet for playout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopOutcome {
    /// A payload was copied out.
    Payload,
    /// The packet's playout time arrived without the packet; counted as lost.
    Missing,
    /// The buffer has not reached its priming depth yet; nothing consumed.
    Priming,
}

/// De-jitter buffer for one receive flow.
#[derive(Debug, Clone)]
pub struct DejitterBuffer {
    capacity: usize,
    depth: usize,
    frames_per_packet: u32,
    max_payload: usize,
    payloads: Vec<Vec<u8>>,
    slot_index: Vec<i64>,
    next_pop: i64,
    started: bool,
    primed: bool,
    newest: i64,
    stats: JitterStats,
}

impl DejitterBuffer {
    /// Create a buffer for packets of `frames_per_packet` frames at 48 kHz, payloads up
    /// to `max_payload_bytes`, `capacity_packets` slots (clamped to ≥ 8 and ≥ 2× depth)
    /// and a priming depth of `depth_packets` (clamped to [1, capacity/2]). AES67's
    /// minimum of 3 packet times corresponds to a depth of 3.
    pub fn new(frames_per_packet: u32, max_payload_bytes: usize, capacity_packets: usize, depth_packets: usize) -> Self {
        let depth_hint = depth_packets.max(1);
        let capacity = capacity_packets.max(8).max(depth_hint * 2);
        let depth = depth_hint.min(capacity / 2);
        Self {
            capacity,
            depth,
            frames_per_packet: frames_per_packet.max(1),
            max_payload: max_payload_bytes,
            payloads: (0..capacity).map(|_| Vec::with_capacity(max_payload_bytes)).collect(),
            slot_index: vec![EMPTY_SLOT; capacity],
            next_pop: 0,
            started: false,
            primed: false,
            newest: 0,
            stats: JitterStats::default(),
        }
    }

    /// Slot count.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Priming depth in packets.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Configured frames per packet.
    pub fn frames_per_packet(&self) -> u32 {
        self.frames_per_packet
    }

    /// Playout cursor: the next packet index pop will return.
    pub fn next_packet(&self) -> i64 {
        self.next_pop
    }

    /// Counters.
    pub fn stats(&self) -> JitterStats {
        self.stats
    }

    /// Whether the priming depth has been reached. Priming latches: once reached, the
    /// buffer keeps playing (starvation surfaces as `Missing` pops, not re-priming).
    pub fn is_primed(&self) -> bool {
        self.primed
    }

    /// Number of stored packets between the playout cursor and the newest arrival.
    pub fn fill_packets(&self) -> usize {
        if !self.started {
            return 0;
        }
        let span = (i128::from(self.newest) - i128::from(self.next_pop) + 1).clamp(0, self.capacity as i128) as i64;
        (0..span)
            .filter(|&k| {
                let idx = self.next_pop + k;
                self.slot_index[idx.rem_euclid(self.capacity as i64) as usize] == idx
            })
            .count()
    }

    /// Store one packet. `packet_index` is the absolute (unwrapped) packet index; use
    /// [`packet_index_from_rtp`](DejitterBuffer::packet_index_from_rtp) to derive it
    /// from an RTP timestamp.
    pub fn push(&mut self, packet_index: i64, payload: &[u8]) -> PushOutcome {
        if payload.len() > self.max_payload {
            self.stats.oversize += 1;
            return PushOutcome::Oversize;
        }
        if !self.started {
            self.started = true;
            self.next_pop = packet_index;
            self.newest = packet_index - 1;
        }
        if packet_index < self.next_pop {
            self.stats.late += 1;
            return PushOutcome::Late;
        }
        if packet_index.saturating_sub(self.next_pop) >= self.capacity as i64 {
            self.stats.overflow += 1;
            return PushOutcome::Overflow;
        }
        let pos = packet_index.rem_euclid(self.capacity as i64) as usize;
        if self.slot_index[pos] == packet_index {
            self.stats.duplicates += 1;
            return PushOutcome::Duplicate;
        }
        // A live packet already occupying this slot is evicted by the newcomer
        // (extreme reordering beyond capacity); it will surface as a Missing pop.
        if self.slot_index[pos] != EMPTY_SLOT {
            self.stats.lost += 1;
        }
        let slot = &mut self.payloads[pos];
        slot.clear();
        slot.extend_from_slice(payload);
        self.slot_index[pos] = packet_index;
        if packet_index > self.newest {
            self.newest = packet_index;
        } else {
            self.stats.reordered += 1;
        }
        if i128::from(self.newest) - i128::from(self.next_pop) + 1 >= self.depth as i128 {
            self.primed = true;
        }
        self.stats.received += 1;
        PushOutcome::Accepted
    }

    /// Pop the packet due for playout into `payload_out` (reused; no steady-state
    /// allocation). Advance one packet time per call, paced by the media clock.
    pub fn pop(&mut self, payload_out: &mut Vec<u8>) -> PopOutcome {
        if !self.primed {
            return PopOutcome::Priming;
        }
        let pos = self.next_pop.rem_euclid(self.capacity as i64) as usize;
        if self.slot_index[pos] == self.next_pop {
            payload_out.clear();
            payload_out.extend_from_slice(&self.payloads[pos]);
            self.slot_index[pos] = EMPTY_SLOT;
            self.next_pop += 1;
            self.stats.delivered += 1;
            PopOutcome::Payload
        } else {
            self.next_pop += 1;
            self.stats.lost += 1;
            PopOutcome::Missing
        }
    }

    /// Absolute packet index of an RTP timestamp, unwrapped against the newest arrival.
    #[must_use]
    pub fn packet_index_from_rtp(&self, timestamp: u32) -> i64 {
        let frames = i64::from(self.frames_per_packet);
        let previous_sample = self.newest.saturating_mul(frames);
        rtp::unwrap_rtp_timestamp(timestamp, previous_sample).div_euclid(frames)
    }

    /// Forget everything and return to the unprimed state.
    pub fn reset(&mut self) {
        self.slot_index.fill(EMPTY_SLOT);
        self.started = false;
        self.primed = false;
        self.next_pop = 0;
        self.newest = 0;
        self.stats = JitterStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtp::rtp_timestamp;

    fn payload_for(index: i64) -> Vec<u8> {
        format!("payload-{index}").into_bytes()
    }

    #[test]
    fn in_order_delivery() {
        let mut buf = DejitterBuffer::new(48, 64, 8, 3);
        assert!(!buf.is_primed());
        for i in 0..3i64 {
            assert_eq!(buf.push(i, &payload_for(i)), PushOutcome::Accepted);
        }
        assert!(buf.is_primed());
        assert_eq!(buf.fill_packets(), 3);
        let mut out = Vec::new();
        for i in 0..3i64 {
            assert_eq!(buf.pop(&mut out), PopOutcome::Payload);
            assert_eq!(out, payload_for(i));
        }
        assert_eq!(buf.stats().delivered, 3);
        assert_eq!(buf.stats().lost, 0);
        // Starved after the last packet: pops advance as losses.
        assert_eq!(buf.pop(&mut out), PopOutcome::Missing);
        assert_eq!(buf.stats().lost, 1);
    }

    #[test]
    fn reordering_is_absorbed() {
        let mut buf = DejitterBuffer::new(48, 64, 8, 3);
        assert_eq!(buf.push(0, &payload_for(0)), PushOutcome::Accepted);
        assert_eq!(buf.push(3, &payload_for(3)), PushOutcome::Accepted);
        assert_eq!(buf.push(2, &payload_for(2)), PushOutcome::Accepted);
        assert_eq!(buf.push(1, &payload_for(1)), PushOutcome::Accepted);
        assert_eq!(buf.stats().reordered, 2);
        assert_eq!(buf.fill_packets(), 4);
        let mut out = Vec::new();
        for i in 0..4i64 {
            assert_eq!(buf.pop(&mut out), PopOutcome::Payload);
            assert_eq!(out, payload_for(i));
        }
        assert_eq!(buf.stats().lost, 0);
    }

    #[test]
    fn loss_and_late_accounting() {
        let mut buf = DejitterBuffer::new(48, 64, 8, 3);
        for i in [0i64, 1, 3, 4] {
            assert_eq!(buf.push(i, &payload_for(i)), PushOutcome::Accepted);
        }
        let mut out = Vec::new();
        assert_eq!(buf.pop(&mut out), PopOutcome::Payload);
        assert_eq!(buf.pop(&mut out), PopOutcome::Payload);
        assert_eq!(buf.pop(&mut out), PopOutcome::Missing); // packet 2 never came
        assert_eq!(buf.pop(&mut out), PopOutcome::Payload);
        assert_eq!(buf.pop(&mut out), PopOutcome::Payload);
        assert_eq!(buf.stats().lost, 1);
        // Packet 2 straggles in behind the cursor: late, discarded.
        assert_eq!(buf.push(2, &payload_for(2)), PushOutcome::Late);
        assert_eq!(buf.stats().late, 1);
        assert_eq!(buf.fill_packets(), 0);
    }

    #[test]
    fn duplicates_and_oversize_are_rejected() {
        let mut buf = DejitterBuffer::new(48, 8, 8, 2);
        assert_eq!(buf.push(0, b"12345678"), PushOutcome::Accepted);
        assert_eq!(buf.push(0, b"12345678"), PushOutcome::Duplicate);
        assert_eq!(buf.push(1, b"123456789"), PushOutcome::Oversize);
        assert_eq!(buf.stats().duplicates, 1);
        assert_eq!(buf.stats().oversize, 1);
        assert_eq!(buf.fill_packets(), 1);
    }

    #[test]
    fn far_future_overflow_is_reported_not_stored() {
        let mut buf = DejitterBuffer::new(48, 64, 8, 3);
        assert_eq!(buf.push(100, &payload_for(100)), PushOutcome::Accepted);
        assert_eq!(buf.push(108, &payload_for(108)), PushOutcome::Overflow);
        assert_eq!(buf.stats().overflow, 1);
        assert_eq!(buf.fill_packets(), 1);
        // Caller resets and re-anchors on the newest packet.
        buf.reset();
        assert!(!buf.is_primed());
        assert_eq!(buf.push(200, &payload_for(200)), PushOutcome::Accepted);
        assert_eq!(buf.next_packet(), 200);
    }

    #[test]
    fn priming_does_not_consume_packets() {
        let mut buf = DejitterBuffer::new(48, 64, 8, 3);
        let mut out = Vec::new();
        assert_eq!(buf.pop(&mut out), PopOutcome::Priming);
        assert_eq!(buf.push(7, &payload_for(7)), PushOutcome::Accepted);
        assert_eq!(buf.push(8, &payload_for(8)), PushOutcome::Accepted);
        assert_eq!(buf.pop(&mut out), PopOutcome::Priming);
        assert_eq!(buf.next_packet(), 7); // cursor unmoved
        assert_eq!(buf.push(9, &payload_for(9)), PushOutcome::Accepted);
        assert_eq!(buf.pop(&mut out), PopOutcome::Payload);
        assert_eq!(out, payload_for(7));
    }

    #[test]
    fn handles_far_from_zero_indices() {
        let base = 1i64 << 40;
        let mut buf = DejitterBuffer::new(48, 64, 8, 2);
        for k in 0..4i64 {
            assert_eq!(buf.push(base + k, &payload_for(base + k)), PushOutcome::Accepted);
        }
        let mut out = Vec::new();
        for k in 0..4i64 {
            assert_eq!(buf.pop(&mut out), PopOutcome::Payload);
            assert_eq!(out, payload_for(base + k));
        }
    }

    #[test]
    fn rtp_index_across_the_wraparound() {
        // Packet whose samples straddle the 2^32 RTP timestamp wrap.
        let frames = 48i64;
        let p0 = (1i64 << 32).div_euclid(frames) + 1; // first packet fully past the wrap
        let mut buf = DejitterBuffer::new(48, 64, 16, 2);
        assert_eq!(buf.push(p0 - 1, &payload_for(p0 - 1)), PushOutcome::Accepted);
        let ts = rtp_timestamp(p0 * frames); // wrapped small value
        assert_eq!(buf.packet_index_from_rtp(ts), p0);
        // Still correct well past the wrap, re-anchored on the newer packet.
        let p1 = p0 + 1000;
        let mut buf2 = DejitterBuffer::new(48, 64, 16, 2);
        assert_eq!(buf2.push(p1, &payload_for(p1)), PushOutcome::Accepted);
        assert_eq!(buf2.packet_index_from_rtp(rtp_timestamp(p1 * frames)), p1);
    }
}
