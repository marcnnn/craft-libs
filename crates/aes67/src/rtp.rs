//! RTP timestamp ↔ sample-index mapping (ST 2110-30 / AES67).
//!
//! For L16/L24 audio the RTP clock rate equals the sample rate (48 kHz here), and the
//! RTP timestamp is the sender's media-clock sample count modulo 2³²:
//! `RTP Timestamp = (PTP samples since epoch) mod 2³²` (HANDOFF §5.7). Receiving code
//! must unwrap the modulo against its previous knowledge — the counter wraps after
//! ~24.9 hours at 48 kHz, and a receiver that treats it as a plain `u32` sees a huge
//! backwards jump once per day.

/// RTP timestamp modulus (32-bit counter).
pub const RTP_TIMESTAMP_MODULUS: i64 = 1 << 32;

/// RTP timestamp of an absolute sample index (wrapping by design).
#[must_use]
pub fn rtp_timestamp(sample_index: i64) -> u32 {
    sample_index.rem_euclid(RTP_TIMESTAMP_MODULUS) as u32
}

/// Recover the absolute sample index from an RTP timestamp, choosing the 2³² extension
/// nearest `previous_sample`. Negative or zero sample indices behave correctly
/// (Euclidean modulus).
#[must_use]
pub fn unwrap_rtp_timestamp(timestamp: u32, previous_sample: i64) -> i64 {
    let previous = previous_sample;
    let base = previous - previous.rem_euclid(RTP_TIMESTAMP_MODULUS); // multiple of 2^32
    let candidate = base + i64::from(timestamp);
    // Nearest of candidate - M, candidate, candidate + M to previous_sample.
    let mut best = candidate;
    for c in [candidate - RTP_TIMESTAMP_MODULUS, candidate, candidate + RTP_TIMESTAMP_MODULUS] {
        if (c - previous).abs() < (best - previous).abs() {
            best = c;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_below_the_modulus() {
        assert_eq!(rtp_timestamp(0), 0);
        assert_eq!(rtp_timestamp(48), 48);
        assert_eq!(rtp_timestamp(RTP_TIMESTAMP_MODULUS - 1), u32::MAX);
    }

    #[test]
    fn wraps_every_2_to_the_32_samples() {
        // Exactly one full counter: back to zero, ~24.9 h at 48 kHz.
        assert_eq!(rtp_timestamp(RTP_TIMESTAMP_MODULUS), 0);
        assert_eq!(rtp_timestamp(RTP_TIMESTAMP_MODULUS + 48), 48);
    }

    #[test]
    fn unwrap_survives_the_wraparound() {
        // Streaming up to the wrap: last sample before, first sample after.
        let before = RTP_TIMESTAMP_MODULUS - 48;
        let after = RTP_TIMESTAMP_MODULUS + 48;
        assert_eq!(unwrap_rtp_timestamp(rtp_timestamp(before), before - 4800), before);
        assert_eq!(unwrap_rtp_timestamp(rtp_timestamp(after), before), after);
        // A late reordered packet from before the wrap still maps backwards.
        assert_eq!(unwrap_rtp_timestamp(rtp_timestamp(before), after), before);
    }

    #[test]
    fn unwrap_near_zero_handles_negative_extensions() {
        assert_eq!(unwrap_rtp_timestamp(96, 0), 96);
        // Deep negative history: the nearest extension is two moduli down.
        let previous = -(RTP_TIMESTAMP_MODULUS * 2 + 10);
        assert_eq!(unwrap_rtp_timestamp(5, previous), 5 - RTP_TIMESTAMP_MODULUS * 2);
    }

    #[test]
    fn packet_indices_round_trip() {
        // Packet-level use: indices at 48 frames per packet across the wrap.
        let frames = 48i64;
        for packet in [0i64, 1, 89_507_772, RTP_TIMESTAMP_MODULUS / frames - 1, RTP_TIMESTAMP_MODULUS / frames + 3] {
            let sample = packet * frames;
            let ts = rtp_timestamp(sample);
            assert_eq!(unwrap_rtp_timestamp(ts, sample - 480), sample);
        }
    }
}
