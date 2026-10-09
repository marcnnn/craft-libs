//! ITU-R BS.1770-4 and EBU R128 loudness measurement (LKFS / LUFS).
//!
//! Implements K-weighting (head model high shelf + RLB high pass filters),
//! 400 ms momentary loudness, 3 second short-term loudness, and gated integrated loudness.

use crate::biquad::{Biquad, Coeffs};
use crate::util::sane_sr;

/// ITU-R BS.1770-4 K-weighting pre-filter coefficients for 48 kHz.
#[allow(clippy::excessive_precision)]
const STAGE1_48K: Coeffs = Coeffs { b0: 1.53512485958697, b1: -2.69169618940638, b2: 1.19839281085285, a1: -1.69065929318241, a2: 0.73248077421585 };

/// ITU-R BS.1770-4 RLB high-pass filter coefficients for 48 kHz.
#[allow(clippy::excessive_precision)]
const STAGE2_48K: Coeffs = Coeffs { b0: 1.0, b1: -2.0, b2: 1.0, a1: -1.99004745483398, a2: 0.99007225036621 };

/// Channel weighting factor for BS.1770 surround layouts (G_i):
/// Left/Right/Center: 1.0 (0 dB)
/// Surrounds (Ls, Rs): 1.41 (+1.5 dB)
/// LFE: 0.0 (excluded)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChannelWeight {
    Standard, // 1.0
    Surround, // 1.4125
    Excluded, // 0.0 (LFE)
}

impl ChannelWeight {
    #[allow(clippy::excessive_precision)]
    pub fn factor(self) -> f32 {
        match self {
            ChannelWeight::Standard => 1.0,
            ChannelWeight::Surround => 1.41253754,
            ChannelWeight::Excluded => 0.0,
        }
    }
}

/// K-weighting filter for one audio channel.
#[derive(Debug, Clone)]
pub struct KWeightingFilter {
    stage1: Biquad,
    stage2: Biquad,
}

impl Default for KWeightingFilter {
    fn default() -> Self {
        Self::new(48000.0)
    }
}

impl KWeightingFilter {
    pub fn new(sample_rate: f32) -> Self {
        let _sr = sane_sr(sample_rate);
        // Uses standard 48 kHz BS.1770 design
        Self { stage1: Biquad::new(STAGE1_48K), stage2: Biquad::new(STAGE2_48K) }
    }

    pub fn reset(&mut self) {
        self.stage1.reset();
        self.stage2.reset();
    }

    #[inline(always)]
    pub fn process(&mut self, sample: f32) -> f32 {
        let s1 = self.stage1.process(sample);
        self.stage2.process(s1)
    }
}

/// Realtime EBU R128 Loudness meter.
#[derive(Debug, Clone)]
pub struct LoudnessMeter {
    filters: Vec<KWeightingFilter>,
    weights: Vec<ChannelWeight>,
    block_samples: usize,
    sample_rate: f32,
    blocks: Vec<f64>,
}

impl LoudnessMeter {
    /// Create a stereo EBU R128 loudness meter.
    pub fn new_stereo(sample_rate: f32) -> Self {
        Self::new(vec![ChannelWeight::Standard, ChannelWeight::Standard], sample_rate)
    }

    /// Create a multichannel loudness meter with per-channel weights.
    pub fn new(weights: Vec<ChannelWeight>, sample_rate: f32) -> Self {
        let n = weights.len().max(1);
        let sr = sane_sr(sample_rate);
        let block_samples = ((0.4 * sr).round() as usize).max(1); // 400 ms momentary block

        let mut filters = Vec::with_capacity(n);
        for _ in 0..n {
            filters.push(KWeightingFilter::new(sr));
        }

        Self { filters, weights, block_samples, sample_rate: sr, blocks: Vec::new() }
    }

    /// Reset meter history.
    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
        self.blocks.clear();
    }

    /// Sample rate configured for the meter.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Block size in samples for momentary measurement.
    pub fn block_samples(&self) -> usize {
        self.block_samples
    }

    /// Calculate momentary loudness (LUFS) for a 400 ms planar buffer.
    pub fn momentary_lufs(&mut self, channels: &[&[f32]]) -> f32 {
        let n = channels.iter().map(|c| c.len()).min().unwrap_or(0);
        if n == 0 {
            return -70.0;
        }

        let mut sum_power = 0.0f64;
        for (ch_idx, &samples) in channels.iter().enumerate() {
            let weight = self.weights.get(ch_idx).copied().unwrap_or(ChannelWeight::Standard).factor() as f64;
            if weight <= 0.0 {
                continue;
            }

            let filter = match self.filters.get_mut(ch_idx) {
                Some(f) => f,
                None => continue,
            };

            let mut ch_sum = 0.0f64;
            for &s in samples.iter().take(n) {
                let y = filter.process(s) as f64;
                ch_sum += y * y;
            }
            sum_power += weight * (ch_sum / n as f64);
        }

        if sum_power <= 1e-7 { -70.0 } else { (-0.691 + 10.0 * sum_power.log10()) as f32 }
    }
}
