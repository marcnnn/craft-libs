//! ITU-R BS.1770-4 / EBU R128 loudness measurement (LKFS / LUFS).
//!
//! - K-weighting is derived from the BS.1770 analog prototype parameters (high-shelf "head"
//!   pre-filter and RLB high-pass) through the bilinear transform with prewarping, so **any**
//!   sample rate is supported; at 48 kHz the design reproduces the coefficient table in the
//!   recommendation. Filters run in f64 to keep the specified coefficient precision.
//! - The meter is streaming: feed arbitrary block sizes via [`LoudnessMeter::process`]; 100 ms
//!   sub-block energies drive momentary (400 ms), short-term (3 s), and gated integrated
//!   loudness (absolute -70 LUFS gate, relative -10 LU gate).
//! - Gating blocks are accumulated into a fixed 0.05 LU histogram that also stores exact energy
//!   sums (gate logic follows the FilmCraft meter, BS.1770-4 §5), so the meter runs for
//!   arbitrarily long programmes with bounded memory and no steady-state allocation.

use std::f64::consts::PI;

use crate::util::{flush64, sane_sr};

const HIST_MIN: f64 = -70.0;
const HIST_MAX: f64 = 20.0;
const HIST_STEP: f64 = 0.05;
const HIST_BINS: usize = ((HIST_MAX - HIST_MIN) / HIST_STEP) as usize;
/// Momentary window in 100 ms sub-blocks (400 ms).
const M_SUBBLOCKS: usize = 4;
/// Short-term window in 100 ms sub-blocks (3 s).
const ST_SUBBLOCKS: usize = 30;

/// Loudness of a channel-weighted mean-square energy, in LUFS.
/// Returns [`f64::NEG_INFINITY`] for zero energy (silence).
#[inline]
pub fn energy_to_lufs(e: f64) -> f64 {
    if e > 0.0 { -0.691 + 10.0 * e.log10() } else { f64::NEG_INFINITY }
}

/// Channel weighting factor for BS.1770 surround layouts (G_i):
/// Left/Right/Center: 1.0 (0 dB), Surrounds (Ls, Rs): 1.41 (+1.5 dB), LFE: 0.0 (excluded).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelWeight {
    Standard,
    Surround,
    Excluded,
}

impl ChannelWeight {
    #[allow(clippy::excessive_precision)]
    pub fn factor(self) -> f64 {
        match self {
            ChannelWeight::Standard => 1.0,
            ChannelWeight::Surround => 1.41253754,
            ChannelWeight::Excluded => 0.0,
        }
    }
}

/// Transposed direct form II biquad with f64 coefficients and state.
#[derive(Debug, Clone, Copy, Default)]
struct Biquad64 {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad64 {
    #[inline(always)]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = flush64(self.b1 * x - self.a1 * y + self.z2);
        self.z2 = flush64(self.b2 * x - self.a2 * y);
        y
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Stage 1 of K-weighting: the high-shelf ("head") pre-filter for `sample_rate`.
/// Analog prototype of the BS.1770 shelving stage, bilinear-transformed with prewarp.
fn k_prefilter(sample_rate: f64) -> Biquad64 {
    let f0 = 1_681.974_450_955_533;
    let gain_db = 3.999_843_853_973_347;
    let q = 0.707_175_236_955_419_6;
    let k = (PI * f0 / sample_rate).tan();
    let vh = 10f64.powf(gain_db / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6);
    let a0 = 1.0 + k / q + k * k;
    Biquad64 {
        b0: (vh + vb * k / q + k * k) / a0,
        b1: 2.0 * (k * k - vh) / a0,
        b2: (vh - vb * k / q + k * k) / a0,
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / q + k * k) / a0,
        z1: 0.0,
        z2: 0.0,
    }
}

/// Stage 2 of K-weighting: the RLB high-pass for `sample_rate`.
fn k_rlb(sample_rate: f64) -> Biquad64 {
    let f0 = 38.135_470_876_024_44;
    let q = 0.500_327_037_323_877_3;
    let k = (PI * f0 / sample_rate).tan();
    let a0 = 1.0 + k / q + k * k;
    Biquad64 { b0: 1.0, b1: -2.0, b2: 1.0, a1: 2.0 * (k * k - 1.0) / a0, a2: (1.0 - k / q + k * k) / a0, z1: 0.0, z2: 0.0 }
}

/// K-weighting filter (both BS.1770 stages) for one audio channel, designed for
/// the actual `sample_rate`. Returns the weighted sample in f64 for energy accumulation.
#[derive(Debug, Clone)]
pub struct KWeightingFilter {
    stage1: Biquad64,
    stage2: Biquad64,
}

impl Default for KWeightingFilter {
    fn default() -> Self {
        Self::new(48000.0)
    }
}

impl KWeightingFilter {
    /// Design the K-weighting stages for `sample_rate` Hz (any rate; 48 kHz reproduces
    /// the coefficient table in ITU-R BS.1770-4).
    pub fn new(sample_rate: f32) -> Self {
        let sr = f64::from(sane_sr(sample_rate));
        Self { stage1: k_prefilter(sr), stage2: k_rlb(sr) }
    }

    /// Clear internal filter state (silence history).
    pub fn reset(&mut self) {
        self.stage1.reset();
        self.stage2.reset();
    }

    /// Process one sample, returning the K-weighted value.
    #[inline(always)]
    pub fn process(&mut self, sample: f32) -> f64 {
        let s1 = self.stage1.process(f64::from(sample));
        self.stage2.process(s1)
    }
}

/// Histogram of 400 ms gating block loudness with exact energy sums per bin.
#[derive(Debug, Clone)]
struct Histogram {
    count: Vec<u64>,
    energy: Vec<f64>,
}

impl Histogram {
    fn new() -> Self {
        Histogram { count: vec![0; HIST_BINS], energy: vec![0.0; HIST_BINS] }
    }

    fn clear(&mut self) {
        self.count.iter_mut().for_each(|c| *c = 0);
        self.energy.iter_mut().for_each(|e| *e = 0.0);
    }

    fn bin_of(lufs: f64) -> usize {
        (((lufs - HIST_MIN) / HIST_STEP).floor().max(0.0) as usize).min(HIST_BINS - 1)
    }

    /// Add a gating block if it passes the absolute (-70 LUFS) gate.
    fn add(&mut self, energy: f64) {
        let l = energy_to_lufs(energy);
        if l >= HIST_MIN {
            let b = Self::bin_of(l);
            self.count[b] = self.count[b].saturating_add(1);
            self.energy[b] += energy;
        }
    }

    /// Mean energy and block count over bins from `from` upward; `None` if empty.
    fn mean_energy_from(&self, from: usize) -> Option<(f64, u64)> {
        let (mut e, mut n) = (0.0f64, 0u64);
        for i in from.min(HIST_BINS)..HIST_BINS {
            e += self.energy[i];
            n = n.saturating_add(self.count[i]);
        }
        (n > 0).then(|| (e / n as f64, n))
    }
}

/// Streaming ITU-R BS.1770-4 / EBU R128 loudness meter for planar `f32` audio.
///
/// Feed audio of arbitrary block sizes with [`process`](LoudnessMeter::process); read
/// [`momentary_lufs`](LoudnessMeter::momentary_lufs), [`short_term_lufs`](LoudnessMeter::short_term_lufs)
/// and gated [`integrated_lufs`](LoudnessMeter::integrated_lufs) at any time. Readouts update in
/// 100 ms steps; during the first 400 ms the momentary window is partial. All readouts floor at
/// -70.0 LUFS for silence.
#[derive(Debug, Clone)]
pub struct LoudnessMeter {
    filters: Vec<KWeightingFilter>,
    weights: Vec<f64>,
    sample_rate: f64,
    sub_len: usize,
    sub_fill: usize,
    sub_energy: f64,
    subs: Vec<f64>,
    sub_pos: usize,
    sub_count: usize,
    hist: Histogram,
}

impl LoudnessMeter {
    /// Create a stereo EBU R128 loudness meter.
    pub fn new_stereo(sample_rate: f32) -> Self {
        Self::new(vec![ChannelWeight::Standard, ChannelWeight::Standard], sample_rate)
    }

    /// Create a multichannel loudness meter with per-channel weights.
    pub fn new(weights: Vec<ChannelWeight>, sample_rate: f32) -> Self {
        let sr = f64::from(sane_sr(sample_rate));
        let sub_len = ((0.1 * sr).round() as usize).max(1); // 100 ms sub-block
        let filters = weights.iter().map(|_| KWeightingFilter::new(sr as f32)).collect();
        Self {
            filters,
            weights: weights.iter().map(|w| w.factor()).collect(),
            sample_rate: sr,
            sub_len,
            sub_fill: 0,
            sub_energy: 0.0,
            subs: vec![0.0; ST_SUBBLOCKS],
            sub_pos: 0,
            sub_count: 0,
            hist: Histogram::new(),
        }
    }

    /// Reset filter state, window history, and the integrated histogram.
    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
        self.sub_fill = 0;
        self.sub_energy = 0.0;
        self.subs.iter_mut().for_each(|s| *s = 0.0);
        self.sub_pos = 0;
        self.sub_count = 0;
        self.hist.clear();
    }

    /// Sample rate configured for the meter.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate as f32
    }

    /// 100 ms sub-block length in samples.
    pub fn sub_block_samples(&self) -> usize {
        self.sub_len
    }

    /// Feed a block of planar audio. Channels beyond the configured weights are ignored;
    /// excluded (weight 0) channels are skipped. Call with any block size.
    pub fn process(&mut self, channels: &[&[f32]]) {
        let n = channels.iter().map(|c| c.len()).min().unwrap_or(0);
        for i in 0..n {
            let mut e = 0.0f64;
            for (ch, samples) in channels.iter().enumerate() {
                let weight = self.weights.get(ch).copied().unwrap_or(1.0);
                if weight <= 0.0 {
                    continue;
                }
                let Some(filter) = self.filters.get_mut(ch) else { continue };
                let x = samples.get(i).copied().unwrap_or(0.0);
                let y = filter.process(x);
                e += weight * y * y;
            }
            self.sub_energy += e;
            self.sub_fill += 1;
            if self.sub_fill >= self.sub_len {
                self.complete_sub_block();
            }
        }
    }

    /// Close the current 100 ms sub-block and update windows plus the gating histogram.
    fn complete_sub_block(&mut self) {
        let mean = self.sub_energy / self.sub_len as f64;
        self.sub_energy = 0.0;
        self.sub_fill = 0;
        self.subs[self.sub_pos] = mean;
        self.sub_pos = (self.sub_pos + 1) % ST_SUBBLOCKS;
        self.sub_count = self.sub_count.saturating_add(1);
        if self.sub_count >= M_SUBBLOCKS {
            let block = self.window_energy(M_SUBBLOCKS);
            self.hist.add(block);
        }
    }

    /// Mean sub-block energy over the most recent `count` completed sub-blocks
    /// (fewer while the meter is filling). Zero when no sub-block has completed.
    fn window_energy(&self, count: usize) -> f64 {
        let avail = self.sub_count.min(count);
        if avail == 0 {
            return 0.0;
        }
        let mut sum = 0.0f64;
        for k in 1..=avail {
            let idx = (self.sub_pos + ST_SUBBLOCKS - k) % ST_SUBBLOCKS;
            sum += self.subs[idx];
        }
        sum / avail as f64
    }

    fn window_lufs(&self, count: usize) -> f32 {
        let l = energy_to_lufs(self.window_energy(count));
        if l.is_finite() && l > HIST_MIN { l as f32 } else { HIST_MIN as f32 }
    }

    /// Momentary loudness (latest 400 ms window), floored at -70.0 LUFS.
    pub fn momentary_lufs(&self) -> f32 {
        self.window_lufs(M_SUBBLOCKS)
    }

    /// Short-term loudness (latest 3 s window), floored at -70.0 LUFS.
    pub fn short_term_lufs(&self) -> f32 {
        self.window_lufs(ST_SUBBLOCKS)
    }

    /// Gated integrated loudness (BS.1770-4: -70 LUFS absolute gate, -10 LU relative gate),
    /// floored at -70.0 LUFS when no block passes the gates.
    pub fn integrated_lufs(&self) -> f32 {
        let Some((abs_mean, _)) = self.hist.mean_energy_from(0) else { return HIST_MIN as f32 };
        let threshold = energy_to_lufs(abs_mean) - 10.0; // relative gate
        let from = if threshold <= HIST_MIN { 0 } else { Histogram::bin_of(threshold) };
        match self.hist.mean_energy_from(from) {
            Some((e, _)) => {
                let l = energy_to_lufs(e);
                if l.is_finite() && l > HIST_MIN { l as f32 } else { HIST_MIN as f32 }
            }
            None => HIST_MIN as f32,
        }
    }
}
