//! Dynamics processing: compressor, look-ahead limiter, and noise gate.
//!
//! Gain computers operate in the decibel domain with quadratic soft knee interpolation.
//! Detector envelopes are stereo-linked to prevent stereo image shifting.

use crate::util::{db_to_gain, flush, gain_to_db, time_coef};

/// Compute static compressor gain reduction in decibels (always >= 0.0).
/// `level_db` is the input signal envelope in dB.
#[inline]
pub fn compressor_gain_reduction(level_db: f32, threshold: f32, ratio: f32, knee: f32) -> f32 {
    let r = ratio.max(1.0);
    let slope = 1.0 - 1.0 / r;
    let over = level_db - threshold;
    let half_knee = knee.max(0.0) * 0.5;

    if half_knee > 1e-4 && over.abs() <= half_knee {
        let t = over + half_knee;
        slope * t * t / (2.0 * half_knee)
    } else if over > 0.0 {
        slope * over
    } else {
        0.0
    }
}

/// Compute downward expander/gate gain reduction in decibels (always >= 0.0).
#[inline]
pub fn expander_gain_reduction(level_db: f32, threshold: f32, ratio: f32, range_db: f32) -> f32 {
    let r = ratio.max(1.0);
    let under = threshold - level_db;
    if under > 0.0 {
        let att = under * (r - 1.0);
        att.min(range_db.max(0.0))
    } else {
        0.0
    }
}

/// Dynamic range compressor with soft knee and make-up gain.
#[derive(Debug, Clone)]
pub struct Compressor {
    threshold: f32,
    ratio: f32,
    knee: f32,
    makeup_gain: f32,
    att_coef: f32,
    rel_coef: f32,
    envelope: f32,
    sample_rate: f32,
}

impl Compressor {
    /// Create a new compressor.
    /// `threshold_db` (-60.0 to 0.0 dB), `ratio` (1.0 to 100.0), `knee_db` (0.0 to 24.0 dB),
    /// `attack_ms` (0.1 to 200.0 ms), `release_ms` (10.0 to 2000.0 ms).
    pub fn new(threshold_db: f32, ratio: f32, knee_db: f32, attack_ms: f32, release_ms: f32, makeup_db: f32, sample_rate: f32) -> Self {
        let mut comp = Self {
            threshold: threshold_db,
            ratio: ratio.max(1.0),
            knee: knee_db.max(0.0),
            makeup_gain: db_to_gain(makeup_db),
            att_coef: 0.0,
            rel_coef: 0.0,
            envelope: 0.0,
            sample_rate,
        };
        comp.set_timing(attack_ms, release_ms, sample_rate);
        comp
    }

    /// Update attack and release times.
    pub fn set_timing(&mut self, attack_ms: f32, release_ms: f32, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.att_coef = time_coef(attack_ms, sample_rate);
        self.rel_coef = time_coef(release_ms, sample_rate);
    }

    /// Reset envelope state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }

    /// Process a stereo frame (interleaved or planar slice pair) in-place.
    /// Returns the current gain reduction amount in dB.
    #[inline]
    pub fn process_stereo(&mut self, left: &mut f32, right: &mut f32) -> f32 {
        let det_in = left.abs().max(right.abs());
        let in_db = gain_to_db(det_in);

        let target_gr = compressor_gain_reduction(in_db, self.threshold, self.ratio, self.knee);

        // Attack when gain reduction increases, release when it decreases
        let coef = if target_gr > self.envelope { self.att_coef } else { self.rel_coef };
        self.envelope = flush(target_gr + (self.envelope - target_gr) * coef);

        let gr_linear = db_to_gain(-self.envelope);
        let total_gain = gr_linear * self.makeup_gain;

        *left *= total_gain;
        *right *= total_gain;

        self.envelope
    }

    /// Process planar multichannel audio buffers in-place.
    pub fn process_planar(&mut self, channels: &mut [&mut [f32]]) -> f32 {
        let n = channels.iter().map(|c| c.len()).min().unwrap_or(0);
        let mut last_gr = 0.0f32;

        for i in 0..n {
            let mut det_in = 0.0f32;
            for ch in channels.iter() {
                if let Some(sample) = ch.get(i) {
                    det_in = det_in.max(sample.abs());
                }
            }

            let in_db = gain_to_db(det_in);
            let target_gr = compressor_gain_reduction(in_db, self.threshold, self.ratio, self.knee);

            let coef = if target_gr > self.envelope { self.att_coef } else { self.rel_coef };
            self.envelope = flush(target_gr + (self.envelope - target_gr) * coef);
            last_gr = self.envelope;

            let gr_linear = db_to_gain(-self.envelope);
            let total_gain = gr_linear * self.makeup_gain;

            for ch in channels.iter_mut() {
                if let Some(sample) = ch.get_mut(i) {
                    *sample *= total_gain;
                }
            }
        }

        last_gr
    }
}

/// Look-ahead brickwall peak limiter.
#[derive(Debug, Clone)]
pub struct Limiter {
    ceiling_db: f32,
    ceiling_linear: f32,
    rel_coef: f32,
    envelope: f32,
    delay_l: Vec<f32>,
    delay_r: Vec<f32>,
    pos: usize,
    lookahead_samples: usize,
}

impl Limiter {
    /// Create a look-ahead limiter with ceiling in dB (e.g. -0.1 dB),
    /// lookahead in milliseconds (e.g. 5.0 ms), and release in milliseconds (e.g. 50.0 ms).
    pub fn new(ceiling_db: f32, lookahead_ms: f32, release_ms: f32, sample_rate: f32) -> Self {
        let ceiling = db_to_gain(ceiling_db.min(0.0));
        let la = ((lookahead_ms * 0.001 * sample_rate).round() as usize).max(1);
        Self {
            ceiling_db,
            ceiling_linear: ceiling,
            rel_coef: time_coef(release_ms, sample_rate),
            envelope: 1.0,
            delay_l: vec![0.0; la],
            delay_r: vec![0.0; la],
            pos: 0,
            lookahead_samples: la,
        }
    }

    /// Reset internal delay line and envelope.
    pub fn reset(&mut self) {
        self.delay_l.fill(0.0);
        self.delay_r.fill(0.0);
        self.pos = 0;
        self.envelope = 1.0;
    }

    /// Ceiling level in decibels.
    pub fn ceiling_db(&self) -> f32 {
        self.ceiling_db
    }

    /// Latency introduced by the look-ahead buffer in samples.
    pub fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }

    /// Process a stereo frame.
    #[inline]
    pub fn process_stereo(&mut self, left: &mut f32, right: &mut f32) -> f32 {
        let in_l = *left;
        let in_r = *right;

        // Peak detector ahead of delayed audio
        let peak = in_l.abs().max(in_r.abs());
        let required_gain = if peak > self.ceiling_linear { self.ceiling_linear / peak } else { 1.0 };

        // Instant attack, smoothed release
        if required_gain < self.envelope {
            self.envelope = required_gain;
        } else {
            self.envelope = flush(required_gain + (self.envelope - required_gain) * self.rel_coef);
        }

        // Read delayed audio from look-ahead buffer
        let out_l = self.delay_l[self.pos];
        let out_r = self.delay_r[self.pos];

        // Store current input
        self.delay_l[self.pos] = in_l;
        self.delay_r[self.pos] = in_r;
        self.pos = (self.pos + 1) % self.lookahead_samples;

        *left = (out_l * self.envelope).clamp(-self.ceiling_linear, self.ceiling_linear);
        *right = (out_r * self.envelope).clamp(-self.ceiling_linear, self.ceiling_linear);

        gain_to_db(self.envelope)
    }
}
