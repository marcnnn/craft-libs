//! Audio metering: peak, RMS, and true-peak level detectors with ballistics.

use crate::util::{db_to_gain, flush, gain_to_db};

/// Peak level meter with configurable falloff ballistics.
#[derive(Debug, Clone)]
pub struct PeakMeter {
    peak: f32,
    hold_peak: f32,
    decay_per_sample: f32,
}

impl Default for PeakMeter {
    fn default() -> Self {
        Self::new(20.0, 48000.0) // 20 dB/second standard decay
    }
}

impl PeakMeter {
    /// Create a peak meter with a release rate in dB/sec at the given sample rate.
    pub fn new(release_db_per_sec: f32, sample_rate: f32) -> Self {
        let sr = sample_rate.max(1.0);
        let decay_db = release_db_per_sec / sr;
        let decay = db_to_gain(-decay_db);
        Self { peak: 0.0, hold_peak: 0.0, decay_per_sample: decay }
    }

    /// Reset meters to silence.
    pub fn reset(&mut self) {
        self.peak = 0.0;
        self.hold_peak = 0.0;
    }

    /// Process a single sample and update internal peak state.
    #[inline]
    pub fn process_sample(&mut self, sample: f32) {
        let abs = sample.abs();
        if abs > self.peak {
            self.peak = abs;
        } else {
            self.peak = flush(self.peak * self.decay_per_sample);
        }

        if abs > self.hold_peak {
            self.hold_peak = abs;
        }
    }

    /// Process a block of samples.
    pub fn process_block(&mut self, samples: &[f32]) {
        for &s in samples {
            self.process_sample(s);
        }
    }

    /// Current peak level in linear amplitude.
    #[inline]
    pub fn peak(&self) -> f32 {
        self.peak
    }

    /// Current peak level in decibels.
    #[inline]
    pub fn peak_db(&self) -> f32 {
        gain_to_db(self.peak)
    }

    /// Read and reset the peak-hold value in decibels.
    pub fn take_hold_db(&mut self) -> f32 {
        let val = gain_to_db(self.hold_peak);
        self.hold_peak = self.peak;
        val
    }
}

/// Windowed Root Mean Square (RMS) level meter.
#[derive(Debug, Clone)]
pub struct RmsMeter {
    window_samples: usize,
    sum_squares: f64,
    history: Vec<f32>,
    pos: usize,
}

impl RmsMeter {
    /// Create an RMS meter with window length in milliseconds (e.g. 300 ms for AES17).
    pub fn new(window_ms: f32, sample_rate: f32) -> Self {
        let len = ((window_ms * 0.001 * sample_rate).round() as usize).max(1);
        Self { window_samples: len, sum_squares: 0.0, history: vec![0.0; len], pos: 0 }
    }

    /// Reset internal RMS history.
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.sum_squares = 0.0;
        self.pos = 0;
    }

    /// Process a sample and return current RMS level in linear amplitude.
    #[inline]
    pub fn process_sample(&mut self, sample: f32) -> f32 {
        let s64 = f64::from(sample);
        let sq = s64 * s64;
        let old = f64::from(self.history[self.pos]);
        let old_sq = old * old;

        self.sum_squares = (self.sum_squares + sq - old_sq).max(0.0);
        self.history[self.pos] = sample;
        self.pos = (self.pos + 1) % self.window_samples;

        let mean = self.sum_squares / (self.window_samples as f64);
        mean.sqrt() as f32
    }

    /// Current RMS level in decibels.
    pub fn current_db(&self) -> f32 {
        let mean = self.sum_squares / (self.window_samples as f64);
        gain_to_db(mean.sqrt() as f32)
    }
}
