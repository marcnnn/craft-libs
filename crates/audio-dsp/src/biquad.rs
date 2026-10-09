//! Second-order IIR biquad filters designed from the Robert Bristow-Johnson (RBJ)
//! "Audio EQ Cookbook" formulas, plus first-order 6 dB/oct sections.
//!
//! Evaluated using transposed direct form II for numerical stability and low state overhead.

use std::f64::consts::PI;

use crate::util::{flush, sane_sr};

/// Filter response curves and shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    Peak,
    LowShelf,
    HighShelf,
    AllPass,
    /// First-order low-pass (6 dB/octave). Q and gain are ignored.
    LowPass1,
    /// First-order high-pass (6 dB/octave). Q and gain are ignored.
    HighPass1,
}

/// Normalized biquad filter coefficients (with `a0 = 1.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl Default for Coeffs {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Coeffs {
    /// Pass-through (identity) filter.
    pub const IDENTITY: Coeffs = Coeffs { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };

    /// Design normalized biquad coefficients using the RBJ Audio EQ Cookbook.
    ///
    /// Frequency is clamped safely between 1.0 Hz and 0.49 * sample_rate (Nyquist safety).
    /// Quality factor `q` is clamped between 0.025 and 100.0.
    /// Gain `gain_db` is clamped between -60.0 and +60.0 dB.
    pub fn design(kind: FilterType, freq: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        let sr = f64::from(sane_sr(sample_rate));
        let f = if freq.is_finite() { f64::from(freq).clamp(1.0, sr * 0.49) } else { 1000.0 };
        let q_val = if q.is_finite() { f64::from(q).clamp(0.025, 100.0) } else { std::f64::consts::FRAC_1_SQRT_2 };
        let g = if gain_db.is_finite() { f64::from(gain_db).clamp(-60.0, 60.0) } else { 0.0 };

        match kind {
            FilterType::LowPass1 => {
                let th = 2.0 * PI * f / sr;
                let gamma = th.cos() / (1.0 + th.sin());
                let b0 = (1.0 - gamma) * 0.5;
                Coeffs { b0: b0 as f32, b1: b0 as f32, b2: 0.0, a1: -gamma as f32, a2: 0.0 }
            }
            FilterType::HighPass1 => {
                let th = 2.0 * PI * f / sr;
                let gamma = th.cos() / (1.0 + th.sin());
                let b0 = (1.0 + gamma) * 0.5;
                Coeffs { b0: b0 as f32, b1: -b0 as f32, b2: 0.0, a1: -gamma as f32, a2: 0.0 }
            }
            _ => {
                let w0 = 2.0 * PI * f / sr;
                let (sw, cw) = w0.sin_cos();
                let alpha = sw / (2.0 * q_val);
                let a = 10f64.powf(g / 40.0);

                let (b0, b1, b2, a0, a1, a2) = match kind {
                    FilterType::LowPass => ((1.0 - cw) * 0.5, 1.0 - cw, (1.0 - cw) * 0.5, 1.0 + alpha, -2.0 * cw, 1.0 - alpha),
                    FilterType::HighPass => ((1.0 + cw) * 0.5, -(1.0 + cw), (1.0 + cw) * 0.5, 1.0 + alpha, -2.0 * cw, 1.0 - alpha),
                    FilterType::BandPass => (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cw, 1.0 - alpha),
                    FilterType::Notch => (1.0, -2.0 * cw, 1.0, 1.0 + alpha, -2.0 * cw, 1.0 - alpha),
                    FilterType::Peak => (1.0 + alpha * a, -2.0 * cw, 1.0 - alpha * a, 1.0 + alpha / a, -2.0 * cw, 1.0 - alpha / a),
                    FilterType::LowShelf => {
                        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
                        (
                            a * ((a + 1.0) - (a - 1.0) * cw + two_sqrt_a_alpha),
                            2.0 * a * ((a - 1.0) - (a + 1.0) * cw),
                            a * ((a + 1.0) - (a - 1.0) * cw - two_sqrt_a_alpha),
                            (a + 1.0) + (a - 1.0) * cw + two_sqrt_a_alpha,
                            -2.0 * ((a - 1.0) + (a + 1.0) * cw),
                            (a + 1.0) + (a - 1.0) * cw - two_sqrt_a_alpha,
                        )
                    }
                    FilterType::HighShelf => {
                        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
                        (
                            a * ((a + 1.0) + (a - 1.0) * cw + two_sqrt_a_alpha),
                            -2.0 * a * ((a - 1.0) + (a + 1.0) * cw),
                            a * ((a + 1.0) + (a - 1.0) * cw - two_sqrt_a_alpha),
                            (a + 1.0) - (a - 1.0) * cw + two_sqrt_a_alpha,
                            2.0 * ((a - 1.0) - (a + 1.0) * cw),
                            (a + 1.0) - (a - 1.0) * cw - two_sqrt_a_alpha,
                        )
                    }
                    FilterType::AllPass => (1.0 - alpha, -2.0 * cw, 1.0 + alpha, 1.0 + alpha, -2.0 * cw, 1.0 - alpha),
                    FilterType::LowPass1 | FilterType::HighPass1 => (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                };

                let inv_a0 = 1.0 / a0;
                Coeffs {
                    b0: (b0 * inv_a0) as f32,
                    b1: (b1 * inv_a0) as f32,
                    b2: (b2 * inv_a0) as f32,
                    a1: (a1 * inv_a0) as f32,
                    a2: (a2 * inv_a0) as f32,
                }
            }
        }
    }

    /// Compute the complex magnitude response in decibels at a given frequency.
    pub fn magnitude_db(&self, freq_hz: f32, sample_rate: f32) -> f32 {
        let sr = f64::from(sane_sr(sample_rate));
        let w = 2.0 * PI * f64::from(freq_hz) / sr;
        let phi = 4.0 * (w * 0.5).sin().powi(2);
        let b0 = f64::from(self.b0);
        let b1 = f64::from(self.b1);
        let b2 = f64::from(self.b2);
        let a1 = f64::from(self.a1);
        let a2 = f64::from(self.a2);

        let num = (b0 + b1 + b2).powi(2) + (b0 * b2 * phi - (b1 * (b0 + b2) + 4.0 * b0 * b2)) * phi;
        let den = (1.0 + a1 + a2).powi(2) + (a2 * phi - (a1 * (1.0 + a2) + 4.0 * a2)) * phi;
        if den <= 1e-12 || num <= 1e-12 { 0.0 } else { (10.0 * (num / den).log10()) as f32 }
    }
}

/// Transposed Direct Form II biquad filter state for one audio channel.
#[derive(Debug, Clone, Copy, Default)]
pub struct Biquad {
    pub coeffs: Coeffs,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Create a new biquad filter with the given coefficients.
    pub const fn new(coeffs: Coeffs) -> Self {
        Self { coeffs, z1: 0.0, z2: 0.0 }
    }

    /// Clear internal delay state (silence history).
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Update filter coefficients without clearing state.
    pub fn update(&mut self, coeffs: Coeffs) {
        self.coeffs = coeffs;
    }

    /// Process a single audio sample.
    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.coeffs.b0 * x + self.z1;
        self.z1 = flush(self.coeffs.b1 * x - self.coeffs.a1 * y + self.z2);
        self.z2 = flush(self.coeffs.b2 * x - self.coeffs.a2 * y);
        y
    }

    /// In-place processing of an audio slice.
    pub fn process_slice(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}
