//! Parametric Equalizer: multi-band IIR filter bank with sample-accurate parameter updating.

use std::f32::consts::FRAC_1_SQRT_2;

use crate::biquad::{Biquad, Coeffs, FilterType};

/// A single parametric EQ band specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqBand {
    pub kind: FilterType,
    pub freq: f32,
    pub q: f32,
    pub gain_db: f32,
    pub enabled: bool,
}

impl Default for EqBand {
    fn default() -> Self {
        Self { kind: FilterType::Peak, freq: 1000.0, q: FRAC_1_SQRT_2, gain_db: 0.0, enabled: true }
    }
}

/// Multi-band Parametric Equalizer for stereo or mono audio.
#[derive(Debug, Clone)]
pub struct ParametricEq {
    bands: Vec<EqBand>,
    filters_l: Vec<Biquad>,
    filters_r: Vec<Biquad>,
    sample_rate: f32,
}

impl ParametricEq {
    /// Create a parametric EQ with a set of default bands.
    pub fn new(bands: Vec<EqBand>, sample_rate: f32) -> Self {
        let n = bands.len();
        let mut filters_l = Vec::with_capacity(n);
        let mut filters_r = Vec::with_capacity(n);

        for band in &bands {
            let coeffs = if band.enabled { Coeffs::design(band.kind, band.freq, band.q, band.gain_db, sample_rate) } else { Coeffs::IDENTITY };
            filters_l.push(Biquad::new(coeffs));
            filters_r.push(Biquad::new(coeffs));
        }

        Self { bands, filters_l, filters_r, sample_rate }
    }

    /// Reset filter states.
    pub fn reset(&mut self) {
        for f in &mut self.filters_l {
            f.reset();
        }
        for f in &mut self.filters_r {
            f.reset();
        }
    }

    /// Update a specific band's settings.
    pub fn set_band(&mut self, index: usize, band: EqBand) {
        if index < self.bands.len() {
            self.bands[index] = band;
            let coeffs = if band.enabled { Coeffs::design(band.kind, band.freq, band.q, band.gain_db, self.sample_rate) } else { Coeffs::IDENTITY };
            self.filters_l[index].update(coeffs);
            self.filters_r[index].update(coeffs);
        }
    }

    /// Process a stereo sample pair in-place.
    #[inline]
    pub fn process_stereo(&mut self, left: &mut f32, right: &mut f32) {
        for (fl, fr) in self.filters_l.iter_mut().zip(self.filters_r.iter_mut()) {
            *left = fl.process(*left);
            *right = fr.process(*right);
        }
    }

    /// Compute composite magnitude response in dB across all enabled bands at `freq_hz`.
    pub fn composite_magnitude_db(&self, freq_hz: f32) -> f32 {
        let mut total_db = 0.0f32;
        for (band, fl) in self.bands.iter().zip(self.filters_l.iter()) {
            if band.enabled {
                total_db += fl.coeffs.magnitude_db(freq_hz, self.sample_rate);
            }
        }
        total_db
    }
}
