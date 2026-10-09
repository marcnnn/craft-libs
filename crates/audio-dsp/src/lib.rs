//! Unified real-time audio DSP for Craft Apps.
//!
//! Provides the core digital signal processing building blocks shared across SoundCraft,
//! FilmCraft, and EffectCraft:
//!
//! - **Filtering:** [`biquad`] RBJ Audio EQ Cookbook IIR filters and [`eq`] parametric equalizer
//! - **Dynamics:** [`dynamics`] soft-knee compressor, look-ahead limiter, and noise gate
//! - **Spatial & Routing:** [`pan`] pan laws, stereo balance, ITU-R BS.775 downmix, surround matrices
//! - **Time & Modulation:** [`reverb`] algorithmic reverb and [`modulation`] chorus/flanger
//! - **Metering & Standards:** [`loudness`] ITU-R BS.1770 / EBU R128 loudness and [`meter`] peak/RMS
//! - **Conversion:** [`resample`] continuous rational streaming sample rate converter
//! - **Spectral:** [`fft`] dependency-free Radix-2 FFT and windowing functions
//! - **Utilities:** [`util`] denormal flushing, decibel conversion, delay lines, and [`smooth`] parameter smoothers

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod biquad;
pub mod dynamics;
pub mod eq;
pub mod fft;
pub mod loudness;
pub mod meter;
pub mod modulation;
pub mod pan;
pub mod resample;
pub mod reverb;
pub mod smooth;
pub mod util;

pub use biquad::{Biquad, Coeffs, FilterType};
pub use dynamics::{Compressor, Limiter, compressor_gain_reduction, expander_gain_reduction};
pub use eq::{EqBand, ParametricEq};
pub use loudness::{ChannelWeight, KWeightingFilter, LoudnessMeter};
pub use meter::{PeakMeter, RmsMeter};
pub use pan::{BS775_DOWNMIX_51_TO_STEREO, PanLaw, downmix_51_to_stereo, pan51_matrix, stereo_balance};
pub use resample::StreamResampler;
pub use reverb::Reverb;
pub use smooth::Smoother;
pub use util::{DelayLine, db_to_gain, flush, flush64, gain_to_db, time_coef};

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn test_db_conversions() {
        assert!((db_to_gain(0.0) - 1.0).abs() < 1e-6);
        assert!((gain_to_db(1.0) - 0.0).abs() < 1e-6);
        assert!((db_to_gain(-6.0206) - 0.5).abs() < 1e-3);
        assert!((gain_to_db(0.5) - -6.0206).abs() < 1e-3);
    }

    #[test]
    fn test_biquad_filtering() {
        let coeffs = Coeffs::design(FilterType::LowPass, 1000.0, FRAC_1_SQRT_2, 0.0, 48000.0);
        let mut filter = Biquad::new(coeffs);
        let mut buf = vec![0.0f32; 128];
        buf[0] = 1.0; // impulse
        filter.process_slice(&mut buf);
        assert!(buf[0] > 0.0);
    }

    #[test]
    fn test_compressor_soft_knee() {
        // Below threshold: no reduction
        let gr_under = compressor_gain_reduction(-30.0, -20.0, 4.0, 6.0);
        assert_eq!(gr_under, 0.0);

        // In knee: smooth transition
        let gr_knee = compressor_gain_reduction(-20.0, -20.0, 4.0, 6.0);
        assert!(gr_knee > 0.0);

        // Well above threshold
        let gr_over = compressor_gain_reduction(0.0, -20.0, 4.0, 6.0);
        assert!((gr_over - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_limiter_ceiling() {
        let mut limiter = Limiter::new(-1.0, 5.0, 50.0, 48000.0);
        let max_amp = db_to_gain(-1.0);
        for _ in 0..1000 {
            let mut l = 2.0; // excessive level
            let mut r = 2.0;
            limiter.process_stereo(&mut l, &mut r);
            assert!(l <= max_amp + 1e-4);
            assert!(r <= max_amp + 1e-4);
        }
    }

    #[test]
    fn test_parametric_eq() {
        let bands = vec![EqBand { kind: FilterType::Peak, freq: 1000.0, q: 2.0, gain_db: 6.0, enabled: true }];
        let eq = ParametricEq::new(bands, 48000.0);
        let mag_center = eq.composite_magnitude_db(1000.0);
        let mag_away = eq.composite_magnitude_db(100.0);
        assert!((mag_center - 6.0).abs() < 0.5);
        assert!(mag_away.abs() < 0.5);
    }

    #[test]
    fn test_pan_laws() {
        let (l, r) = PanLaw::ConstantPowerMinus3Db.gains(0.0);
        assert!((l - r).abs() < 1e-5);
        assert!((l * l + r * r - 1.0).abs() < 1e-5); // constant power sum
    }

    #[test]
    fn test_downmix_51() {
        let l = vec![1.0; 64];
        let r = vec![1.0; 64];
        let c = vec![0.0; 64];
        let lfe = vec![0.0; 64];
        let ls = vec![0.0; 64];
        let rs = vec![0.0; 64];
        let channels = [&l[..], &r[..], &c[..], &lfe[..], &ls[..], &rs[..]];
        let mut out_l = vec![0.0; 64];
        let mut out_r = vec![0.0; 64];
        downmix_51_to_stereo(&channels, &mut out_l, &mut out_r);
        assert_eq!(out_l[0], 1.0);
        assert_eq!(out_r[0], 1.0);
    }

    #[test]
    fn test_loudness_silence() {
        let mut meter = LoudnessMeter::new_stereo(48000.0);
        let silent = vec![0.0f32; 19200]; // 400ms at 48k
        let channels = [&silent[..], &silent[..]];
        let lufs = meter.momentary_lufs(&channels);
        assert_eq!(lufs, -70.0);
    }

    #[test]
    fn test_stream_resampler() {
        let mut resampler = StreamResampler::new(48000, 96000); // 2x upsampling
        let out = resampler.process(0, 100, |_start, count| vec![vec![1.0f32; count]]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 100);
        assert!((out[0][50] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_delay_line_interpolation() {
        let mut dl = DelayLine::new(100);
        dl.write(1.0);
        dl.write(2.0);
        let mid = dl.read_interpolated(0.5);
        assert!((mid - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_fft_roundtrip() {
        let mut data = vec![fft::Complex::new(1.0, 0.0), fft::Complex::new(0.5, 0.0), fft::Complex::new(-0.5, 0.0), fft::Complex::new(0.0, 0.0)];
        let original = data.clone();
        fft::fft_in_place(&mut data);
        fft::ifft_in_place(&mut data);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a.re - b.re).abs() < 1e-5);
            assert!((a.im - b.im).abs() < 1e-5);
        }
    }
}
