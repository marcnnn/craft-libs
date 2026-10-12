//! Unified real-time audio DSP for Craft Apps.
//!
//! Provides the core digital signal processing building blocks shared across SoundCraft,
//! FilmCraft, and EffectCraft:
//!
//! - **Filtering:** [`biquad`] RBJ Audio EQ Cookbook IIR filters and [`eq`] parametric equalizer
//! - **Dynamics:** [`dynamics`] soft-knee compressor, look-ahead mastering limiter,
//!   zero-latency protection limiter, and noise gate
//! - **Spatial & Routing:** [`pan`] pan laws, stereo balance, ITU-R BS.775 downmix, surround matrices
//! - **Time & Modulation:** [`reverb`] algorithmic reverb and [`modulation`] chorus/flanger
//! - **Metering & Standards:** [`loudness`] ITU-R BS.1770 / EBU R128 loudness and [`meter`] peak/RMS
//! - **Conversion:** [`resample`] continuous rational streaming sample rate converter with
//!   phase-continuous dynamic rate updates
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
pub use dynamics::{Compressor, MasteringLimiter, ProtectionLimiter, compressor_gain_reduction, expander_gain_reduction};
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
    fn test_mastering_limiter_ceiling() {
        let mut limiter = MasteringLimiter::new(-1.0, 5.0, 50.0, 48000.0);
        assert_eq!(limiter.latency_samples(), 240); // 5 ms at 48 kHz, announced latency
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
    fn test_protection_limiter_zero_latency() {
        let mut limiter = ProtectionLimiter::new(-1.0, 50.0, 48000.0);
        assert_eq!(limiter.latency_samples(), 0); // monitoring never pays latency for limiting
        let max_amp = db_to_gain(-1.0);
        // The very first excessive sample is already caught: instant attack, no delay line.
        let mut l = 2.0;
        let mut r = -2.0;
        limiter.process_stereo(&mut l, &mut r);
        assert!((l - max_amp).abs() < 1e-5);
        assert!((r + max_amp).abs() < 1e-5);
        for _ in 0..1000 {
            limiter.process_stereo(&mut l, &mut r);
            assert!(l <= max_amp + 1e-4 && l >= -max_amp - 1e-4);
            assert!(r <= max_amp + 1e-4 && r >= -max_amp - 1e-4);
        }
        // Release returns toward unity after the excess disappears.
        let mut ql = 0.1f32;
        let mut qr = 0.1f32;
        let mut gr = 0.0f32;
        for _ in 0..4800 {
            // 100 ms of quiet input
            gr = limiter.process_stereo(&mut ql, &mut qr);
        }
        assert!(gr > -0.5, "gain should recover, got {gr} dB");
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
        meter.process(&channels);
        assert_eq!(meter.momentary_lufs(), -70.0);
        assert_eq!(meter.short_term_lufs(), -70.0);
        assert_eq!(meter.integrated_lufs(), -70.0);
    }

    #[test]
    fn test_loudness_1khz_calibration() {
        // The BS.1770 conformance anchor: a stereo 1 kHz sine at -23 dBFS reads -23.0 LUFS,
        // at any sample rate (K-weighting is designed per rate).
        for sr in [48000.0f64, 44100.0] {
            let mut meter = LoudnessMeter::new_stereo(sr as f32);
            let amp = 10f64.powf(-23.0 / 20.0);
            let block_len = 1024usize;
            let blocks = (3.0 * sr / block_len as f64).ceil() as usize;
            for b in 0..blocks {
                let block = tone(b * block_len, block_len, 1000.0, amp, sr);
                let channels = [&block[..], &block[..]];
                meter.process(&channels);
            }
            assert!((meter.momentary_lufs() - -23.0).abs() < 0.15, "momentary at {sr}: {}", meter.momentary_lufs());
            assert!((meter.short_term_lufs() - -23.0).abs() < 0.15, "short-term at {sr}: {}", meter.short_term_lufs());
            assert!((meter.integrated_lufs() - -23.0).abs() < 0.15, "integrated at {sr}: {}", meter.integrated_lufs());
        }
    }

    #[test]
    fn test_loudness_gating() {
        let sr = 48000.0f64;
        // Absolute gate: 1 s of -20 dBFS tone followed by 3 s of silence. Silent blocks are
        // excluded; boundary blocks dilute the mean to about -21.2, but far above the -22.9
        // that counting the silence would give.
        let mut meter = LoudnessMeter::new_stereo(sr as f32);
        let amp = 10f64.powf(-20.0 / 20.0);
        let loud_blocks = (1.0 * sr / 1024.0).ceil() as usize;
        for b in 0..loud_blocks {
            let block = tone(b * 1024, 1024, 1000.0, amp, sr);
            let channels = [&block[..], &block[..]];
            meter.process(&channels);
        }
        for _ in 0..(3.0 * sr / 1024.0).ceil() as usize {
            let block = vec![0.0f32; 1024];
            let channels = [&block[..], &block[..]];
            meter.process(&channels);
        }
        let integrated = meter.integrated_lufs();
        assert!(integrated > -21.7 && integrated < -20.5, "absolute-gated integrated: {integrated}");

        // Relative gate: 2 s of -16 dBFS then 2 s of -36 dBFS. Ungated mean is about -19.2;
        // the -10 LU gate excludes the quiet material, giving about -16.6.
        let mut meter = LoudnessMeter::new_stereo(sr as f32);
        for (level, secs) in [(-16.0f64, 2.0f64), (-36.0, 2.0)] {
            let amp = 10f64.powf(level / 20.0);
            let n_blocks = (secs * sr / 1024.0).ceil() as usize;
            for b in 0..n_blocks {
                let block = tone(b * 1024, 1024, 1000.0, amp, sr);
                let channels = [&block[..], &block[..]];
                meter.process(&channels);
            }
        }
        let integrated = meter.integrated_lufs();
        assert!(integrated > -17.3 && integrated < -15.7, "relative-gated integrated: {integrated}");
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
    fn test_resampler_rate_change_continuity() {
        fn ramp(start: i64, count: usize) -> Vec<Vec<f32>> {
            // Source frame `i` carries the value `i` (negative frames included: the
            // resampler reads one frame of interpolation context before the first output).
            vec![(start..start + count as i64).map(|i| i as f32).collect::<Vec<f32>>()]
        }
        let mut resampler = StreamResampler::new(48000, 48000);
        let out = resampler.process(0, 480, ramp);
        assert!((out[0][479] - 479.0).abs() < 1e-3);

        // Halve the output rate mid-stream: the source position must stay continuous.
        // Without re-anchoring, source_index(480) would jump to frame 960.
        resampler.set_rates(48000, 24000);
        let out = resampler.process(480, 4, ramp);
        assert!((out[0][0] - 480.0).abs() <= 1.0, "first frame after rate change: {}", out[0][0]);
        assert!((out[0][1] - 482.0).abs() <= 1.0, "second frame after rate change: {}", out[0][1]);
    }

    #[test]
    fn test_resampler_process_into() {
        // Zero-allocation real-time path: renders into caller buffers via a reused scratch.
        let mut resampler = StreamResampler::new(48000, 96000);
        let mut fetch = |start: i64, count: usize, scratch: &mut Vec<Vec<f32>>| {
            if scratch.is_empty() {
                scratch.push(Vec::new());
            }
            scratch[0].clear();
            scratch[0].extend((start..start + count as i64).map(|i| i as f32));
        };
        let mut out = vec![0.0f32; 100];
        let nch = resampler.process_into(0, &mut [&mut out], &mut fetch);
        assert_eq!(nch, 1);
        // 2x upsampling of the identity ramp: out[i] = i / 2.
        assert!((out[50] - 25.0).abs() < 1e-3);
        assert!((out[99] - 49.5).abs() < 1e-3);
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
    fn test_resampler_block_chaining_matches_whole_buffer() {
        // Regression: interpolation of the final frame of each block reads one source frame
        // beyond `floor(src)`; losing it made every block boundary deviate from a whole-buffer
        // render. Chained 64-frame blocks must now match a single 512-frame render exactly.
        fn ramp(start: i64, count: usize) -> Vec<Vec<f32>> {
            vec![(start..start + count as i64).map(|i| i as f32).collect::<Vec<f32>>()]
        }
        let mut whole = StreamResampler::new(48000, 44100);
        let expected = whole.process(0, 512, ramp);

        let mut chained = StreamResampler::new(48000, 44100);
        let mut got: Vec<f32> = Vec::new();
        let mut pos = 0i64;
        while got.len() < 512 {
            let out = chained.process(pos, 64, ramp);
            got.extend(out[0].iter().copied());
            pos += 64;
        }
        assert_eq!(got.len(), expected[0].len());
        for (i, (a, b)) in got.iter().zip(expected[0].iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "frame {i}: chained {a} vs whole-buffer {b}");
        }
    }

    #[test]
    fn test_fft_roundtrip() {
        let mut data = vec![fft::Complex::new(1.0, 0.0), fft::Complex::new(0.5, 0.0), fft::Complex::new(-0.5, 0.0), fft::Complex::new(0.0, 0.0)];
        let original = data.clone();
        assert!(fft::fft_in_place(&mut data));
        assert!(fft::ifft_in_place(&mut data));
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a.re - b.re).abs() < 1e-5);
            assert!((a.im - b.im).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fft_rejects_non_power_of_two() {
        let mut data = vec![fft::Complex::new(1.0, 0.0); 6];
        assert!(!fft::fft_in_place(&mut data));
        assert!(!fft::ifft_in_place(&mut data));
        assert_eq!(data[0].re, 1.0); // left untouched
    }

    /// `n` samples of a sine starting at absolute sample index `start`, computed in f64.
    fn tone(start: usize, n: usize, freq: f64, amp: f64, sample_rate: f64) -> Vec<f32> {
        use std::f64::consts::TAU;
        (0..n)
            .map(|i| {
                let t = (start + i) as f64 / sample_rate;
                (amp * (TAU * freq * t).sin()) as f32
            })
            .collect()
    }
}
