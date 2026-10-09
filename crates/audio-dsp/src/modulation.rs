//! Modulation effects: Chorus, Flanger, and Tremolo.

use std::f32::consts::PI;

use crate::util::{DelayLine, sane_sr};

/// Multi-voice stereo chorus effect.
#[derive(Debug, Clone)]
pub struct Chorus {
    delay_l: DelayLine,
    delay_r: DelayLine,
    lfo_phase: f32,
    rate_hz: f32,
    depth_ms: f32,
    base_delay_ms: f32,
    feedback: f32,
    mix: f32,
    sample_rate: f32,
}

impl Chorus {
    /// Create a chorus effect.
    pub fn new(sample_rate: f32) -> Self {
        let sr = sane_sr(sample_rate);
        let max_samples = (0.05 * sr).ceil() as usize; // 50 ms max delay
        Self {
            delay_l: DelayLine::new(max_samples),
            delay_r: DelayLine::new(max_samples),
            lfo_phase: 0.0,
            rate_hz: 1.2,
            depth_ms: 3.5,
            base_delay_ms: 15.0,
            feedback: 0.2,
            mix: 0.5,
            sample_rate: sr,
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.delay_l.clear();
        self.delay_r.clear();
        self.lfo_phase = 0.0;
    }

    /// Update parameters: `rate_hz` [0.1, 10.0], `depth_ms` [0.5, 20.0], `mix` [0.0, 1.0].
    pub fn set_params(&mut self, rate_hz: f32, depth_ms: f32, mix: f32) {
        self.rate_hz = rate_hz.clamp(0.1, 10.0);
        self.depth_ms = depth_ms.clamp(0.1, 20.0);
        self.mix = mix.clamp(0.0, 1.0);
    }

    /// Process a stereo sample pair in-place.
    #[inline]
    pub fn process_stereo(&mut self, left: &mut f32, right: &mut f32) {
        let in_l = *left;
        let in_r = *right;

        let phase_inc = 2.0 * PI * self.rate_hz / self.sample_rate;
        self.lfo_phase = (self.lfo_phase + phase_inc) % (2.0 * PI);

        let mod_l = (self.lfo_phase.sin() * 0.5 + 0.5) * self.depth_ms;
        let mod_r = ((self.lfo_phase + PI * 0.5).sin() * 0.5 + 0.5) * self.depth_ms;

        let delay_samples_l = (self.base_delay_ms + mod_l) * 0.001 * self.sample_rate;
        let delay_samples_r = (self.base_delay_ms + mod_r) * 0.001 * self.sample_rate;

        let delayed_l = self.delay_l.read_interpolated(delay_samples_l);
        let delayed_r = self.delay_r.read_interpolated(delay_samples_r);

        self.delay_l.write(in_l + delayed_l * self.feedback);
        self.delay_r.write(in_r + delayed_r * self.feedback);

        *left = in_l * (1.0 - self.mix) + delayed_l * self.mix;
        *right = in_r * (1.0 - self.mix) + delayed_r * self.mix;
    }
}
