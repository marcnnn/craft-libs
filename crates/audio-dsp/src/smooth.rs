//! Parameter smoothing to prevent zipper noise and clicks during realtime parameter edits.

use crate::util::{flush, time_coef};

/// One-pole low-pass smoother for parameter values.
#[derive(Debug, Clone, Copy)]
pub struct Smoother {
    current: f32,
    target: f32,
    coef: f32,
}

impl Default for Smoother {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Smoother {
    /// Create a smoother initialized to `initial_value`.
    pub fn new(initial_value: f32) -> Self {
        Self { current: initial_value, target: initial_value, coef: 0.0 }
    }

    /// Set smoothing time constant in milliseconds at the given sample rate.
    pub fn set_time_ms(&mut self, ms: f32, sample_rate: f32) {
        self.coef = time_coef(ms, sample_rate);
    }

    /// Set a new target value.
    #[inline]
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Jump directly to `value`, resetting any in-flight smoothing.
    #[inline]
    pub fn snap_to(&mut self, value: f32) {
        self.current = value;
        self.target = value;
    }

    /// Advance one sample and return the smoothed value.
    #[inline(always)]
    pub fn advance(&mut self) -> f32 {
        if (self.current - self.target).abs() < 1e-6 {
            self.current = self.target;
        } else {
            self.current = flush(self.target + (self.current - self.target) * self.coef);
        }
        self.current
    }

    /// Current smoothed value.
    #[inline]
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Target value.
    #[inline]
    pub fn target(&self) -> f32 {
        self.target
    }

    /// Whether the smoother has reached its target within epsilon.
    #[inline]
    pub fn is_settled(&self) -> bool {
        (self.current - self.target).abs() < 1e-6
    }
}
