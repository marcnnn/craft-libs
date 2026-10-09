//! Utility DSP functions: denormal flushing, dB conversions, smoothing coefficients,
//! and circular delay lines.

/// Flush subnormal (denormal) 32-bit floats to 0.0 to avoid CPU performance penalties.
#[inline(always)]
pub fn flush(v: f32) -> f32 {
    if v.abs() < 1e-15 || !v.is_finite() { 0.0 } else { v }
}

/// Flush subnormal (denormal) 64-bit floats to 0.0.
#[inline(always)]
pub fn flush64(v: f64) -> f64 {
    if v.abs() < 1e-25 || !v.is_finite() { 0.0 } else { v }
}

/// Convert a gain in decibels to linear amplitude: `10^(db / 20)`.
#[inline]
pub fn db_to_gain(db: f32) -> f32 {
    if db <= -120.0 { 0.0 } else { 10.0f32.powf(db / 20.0) }
}

/// Convert linear amplitude to decibels: `20 * log10(gain)`.
#[inline]
pub fn gain_to_db(gain: f32) -> f32 {
    if gain <= 1e-6 { -120.0 } else { 20.0 * gain.log10() }
}

/// One-pole smoothing filter coefficient for a time constant in milliseconds:
/// `exp(-1.0 / (ms * 0.001 * sr))`. Returns 0.0 for non-positive time constants.
#[inline]
pub fn time_coef(ms: f32, sample_rate: f32) -> f32 {
    let sr = sane_sr(sample_rate);
    if ms <= 0.0 { 0.0 } else { (-1.0 / (ms * 0.001 * sr)).exp() }
}

/// Sanitize sample rate to a valid range [8000.0, 768000.0] Hz.
#[inline]
pub fn sane_sr(sr: f32) -> f32 {
    if sr.is_finite() { sr.clamp(8000.0, 768000.0) } else { 48000.0 }
}

/// Sanitize channel count to [1, 64].
#[inline]
pub fn sane_channels(channels: usize) -> usize {
    channels.clamp(1, 64)
}

/// Circular delay line for sample-accurate and fractional delay effects.
#[derive(Debug, Clone)]
pub struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl DelayLine {
    /// Create a delay line with a maximum capacity in samples.
    pub fn new(max_delay_samples: usize) -> Self {
        let capacity = max_delay_samples.max(1) + 1;
        Self { buffer: vec![0.0; capacity], write_pos: 0 }
    }

    /// Reset internal buffer to silence.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }

    /// Write an input sample into the delay line.
    #[inline]
    pub fn write(&mut self, sample: f32) {
        self.buffer[self.write_pos] = flush(sample);
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
    }

    /// Read an integer number of samples back in time.
    #[inline]
    pub fn read_exact(&self, delay_samples: usize) -> f32 {
        let len = self.buffer.len();
        let idx = (self.write_pos + len - (delay_samples % len)) % len;
        self.buffer[idx]
    }

    /// Read a fractional number of samples back in time using linear interpolation.
    #[inline]
    pub fn read_interpolated(&self, delay_samples: f32) -> f32 {
        let d = delay_samples.max(0.0);
        let d_int = d.floor() as usize;
        let frac = d - d_int as f32;

        let len = self.buffer.len();
        let idx0 = (self.write_pos + len - 1 - (d_int % len)) % len;
        let idx1 = (idx0 + len - 1) % len;

        let s0 = self.buffer[idx0];
        let s1 = self.buffer[idx1];
        s0 + (s1 - s0) * frac
    }

    /// Capacity of the delay line in samples.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len().saturating_sub(1)
    }
}
