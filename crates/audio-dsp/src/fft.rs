//! Dependency-free in-place Radix-2 Cooley-Tukey Fast Fourier Transform (FFT)
//! and standard spectral windowing functions.

use std::f32::consts::PI;
use std::ops::{Add, Mul, Sub};

/// Complex number representation for spectral processing.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn norm(self) -> f32 {
        self.norm_sq().sqrt()
    }
}

impl Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self { re: self.re * rhs.re - self.im * rhs.im, im: self.re * rhs.im + self.im * rhs.re }
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

/// Compute in-place Radix-2 FFT. `buffer.len()` must be a power of two.
///
/// Returns `true` when the transform was applied. A zero-length or non-power-of-two
/// buffer is left **untouched** and `false` is returned (this crate never panics, so an
/// unsupported length is reported rather than asserted).
pub fn fft_in_place(buffer: &mut [Complex]) -> bool {
    let n = buffer.len();
    if n == 0 || !n.is_power_of_two() {
        return false;
    }

    // Bit reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j {
            buffer.swap(i, j);
        }
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
    }

    // Cooley-Tukey butterflies
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / (len as f32);
        let w_step = Complex::new(angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buffer[i + k];
                let v = buffer[i + k + half] * w;
                buffer[i + k] = u + v;
                buffer[i + k + half] = u - v;
                w = w * w_step;
            }
            i += len;
        }
        len <<= 1;
    }
    true
}

/// Compute in-place Radix-2 Inverse FFT (normalized by 1/N). `buffer.len()` must be a
/// power of two. Returns `false` (buffer untouched) for unsupported lengths; see
/// [`fft_in_place`].
pub fn ifft_in_place(buffer: &mut [Complex]) -> bool {
    let n = buffer.len();
    if n == 0 || !n.is_power_of_two() {
        return false;
    }
    if n == 1 {
        return true;
    }

    // Conjugate inputs
    for c in buffer.iter_mut() {
        c.im = -c.im;
    }

    // Forward FFT
    fft_in_place(buffer);

    // Conjugate and normalize
    let scale = 1.0 / (n as f32);
    for c in buffer.iter_mut() {
        c.re *= scale;
        c.im = -c.im * scale;
    }
    true
}

/// Generate a Hann window of length `n`.
pub fn hann_window(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f32) / (n as f32 - 1.0)).cos())).collect()
}

/// Generate a Blackman window of length `n`.
pub fn blackman_window(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| {
            let a = 2.0 * PI * (i as f32) / (n as f32 - 1.0);
            0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos()
        })
        .collect()
}
