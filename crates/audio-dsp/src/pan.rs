//! Panning, balance, channel matrix remixing, and ITU-R BS.775 downmixing.

use std::f32::consts::{FRAC_1_SQRT_2, FRAC_PI_2, SQRT_2};

/// Pan laws governing the attenuation curve across the stereo field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PanLaw {
    /// -3.0 dB constant power law (centre = 0.7071 / -3 dB each side). Standard broadcast default.
    #[default]
    ConstantPowerMinus3Db,
    /// -4.5 dB compromise law.
    Minus45Db,
    /// -6.0 dB linear voltage law (centre = 0.5 / -6 dB each side).
    LinearMinus6Db,
    /// 0.0 dB (centre = 1.0, sides boost to +3 dB or unity).
    ZeroDb,
}

impl PanLaw {
    /// Compute (left, right) gains for a pan position in [-1.0, +1.0].
    /// -1.0 = hard left, 0.0 = center, +1.0 = hard right.
    pub fn gains(self, pan: f32) -> (f32, f32) {
        let p = pan.clamp(-1.0, 1.0);
        let pos = (p + 1.0) * 0.5; // [0.0, 1.0]

        match self {
            PanLaw::ConstantPowerMinus3Db => {
                let theta = pos * FRAC_PI_2;
                (theta.cos(), theta.sin())
            }
            PanLaw::Minus45Db => {
                let theta = pos * FRAC_PI_2;
                let cp_l = theta.cos();
                let cp_r = theta.sin();
                let lin_l = 1.0 - pos;
                let lin_r = pos;
                // Geometric mean between -3 dB and -6 dB gives -4.5 dB center
                ((cp_l * lin_l).sqrt(), (cp_r * lin_r).sqrt())
            }
            PanLaw::LinearMinus6Db => (1.0 - pos, pos),
            PanLaw::ZeroDb => {
                let theta = pos * FRAC_PI_2;
                (theta.cos() * SQRT_2, theta.sin() * SQRT_2)
            }
        }
    }
}

/// Balance control for an existing stereo track.
/// Center keeps both channels at 1.0; panning left attenuates right, and vice versa.
#[inline]
pub fn stereo_balance(balance: f32) -> (f32, f32) {
    let b = balance.clamp(-1.0, 1.0);
    if b < 0.0 { (1.0, 1.0 + b) } else { (1.0 - b, 1.0) }
}

/// ITU-R BS.775 standard coefficients for 5.1 surround to stereo downmix:
/// - Left = L + 0.7071 * C + 0.7071 * Ls
/// - Right = R + 0.7071 * C + 0.7071 * Rs
///
/// LFE is discarded by standard recommendation to prevent subwoofer overload.
pub const BS775_DOWNMIX_51_TO_STEREO: [(f32, f32); 6] = [
    (1.0, 0.0),                     // L
    (0.0, 1.0),                     // R
    (FRAC_1_SQRT_2, FRAC_1_SQRT_2), // C
    (0.0, 0.0),                     // LFE (discarded)
    (FRAC_1_SQRT_2, 0.0),           // Ls
    (0.0, FRAC_1_SQRT_2),           // Rs
];

/// Apply ITU-R BS.775 5.1 to stereo downmix in-place from planar 5.1 input to stereo output.
pub fn downmix_51_to_stereo(surround: &[&[f32]], out_l: &mut [f32], out_r: &mut [f32]) {
    let n = out_l.len().min(out_r.len());
    let get_ch = |idx: usize, i: usize| -> f32 { surround.get(idx).and_then(|c| c.get(i)).copied().unwrap_or(0.0) };

    for i in 0..n {
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        for (ch, &(gl, gr)) in BS775_DOWNMIX_51_TO_STEREO.iter().enumerate() {
            let s = get_ch(ch, i);
            l += s * gl;
            r += s * gr;
        }
        out_l[i] = l.clamp(-1.0, 1.0);
        out_r[i] = r.clamp(-1.0, 1.0);
    }
}

/// 5.1 surround speaker gains: `[L, R, C, LFE, Ls, Rs]`.
/// `x` in [-1.0, 1.0] (left to right), `y` in [-1.0, 1.0] (rear to front),
/// `center_pct` in [0.0, 1.0], `lfe_gain` in [0.0, 1.0].
pub fn pan51_matrix(x: f32, y: f32, center_pct: f32, lfe_gain: f32) -> [f32; 6] {
    let x = x.clamp(-1.0, 1.0);
    let y = y.clamp(-1.0, 1.0);
    let c_pct = center_pct.clamp(0.0, 1.0);
    let lfe = lfe_gain.clamp(0.0, 1.0);

    // Front / rear balance
    let front = (y + 1.0) * 0.5; // 0 = full rear, 1 = full front
    let rear = 1.0 - front;

    // Left / right balance
    let (front_l, front_r) = PanLaw::ConstantPowerMinus3Db.gains(x);
    let (rear_l, rear_r) = PanLaw::ConstantPowerMinus3Db.gains(x);

    let l = front_l * front * (1.0 - c_pct * 0.5);
    let r = front_r * front * (1.0 - c_pct * 0.5);
    let c = front * c_pct * ((front_l + front_r) * 0.5 * SQRT_2);
    let ls = rear_l * rear;
    let rs = rear_r * rear;

    [l, r, c, lfe, ls, rs]
}
