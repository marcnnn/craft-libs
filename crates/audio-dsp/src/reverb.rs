//! Algorithmic stereo reverb: parallel comb filters with high-frequency damping
//! followed by series allpass diffuser stages.

use crate::util::{DelayLine, flush, sane_sr};

const NUM_COMBS: usize = 4;
const NUM_ALLPASS: usize = 2;

// Tuned prime delay lengths in samples at 44.1 kHz
const COMB_TUNING: [usize; NUM_COMBS] = [1116, 1188, 1277, 1356];
const ALLPASS_TUNING: [usize; NUM_ALLPASS] = [556, 441];

/// Stereo algorithmic reverb.
#[derive(Debug, Clone)]
pub struct Reverb {
    combs_l: Vec<DelayLine>,
    combs_r: Vec<DelayLine>,
    comb_filters_l: [f32; NUM_COMBS],
    comb_filters_r: [f32; NUM_COMBS],
    allpasses_l: Vec<DelayLine>,
    allpasses_r: Vec<DelayLine>,
    room_size: f32,
    damp: f32,
    wet: f32,
    dry: f32,
    width: f32,
}

impl Reverb {
    /// Create a reverb processor at the given sample rate.
    pub fn new(sample_rate: f32) -> Self {
        let sr_scale = sane_sr(sample_rate) / 44100.0;

        let mut combs_l = Vec::with_capacity(NUM_COMBS);
        let mut combs_r = Vec::with_capacity(NUM_COMBS);
        for &t in &COMB_TUNING {
            let samples = (t as f32 * sr_scale).round() as usize;
            combs_l.push(DelayLine::new(samples + 10));
            combs_r.push(DelayLine::new(samples + 33)); // slight stereo spread
        }

        let mut allpasses_l = Vec::with_capacity(NUM_ALLPASS);
        let mut allpasses_r = Vec::with_capacity(NUM_ALLPASS);
        for &t in &ALLPASS_TUNING {
            let samples = (t as f32 * sr_scale).round() as usize;
            allpasses_l.push(DelayLine::new(samples + 10));
            allpasses_r.push(DelayLine::new(samples + 27));
        }

        Self {
            combs_l,
            combs_r,
            comb_filters_l: [0.0; NUM_COMBS],
            comb_filters_r: [0.0; NUM_COMBS],
            allpasses_l,
            allpasses_r,
            room_size: 0.8,
            damp: 0.2,
            wet: 0.33,
            dry: 0.67,
            width: 1.0,
        }
    }

    /// Reset internal delay memory.
    pub fn reset(&mut self) {
        for c in &mut self.combs_l {
            c.clear();
        }
        for c in &mut self.combs_r {
            c.clear();
        }
        for a in &mut self.allpasses_l {
            a.clear();
        }
        for a in &mut self.allpasses_r {
            a.clear();
        }
        self.comb_filters_l.fill(0.0);
        self.comb_filters_r.fill(0.0);
    }

    /// Set parameters: `room_size` [0.0, 1.0], `damp` [0.0, 1.0], `wet` [0.0, 1.0], `dry` [0.0, 1.0].
    pub fn set_params(&mut self, room_size: f32, damp: f32, wet: f32, dry: f32) {
        self.room_size = room_size.clamp(0.0, 0.98);
        self.damp = damp.clamp(0.0, 1.0);
        self.wet = wet.clamp(0.0, 1.0);
        self.dry = dry.clamp(0.0, 1.0);
    }

    /// Process a stereo sample pair in-place.
    #[inline]
    pub fn process_stereo(&mut self, left: &mut f32, right: &mut f32) {
        let in_l = *left;
        let in_r = *right;
        let input = (in_l + in_r) * 0.015;

        let mut out_l = 0.0f32;
        let mut out_r = 0.0f32;

        // Parallel comb filters with one-pole damping
        for i in 0..NUM_COMBS {
            let cap_l = self.combs_l[i].capacity();
            let c_out_l = self.combs_l[i].read_exact(cap_l);
            self.comb_filters_l[i] = flush(c_out_l * (1.0 - self.damp) + self.comb_filters_l[i] * self.damp);
            self.combs_l[i].write(input + self.comb_filters_l[i] * self.room_size);
            out_l += c_out_l;

            let cap_r = self.combs_r[i].capacity();
            let c_out_r = self.combs_r[i].read_exact(cap_r);
            self.comb_filters_r[i] = flush(c_out_r * (1.0 - self.damp) + self.comb_filters_r[i] * self.damp);
            self.combs_r[i].write(input + self.comb_filters_r[i] * self.room_size);
            out_r += c_out_r;
        }

        // Series allpass diffusers
        for ap in &mut self.allpasses_l {
            let cap = ap.capacity();
            let buf_out = ap.read_exact(cap);
            let y = -out_l + buf_out;
            ap.write(flush(out_l + buf_out * 0.5));
            out_l = y;
        }

        for ap in &mut self.allpasses_r {
            let cap = ap.capacity();
            let buf_out = ap.read_exact(cap);
            let y = -out_r + buf_out;
            ap.write(flush(out_r + buf_out * 0.5));
            out_r = y;
        }

        let wet_l = out_l * (self.width * 0.5 + 0.5) + out_r * ((1.0 - self.width) * 0.5);
        let wet_r = out_r * (self.width * 0.5 + 0.5) + out_l * ((1.0 - self.width) * 0.5);

        *left = in_l * self.dry + wet_l * self.wet;
        *right = in_r * self.dry + wet_r * self.wet;
    }
}
