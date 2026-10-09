//! Streaming sample-rate conversion and adaptive clock drift compensation.
//!
//! Converts planar audio between arbitrary sample rates using continuous rational
//! phase tracking, preventing phase jumps and discontinuities across block boundaries.
//! Essential for bridging drifting hardware clocks with master PTP/TAI timelines.

/// Streaming planar sample-rate converter.
#[derive(Debug, Clone)]
pub struct StreamResampler {
    from: u32,
    to: u32,
    held_start: i64,
    held: Vec<Vec<f32>>,
    next_out: Option<i64>,
}

impl StreamResampler {
    /// Create a resampler between `from_rate` and `to_rate`.
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        Self { from: from_rate.max(1), to: to_rate.max(1), held_start: 0, held: Vec::new(), next_out: None }
    }

    /// Update resampling rates dynamically (e.g. For DLL clock tracking).
    pub fn set_rates(&mut self, from_rate: u32, to_rate: u32) {
        self.from = from_rate.max(1);
        self.to = to_rate.max(1);
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.held.clear();
        self.held_start = 0;
        self.next_out = None;
    }

    /// Map output frame index to fractional source position: `(source_frame, frac)`.
    fn source_index(&self, out_frame: i64) -> (i64, f32) {
        let p = out_frame as i128 * self.from as i128;
        let to = self.to as i128;
        (p.div_euclid(to) as i64, (p.rem_euclid(to) as f64 / to as f64) as f32)
    }

    /// Process a block of `n` output frames starting at `out_frame`.
    /// `pull(start, count)` is called when additional source frames are needed.
    pub fn process(&mut self, out_frame: i64, n: usize, mut pull: impl FnMut(i64, usize) -> Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if n == 0 {
            return vec![Vec::new(); self.held.len().max(1)];
        }

        let (first, _) = self.source_index(out_frame);
        let (last, _) = self.source_index(out_frame.saturating_add(n as i64 - 1));

        if self.next_out != Some(out_frame) || self.held.is_empty() {
            self.held_start = first - 1;
            self.held.clear();
        }

        let held_end = self.held_start + self.held.first().map_or(0, Vec::len) as i64;
        if last + 1 > held_end {
            let want = (last + 1 - held_end) as usize;
            let got = pull(held_end, want);
            if self.held.is_empty() {
                self.held = vec![Vec::new(); got.len().max(1)];
            }
            for (dest, src) in self.held.iter_mut().zip(got) {
                dest.extend(src);
            }
        }

        let nch = self.held.len().max(1);
        let mut out = vec![vec![0.0f32; n]; nch];

        for i in 0..n {
            let f = out_frame.saturating_add(i as i64);
            let (src_idx, frac) = self.source_index(f);
            let local_idx = (src_idx - self.held_start) as usize;

            for (ch, buf) in out.iter_mut().enumerate() {
                let h = self.held.get(ch);
                let s0 = h.and_then(|v| v.get(local_idx)).copied().unwrap_or(0.0);
                let s1 = h.and_then(|v| v.get(local_idx + 1)).copied().unwrap_or(s0);
                buf[i] = s0 + (s1 - s0) * frac;
            }
        }

        // Prune consumed history
        let keep_from = (first - 1).saturating_sub(self.held_start).max(0) as usize;
        for ch in &mut self.held {
            if keep_from < ch.len() {
                ch.drain(..keep_from);
            }
        }
        self.held_start += keep_from as i64;
        self.next_out = Some(out_frame.saturating_add(n as i64));

        out
    }
}
