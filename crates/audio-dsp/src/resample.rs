//! Streaming sample-rate conversion for clock-drift correction.
//!
//! This is the **last-resort** correction stage. Engines should default to a 1:1
//! passthrough and derive timeline positions from timestamps
//! (`craft-audio-clock::TimestampTracker`); engage this resampler only when a measured,
//! sustained clock deviation would otherwise overrun bounded buffers.
//!
//! Converts planar audio between arbitrary sample rates using continuous rational
//! phase tracking, preventing phase jumps and discontinuities across block boundaries.
//! The source position of output frame `n` is `(n * from + corr) / to`, where the integer
//! phase correction `corr` is recomputed on every [`set_rates`] call so that the source
//! position stays **continuous** when the ratio changes mid-stream (exactly what a
//! delay-locked loop driving crystal-drift correction needs). Rate changes therefore never
//! cause a jump; the stream simply advances at the new ratio from the current position.

/// Reusable planar scratch buffer (`scratch[channel][frame]`) passed to fetch callbacks.
pub type PlanarBuffer = Vec<Vec<f32>>;

/// Streaming planar sample-rate converter with phase-continuous rate updates.
#[derive(Debug, Clone)]
pub struct StreamResampler {
    from: u32,
    to: u32,
    /// Integer phase correction (in units of `1/to` source frames) preserving source-position
    /// continuity across [`set_rates`](StreamResampler::set_rates) changes.
    corr: i128,
    held_start: i64,
    held: Vec<Vec<f32>>,
    next_out: Option<i64>,
    scratch: Vec<Vec<f32>>,
}

/// Nearest-integer Euclidean division (`den > 0`).
fn div_round(num: i128, den: i128) -> i128 {
    let q = num.div_euclid(den);
    let r = num.rem_euclid(den);
    if 2 * r >= den { q + 1 } else { q }
}

impl StreamResampler {
    /// Create a resampler between `from_rate` and `to_rate`.
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        Self { from: from_rate.max(1), to: to_rate.max(1), corr: 0, held_start: 0, held: Vec::new(), next_out: None, scratch: Vec::new() }
    }

    /// Current `(from, to)` rates.
    pub fn rates(&self) -> (u32, u32) {
        (self.from, self.to)
    }

    /// Update resampling rates dynamically (e.g. from a DLL clock tracker).
    ///
    /// The fractional source position at the next output frame is preserved across the
    /// change (to within `< 1/to` of a sample), so the rendered waveform stays continuous:
    /// no jump, no re-anchor, no buffer reset. Subsequent output frames advance at the new
    /// ratio, gradually inserting or dropping source samples to track the new rate.
    pub fn set_rates(&mut self, from_rate: u32, to_rate: u32) {
        let new_from = from_rate.max(1) as i128;
        let new_to = to_rate.max(1) as i128;
        if let Some(n) = self.next_out {
            // Source position numerator at `n` under the old ratio, in old `to` units.
            let old_to = self.to.max(1) as i128;
            let old_num = n as i128 * self.from.max(1) as i128 + self.corr;
            // Solve n*new_from + corr_new ~= old_num * new_to / old_to for corr_new.
            self.corr = div_round(old_num * new_to, old_to) - n as i128 * new_from;
        } else {
            self.corr = 0;
        }
        self.from = from_rate.max(1);
        self.to = to_rate.max(1);
    }

    /// Reset internal state (also clears the phase correction and any stream anchor).
    pub fn reset(&mut self) {
        self.held.clear();
        self.held_start = 0;
        self.next_out = None;
        self.corr = 0;
    }

    /// Map output frame index to fractional source position: `(source_frame, frac)`.
    fn source_index(&self, out_frame: i64) -> (i64, f32) {
        let p = out_frame as i128 * self.from as i128 + self.corr;
        let to = self.to as i128;
        (p.div_euclid(to) as i64, (p.rem_euclid(to) as f64 / to as f64) as f32)
    }

    /// Re-anchor if needed, pull missing source frames through `fetch`, and prune history.
    /// `fetch(start, count, scratch)` must fill `scratch`'s per-channel buffers with `count`
    /// source frames starting at absolute frame `start`.
    fn sync_held(&mut self, out_frame: i64, n: usize, fetch: &mut dyn FnMut(i64, usize, &mut PlanarBuffer)) {
        let (first, _) = self.source_index(out_frame);
        let (last, _) = self.source_index(out_frame.saturating_add(n as i64 - 1));

        // Non-consecutive out_frame (or first call) is a seek: re-anchor from scratch.
        if self.next_out != Some(out_frame) || self.held.is_empty() {
            self.held_start = first - 1;
            for ch in &mut self.held {
                ch.clear();
            }
        }

        let held_end = self.held_start + self.held.first().map_or(0, Vec::len) as i64;
        // Interpolation of the last output frame reads source frame `last + 1`, so the held
        // range must cover `first - 1 ..= last + 1` (exclusive end `last + 2`).
        if last + 2 > held_end {
            let want = (last + 2 - held_end) as usize;
            let mut scratch = std::mem::take(&mut self.scratch);
            fetch(held_end, want, &mut scratch);
            if self.held.is_empty() {
                let nch = scratch.len().max(1);
                self.held = vec![Vec::new(); nch];
            }
            for (dest, src) in self.held.iter_mut().zip(scratch.iter()) {
                dest.extend_from_slice(src);
            }
            self.scratch = scratch;
        }

        // Prune consumed history, keeping one sample of interpolation context.
        let keep_from = (first - 1).saturating_sub(self.held_start).max(0) as usize;
        for ch in &mut self.held {
            if keep_from < ch.len() {
                ch.drain(..keep_from);
            }
        }
        self.held_start += keep_from as i64;
        self.next_out = Some(out_frame.saturating_add(n as i64));
    }

    /// Linear-interpolate `n` frames from `out_frame` into `write(channel, frame, sample)`.
    fn render_into(&self, out_frame: i64, n: usize, mut write: impl FnMut(usize, usize, f32)) {
        for i in 0..n {
            let f = out_frame.saturating_add(i as i64);
            let (src_idx, frac) = self.source_index(f);
            let local = (src_idx - self.held_start).max(0) as usize;
            for (ch, buf) in self.held.iter().enumerate() {
                let s0 = buf.get(local).copied().unwrap_or(0.0);
                let s1 = buf.get(local + 1).copied().unwrap_or(s0);
                write(ch, i, s0 + (s1 - s0) * frac);
            }
        }
    }

    /// Real-time processing path with **zero steady-state allocations**: renders `n` frames
    /// (the minimum length across `outs`) into the caller's planar buffers, pulling source
    /// audio through `fetch(start, count, scratch)`. The `scratch` buffer passed to `fetch`
    /// is owned by the resampler and reused across calls; resize it in place (allocation-free
    /// once warm) and fill each channel with `count` frames. Returns the channel count.
    pub fn process_into(&mut self, out_frame: i64, outs: &mut [&mut [f32]], fetch: &mut dyn FnMut(i64, usize, &mut PlanarBuffer)) -> usize {
        let n = outs.iter().map(|o| o.len()).min().unwrap_or(0);
        if n == 0 || outs.is_empty() {
            return self.held.len().max(1);
        }
        self.sync_held(out_frame, n, fetch);
        self.render_into(out_frame, n, |ch, i, v| {
            if let Some(out) = outs.get_mut(ch)
                && let Some(slot) = out.get_mut(i)
            {
                *slot = v;
            }
        });
        self.held.len()
    }

    /// Convenience variant of [`process_into`](StreamResampler::process_into) that allocates
    /// the output (and the pull source) per call — fine for tests and offline rendering; use
    /// `process_into` on real-time threads. `pull(start, count)` returns planar source frames.
    pub fn process(&mut self, out_frame: i64, n: usize, mut pull: impl FnMut(i64, usize) -> Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if n == 0 {
            return vec![Vec::new(); self.held.len().max(1)];
        }
        let mut fetch = |start: i64, count: usize, scratch: &mut PlanarBuffer| {
            let got = pull(start, count);
            if scratch.len() < got.len() {
                scratch.resize(got.len(), Vec::new());
            }
            for (s, g) in scratch.iter_mut().zip(got.iter()) {
                s.clear();
                s.extend_from_slice(g);
            }
        };
        self.sync_held(out_frame, n, &mut fetch);
        let mut out = vec![vec![0.0f32; n]; self.held.len().max(1)];
        self.render_into(out_frame, n, |ch, i, v| {
            if let Some(buf) = out.get_mut(ch)
                && let Some(slot) = buf.get_mut(i)
            {
                *slot = v;
            }
        });
        out
    }
}
