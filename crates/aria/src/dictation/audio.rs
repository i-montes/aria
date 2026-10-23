//! The microphone's samples as Whisper reads them: one channel at 16 kHz. Any rate and channel count come in, in pieces
//! of any size as the device delivers them; the channels are averaged and the rate is changed by linear interpolation.

/// Whisper's sample rate.
pub const RATE: u32 = 16_000;

pub struct Mixer {
    channels: usize,
    /// Input samples per output sample.
    step: f64,
    /// Where the next output sample falls, in input samples from the start of `pending`.
    at: f64,
    /// Mono input samples not yet used up (the last one is kept to interpolate against the next piece).
    pending: Vec<f32>,
    out: Vec<f32>,
}

impl Mixer {
    pub fn new(rate: u32, channels: u16) -> Mixer {
        Mixer {
            channels: channels.max(1) as usize,
            step: rate.max(1) as f64 / RATE as f64,
            at: 0.0,
            pending: Vec::new(),
            out: Vec::new(),
        }
    }

    /// A piece of interleaved frames (a trailing partial frame is dropped).
    pub fn push(&mut self, interleaved: &[f32]) {
        let n = self.channels as f32;
        self.pending.extend(interleaved.chunks_exact(self.channels).map(|frame| frame.iter().sum::<f32>() / n));
        while self.at + 1.0 < self.pending.len() as f64 {
            let i = self.at.floor() as usize;
            let frac = (self.at - i as f64) as f32;
            let (a, b) = (self.pending[i], self.pending[i + 1]);
            self.out.push(a + (b - a) * frac);
            self.at += self.step;
        }
        let used = (self.at.floor() as usize).min(self.pending.len());
        self.pending.drain(..used);
        self.at -= used as f64;
    }

    /// Everything so far, as 16 kHz mono.
    pub fn take(self) -> Vec<f32> {
        self.out
    }

    /// How many seconds of audio it holds.
    pub fn seconds(&self) -> f64 {
        self.out.len() as f64 / RATE as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_16k_passes_through() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 / 100.0).sin()).collect();
        let mut m = Mixer::new(16_000, 1);
        m.push(&input);
        let out = m.take();
        assert_eq!(out.len(), 999, "the last sample waits for the next one to interpolate against");
        assert!(out.iter().zip(&input).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn stereo_48k_becomes_mono_16k() {
        let mut m = Mixer::new(48_000, 2);
        let frames: Vec<f32> = (0..48_000).flat_map(|_| [1.0, 0.0]).collect();
        m.push(&frames);
        let out = m.take();
        assert_eq!(out.len(), 16_000);
        assert!(out.iter().all(|s| (s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn pushes_in_pieces_equal_one_push() {
        let input: Vec<f32> = (0..44_100).map(|i| (i as f32 / 37.0).sin()).collect();
        let mut whole = Mixer::new(44_100, 1);
        whole.push(&input);
        let mut pieces = Mixer::new(44_100, 1);
        let mut at: usize = 0;
        for size in [1usize, 999, 3, 10_000, 7, 20_000, usize::MAX] {
            let end = at.saturating_add(size).min(input.len());
            pieces.push(&input[at..end]);
            at = end;
        }
        let (a, b) = (whole.take(), pieces.take());
        assert!(a.len() > 15_990);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-6));
    }
}
