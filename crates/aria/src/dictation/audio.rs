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

/// The speech in a recording (16 kHz mono): the stretch from the first to the last frame loud enough to be a voice,
/// with a little margin, or `None` when that stretch is not longer than `MIN_SPEECH` (a click, a cough, silence:
/// Whisper given nothing to hear makes up words).
pub fn speech(samples: &[f32]) -> Option<&[f32]> {
    const FRAME: usize = RATE as usize / 50; // 20 ms
    let rms: Vec<f32> =
        samples.chunks(FRAME).map(|f| (f.iter().map(|s| s * s).sum::<f32>() / f.len() as f32).sqrt()).collect();
    if rms.is_empty() {
        return None;
    }
    // The room's own level (a quiet fifth of the frames), and a voice well above it.
    let mut sorted = rms.clone();
    sorted.sort_by(f32::total_cmp);
    let floor = sorted[sorted.len() / 5];
    let threshold = (floor * 4.0).max(0.01);
    let first = rms.iter().position(|r| *r > threshold)?;
    let last = rms.iter().rposition(|r| *r > threshold)?;
    if (last + 1 - first) * FRAME <= MIN_SPEECH {
        return None;
    }
    let start = first.saturating_sub(MARGIN) * FRAME;
    let end = ((last + 1 + MARGIN) * FRAME).min(samples.len());
    Some(&samples[start..end])
}

/// Speech must last longer than this (one second) to be transcribed.
pub const MIN_SPEECH: usize = RATE as usize;
/// Frames kept around the speech (200 ms each side), so the first and the last syllables are whole.
const MARGIN: usize = 10;

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(seconds: f64, level: f32) -> Vec<f32> {
        let mut x: u32 = 12345;
        (0..(seconds * RATE as f64) as usize)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32 - 0.5) * 2.0 * level
            })
            .collect()
    }

    fn voice(seconds: f64) -> Vec<f32> {
        (0..(seconds * RATE as f64) as usize).map(|i| (i as f32 * 0.07).sin() * 0.3).collect()
    }

    fn secs(samples: &[f32]) -> f64 {
        samples.len() as f64 / RATE as f64
    }

    #[test]
    fn silence_and_room_noise_have_no_speech() {
        assert!(speech(&vec![0.0; 48_000]).is_none());
        assert!(speech(&noise(3.0, 0.004)).is_none());
        assert!(speech(&[]).is_none());
    }

    #[test]
    fn a_second_or_less_of_voice_is_not_speech() {
        let mut audio = noise(1.0, 0.004);
        audio.extend(voice(0.8));
        audio.extend(noise(1.0, 0.004));
        assert!(speech(&audio).is_none());
    }

    #[test]
    fn more_than_a_second_is_kept_without_the_silence_around_it() {
        let mut audio = noise(2.0, 0.004);
        audio.extend(voice(1.5));
        audio.extend(noise(2.0, 0.004));
        let kept = speech(&audio).expect("a second and a half of voice");
        assert!((1.5..=2.0).contains(&secs(kept)), "{}", secs(kept));
    }

    #[test]
    fn a_pause_inside_the_speech_stays() {
        let mut audio = voice(0.8);
        audio.extend(noise(1.0, 0.004));
        audio.extend(voice(0.8));
        let kept = speech(&audio).expect("two bits of voice with a pause");
        assert!(secs(kept) >= 2.5, "{}", secs(kept));
    }

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
