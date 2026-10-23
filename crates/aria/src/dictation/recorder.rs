//! The microphone (cpal: CoreAudio on macOS, ALSA/PipeWire on Linux): the default input device, mixed to 16 kHz mono as
//! it arrives. The stream lives on the dictation's thread (cpal streams cannot change threads on every platform).

use super::audio::Mixer;
use super::worker::{DictationError, Recorder};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct Microphone {
    stream: Option<cpal::Stream>,
    mixer: Option<Arc<Mutex<Mixer>>>,
}

fn failed(e: impl std::fmt::Display) -> DictationError {
    DictationError::Failed(e.to_string())
}

impl Recorder for Microphone {
    fn start(&mut self) -> Result<(), DictationError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(DictationError::NoMicrophone)?;
        let config = device.default_input_config().map_err(failed)?;
        let mixer = Arc::new(Mutex::new(Mixer::new(config.sample_rate().0, config.channels())));
        let into = mixer.clone();
        let on_error = |e| crate::note(&format!("the microphone stopped: {e}"));
        let stream_config: cpal::StreamConfig = config.clone().into();
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &_| into.lock().unwrap_or_else(|e| e.into_inner()).push(data),
                on_error,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &_| {
                    let samples: Vec<f32> = data.iter().map(|s| *s as f32 / 32768.0).collect();
                    into.lock().unwrap_or_else(|e| e.into_inner()).push(&samples)
                },
                on_error,
                None,
            ),
            other => return Err(failed(format!("the microphone gives {other:?} samples"))),
        }
        .map_err(failed)?;
        stream.play().map_err(failed)?;
        self.stream = Some(stream);
        self.mixer = Some(mixer);
        Ok(())
    }

    fn seconds(&self) -> f64 {
        self.mixer.as_ref().map_or(0.0, |m| m.lock().unwrap_or_else(|e| e.into_inner()).seconds())
    }

    fn finish(&mut self) -> Vec<f32> {
        drop(self.stream.take()); // the stream stops when dropped
        let Some(mixer) = self.mixer.take() else { return Vec::new() };
        match Arc::try_unwrap(mixer) {
            Ok(m) => m.into_inner().unwrap_or_else(|e| e.into_inner()).take(),
            Err(shared) => {
                std::mem::replace(&mut *shared.lock().unwrap_or_else(|e| e.into_inner()), Mixer::new(16_000, 1)).take()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real microphone for a few seconds: `cargo test -p aria records_the_microphone -- --ignored --nocapture`
    /// (with `ARIA_DICTATION_MODEL` set, what was said is transcribed too).
    #[test]
    #[ignore]
    fn records_the_microphone() {
        let mut mic = Microphone::default();
        mic.start().unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
        let seconds = mic.seconds();
        let audio = mic.finish();
        let loudest = audio.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        eprintln!("{seconds:.2} s, {} samples, loudest {loudest:.4}", audio.len());
        assert!((4.5..5.6).contains(&seconds), "{seconds}");
        assert!(loudest > 0.0, "exact silence: the microphone is not allowed");
        if let Ok(model) = std::env::var("ARIA_DICTATION_MODEL") {
            use super::super::worker::Transcriber;
            let mut whisper = super::super::whisper::Whisper::new(model.into());
            whisper.load().unwrap();
            eprintln!("heard: {:?}", whisper.transcribe(&audio, "es").unwrap());
        }
    }
}
