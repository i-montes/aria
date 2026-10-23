//! Whisper (whisper.cpp through `whisper-rs`; Metal on macOS): loaded when a dictation starts, let go when idle.

use super::worker::{DictationError, Transcriber};
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct Whisper {
    model: PathBuf,
    context: Option<WhisperContext>,
}

impl Whisper {
    pub fn new(model: PathBuf) -> Whisper {
        Whisper { model, context: None }
    }
}

fn failed(e: impl std::fmt::Display) -> DictationError {
    DictationError::Failed(e.to_string())
}

impl Transcriber for Whisper {
    fn loaded(&self) -> bool {
        self.context.is_some()
    }

    fn load(&mut self) -> Result<(), DictationError> {
        // whisper.cpp and ggml print every step of a load to stderr (the dock's log): their hooks go to `log`, which
        // Aria does not use, so the lines are dropped.
        static QUIET: std::sync::Once = std::sync::Once::new();
        QUIET.call_once(whisper_rs::install_logging_hooks);
        let mut params = WhisperContextParameters::default();
        params.flash_attn(true);
        let path = self.model.to_str().ok_or_else(|| failed("the model's path is not UTF-8"))?;
        self.context = Some(WhisperContext::new_with_params(path, params).map_err(failed)?);
        Ok(())
    }

    fn transcribe(&mut self, audio: &[f32], language: &str) -> Result<String, DictationError> {
        let context = self.context.as_ref().ok_or_else(|| failed("the model is not loaded"))?;
        if audio.len() < 1600 {
            return Ok(String::new()); // under a tenth of a second: nothing was said
        }
        let mut state = context.create_state().map_err(failed)?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(if language == "es" { "es" } else { "en" }));
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_no_timestamps(true);
        state.full(params, audio).map_err(failed)?;
        Ok(state.as_iter().map(|segment| segment.to_string()).collect::<String>().trim().to_string())
    }

    fn unload(&mut self) {
        self.context = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16 kHz mono 16-bit WAV's samples (its `data` chunk).
    fn wav(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        let mut at = 12;
        while at + 8 <= bytes.len() {
            let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
            if &bytes[at..at + 4] == b"data" {
                return bytes[at + 8..(at + 8 + size).min(bytes.len())]
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
            }
            at += 8 + size;
        }
        panic!("no data chunk in {path}");
    }

    /// The real model on a real recording:
    /// `ARIA_DICTATION_MODEL=<model> ARIA_DICTATION_SAMPLE=<16 kHz mono wav> cargo test -p aria transcribes -- --ignored`.
    #[test]
    #[ignore]
    fn transcribes_a_recording() {
        let (Ok(model), Ok(sample)) = (std::env::var("ARIA_DICTATION_MODEL"), std::env::var("ARIA_DICTATION_SAMPLE"))
        else {
            return;
        };
        let mut whisper = Whisper::new(model.into());
        whisper.load().unwrap();
        let started = std::time::Instant::now();
        let text = whisper.transcribe(&wav(&sample), "es").unwrap();
        eprintln!("{:.2} s: {text}", started.elapsed().as_secs_f64());
        assert!(text.to_lowercase().contains("grab"), "{text}");
        whisper.unload();
        assert!(!whisper.loaded());
    }
}
