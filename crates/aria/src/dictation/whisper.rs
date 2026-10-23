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
        // Only the speech, and only when there is more than a second of it: given silence, Whisper makes up words.
        let Some(audio) = super::audio::speech(audio) else { return Ok(String::new()) };
        let mut state = context.create_state().map_err(failed)?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(if language == "es" { "es" } else { "en" }));
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_no_timestamps(true);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        // whisper-rs's default is 4 threads: on a CPU (Linux, no Metal) a recording takes several times longer than
        // with one per core. Half the logical CPUs is one per physical core on SMT machines, at most 8.
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        params.set_n_threads((cores / 2).clamp(4, 8) as i32);
        state.full(params, audio).map_err(failed)?;
        let segments: Vec<(String, f32)> =
            state.as_iter().map(|segment| (segment.to_string(), segment.no_speech_probability())).collect();
        Ok(clean(&segments))
    }

    fn unload(&mut self) {
        self.context = None;
    }
}

/// What Whisper heard, without what it makes up: segments it says are probably not speech, and the closing lines of
/// the videos it learned from ("Gracias por ver el video") when they are a whole segment.
pub fn clean(segments: &[(String, f32)]) -> String {
    segments
        .iter()
        .filter(|(text, no_speech)| *no_speech <= NO_SPEECH && !MADE_UP.contains(&plain(text).as_str()))
        .map(|(text, _)| text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Above this, Whisper itself thinks a segment is not speech.
const NO_SPEECH: f32 = 0.6;

/// The lines Whisper makes up from silence (the endings of the videos it learned from), as `plain` writes them.
const MADE_UP: &[&str] = &[
    "gracias por ver el video",
    "gracias por ver el video hasta la proxima",
    "gracias por ver",
    "suscribete",
    "suscribete al canal",
    "no olvides suscribirte",
    "subtitulos realizados por la comunidad de amaraorg",
    "subtitulos por la comunidad de amaraorg",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subtitles by the amaraorg community",
];

/// Lower case, no accents, letters and digits and single spaces only.
fn plain(text: &str) -> String {
    let folded: String = text
        .to_lowercase()
        .chars()
        .map(|c| match c {
            'á' | 'à' | 'ä' => 'a',
            'é' | 'è' | 'ë' => 'e',
            'í' | 'ì' | 'ï' => 'i',
            'ó' | 'ò' | 'ö' => 'o',
            'ú' | 'ù' | 'ü' => 'u',
            'ñ' => 'n',
            c if c.is_alphanumeric() || c.is_whitespace() => c,
            _ => ' ',
        })
        .collect();
    folded.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(text: &str, no_speech: f32) -> (String, f32) {
        (text.into(), no_speech)
    }

    #[test]
    fn made_up_closings_and_non_speech_go() {
        let heard = [
            seg(" Revisa el despliegue de Vercel.", 0.02),
            seg(" Gracias por ver el video.", 0.1),
            seg(" ¡Suscríbete!", 0.05),
            seg(" algo que no se oyó", 0.9),
            seg(" y avísame.", 0.03),
        ];
        assert_eq!(clean(&heard), "Revisa el despliegue de Vercel. y avísame.");
    }

    #[test]
    fn a_closing_said_inside_a_sentence_stays() {
        assert_eq!(
            clean(&[seg(" Dile gracias por ver el video a Ana.", 0.02)]),
            "Dile gracias por ver el video a Ana."
        );
    }

    #[test]
    fn english_closings_go_too() {
        assert_eq!(clean(&[seg(" Thanks for watching!", 0.1)]), "");
    }

    /// A 16 kHz mono 16-bit WAV's samples (its `data` chunk).
    fn wav(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        let mut at = 12;
        while at + 8 <= bytes.len() {
            let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
            if &bytes[at..at + 4] == b"data" {
                return bytes[at + 8..(at + 8 + size).min(bytes.len())]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
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
