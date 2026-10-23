//! Dictation: the owner speaks, Whisper (large-v3-turbo q5_0, on this computer) writes what was said, and the page puts
//! it in the chat's input. `audio` turns the microphone's samples into what Whisper reads (16 kHz mono), `model` is the
//! voice model's file (downloaded once, when the owner agrees).

pub mod audio;
pub mod model;
pub mod recorder;
pub mod whisper;
pub mod worker;

pub use worker::{Dictation, DictationError, Event, EventKind, Recorder, State, Transcriber};

/// The dock's dictation: the model in `<aria home>/models/`, the default microphone, Whisper; `on` hears every event.
pub fn for_dock(paths: &aria_core::paths::AriaPaths, on: Box<dyn Fn(Event) + Send>) -> Dictation {
    let model = model::path(paths);
    let (ready, dest, load) = (model.clone(), model.clone(), model);
    Dictation::new(
        worker::Parts {
            model_ready: Box::new(move || model::present(&ready, model::SIZE)),
            download: Box::new(move |progress| {
                model::download(model::URL, &dest, model::SIZE, model::SHA256, progress)
            }),
            recorder: Box::new(|| Box::new(recorder::Microphone::default())),
            transcriber: Box::new(move || Box::new(whisper::Whisper::new(load))),
            on,
        },
        worker::Limits::default(),
    )
}
