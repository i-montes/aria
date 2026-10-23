//! Dictation: the owner speaks, Whisper (large-v3-turbo q5_0, on this computer) writes what was said, and the page puts
//! it in the chat's input. `audio` turns the microphone's samples into what Whisper reads (16 kHz mono), `model` is the
//! voice model's file (downloaded once, when the owner agrees).

pub mod audio;
pub mod model;
