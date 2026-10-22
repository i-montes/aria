# Dictation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A microphone button (and Alt+D) in the composer records the owner, Whisper (large-v3-turbo q5_0, on the
device) transcribes it when they stop, and the text lands in the input.

**Architecture:** `crates/aria/src/dictation/` holds the engine for both platforms: `audio.rs` (mono + 16 kHz, pure),
`model.rs` (the model file: download, verify, path), `recorder.rs` (cpal), `mod.rs` (`Dictation`: a worker thread that
records, loads/unloads Whisper and transcribes). The chat's `Sessions` routes the page's `dictation*` messages to the one
`Dictation` and its answers back to the tab. The page adds the button, keys and cards.

**Tech Stack:** Rust, whisper-rs 0.15 (`metal` on macOS), cpal 0.16, ureq + sha2 (already in aria-core), plain JS.

**Spec:** `docs/superpowers/specs/2026-10-22-dictation-design.md`

## Global Constraints

- Model: `ggml-large-v3-turbo-q5_0.bin`, 574041195 bytes, SHA-256 `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`, from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin`, stored in `<aria home>/models/`.
- Unload after 5 minutes idle; at most 5 minutes of audio per dictation; flash attention on; language from `settings.assistant.language` (`es`/`en`).
- Nothing is downloaded or recorded without the owner pressing the button (and accepting the download).
- Never sent by itself, except Enter while recording (stop, then send).
- Both platforms build: `whisper-rs` `metal` only on macOS; Linux builds CPU (its GPU backend is its own task later).
- English everywhere; conventional commits; no co-author lines.

## Review Focus

1. Stop pressed before the model finished loading → the transcription waits for the load, then runs (Task 2 `stop_waits_for_the_load`).
2. A tab closed while it records → the recording is cancelled, no text goes anywhere (Task 3 `a_closed_tab_cancels_its_dictation`).
3. No input device or permission denied → `error` state, the button back to idle (Task 2 `a_recorder_that_fails_says_why`).
4. A second tab starts while one records → `busy` to the second, the first goes on (Task 3 `one_dictation_at_a_time`).
5. Silence or a click → empty text: nothing inserted, no error (Task 4 sandbox check).

---

### Task 1: Audio and the model file

**Files:** Create `crates/aria/src/dictation/audio.rs`, `crates/aria/src/dictation/model.rs`, `crates/aria/src/dictation/mod.rs` (module skeleton); modify `crates/aria/src/main.rs` (`mod dictation;`), `crates/aria/Cargo.toml` (`sha2 = "0.10"`, `ureq` as aria-core has it).

**Produces:**
- `audio::Mixer::new(rate: u32, channels: u16) -> Mixer`; `push(&mut self, interleaved: &[f32])`; `take(self) -> Vec<f32>` (mono, 16 kHz, linear resampling, state kept across pushes).
- `model::MODEL_FILE`, `model::SIZE`, `model::SHA256`, `model::URL`; `model::path(paths: &AriaPaths) -> PathBuf`; `model::present(path) -> bool` (size matches); `model::download(url, dest, size, sha, progress: impl FnMut(f64)) -> Result<(), ModelError>` (`.part`, verify, rename; `ModelError::{Failed(String), Broken}`).

- [ ] **Step 1: Failing tests:** `mono_16k_passes_through`, `stereo_48k_becomes_mono_16k` (48000 frames of stereo (1.0, 0.0) → 16000 samples of 0.5 ±1e-6), `pushes_in_pieces_equal_one_push` (44.1 kHz in 7 uneven pieces = one push, ±1e-6); `a_complete_file_is_present`, `a_short_file_is_not`; `download_verifies_then_renames` (a `std::net::TcpListener` serving 1 KB; right hash → file there, `.part` gone, progress ends at 1.0); `a_wrong_hash_is_broken_and_leaves_nothing` (→ `Err(Broken)`, neither file); `a_failed_connection_is_failed`.
- [ ] **Step 2:** `cargo test -p aria dictation::` → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** PASS; clippy clean.
- [ ] **Step 5: Commit** `feat(dictation): audio to 16 kHz mono, and the voice model's file`

### Task 2: The dictation worker

**Files:** Create `crates/aria/src/dictation/recorder.rs`; modify `dictation/mod.rs`, `crates/aria/Cargo.toml` (`whisper-rs = "0.15"` with `features = ["metal"]` under macOS, plain under Linux; `cpal = "0.16"`).

**Produces:**
- `pub trait Recorder: Send { fn start(&mut self) -> Result<(), DictationError>; fn take(&mut self) -> Vec<f32>; }` (16 kHz mono so far); `recorder::Microphone` (cpal: default input, `Mixer`, `NoMicrophone` / `Denied` / `Failed` mapped from cpal errors).
- `pub enum State { Idle, Recording { seconds: u32 }, Transcribing, NeedsModel { size: u64 }, Downloading { progress: f64 }, Error(&'static str) }`.
- `pub struct Dictation`: `Dictation::new(model: PathBuf, recorder: Box<dyn FnMut() -> Box<dyn Recorder> + Send>, on: Box<dyn Fn(Event) + Send>) -> Dictation`, `start(&self, owner: String, language: String)`, `stop(&self)`, `cancel(&self)`, `download(&self, owner: String)`, `owner(&self) -> Option<String>`. `Event { owner: String, kind: EventKind }`, `EventKind::{State(State), Text(String)}`. Timers: `IDLE_UNLOAD = 300 s`, `MAX_AUDIO = 300 s` (tests set shorter through `Dictation::with_limits`).

- [ ] **Step 1: Failing tests** (fake recorder from a WAV of silence or the test sample; the real model only in `#[ignore]` tests): `start_without_the_model_needs_it` (`NeedsModel { size: 574041195 }`, nothing recorded), `cancel_drops_the_recording` (no `Text`), `stop_without_speech_gives_empty_text` (fake recorder, a stub transcriber returning ""), `stop_waits_for_the_load` (a stub transcriber whose load sleeps 200 ms: `Text` still arrives), `a_recorder_that_fails_says_why` (`Error("no-microphone")`, then `Idle`), `the_audio_cap_stops_by_itself`, `idle_unloads_the_model` (shortened timer: the stub sees `unload`), `#[ignore] transcribes_the_sample` (real model + `ARIA_DICTATION_SAMPLE` WAV: non-empty Spanish text).
  The transcriber is behind `trait Transcriber { fn load(&mut self) -> Result<(), DictationError>; fn transcribe(&mut self, audio: &[f32], language: &str) -> Result<String, DictationError>; fn unload(&mut self); }` with `WhisperTranscriber` (whisper-rs, flash attention, greedy, no timestamps) and test stubs.
- [ ] **Step 2–4:** FAIL → implement → PASS (`cargo test -p aria dictation::`), then the ignored one with the model from the test.
- [ ] **Step 5: Commit** `feat(dictation): record, transcribe with Whisper, unload when idle`

### Task 3: Messages and the dock

**Files:** Modify `crates/aria-core/src/bridge.rs` (page: `DictationStart`, `DictationStop`, `DictationCancel`, `DictationDownload`, all unit, `rename` camelCase; dock: `DictationState { state: String, seconds: Option<u32>, progress: Option<f64>, size: Option<u64>, error: Option<String> }`, `DictationText { text: String }`), `crates/aria/src/chat.rs` (`Sessions` owns one `Dictation`; a session's `dictation*` goes to it with the settings' language; its events go back wrapped in that session; `SessionClose` of the owner cancels), `crates/aria/build.rs` + `crates/aria/Info.plist` (macOS: `-sectcreate __TEXT __info_plist`, with `CFBundleIdentifier` `com.aria.dock` and `NSMicrophoneUsageDescription` "Aria listens only while you dictate, and the audio stays on this computer.").

- [ ] **Step 1: Failing tests:** bridge names round-trip; chat.rs `dictation_answers_the_tab_that_asked` (a fake `Dictation` factory), `one_dictation_at_a_time` (second tab → `DictationState { state: "error", error: "busy" }`), `a_closed_tab_cancels_its_dictation`.
- [ ] **Step 2–4:** FAIL → implement → PASS; `otool -s __TEXT __info_plist target/debug/aria` shows the plist on macOS.
- [ ] **Step 5: Commit** `feat(dictation): the dock dictates for the tab that asks`

### Task 4: The page

**Files:** Modify `ui/chat/parts/composer.js` (+ css, tests), `ui/chat/view.js` (state per active dictation, Alt+D, Esc/Enter while recording, insertion at the cursor, the download card, error notes), `ui/chat/texts.js`, `tools/ui-sandbox/check.ts` (and `serve.ts` stubs for the dictation messages).

**Produces:** composer props `dictation: { state, seconds }` → the mic button's look (`data-chat="dictate"`, `aria-pressed` while recording); texts `chat.dictation.*` in en/es.

- [ ] **Step 1: Failing tests:** composer renders the three looks and the seconds; view's `insertDictation(value, cursor, text) -> { value, cursor }` (pure: space rule, empty text unchanged).
- [ ] **Step 2–4:** FAIL → implement → PASS (`bun test ui`); sandbox checks: button → `dictationStart`; `needsModel` → the card; Download → `dictationDownload`; `recording` → red with seconds; Esc → `dictationCancel`; Alt+D → `dictationStop`; `dictationText` → text at the cursor, focus kept; empty text → nothing; Enter while recording → stop then send once the text is in; an error → a note, the button idle.
- [ ] **Step 5: Commit** `feat(chat): dictate with the microphone button`

### Task 5: Docs and live check

- [ ] README (the feature, the model download), architecture (`dictation/`), `docs/ideas.md` row 12 (first half done), `docs/pending.md` (Linux backend, permission after updates if seen).
- [ ] Live on the Mac with a throwaway `ARIA_HOME` (the model copied in, not downloaded again): the permission prompt, a Spanish and an English dictation, memory loaded and after unload (shortened by `ARIA_DICTATION_IDLE` if added for the check only — else wait 5 min).
- [ ] All suites; commit `docs: dictation`.
