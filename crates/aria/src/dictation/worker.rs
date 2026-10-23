//! The dictation's thread: one dictation at a time for the whole dock. It records, makes sure the model is loaded,
//! transcribes when the owner stops, and lets the model go after a while without dictating.

use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What went wrong, in the page's words (`dictationState { error }`).
#[derive(Debug, Clone, PartialEq)]
pub enum DictationError {
    NoMicrophone,
    Denied,
    Busy,
    Failed(String),
}

/// A source of 16 kHz mono audio.
pub trait Recorder {
    fn start(&mut self) -> Result<(), DictationError>;
    /// Seconds recorded so far.
    fn seconds(&self) -> f64;
    /// Stops and gives everything recorded.
    fn finish(&mut self) -> Vec<f32>;
}

/// Whisper, or a stand-in in tests.
pub trait Transcriber {
    fn loaded(&self) -> bool;
    fn load(&mut self) -> Result<(), DictationError>;
    fn transcribe(&mut self, audio: &[f32], language: &str) -> Result<String, DictationError>;
    fn unload(&mut self);
}

#[derive(Debug, Clone, PartialEq)]
pub enum State {
    Idle,
    Recording { seconds: u32 },
    Transcribing,
    NeedsModel { size: u64 },
    Downloading { progress: f64 },
    Error(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EventKind {
    State(State),
    Text(String),
}

/// Something the tab `owner` should hear.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub owner: String,
    pub kind: EventKind,
}

impl DictationError {
    pub fn code(&self) -> &'static str {
        match self {
            DictationError::NoMicrophone => "no-microphone",
            DictationError::Denied => "microphone-denied",
            DictationError::Busy => "busy",
            DictationError::Failed(_) => "failed",
        }
    }
}

enum Command {
    Start { owner: String, language: String },
    Stop,
    Cancel,
    Download { owner: String, language: String },
}

/// The dock's one dictation: commands go to its thread, events come back through `Parts::on`.
pub struct Dictation {
    tx: Sender<Command>,
    owner: Arc<Mutex<Option<String>>>,
}

/// What the dictation is made of (the real ones in `dictation::for_dock`, stand-ins in tests).
/// Downloads the model, telling the share done.
pub type Download = Box<dyn FnMut(&mut dyn FnMut(f64)) -> Result<(), crate::dictation::model::ModelError> + Send>;

pub struct Parts {
    pub model_ready: Box<dyn Fn() -> bool + Send>,
    pub download: Download,
    pub recorder: Box<dyn FnMut() -> Box<dyn Recorder> + Send>,
    pub transcriber: Box<dyn FnOnce() -> Box<dyn Transcriber> + Send>,
    pub on: Box<dyn Fn(Event) + Send>,
}

pub struct Limits {
    pub idle_unload: Duration,
    pub max_audio: Duration,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits { idle_unload: Duration::from_secs(300), max_audio: Duration::from_secs(300) }
    }
}

impl Dictation {
    pub fn new(parts: Parts, limits: Limits) -> Dictation {
        let (tx, rx) = channel();
        let owner = Arc::new(Mutex::new(None));
        let shared = owner.clone();
        std::thread::Builder::new()
            .name("dictation".into())
            .spawn(move || {
                let mut worker = Worker::new(parts, limits, shared);
                loop {
                    match rx.recv_timeout(TICK) {
                        Ok(command) => worker.handle(command),
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                    worker.tick();
                }
            })
            .expect("the dictation thread starts");
        Dictation { tx, owner }
    }

    /// Starts recording for the tab `owner` (or says why it cannot).
    pub fn start(&self, owner: &str, language: &str) {
        let _ = self.tx.send(Command::Start { owner: owner.into(), language: language.into() });
    }
    /// Ends the recording and transcribes it.
    pub fn stop(&self) {
        let _ = self.tx.send(Command::Stop);
    }
    /// Ends the recording and drops it.
    pub fn cancel(&self) {
        let _ = self.tx.send(Command::Cancel);
    }
    /// Downloads the model (the owner agreed), then starts recording.
    pub fn download(&self, owner: &str, language: &str) {
        let _ = self.tx.send(Command::Download { owner: owner.into(), language: language.into() });
    }
    /// The tab the dictation under way belongs to.
    pub fn owner(&self) -> Option<String> {
        self.owner.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// How often the thread looks at the clock (the seconds shown, the audio cap, the unload).
const TICK: Duration = Duration::from_millis(100);

struct Recording {
    owner: String,
    language: String,
    recorder: Box<dyn Recorder>,
    shown: u32,
}

struct Worker {
    parts: Parts,
    limits: Limits,
    owner: Arc<Mutex<Option<String>>>,
    whisper: Option<Box<dyn Transcriber>>,
    recording: Option<Recording>,
    last_used: Instant,
}

impl Worker {
    fn new(parts: Parts, limits: Limits, owner: Arc<Mutex<Option<String>>>) -> Worker {
        Worker { parts, limits, owner, whisper: None, recording: None, last_used: Instant::now() }
    }

    fn tell(&self, owner: &str, kind: EventKind) {
        (self.parts.on)(Event { owner: owner.into(), kind });
    }

    fn state(&self, owner: &str, state: State) {
        self.tell(owner, EventKind::State(state));
    }

    fn set_owner(&self, owner: Option<&str>) {
        *self.owner.lock().unwrap_or_else(|e| e.into_inner()) = owner.map(str::to_string);
    }

    fn fail(&self, owner: &str, error: &DictationError) {
        if let DictationError::Failed(why) = error {
            crate::note(&format!("dictation failed: {why}"));
        }
        self.state(owner, State::Error(error.code()));
        self.state(owner, State::Idle);
        self.set_owner(None);
    }

    fn handle(&mut self, command: Command) {
        match command {
            Command::Start { owner, language } => self.start(owner, language),
            Command::Stop => self.stop(),
            Command::Cancel => {
                if let Some(mut r) = self.recording.take() {
                    drop(r.recorder.finish());
                    self.state(&r.owner, State::Idle);
                    self.set_owner(None);
                }
            }
            Command::Download { owner, language } => {
                if self.recording.is_some() {
                    return self.state(&owner, State::Error(DictationError::Busy.code()));
                }
                self.set_owner(Some(&owner));
                let mut told = -1.0;
                let mut progress = |p: f64| {
                    if p - told >= 0.01 || p >= 1.0 {
                        told = p;
                        (self.parts.on)(Event {
                            owner: owner.clone(),
                            kind: EventKind::State(State::Downloading { progress: p }),
                        });
                    }
                };
                let result = (self.parts.download)(&mut progress);
                match result {
                    Ok(()) => self.start(owner, language),
                    Err(crate::dictation::model::ModelError::Broken) => {
                        self.state(&owner, State::Error("model-broken"));
                        self.state(&owner, State::Idle);
                        self.set_owner(None);
                    }
                    Err(crate::dictation::model::ModelError::Failed(why)) => {
                        crate::note(&format!("the voice model could not be downloaded: {why}"));
                        self.state(&owner, State::Error("download-failed"));
                        self.state(&owner, State::Idle);
                        self.set_owner(None);
                    }
                }
            }
        }
    }

    fn start(&mut self, owner: String, language: String) {
        if let Some(r) = &self.recording {
            if r.owner != owner {
                self.state(&owner, State::Error(DictationError::Busy.code()));
            }
            return;
        }
        if !(self.parts.model_ready)() {
            return self.state(&owner, State::NeedsModel { size: crate::dictation::model::SIZE });
        }
        let mut recorder = (self.parts.recorder)();
        if let Err(e) = recorder.start() {
            return self.fail(&owner, &e);
        }
        self.set_owner(Some(&owner));
        self.state(&owner, State::Recording { seconds: 0 });
        self.recording = Some(Recording { owner, language, recorder, shown: 0 });
        // The model loads while the owner speaks (the recorder runs on its own).
        if let Err(e) = self.load() {
            if let Some(mut r) = self.recording.take() {
                drop(r.recorder.finish());
                self.fail(&r.owner, &e);
            }
        }
    }

    fn load(&mut self) -> Result<(), DictationError> {
        if self.whisper.is_none() {
            let make = std::mem::replace(&mut self.parts.transcriber, Box::new(|| unreachable!("made once")));
            self.whisper = Some(make());
        }
        let whisper = self.whisper.as_mut().unwrap();
        if !whisper.loaded() {
            whisper.load()?;
        }
        Ok(())
    }

    fn stop(&mut self) {
        let Some(mut r) = self.recording.take() else { return };
        let audio = r.recorder.finish();
        // A microphone macOS was not allowed to use gives exact zeros.
        if !audio.is_empty() && audio.iter().all(|s| *s == 0.0) {
            return self.fail(&r.owner, &DictationError::Denied);
        }
        self.state(&r.owner, State::Transcribing);
        let result = self.load().and_then(|()| self.whisper.as_mut().unwrap().transcribe(&audio, &r.language));
        self.last_used = Instant::now();
        match result {
            Ok(text) => {
                self.tell(&r.owner, EventKind::Text(text.trim().to_string()));
                self.state(&r.owner, State::Idle);
                self.set_owner(None);
            }
            Err(e) => self.fail(&r.owner, &e),
        }
    }

    fn tick(&mut self) {
        if let Some(r) = self.recording.as_mut() {
            let seconds = r.recorder.seconds();
            if seconds >= self.limits.max_audio.as_secs_f64() {
                return self.stop();
            }
            if seconds as u32 > r.shown {
                r.shown = seconds as u32;
                let (owner, shown) = (r.owner.clone(), r.shown);
                self.state(&owner, State::Recording { seconds: shown });
            }
            return;
        }
        if self.last_used.elapsed() >= self.limits.idle_unload {
            if let Some(whisper) = self.whisper.as_mut().filter(|w| w.loaded()) {
                whisper.unload();
            }
        }
    }
}

/// A dictation that is ready, records a second of tone and says `says` (for other modules' tests).
#[cfg(test)]
pub fn fake(says: &'static str, on: Box<dyn Fn(Event) + Send>) -> Dictation {
    struct Mic(Option<Instant>);
    impl Recorder for Mic {
        fn start(&mut self) -> Result<(), DictationError> {
            self.0 = Some(Instant::now());
            Ok(())
        }
        fn seconds(&self) -> f64 {
            self.0.map_or(0.0, |s| s.elapsed().as_secs_f64())
        }
        fn finish(&mut self) -> Vec<f32> {
            vec![0.1; 16_000]
        }
    }
    struct Says(&'static str, bool);
    impl Transcriber for Says {
        fn loaded(&self) -> bool {
            self.1
        }
        fn load(&mut self) -> Result<(), DictationError> {
            self.1 = true;
            Ok(())
        }
        fn transcribe(&mut self, _: &[f32], _: &str) -> Result<String, DictationError> {
            Ok(self.0.into())
        }
        fn unload(&mut self) {
            self.1 = false;
        }
    }
    Dictation::new(
        Parts {
            model_ready: Box::new(|| true),
            download: Box::new(|_| Ok(())),
            recorder: Box::new(|| Box::new(Mic(None))),
            transcriber: Box::new(move || Box::new(Says(says, false))),
            on,
        },
        Limits::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictation::model::ModelError;
    use std::sync::mpsc::{channel, Receiver};
    use std::sync::{Arc, Mutex};

    /// Ten seconds of a tone, or of exact zeros (what a denied microphone gives on macOS).
    struct FakeMic {
        silent: bool,
        fail: Option<DictationError>,
        started: Option<std::time::Instant>,
    }

    impl Recorder for FakeMic {
        fn start(&mut self) -> Result<(), DictationError> {
            if let Some(e) = self.fail.clone() {
                return Err(e);
            }
            self.started = Some(std::time::Instant::now());
            Ok(())
        }
        fn seconds(&self) -> f64 {
            self.started.map_or(0.0, |s| s.elapsed().as_secs_f64())
        }
        fn finish(&mut self) -> Vec<f32> {
            (0..16_000).map(|i| if self.silent { 0.0 } else { (i as f32 / 10.0).sin() * 0.2 }).collect()
        }
    }

    #[derive(Default)]
    struct Log {
        loads: u32,
        unloads: u32,
        heard: Vec<(usize, String)>,
    }

    struct FakeWhisper {
        log: Arc<Mutex<Log>>,
        loaded: bool,
        load_takes: Duration,
        says: String,
    }

    impl Transcriber for FakeWhisper {
        fn loaded(&self) -> bool {
            self.loaded
        }
        fn load(&mut self) -> Result<(), DictationError> {
            std::thread::sleep(self.load_takes);
            self.loaded = true;
            self.log.lock().unwrap().loads += 1;
            Ok(())
        }
        fn transcribe(&mut self, audio: &[f32], language: &str) -> Result<String, DictationError> {
            assert!(self.loaded, "transcribing before the model is loaded");
            self.log.lock().unwrap().heard.push((audio.len(), language.into()));
            Ok(self.says.clone())
        }
        fn unload(&mut self) {
            self.loaded = false;
            self.log.lock().unwrap().unloads += 1;
        }
    }

    struct Setup {
        model: bool,
        silent: bool,
        fail: Option<DictationError>,
        load_takes: Duration,
        says: &'static str,
        limits: Limits,
        download: Result<(), ModelError>,
    }

    impl Default for Setup {
        fn default() -> Setup {
            Setup {
                model: true,
                silent: false,
                fail: None,
                load_takes: Duration::ZERO,
                says: " hola Aria ",
                limits: Limits::default(),
                download: Ok(()),
            }
        }
    }

    fn dictation(setup: Setup) -> (Dictation, Receiver<Event>, Arc<Mutex<Log>>) {
        let (tx, rx) = channel();
        let tx = Mutex::new(tx);
        let log = Arc::new(Mutex::new(Log::default()));
        let model = Arc::new(Mutex::new(setup.model));
        let (silent, fail, load_takes, says) =
            (setup.silent, setup.fail.clone(), setup.load_takes, setup.says.to_string());
        let whisper_log = log.clone();
        let ready = model.clone();
        let mut download = Some(setup.download.clone());
        let d = Dictation::new(
            Parts {
                model_ready: Box::new(move || *ready.lock().unwrap()),
                download: Box::new(move |progress| {
                    progress(0.5);
                    let result = download.take().unwrap_or(Ok(()));
                    if result.is_ok() {
                        *model.lock().unwrap() = true;
                    }
                    result
                }),
                recorder: Box::new(move || Box::new(FakeMic { silent, fail: fail.clone(), started: None })),
                transcriber: Box::new(move || {
                    Box::new(FakeWhisper { log: whisper_log, loaded: false, load_takes, says })
                }),
                on: Box::new(move |e| {
                    let _ = tx.lock().unwrap().send(e);
                }),
            },
            setup.limits,
        );
        (d, rx, log)
    }

    fn until(rx: &Receiver<Event>, what: &str, found: impl Fn(&Event) -> bool) -> Event {
        loop {
            match rx.recv_timeout(Duration::from_secs(5)) {
                Ok(e) if found(&e) => return e,
                Ok(_) => {}
                Err(_) => panic!("timed out waiting for {what}"),
            }
        }
    }

    fn state(e: &Event) -> Option<&State> {
        match &e.kind {
            EventKind::State(s) => Some(s),
            _ => None,
        }
    }

    #[test]
    fn start_then_stop_gives_the_text_to_the_tab() {
        let (d, rx, log) = dictation(Setup::default());
        d.start("t1", "es");
        until(&rx, "recording", |e| state(e) == Some(&State::Recording { seconds: 0 }));
        assert_eq!(d.owner().as_deref(), Some("t1"));
        d.stop();
        until(&rx, "transcribing", |e| state(e) == Some(&State::Transcribing));
        let text = until(&rx, "the text", |e| matches!(e.kind, EventKind::Text(_)));
        assert_eq!(text, Event { owner: "t1".into(), kind: EventKind::Text("hola Aria".into()) });
        until(&rx, "idle", |e| state(e) == Some(&State::Idle));
        assert_eq!(log.lock().unwrap().heard, vec![(16_000, "es".to_string())]);
        assert_eq!(d.owner(), None);
    }

    #[test]
    fn start_without_the_model_needs_it() {
        let (d, rx, _) = dictation(Setup { model: false, ..Setup::default() });
        d.start("t1", "es");
        let e = until(&rx, "needs model", |e| state(e).is_some());
        assert_eq!(e.kind, EventKind::State(State::NeedsModel { size: crate::dictation::model::SIZE }));
        assert_eq!(d.owner(), None);
    }

    #[test]
    fn a_download_then_records() {
        let (d, rx, _) = dictation(Setup { model: false, ..Setup::default() });
        d.download("t1", "en");
        until(&rx, "progress", |e| state(e) == Some(&State::Downloading { progress: 0.5 }));
        until(&rx, "recording after the download", |e| state(e) == Some(&State::Recording { seconds: 0 }));
    }

    #[test]
    fn a_broken_download_says_so() {
        let (d, rx, _) = dictation(Setup { model: false, download: Err(ModelError::Broken), ..Setup::default() });
        d.download("t1", "en");
        until(&rx, "broken", |e| state(e) == Some(&State::Error("model-broken")));
        until(&rx, "idle", |e| state(e) == Some(&State::Idle));
    }

    #[test]
    fn cancel_drops_the_recording() {
        let (d, rx, log) = dictation(Setup::default());
        d.start("t1", "es");
        until(&rx, "recording", |e| state(e) == Some(&State::Recording { seconds: 0 }));
        d.cancel();
        until(&rx, "idle", |e| state(e) == Some(&State::Idle));
        std::thread::sleep(Duration::from_millis(100));
        assert!(rx.try_iter().all(|e| !matches!(e.kind, EventKind::Text(_))));
        assert!(log.lock().unwrap().heard.is_empty());
    }

    #[test]
    fn stop_waits_for_the_load() {
        let (d, rx, _) = dictation(Setup { load_takes: Duration::from_millis(200), ..Setup::default() });
        d.start("t1", "es");
        d.stop();
        let text = until(&rx, "the text", |e| matches!(e.kind, EventKind::Text(_)));
        assert_eq!(text.kind, EventKind::Text("hola Aria".into()));
    }

    #[test]
    fn a_recorder_that_fails_says_why() {
        let (d, rx, _) = dictation(Setup { fail: Some(DictationError::NoMicrophone), ..Setup::default() });
        d.start("t1", "es");
        until(&rx, "the error", |e| state(e) == Some(&State::Error("no-microphone")));
        until(&rx, "idle", |e| state(e) == Some(&State::Idle));
    }

    #[test]
    fn exact_silence_is_a_denied_microphone() {
        let (d, rx, log) = dictation(Setup { silent: true, ..Setup::default() });
        d.start("t1", "es");
        d.stop();
        until(&rx, "denied", |e| state(e) == Some(&State::Error("microphone-denied")));
        assert!(log.lock().unwrap().heard.is_empty());
    }

    #[test]
    fn one_dictation_at_a_time() {
        let (d, rx, _) = dictation(Setup::default());
        d.start("t1", "es");
        until(&rx, "recording", |e| state(e) == Some(&State::Recording { seconds: 0 }));
        d.start("t2", "es");
        let busy = until(&rx, "busy", |e| e.owner == "t2");
        assert_eq!(busy.kind, EventKind::State(State::Error("busy")));
        assert_eq!(d.owner().as_deref(), Some("t1"));
    }

    #[test]
    fn the_audio_cap_stops_by_itself() {
        let limits = Limits { max_audio: Duration::from_millis(300), ..Limits::default() };
        let (d, rx, _) = dictation(Setup { limits, ..Setup::default() });
        d.start("t1", "es");
        until(&rx, "the text without a stop", |e| matches!(e.kind, EventKind::Text(_)));
    }

    #[test]
    fn idle_unloads_the_model() {
        let limits = Limits { idle_unload: Duration::from_millis(300), ..Limits::default() };
        let (d, rx, log) = dictation(Setup { limits, ..Setup::default() });
        d.start("t1", "es");
        d.stop();
        until(&rx, "the text", |e| matches!(e.kind, EventKind::Text(_)));
        std::thread::sleep(Duration::from_millis(800));
        let log = log.lock().unwrap();
        assert_eq!((log.loads, log.unloads), (1, 1));
    }
}
