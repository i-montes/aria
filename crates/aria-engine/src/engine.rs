//! One supervised Pi child in RPC mode. Commands go in as JSON lines; responses are matched to the
//! `request` waiting for them, everything else (session events, dialogs, the exit) arrives on the
//! event channel.

use crate::framing::LineSplitter;
use crate::protocol::{self, Incoming, UiAnswer};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct SpawnSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    /// Names removed from the inherited environment before `env` is applied.
    pub unset: Vec<String>,
    pub cwd: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// A session event, verbatim.
    Event(Value),
    /// A dialog or fire-and-forget request from an extension.
    UiRequest { id: String, method: String, raw: Value },
    /// A response nobody was waiting for (a command sent with `send`); failures matter here.
    Response { id: Option<String>, command: String, success: bool, error: Option<String> },
    /// Pi's stdout closed and the process is gone.
    Exited { code: Option<i32> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub success: bool,
    pub data: Value,
    pub error: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("could not start Pi: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("Pi is no longer running")]
    Gone,
    #[error("Pi did not answer in time")]
    Timeout,
}

type Pending = Arc<Mutex<HashMap<String, Sender<Response>>>>;

pub struct Engine {
    stdin: Mutex<Option<ChildStdin>>,
    child: Arc<Mutex<Child>>,
    pending: Pending,
    alive: Arc<AtomicBool>,
}

impl Engine {
    pub fn spawn(spec: SpawnSpec) -> Result<(Engine, Receiver<EngineEvent>), EngineError> {
        let mut command = Command::new(&spec.program);
        command.args(&spec.args).current_dir(&spec.cwd).stdin(Stdio::piped()).stdout(Stdio::piped())// Pi's diagnostics go nowhere unless ARIA_DEBUG_PI is set (then to the dock's own stderr).
        .stderr(if std::env::var_os("ARIA_DEBUG_PI").is_some() { Stdio::inherit() } else { Stdio::null() });
        for name in &spec.unset {
            command.env_remove(name);
        }
        // Pi leads a process group of its own: what it starts (MCP servers, subagents, a dev server) joins it, so
        // `close_group` can stop all of it.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        for (key, value) in &spec.env {
            command.env(key, value);
        }
        let mut child = command.spawn()?;
        let stdin = child.stdin.take().ok_or(EngineError::Gone)?;
        let mut stdout = child.stdout.take().ok_or(EngineError::Gone)?;
        let child = Arc::new(Mutex::new(child));
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let (events, receiver) = mpsc::channel();

        {
            let (child, pending, alive) = (child.clone(), pending.clone(), alive.clone());
            std::thread::spawn(move || {
                let mut splitter = LineSplitter::new();
                let mut chunk = [0u8; 8192];
                loop {
                    match stdout.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(read) => {
                            for line in splitter.push(&chunk[..read]) {
                                route(&line, &pending, &events);
                            }
                        }
                    }
                }
                alive.store(false, Ordering::SeqCst);
                pending.lock().unwrap().clear(); // wakes every request still waiting
                let code = loop {
                    if let Ok(Some(status)) = child.lock().unwrap().try_wait() {
                        break status.code();
                    }
                    std::thread::sleep(Duration::from_millis(25));
                };
                let _ = events.send(EngineEvent::Exited { code });
            });
        }

        Ok((Engine { stdin: Mutex::new(Some(stdin)), child, pending, alive }, receiver))
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    fn write_line(&self, line: &str) -> Result<(), EngineError> {
        if !self.is_alive() {
            return Err(EngineError::Gone);
        }
        let mut guard = self.stdin.lock().unwrap();
        let stdin = guard.as_mut().ok_or(EngineError::Gone)?;
        stdin.write_all(line.as_bytes()).and_then(|_| stdin.write_all(b"\n")).and_then(|_| stdin.flush()).map_err(|_| EngineError::Gone)
    }

    /// Sends a command and returns its id; the response arrives as `EngineEvent::Response`.
    pub fn send(&self, kind: &str, fields: Value) -> Result<String, EngineError> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst).to_string();
        self.write_line(&protocol::command(kind, &id, fields))?;
        Ok(id)
    }

    /// Sends a command and waits for its response.
    pub fn request(&self, kind: &str, fields: Value, timeout: Duration) -> Result<Response, EngineError> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst).to_string();
        let (sender, receiver) = mpsc::channel();
        self.pending.lock().unwrap().insert(id.clone(), sender);
        if let Err(error) = self.write_line(&protocol::command(kind, &id, fields)) {
            self.pending.lock().unwrap().remove(&id);
            return Err(error);
        }
        match receiver.recv_timeout(timeout) {
            Ok(response) => Ok(response),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.pending.lock().unwrap().remove(&id);
                Err(EngineError::Timeout)
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(EngineError::Gone),
        }
    }

    pub fn answer_ui(&self, id: &str, answer: UiAnswer) -> Result<(), EngineError> {
        self.write_line(&protocol::ui_response(id, &answer))
    }

    /// Closes stdin (Pi's orderly shutdown), waits up to `grace`, then kills the process. Works through a shared
    /// reference, so whoever holds the engine can stop it.
    pub fn close(&self, grace: Duration) {
        self.stdin.lock().unwrap().take();
        let deadline = Instant::now() + grace;
        while Instant::now() < deadline {
            if matches!(self.child.lock().unwrap().try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.child.lock().unwrap().kill();
    }

    /// `close`, then everything Pi started that is still in its process group: asked to stop, killed a second later.
    /// For a session that is closed for good; a restart uses `close` and leaves the session's processes running.
    pub fn close_group(&self, grace: Duration) {
        let group = self.child.lock().unwrap().id() as i32;
        self.close(grace);
        #[cfg(unix)]
        {
            let signal = |signal: i32| unsafe { libc::killpg(group, signal) == 0 };
            if !signal(libc::SIGTERM) {
                return; // nobody is left in the group
            }
            let deadline = Instant::now() + Duration::from_secs(1);
            while Instant::now() < deadline && signal(0) {
                std::thread::sleep(Duration::from_millis(20));
            }
            signal(libc::SIGKILL);
        }
    }

    /// `close` with the default three seconds.
    pub fn shutdown(self) {
        self.close(Duration::from_secs(3));
    }
}

fn route(line: &str, pending: &Pending, events: &Sender<EngineEvent>) {
    match protocol::parse_line(line) {
        Incoming::Response { id, command, success, data, error } => {
            let waiting = id.as_ref().and_then(|id| pending.lock().unwrap().remove(id));
            match waiting {
                Some(sender) => {
                    let _ = sender.send(Response { success, data, error });
                }
                None => {
                    let _ = events.send(EngineEvent::Response { id, command, success, error });
                }
            }
        }
        Incoming::UiRequest { id, method, raw } => {
            let _ = events.send(EngineEvent::UiRequest { id, method, raw });
        }
        Incoming::Event { raw, .. } => {
            let _ = events.send(EngineEvent::Event(raw));
        }
        Incoming::Unparsable(_) => {}
    }
}
