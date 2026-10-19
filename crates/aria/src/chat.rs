//! The chat: owns Pi (started when first needed, kept while the dock is hidden), turns page
//! messages into Pi commands and Pi's events into dock messages, and asks for attention with a
//! desktop notification when an extension dialog arrives while the dock is hidden.

use aria_core::agent;
use aria_core::bridge::{DockMessage, PageMessage};
use aria_core::files;
use aria_core::paths::AriaPaths;
use aria_core::pi::{self, PiError};
use aria_core::settings::{self, Settings};
use aria_engine::engine::{Engine, EngineEvent, SpawnSpec};
use aria_engine::protocol::UiAnswer;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub type Emit = Arc<dyn Fn(DockMessage) + Send + Sync>;

const NOTIFICATION_TITLE: &str = "Aria needs an answer";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const FILE_LIMIT: usize = 50;

/// Raises and removes the desktop notification for a prompt nobody can see.
pub trait Notifier: Send + Sync {
    fn send(&self, title: &str, body: &str);
    fn dismiss(&self);
}

/// Omarchy's notifications: critical urgency and no expiry, so it stays until it is dismissed or
/// clicked; the click runs `aria show`, which opens the dock.
pub struct OmarchyNotifier;

/// The arguments for `omarchy-notification-send`.
pub fn notify_args(exe: &Path, title: &str, body: &str) -> Vec<String> {
    ["--app-name", "aria", "-u", "critical", "-t", "0", title, body, "--exec", &exe.display().to_string(), "show"]
        .iter()
        .map(|part| part.to_string())
        .collect()
}

fn detached(program: &str, args: &[String]) {
    let _ = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|mut child| std::thread::spawn(move || child.wait()));
}

impl Notifier for OmarchyNotifier {
    fn send(&self, title: &str, body: &str) {
        if let Ok(exe) = std::env::current_exe() {
            detached("omarchy-notification-send", &notify_args(&exe, title, body));
        }
    }

    fn dismiss(&self) {
        detached("omarchy-notification-dismiss", &[NOTIFICATION_TITLE.to_string()]);
    }
}

pub struct ChatCtx {
    pub paths: AriaPaths,
    pub notifier: Arc<dyn Notifier>,
    pub dock_visible: Arc<AtomicBool>,
    /// The `PATH` Pi inherits (the shim directory goes in front of it).
    pub path_var: String,
}

#[derive(Default)]
struct Inner {
    engine: Option<Arc<Engine>>,
    /// Bumped for every Pi that starts, so a dying Pi cannot clear its successor.
    generation: u64,
    running: bool,
    /// Dialogs waiting for an answer, by Pi's request id.
    dialogs: HashMap<String, Value>,
    /// The dialogs a notification was raised for.
    notified: HashSet<String>,
    model: Option<String>,
    thinking: Option<String>,
    /// The folder the running Pi was started in (what the header must show, whatever the settings say now).
    workdir: Option<PathBuf>,
}

/// What `get_state` tells about the model and thinking level. `running` is deliberately not taken from
/// it: the events (`agent_start`, `agent_settled`) own that, and a late reply must not undo them.
fn apply_state(inner: &mut Inner, data: &Value) {
    let model = data["model"]["provider"].as_str().zip(data["model"]["id"].as_str()).map(|(p, id)| format!("{p}/{id}"));
    inner.model = model.or(inner.model.take());
    inner.thinking = data["thinkingLevel"].as_str().map(str::to_string);
}

#[derive(Clone)]
pub struct Chat {
    ctx: Arc<ChatCtx>,
    emit: Emit,
    inner: Arc<Mutex<Inner>>,
    /// Starting Pi is check-then-act across threads (a send and a `/` picker can race); one at a time.
    spawn_lock: Arc<Mutex<()>>,
}

fn dialog_body(raw: &Value) -> String {
    raw["title"].as_str().or_else(|| raw["message"].as_str()).unwrap_or_default().to_string()
}

fn is_dialog(method: &str) -> bool {
    matches!(method, "select" | "confirm" | "input" | "editor")
}

impl Chat {
    pub fn new(ctx: ChatCtx, emit: Emit) -> Chat {
        Chat { ctx: Arc::new(ctx), emit, inner: Arc::new(Mutex::new(Inner::default())), spawn_lock: Arc::new(Mutex::new(())) }
    }

    fn emit(&self, message: DockMessage) {
        (self.emit)(message);
    }

    fn settings(&self) -> Option<Settings> {
        settings::load(&self.ctx.paths).ok().flatten()
    }

    fn state_message(&self, problem: Option<&str>) -> DockMessage {
        let settings = self.settings();
        let inner = self.inner.lock().unwrap();
        DockMessage::ChatState {
            workdir: inner.workdir.clone().unwrap_or_else(|| agent::workdir(settings.as_ref())).display().to_string(),
            running: inner.running,
            model: inner.model.clone().or_else(|| settings.as_ref().and_then(|s| s.models.daily.clone())),
            thinking: inner.thinking.clone(),
            problem: problem.map(str::to_string),
        }
    }

    /// Why Pi cannot run on this machine, as a UI string key.
    fn problem(&self) -> Option<&'static str> {
        if let Err(error) = pi::locate(&self.ctx.paths, pi::pinned_version()) {
            return Some(error.problem_code());
        }
        if !pi::node_ok(&self.ctx.paths, &self.ctx.path_var) {
            return Some(PiError::NodeMissing.problem_code());
        }
        None
    }

    pub fn handle(&self, message: PageMessage) {
        match message {
            PageMessage::ChatLoad => self.load(),
            PageMessage::ChatSend { text, behavior } => self.send_prompt(&text, behavior.as_deref()),
            PageMessage::ChatAbort => {
                if let Some(engine) = self.current_engine() {
                    let _ = engine.send("clear_queue", json!({}));
                    let _ = engine.send("abort", json!({}));
                }
            }
            PageMessage::ChatNew => {
                if let Some(engine) = self.current_engine() {
                    let _ = engine.send("new_session", json!({}));
                    self.emit(DockMessage::ChatEvent { event: json!({"type": "session_reset"}) });
                }
            }
            PageMessage::UiAnswer { id, value, confirmed, cancelled } => self.answer(&id, value, confirmed, cancelled),
            PageMessage::ListCommands { id } => self.list(id, "get_commands", "commands"),
            PageMessage::ListModels { id } => self.list(id, "get_available_models", "models"),
            PageMessage::SetModel { id, model } => self.set_model(id, model),
            PageMessage::ListFiles { id, query } => {
                let chat = self.clone();
                std::thread::spawn(move || {
                    let root = agent::workdir(chat.settings().as_ref());
                    chat.emit(DockMessage::Files { id, files: files::list_files(&root, &query, FILE_LIMIT) });
                });
            }
            _ => {}
        }
    }

    fn current_engine(&self) -> Option<Arc<Engine>> {
        self.inner.lock().unwrap().engine.clone().filter(|engine| engine.is_alive())
    }

    fn load(&self) {
        self.emit(self.state_message(self.problem()));
        let open: Vec<(String, Value)> = self.inner.lock().unwrap().dialogs.iter().map(|(id, raw)| (id.clone(), raw.clone())).collect();
        for (id, raw) in open {
            let method = raw["method"].as_str().unwrap_or_default().to_string();
            self.emit(DockMessage::ChatDialog { id, method, request: raw, hidden: false });
        }
    }

    fn send_prompt(&self, text: &str, behavior: Option<&str>) {
        let engine = match self.ensure_engine() {
            Ok(engine) => engine,
            Err(code) => return self.emit(self.state_message(Some(code))),
        };
        let mut fields = json!({"message": text});
        if let Some(behavior) = behavior {
            fields["streamingBehavior"] = json!(behavior);
        }
        if engine.send("prompt", fields).is_err() {
            self.emit(DockMessage::ChatEvent { event: json!({"type": "command_failed", "command": "prompt", "error": "Pi is no longer running"}) });
        }
    }

    /// The running Pi, starting one when there is none. `Err` is a UI problem key.
    fn ensure_engine(&self) -> Result<Arc<Engine>, &'static str> {
        if let Some(engine) = self.current_engine() {
            return Ok(engine);
        }
        let _one_at_a_time = self.spawn_lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(engine) = self.current_engine() {
            return Ok(engine); // another thread started it while this one waited
        }
        let install = pi::locate(&self.ctx.paths, pi::pinned_version()).map_err(|e| e.problem_code())?;
        if !pi::node_ok(&self.ctx.paths, &self.ctx.path_var) {
            return Err(PiError::NodeMissing.problem_code());
        }
        let settings = self.settings();
        agent::prepare(&self.ctx.paths, settings.as_ref()).map_err(|_| "agent-dir-unwritable")?;
        pi::write_shim(&self.ctx.paths, &install).map_err(|_| "agent-dir-unwritable")?;
        let workdir = working_folder(settings.as_ref());

        self.emit(DockMessage::ChatEngine { state: "starting".into() });
        let spec = SpawnSpec {
            program: install.binary,
            args: vec!["--mode".into(), "rpc".into(), "--continue".into()],
            env: vec![
                ("PI_CODING_AGENT_DIR".into(), self.ctx.paths.agent().display().to_string()),
                ("PATH".into(), pi::path_with_bin(&self.ctx.paths, &self.ctx.path_var)),
                ("PI_SKIP_VERSION_CHECK".into(), "1".into()),
            ],
            cwd: workdir.clone(),
        };
        let (engine, events) = Engine::spawn(spec).map_err(|_| "pi-failed")?;
        let engine = Arc::new(engine);
        let (generation, stale) = {
            let mut inner = self.inner.lock().unwrap();
            inner.generation += 1;
            inner.engine = Some(engine.clone());
            inner.running = false;
            inner.workdir = Some(workdir);
            // Whatever the previous Pi left open can never be answered now.
            let stale: Vec<String> = inner.dialogs.drain().map(|(id, _)| id).collect();
            let dismiss = !inner.notified.is_empty();
            inner.notified.clear();
            (inner.generation, (stale, dismiss))
        };
        self.close_stale(stale);
        let chat = self.clone();
        std::thread::spawn(move || chat.pump(events, generation));
        self.emit(DockMessage::ChatEngine { state: "ready".into() });
        let chat = self.clone();
        let for_state = engine.clone();
        std::thread::spawn(move || chat.announce(&for_state));
        Ok(engine)
    }

    /// Tells the page what Pi is using and replays the session's history.
    fn announce(&self, engine: &Engine) {
        if let Ok(response) = engine.request("get_state", json!({}), REQUEST_TIMEOUT) {
            if response.success {
                apply_state(&mut self.inner.lock().unwrap(), &response.data);
                self.emit(self.state_message(None));
            }
        }
        if let Ok(response) = engine.request("get_messages", json!({}), REQUEST_TIMEOUT) {
            if response.success {
                self.emit(DockMessage::ChatEvent { event: json!({"type": "history", "messages": response.data["messages"]}) });
            }
        }
    }

    fn list(&self, id: u64, command: &'static str, key: &'static str) {
        let chat = self.clone();
        std::thread::spawn(move || {
            let data = chat
                .ensure_engine()
                .ok()
                .and_then(|engine| engine.request(command, json!({}), REQUEST_TIMEOUT).ok())
                .filter(|response| response.success)
                .map(|response| response.data[key].clone())
                .filter(Value::is_array)
                .unwrap_or_else(|| json!([]));
            chat.emit(match key {
                "commands" => DockMessage::Commands { id, commands: data },
                _ => DockMessage::Models { id, models: data },
            });
        });
    }

    fn set_model(&self, id: u64, model: String) {
        let chat = self.clone();
        std::thread::spawn(move || {
            let Some((provider, model_id)) = agent::split_model(&model) else {
                return chat.emit(DockMessage::Error { id, code: "model-invalid".into() });
            };
            let engine = match chat.ensure_engine() {
                Ok(engine) => engine,
                Err(code) => return chat.emit(DockMessage::Error { id, code: code.into() }),
            };
            match engine.request("set_model", json!({"provider": provider, "modelId": model_id}), REQUEST_TIMEOUT) {
                Ok(response) if response.success => {
                    chat.inner.lock().unwrap().model = Some(model.clone());
                    chat.emit(chat.state_message(None));
                    chat.emit(DockMessage::Saved { id });
                }
                _ => chat.emit(DockMessage::Error { id, code: "model-rejected".into() }),
            }
        });
    }

    fn answer(&self, id: &str, value: Option<String>, confirmed: Option<bool>, cancelled: Option<bool>) {
        if !self.inner.lock().unwrap().dialogs.contains_key(id) {
            return; // an answer for a dialog that is gone (timed out, Pi exited): nothing to do
        }
        let answer = match (value, confirmed, cancelled) {
            (_, _, Some(true)) => UiAnswer::Cancelled,
            (Some(value), _, _) => UiAnswer::Value(value),
            (_, Some(confirmed), _) => UiAnswer::Confirmed(confirmed),
            _ => UiAnswer::Cancelled,
        };
        if let Some(engine) = self.current_engine() {
            let _ = engine.answer_ui(id, answer);
        }
        self.close_dialog(id);
    }

    /// Removes a dialog (once) and, when it was the last one a notification was raised for, the notification.
    fn close_dialog(&self, id: &str) {
        let dismiss = {
            let mut inner = self.inner.lock().unwrap();
            if inner.dialogs.remove(id).is_none() {
                return;
            }
            inner.notified.remove(id) && inner.notified.is_empty()
        };
        if dismiss {
            self.ctx.notifier.dismiss();
        }
        self.emit(DockMessage::ChatDialogClosed { id: id.to_string() });
    }

    /// Consumes one Pi's events until it exits.
    fn pump(&self, events: Receiver<EngineEvent>, generation: u64) {
        for event in events {
            let current = self.inner.lock().unwrap().generation == generation;
            if !current && !matches!(event, EngineEvent::Exited { .. }) {
                continue; // a Pi that was replaced no longer speaks for the chat
            }
            match event {
                EngineEvent::Event(raw) => {
                    match raw["type"].as_str() {
                        Some("agent_start") => self.inner.lock().unwrap().running = true,
                        Some("agent_settled") => self.inner.lock().unwrap().running = false,
                        _ => {}
                    }
                    self.emit(DockMessage::ChatEvent { event: raw });
                }
                EngineEvent::UiRequest { id, method, raw } if is_dialog(&method) => self.open_dialog(id, method, raw),
                EngineEvent::UiRequest { raw, .. } => {
                    self.emit(DockMessage::ChatEvent { event: json!({"type": "ui_notice", "request": raw}) });
                }
                EngineEvent::Response { success: false, command, error, .. } => {
                    self.emit(DockMessage::ChatEvent { event: json!({"type": "command_failed", "command": command, "error": error}) });
                }
                EngineEvent::Response { .. } => {}
                EngineEvent::Exited { code } => {
                    self.exited(generation, code);
                    break;
                }
            }
        }
    }

    fn close_stale(&self, (open, dismiss): (Vec<String>, bool)) {
        if dismiss {
            self.ctx.notifier.dismiss();
        }
        for id in open {
            self.emit(DockMessage::ChatDialogClosed { id });
        }
    }

    /// The dock's visibility changed: when it is now hidden, every open dialog that has no notification
    /// yet gets one (the owner may have pressed Escape with a question still open).
    pub fn dock_visibility_changed(&self) {
        if self.ctx.dock_visible.load(Ordering::SeqCst) {
            return;
        }
        let waiting: Vec<String> = {
            let mut inner = self.inner.lock().unwrap();
            let fresh: Vec<String> = inner.dialogs.keys().filter(|id| !inner.notified.contains(*id)).cloned().collect();
            let bodies = fresh.iter().map(|id| inner.dialogs[id].clone()).collect::<Vec<_>>();
            inner.notified.extend(fresh);
            bodies.iter().map(dialog_body).collect()
        };
        for body in waiting {
            self.ctx.notifier.send(NOTIFICATION_TITLE, &body);
        }
    }

    fn open_dialog(&self, id: String, method: String, raw: Value) {
        let hidden = !self.ctx.dock_visible.load(Ordering::SeqCst);
        {
            let mut inner = self.inner.lock().unwrap();
            inner.dialogs.insert(id.clone(), raw.clone());
            if hidden {
                inner.notified.insert(id.clone());
            }
        }
        if hidden {
            self.ctx.notifier.send(NOTIFICATION_TITLE, &dialog_body(&raw));
        }
        if let Some(timeout) = raw["timeout"].as_u64() {
            // Pi resolves the dialog by itself when its timeout runs out; the page and the notification follow.
            let (chat, id) = (self.clone(), id.clone());
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(timeout));
                chat.close_dialog(&id);
            });
        }
        self.emit(DockMessage::ChatDialog { id, method, request: raw, hidden });
    }

    fn exited(&self, generation: u64, code: Option<i32>) {
        let (open, dismiss) = {
            let mut inner = self.inner.lock().unwrap();
            if inner.generation != generation {
                return;
            }
            inner.engine = None;
            inner.running = false;
            let open: Vec<String> = inner.dialogs.drain().map(|(id, _)| id).collect();
            let dismiss = !inner.notified.is_empty();
            inner.notified.clear();
            (open, dismiss)
        };
        if dismiss {
            self.ctx.notifier.dismiss();
        }
        for id in open {
            self.emit(DockMessage::ChatDialogClosed { id });
        }
        self.emit(DockMessage::ChatEngine { state: "exited".into() });
        self.emit(DockMessage::ChatEvent { event: json!({"type": "engine_exited", "code": code}) });
    }

    #[cfg(test)]
    fn test_send_raw(&self, kind: &str) {
        if let Some(engine) = self.current_engine() {
            let _ = engine.send(kind, json!({}));
        }
    }
}

/// Pi's folder: the configured one, created when missing, `$HOME` when that is not possible.
fn working_folder(settings: Option<&Settings>) -> PathBuf {
    let folder = agent::workdir(settings);
    if std::fs::create_dir_all(&folder).is_ok() {
        folder
    } else {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::settings::{self, Settings};
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::{self, Receiver};
    use std::sync::Mutex;
    use std::time::Duration;

    #[derive(Default)]
    struct TestNotifier {
        log: Mutex<Vec<String>>,
    }

    impl Notifier for TestNotifier {
        fn send(&self, title: &str, body: &str) {
            self.log.lock().unwrap().push(format!("send:{title}:{body}"));
        }
        fn dismiss(&self) {
            self.log.lock().unwrap().push("dismiss".into());
        }
    }

    impl TestNotifier {
        fn log(&self) -> Vec<String> {
            self.log.lock().unwrap().clone()
        }
    }

    struct Harness {
        _turn: std::sync::MutexGuard<'static, ()>,
        /// Messages `wait_for` skipped over, kept so a later wait (or `drain`) can still see them.
        backlog: Mutex<Vec<DockMessage>>,
        _dir: tempfile::TempDir,
        paths: AriaPaths,
        chat: Chat,
        messages: Receiver<DockMessage>,
        notifier: Arc<TestNotifier>,
        visible: Arc<AtomicBool>,
        workdir: PathBuf,
    }

    fn fake_pi_script() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../aria-engine/tests/fake_pi.sh").canonicalize().unwrap()
    }

    /// An Aria home with a "pinned Pi" that is the fake script and a Node that says it is new enough.
    fn harness_with(install_pi: bool) -> Harness {
        let turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths { root: dir.path().join(".aria") };
        paths.ensure_root().unwrap();
        if install_pi {
            let binary = paths.pi_install(aria_core::pi::pinned_version()).join("node_modules/.bin/pi");
            std::fs::create_dir_all(binary.parent().unwrap()).unwrap();
            std::os::unix::fs::symlink(fake_pi_script(), &binary).unwrap();
        }
        std::fs::create_dir_all(paths.bin()).unwrap();
        // A link to a checked-in script, not a file written now: executing a file that another test thread's
        // fork still holds open for writing fails with "text file busy".
        let fake_node = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fake_node.sh");
        std::os::unix::fs::symlink(fake_node, paths.bin().join("node")).unwrap();
        let workdir = dir.path().join("work");
        let mut settings = Settings::default();
        settings.sandbox.projects_dir = workdir.display().to_string();
        settings.models.daily = Some("opencode-go/deepseek-v4.1-flash".into());
        settings::save(&paths, &settings).unwrap();

        let (sender, messages) = mpsc::channel();
        let sender = Mutex::new(sender);
        let emit: Emit = Arc::new(move |message| {
            let _ = sender.lock().unwrap().send(message);
        });
        let notifier = Arc::new(TestNotifier::default());
        let visible = Arc::new(AtomicBool::new(true));
        let ctx = ChatCtx { paths: paths.clone(), notifier: notifier.clone(), dock_visible: visible.clone(), path_var: "/usr/bin:/bin".into() };
        Harness { _turn: turn, backlog: Mutex::new(Vec::new()), _dir: dir, paths, chat: Chat::new(ctx, emit), messages, notifier, visible, workdir }
    }

    fn harness() -> Harness {
        harness_with(true)
    }

    fn wait_for(h: &Harness, what: &str, found: impl Fn(&DockMessage) -> bool) -> DockMessage {
        {
            let mut backlog = h.backlog.lock().unwrap();
            if let Some(at) = backlog.iter().position(&found) {
                return backlog.remove(at);
            }
        }
        loop {
            match h.messages.recv_timeout(Duration::from_secs(5)) {
                Ok(message) if found(&message) => return message,
                Ok(message) => h.backlog.lock().unwrap().push(message),
                Err(_) => panic!("timed out waiting for {what}"),
            }
        }
    }

    /// Everything that has arrived and nobody waited for.
    fn drain(h: &Harness) -> Vec<DockMessage> {
        let mut all = std::mem::take(&mut *h.backlog.lock().unwrap());
        while let Ok(message) = h.messages.try_recv() {
            all.push(message);
        }
        all
    }

    fn event_of(message: &DockMessage, kind: &str) -> bool {
        matches!(message, DockMessage::ChatEvent { event } if event["type"] == kind)
    }

    fn send(h: &Harness, text: &str) {
        h.chat.handle(PageMessage::ChatSend { text: text.into(), behavior: None });
    }

    #[test]
    fn first_send_starts_pi_and_forwards_events() {
        let h = harness();
        send(&h, "hello");
        wait_for(&h, "starting", |m| matches!(m, DockMessage::ChatEngine { state } if state == "starting"));
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        wait_for(&h, "agent_start", |m| event_of(m, "agent_start"));
        wait_for(&h, "the delta", |m| event_of(m, "message_update"));
        wait_for(&h, "agent_settled", |m| event_of(m, "agent_settled"));
    }

    #[test]
    fn pi_runs_in_the_working_folder_and_gets_its_own_agent_dir() {
        let h = harness();
        send(&h, "hello");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        assert!(h.workdir.is_dir(), "the working folder is created when missing");
        assert!(h.paths.agent().join("settings.json").exists());
        assert!(h.paths.bin().join("pi").exists(), "the pi shim is written");
    }

    #[test]
    fn second_send_reuses_the_engine() {
        let h = harness();
        send(&h, "one");
        wait_for(&h, "first settle", |m| event_of(m, "agent_settled"));
        send(&h, "two");
        wait_for(&h, "second settle", |m| event_of(m, "agent_settled"));
        let starts = drain(&h).iter().filter(|m| matches!(m, DockMessage::ChatEngine { state } if state == "starting")).count();
        assert_eq!(starts, 1, "Pi was started more than once");
    }

    #[test]
    fn chat_load_reports_the_working_folder_and_model_without_starting_pi() {
        let h = harness();
        h.chat.handle(PageMessage::ChatLoad);
        match wait_for(&h, "chatState", |m| matches!(m, DockMessage::ChatState { .. })) {
            DockMessage::ChatState { workdir, running, model, problem, .. } => {
                assert_eq!(workdir, h.workdir.display().to_string());
                assert!(!running);
                assert_eq!(model.as_deref(), Some("opencode-go/deepseek-v4.1-flash"));
                assert_eq!(problem, None);
            }
            _ => unreachable!(),
        }
        assert!(drain(&h).is_empty(), "loading must not start Pi");
    }

    #[test]
    fn the_engine_reports_its_model_and_thinking_level_once_ready() {
        let h = harness();
        send(&h, "hello");
        let state = wait_for(&h, "state from get_state", |m| matches!(m, DockMessage::ChatState { thinking: Some(_), .. }));
        assert!(matches!(state, DockMessage::ChatState { thinking: Some(t), model: Some(m), .. } if t == "medium" && m == "opencode-go/deepseek-v4.1-flash"));
    }

    #[test]
    fn a_failed_command_nobody_waited_for_is_reported() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.handle(PageMessage::SetModel { id: 1, model: "nonsense".into() });
        wait_for(&h, "an error reply", |m| matches!(m, DockMessage::Error { id: 1, .. }));
    }

    #[test]
    fn dialog_while_hidden_notifies_once_and_answer_dismisses() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.visible.store(false, Ordering::SeqCst);
        // `ask` is the fake Pi's test-only command that opens a select dialog (u1)
        h.chat.handle(PageMessage::ChatSend { text: "/ask".into(), behavior: None });
        wait_for(&h, "no dialog for plain prompts", |m| event_of(m, "agent_settled"));
        h.chat.test_send_raw("ask");
        let dialog = wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { .. }));
        assert!(matches!(dialog, DockMessage::ChatDialog { ref id, hidden: true, .. } if id == "u1"));
        assert_eq!(h.notifier.log(), vec!["send:Aria needs an answer:Allow?"]);
        h.chat.handle(PageMessage::UiAnswer { id: "u1".into(), value: Some("Allow".into()), confirmed: None, cancelled: None });
        wait_for(&h, "closed", |m| matches!(m, DockMessage::ChatDialogClosed { id } if id == "u1"));
        wait_for(&h, "the answer reaching Pi", |m| event_of(m, "answered"));
        assert_eq!(h.notifier.log(), vec!["send:Aria needs an answer:Allow?", "dismiss"]);
    }

    #[test]
    fn dialog_while_visible_does_not_notify() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.test_send_raw("ask");
        let dialog = wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { .. }));
        assert!(matches!(dialog, DockMessage::ChatDialog { hidden: false, .. }));
        assert!(h.notifier.log().is_empty());
    }

    #[test]
    fn chat_load_replays_dialogs_that_are_still_open() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.visible.store(false, Ordering::SeqCst);
        h.chat.test_send_raw("ask");
        wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { .. }));
        h.chat.handle(PageMessage::ChatLoad);
        wait_for(&h, "the replayed dialog", |m| matches!(m, DockMessage::ChatDialog { id, hidden: false, .. } if id == "u1"));
    }

    #[test]
    fn a_dialog_timeout_closes_it_and_dismisses_the_notification() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.visible.store(false, Ordering::SeqCst);
        h.chat.test_send_raw("ask_timeout");
        wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { id, .. } if id == "u2"));
        wait_for(&h, "closed by timeout", |m| matches!(m, DockMessage::ChatDialogClosed { id } if id == "u2"));
        assert_eq!(h.notifier.log().last().map(String::as_str), Some("dismiss"));
    }

    #[test]
    fn notices_from_extensions_are_events_not_dialogs() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.test_send_raw("notify");
        let notice = wait_for(&h, "the notice", |m| event_of(m, "ui_notice"));
        match notice {
            DockMessage::ChatEvent { event } => assert_eq!(event["request"]["message"], json!("careful")),
            _ => unreachable!(),
        }
        assert!(h.notifier.log().is_empty());
    }

    #[test]
    fn exit_clears_the_engine_dismisses_and_next_send_respawns() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.visible.store(false, Ordering::SeqCst);
        h.chat.test_send_raw("ask");
        wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { .. }));
        h.chat.test_send_raw("crash");
        wait_for(&h, "the dialog closing", |m| matches!(m, DockMessage::ChatDialogClosed { id } if id == "u1"));
        wait_for(&h, "exited", |m| matches!(m, DockMessage::ChatEngine { state } if state == "exited"));
        wait_for(&h, "an engine_exited event", |m| event_of(m, "engine_exited"));
        assert_eq!(h.notifier.log().last().map(String::as_str), Some("dismiss"));
        send(&h, "again");
        wait_for(&h, "a second start", |m| matches!(m, DockMessage::ChatEngine { state } if state == "starting"));
        wait_for(&h, "settled after the respawn", |m| event_of(m, "agent_settled"));
    }

    #[test]
    fn ui_answer_for_an_unknown_id_is_ignored() {
        let h = harness();
        h.chat.handle(PageMessage::UiAnswer { id: "ghost".into(), value: Some("x".into()), confirmed: None, cancelled: None });
        send(&h, "x");
        wait_for(&h, "the chat still works", |m| event_of(m, "agent_settled"));
        h.chat.handle(PageMessage::UiAnswer { id: "ghost".into(), value: None, confirmed: None, cancelled: Some(true) });
        assert!(h.notifier.log().is_empty());
    }

    #[test]
    fn abort_with_no_engine_is_ignored() {
        let h = harness();
        h.chat.handle(PageMessage::ChatAbort);
        h.chat.handle(PageMessage::ChatNew);
        assert!(drain(&h).is_empty(), "no engine means nothing to abort and nothing to say");
    }

    #[test]
    fn missing_pi_reports_the_problem_code_and_never_crashes() {
        let h = harness_with(false);
        send(&h, "hello");
        match wait_for(&h, "a problem", |m| matches!(m, DockMessage::ChatState { problem: Some(_), .. })) {
            DockMessage::ChatState { problem, .. } => assert_eq!(problem.as_deref(), Some("pi-missing")),
            _ => unreachable!(),
        }
        h.chat.handle(PageMessage::ChatLoad);
        wait_for(&h, "the problem again on load", |m| matches!(m, DockMessage::ChatState { problem: Some(p), .. } if p == "pi-missing"));
    }

    #[test]
    fn a_missing_node_is_its_own_problem() {
        let h = harness();
        std::fs::remove_file(h.paths.bin().join("node")).unwrap();
        // /usr/bin:/bin may hold a real node on this machine; point the chat at an empty PATH instead.
        let ctx = ChatCtx { paths: h.paths.clone(), notifier: h.notifier.clone(), dock_visible: h.visible.clone(), path_var: "/nonexistent".into() };
        let (sender, messages) = mpsc::channel();
        let sender = Mutex::new(sender);
        let chat = Chat::new(ctx, Arc::new(move |m| { let _ = sender.lock().unwrap().send(m); }));
        chat.handle(PageMessage::ChatSend { text: "hi".into(), behavior: None });
        let message = messages.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(message, DockMessage::ChatState { problem: Some(ref p), .. } if p == "node-missing"), "{message:?}");
    }

    #[test]
    fn commands_models_and_history_come_from_pi() {
        let h = harness();
        h.chat.handle(PageMessage::ListCommands { id: 7 });
        match wait_for(&h, "commands", |m| matches!(m, DockMessage::Commands { id: 7, .. })) {
            DockMessage::Commands { commands, .. } => assert_eq!(commands[0]["name"], json!("effort")),
            _ => unreachable!(),
        }
        h.chat.handle(PageMessage::ListModels { id: 8 });
        match wait_for(&h, "models", |m| matches!(m, DockMessage::Models { id: 8, .. })) {
            DockMessage::Models { models, .. } => assert_eq!(models[0]["id"], json!("glm-5.3")),
            _ => unreachable!(),
        }
        let history = wait_for(&h, "history", |m| event_of(m, "history"));
        match history {
            DockMessage::ChatEvent { event } => assert_eq!(event["messages"][0]["content"], json!("earlier")),
            _ => unreachable!(),
        }
    }

    #[test]
    fn set_model_splits_provider_and_id_and_confirms() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.handle(PageMessage::SetModel { id: 3, model: "opencode-go/glm-5.3".into() });
        wait_for(&h, "saved", |m| matches!(m, DockMessage::Saved { id: 3 }));
    }

    #[test]
    fn list_files_uses_the_workdir() {
        let h = harness();
        std::fs::create_dir_all(h.workdir.join("proj")).unwrap();
        std::fs::write(h.workdir.join("proj/main.rs"), "x").unwrap();
        h.chat.handle(PageMessage::ListFiles { id: 5, query: "main".into() });
        match wait_for(&h, "files", |m| matches!(m, DockMessage::Files { id: 5, .. })) {
            DockMessage::Files { files, .. } => assert_eq!(files, vec!["proj/main.rs"]),
            _ => unreachable!(),
        }
    }


    #[test]
    fn two_threads_asking_at_once_start_one_pi() {
        let h = harness();
        let (a, b) = (h.chat.clone(), h.chat.clone());
        let t1 = std::thread::spawn(move || a.handle(PageMessage::ListCommands { id: 1 }));
        let t2 = std::thread::spawn(move || b.handle(PageMessage::ChatSend { text: "hi".into(), behavior: None }));
        t1.join().unwrap();
        t2.join().unwrap();
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        wait_for(&h, "commands", |m| matches!(m, DockMessage::Commands { id: 1, .. }));
        let starts = drain(&h).iter().filter(|m| matches!(m, DockMessage::ChatEngine { state } if state == "starting")).count();
        assert_eq!(starts, 1, "two threads must share one Pi");
    }

    #[test]
    fn get_state_never_overrides_what_the_events_say_about_running() {
        let mut inner = Inner { running: true, ..Inner::default() };
        apply_state(&mut inner, &json!({"isStreaming": false, "thinkingLevel": "high", "model": {"provider": "p", "id": "m"}}));
        assert!(inner.running, "a late get_state reply must not stop a run the events started");
        assert_eq!(inner.thinking.as_deref(), Some("high"));
        assert_eq!(inner.model.as_deref(), Some("p/m"));
    }

    #[test]
    fn a_dialog_open_when_the_dock_hides_gets_its_notification_once() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.test_send_raw("ask");
        wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { hidden: false, .. }));
        assert!(h.notifier.log().is_empty());
        h.visible.store(false, Ordering::SeqCst);
        h.chat.dock_visibility_changed();
        h.chat.dock_visibility_changed();
        assert_eq!(h.notifier.log(), vec!["send:Aria needs an answer:Allow?"]);
        h.chat.handle(PageMessage::UiAnswer { id: "u1".into(), value: Some("Allow".into()), confirmed: None, cancelled: None });
        assert_eq!(h.notifier.log().last().map(String::as_str), Some("dismiss"));
    }

    #[test]
    fn the_dialogs_of_a_dead_pi_are_closed_when_the_next_one_starts() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.visible.store(false, Ordering::SeqCst);
        h.chat.test_send_raw("ask");
        wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { .. }));
        h.chat.test_send_raw("crash");
        send(&h, "again"); // may come before the old Pi's exit has been processed
        wait_for(&h, "the old dialog closing", |m| matches!(m, DockMessage::ChatDialogClosed { id } if id == "u1"));
        wait_for(&h, "settled after the respawn", |m| event_of(m, "agent_settled"));
        assert_eq!(h.notifier.log().last().map(String::as_str), Some("dismiss"));
        h.chat.handle(PageMessage::ChatLoad);
        std::thread::sleep(Duration::from_millis(100));
        assert!(!drain(&h).iter().any(|m| matches!(m, DockMessage::ChatDialog { .. })), "a dead Pi's dialog was replayed");
    }

    #[test]
    fn the_header_shows_the_folder_pi_really_works_in() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        let mut settings = settings::load(&h.paths).unwrap().unwrap();
        settings.sandbox.projects_dir = h._dir.path().join("elsewhere").display().to_string();
        settings::save(&h.paths, &settings).unwrap();
        h.chat.handle(PageMessage::ChatLoad);
        match wait_for(&h, "chatState after the change", |m| matches!(m, DockMessage::ChatState { .. })) {
            DockMessage::ChatState { workdir, .. } => assert_eq!(workdir, h.workdir.display().to_string()),
            _ => unreachable!(),
        }
    }

    #[test]
    fn the_notification_stays_until_dismissed_and_a_click_shows_the_dock() {
        let args = notify_args(std::path::Path::new("/opt/aria"), "Aria needs an answer", "Allow?");
        assert_eq!(
            args,
            ["--app-name", "aria", "-u", "critical", "-t", "0", "Aria needs an answer", "Allow?", "--exec", "/opt/aria", "show"]
        );
    }
}
