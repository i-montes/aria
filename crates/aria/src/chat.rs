//! The chat: owns Pi (started when first needed, kept while the dock is hidden), turns page
//! messages into Pi commands and Pi's events into dock messages, and asks for attention with a
//! desktop notification when an extension dialog arrives while the dock is hidden.

use aria_core::agent;
use aria_core::bridge::{DockMessage, PageMessage};
use aria_core::files;
use aria_core::paths::AriaPaths;
use aria_core::pi::{self, PiError};
use aria_core::sessions::{self, SessionList};
use aria_core::settings::{self, Settings};
use aria_engine::engine::{Engine, EngineEvent, SpawnSpec};
use aria_engine::protocol::UiAnswer;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
#[cfg(target_os = "linux")]
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub type Emit = Arc<dyn Fn(DockMessage) + Send + Sync>;

const NOTIFICATION_TITLE: &str = "Aria needs an answer";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const FILE_LIMIT: usize = 50;
const QUOTA_MAX_AGE: Duration = Duration::from_secs(20);

/// Raises and removes the desktop notification for a prompt nobody can see.
///
/// `key` is the session the prompt belongs to, so several sessions can share one notification.
pub trait Notifier: Send + Sync {
    fn send(&self, key: &str, title: &str, body: &str);
    fn dismiss(&self, key: &str);
}

/// Wraps a notifier for several sessions: the real notification is removed only when no session has one left.
pub struct CountingNotifier {
    inner: Arc<dyn Notifier>,
    live: Mutex<HashSet<String>>,
}

impl CountingNotifier {
    pub fn new(inner: Arc<dyn Notifier>) -> CountingNotifier {
        CountingNotifier { inner, live: Mutex::new(HashSet::new()) }
    }
}

impl Notifier for CountingNotifier {
    fn send(&self, key: &str, title: &str, body: &str) {
        self.live.lock().unwrap().insert(key.to_string());
        self.inner.send(key, title, body);
    }

    fn dismiss(&self, key: &str) {
        let last = {
            let mut live = self.live.lock().unwrap();
            live.remove(key) && live.is_empty()
        };
        if last {
            self.inner.dismiss(key);
        }
    }
}

/// Omarchy's notifications: critical urgency and no expiry, so it stays until it is dismissed or
/// clicked; the click runs `aria show`, which opens the dock.
#[cfg(target_os = "linux")]
pub struct OmarchyNotifier;

/// The arguments for `omarchy-notification-send`.
#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
impl Notifier for OmarchyNotifier {
    fn send(&self, _key: &str, title: &str, body: &str) {
        if let Ok(exe) = std::env::current_exe() {
            detached("omarchy-notification-send", &notify_args(&exe, title, body));
        }
    }

    fn dismiss(&self, _key: &str) {
        detached("omarchy-notification-dismiss", &[NOTIFICATION_TITLE.to_string()]);
    }
}

/// The notifier the binary uses: Omarchy's on Linux, macOS's on Mac.
#[cfg(target_os = "linux")]
pub use OmarchyNotifier as PlatformNotifier;

/// macOS notifications, through `osascript` for now. A notification from a program without an app
/// bundle cannot carry a click action, so clicking it does not open Aria yet (docs/pending.md).
#[cfg(target_os = "macos")]
pub struct MacNotifier;

/// A string literal for AppleScript: backslashes and quotes escaped.
#[cfg(target_os = "macos")]
pub fn apple_script_string(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}

#[cfg(target_os = "macos")]
impl Notifier for MacNotifier {
    fn send(&self, _key: &str, title: &str, body: &str) {
        let script = format!("display notification {} with title {}", apple_script_string(body), apple_script_string(title));
        detached("osascript", &["-e".to_string(), script]);
    }

    fn dismiss(&self, _key: &str) {}
}

#[cfg(target_os = "macos")]
pub use MacNotifier as PlatformNotifier;

pub struct ChatCtx {
    pub paths: AriaPaths,
    pub notifier: Arc<dyn Notifier>,
    pub dock_visible: Arc<AtomicBool>,
    /// Is the chat what the dock shows? False while the settings are open: a prompt then needs a notification too.
    pub chat_shown: Arc<AtomicBool>,
    /// Where the sessions' subagent sockets live (`$XDG_RUNTIME_DIR/aria`).
    pub run_dir: PathBuf,
    /// The `PATH` Pi inherits (the shim directory goes in front of it).
    pub path_var: String,
    /// Reads a provider's quota (a fake in tests).
    pub quota_fetcher: Arc<aria_core::quota::Fetcher>,
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
    /// How full the context window is (percent), from `get_session_stats`; `None` until known or just after a compaction.
    context: Option<f64>,
    /// The provider's quota windows as last read, and when (so a run that follows another does not ask again).
    quota: Option<Vec<aria_core::quota::Window>>,
    quota_at: Option<std::time::Instant>,
    /// The notices of `tell_once` were given.
    told: bool,
    /// The socket this session's subagents stream to (kept across Pi restarts, removed when the tab closes).
    watch: Option<Arc<crate::watch::Watch>>,
    /// A restart was asked while a run was going: it happens when the run settles.
    restart_pending: bool,
    /// Prompts sent and not yet accepted, by command id, so a refusal can name its text.
    pending_prompts: HashMap<String, String>,
    /// The arguments the running Pi was started with (tests read them).
    spawn_args: Vec<String>,
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
    /// The session (tab) this chat is: Pi's `--session-id` and the notifications' key.
    session: String,
    ctx: Arc<ChatCtx>,
    emit: Emit,
    inner: Arc<Mutex<Inner>>,
    /// Starting Pi is check-then-act across threads (a send and a `/` picker can race); one at a time.
    spawn_lock: Arc<Mutex<()>>,
}

/// The notification's text: a card's summary (its path, its quota, its first question), or the dialog's own title.
fn dialog_body(raw: &Value) -> String {
    let title = raw["title"].as_str().or_else(|| raw["message"].as_str()).unwrap_or_default();
    let card = serde_json::from_str::<Value>(title).ok().filter(|card| card["aria"] == json!(1));
    let Some(card) = card else { return title.to_string() };
    let text = |key: &str| card[key].as_str().filter(|s| !s.is_empty()).map(str::to_string);
    match card["kind"].as_str() {
        // Never the raw command (it may hold a token, and notifications stay in history): its summary, or a path.
        Some("sandbox") => text("summary").or_else(|| Some("Aria wants to run a command outside the sandbox".into())),
        Some("write") => text("summary").or_else(|| text("path").map(|p| format!("Aria wants to write {p}"))),
        Some("quota") => card["percent"].as_f64().map(|p| format!("Quota at {}%", p.round())),
        Some("question") => card["questions"][0]["question"].as_str().map(str::to_string),
        _ => None,
    }
    .unwrap_or_else(|| "Aria is waiting for your answer".to_string())
}

fn is_dialog(method: &str) -> bool {
    matches!(method, "select" | "confirm" | "input" | "editor")
}

impl Chat {
    pub fn new(ctx: Arc<ChatCtx>, emit: Emit, session: String) -> Chat {
        Chat { session, ctx, emit, inner: Arc::new(Mutex::new(Inner::default())), spawn_lock: Arc::new(Mutex::new(())) }
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
            context: inner.context,
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
            PageMessage::ChatClearQueue { keep } => {
                if let Some(engine) = self.current_engine() {
                    // Pi can only empty its queue; the messages that stay are queued again, in their order.
                    std::thread::spawn(move || {
                        let _ = engine.request("clear_queue", json!({}), REQUEST_TIMEOUT);
                        for text in keep {
                            let _ = engine.send("prompt", json!({"message": text, "streamingBehavior": "followUp"}));
                        }
                    });
                }
            }
            PageMessage::ChatAbort => {
                if let Some(engine) = self.current_engine() {
                    let _ = engine.send("clear_queue", json!({}));
                    let _ = engine.send("abort", json!({}));
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

    /// Is `engine` still this chat's Pi? A reply that comes back after a restart or a close must say nothing.
    fn is_current(&self, engine: &Engine) -> bool {
        self.inner.lock().unwrap().engine.as_deref().is_some_and(|e| std::ptr::eq(e, engine))
    }

    /// The session's subagent socket, opened the first time it is needed. Without one the subagents still run, unwatched.
    fn watch(&self) -> Option<Arc<crate::watch::Watch>> {
        if let Some(watch) = self.inner.lock().unwrap().watch.clone() {
            return Some(watch);
        }
        let path = crate::watch::socket_path(&self.ctx.run_dir, &self.session);
        let slot: Arc<std::sync::OnceLock<std::sync::Weak<crate::watch::Watch>>> = Arc::new(std::sync::OnceLock::new());
        let (chat, seen) = (self.clone(), slot.clone());
        let watch = Arc::new(
            crate::watch::Watch::bind(&path, Arc::new(move || {
                if let Some(watch) = seen.get().and_then(std::sync::Weak::upgrade) {
                    chat.emit(DockMessage::ChatAgents { agents: watch.agents() });
                }
            }))
            .ok()?,
        );
        let _ = slot.set(Arc::downgrade(&watch));
        self.inner.lock().unwrap().watch = Some(watch.clone());
        Some(watch)
    }

    /// Once per tab: files of Aria the owner edited (kept, so updates to them are not applied), and missing packages.
    fn tell_once(&self, kept: &[String]) {
        if std::mem::replace(&mut self.inner.lock().unwrap().told, true) {
            return;
        }
        let notice = |message: String| {
            self.emit(DockMessage::ChatEvent { event: json!({"type": "ui_notice", "request": {"method": "notify", "message": message, "notifyType": "warning"}}) });
        };
        if !kept.is_empty() {
            notice(format!("Aria kept your edited {} in ~/.aria/agent: its own updates to them are not applied. To get Aria's version back, delete the file and its line in ~/.aria/agent/.aria-files.", kept.join(", ")));
        }
        if !agent::packages_installed(&self.ctx.paths) {
            notice("Some packages the extensions use are missing (subagents, skills). Run the Aria installer again.".into());
        }
    }

    /// The tab closed: its subagent socket goes away.
    pub fn forget_watch(&self) {
        self.inner.lock().unwrap().watch = None;
    }

    fn current_engine(&self) -> Option<Arc<Engine>> {
        self.inner.lock().unwrap().engine.clone().filter(|engine| engine.is_alive())
    }

    fn load(&self) {
        let problem = self.problem();
        self.emit(self.state_message(problem));
        if let Some(windows) = self.inner.lock().unwrap().quota.clone() {
            self.emit(DockMessage::ChatQuota { windows });
        }
        if let Some(watch) = self.inner.lock().unwrap().watch.clone() {
            self.emit(DockMessage::ChatAgents { agents: watch.agents() });
        }
        if problem.is_none() {
            // Showing a session wakes its Pi: that is what brings the conversation (and its model) back.
            let _ = self.ensure_engine();
        }
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
        match engine.send("prompt", fields) {
            Ok(id) => {
                self.inner.lock().unwrap().pending_prompts.insert(id, text.to_string());
            }
            Err(_) => self.emit(DockMessage::ChatEvent {
                event: json!({"type": "command_failed", "command": "prompt", "error": "Pi is no longer running", "text": text}),
            }),
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
        // Settings that exist but cannot be read stop the start: falling back to defaults would run a Pi that is not
        // the one the owner set up (other persona, no memory, other models), without a word.
        let settings = match settings::load(&self.ctx.paths) {
            Ok(settings) => settings,
            Err(settings::SettingsError::File(aria_core::json_file::JsonFileError::Io { .. })) => return Err("settings-unreadable"),
            Err(_) => return Err("settings-invalid"),
        };
        let kept = agent::prepare_reporting(&self.ctx.paths, settings.as_ref()).map_err(|error| match error.kind() {
            std::io::ErrorKind::InvalidData => "pi-settings-invalid",
            _ => "agent-dir-unwritable",
        })?;
        let save_sessions = settings.as_ref().is_some_and(|s| s.subagents.save_sessions);
        pi::write_shim(&self.ctx.paths, &install, save_sessions).map_err(|_| "agent-dir-unwritable")?;
        let workdir = working_folder(settings.as_ref()).ok_or("workdir-unavailable")?;
        self.tell_once(&kept);

        self.emit(DockMessage::ChatEngine { state: "starting".into() });
        let spec = SpawnSpec {
            program: install.binary,
            // One sessions folder, whatever the working folder: a changed projects folder must not lose a tab's history.
            args: vec![
                "--mode".into(),
                "rpc".into(),
                "--session-dir".into(),
                agent::sessions_dir(&self.ctx.paths).display().to_string(),
                "--session-id".into(),
                self.session.clone(),
            ],
            env: pi::session_env(&self.ctx.paths, &self.ctx.path_var, &workdir),
            unset: pi::unset_for_pi(std::env::vars()),
            cwd: workdir.clone(),
        };
        let mut spec = spec;
        if let Some(watch) = self.watch() {
            spec.env.push(("PI_WATCH_SOCK".into(), watch.path().display().to_string()));
        }
        let spec_args = spec.args.clone();
        let (engine, events) = Engine::spawn(spec).map_err(|_| "pi-failed")?;
        let engine = Arc::new(engine);
        let (generation, stale) = {
            let mut inner = self.inner.lock().unwrap();
            inner.generation += 1;
            inner.engine = Some(engine.clone());
            inner.running = false;
            inner.workdir = Some(workdir);
            inner.spawn_args = spec_args.clone();
            inner.pending_prompts.clear();
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

    /// Reads how full the context window is and tells the page.
    fn refresh_context(&self, engine: &Engine) {
        let Ok(response) = engine.request("get_session_stats", json!({}), REQUEST_TIMEOUT) else { return };
        if !response.success || !self.is_current(engine) {
            return;
        }
        self.inner.lock().unwrap().context = response.data["contextUsage"]["percent"].as_f64();
        self.emit(self.state_message(None));
    }

    /// Reads the quota of the model's provider and tells the page. Unless `force`, a reading less than 20 s old is kept (runs
    /// can follow each other closely); a model change always asks. A model whose provider has nothing to read reports none.
    fn refresh_quota(&self, force: bool) {
        let model = {
            let inner = self.inner.lock().unwrap();
            if !force && inner.quota_at.is_some_and(|at| at.elapsed() < QUOTA_MAX_AGE) {
                return;
            }
            inner.model.clone().or_else(|| self.settings().and_then(|s| s.models.daily))
        };
        let Some(model) = model else { return };
        let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0);
        let source = aria_core::quota::source_for(&model, &aria_core::auth::AuthStore::new(self.ctx.paths.auth()), now_ms);
        self.inner.lock().unwrap().quota_at = Some(std::time::Instant::now());
        let chat = self.clone();
        std::thread::spawn(move || {
            let windows = source.and_then(|s| aria_core::quota::fetch(&s, &*chat.ctx.quota_fetcher)).unwrap_or_default();
            chat.store_quota_for(&model, windows);
        });
    }

    /// Tells the page what Pi is using and replays the session's history.
    fn announce(&self, engine: &Engine) {
        // A cold start can be slow (packages, MCP servers): wait longer than for an ordinary request, and ask twice.
        let ask = |command: &str| {
            (0..2).find_map(|_| engine.request(command, json!({}), STARTUP_TIMEOUT).ok().filter(|r| r.success))
        };
        if let Some(response) = ask("get_state") {
            if !self.is_current(engine) {
                return;
            }
            apply_state(&mut self.inner.lock().unwrap(), &response.data);
            self.emit(self.state_message(None));
        }
        if let Some(response) = ask("get_messages") {
            if !self.is_current(engine) {
                return;
            }
            self.emit(DockMessage::ChatEvent { event: for_page(json!({"type": "history", "messages": response.data["messages"]})) });
        }
        self.refresh_context(engine);
        // Not forced: a run that already settled may have asked a moment ago.
        self.refresh_quota(false);
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
                    chat.refresh_quota(true);
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
            self.ctx.notifier.dismiss(&self.session);
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
                        // The tasks of a `parallel` call name the subagents that connect next.
                        Some("tool_execution_start") if raw["toolName"] == "parallel" => {
                            if let Some(watch) = self.inner.lock().unwrap().watch.clone() {
                                watch.expect_tasks(&raw["args"]);
                            }
                        }
                        Some("agent_settled") => {
                            // Pi drops a run's dialogs silently when the run is aborted: they go here too.
                            let (open, restart) = {
                                let mut inner = self.inner.lock().unwrap();
                                inner.running = false;
                                let restart = std::mem::take(&mut inner.restart_pending);
                                (inner.dialogs.keys().cloned().collect::<Vec<_>>(), restart)
                            };
                            for id in open {
                                self.close_dialog(&id);
                            }
                            if restart {
                                self.emit(DockMessage::ChatEvent { event: for_page(raw.clone()) });
                                self.shutdown();
                                continue;
                            }
                            if let Some(engine) = self.current_engine() {
                                let chat = self.clone();
                                std::thread::spawn(move || chat.refresh_context(&engine));
                                self.refresh_quota(false);
                            }
                        }
                        _ => {}
                    }
                    self.emit(DockMessage::ChatEvent { event: for_page(raw) });
                }
                EngineEvent::UiRequest { id, method, raw } if is_dialog(&method) => self.open_dialog(id, method, raw),
                EngineEvent::UiRequest { raw, .. } => {
                    self.emit(DockMessage::ChatEvent { event: json!({"type": "ui_notice", "request": raw}) });
                }
                EngineEvent::Response { id, success, command, error } => {
                    // A refused prompt names its text, so the page can take it back out of the thread or the queue.
                    let text = id.and_then(|id| self.inner.lock().unwrap().pending_prompts.remove(&id));
                    if !success {
                        self.emit(DockMessage::ChatEvent { event: json!({"type": "command_failed", "command": command, "error": error, "text": text}) });
                    }
                }
                EngineEvent::Exited { code } => {
                    self.exited(generation, code);
                    break;
                }
            }
        }
    }

    fn close_stale(&self, (open, dismiss): (Vec<String>, bool)) {
        if dismiss {
            self.ctx.notifier.dismiss(&self.session);
        }
        for id in open {
            self.emit(DockMessage::ChatDialogClosed { id });
        }
    }

    /// The dock's visibility changed: when it is now hidden, every open dialog that has no notification
    /// yet gets one (the owner may have pressed Escape with a question still open).
    /// Can the owner see this chat? The dock is open and showing the chat, not the settings.
    fn seen(&self) -> bool {
        self.ctx.dock_visible.load(Ordering::SeqCst) && self.ctx.chat_shown.load(Ordering::SeqCst)
    }

    pub fn dock_visibility_changed(&self) {
        if self.seen() {
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
            self.ctx.notifier.send(&self.session, NOTIFICATION_TITLE, &body);
        }
    }

    fn open_dialog(&self, id: String, method: String, raw: Value) {
        let hidden = !self.seen();
        {
            let mut inner = self.inner.lock().unwrap();
            inner.dialogs.insert(id.clone(), raw.clone());
            if hidden {
                inner.notified.insert(id.clone());
            }
        }
        if hidden {
            self.ctx.notifier.send(&self.session, NOTIFICATION_TITLE, &dialog_body(&raw));
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
            self.ctx.notifier.dismiss(&self.session);
        }
        for id in open {
            self.emit(DockMessage::ChatDialogClosed { id });
        }
        self.emit(DockMessage::ChatEngine { state: "exited".into() });
        self.emit(DockMessage::ChatEvent { event: json!({"type": "engine_exited", "code": code}) });
    }

    /// The folder this chat's running Pi works in (None when no Pi runs).
    fn workdir(&self) -> Option<PathBuf> {
        let inner = self.inner.lock().unwrap();
        inner.engine.as_ref().and(inner.workdir.clone())
    }

    /// Restart for a changed credential or package: at once when idle, when the current run settles otherwise.
    pub fn restart_when_idle(&self) {
        let running = {
            let mut inner = self.inner.lock().unwrap();
            if inner.running {
                inner.restart_pending = true;
            }
            inner.running
        };
        if !running {
            self.shutdown();
        }
    }

    /// Pi stops (stdin closed, killed after three seconds), its dialogs and notification go away, and the page is told.
    /// What Pi started keeps running: a restart must not stop the owner's dev server.
    pub fn shutdown(&self) {
        self.stop(false);
    }

    /// The session is closed for good: Pi stops, and so does everything it started (MCP servers, subagents, a dev
    /// server), so nothing of a closed tab keeps running.
    pub fn close_for_good(&self) {
        self.stop(true);
    }

    fn stop(&self, group: bool) {
        let (stale, engine) = {
            let mut inner = self.inner.lock().unwrap();
            inner.generation += 1; // whatever the old Pi still says is ignored
            let engine = inner.engine.take();
            inner.running = false;
            inner.restart_pending = false;
            // A credential may have changed (that is why Pis restart): the quota is read afresh next time.
            inner.quota = None;
            inner.quota_at = None;
            let open: Vec<String> = inner.dialogs.drain().map(|(id, _)| id).collect();
            let dismiss = !inner.notified.is_empty();
            inner.notified.clear();
            ((open, dismiss), engine)
        };
        self.close_stale(stale);
        if let Some(engine) = engine {
            std::thread::spawn(move || if group { engine.close_group(Duration::from_secs(3)) } else { engine.close(Duration::from_secs(3)) });
            self.emit(DockMessage::ChatEngine { state: "stopped".into() });
            self.emit(self.state_message(None));
        }
    }

    /// Keeps a quota reading only if it is for the model in use now (a reading for the previous model can land late).
    fn store_quota_for(&self, model: &str, windows: Vec<aria_core::quota::Window>) {
        {
            let mut inner = self.inner.lock().unwrap();
            let now = inner.model.clone().or_else(|| self.settings().and_then(|s| s.models.daily));
            if now.as_deref() != Some(model) {
                return;
            }
            inner.quota = Some(windows.clone());
        }
        self.emit(DockMessage::ChatQuota { windows });
    }

    #[cfg(test)]
    fn test_spawn_args(&self) -> Vec<String> {
        self.inner.lock().unwrap().spawn_args.clone()
    }

    #[cfg(test)]
    fn test_send_raw(&self, kind: &str) {
        if let Some(engine) = self.current_engine() {
            let _ = engine.send(kind, json!({}));
        }
    }
}

/// The dock's tabs: one `Chat` (and so one Pi) per session id, created the first time a message names it.
#[derive(Clone)]
pub struct Sessions {
    ctx: Arc<ChatCtx>,
    emit: Emit,
    chats: Arc<Mutex<HashMap<String, Chat>>>,
    closed: Arc<Mutex<HashSet<String>>>,
}

/// Watches `settings.json` by content: a change from the settings screen, from `/effort` or `/autocompact` in Pi, or
/// from an editor is seen once, and the same content written again is not a change.
pub struct SettingsWatch {
    path: PathBuf,
    last: Option<u64>,
}

impl SettingsWatch {
    pub fn new(path: PathBuf) -> SettingsWatch {
        let last = Self::hash(&path);
        SettingsWatch { path, last }
    }

    fn hash(path: &std::path::Path) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let bytes = std::fs::read(path).ok()?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        Some(hasher.finish())
    }

    pub fn changed(&mut self) -> bool {
        let now = Self::hash(&self.path);
        if now.is_none() || now == self.last {
            return false;
        }
        self.last = now;
        true
    }
}

impl Sessions {
    /// The credentials or packages changed: every session's Pi starts again, a busy one when its run settles, never
    /// during it; the next start prepares the agent directory.
    pub fn restart_all(&self) {
        let chats: Vec<Chat> = self.chats.lock().unwrap().values().cloned().collect();
        for chat in chats {
            chat.restart_when_idle();
        }
    }

    /// `settings.json` changed. The agent directory is prepared again at once (`aria.json`, which the extensions read on
    /// every use, so `/effort`, the quota and the sandbox apply without a restart). Only when something Pi reads at start
    /// changed (the persona, the experts, its settings with the compaction reserves, `mcp.json`, the shim, the working
    /// folder) do the Pis restart, each when its run settles: a restart loses plan mode, the sandbox's session grants and
    /// the quota switch.
    pub fn settings_changed(&self) {
        let paths = &self.ctx.paths;
        let Ok(settings) = settings::load(paths) else { return }; // a broken file: the next start says so
        let before = agent::start_fingerprint(paths);
        if agent::prepare(paths, settings.as_ref()).is_err() {
            return;
        }
        if let Ok(install) = pi::locate(paths, pi::pinned_version()) {
            let _ = pi::write_shim(paths, &install, settings.as_ref().is_some_and(|s| s.subagents.save_sessions));
        }
        if agent::start_fingerprint(paths) != before {
            return self.restart_all();
        }
        // The folder a Pi works in is fixed when it starts: one working in another folder than the settings' starts again.
        let wanted = agent::workdir(settings.as_ref());
        let chats: Vec<Chat> = self.chats.lock().unwrap().values().cloned().collect();
        for chat in chats.into_iter().filter(|chat| chat.workdir().is_some_and(|dir| dir != wanted)) {
            chat.restart_when_idle();
        }
    }

    /// `ctx.notifier` is wrapped so that one session's answer does not remove another's notification.
    pub fn new(ctx: ChatCtx, emit: Emit) -> Sessions {
        let notifier: Arc<dyn Notifier> = Arc::new(CountingNotifier::new(ctx.notifier.clone()));
        let ctx = Arc::new(ChatCtx { notifier, ..ctx });
        Sessions { ctx, emit, chats: Arc::new(Mutex::new(HashMap::new())), closed: Arc::new(Mutex::new(HashSet::new())) }
    }

    fn chat(&self, session: &str) -> Option<Chat> {
        // A late reply for a closed tab must not bring its Pi back.
        if !sessions::valid_id(session) || self.closed.lock().unwrap().contains(session) {
            return None;
        }
        let mut chats = self.chats.lock().unwrap();
        let chat = chats.entry(session.to_string()).or_insert_with(|| {
            let emit = self.emit.clone();
            let tag = session.to_string();
            let tagged: Emit = Arc::new(move |message| emit(DockMessage::Session { session: tag.clone(), message: Box::new(message) }));
            Chat::new(self.ctx.clone(), tagged, session.to_string())
        });
        Some(chat.clone())
    }

    pub fn handle(&self, message: PageMessage) {
        match message {
            PageMessage::Session { session, message } => {
                if matches!(*message, PageMessage::Session { .. }) {
                    return;
                }
                if let Some(chat) = self.chat(&session) {
                    chat.handle(*message);
                }
            }
            PageMessage::SessionsLoad { id } => {
                let list = sessions::load(&self.ctx.paths);
                (self.emit)(DockMessage::Sessions { id, sessions: list.sessions, active: list.active });
            }
            PageMessage::SessionsSave { sessions: entries, active } => {
                let _ = self.ctx.paths.ensure_root();
                let _ = sessions::save(&self.ctx.paths, &SessionList { sessions: entries, active });
            }
            // Credentials or packages changed.
            PageMessage::EnginesRestart => self.restart_all(),
            PageMessage::ChatShown { shown } => {
                self.ctx.chat_shown.store(shown, Ordering::SeqCst);
                if !shown {
                    self.dock_visibility_changed();
                }
            }
            PageMessage::SessionClose { session } => {
                self.closed.lock().unwrap().insert(session.clone());
                let chat = self.chats.lock().unwrap().remove(&session);
                if let Some(chat) = chat {
                    chat.close_for_good();
                    chat.forget_watch();
                }
            }
            _ => {}
        }
    }

    pub fn dock_visibility_changed(&self) {
        let chats: Vec<Chat> = self.chats.lock().unwrap().values().cloned().collect();
        for chat in chats {
            chat.dock_visibility_changed();
        }
    }

    #[cfg(test)]
    fn test_send_raw(&self, session: &str, kind: &str) {
        if let Some(chat) = self.chat(session) {
            chat.test_send_raw(kind);
        }
    }
}

/// Pi's folder: the configured one, created when missing. `None` when it cannot be made: never another folder in its
/// place (the sandbox lets Pi write in it).
/// A Pi event as the page gets it: images keep their type and kind but lose their data. The page never shows them, and
/// a session's screenshots can make its history tens of megabytes, which the dock's web view does not survive.
fn for_page(mut event: Value) -> Value {
    fn strip(value: &mut Value) {
        match value {
            Value::Object(map) => {
                if map.get("type").and_then(Value::as_str) == Some("image") {
                    map.remove("data");
                }
                map.values_mut().for_each(strip);
            }
            Value::Array(items) => items.iter_mut().for_each(strip),
            _ => {}
        }
    }
    strip(&mut event);
    event
}

fn working_folder(settings: Option<&Settings>) -> Option<PathBuf> {
    let folder = agent::workdir(settings);
    std::fs::create_dir_all(&folder).ok().map(|_| folder)
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
        fn send(&self, _key: &str, title: &str, body: &str) {
            self.log.lock().unwrap().push(format!("send:{title}:{body}"));
        }
        fn dismiss(&self, _key: &str) {
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
        quota_calls: Arc<std::sync::atomic::AtomicUsize>,
        chat_shown: Arc<AtomicBool>,
        run_dir: PathBuf,
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
        std::fs::write(paths.auth(), r#"{"opencode-go":{"type":"api_key","key":"go-key"}}"#).unwrap();
        let quota_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = quota_calls.clone();
        let quota_fetcher: Arc<aria_core::quota::Fetcher> = Arc::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            Some(json!({"usage": {"rolling": {"status": "ok", "percent": 42}, "weekly": {"status": "ok", "percent": 17}, "monthly": {"status": "ok", "percent": 6}}}))
        });
        let chat_shown = Arc::new(AtomicBool::new(true));
        let run_dir = dir.path().join("run");
        let ctx = ChatCtx { paths: paths.clone(), notifier: notifier.clone(), dock_visible: visible.clone(), chat_shown: chat_shown.clone(), run_dir: run_dir.clone(), path_var: "/usr/bin:/bin".into(), quota_fetcher };
        Harness { _turn: turn, quota_calls, chat_shown, run_dir, backlog: Mutex::new(Vec::new()), _dir: dir, paths, chat: Chat::new(Arc::new(ctx), emit, "s1".into()), messages, notifier, visible, workdir }
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
    fn images_never_reach_the_page() {
        let image = json!({"type": "image", "data": "A".repeat(1000), "mimeType": "image/png"});
        let history = json!({"type": "history", "messages": [
            {"role": "user", "content": "look"},
            {"role": "toolResult", "content": [{"type": "text", "text": "a screenshot"}, image.clone()]},
        ]});
        let page = for_page(history);
        assert_eq!(page["messages"][1]["content"][0]["text"], "a screenshot");
        assert_eq!(page["messages"][1]["content"][1], json!({"type": "image", "mimeType": "image/png"}));
        let live = for_page(json!({"type": "tool_execution_end", "result": {"content": [image]}}));
        assert_eq!(live["result"]["content"][0], json!({"type": "image", "mimeType": "image/png"}));
        let text = json!({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "data"}});
        assert_eq!(for_page(text.clone()), text);
    }

    #[test]
    fn chat_load_reports_the_folder_and_model_then_starts_pi_and_replays_the_history() {
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
        // Showing a session wakes its Pi, so the conversation it had comes back without a first message.
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        let history = wait_for(&h, "history", |m| event_of(m, "history"));
        assert!(matches!(history, DockMessage::ChatEvent { ref event } if event["messages"][0]["content"] == json!("earlier")));
    }

    #[test]
    fn chat_load_with_a_missing_pi_only_reports_the_problem() {
        let h = harness_with(false);
        h.chat.handle(PageMessage::ChatLoad);
        wait_for(&h, "the problem", |m| matches!(m, DockMessage::ChatState { problem: Some(p), .. } if p == "pi-missing"));
        assert!(drain(&h).iter().all(|m| !matches!(m, DockMessage::ChatEngine { .. })));
    }

    #[test]
    fn the_engine_reports_its_model_and_thinking_level_once_ready() {
        let h = harness();
        send(&h, "hello");
        let state = wait_for(&h, "state from get_state", |m| matches!(m, DockMessage::ChatState { thinking: Some(_), .. }));
        assert!(matches!(state, DockMessage::ChatState { thinking: Some(t), model: Some(m), .. } if t == "medium" && m == "opencode-go/deepseek-v4.1-flash"));
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
    fn a_tab_is_told_once_when_the_packages_are_missing() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "the notice", |m| matches!(m, DockMessage::ChatEvent { event } if event["request"]["message"].as_str().is_some_and(|t| t.contains("packages"))));
        h.chat.restart_when_idle();
        send(&h, "y");
        wait_for(&h, "ready again", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        std::thread::sleep(Duration::from_millis(200));
        assert!(drain(&h).iter().all(|m| !matches!(m, DockMessage::ChatEvent { event } if event["request"]["message"].as_str().is_some_and(|t| t.contains("packages")))));
    }

    #[test]
    fn notices_from_extensions_are_events_not_dialogs() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.test_send_raw("notify");
        let notice = wait_for(&h, "the notice", |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "ui_notice" && event["request"]["message"] == "careful"));
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
        let ctx = ChatCtx { paths: h.paths.clone(), notifier: h.notifier.clone(), dock_visible: h.visible.clone(), chat_shown: Arc::new(AtomicBool::new(true)), run_dir: h.run_dir.clone(), path_var: "/nonexistent".into(), quota_fetcher: Arc::new(|_| None) };
        let (sender, messages) = mpsc::channel();
        let sender = Mutex::new(sender);
        let chat = Chat::new(Arc::new(ctx), Arc::new(move |m| { let _ = sender.lock().unwrap().send(m); }), "s1".into());
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


    // ---- sessions: one Pi per tab -------------------------------------------------------------

    struct SessionsHarness {
        h: Harness,
        sessions: Sessions,
        messages: Receiver<DockMessage>,
    }

    fn sessions_harness() -> SessionsHarness {
        let h = harness();
        let (sender, messages) = mpsc::channel();
        let sender = Mutex::new(sender);
        let emit: Emit = Arc::new(move |m| {
            let _ = sender.lock().unwrap().send(m);
        });
        let ctx = ChatCtx { paths: h.paths.clone(), notifier: h.notifier.clone(), dock_visible: h.visible.clone(), chat_shown: Arc::new(AtomicBool::new(true)), run_dir: h.run_dir.clone(), path_var: "/usr/bin:/bin".into(), quota_fetcher: Arc::new(|_| None) };
        let sessions = Sessions::new(ctx, emit);
        SessionsHarness { h, sessions, messages }
    }

    fn in_session(session: &str, message: PageMessage) -> PageMessage {
        PageMessage::Session { session: session.into(), message: Box::new(message) }
    }

    fn wait_session(s: &SessionsHarness, what: &str, session: &str, found: impl Fn(&DockMessage) -> bool) -> DockMessage {
        loop {
            match s.messages.recv_timeout(Duration::from_secs(5)) {
                Ok(DockMessage::Session { session: got, message }) if got == session && found(&message) => return *message,
                Ok(_) => {}
                Err(_) => panic!("timed out waiting for {what} in session {session}"),
            }
        }
    }

    fn settled(m: &DockMessage) -> bool {
        event_of(m, "agent_settled")
    }

    #[test]
    fn two_sessions_run_their_own_pi_at_once() {
        let s = sessions_harness();
        s.sessions.handle(in_session("aa", PageMessage::ChatSend { text: "one".into(), behavior: None }));
        s.sessions.handle(in_session("bb", PageMessage::ChatSend { text: "two".into(), behavior: None }));
        wait_session(&s, "the first reply", "aa", settled);
        wait_session(&s, "the second reply", "bb", settled);
    }

    #[test]
    fn every_message_comes_back_tagged_with_its_session() {
        let s = sessions_harness();
        s.sessions.handle(in_session("aa", PageMessage::ListCommands { id: 4 }));
        let reply = wait_session(&s, "commands", "aa", |m| matches!(m, DockMessage::Commands { id: 4, .. }));
        assert!(matches!(reply, DockMessage::Commands { .. }));
    }

    #[test]
    fn a_session_message_for_an_unknown_id_creates_it_and_a_bad_id_is_ignored() {
        let s = sessions_harness();
        s.sessions.handle(in_session("../evil", PageMessage::ChatSend { text: "x".into(), behavior: None }));
        s.sessions.handle(in_session("fresh-1", PageMessage::ChatLoad));
        wait_session(&s, "its state", "fresh-1", |m| matches!(m, DockMessage::ChatState { .. }));
        assert!(!s.h.paths.agent().join("sessions").join("../evil").exists());
    }

    #[test]
    fn a_dialog_in_one_session_does_not_dismiss_anothers_notification() {
        let s = sessions_harness();
        s.h.visible.store(false, Ordering::SeqCst);
        for id in ["aa", "bb"] {
            s.sessions.handle(in_session(id, PageMessage::ChatSend { text: "x".into(), behavior: None }));
            wait_session(&s, "ready", id, |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        }
        s.sessions.test_send_raw("aa", "ask");
        wait_session(&s, "aa's dialog", "aa", |m| matches!(m, DockMessage::ChatDialog { .. }));
        s.sessions.test_send_raw("bb", "ask");
        wait_session(&s, "bb's dialog", "bb", |m| matches!(m, DockMessage::ChatDialog { .. }));
        assert_eq!(s.h.notifier.log().len(), 2);
        s.sessions.handle(in_session("aa", PageMessage::UiAnswer { id: "u1".into(), value: Some("Allow".into()), confirmed: None, cancelled: None }));
        wait_session(&s, "aa's dialog closing", "aa", |m| matches!(m, DockMessage::ChatDialogClosed { .. }));
        assert!(!s.h.notifier.log().contains(&"dismiss".to_string()), "bb still waits for an answer: {:?}", s.h.notifier.log());
        s.sessions.handle(in_session("bb", PageMessage::UiAnswer { id: "u1".into(), value: Some("Allow".into()), confirmed: None, cancelled: None }));
        wait_session(&s, "bb's dialog closing", "bb", |m| matches!(m, DockMessage::ChatDialogClosed { .. }));
        assert_eq!(s.h.notifier.log().last().map(String::as_str), Some("dismiss"));
    }

    #[test]
    fn closing_a_session_stops_its_pi_and_closes_its_dialogs() {
        let s = sessions_harness();
        s.h.visible.store(false, Ordering::SeqCst);
        s.sessions.handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        s.sessions.test_send_raw("aa", "ask");
        wait_session(&s, "the dialog", "aa", |m| matches!(m, DockMessage::ChatDialog { .. }));
        s.sessions.handle(PageMessage::SessionClose { session: "aa".into() });
        wait_session(&s, "the dialog closing", "aa", |m| matches!(m, DockMessage::ChatDialogClosed { id } if id == "u1"));
        assert_eq!(s.h.notifier.log().last().map(String::as_str), Some("dismiss"));
        // the same id opened again starts a fresh Pi
        // a late message for the closed tab (a picker reply, a model change) does not bring its Pi back
        while s.messages.try_recv().is_ok() {}
        s.sessions.handle(in_session("aa", PageMessage::ListCommands { id: 9 }));
        std::thread::sleep(Duration::from_millis(200));
        let late = s.messages.try_recv();
        assert!(late.is_err(), "nothing answered for a closed tab: {late:?}");
    }

    #[test]
    fn restarting_the_engines_stops_every_pi_and_the_next_use_starts_a_fresh_one() {
        let s = sessions_harness();
        for id in ["aa", "bb"] {
            s.sessions.handle(in_session(id, PageMessage::ChatSend { text: "x".into(), behavior: None }));
            wait_session(&s, "ready", id, |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        }
        s.h.visible.store(false, Ordering::SeqCst);
        s.sessions.test_send_raw("aa", "ask");
        wait_session(&s, "the dialog", "aa", |m| matches!(m, DockMessage::ChatDialog { .. }));
        s.sessions.handle(PageMessage::EnginesRestart);
        wait_session(&s, "the dialog closing", "aa", |m| matches!(m, DockMessage::ChatDialogClosed { .. }));
        // showing a session again wakes a new Pi, which sees the changed credentials
        s.sessions.handle(in_session("bb", PageMessage::ChatLoad));
        wait_session(&s, "a new start", "bb", |m| matches!(m, DockMessage::ChatEngine { state } if state == "starting"));
        wait_session(&s, "ready again", "bb", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
    }

    #[test]
    fn a_settings_change_waits_for_the_run_then_restarts_pi() {
        let s = sessions_harness();
        s.sessions.handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        // the first message's own run (start and settle) must not be mistaken for the one below
        wait_session(&s, "the first run settled", "aa", |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "agent_settled"));
        std::thread::sleep(Duration::from_millis(100));
        while s.messages.try_recv().is_ok() {}
        s.sessions.test_send_raw("aa", "start"); // a run that has not settled
        wait_session(&s, "the run", "aa", |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "agent_start"));
        let mut settings = Settings::defaults("es");
        settings.assistant.name = "Nova".into(); // the persona changes: Pi reads it only at start
        settings::save(&s.h.paths, &settings).unwrap();
        s.sessions.settings_changed();
        std::thread::sleep(Duration::from_millis(200));
        let mut cut = false;
        while let Ok(m) = s.messages.try_recv() {
            cut |= matches!(m, DockMessage::Session { ref message, .. } if matches!(**message, DockMessage::ChatEngine { ref state } if state == "stopped"));
        }
        assert!(!cut, "the run is not cut");
        s.sessions.test_send_raw("aa", "settle");
        wait_session(&s, "stopped once the run settled", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "stopped"));
    }

    fn stopped_within(s: &SessionsHarness, ms: u64) -> bool {
        std::thread::sleep(Duration::from_millis(ms));
        let mut stopped = false;
        while let Ok(m) = s.messages.try_recv() {
            stopped |= matches!(m, DockMessage::Session { ref message, .. } if matches!(**message, DockMessage::ChatEngine { ref state } if state == "stopped"));
        }
        stopped
    }

    #[test]
    fn a_change_the_extensions_read_live_keeps_every_pi_and_its_session_state() {
        let s = sessions_harness();
        settings::save(&s.h.paths, &Settings::defaults("es")).unwrap();
        s.sessions.handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        let mut settings = Settings::defaults("es");
        settings.effort.mode = "fixed".into();
        settings.effort.fixed_level = Some("high".into());
        settings.compaction.percent = 70;
        settings::save(&s.h.paths, &settings).unwrap();
        s.sessions.settings_changed();
        assert!(!stopped_within(&s, 300), "plan mode, session grants and the quota switch survive an /effort");
        let config: Value = serde_json::from_str(&std::fs::read_to_string(s.h.paths.agent().join("aria.json")).unwrap()).unwrap();
        assert_eq!(config["effort"]["fixedLevel"], "high", "the extensions read the change at once");
    }

    #[test]
    fn a_change_pi_reads_only_at_start_restarts_it() {
        let s = sessions_harness();
        settings::save(&s.h.paths, &Settings::defaults("es")).unwrap();
        s.sessions.handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        let mut settings = Settings::defaults("es");
        settings.subagents.experts.iter_mut().for_each(|e| e.enabled = e.name != "security");
        settings::save(&s.h.paths, &settings).unwrap();
        s.sessions.settings_changed();
        wait_session(&s, "a restart", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "stopped"));
    }

    #[test]
    fn a_new_projects_folder_restarts_pi_so_it_works_there() {
        let s = sessions_harness();
        settings::save(&s.h.paths, &Settings::defaults("es")).unwrap();
        s.sessions.handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        let mut settings = Settings::defaults("es");
        let elsewhere = s.h._dir.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        settings.sandbox.projects_dir = elsewhere.display().to_string();
        settings::save(&s.h.paths, &settings).unwrap();
        s.sessions.settings_changed();
        wait_session(&s, "a restart", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "stopped"));
    }

    #[test]
    fn a_change_to_the_settings_file_is_seen_once() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("settings.json");
        let mut watch = SettingsWatch::new(file.clone());
        assert!(!watch.changed(), "no file yet: nothing changed");
        std::fs::write(&file, "{\"a\":1}").unwrap();
        assert!(watch.changed());
        assert!(!watch.changed(), "the same content is not a change");
        std::fs::write(&file, "{\"a\":1}").unwrap();
        assert!(!watch.changed(), "written again with the same content: still no change");
        std::fs::write(&file, "{\"a\":2}").unwrap();
        assert!(watch.changed());
    }

    #[test]
    fn the_session_list_persists_across_restarts() {
        let s = sessions_harness();
        let entries = vec![aria_core::sessions::SessionEntry { id: "aa".into(), title: "Fix the build".into() }];
        s.sessions.handle(PageMessage::SessionsSave { sessions: entries.clone(), active: Some("aa".into()) });
        s.sessions.handle(PageMessage::SessionsLoad { id: 9 });
        match s.messages.recv_timeout(Duration::from_secs(5)).unwrap() {
            DockMessage::Sessions { id: 9, sessions, active } => {
                assert_eq!(sessions, entries);
                assert_eq!(active.as_deref(), Some("aa"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_hidden_dock_notifies_for_dialogs_in_every_session() {
        let s = sessions_harness();
        for id in ["aa", "bb"] {
            s.sessions.handle(in_session(id, PageMessage::ChatSend { text: "x".into(), behavior: None }));
            wait_session(&s, "ready", id, |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
            s.sessions.test_send_raw(id, "ask");
            wait_session(&s, "its dialog", id, |m| matches!(m, DockMessage::ChatDialog { .. }));
        }
        assert!(s.h.notifier.log().is_empty());
        s.h.visible.store(false, Ordering::SeqCst);
        s.sessions.dock_visibility_changed();
        assert_eq!(s.h.notifier.log().len(), 2);
    }


    #[test]
    fn the_context_usage_is_reported_when_pi_is_ready_and_after_every_run() {
        let h = harness();
        send(&h, "hello");
        let first = wait_for(&h, "a state with the context", |m| matches!(m, DockMessage::ChatState { context: Some(_), .. }));
        assert!(matches!(first, DockMessage::ChatState { context: Some(c), .. } if (c - 30.0).abs() < f64::EPSILON));
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        // the settle triggers a fresh reading
        wait_for(&h, "the state after the run", |m| matches!(m, DockMessage::ChatState { context: Some(_), .. }));
    }

    #[test]
    fn clearing_the_queue_reaches_pi_and_is_ignored_without_one() {
        let h = harness();
        h.chat.handle(PageMessage::ChatClearQueue { keep: vec![] });
        assert!(drain(&h).is_empty());
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.handle(PageMessage::ChatClearQueue { keep: vec![] });
        wait_for(&h, "Pi's queue update", |m| event_of(m, "queue_update"));
    }

    #[test]
    fn cancelling_one_queued_message_queues_the_others_again() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "first settle", |m| event_of(m, "agent_settled"));
        drain(&h); // forget the first run's events: only the resend may satisfy the wait below
        h.chat.handle(PageMessage::ChatClearQueue { keep: vec!["the one that stays".into()] });
        // the fake Pi answers a prompt with agent_start: it was sent again after the queue was emptied
        wait_for(&h, "the kept message sent again", |m| event_of(m, "agent_start"));
    }

    #[test]
    fn the_provider_quota_is_reported_once_pi_is_ready() {
        let h = harness();
        send(&h, "hello");
        match wait_for(&h, "the quota", |m| matches!(m, DockMessage::ChatQuota { .. })) {
            DockMessage::ChatQuota { windows } => {
                assert_eq!(windows.iter().map(|w| (w.id.as_str(), w.percent)).collect::<Vec<_>>(), vec![("5h", 42.0), ("week", 17.0), ("month", 6.0)]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn a_model_whose_provider_has_no_quota_to_read_reports_none() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "the first quota", |m| matches!(m, DockMessage::ChatQuota { windows } if !windows.is_empty()));
        h.chat.handle(PageMessage::SetModel { id: 1, model: "anthropic/claude-sonnet-5-5".into() });
        wait_for(&h, "an empty quota for the new model", |m| matches!(m, DockMessage::ChatQuota { windows } if windows.is_empty()));
    }

    #[test]
    fn the_quota_is_not_asked_again_at_every_run_but_is_after_a_model_change() {
        let h = harness();
        send(&h, "one");
        wait_for(&h, "the first quota", |m| matches!(m, DockMessage::ChatQuota { .. }));
        wait_for(&h, "first settle", |m| event_of(m, "agent_settled"));
        send(&h, "two");
        wait_for(&h, "second settle", |m| event_of(m, "agent_settled"));
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(h.quota_calls.load(Ordering::SeqCst), 1, "the second run came right after: no new request");
        h.chat.handle(PageMessage::SetModel { id: 2, model: "opencode-go/glm-5.3".into() });
        wait_for(&h, "saved", |m| matches!(m, DockMessage::Saved { id: 2 }));
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(h.quota_calls.load(Ordering::SeqCst), 2, "a model change asks at once");
    }

    #[test]
    fn showing_a_session_again_replays_the_last_quota() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "the quota", |m| matches!(m, DockMessage::ChatQuota { .. }));
        drain(&h);
        h.chat.handle(PageMessage::ChatLoad);
        wait_for(&h, "the quota again", |m| matches!(m, DockMessage::ChatQuota { windows } if windows.len() == 3));
        assert_eq!(h.quota_calls.load(Ordering::SeqCst), 1, "replayed, not fetched again");
    }

    // ---- review fixes before 2b -------------------------------------------------------------------

    fn ready(h: &Harness) {
        send(h, "x");
        wait_for(h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        wait_for(h, "settled", |m| event_of(m, "agent_settled"));
    }

    #[test]
    fn a_restart_waits_for_a_run_to_settle_and_says_so() {
        let h = harness();
        ready(&h);
        drain(&h); // the first run's events must not satisfy the wait below
        h.chat.test_send_raw("start");
        wait_for(&h, "a run", |m| event_of(m, "agent_start"));
        h.chat.restart_when_idle();
        std::thread::sleep(Duration::from_millis(100));
        assert!(h.chat.current_engine().is_some(), "a run in progress is not cut");
        h.chat.test_send_raw("settle");
        wait_for(&h, "the stop announced", |m| matches!(m, DockMessage::ChatEngine { state } if state == "stopped"));
        wait_for(&h, "an idle state", |m| matches!(m, DockMessage::ChatState { running: false, .. }));
        assert!(h.chat.current_engine().is_none());
    }

    #[test]
    fn a_restart_of_an_idle_session_stops_it_at_once_and_says_so() {
        let h = harness();
        ready(&h);
        h.chat.restart_when_idle();
        wait_for(&h, "the stop announced", |m| matches!(m, DockMessage::ChatEngine { state } if state == "stopped"));
        assert!(h.chat.current_engine().is_none());
    }

    #[test]
    fn dialogs_still_open_when_a_run_settles_are_closed() {
        let h = harness();
        ready(&h);
        h.visible.store(false, Ordering::SeqCst);
        h.chat.test_send_raw("ask");
        wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { .. }));
        h.chat.test_send_raw("settle"); // Pi resolves a dialog silently when the run is aborted
        wait_for(&h, "the dialog closing", |m| matches!(m, DockMessage::ChatDialogClosed { id } if id == "u1"));
        assert_eq!(h.notifier.log().last().map(String::as_str), Some("dismiss"));
    }

    #[test]
    fn the_notification_of_a_card_reads_its_summary_not_its_json() {
        assert_eq!(dialog_body(&json!({"title": "{\"aria\":1,\"kind\":\"sandbox\",\"summary\":\"Run rm -rf build\"}"})), "Run rm -rf build");
        assert_eq!(dialog_body(&json!({"title": "{\"aria\":1,\"kind\":\"write\",\"path\":\"~/notes.md\"}"})), "Aria wants to write ~/notes.md");
        assert_eq!(dialog_body(&json!({"title": "{\"aria\":1,\"kind\":\"sandbox\",\"detail\":\"curl -H token:abc\"}"})), "Aria wants to run a command outside the sandbox");
        assert_eq!(dialog_body(&json!({"title": "{\"aria\":1,\"kind\":\"quota\",\"percent\":98}"})), "Quota at 98%");
        assert_eq!(dialog_body(&json!({"title": "{\"aria\":1,\"kind\":\"question\",\"questions\":[{\"question\":\"Which fallback?\"}]}"})), "Which fallback?");
        assert_eq!(dialog_body(&json!({"title": "Allow?"})), "Allow?");
        assert_eq!(dialog_body(&json!({"title": "{not json"})), "{not json");
        let h = harness();
        ready(&h);
        h.visible.store(false, Ordering::SeqCst);
        h.chat.test_send_raw("ask_card");
        wait_for(&h, "the card", |m| matches!(m, DockMessage::ChatDialog { .. }));
        assert_eq!(h.notifier.log(), vec!["send:Aria needs an answer:Wants to run rm -rf build"]);
    }

    #[test]
    fn a_working_folder_that_cannot_be_made_is_a_problem_not_the_home_folder() {
        let h = harness();
        let mut settings = settings::load(&h.paths).unwrap().unwrap();
        settings.sandbox.projects_dir = "/proc/aria-cannot-be-here".into();
        settings::save(&h.paths, &settings).unwrap();
        send(&h, "hi");
        wait_for(&h, "the problem", |m| matches!(m, DockMessage::ChatState { problem: Some(p), .. } if p == "workdir-unavailable"));
    }

    #[test]
    fn a_refused_prompt_names_its_text_so_the_page_can_take_it_back() {
        let h = harness();
        ready(&h);
        send(&h, "FAIL");
        match wait_for(&h, "the failure", |m| event_of(m, "command_failed")) {
            DockMessage::ChatEvent { event } => {
                assert_eq!(event["text"], json!("FAIL"));
                assert_eq!(event["command"], json!("prompt"));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn a_failed_command_nobody_waited_for_reaches_the_page() {
        let h = harness();
        ready(&h);
        h.chat.handle(PageMessage::SetModel { id: 1, model: "opencode-go/rejected".into() });
        wait_for(&h, "an error reply", |m| matches!(m, DockMessage::Error { id: 1, code } if code == "model-rejected"));
    }

    #[test]
    fn while_the_chat_is_not_shown_a_dialog_notifies_even_with_the_dock_open() {
        let h = harness();
        ready(&h);
        h.chat_shown.store(false, Ordering::SeqCst); // the owner opened the settings
        h.chat.test_send_raw("ask");
        wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { hidden: true, .. }));
        assert_eq!(h.notifier.log().len(), 1);
    }

    #[test]
    fn pi_keeps_its_sessions_in_one_folder_whatever_the_working_folder() {
        let h = harness();
        std::env::set_var("ARIA_TEST_ARGS", "unused");
        ready(&h);
        let args = h.chat.test_spawn_args();
        let at = args.iter().position(|a| a == "--session-dir").expect("--session-dir is passed");
        assert_eq!(args[at + 1], h.paths.agent().join("sessions").join("aria").display().to_string());
    }

    #[test]
    fn a_quota_read_for_an_old_model_does_not_overwrite_the_new_one() {
        let h = harness();
        ready(&h);
        wait_for(&h, "the first quota", |m| matches!(m, DockMessage::ChatQuota { .. }));
        // the fetch for the old model lands after the model changed
        h.chat.store_quota_for("opencode-go/old", vec![]);
        assert_eq!(h.chat.inner.lock().unwrap().quota.as_ref().map(Vec::len), Some(3));
    }

    #[test]
    fn pi_is_told_the_working_folder_for_the_extensions() {
        let h = harness();
        ready(&h);
        h.chat.test_send_raw("env_projects");
        match wait_for(&h, "the env echo", |m| event_of(m, "env")) {
            DockMessage::ChatEvent { event } => assert_eq!(event["projects"], json!(h.workdir.display().to_string())),
            _ => unreachable!(),
        }
    }

    #[test]
    fn settings_aria_cannot_read_stop_the_start_instead_of_falling_back_to_defaults() {
        let h = harness();
        std::fs::write(h.paths.settings(), r#"{"version": 1, "compaction": {"percent": "lots"}}"#).unwrap();
        send(&h, "x");
        match wait_for(&h, "the problem", |m| matches!(m, DockMessage::ChatState { problem: Some(_), .. })) {
            DockMessage::ChatState { problem, .. } => assert_eq!(problem.as_deref(), Some("settings-invalid")),
            _ => unreachable!(),
        }
        assert!(!h.paths.agent().join("aria.json").exists(), "nothing was prepared from defaults");
    }

    #[test]
    fn pi_gets_aria_settings_in_its_env() {
        let h = harness();
        ready(&h);
        h.chat.test_send_raw("env_settings");
        match wait_for(&h, "the env echo", |m| event_of(m, "env")) {
            DockMessage::ChatEvent { event } => assert_eq!(event["settings"], json!(h.paths.settings().display().to_string())),
            _ => unreachable!(),
        }
    }

    #[test]
    fn the_subagents_of_a_session_reach_the_page() {
        use std::io::Write;
        let h = harness();
        ready(&h);
        h.chat.test_send_raw("env_watch");
        let path = match wait_for(&h, "the socket path", |m| event_of(m, "env")) {
            DockMessage::ChatEvent { event } => event["watch"].as_str().unwrap().to_string(),
            _ => unreachable!(),
        };
        assert!(path.ends_with("watch-s1.sock"), "{path}");
        h.chat.test_send_raw("parallel_call"); // names the child that connects next
        wait_for(&h, "the parallel call", |m| event_of(m, "tool_execution_start"));
        let mut child = std::os::unix::net::UnixStream::connect(&path).unwrap();
        writeln!(child, "{}", json!({"t": "prompt", "text": "Task: Review the policy"})).unwrap();
        match wait_for(&h, "the agents", |m| matches!(m, DockMessage::ChatAgents { agents } if agents[0]["role"] == "security")) {
            DockMessage::ChatAgents { agents } => assert_eq!(agents[0]["kind"], "active"),
            _ => unreachable!(),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn closing_a_session_stops_what_its_pi_started() {
        let s = sessions_harness();
        s.sessions.handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        s.sessions.test_send_raw("aa", "background");
        let pid: u32 = match wait_session(&s, "the background process", "aa", |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "background")) {
            DockMessage::ChatEvent { event } => event["pid"].as_str().unwrap().parse().unwrap(),
            _ => unreachable!(),
        };
        let alive = |pid: u32| std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| stat.rsplit(')').next().is_some_and(|rest| !rest.trim_start().starts_with('Z')));
        assert!(alive(pid));
        s.sessions.handle(PageMessage::SessionClose { session: "aa".into() });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while alive(pid) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(!alive(pid), "a process the closed session's Pi started is still running");
    }

    #[test]
    fn closing_the_tab_removes_its_socket() {
        let s = sessions_harness();
        s.sessions.handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        let path = s.h.run_dir.join("watch-aa.sock");
        assert!(path.exists());
        s.sessions.handle(PageMessage::SessionClose { session: "aa".into() });
        std::thread::sleep(Duration::from_millis(100));
        assert!(!path.exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_notification_stays_until_dismissed_and_a_click_shows_the_dock() {
        let args = notify_args(std::path::Path::new("/opt/aria"), "Aria needs an answer", "Allow?");
        assert_eq!(
            args,
            ["--app-name", "aria", "-u", "critical", "-t", "0", "Aria needs an answer", "Allow?", "--exec", "/opt/aria", "show"]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mac_notifications_escape_applescript_literals() {
        assert_eq!(apple_script_string(r#"a "b" \c"#), r#""a \"b\" \\c""#);
    }
}
