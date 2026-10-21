//! The chat: owns Pi (started when first needed, kept while the dock is hidden), turns page
//! messages into Pi commands and Pi's events into dock messages, and asks for attention with a
//! desktop notification when an extension dialog arrives while the dock is hidden.

use crate::jobs::{Job, JobState, Jobs};
use crate::jobs_socket::{self, AgentLaunch, JobsSocket, Launcher};
use aria_core::agent;
use aria_core::bridge::{Attachment, DockMessage, PageMessage};
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
use std::time::{Duration, Instant};

pub type Emit = Arc<dyn Fn(DockMessage) + Send + Sync>;

const NOTIFICATION_TITLE: &str = "Aria needs an answer";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
/// Pi's own commands the `/` menu offers, and how long each may take: a compaction is a model call over the whole context.
const PI_COMMANDS: [(&str, Duration); 3] = [
    ("compact", Duration::from_secs(600)),
    ("get_session_stats", REQUEST_TIMEOUT),
    ("export_html", Duration::from_secs(60)),
];
const FILE_LIMIT: usize = 50;
const QUOTA_MAX_AGE: Duration = Duration::from_secs(20);
/// What a session with background work is told when an update has waited `UPDATE_NOTICE_AFTER` for that work alone.
pub const UPDATE_WAITS: &str =
    "An Aria update is waiting for the background work to finish; stop it from the background list to update now.";
/// Output lines the page gets when its `jobOutput` names no number.
const PAGE_OUTPUT_LINES: usize = 80;

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
        let script =
            format!("display notification {} with title {}", apple_script_string(body), apple_script_string(title));
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
    /// The folder boxed commands cannot see (`$XDG_RUNTIME_DIR`, which the sandbox hides; `None` when unset). A jobs
    /// socket is opened only inside it: anywhere else a boxed command could reach it and start work outside the box.
    pub hidden_dir: Option<PathBuf>,
    /// The `PATH` Pi inherits (the shim directory goes in front of it).
    pub path_var: String,
    /// Reads a provider's quota (a fake in tests).
    pub quota_fetcher: Arc<aria_core::quota::Fetcher>,
    /// The dock's background work, every session's. One for the dock's whole life: its spawner thread is the parent
    /// every job is tied to (`PR_SET_PDEATHSIG`), so the last clone going away ends them all.
    pub jobs: Jobs,
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
    /// The socket this session's Pi reaches its background work through (kept across Pi restarts, so a job that ends
    /// while Pi restarts is heard by the next one; removed when the tab closes).
    jobs: Option<JobsSocket>,
    /// A restart was asked while a run was going: it happens when the run settles.
    restart_pending: bool,
    /// Prompts sent and not yet accepted, by command id, so a refusal can name its text.
    pending_prompts: HashMap<String, String>,
    /// Which models can be sent images (`input` includes `image` in `get_available_models`), by `provider/id`.
    sees_images: HashMap<String, bool>,
    /// The arguments the running Pi was started with (tests read them).
    spawn_args: Vec<String>,
    /// The folder the running Pi was started in (what the header must show, whatever the settings say now).
    workdir: Option<PathBuf>,
    /// When the current Pi last said anything (an event, a reply, a UI request): the turn watchdog's clock.
    last_activity: Option<Instant>,
    /// The tool executions under way (by `toolCallId`, nested ones too): while any is open the watchdog waits, since a
    /// long build or test is silent and has its own timeouts.
    tools: HashSet<String>,
    /// The turn watchdog stopped this run already (it stops a run once; the next `agent_start` clears it).
    watchdog_fired: bool,
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
        // Never the arguments (the detail): what the connector wants to do, in a sentence.
        Some("connector") => text("summary").or_else(|| Some("A connector wants to act".into())),
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
            PageMessage::ChatSend { text, behavior, attachments } => {
                self.send_prompt(&text, behavior.as_deref(), &attachments)
            }
            PageMessage::ChatClearQueue { steer, keep } => {
                if let Some(engine) = self.current_engine() {
                    // Pi can only empty its queue; the messages that stay are queued again, each where it was (Pi takes
                    // the steering ones before any follow-up, so that order is kept too).
                    std::thread::spawn(move || {
                        let _ = engine.request("clear_queue", json!({}), REQUEST_TIMEOUT);
                        let again = steer
                            .into_iter()
                            .map(|text| (text, "steer"))
                            .chain(keep.into_iter().map(|text| (text, "followUp")));
                        for (text, behavior) in again {
                            let _ = engine.send("prompt", json!({"message": text, "streamingBehavior": behavior}));
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
            PageMessage::PiCommand { id, command, args } => self.pi_command(id, command, args),
            // `/reload`: this tab's Pi starts again (reading its extensions, skills and templates), after its run.
            PageMessage::EnginesRestart => self.restart_when_idle(),
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
            crate::watch::Watch::bind(
                &path,
                Arc::new(move || {
                    if let Some(watch) = seen.get().and_then(std::sync::Weak::upgrade) {
                        chat.emit(DockMessage::ChatAgents { agents: watch.agents() });
                    }
                }),
            )
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
        let notice = |message: String| self.notice(message);
        if !kept.is_empty() {
            notice(format!("Aria kept your edited {} in ~/.aria/agent: its own updates to them are not applied. To get Aria's version back, delete the file and its line in ~/.aria/agent/.aria-files.", kept.join(", ")));
        }
        if !agent::packages_installed(&self.ctx.paths) {
            notice(
                "Some packages the extensions use are missing (subagents, skills). Run the Aria installer again."
                    .into(),
            );
        }
    }

    /// A warning in the chat the page says in its own language: a translation key and its values.
    fn notice_key(&self, key: &str, vars: Value) {
        self.emit(DockMessage::ChatEvent {
            event: json!({"type": "ui_notice", "request": {"method": "notify", "key": key, "vars": vars, "notifyType": "warning"}}),
        });
    }

    /// The turn watchdog: it covers the model being silent. A run that got nothing from Pi (no event, reply or UI
    /// request) for `limit` is stopped as the page's stop does (`clear_queue`, `abort`), once per run, and the chat says
    /// why; the run then settles as usual. A dialog waiting for the person or a tool under way pauses it: its clock
    /// starts again when the dialog closes or the last tool ends. A zero `limit` turns it off. True when it stopped the
    /// run.
    pub fn check_stalled(&self, now: Instant, limit: Duration) -> bool {
        let engine = {
            let mut inner = self.inner.lock().unwrap();
            if limit.is_zero() || !inner.running || inner.watchdog_fired {
                return false;
            }
            if !inner.dialogs.is_empty() || !inner.tools.is_empty() {
                inner.last_activity = Some(now);
                return false;
            }
            let last = *inner.last_activity.get_or_insert(now);
            if now.saturating_duration_since(last) < limit {
                return false;
            }
            let Some(engine) = inner.engine.clone() else { return false };
            inner.watchdog_fired = true;
            engine
        };
        let _ = engine.send("clear_queue", json!({}));
        let _ = engine.send("abort", json!({}));
        self.notice_key("chat.watchdog.stopped", json!({"minutes": limit.as_secs().div_ceil(60)}));
        true
    }

    /// A warning in the chat, as an extension's `notify` shows.
    fn notice(&self, message: String) {
        self.emit(DockMessage::ChatEvent {
            event: json!({"type": "ui_notice", "request": {"method": "notify", "message": message, "notifyType": "warning"}}),
        });
    }

    /// The tab closed: its subagent socket goes away.
    pub fn forget_watch(&self) {
        self.inner.lock().unwrap().watch = None;
    }

    /// The session's jobs socket, opened the first time Pi starts. Without one Pi runs without background work (its
    /// tools say they need the dock): when it cannot be bound, or when the run dir is somewhere a boxed command sees.
    fn jobs_socket(&self) -> Option<PathBuf> {
        if let Some(socket) = self.inner.lock().unwrap().jobs.as_ref() {
            return Some(socket.path().to_path_buf());
        }
        if !hidden_from_the_box(&self.ctx.run_dir, self.ctx.hidden_dir.as_deref()) {
            return None;
        }
        let path = jobs_socket::socket_path(&self.ctx.run_dir, &self.session);
        let launcher = Arc::new(SessionLauncher { chat: self.clone() });
        let socket = match JobsSocket::bind(&path, self.session.clone(), self.ctx.jobs.clone(), launcher) {
            Ok(socket) => socket,
            Err(error) => {
                eprintln!("aria: no background work for session {}: the jobs socket: {error}", self.session);
                return None;
            }
        };
        let path = socket.path().to_path_buf();
        self.inner.lock().unwrap().jobs = Some(socket);
        Some(path)
    }

    /// The tab closed: its jobs socket goes away (and with it the launcher, which holds this chat).
    pub fn forget_jobs(&self) {
        self.inner.lock().unwrap().jobs = None;
    }

    /// How a subagent of this session runs: the `pi` shim with the session's environment, in its folder, watched by its
    /// subagent socket. `model` `"session"` is the model and thinking level this session's Pi reported (the daily model
    /// before it did); `"heavy"` is the planning model and its effort, or the session's when none is set. `None` when
    /// there is neither a model nor settings to run with. Never panics: it runs on a socket thread.
    fn agent_launch(&self, model: &str) -> Option<AgentLaunch> {
        let settings = self.settings();
        let (own, thinking, workdir, watch) = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let watch = inner.watch.as_ref().map(|w| w.path().display().to_string());
            (inner.model.clone(), inner.thinking.clone(), inner.workdir.clone(), watch)
        };
        let args = agent_args(model, own, thinking, settings.as_ref())?;
        // Out of the person's repositories, where their test runners and watchers would find them.
        let worktrees = agent::workdir(settings.as_ref()).join(crate::worktree::DIR);
        let workdir = workdir.unwrap_or_else(|| agent::workdir(settings.as_ref()));
        let mut env = pi::session_env(&self.ctx.paths, &self.ctx.path_var, &workdir);
        if let Some(path) = watch {
            env.push(("PI_WATCH_SOCK".into(), path));
        }
        let mut unset = pi::unset_for_pi(std::env::vars());
        // No background work inside background work.
        unset.push("ARIA_JOBS_SOCK".into());
        Some((self.ctx.paths.bin().join("pi"), args, env, unset, worktrees))
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
        let open: Vec<(String, Value)> =
            self.inner.lock().unwrap().dialogs.iter().map(|(id, raw)| (id.clone(), raw.clone())).collect();
        for (id, raw) in open {
            let method = raw["method"].as_str().unwrap_or_default().to_string();
            self.emit(DockMessage::ChatDialog { id, method, request: raw, hidden: false });
        }
    }

    /// Sends a message. Its attachments are saved first (a message that cannot be saved is refused, and its text goes back
    /// to the composer, as Pi's own refusals do); images go to Pi as images only when the session's model can see them.
    fn send_prompt(&self, text: &str, behavior: Option<&str>, attachments: &[Attachment]) {
        let refuse = |error: &str| {
            self.emit(DockMessage::ChatEvent {
                event: json!({"type": "command_failed", "command": "prompt", "error": error, "text": text}),
            })
        };
        let engine = match self.ensure_engine() {
            Ok(engine) => engine,
            Err(code) => {
                self.emit(self.state_message(Some(code)));
                // A message with files is refused like any failed prompt: its text and files go back to the composer.
                if !attachments.is_empty() {
                    refuse("Pi could not start, so the message was not sent");
                }
                return;
            }
        };
        let images_ok = attachments.iter().any(crate::attachments::is_image) && self.model_sees_images(&engine);
        let saved = match crate::attachments::save(
            &self.ctx.paths.attachments(),
            &self.session,
            &crate::attachments::today(),
            attachments,
            images_ok,
        ) {
            Ok(saved) => saved,
            Err(error) => return refuse(&error),
        };
        let (message, images) = crate::attachments::message(text, &saved);
        let mut fields = json!({"message": message});
        if !images.is_empty() {
            fields["images"] = json!(images);
        }
        if let Some(behavior) = behavior {
            fields["streamingBehavior"] = json!(behavior);
        }
        match engine.send("prompt", fields) {
            Ok(id) => {
                // The page's own text: what its thread and queue show, and what a refusal gives back.
                self.inner.lock().unwrap().pending_prompts.insert(id, text.to_string());
            }
            Err(_) => refuse("Pi is no longer running"),
        }
    }

    /// Can the session's model be sent images? Read from `get_available_models` and kept per model until Pi restarts; a
    /// model Pi does not list, or a list Pi does not give, is a model that cannot, and is asked about again next time.
    fn model_sees_images(&self, engine: &Engine) -> bool {
        // Bound first: no settings read while `inner` is locked.
        let chosen = self.inner.lock().unwrap().model.clone();
        let Some(model) = chosen.or_else(|| self.settings().and_then(|s| s.models.daily)) else { return false };
        if let Some(&known) = self.inner.lock().unwrap().sees_images.get(&model) {
            return known;
        }
        let Ok(response) = engine.request("get_available_models", json!({}), REQUEST_TIMEOUT) else { return false };
        let Some(models) = response.data["models"].as_array().filter(|_| response.success) else { return false };
        let mut inner = self.inner.lock().unwrap();
        for listed in models {
            if let (Some(provider), Some(id)) = (listed["provider"].as_str(), listed["id"].as_str()) {
                let image = listed["input"].as_array().is_some_and(|input| input.iter().any(|kind| kind == "image"));
                inner.sees_images.insert(format!("{provider}/{id}"), image);
            }
        }
        inner.sees_images.get(&model).copied().unwrap_or(false)
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
            Err(settings::SettingsError::File(aria_core::json_file::JsonFileError::Io { .. })) => {
                return Err("settings-unreadable")
            }
            Err(_) => return Err("settings-invalid"),
        };
        let kept =
            agent::prepare_reporting(&self.ctx.paths, settings.as_ref()).map_err(|error| match error.kind() {
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
        if let Some(path) = self.jobs_socket() {
            spec.env.push(("ARIA_JOBS_SOCK".into(), path.display().to_string()));
        }
        let spec_args = spec.args.clone();
        let (engine, events) = Engine::spawn(spec).map_err(|_| "pi-failed")?;
        let engine = Arc::new(engine);
        let (generation, stale) = {
            let mut inner = self.inner.lock().unwrap();
            inner.generation += 1;
            inner.engine = Some(engine.clone());
            inner.running = false;
            inner.tools.clear();
            inner.workdir = Some(workdir);
            inner.spawn_args = spec_args.clone();
            inner.pending_prompts.clear();
            // Another Pi may list other models (an extension, a changed models file): ask it again.
            inner.sees_images.clear();
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
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let source =
            aria_core::quota::source_for(&model, &aria_core::auth::AuthStore::new(self.ctx.paths.auth()), now_ms);
        self.inner.lock().unwrap().quota_at = Some(std::time::Instant::now());
        let chat = self.clone();
        std::thread::spawn(move || {
            let windows =
                source.and_then(|s| aria_core::quota::fetch(&s, &*chat.ctx.quota_fetcher)).unwrap_or_default();
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
            self.emit(DockMessage::ChatEvent {
                event: for_page(json!({"type": "history", "messages": response.data["messages"]})),
            });
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

    /// A command of Pi's own from the `/` menu. Only the ones in `PI_COMMANDS` are sent; the page cannot reach the rest.
    fn pi_command(&self, id: u64, command: String, args: Value) {
        let Some(&(command, timeout)) = PI_COMMANDS.iter().find(|(name, _)| *name == command) else {
            let error = Some("command-unknown".into());
            return self.emit(DockMessage::PiReply { id, ok: false, data: Value::Null, error });
        };
        let args = if args.is_object() { args } else { json!({}) };
        let chat = self.clone();
        std::thread::spawn(move || {
            let reply = chat
                .ensure_engine()
                .map_err(str::to_string)
                .and_then(|engine| engine.request(command, args, timeout).map_err(|_| "command-failed".to_string()));
            chat.emit(match reply {
                Ok(response) if response.success => {
                    DockMessage::PiReply { id, ok: true, data: response.data, error: None }
                }
                Ok(response) => DockMessage::PiReply {
                    id,
                    ok: false,
                    data: Value::Null,
                    error: Some(response.error.unwrap_or_else(|| "command-failed".into())),
                },
                Err(code) => DockMessage::PiReply { id, ok: false, data: Value::Null, error: Some(code) },
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
            // The watchdog was paused while the dialog waited: its clock starts now.
            inner.last_activity = Some(Instant::now());
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
            let current = {
                let mut inner = self.inner.lock().unwrap();
                let current = inner.generation == generation;
                if current {
                    inner.last_activity = Some(Instant::now()); // anything Pi says keeps the watchdog quiet
                }
                current
            };
            if !current && !matches!(event, EngineEvent::Exited { .. }) {
                continue; // a Pi that was replaced no longer speaks for the chat
            }
            match event {
                EngineEvent::Event(raw) => {
                    match raw["type"].as_str() {
                        Some("agent_start") => {
                            let mut inner = self.inner.lock().unwrap();
                            inner.running = true;
                            inner.watchdog_fired = false;
                            inner.tools.clear();
                        }
                        // The last tool ending restarts the watchdog's clock (this event already set it).
                        Some("tool_execution_end") => {
                            let id = raw["toolCallId"].as_str().unwrap_or_default();
                            self.inner.lock().unwrap().tools.remove(id);
                        }
                        // The tasks of a `parallel` call name the subagents that connect next.
                        Some("tool_execution_start") => {
                            let watch = {
                                let mut inner = self.inner.lock().unwrap();
                                inner.tools.insert(raw["toolCallId"].as_str().unwrap_or_default().to_string());
                                inner.watch.clone().filter(|_| raw["toolName"] == "parallel")
                            };
                            if let Some(watch) = watch {
                                watch.expect_tasks(&raw["args"]);
                            }
                        }
                        Some("agent_settled") => {
                            // Pi drops a run's dialogs silently when the run is aborted: they go here too.
                            let (open, restart) = {
                                let mut inner = self.inner.lock().unwrap();
                                inner.running = false;
                                inner.tools.clear();
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
                        self.emit(DockMessage::ChatEvent {
                            event: json!({"type": "command_failed", "command": command, "error": error, "text": text}),
                        });
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
            inner.tools.clear();
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

    /// A run is going or a dialog waits for an answer.
    fn busy(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.running || !inner.dialogs.is_empty()
    }

    /// The returned thread is closing Pi; joining it waits until Pi is gone.
    fn stop(&self, group: bool) -> Option<std::thread::JoinHandle<()>> {
        let (stale, engine) = {
            let mut inner = self.inner.lock().unwrap();
            inner.generation += 1; // whatever the old Pi still says is ignored
            let engine = inner.engine.take();
            inner.running = false;
            inner.tools.clear();
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
        let engine = engine?;
        let closing = std::thread::spawn(move || {
            if group {
                engine.close_group(Duration::from_secs(3))
            } else {
                engine.close(Duration::from_secs(3))
            }
        });
        self.emit(DockMessage::ChatEngine { state: "stopped".into() });
        self.emit(self.state_message(None));
        Some(closing)
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

/// The `--model`/`--thinking` a subagent runs with (see `Chat::agent_launch`).
fn agent_args(
    model: &str,
    own: Option<String>,
    thinking: Option<String>,
    settings: Option<&Settings>,
) -> Option<Vec<String>> {
    let with_thinking = |model: String, thinking: Option<String>| {
        let mut args = vec!["--model".to_string(), model];
        if let Some(level) = thinking.filter(|level| !level.is_empty()) {
            args.extend(["--thinking".to_string(), level]);
        }
        args
    };
    let planning = settings.and_then(|s| s.models.planning.clone().map(|m| (m, s.models.planning_effort.clone())));
    match (model, planning) {
        ("heavy", Some((planning, effort))) => Some(with_thinking(planning, Some(effort))),
        _ => match own {
            Some(own) => Some(with_thinking(own, thinking)),
            // Pi's own default when the settings name no daily model either.
            None => {
                settings.map(|s| s.models.daily.clone().map(|daily| with_thinking(daily, None)).unwrap_or_default())
            }
        },
    }
}

/// How the session's jobs socket starts subagents: as this chat's Pi would run them.
struct SessionLauncher {
    chat: Chat,
}

impl Launcher for SessionLauncher {
    fn agent(&self, model: &str) -> Option<AgentLaunch> {
        self.chat.agent_launch(model)
    }

    /// Names the subagent in the session's watch, so its card reads its title rather than the first line of its brief.
    /// It runs on a jobs thread, which must never be taken down by it: a poisoned lock or a panic in the watch is
    /// ignored (the card is then named after its task).
    fn before_spawn(&self, job: &Job) {
        let watch = self.chat.inner.lock().map(|inner| inner.watch.clone()).unwrap_or(None);
        if let Some(watch) = watch {
            let args = json!({"task": job.task, "agent": job.title});
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| watch.expect_tasks(&args)));
        }
    }
}

/// Whether `run_dir` is inside `hidden`, the folder the sandbox hides from boxed commands.
fn hidden_from_the_box(run_dir: &Path, hidden: Option<&Path>) -> bool {
    hidden.is_some_and(|hidden| hidden.is_absolute() && run_dir.starts_with(hidden))
}

/// Every session's jobs, as the page's background list reads them.
pub fn jobs_message(jobs: &Jobs) -> DockMessage {
    DockMessage::Jobs { jobs: json!(jobs.list(None)) }
}

fn is_active(job: &Job) -> bool {
    matches!(job.state, JobState::Queued | JobState::Running)
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
        Sessions {
            ctx,
            emit,
            chats: Arc::new(Mutex::new(HashMap::new())),
            closed: Arc::new(Mutex::new(HashSet::new())),
        }
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
            let tagged: Emit = Arc::new(move |message| {
                emit(DockMessage::Session { session: tag.clone(), message: Box::new(message) })
            });
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
                    let first_look = matches!(*message, PageMessage::ChatLoad);
                    chat.handle(*message);
                    // The background list shows every session's jobs; a session's first look is the page's too.
                    if first_look {
                        (self.emit)(jobs_message(&self.ctx.jobs));
                    }
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
                    chat.forget_jobs();
                }
                self.close_jobs(session);
            }
            PageMessage::JobStop { job } => {
                self.ctx.jobs.stop(&job);
            }
            PageMessage::JobDismiss { job } => self.ctx.jobs.dismiss(&job),
            PageMessage::JobOutput { id, job, lines } => {
                let lines = self.ctx.jobs.output(&job, lines.unwrap_or(PAGE_OUTPUT_LINES)).unwrap_or_default();
                (self.emit)(DockMessage::JobOutput { id, lines });
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

    /// A closed tab's jobs stop (its queued subagents never start), and once they have ended its events are dropped:
    /// nobody will ever acknowledge them. Off the caller's thread: a group has `STOP_GRACE` to go.
    fn close_jobs(&self, session: String) {
        let jobs = self.ctx.jobs.clone();
        jobs.stop_session(&session);
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while jobs.list(Some(&session)).iter().any(is_active) && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(50));
            }
            jobs.forget_session(&session);
        });
    }

    /// At start: the jobs of sessions whose tab is gone (closed while the dock was away, or never saved) drop their
    /// events, which nobody will ever acknowledge.
    pub fn forget_gone_sessions(&self) {
        let open: HashSet<String> =
            sessions::load(&self.ctx.paths).sessions.into_iter().map(|entry| entry.id).collect();
        let stored: HashSet<String> = self.ctx.jobs.list(None).into_iter().map(|job| job.session).collect();
        for gone in stored.difference(&open) {
            self.ctx.jobs.forget_session(gone);
        }
    }

    /// Some session has a run going or a dialog waiting, or background work is running or queued: an update must not
    /// restart the dock now (a restart would end that work).
    pub fn busy(&self) -> bool {
        self.busy_without_jobs() || self.ctx.jobs.active() > 0
    }

    /// Some session has a run going or a dialog waiting (background work aside): when this is false and the dock is
    /// still busy, the update waits for background work alone.
    pub fn busy_without_jobs(&self) -> bool {
        self.chats.lock().unwrap().values().any(Chat::busy)
    }

    /// An update has waited long for background work alone: each session with work running or queued says so, with how
    /// to update now.
    pub fn tell_update_waits(&self) {
        let waiting: HashSet<String> =
            self.ctx.jobs.list(None).into_iter().filter(is_active).map(|job| job.session).collect();
        let chats: Vec<Chat> = {
            let chats = self.chats.lock().unwrap();
            waiting.iter().filter_map(|session| chats.get(session).cloned()).collect()
        };
        for chat in chats {
            chat.notice(UPDATE_WAITS.into());
        }
    }

    /// The turn watchdog's limit, read from the settings on every check (a change applies without a restart). Settings
    /// that cannot be read keep the default.
    fn limit(paths: &AriaPaths) -> Duration {
        let minutes = settings::load(paths)
            .ok()
            .flatten()
            .map(|s| s.models.turn_timeout_minutes)
            .unwrap_or_else(|| aria_core::settings::Models::default().turn_timeout_minutes);
        Duration::from_secs(u64::from(minutes) * 60)
    }

    /// Every 30 s: each session's run that got nothing from its Pi for the settings' limit is stopped (see
    /// `Chat::check_stalled`). Returns how many were.
    pub fn check_stalled(&self, now: Instant) -> usize {
        let limit = Self::limit(&self.ctx.paths);
        if limit.is_zero() {
            return 0;
        }
        let chats: Vec<Chat> = self.chats.lock().unwrap().values().cloned().collect();
        chats.iter().filter(|chat| chat.check_stalled(now, limit)).count()
    }

    /// Every Pi stops, as for a restart (what they started keeps running), and this returns once they are gone, so the
    /// new dock does not open a session an old Pi still writes.
    pub fn stop_for_update(&self) {
        let chats: Vec<Chat> = self.chats.lock().unwrap().values().cloned().collect();
        let closing: Vec<_> = chats.iter().filter_map(|chat| chat.stop(false)).collect();
        for thread in closing {
            let _ = thread.join();
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
    use std::time::{Duration, Instant};

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
            Some(
                json!({"usage": {"rolling": {"status": "ok", "percent": 42}, "weekly": {"status": "ok", "percent": 17}, "monthly": {"status": "ok", "percent": 6}}}),
            )
        });
        let chat_shown = Arc::new(AtomicBool::new(true));
        let run_dir = dir.path().join("run");
        let ctx = ChatCtx {
            paths: paths.clone(),
            notifier: notifier.clone(),
            dock_visible: visible.clone(),
            chat_shown: chat_shown.clone(),
            run_dir: run_dir.clone(),
            hidden_dir: Some(dir.path().to_path_buf()),
            path_var: "/usr/bin:/bin".into(),
            quota_fetcher,
            jobs: Jobs::new(dir.path().join("jobs.json"), Arc::new(|| {})),
        };
        Harness {
            _turn: turn,
            quota_calls,
            chat_shown,
            run_dir,
            backlog: Mutex::new(Vec::new()),
            _dir: dir,
            paths,
            chat: Chat::new(Arc::new(ctx), emit, "s1".into()),
            messages,
            notifier,
            visible,
            workdir,
        }
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
        h.chat.handle(PageMessage::ChatSend { text: text.into(), behavior: None, attachments: vec![] });
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
        let starts =
            drain(&h).iter().filter(|m| matches!(m, DockMessage::ChatEngine { state } if state == "starting")).count();
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
        assert!(
            matches!(history, DockMessage::ChatEvent { ref event } if event["messages"][0]["content"] == json!("earlier"))
        );
    }

    #[test]
    fn chat_load_with_a_missing_pi_only_reports_the_problem() {
        let h = harness_with(false);
        h.chat.handle(PageMessage::ChatLoad);
        wait_for(
            &h,
            "the problem",
            |m| matches!(m, DockMessage::ChatState { problem: Some(p), .. } if p == "pi-missing"),
        );
        assert!(drain(&h).iter().all(|m| !matches!(m, DockMessage::ChatEngine { .. })));
    }

    #[test]
    fn the_engine_reports_its_model_and_thinking_level_once_ready() {
        let h = harness();
        send(&h, "hello");
        let state =
            wait_for(&h, "state from get_state", |m| matches!(m, DockMessage::ChatState { thinking: Some(_), .. }));
        assert!(
            matches!(state, DockMessage::ChatState { thinking: Some(t), model: Some(m), .. } if t == "medium" && m == "opencode-go/deepseek-v4.1-flash")
        );
    }

    #[test]
    fn dialog_while_hidden_notifies_once_and_answer_dismisses() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.visible.store(false, Ordering::SeqCst);
        // `ask` is the fake Pi's test-only command that opens a select dialog (u1)
        h.chat.handle(PageMessage::ChatSend { text: "/ask".into(), behavior: None, attachments: vec![] });
        wait_for(&h, "no dialog for plain prompts", |m| event_of(m, "agent_settled"));
        h.chat.test_send_raw("ask");
        let dialog = wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { .. }));
        assert!(matches!(dialog, DockMessage::ChatDialog { ref id, hidden: true, .. } if id == "u1"));
        assert_eq!(h.notifier.log(), vec!["send:Aria needs an answer:Allow?"]);
        h.chat.handle(PageMessage::UiAnswer {
            id: "u1".into(),
            value: Some("Allow".into()),
            confirmed: None,
            cancelled: None,
        });
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
        wait_for(
            &h,
            "the replayed dialog",
            |m| matches!(m, DockMessage::ChatDialog { id, hidden: false, .. } if id == "u1"),
        );
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
        wait_for(
            &h,
            "the notice",
            |m| matches!(m, DockMessage::ChatEvent { event } if event["request"]["message"].as_str().is_some_and(|t| t.contains("packages"))),
        );
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
        let notice = wait_for(
            &h,
            "the notice",
            |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "ui_notice" && event["request"]["message"] == "careful"),
        );
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
        h.chat.handle(PageMessage::UiAnswer {
            id: "ghost".into(),
            value: Some("x".into()),
            confirmed: None,
            cancelled: None,
        });
        send(&h, "x");
        wait_for(&h, "the chat still works", |m| event_of(m, "agent_settled"));
        h.chat.handle(PageMessage::UiAnswer {
            id: "ghost".into(),
            value: None,
            confirmed: None,
            cancelled: Some(true),
        });
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
        wait_for(
            &h,
            "the problem again on load",
            |m| matches!(m, DockMessage::ChatState { problem: Some(p), .. } if p == "pi-missing"),
        );
    }

    #[test]
    fn a_missing_node_is_its_own_problem() {
        let h = harness();
        std::fs::remove_file(h.paths.bin().join("node")).unwrap();
        // /usr/bin:/bin may hold a real node on this machine; point the chat at an empty PATH instead.
        let ctx = ChatCtx {
            paths: h.paths.clone(),
            notifier: h.notifier.clone(),
            dock_visible: h.visible.clone(),
            chat_shown: Arc::new(AtomicBool::new(true)),
            run_dir: h.run_dir.clone(),
            hidden_dir: Some(h._dir.path().to_path_buf()),
            path_var: "/nonexistent".into(),
            quota_fetcher: Arc::new(|_| None),
            jobs: h.chat.ctx.jobs.clone(),
        };
        let (sender, messages) = mpsc::channel();
        let sender = Mutex::new(sender);
        let chat = Chat::new(
            Arc::new(ctx),
            Arc::new(move |m| {
                let _ = sender.lock().unwrap().send(m);
            }),
            "s1".into(),
        );
        chat.handle(PageMessage::ChatSend { text: "hi".into(), behavior: None, attachments: vec![] });
        let message = messages.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            matches!(message, DockMessage::ChatState { problem: Some(ref p), .. } if p == "node-missing"),
            "{message:?}"
        );
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

    fn pi_reply(h: &Harness, id: u64) -> (bool, Value, Option<String>) {
        match wait_for(h, "the reply", |m| matches!(m, DockMessage::PiReply { id: got, .. } if *got == id)) {
            DockMessage::PiReply { ok, data, error, .. } => (ok, data, error),
            _ => unreachable!(),
        }
    }

    #[test]
    fn the_menus_pi_commands_reach_pi_with_their_fields() {
        let h = harness();
        h.chat.handle(PageMessage::PiCommand { id: 1, command: "get_session_stats".into(), args: Value::Null });
        let (ok, data, _) = pi_reply(&h, 1);
        assert!(ok);
        assert_eq!(data["contextUsage"]["percent"], json!(30));
        h.chat.handle(PageMessage::PiCommand {
            id: 2,
            command: "compact".into(),
            args: json!({"customInstructions": "keep the code"}),
        });
        let (ok, data, _) = pi_reply(&h, 2);
        assert!(ok);
        assert_eq!(data["instructions"], json!("keep the code"));
        h.chat.handle(PageMessage::PiCommand { id: 3, command: "export_html".into(), args: Value::Null });
        assert_eq!(pi_reply(&h, 3), (false, Value::Null, Some("Nothing to export".into())));
    }

    #[test]
    fn a_pi_command_the_menu_does_not_offer_is_never_sent() {
        let h = harness();
        h.chat.handle(PageMessage::PiCommand { id: 4, command: "crash".into(), args: Value::Null });
        assert_eq!(pi_reply(&h, 4), (false, Value::Null, Some("command-unknown".into())));
        assert!(
            !drain(&h).iter().any(|m| matches!(m, DockMessage::ChatEngine { state } if state == "exited")),
            "Pi never got it"
        );
    }

    #[test]
    fn reload_restarts_only_its_own_pi() {
        let s = sessions_harness();
        for id in ["aa", "bb"] {
            s.sessions.handle(in_session(
                id,
                PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] },
            ));
            wait_session(&s, "settled", id, settled);
        }
        s.sessions.handle(in_session("aa", PageMessage::EnginesRestart));
        wait_session(&s, "aa stopped", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "stopped"));
        std::thread::sleep(Duration::from_millis(200));
        let bb_stopped = std::iter::from_fn(|| s.messages.try_recv().ok()).any(|m| {
            matches!(m, DockMessage::Session { ref session, ref message } if session == "bb" && matches!(**message, DockMessage::ChatEngine { ref state } if state == "stopped"))
        });
        assert!(!bb_stopped, "the other tab's Pi keeps running");
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
        let t2 = std::thread::spawn(move || {
            b.handle(PageMessage::ChatSend { text: "hi".into(), behavior: None, attachments: vec![] })
        });
        t1.join().unwrap();
        t2.join().unwrap();
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        wait_for(&h, "commands", |m| matches!(m, DockMessage::Commands { id: 1, .. }));
        let starts =
            drain(&h).iter().filter(|m| matches!(m, DockMessage::ChatEngine { state } if state == "starting")).count();
        assert_eq!(starts, 1, "two threads must share one Pi");
    }

    #[test]
    fn get_state_never_overrides_what_the_events_say_about_running() {
        let mut inner = Inner { running: true, ..Inner::default() };
        apply_state(
            &mut inner,
            &json!({"isStreaming": false, "thinkingLevel": "high", "model": {"provider": "p", "id": "m"}}),
        );
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
        h.chat.handle(PageMessage::UiAnswer {
            id: "u1".into(),
            value: Some("Allow".into()),
            confirmed: None,
            cancelled: None,
        });
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
        assert!(
            !drain(&h).iter().any(|m| matches!(m, DockMessage::ChatDialog { .. })),
            "a dead Pi's dialog was replayed"
        );
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
        let ctx = ChatCtx {
            paths: h.paths.clone(),
            notifier: h.notifier.clone(),
            dock_visible: h.visible.clone(),
            chat_shown: Arc::new(AtomicBool::new(true)),
            run_dir: h.run_dir.clone(),
            hidden_dir: Some(h._dir.path().to_path_buf()),
            path_var: "/usr/bin:/bin".into(),
            quota_fetcher: Arc::new(|_| None),
            jobs: h.chat.ctx.jobs.clone(),
        };
        let sessions = Sessions::new(ctx, emit);
        SessionsHarness { h, sessions, messages }
    }

    fn in_session(session: &str, message: PageMessage) -> PageMessage {
        PageMessage::Session { session: session.into(), message: Box::new(message) }
    }

    fn wait_session(
        s: &SessionsHarness,
        what: &str,
        session: &str,
        found: impl Fn(&DockMessage) -> bool,
    ) -> DockMessage {
        loop {
            match s.messages.recv_timeout(Duration::from_secs(5)) {
                Ok(DockMessage::Session { session: got, message }) if got == session && found(&message) => {
                    return *message
                }
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
        s.sessions.handle(in_session(
            "aa",
            PageMessage::ChatSend { text: "one".into(), behavior: None, attachments: vec![] },
        ));
        s.sessions.handle(in_session(
            "bb",
            PageMessage::ChatSend { text: "two".into(), behavior: None, attachments: vec![] },
        ));
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
        s.sessions.handle(in_session(
            "../evil",
            PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] },
        ));
        s.sessions.handle(in_session("fresh-1", PageMessage::ChatLoad));
        wait_session(&s, "its state", "fresh-1", |m| matches!(m, DockMessage::ChatState { .. }));
        assert!(!s.h.paths.agent().join("sessions").join("../evil").exists());
    }

    #[test]
    fn a_dialog_in_one_session_does_not_dismiss_anothers_notification() {
        let s = sessions_harness();
        s.h.visible.store(false, Ordering::SeqCst);
        for id in ["aa", "bb"] {
            s.sessions.handle(in_session(
                id,
                PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] },
            ));
            wait_session(&s, "ready", id, |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        }
        s.sessions.test_send_raw("aa", "ask");
        wait_session(&s, "aa's dialog", "aa", |m| matches!(m, DockMessage::ChatDialog { .. }));
        s.sessions.test_send_raw("bb", "ask");
        wait_session(&s, "bb's dialog", "bb", |m| matches!(m, DockMessage::ChatDialog { .. }));
        assert_eq!(s.h.notifier.log().len(), 2);
        s.sessions.handle(in_session(
            "aa",
            PageMessage::UiAnswer { id: "u1".into(), value: Some("Allow".into()), confirmed: None, cancelled: None },
        ));
        wait_session(&s, "aa's dialog closing", "aa", |m| matches!(m, DockMessage::ChatDialogClosed { .. }));
        assert!(
            !s.h.notifier.log().contains(&"dismiss".to_string()),
            "bb still waits for an answer: {:?}",
            s.h.notifier.log()
        );
        s.sessions.handle(in_session(
            "bb",
            PageMessage::UiAnswer { id: "u1".into(), value: Some("Allow".into()), confirmed: None, cancelled: None },
        ));
        wait_session(&s, "bb's dialog closing", "bb", |m| matches!(m, DockMessage::ChatDialogClosed { .. }));
        assert_eq!(s.h.notifier.log().last().map(String::as_str), Some("dismiss"));
    }

    #[test]
    fn closing_a_session_stops_its_pi_and_closes_its_dialogs() {
        let s = sessions_harness();
        s.h.visible.store(false, Ordering::SeqCst);
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        s.sessions.test_send_raw("aa", "ask");
        wait_session(&s, "the dialog", "aa", |m| matches!(m, DockMessage::ChatDialog { .. }));
        s.sessions.handle(PageMessage::SessionClose { session: "aa".into() });
        wait_session(
            &s,
            "the dialog closing",
            "aa",
            |m| matches!(m, DockMessage::ChatDialogClosed { id } if id == "u1"),
        );
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
            s.sessions.handle(in_session(
                id,
                PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] },
            ));
            wait_session(&s, "ready", id, |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        }
        s.h.visible.store(false, Ordering::SeqCst);
        s.sessions.test_send_raw("aa", "ask");
        wait_session(&s, "the dialog", "aa", |m| matches!(m, DockMessage::ChatDialog { .. }));
        s.sessions.handle(PageMessage::EnginesRestart);
        wait_session(&s, "the dialog closing", "aa", |m| matches!(m, DockMessage::ChatDialogClosed { .. }));
        // showing a session again wakes a new Pi, which sees the changed credentials
        s.sessions.handle(in_session("bb", PageMessage::ChatLoad));
        wait_session(
            &s,
            "a new start",
            "bb",
            |m| matches!(m, DockMessage::ChatEngine { state } if state == "starting"),
        );
        wait_session(&s, "ready again", "bb", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
    }

    #[test]
    fn a_settings_change_waits_for_the_run_then_restarts_pi() {
        let s = sessions_harness();
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        // the first message's own run (start and settle) must not be mistaken for the one below
        wait_session(
            &s,
            "the first run settled",
            "aa",
            |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "agent_settled"),
        );
        std::thread::sleep(Duration::from_millis(100));
        while s.messages.try_recv().is_ok() {}
        s.sessions.test_send_raw("aa", "start"); // a run that has not settled
        wait_session(
            &s,
            "the run",
            "aa",
            |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "agent_start"),
        );
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
        wait_session(
            &s,
            "stopped once the run settled",
            "aa",
            |m| matches!(m, DockMessage::ChatEngine { state } if state == "stopped"),
        );
    }

    #[test]
    fn an_update_waits_while_a_run_goes_and_then_stops_every_pi() {
        let s = sessions_harness();
        assert!(!s.sessions.busy(), "no session has run yet");
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
        wait_session(&s, "the first run settled", "aa", settled);
        std::thread::sleep(Duration::from_millis(100));
        while s.messages.try_recv().is_ok() {}
        s.sessions.test_send_raw("aa", "start"); // a run that has not settled
        wait_session(&s, "the run", "aa", |m| event_of(m, "agent_start"));
        assert!(s.sessions.busy(), "a run is going");
        s.sessions.test_send_raw("aa", "settle");
        wait_session(&s, "settled", "aa", settled);
        assert!(!s.sessions.busy(), "the run settled");
        s.sessions.stop_for_update();
        wait_session(&s, "stopped", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "stopped"));
        assert!(s.sessions.chat("aa").unwrap().current_engine().is_none(), "no Pi is left");
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
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        let mut settings = Settings::defaults("es");
        settings.effort.mode = "fixed".into();
        settings.effort.fixed_level = Some("high".into());
        settings.compaction.percent = 70;
        settings::save(&s.h.paths, &settings).unwrap();
        s.sessions.settings_changed();
        assert!(!stopped_within(&s, 300), "plan mode, session grants and the quota switch survive an /effort");
        let config: Value =
            serde_json::from_str(&std::fs::read_to_string(s.h.paths.agent().join("aria.json")).unwrap()).unwrap();
        assert_eq!(config["effort"]["fixedLevel"], "high", "the extensions read the change at once");
    }

    #[test]
    fn a_change_pi_reads_only_at_start_restarts_it() {
        let s = sessions_harness();
        settings::save(&s.h.paths, &Settings::defaults("es")).unwrap();
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
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
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
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
            s.sessions.handle(in_session(
                id,
                PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] },
            ));
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
        let first =
            wait_for(&h, "a state with the context", |m| matches!(m, DockMessage::ChatState { context: Some(_), .. }));
        assert!(matches!(first, DockMessage::ChatState { context: Some(c), .. } if (c - 30.0).abs() < f64::EPSILON));
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        // the settle triggers a fresh reading
        wait_for(&h, "the state after the run", |m| matches!(m, DockMessage::ChatState { context: Some(_), .. }));
    }

    #[test]
    fn clearing_the_queue_reaches_pi_and_is_ignored_without_one() {
        let h = harness();
        h.chat.handle(PageMessage::ChatClearQueue { steer: vec![], keep: vec![] });
        assert!(drain(&h).is_empty());
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.handle(PageMessage::ChatClearQueue { steer: vec![], keep: vec![] });
        wait_for(&h, "Pi's queue update", |m| event_of(m, "queue_update"));
    }

    #[test]
    fn cancelling_one_queued_message_queues_the_others_again() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "first settle", |m| event_of(m, "agent_settled"));
        drain(&h); // forget the first run's events: only the resend may satisfy the wait below
        h.chat.handle(PageMessage::ChatClearQueue { steer: vec![], keep: vec!["the one that stays".into()] });
        // the fake Pi answers a prompt with agent_start: it was sent again after the queue was emptied
        wait_for(&h, "the kept message sent again", |m| event_of(m, "agent_start"));
    }

    #[test]
    fn the_messages_kept_in_the_queue_keep_their_place_in_the_gap_or_at_the_end() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "first settle", |m| event_of(m, "agent_settled"));
        drain(&h);
        h.chat.handle(PageMessage::ChatClearQueue { steer: vec!["now".into()], keep: vec!["later".into()] });
        let queued = |m: &DockMessage| event_of(m, "queued_as");
        let behavior = |m: DockMessage| match m {
            DockMessage::ChatEvent { event } => (event["behavior"].clone(), event["message"].clone()),
            _ => unreachable!(),
        };
        assert_eq!(behavior(wait_for(&h, "the steering one", queued)), (json!("steer"), json!("now")));
        assert_eq!(behavior(wait_for(&h, "the follow-up", queued)), (json!("followUp"), json!("later")));
    }

    #[test]
    fn the_provider_quota_is_reported_once_pi_is_ready() {
        let h = harness();
        send(&h, "hello");
        match wait_for(&h, "the quota", |m| matches!(m, DockMessage::ChatQuota { .. })) {
            DockMessage::ChatQuota { windows } => {
                assert_eq!(
                    windows.iter().map(|w| (w.id.as_str(), w.percent)).collect::<Vec<_>>(),
                    vec![("5h", 42.0), ("week", 17.0), ("month", 6.0)]
                );
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
        wait_for(
            &h,
            "an empty quota for the new model",
            |m| matches!(m, DockMessage::ChatQuota { windows } if windows.is_empty()),
        );
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

    /// A run started and left silent: `start` sends `agent_start` and nothing more.
    fn silent_run(h: &Harness) {
        ready(h);
        drain(h);
        h.chat.test_send_raw("start");
        wait_for(h, "a run", |m| event_of(m, "agent_start"));
    }

    fn is_watchdog_notice(message: &DockMessage) -> bool {
        matches!(message, DockMessage::ChatEvent { event }
            if event["type"] == "ui_notice" && event["request"]["key"] == "chat.watchdog.stopped")
    }

    const LIMIT: Duration = Duration::from_secs(1);

    #[test]
    fn a_silent_run_is_stopped_once() {
        let h = harness();
        silent_run(&h);
        assert!(!h.chat.check_stalled(Instant::now(), LIMIT), "a run that just spoke is left alone");
        assert!(h.chat.check_stalled(Instant::now() + 2 * LIMIT, LIMIT));
        wait_for(&h, "the abort reaching Pi", |m| event_of(m, "abort_seen"));
        let notice = wait_for(&h, "the notice", is_watchdog_notice);
        let DockMessage::ChatEvent { event } = notice else { unreachable!() };
        assert_eq!(event["request"]["vars"]["minutes"], 1);
        assert_eq!(event["request"]["notifyType"], "warning");
        assert!(event["request"].get("message").is_none(), "the page says it in its language, from the key");
        assert!(!h.chat.check_stalled(Instant::now() + Duration::from_secs(3600), LIMIT), "once per run");
        std::thread::sleep(Duration::from_millis(200));
        let later = drain(&h);
        assert!(!later.iter().any(|m| event_of(m, "abort_seen") || is_watchdog_notice(m)), "{later:?}");
        // The next run has its own watchdog.
        h.chat.test_send_raw("settle");
        wait_for(&h, "the settle", |m| event_of(m, "agent_settled"));
        h.chat.test_send_raw("start");
        wait_for(&h, "the next run", |m| event_of(m, "agent_start"));
        assert!(h.chat.check_stalled(Instant::now() + 2 * LIMIT, LIMIT));
    }

    #[test]
    fn a_waiting_dialog_pauses_the_watchdog() {
        let h = harness();
        silent_run(&h);
        h.chat.test_send_raw("ask");
        wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { .. }));
        let hour = Duration::from_secs(3600);
        assert!(!h.chat.check_stalled(Instant::now() + hour, LIMIT), "an hour on a question is not a hang");
        h.chat.handle(PageMessage::UiAnswer {
            id: "u1".into(),
            value: Some("Allow".into()),
            confirmed: None,
            cancelled: None,
        });
        wait_for(&h, "the answer reaching Pi", |m| event_of(m, "answered"));
        assert!(!drain(&h).iter().any(|m| event_of(m, "abort_seen") || is_watchdog_notice(m)));
        assert!(!h.chat.check_stalled(Instant::now(), LIMIT), "the clock starts again when the dialog closes");
        assert!(h.chat.check_stalled(Instant::now() + 2 * LIMIT, LIMIT), "silence after the answer still counts");
    }

    #[test]
    fn a_running_tool_pauses_the_watchdog() {
        let h = harness();
        silent_run(&h);
        h.chat.test_send_raw("tool_start");
        wait_for(&h, "the tool", |m| event_of(m, "tool_execution_start"));
        let hour = Duration::from_secs(3600);
        assert!(!h.chat.check_stalled(Instant::now() + hour, LIMIT), "a long silent build is not a hang");
        std::thread::sleep(Duration::from_millis(200));
        assert!(!drain(&h).iter().any(|m| event_of(m, "abort_seen") || is_watchdog_notice(m)));
        h.chat.test_send_raw("tool_end");
        wait_for(&h, "the tool's end", |m| event_of(m, "tool_execution_end"));
        assert!(!h.chat.check_stalled(Instant::now(), LIMIT), "the clock starts again when the last tool ends");
        assert!(h.chat.check_stalled(Instant::now() + 2 * LIMIT, LIMIT), "the model's silence after it counts");
        wait_for(&h, "the abort reaching Pi", |m| event_of(m, "abort_seen"));
        wait_for(&h, "the notice", is_watchdog_notice);
    }

    #[test]
    fn zero_turns_it_off() {
        let h = harness();
        silent_run(&h);
        assert!(!h.chat.check_stalled(Instant::now() + Duration::from_secs(36_000), Duration::ZERO));
        std::thread::sleep(Duration::from_millis(200));
        assert!(!drain(&h).iter().any(|m| event_of(m, "abort_seen") || is_watchdog_notice(m)));
    }

    #[test]
    fn events_keep_it_alive() {
        let h = harness();
        silent_run(&h);
        let started = Instant::now();
        std::thread::sleep(Duration::from_millis(300));
        h.chat.test_send_raw("notify"); // a UI request
        wait_for(&h, "the notice", |m| event_of(m, "ui_notice"));
        assert!(!h.chat.check_stalled(started + LIMIT + Duration::from_millis(100), LIMIT));
        std::thread::sleep(Duration::from_millis(300));
        h.chat.test_send_raw("args"); // an event
        wait_for(&h, "the event", |m| event_of(m, "args"));
        assert!(!h.chat.check_stalled(started + LIMIT + Duration::from_millis(400), LIMIT));
        std::thread::sleep(Duration::from_millis(300));
        h.chat.test_send_raw("reject"); // a response
        wait_for(&h, "the response", |m| event_of(m, "command_failed"));
        assert!(!h.chat.check_stalled(started + LIMIT + Duration::from_millis(700), LIMIT));
        assert!(h.chat.check_stalled(Instant::now() + 2 * LIMIT, LIMIT), "silence after them still counts");
    }

    #[test]
    fn an_idle_chat_is_never_stopped() {
        let h = harness();
        ready(&h);
        assert!(!h.chat.check_stalled(Instant::now() + Duration::from_secs(3600), LIMIT));
    }

    #[test]
    fn sessions_read_the_limit_from_the_settings() {
        let h = harness();
        silent_run(&h);
        let mut settings = settings::load(&h.paths).unwrap().unwrap();
        settings.models.turn_timeout_minutes = 0;
        settings::save(&h.paths, &settings).unwrap();
        assert_eq!(Sessions::limit(&h.paths), Duration::ZERO);
        settings.models.turn_timeout_minutes = 10;
        settings::save(&h.paths, &settings).unwrap();
        assert_eq!(Sessions::limit(&h.paths), Duration::from_secs(600));
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
        assert_eq!(
            dialog_body(&json!({"title": "{\"aria\":1,\"kind\":\"sandbox\",\"summary\":\"Run rm -rf build\"}"})),
            "Run rm -rf build"
        );
        assert_eq!(
            dialog_body(&json!({"title": "{\"aria\":1,\"kind\":\"write\",\"path\":\"~/notes.md\"}"})),
            "Aria wants to write ~/notes.md"
        );
        assert_eq!(
            dialog_body(&json!({"title": "{\"aria\":1,\"kind\":\"sandbox\",\"detail\":\"curl -H token:abc\"}"})),
            "Aria wants to run a command outside the sandbox"
        );
        assert_eq!(dialog_body(&json!({"title": "{\"aria\":1,\"kind\":\"quota\",\"percent\":98}"})), "Quota at 98%");
        assert_eq!(
            dialog_body(
                &json!({"title": "{\"aria\":1,\"kind\":\"question\",\"questions\":[{\"question\":\"Which fallback?\"}]}"})
            ),
            "Which fallback?"
        );
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
    fn a_connector_card_notifies_with_its_summary() {
        let card = json!({"aria": 1, "kind": "connector", "summary": "Notion wants to create pages", "detail": "title: secret plans"});
        assert_eq!(dialog_body(&json!({"title": card.to_string()})), "Notion wants to create pages");
        // Never the arguments: without a summary, a fixed sentence.
        let bare = json!({"aria": 1, "kind": "connector", "detail": "title: secret plans"});
        assert_eq!(dialog_body(&json!({"title": bare.to_string()})), "A connector wants to act");
    }

    #[test]
    fn a_working_folder_that_cannot_be_made_is_a_problem_not_the_home_folder() {
        let h = harness();
        let mut settings = settings::load(&h.paths).unwrap().unwrap();
        settings.sandbox.projects_dir = "/proc/aria-cannot-be-here".into();
        settings::save(&h.paths, &settings).unwrap();
        send(&h, "hi");
        wait_for(
            &h,
            "the problem",
            |m| matches!(m, DockMessage::ChatState { problem: Some(p), .. } if p == "workdir-unavailable"),
        );
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

    fn with_image(text: &str) -> PageMessage {
        use base64::Engine as _;
        let data = base64::engine::general_purpose::STANDARD.encode(b"PNGDATA");
        let attachments =
            vec![aria_core::bridge::Attachment { name: "shot.png".into(), mime: "image/png".into(), data }];
        PageMessage::ChatSend { text: text.into(), behavior: None, attachments }
    }

    fn prompt_seen(h: &Harness) -> (u64, String) {
        match wait_for(h, "the prompt with attachments", |m| event_of(m, "prompt_seen")) {
            DockMessage::ChatEvent { event } => {
                (event["images"].as_u64().unwrap(), event["message"].as_str().unwrap().replace("\\n", "\n"))
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn the_prompt_carries_images_for_a_vision_model() {
        let h = harness();
        ready(&h);
        // The session's model (the daily one) is not listed as seeing images: the image goes as a path.
        h.chat.handle(with_image("look"));
        let (images, message) = prompt_seen(&h);
        assert_eq!(images, 0);
        assert!(message.starts_with("look\n\n[attached: shot.png (image/png, 7 B) at "), "{message}");
        let saved = h.paths.attachments().join("s1").join(crate::attachments::today()).join("1-shot.png");
        assert!(message.contains(&saved.display().to_string()), "{message}");
        assert_eq!(std::fs::read(&saved).unwrap(), b"PNGDATA");
        {
            // A model Pi does not list is not taken for text-only for good; the listed ones are kept.
            let inner = h.chat.inner.lock().unwrap();
            assert!(!inner.sees_images.contains_key("opencode-go/deepseek-v4.1-flash"), "{:?}", inner.sees_images);
            assert_eq!(inner.sees_images.get("opencode-go/kimi-k3"), Some(&true));
        }

        h.chat.handle(PageMessage::SetModel { id: 2, model: "opencode-go/kimi-k3".into() });
        wait_for(&h, "the model change", |m| matches!(m, DockMessage::Saved { id: 2 }));
        h.chat.handle(with_image("again"));
        let (images, message) = prompt_seen(&h);
        assert_eq!(images, 1);
        assert!(message.starts_with("again\n\n[attached image: shot.png at "), "{message}");
        assert!(message.ends_with("2-shot.png]"), "{message}");
    }

    #[test]
    fn a_new_pi_is_asked_again_which_models_see_images() {
        let h = harness();
        ready(&h);
        h.chat.handle(with_image("look"));
        prompt_seen(&h);
        assert!(!h.chat.inner.lock().unwrap().sees_images.is_empty());
        h.chat.shutdown();
        ready(&h);
        assert!(h.chat.inner.lock().unwrap().sees_images.is_empty());
    }

    #[test]
    fn a_message_with_files_is_given_back_when_pi_cannot_start() {
        let h = harness_with(false);
        h.chat.handle(with_image("look"));
        match wait_for(&h, "the refusal", |m| event_of(m, "command_failed")) {
            DockMessage::ChatEvent { event } => {
                assert_eq!(event["command"], json!("prompt"));
                assert_eq!(event["text"], json!("look"));
            }
            _ => unreachable!(),
        }
        assert!(!h.paths.attachments().exists(), "nothing is saved for a message that was not sent");
    }

    #[test]
    fn a_failed_save_gives_the_text_back() {
        let h = harness();
        ready(&h);
        std::fs::write(h.paths.attachments(), "not a folder").unwrap();
        drain(&h); // the first run's events: only what this send causes counts below
        h.chat.handle(with_image("look"));
        match wait_for(&h, "the refusal", |m| event_of(m, "command_failed")) {
            DockMessage::ChatEvent { event } => {
                assert_eq!(event["command"], json!("prompt"));
                assert_eq!(event["text"], json!("look"));
                assert!(event["error"].is_string());
            }
            _ => unreachable!(),
        }
        assert!(!drain(&h).iter().any(|m| event_of(m, "prompt_seen") || event_of(m, "agent_start")));
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
            DockMessage::ChatEvent { event } => {
                assert_eq!(event["settings"], json!(h.paths.settings().display().to_string()))
            }
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
        match wait_for(
            &h,
            "the agents",
            |m| matches!(m, DockMessage::ChatAgents { agents } if agents[0]["role"] == "security"),
        ) {
            DockMessage::ChatAgents { agents } => assert_eq!(agents[0]["kind"], "active"),
            _ => unreachable!(),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn closing_a_session_stops_what_its_pi_started() {
        let s = sessions_harness();
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        s.sessions.test_send_raw("aa", "background");
        let pid: u32 = match wait_session(
            &s,
            "the background process",
            "aa",
            |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "background"),
        ) {
            DockMessage::ChatEvent { event } => event["pid"].as_str().unwrap().parse().unwrap(),
            _ => unreachable!(),
        };
        let alive = |pid: u32| {
            std::fs::read_to_string(format!("/proc/{pid}/stat"))
                .is_ok_and(|stat| stat.rsplit(')').next().is_some_and(|rest| !rest.trim_start().starts_with('Z')))
        };
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
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        let path = s.h.run_dir.join("watch-aa.sock");
        assert!(path.exists());
        s.sessions.handle(PageMessage::SessionClose { session: "aa".into() });
        std::thread::sleep(Duration::from_millis(100));
        assert!(!path.exists());
    }

    // ---- background work ----------------------------------------------------------------------------

    fn sleeper(jobs: &Jobs, session: &str, folder: &std::path::Path) -> crate::jobs::Job {
        jobs.start_process(crate::jobs::ProcessStart {
            session: session.into(),
            title: "sleep".into(),
            command: "sleep 30".into(),
            line: "sleep 30".into(),
            folder: folder.to_path_buf(),
            env: Vec::new(),
        })
        .unwrap()
    }

    fn eventually(what: &str, test: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while std::time::Instant::now() < deadline {
            if test() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("{what}");
    }

    #[test]
    fn pi_gets_the_jobs_socket() {
        let h = harness();
        ready(&h);
        h.chat.test_send_raw("env_jobs");
        let path = match wait_for(
            &h,
            "the socket path",
            |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "env" && event.get("jobs").is_some()),
        ) {
            DockMessage::ChatEvent { event } => event["jobs"].as_str().unwrap().to_string(),
            _ => unreachable!(),
        };
        assert!(path.ends_with("jobs-s1.sock"), "{path}");
        assert!(std::path::Path::new(&path).exists(), "the socket is listening");
    }

    #[test]
    fn no_jobs_socket_where_a_boxed_command_could_reach_it() {
        let h = harness();
        // A run dir outside the hidden folder (`$XDG_RUNTIME_DIR` unset: the run dir falls back to /tmp).
        let ctx = ChatCtx {
            paths: h.paths.clone(),
            notifier: h.notifier.clone(),
            dock_visible: h.visible.clone(),
            chat_shown: Arc::new(AtomicBool::new(true)),
            run_dir: h.run_dir.clone(),
            hidden_dir: None,
            path_var: "/usr/bin:/bin".into(),
            quota_fetcher: Arc::new(|_| None),
            jobs: h.chat.ctx.jobs.clone(),
        };
        let (sender, messages) = mpsc::channel();
        let sender = Mutex::new(sender);
        let chat = Chat::new(
            Arc::new(ctx),
            Arc::new(move |m| {
                let _ = sender.lock().unwrap().send(m);
            }),
            "s2".into(),
        );
        chat.handle(PageMessage::ChatSend { text: "hi".into(), behavior: None, attachments: vec![] });
        chat.test_send_raw("env_jobs");
        let path = loop {
            match messages.recv_timeout(Duration::from_secs(5)).expect("the env") {
                DockMessage::ChatEvent { event } if event["type"] == "env" && event.get("jobs").is_some() => {
                    break event["jobs"].as_str().unwrap().to_string()
                }
                _ => continue,
            }
        };
        assert_eq!(path, "", "Pi gets no ARIA_JOBS_SOCK, so its tools say they need the dock");
        assert!(!h.run_dir.join("jobs-s2.sock").exists());

        assert!(hidden_from_the_box(Path::new("/run/user/1000/aria"), Some(Path::new("/run/user/1000"))));
        assert!(!hidden_from_the_box(Path::new("/tmp/aria"), Some(Path::new("/run/user/1000"))));
        assert!(!hidden_from_the_box(Path::new("/tmp/aria"), None));
        assert!(!hidden_from_the_box(Path::new("aria"), Some(Path::new(""))), "a relative folder hides nothing");
    }

    #[test]
    fn closing_a_tab_stops_its_jobs() {
        let s = sessions_harness();
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        let socket = s.h.run_dir.join("jobs-aa.sock");
        assert!(socket.exists());
        let jobs = s.sessions.ctx.jobs.clone();
        let mine = sleeper(&jobs, "aa", s.h._dir.path());
        let other = sleeper(&jobs, "bb", s.h._dir.path());
        s.sessions.handle(PageMessage::SessionClose { session: "aa".into() });
        eventually("the closed tab's job stopped", || {
            jobs.get(&mine.id).is_some_and(|j| j.state == crate::jobs::JobState::Stopped)
        });
        eventually("nobody will ever ack a closed tab's events: they are dropped", || jobs.events("aa").is_empty());
        assert_eq!(
            jobs.get(&other.id).unwrap().state,
            crate::jobs::JobState::Running,
            "another tab's job keeps running"
        );
        std::thread::sleep(Duration::from_millis(100));
        assert!(!socket.exists(), "the closed tab's jobs socket is gone");
        jobs.stop_all();
    }

    #[test]
    fn background_jobs_keep_the_dock_busy() {
        let s = sessions_harness();
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
        wait_session(&s, "settled", "aa", settled);
        assert!(!s.sessions.busy());
        let jobs = s.sessions.ctx.jobs.clone();
        sleeper(&jobs, "aa", s.h._dir.path());
        assert!(s.sessions.busy(), "a running job keeps the update waiting");
        assert!(!s.sessions.busy_without_jobs(), "only the job is in the way");
        jobs.stop_all();
        assert!(!s.sessions.busy());
    }

    #[test]
    fn a_session_launcher_uses_the_sessions_model() {
        use crate::jobs_socket::Launcher;
        let h = harness();
        h.chat.watch().expect("the watch socket");
        {
            let mut inner = h.chat.inner.lock().unwrap();
            inner.model = Some("opencode-go/x".into());
            inner.thinking = Some("high".into());
        }
        let launcher = SessionLauncher { chat: h.chat.clone() };
        let (program, args, env, unset, worktrees) = launcher.agent("session").expect("a launch");
        assert_eq!(worktrees, h.workdir.join(".aria-worktrees"), "in the projects folder");
        assert_eq!(program, h.paths.bin().join("pi"));
        assert_eq!(args, ["--model", "opencode-go/x", "--thinking", "high"]);
        assert!(env.iter().any(|(k, v)| k == "PI_WATCH_SOCK" && v.ends_with("watch-s1.sock")), "{env:?}");
        assert!(env.iter().any(|(k, _)| k == "PI_CODING_AGENT_DIR"), "{env:?}");
        assert!(unset.iter().any(|name| name == "ARIA_JOBS_SOCK"), "{unset:?}");

        let mut settings = settings::load(&h.paths).unwrap().unwrap();
        settings.models.planning = Some("opencode-go/big".into());
        settings.models.planning_effort = "xhigh".into();
        settings::save(&h.paths, &settings).unwrap();
        let (_, args, _, _, _) = launcher.agent("heavy").unwrap();
        assert_eq!(args, ["--model", "opencode-go/big", "--thinking", "xhigh"]);

        // Without a model of its own the session falls back to the daily model; with no settings and no model there is
        // nothing to run.
        h.chat.inner.lock().unwrap().model = None;
        h.chat.inner.lock().unwrap().thinking = None;
        let (_, args, _, _, _) = launcher.agent("session").unwrap();
        assert_eq!(args, ["--model", "opencode-go/deepseek-v4.1-flash"]);
        std::fs::remove_file(h.paths.settings()).unwrap();
        assert!(launcher.agent("session").is_none());
        assert!(launcher.agent("heavy").is_none());
    }

    #[test]
    fn a_launcher_names_the_subagent_in_the_watch_before_it_starts() {
        use crate::jobs_socket::Launcher;
        use std::io::Write;
        let h = harness();
        ready(&h);
        let launcher = SessionLauncher { chat: h.chat.clone() };
        let mut job = sleeper(&h.chat.ctx.jobs, "s1", h._dir.path());
        h.chat.ctx.jobs.stop_all();
        job.title = "Fix the tests".into();
        job.task = Some("Make the suite pass".into());
        launcher.before_spawn(&job);
        let path = h.run_dir.join("watch-s1.sock");
        let mut child = std::os::unix::net::UnixStream::connect(&path).unwrap();
        writeln!(child, "{}", json!({"t": "prompt", "text": "Task: Make the suite pass\n\nWork in /x."})).unwrap();
        wait_for(
            &h,
            "the named agent",
            |m| matches!(m, DockMessage::ChatAgents { agents } if agents[0]["role"] == "Fix the tests"),
        );
    }

    #[test]
    fn the_page_reaches_every_job_from_any_session() {
        let s = sessions_harness();
        let jobs = s.sessions.ctx.jobs.clone();
        let job = sleeper(&jobs, "aa", s.h._dir.path());
        // The page's first look: any session's chatLoad brings the whole list, unwrapped.
        s.sessions.handle(in_session("bb", PageMessage::ChatLoad));
        let listed = loop {
            match s.messages.recv_timeout(Duration::from_secs(5)).expect("the jobs list") {
                DockMessage::Jobs { jobs } => break jobs,
                _ => continue,
            }
        };
        assert_eq!(listed[0]["id"], json!(job.id));
        assert_eq!(listed[0]["session"], json!("aa"));
        s.sessions.handle(PageMessage::JobOutput { id: 7, job: "j999".into(), lines: None });
        let reply = loop {
            match s.messages.recv_timeout(Duration::from_secs(5)).expect("the output") {
                m @ DockMessage::JobOutput { .. } => break m,
                _ => continue,
            }
        };
        assert_eq!(reply, DockMessage::JobOutput { id: 7, lines: vec![] }, "an unknown job has no output");
        s.sessions.handle(PageMessage::JobStop { job: job.id.clone() });
        eventually("the job stopped", || jobs.get(&job.id).is_some_and(|j| j.state == crate::jobs::JobState::Stopped));
        s.sessions.handle(PageMessage::JobDismiss { job: job.id.clone() });
        assert!(jobs.get(&job.id).unwrap().dismissed);
    }

    #[test]
    fn a_job_s_output_reaches_the_page() {
        let s = sessions_harness();
        let jobs = s.sessions.ctx.jobs.clone();
        let job = jobs
            .start_process(crate::jobs::ProcessStart {
                session: "aa".into(),
                title: "count".into(),
                command: "seq 100".into(),
                line: "seq 100".into(),
                folder: s.h._dir.path().to_path_buf(),
                env: Vec::new(),
            })
            .unwrap();
        eventually("it ended", || jobs.get(&job.id).is_some_and(|j| j.state != crate::jobs::JobState::Running));
        s.sessions.handle(PageMessage::JobOutput { id: 1, job: job.id.clone(), lines: None });
        s.sessions.handle(PageMessage::JobOutput { id: 2, job: job.id.clone(), lines: Some(3) });
        let mut replies = Vec::new();
        while replies.len() < 2 {
            if let DockMessage::JobOutput { id, lines } = s.messages.recv_timeout(Duration::from_secs(5)).unwrap() {
                replies.push((id, lines));
            }
        }
        assert_eq!(replies[0].1.len(), 80, "80 lines when the page names no number");
        assert_eq!(replies[0].1.last().map(String::as_str), Some("100"));
        assert_eq!(replies[1], (2, vec!["98".to_string(), "99".into(), "100".into()]));
    }

    #[test]
    fn sessions_whose_tab_is_gone_are_forgotten_at_start() {
        let s = sessions_harness();
        let jobs = s.sessions.ctx.jobs.clone();
        for session in ["aa", "gone"] {
            let job = jobs
                .start_process(crate::jobs::ProcessStart {
                    session: session.into(),
                    title: "true".into(),
                    command: "true".into(),
                    line: "true".into(),
                    folder: s.h._dir.path().to_path_buf(),
                    env: Vec::new(),
                })
                .unwrap();
            eventually("it ended", || jobs.get(&job.id).is_some_and(|j| j.state != crate::jobs::JobState::Running));
        }
        let entries = vec![aria_core::sessions::SessionEntry { id: "aa".into(), title: "Kept".into() }];
        aria_core::sessions::save(&s.h.paths, &SessionList { sessions: entries, active: None }).unwrap();
        s.sessions.forget_gone_sessions();
        assert_eq!(jobs.events("aa").len(), 1, "an open tab keeps its event for its Pi");
        assert!(jobs.events("gone").is_empty(), "nobody will ever ack a gone tab's event");
    }

    #[test]
    fn sessions_with_background_work_are_told_an_update_waits() {
        let s = sessions_harness();
        for id in ["aa", "bb"] {
            s.sessions.handle(in_session(id, PageMessage::ChatLoad));
            wait_session(&s, "its state", id, |m| matches!(m, DockMessage::ChatState { .. }));
        }
        let jobs = s.sessions.ctx.jobs.clone();
        sleeper(&jobs, "aa", s.h._dir.path());
        std::thread::sleep(Duration::from_millis(100));
        while s.messages.try_recv().is_ok() {}
        s.sessions.tell_update_waits();
        let notice = wait_session(
            &s,
            "the notice",
            "aa",
            |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "ui_notice"),
        );
        match notice {
            DockMessage::ChatEvent { event } => {
                assert_eq!(event["request"]["notifyType"], json!("warning"));
                assert_eq!(event["request"]["message"], json!(UPDATE_WAITS));
            }
            _ => unreachable!(),
        }
        std::thread::sleep(Duration::from_millis(100));
        let to_bb = std::iter::from_fn(|| s.messages.try_recv().ok()).any(|m| {
            matches!(m, DockMessage::Session { ref session, ref message } if session == "bb" && matches!(**message, DockMessage::ChatEvent { .. }))
        });
        assert!(!to_bb, "a session without background work is not told");
        jobs.stop_all();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_notification_stays_until_dismissed_and_a_click_shows_the_dock() {
        let args = notify_args(std::path::Path::new("/opt/aria"), "Aria needs an answer", "Allow?");
        assert_eq!(
            args,
            [
                "--app-name",
                "aria",
                "-u",
                "critical",
                "-t",
                "0",
                "Aria needs an answer",
                "Allow?",
                "--exec",
                "/opt/aria",
                "show"
            ]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mac_notifications_escape_applescript_literals() {
        assert_eq!(apple_script_string(r#"a "b" \c"#), r#""a \"b\" \\c""#);
    }
}
