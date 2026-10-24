//! The chat: owns Pi (started when first needed, kept while the dock is hidden), turns page
//! messages into Pi commands and Pi's events into dock messages, and asks for attention with a
//! desktop notification when an extension dialog arrives while the dock is hidden.

use crate::face;
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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub type Emit = Arc<dyn Fn(DockMessage) + Send + Sync>;

const NOTIFICATION_TITLE: &str = "Aria needs an answer";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
/// A traced tab's tree goes to the page at most this often while it changes (and once more when the turn settles).
const TRACE_EVERY: Duration = Duration::from_millis(250);

/// Wall-clock ms: the trace's times (a subagent's lines carry the same clock).
fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a compaction may take: a model call over the whole context.
const COMPACT_TIMEOUT: Duration = Duration::from_secs(600);
/// Pi's own commands the `/` menu offers, and how long each may take: a compaction is a model call over the whole context.
const PI_COMMANDS: [(&str, Duration); 3] =
    [("compact", COMPACT_TIMEOUT), ("get_session_stats", REQUEST_TIMEOUT), ("export_html", Duration::from_secs(60))];
/// A terminal's `!command` (`ChatBash`) may run this long.
const BASH_TIMEOUT: Duration = Duration::from_secs(600);
const FILE_LIMIT: usize = 50;
const QUOTA_MAX_AGE: Duration = Duration::from_secs(20);
/// What a session with background work is told when an update has waited `UPDATE_NOTICE_AFTER` for that work alone.
pub const UPDATE_WAITS: &str =
    "An Aria update is waiting for the background work to finish; stop it from the background list to update now.";
/// Output lines the page gets when its `jobOutput` names no number.
const PAGE_OUTPUT_LINES: usize = 80;
/// The rotate extension's command: continue the conversation in a new session file that starts from the compaction.
const ROTATE_COMMAND: &str = "aria-rotate";
/// How long the command may take (it writes the new file; no model call).
const ROTATE_TIMEOUT: Duration = Duration::from_secs(60);
/// A segment file larger than this is compacted when the tab is quiet, which then rotates it.
const SEGMENT_LIMIT: u64 = 8 * 1024 * 1024;
/// `sessions.json` is read, changed and written by the page's saves and by rotations: one at a time.
static SESSIONS_FILE: Mutex<()> = Mutex::new(());

/// Raises and removes the desktop notification for a prompt nobody can see.
///
/// `key` is the session the prompt belongs to, so several sessions can share one notification.
pub trait Notifier: Send + Sync {
    fn send(&self, key: &str, title: &str, body: &str);
    fn dismiss(&self, key: &str);
    /// Takes away the notification with this title (a session's "it finished"), where the system can.
    fn dismiss_titled(&self, _title: &str) {}
    /// A session's "it finished": taken away by its title, never by `dismiss`.
    fn send_finished(&self, key: &str, title: &str, body: &str) {
        self.send(key, title, body);
    }
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

    fn dismiss_titled(&self, title: &str) {
        self.inner.dismiss_titled(title);
    }

    /// Not counted: the dialogs' notification goes away when their last one is answered, whatever finished meanwhile.
    fn send_finished(&self, key: &str, title: &str, body: &str) {
        self.inner.send(key, title, body);
    }
}

/// Omarchy's notifications: critical urgency and no expiry, so it stays until it is dismissed or
/// clicked; the click runs `aria show`, which opens the dock.
#[cfg(all(target_os = "linux", feature = "gui"))]
pub struct OmarchyNotifier;

/// The arguments for `omarchy-notification-send`.
#[cfg(all(target_os = "linux", feature = "gui"))]
pub fn notify_args(exe: &Path, title: &str, body: &str) -> Vec<String> {
    ["--app-name", "aria", "-u", "critical", "-t", "0", title, body, "--exec", &exe.display().to_string(), "show"]
        .iter()
        .map(|part| part.to_string())
        .collect()
}

#[cfg(feature = "gui")]
fn detached(program: &str, args: &[String]) {
    let _ = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|mut child| std::thread::spawn(move || child.wait()));
}

#[cfg(all(target_os = "linux", feature = "gui"))]
impl Notifier for OmarchyNotifier {
    fn send(&self, _key: &str, title: &str, body: &str) {
        if let Ok(exe) = std::env::current_exe() {
            detached("omarchy-notification-send", &notify_args(&exe, title, body));
        }
    }

    fn dismiss(&self, _key: &str) {
        detached("omarchy-notification-dismiss", &[NOTIFICATION_TITLE.to_string()]);
    }

    fn dismiss_titled(&self, title: &str) {
        detached("omarchy-notification-dismiss", &[title.to_string()]);
    }
}

/// No notification at all: a service with no desktop (`aria serve`) has nobody to show one to.
pub struct NullNotifier;

impl Notifier for NullNotifier {
    fn send(&self, _key: &str, _title: &str, _body: &str) {}

    fn dismiss(&self, _key: &str) {}
}

/// The notifier the dock uses: Omarchy's on Linux, macOS's on Mac (a service with no window uses [`NullNotifier`]).
#[cfg(all(target_os = "linux", feature = "gui"))]
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
    /// Where the sessions' subagent sockets live (`aria/` in the runtime directory, `control::runtime_dir`).
    pub run_dir: PathBuf,
    /// The folder boxed commands cannot see (the runtime directory, which the sandbox hides; `None` when it is only the
    /// `/tmp` fallback, which a boxed command sees). A jobs
    /// socket is opened only inside it: anywhere else a boxed command could reach it and start work outside the box.
    pub hidden_dir: Option<PathBuf>,
    /// The `PATH` Pi inherits (the shim directory goes in front of it).
    pub path_var: String,
    /// Reads a provider's quota (a fake in tests).
    pub quota_fetcher: Arc<aria_core::quota::Fetcher>,
    /// The dock's background work, every session's. One for the dock's whole life: its spawner thread is the parent
    /// every job is tied to (`PR_SET_PDEATHSIG`), so the last clone going away ends them all.
    pub jobs: Jobs,
    /// The scheduled tasks every session's jobs socket can change.
    pub schedules: crate::schedule_store::ScheduleStore,
    /// The terminals attached to each session (crate::clients): a focused one sees it.
    pub watchers: crate::clients::Watchers,
}

/// A scheduled task's tab: its task, and the grants its Pi runs with (`ARIA_GRANTS`, docs/superpowers/specs/
/// 2026-10-21-schedules-design.md).
#[derive(Clone)]
struct Scheduled {
    task: String,
    title: String,
    grants: String,
}

#[derive(Default)]
struct Inner {
    engine: Option<Arc<Engine>>,
    scheduled: Option<Scheduled>,
    /// A scheduled run is going: when it settles it is recorded and, unless seen, notified.
    scheduled_run: bool,
    /// The last answer's text and whether it ended in an error (a scheduled run's summary).
    last_answer: String,
    last_failed: bool,
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
    /// What the session is doing for the terminal's list, since when (ms), and whether a run ended that nobody saw.
    status: &'static str,
    status_at: i64,
    unseen_done: bool,
    /// The title of the "it finished" notification raised for this session, while it stands.
    settled_notice: Option<String>,
    /// The checkpoint taken before the running turn's first message: its changes are said when it settles.
    run_checkpoint: Option<crate::checkpoints::Checkpoint>,
    /// The tool executions under way (by `toolCallId`, nested ones too): while any is open the watchdog waits, since a
    /// long build or test is silent and has its own timeouts. The value is the tool's name (what the face shows).
    tools: HashMap<String, String>,
    /// How the last run ended (for the face), cleared when the next starts.
    outcome: Option<(face::Outcome, Instant)>,
    /// Pi is compacting the context inside the run (`compaction_start` … `compaction_end`, silent meanwhile): the
    /// watchdog waits, as for a tool.
    compacting: bool,
    /// The turn watchdog stopped this run already (it stops a run once; the next `agent_start` clears it).
    watchdog_fired: bool,
    /// A compaction ended and the tab has not rotated since: it rotates the next time it is quiet (kept across Pi
    /// restarts; cleared when the rotation is tried, whatever its outcome, so a failure waits for the next compaction).
    rotate_due: bool,
    /// This Pi was asked to compact an oversized segment already (once per Pi).
    compact_asked: bool,
    /// The file of the segment the running Pi is on, from `get_state`.
    session_file: Option<PathBuf>,
    /// A rotation is under way (`/aria-rotate` and recording its segment). Pi tears its session down to switch, so
    /// what would reach it meanwhile is held in `held` and runs, in order, when the rotation ends.
    rotating: bool,
    /// The page's messages (and restarts) that came during a rotation or a compaction, oldest first.
    held: std::collections::VecDeque<PageMessage>,
    /// What was held is being sent now (one replay at a time): later messages queue behind it, and a stop meanwhile
    /// stops Pi and drops what is still held.
    replaying: bool,
    /// How long `/aria-rotate` may take (tests shorten it); `ROTATE_TIMEOUT` when unset.
    rotate_timeout: Option<Duration>,
    /// The tab was closed for good: nothing held runs and no Pi starts for it again.
    closed: bool,
    /// `/debug` is on in this tab: its turns are traced (in memory only).
    trace_on: bool,
    /// The current turn's trace (from its `agent_start`; kept after it settles, until the next one).
    trace: Option<crate::trace::Trace>,
    /// When the trace was last sent, and whether a send is already waiting for the throttle.
    trace_sent: Option<Instant>,
    trace_pending: bool,
}

/// Runs when the rotation thread ends, however it ends: normally, the held work runs; after a panic, `rotating` is
/// cleared and the held work is dropped (its prompts given back to the composer) with a log line, so the tab never holds
/// everything forever.
struct RotationEnd {
    chat: Chat,
}

impl Drop for RotationEnd {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return self.chat.rotation_ended();
        }
        let dropped: Vec<PageMessage> = {
            let mut inner = self.chat.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.rotating = false;
            inner.replaying = false;
            inner.held.drain(..).collect()
        };
        crate::note(&format!(
            "tab {}: the session rotation failed unexpectedly; {} held message(s) dropped",
            self.chat.session,
            dropped.len()
        ));
        self.chat.give_back(dropped, "Aria could not send this message; send it again");
    }
}

/// What an undo puts back.
struct Rewind {
    /// The message's entry in Pi's session.
    entry: String,
    /// The checkpoint taken before it, if any.
    checkpoint: Option<String>,
    /// Put the folder back to that checkpoint.
    files: bool,
    /// Fork the conversation before the message.
    conversation: bool,
}

/// The rewind list: the person's messages in the conversation (Pi's `get_fork_messages`), oldest first, each with the
/// checkpoint taken before it and what its turn changed. Pi's text may carry the dock's attachment lines; the same text
/// more than once pairs newest with newest. `kept` is newest first, as the checkpoints list them.
fn rewind_entries(messages: &Value, kept: &[crate::checkpoints::Checkpoint]) -> Value {
    let mine = |pi_text: &str| -> String {
        pi_text.lines().filter(|l| !l.starts_with("[attached")).collect::<Vec<_>>().join("\n").trim().to_string()
    };
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut list: Vec<Value> = messages
        .as_array()
        .into_iter()
        .flatten()
        .rev()
        .filter_map(|m| {
            let entry = m["entryId"].as_str()?;
            let text = mine(m["text"].as_str()?);
            let rank = seen.entry(text.clone()).or_default();
            let checkpoint = kept.iter().filter(|c| c.text.trim() == text).nth(*rank);
            *rank += 1;
            Some(json!({
                "entry": entry,
                "text": text,
                "checkpoint": checkpoint.map(|c| c.id.clone()),
                "files": checkpoint.map_or(0, |c| c.files),
                "added": checkpoint.map_or(0, |c| c.added),
                "removed": checkpoint.map_or(0, |c| c.removed),
            }))
        })
        .collect();
    list.reverse();
    Value::Array(list)
}

/// What must wait for a rotation to end: everything that reaches Pi's session or restarts it.
fn waits_for_rotation(message: &PageMessage) -> bool {
    matches!(
        message,
        PageMessage::ChatSend { .. }
            | PageMessage::PiCommand { .. }
            | PageMessage::ChatBash { .. }
            | PageMessage::SetModel { .. }
            | PageMessage::ChatClearQueue { .. }
            | PageMessage::EnginesRestart
    )
}

/// Must this page message wait in `held`? During a rotation: everything that reaches Pi's session or restarts it, and
/// `chatLoad` (its `get_messages` could read a half-seeded session). During a compaction the tab has not asked for
/// itself: what Pi would refuse meanwhile ("Cannot submit a prompt while compaction is in progress"). While anything is
/// held or replayed: those too, so they keep their order.
fn must_wait(inner: &Inner, message: &PageMessage) -> bool {
    match message {
        PageMessage::ChatLoad { .. } => inner.rotating || inner.replaying,
        message if waits_for_rotation(message) => holding(inner),
        _ => false,
    }
}

fn holding(inner: &Inner) -> bool {
    inner.rotating || inner.replaying || !inner.held.is_empty() || compaction_holds(inner)
}

/// Pi compacts and the tab has no prompt of its own waiting on that compaction.
fn compaction_holds(inner: &Inner) -> bool {
    inner.compacting && inner.pending_prompts.is_empty()
}

/// Notes a prompt as sent before anything slow happens (attachments, the model list, the send itself): from then on
/// the tab is not quiet, so no rotation or compaction starts under it. The id is the one the prompt is sent with.
fn reserve_prompt(inner: &mut Inner, text: &str) -> String {
    let id = Engine::new_id();
    inner.pending_prompts.insert(id.clone(), text.to_string());
    id
}

/// The tab is quiet: its Pi runs, with no run going, no prompt waiting to be accepted, no dialog open, no compaction
/// under way and no replay of held messages. Messages still held do not count: a rotation due after a compaction
/// goes first, so they reach the new segment.
fn quiet(inner: &Inner) -> bool {
    inner.engine.is_some()
        && !inner.rotating
        && !inner.replaying
        && !inner.running
        && !inner.compacting
        && !inner.restart_pending
        && inner.dialogs.is_empty()
        && inner.pending_prompts.is_empty()
}

/// What `get_state` tells about the model and thinking level. `running` is deliberately not taken from
/// it: the events (`agent_start`, `agent_settled`) own that, and a late reply must not undo them.
fn apply_state(inner: &mut Inner, data: &Value) {
    inner.model = model_of(data).or(inner.model.take());
    inner.thinking = data["thinkingLevel"].as_str().map(str::to_string);
    if let Some(file) = data["sessionFile"].as_str().filter(|file| !file.is_empty()) {
        inner.session_file = Some(PathBuf::from(file));
    }
}

/// The `provider/id` of the model a `get_state` reply names.
fn model_of(state: &Value) -> Option<String> {
    state["model"]["provider"].as_str().zip(state["model"]["id"].as_str()).map(|(p, id)| format!("{p}/{id}"))
}

/// The session id a `get_state` reply names.
fn session_of(state: &Value) -> Option<&str> {
    state["sessionId"].as_str().filter(|id| !id.is_empty())
}

#[derive(Clone)]
pub struct Chat {
    /// The session (tab) this chat is: its id everywhere (sockets, jobs, notifications); Pi runs on its current segment.
    session: String,
    ctx: Arc<ChatCtx>,
    emit: Emit,
    inner: Arc<Mutex<Inner>>,
    /// Starting Pi is check-then-act across threads (a send and a `/` picker can race); one at a time.
    spawn_lock: Arc<Mutex<()>>,
    /// One operation at a time on this session's checkpoints (a snapshot, its changes, an undo, the list).
    checkpoint_lock: Arc<Mutex<()>>,
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
        Chat {
            session,
            ctx,
            emit,
            inner: Arc::new(Mutex::new(Inner::default())),
            spawn_lock: Arc::new(Mutex::new(())),
            checkpoint_lock: Arc::new(Mutex::new(())),
        }
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
            compacting: inner.compacting && !inner.running,
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
        let (stopped, rotating, reserved) = {
            let mut inner = self.inner.lock().unwrap();
            if must_wait(&inner, &message) {
                inner.held.push_back(message);
                return;
            }
            // Stop: the held prompts never start (whatever else was held still runs).
            let stopped: Vec<PageMessage> = if matches!(message, PageMessage::ChatAbort) {
                let (sends, rest) = std::mem::take(&mut inner.held).into_iter().partition(|held| {
                    matches!(held, PageMessage::ChatSend { .. } | PageMessage::ChatClearQueue { .. })
                });
                inner.held = rest;
                Vec::from(sends)
            } else {
                Vec::new()
            };
            // Under the same lock as the hold check: a rotation cannot start between the two.
            let reserved = match &message {
                PageMessage::ChatSend { text, .. } => Some(reserve_prompt(&mut inner, text)),
                _ => None,
            };
            (stopped, inner.rotating, reserved)
        };
        self.give_back(stopped, "Stopped before it was sent");
        // Stop during a rotation: Pi has nothing running, and the rotation is not cut. Otherwise (a replay of what was
        // held under way, a compaction) the stop reaches Pi.
        if rotating && matches!(message, PageMessage::ChatAbort) {
            return;
        }
        self.dispatch(message, reserved);
    }

    /// Held messages that will never be sent: each prompt leaves the thread and its text goes back to the composer, as
    /// for a prompt Pi refused.
    fn give_back(&self, held: impl IntoIterator<Item = PageMessage>, why: &str) {
        for message in held {
            if let PageMessage::ChatSend { text, .. } = message {
                self.prompt_failed(why);
                self.emit(DockMessage::ChatEvent {
                    event: json!({"type": "command_failed", "command": "prompt", "error": why, "text": text}),
                });
            }
        }
    }

    /// `reserved` is the id a `ChatSend` was noted under (`reserve_prompt`); one is taken here when it has none.
    fn dispatch(&self, message: PageMessage, reserved: Option<String>) {
        match message {
            PageMessage::ChatLoad { fresh } => {
                self.looked_at();
                self.load(fresh)
            }
            PageMessage::ChatSend { text, behavior, attachments } => {
                let id = reserved.unwrap_or_else(|| reserve_prompt(&mut self.inner.lock().unwrap(), &text));
                self.send_prompt(id, &text, behavior.as_deref(), &attachments)
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
            PageMessage::ChatCheckpoints { id } => self.rewind_list(id),
            PageMessage::ChatUndo { id, entry, checkpoint, files, conversation } => {
                self.undo(id, Rewind { entry, checkpoint, files, conversation })
            }
            PageMessage::ChatBash { id, command } => {
                self.pi_request(id, "bash", json!({ "command": command }), BASH_TIMEOUT)
            }
            // `/reload`: this tab's Pi starts again (reading its extensions, skills and templates), after its run.
            PageMessage::EnginesRestart => self.restart_after_run(),
            PageMessage::SetModel { id, model } => self.set_model(id, model),
            PageMessage::TraceSet { on } => self.trace_set(on),
            PageMessage::ChatEarlier { cursor } => {
                let chat = self.clone();
                std::thread::spawn(move || {
                    let page = aria_core::history::earlier_page(
                        &agent::sessions_dir(&chat.ctx.paths),
                        &chat.segments(),
                        cursor,
                    );
                    chat.emit(DockMessage::ChatEvent {
                        event: for_page(
                            json!({"type": "earlier", "messages": page.messages, "cursor": page.cursor, "more": page.more}),
                        ),
                    });
                });
            }
            PageMessage::ListFiles { id, query } => {
                let chat = self.clone();
                std::thread::spawn(move || {
                    let root = chat.folder();
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
                        chat.trace_changed();
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
        self.shown(
            json!({"type": "ui_notice", "request": {"method": "notify", "key": key, "vars": vars, "notifyType": "warning"}}),
        );
    }

    /// The turn watchdog: it covers the model being silent. A run that got nothing from Pi (no event, reply or UI
    /// request) for `limit` is stopped with `abort` alone, once per run, and the chat says why; the run then settles as
    /// usual. Unlike the page's stop it leaves Pi's queue alone: what the person queued while the turn hung still runs.
    /// A dialog waiting for the person or a tool under way pauses it: its clock starts again when the dialog closes or
    /// the last tool ends. A compaction pauses it too. A zero `limit` turns it off. True when it stopped the run.
    pub fn check_stalled(&self, now: Instant, limit: Duration) -> bool {
        let engine = {
            let mut inner = self.inner.lock().unwrap();
            if limit.is_zero() || !inner.running || inner.watchdog_fired {
                return false;
            }
            if !inner.dialogs.is_empty() || !inner.tools.is_empty() || inner.compacting {
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
        let _ = engine.send("abort", json!({}));
        self.notice_key("chat.watchdog.stopped", json!({"minutes": limit.as_secs().div_ceil(60)}));
        true
    }

    /// A warning in the chat, as an extension's `notify` shows.
    fn notice(&self, message: String) {
        self.shown(
            json!({"type": "ui_notice", "request": {"method": "notify", "message": message, "notifyType": "warning"}}),
        );
    }

    /// An event the chat shows that Pi does not keep (a notice, an instruction's changes): kept with its time, so the
    /// history brings it back in place, then sent.
    fn shown(&self, event: Value) {
        if let Some(file) = notices_file(&self.ctx.paths.root, &self.session) {
            let line = json!({"at": now_ms(), "event": event}).to_string();
            let mut lines: Vec<String> =
                std::fs::read_to_string(&file).unwrap_or_default().lines().map(str::to_string).collect();
            lines.push(line);
            let start = lines.len().saturating_sub(NOTICES_KEPT);
            if let Some(dir) = file.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(&file, lines[start..].join("\n") + "\n");
        }
        self.emit(DockMessage::ChatEvent { event });
    }

    /// The notices this tab kept, oldest first.
    fn kept_notices(&self) -> Vec<(i64, Value)> {
        let Some(file) = notices_file(&self.ctx.paths.root, &self.session) else { return Vec::new() };
        std::fs::read_to_string(file)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter_map(|kept| Some((kept["at"].as_i64()?, kept["event"].clone())))
            .collect()
    }

    /// After the conversation went back: the notices newer than its last message go with what was undone.
    fn forget_notices_after(&self, at: i64) {
        let Some(file) = notices_file(&self.ctx.paths.root, &self.session) else { return };
        let kept: Vec<String> = self
            .kept_notices()
            .into_iter()
            .filter(|(when, _)| *when <= at)
            .map(|(when, event)| json!({"at": when, "event": event}).to_string() + "\n")
            .collect();
        let _ = std::fs::write(file, kept.concat());
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
        let socket = match JobsSocket::bind(
            &path,
            self.session.clone(),
            self.ctx.jobs.clone(),
            launcher,
            self.ctx.schedules.clone(),
            self.ctx.paths.clone(),
        ) {
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
        // A scheduled task's agents run under its grants too, never the person's wider settings.
        if let Some(grants) = self.inner.lock().unwrap().scheduled.as_ref().map(|s| s.grants.clone()) {
            env.push(("ARIA_GRANTS".into(), grants));
        }
        let mut unset = pi::unset_for_pi(std::env::vars());
        // No background work inside background work.
        unset.push("ARIA_JOBS_SOCK".into());
        Some((self.ctx.paths.bin().join("pi"), args, env, unset, worktrees))
    }

    fn current_engine(&self) -> Option<Arc<Engine>> {
        self.inner.lock().unwrap().engine.clone().filter(|engine| engine.is_alive())
    }

    /// `fresh` is a page that just booted: a Pi that was already running replays its conversation to it (a Pi started
    /// here announces itself, once).
    fn load(&self, fresh: bool) {
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
            let running = self.current_engine();
            let engine = self.ensure_engine();
            // Only the Pi that was already running: one started just now has announced itself.
            if let (true, Some(running), Ok(engine)) = (fresh, running, engine) {
                if Arc::ptr_eq(&running, &engine) {
                    let chat = self.clone();
                    std::thread::spawn(move || chat.announce(&engine));
                }
            }
        }
        let open: Vec<(String, Value)> =
            self.inner.lock().unwrap().dialogs.iter().map(|(id, raw)| (id.clone(), raw.clone())).collect();
        for (id, raw) in open {
            let method = raw["method"].as_str().unwrap_or_default().to_string();
            self.emit(DockMessage::ChatDialog { id, method, request: raw, hidden: false });
        }
    }

    /// Sends a message under `id` (already in `pending_prompts`; taken out again when it is not sent). Its attachments are
    /// saved first (a message that cannot be saved is refused, and its text goes back to the composer, as Pi's own
    /// refusals do); images go to Pi as images only when the session's model can see them.
    fn send_prompt(&self, id: String, text: &str, behavior: Option<&str>, attachments: &[Attachment]) {
        let unsent = || {
            self.inner.lock().unwrap().pending_prompts.remove(&id);
        };
        let refuse = |error: &str| {
            unsent();
            self.prompt_failed(error);
            self.emit(DockMessage::ChatEvent {
                event: json!({"type": "command_failed", "command": "prompt", "error": error, "text": text}),
            })
        };
        let engine = match self.ensure_engine() {
            Ok(engine) => {
                // A Pi started just now forgot the prompts of the one before, this one's note included.
                self.inner.lock().unwrap().pending_prompts.insert(id.clone(), text.to_string());
                engine
            }
            Err(code) => {
                unsent();
                self.emit(self.state_message(Some(code)));
                self.prompt_failed(&format!("Pi could not start ({code})"));
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
        // The folder as it is before this message, so its changes can be told and undone.
        // A `/command` (an extension's or Pi's) changes no files of its own: it gets no checkpoint.
        if behavior.is_none() && !text.trim_start().starts_with('/') {
            self.take_checkpoint(text);
        }
        let (message, images) = crate::attachments::message(text, &saved);
        let mut fields = json!({"message": message});
        if !images.is_empty() {
            fields["images"] = json!(images);
        }
        if let Some(behavior) = behavior {
            fields["streamingBehavior"] = json!(behavior);
        }
        // The page's own text is noted under `id`: what its thread and queue show, and what a refusal gives back.
        if engine.send_as(&id, "prompt", fields).is_err() {
            refuse("Pi is no longer running");
        }
    }

    /// The checkpoints of this session's folder (the one its Pi works in), or why it keeps none, as the client reads it.
    /// None for the projects folder itself: it holds the projects, each with its own `.git` a checkpoint cannot see
    /// into, so keeping it would be heavy and useless.
    fn checkpoints(&self) -> Result<crate::checkpoints::Checkpoints, &'static str> {
        if self.in_projects_folder() {
            return Err("projects");
        }
        crate::checkpoints::Checkpoints::open(&self.ctx.paths.root.join("checkpoints"), &self.session, &self.folder())
            .map_err(|off| off.code())
    }

    /// The folder this session's Pi works in.
    fn folder(&self) -> std::path::PathBuf {
        let workdir = self.inner.lock().unwrap().workdir.clone();
        workdir.or_else(|| self.own_folder()).unwrap_or_else(|| agent::workdir(self.settings().as_ref()))
    }

    /// Whether this session works in the projects folder itself.
    fn in_projects_folder(&self) -> bool {
        let (folder, projects) = (self.folder(), agent::workdir(self.settings().as_ref()));
        match (folder.canonicalize(), projects.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => folder == projects,
        }
    }

    /// Keeps the folder before a turn's first message (once per turn).
    fn take_checkpoint(&self, text: &str) {
        if self.inner.lock().unwrap().run_checkpoint.is_some() {
            return;
        }
        let _one = self.checkpoint_lock.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(checkpoints) = self.checkpoints() else { return };
        match checkpoints.take(text) {
            Ok(checkpoint) => self.inner.lock().unwrap().run_checkpoint = Some(checkpoint),
            Err(error) => crate::note(&format!("tab {}: no checkpoint: {error}", self.session)),
        }
    }

    /// A turn settled: what it changed in the folder, told when it changed something.
    fn tell_changes(&self) {
        let Some(checkpoint) = self.inner.lock().unwrap().run_checkpoint.take() else { return };
        let chat = self.clone();
        std::thread::spawn(move || {
            let _one = chat.checkpoint_lock.lock().unwrap_or_else(|e| e.into_inner());
            let Ok(checkpoints) = chat.checkpoints() else { return };
            if let Ok(changes) = checkpoints.changes_since(&checkpoint) {
                if changes.files > 0 {
                    let _ = checkpoints.set_changes(&checkpoint, changes);
                    chat.shown(
                        json!({"type": "changes", "files": changes.files, "added": changes.added, "removed": changes.removed}),
                    );
                }
            }
        });
    }

    /// The messages the conversation can go back to, with their checkpoints (none where the folder keeps none: `off`
    /// says why). Off the caller's thread: Pi may have to start.
    fn rewind_list(&self, id: u64) {
        let chat = self.clone();
        std::thread::spawn(move || {
            let (kept, off) = {
                let _one = chat.checkpoint_lock.lock().unwrap_or_else(|e| e.into_inner());
                match chat.checkpoints() {
                    Ok(c) => (c.list(), None),
                    Err(why) => (Vec::new(), Some(why.to_string())),
                }
            };
            let messages = chat
                .ensure_engine()
                .ok()
                .and_then(|engine| engine.request("get_fork_messages", json!({}), REQUEST_TIMEOUT).ok())
                .filter(|r| r.success)
                .map(|r| r.data["messages"].clone())
                .unwrap_or_else(|| json!([]));
            chat.emit(DockMessage::Checkpoints { id, list: rewind_entries(&messages, &kept), off });
        });
    }

    /// Back to before a message: its files (to the checkpoint taken before it), its conversation (Pi forks at its
    /// entry; the tab records the new session and the conversation is told again), or both. The conversation's place is
    /// found before any file moves. Only on a quiet tab; what arrives meanwhile waits, as during a rotation.
    fn undo(&self, id: u64, rewind: Rewind) {
        {
            let mut inner = self.inner.lock().unwrap();
            if !quiet(&inner) {
                drop(inner);
                return self.emit(DockMessage::Error { id, code: "busy".into() });
            }
            inner.rotating = true;
        }
        let chat = self.clone();
        std::thread::spawn(move || {
            let _end = RotationEnd { chat: chat.clone() };
            let _one = chat.checkpoint_lock.lock().unwrap_or_else(|e| e.into_inner());
            let failed = |chat: &Chat, why: &str| {
                crate::note(&format!("tab {}: undo failed: {why}", chat.session));
                chat.emit(DockMessage::Error { id, code: "failed".into() });
            };
            let engine = match chat.ensure_engine() {
                Ok(engine) => engine,
                Err(code) => return failed(&chat, code),
            };
            let checkpoints = chat.checkpoints().ok();
            let target = rewind.checkpoint.as_ref().and_then(|wanted| {
                checkpoints.as_ref().and_then(|c| c.list().into_iter().find(|kept| &kept.id == wanted))
            });
            if rewind.files && target.is_none() {
                return failed(&chat, "unknown checkpoint");
            }
            if rewind.conversation {
                // Nothing moves unless the conversation can go back too.
                let there = engine
                    .request("get_fork_messages", json!({}), REQUEST_TIMEOUT)
                    .ok()
                    .filter(|r| r.success)
                    .and_then(|r| r.data["messages"].as_array().cloned())
                    .is_some_and(|messages| messages.iter().any(|m| m["entryId"] == json!(rewind.entry)));
                if !there {
                    return failed(&chat, "the message is not in the conversation");
                }
            }
            if let (true, Some(c), Some(target)) = (rewind.files, &checkpoints, &target) {
                if let Err(error) = c.restore(target) {
                    return failed(&chat, &error.to_string());
                }
            }
            if rewind.conversation {
                match engine.request("fork", json!({ "entryId": rewind.entry }), REQUEST_TIMEOUT) {
                    Ok(response) if response.success => {}
                    _ => return failed(&chat, "Pi did not fork"),
                }
                if let (Some(c), Some(target)) = (&checkpoints, &target) {
                    let _ = c.forget_from(target);
                }
                // What the notices said after the last message left goes with it.
                let last =
                    engine.request("get_messages", json!({}), REQUEST_TIMEOUT).ok().filter(|r| r.success).and_then(
                        |r| r.data["messages"].as_array()?.iter().rev().find_map(|m| m["timestamp"].as_i64()),
                    );
                chat.forget_notices_after(last.unwrap_or(0));
                chat.emit(DockMessage::ChatEvent { event: json!({"type": "session_reset"}) });
                chat.announce(&engine);
            }
            chat.emit(DockMessage::Saved { id });
        });
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
        if self.inner.lock().unwrap().closed {
            return Err("session-closed"); // a late message for a closed tab never brings its Pi back
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
        // A tab opened from a terminal works where it was opened; a folder gone since then is never made again.
        let workdir = match self.own_folder() {
            Some(own) if own.is_dir() => own,
            Some(own) => {
                self.notice_key("chat.folderMissing", json!({"path": own.display().to_string()}));
                working_folder(settings.as_ref()).ok_or("workdir-unavailable")?
            }
            None => working_folder(settings.as_ref()).ok_or("workdir-unavailable")?,
        };
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
                self.current_segment(),
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
        // The `aria` the secrets extension reads a stored secret with (`aria secret`), for a command that names it.
        if let Some(vault) = self.ctx.paths.vault.as_ref() {
            spec.env.push(("ARIA_EXE".into(), vault.exe.display().to_string()));
        }
        let grants = self.inner.lock().unwrap().scheduled.as_ref().map(|s| s.grants.clone());
        if let Some(grants) = grants {
            spec.env.push(("ARIA_GRANTS".into(), grants));
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
            inner.compacting = false;
            inner.workdir = Some(workdir);
            inner.compact_asked = false;
            inner.session_file = None;
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
            aria_core::quota::source_for(&model, &aria_core::auth::AuthStore::for_paths(&self.ctx.paths), now_ms);
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
            self.reconcile(&response.data);
            self.emit(self.state_message(None));
        }
        if let Some(response) = ask("get_messages") {
            if !self.is_current(engine) {
                return;
            }
            let messages = with_notices(&response.data["messages"], &self.kept_notices());
            self.emit(DockMessage::ChatEvent { event: for_page(json!({"type": "history", "messages": messages})) });
        }
        self.refresh_context(engine);
        // Not forced: a run that already settled may have asked a moment ago.
        self.refresh_quota(false);
        if self.is_current(engine) {
            self.when_quiet();
        }
    }

    /// The tab's segments (Pi session ids, oldest first) as `sessions.json` keeps them; a tab not in it is its own id.
    fn segments(&self) -> Vec<String> {
        let list = sessions::load(&self.ctx.paths);
        list.sessions
            .iter()
            .find(|entry| entry.id == self.session)
            .map(sessions::segments_of)
            .unwrap_or_else(|| vec![self.session.clone()])
    }

    /// The Pi session the tab runs on: its last segment.
    fn current_segment(&self) -> String {
        self.segments().pop().unwrap_or_else(|| self.session.clone())
    }

    /// The tab may have become quiet (a run settled, a compaction ended, its Pi started). A rotation that is due is
    /// tried; otherwise a segment file over `SEGMENT_LIMIT` is compacted (once per Pi), and the compaction's end then
    /// rotates it. The work runs on its own thread.
    fn when_quiet(&self) {
        let (engine, rotate, file) = {
            let mut inner = self.inner.lock().unwrap();
            if !quiet(&inner) {
                return;
            }
            let Some(engine) = inner.engine.clone() else { return };
            let rotate = std::mem::take(&mut inner.rotate_due);
            inner.rotating = rotate;
            let file = if inner.compact_asked { None } else { inner.session_file.clone() };
            (engine, rotate, file)
        };
        if rotate {
            let chat = self.clone();
            std::thread::spawn(move || {
                let _end = RotationEnd { chat: chat.clone() };
                chat.rotate(&engine);
            });
            return;
        }
        let Some(file) = file.filter(|file| std::fs::metadata(file).is_ok_and(|meta| meta.len() > SEGMENT_LIMIT))
        else {
            return;
        };
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.compact_asked || inner.rotate_due || !quiet(&inner) {
                return;
            }
            inner.compact_asked = true;
        }
        let session = self.session.clone();
        std::thread::spawn(move || {
            crate::note(&format!("tab {session}: its session file {} is over 8 MB: compacting it", file.display()));
            match engine.request("compact", json!({}), COMPACT_TIMEOUT) {
                Ok(response) if response.success => {}
                Ok(response) => crate::note(&format!(
                    "tab {session}: the compaction of an oversized session failed: {}",
                    response.error.unwrap_or_default()
                )),
                Err(error) => {
                    crate::note(&format!("tab {session}: the compaction of an oversized session failed: {error}"))
                }
            }
        });
    }

    /// The rotation is over: what was held runs, then the tab may be quiet again.
    fn rotation_ended(&self) {
        self.inner.lock().unwrap().rotating = false;
        self.release_held();
    }

    /// Something that held the page's messages may have ended (a compaction, a run's start or end, Pi's exit): what is
    /// held runs, on its own thread.
    fn release_soon(&self) {
        if self.inner.lock().unwrap().held.is_empty() {
            return;
        }
        let chat = self.clone();
        std::thread::spawn(move || chat.release_held());
    }

    /// What was held runs, in order, once nothing holds it (later arrivals queue behind it), then the tab may be quiet
    /// again. One replay at a time: a call while another replays leaves the work to it. A prompt sent after another
    /// one of the replay, or while a run goes, is queued as a follow-up, as the page's own queue does (Pi refuses a
    /// plain prompt while it streams). A closed tab drops what it held, and no Pi starts for it.
    fn release_held(&self) {
        let (mut owner, mut prompted) = (false, false);
        loop {
            let (next, reserved) = {
                let mut inner = self.inner.lock().unwrap();
                if inner.closed {
                    inner.held.clear();
                    inner.replaying = false;
                    return;
                }
                if !owner && inner.replaying {
                    return;
                }
                if inner.rotating || compaction_holds(&inner) {
                    // Held again (a compaction began meanwhile): its end releases the rest.
                    if owner {
                        inner.replaying = false;
                    }
                    return;
                }
                let Some(mut message) = inner.held.pop_front() else {
                    inner.replaying = false;
                    break;
                };
                owner = true;
                inner.replaying = true;
                let mut reserved = None;
                if let PageMessage::ChatSend { text, behavior, .. } = &mut message {
                    if behavior.is_none() && (prompted || inner.running) {
                        *behavior = Some("followUp".into());
                    }
                    prompted = true;
                    reserved = Some(reserve_prompt(&mut inner, text));
                }
                (message, reserved)
            };
            self.dispatch(next, reserved);
        }
        self.when_quiet();
    }

    /// Pi's live session, from a `get_state` reply: when it is not the tab's current segment (a rotation whose answer
    /// was lost, or that failed after Pi switched), it becomes it. Returns that session id.
    fn reconcile(&self, state: &Value) -> Option<String> {
        let segment = session_of(state)?.to_string();
        let recorded = {
            let _file = SESSIONS_FILE.lock().unwrap_or_else(|e| e.into_inner());
            if self.current_segment() == segment {
                return Some(segment);
            }
            sessions::append_segment(&self.ctx.paths, &self.session, &segment)
        };
        match recorded {
            Ok(true) => crate::note(&format!("tab {}: the conversation continues in session {segment}", self.session)),
            Ok(false) => crate::note(&format!(
                "tab {}: the tab is not in sessions.json, so session {segment} was not recorded",
                self.session
            )),
            Err(error) => {
                crate::note(&format!("tab {}: session {segment} could not be recorded: {error}", self.session))
            }
        }
        Some(segment)
    }

    /// Sends `/aria-rotate`; when Pi then runs on another session, that session becomes the tab's current segment, and
    /// the tab's model and thinking level are set again (Pi's new session starts from the settings' default ones). Pi's
    /// session is read after the command whatever its answer, so a switch whose answer was lost is still recorded. Any
    /// failure is logged and waits for the next compaction.
    fn rotate(&self, engine: &Engine) {
        let fail = |why: String| {
            crate::note(&format!(
                "tab {}: the session did not rotate ({why}); it is tried again after the next compaction",
                self.session
            ))
        };
        // Unknown to Pi, the command would go to the model as an ordinary message.
        let known = engine
            .request("get_commands", json!({}), REQUEST_TIMEOUT)
            .ok()
            .filter(|response| response.success)
            .is_some_and(|response| {
                response.data["commands"]
                    .as_array()
                    .is_some_and(|commands| commands.iter().any(|c| c["name"] == ROTATE_COMMAND))
            });
        if !known {
            return fail(format!("Pi has no /{ROTATE_COMMAND} command"));
        }
        let state = |engine: &Engine| {
            engine.request("get_state", json!({}), REQUEST_TIMEOUT).ok().filter(|r| r.success).map(|r| r.data)
        };
        // What the tab runs with now, as Pi says (`/effort` may have changed the level), else as Aria last knew it.
        let (model, thinking) = match state(engine) {
            Some(before) => (model_of(&before), before["thinkingLevel"].as_str().map(str::to_string)),
            None => {
                let inner = self.inner.lock().unwrap();
                (inner.model.clone(), inner.thinking.clone())
            }
        };
        let before = self.current_segment();
        let timeout = self.inner.lock().unwrap().rotate_timeout.unwrap_or(ROTATE_TIMEOUT);
        let answer = match engine.request("prompt", json!({"message": format!("/{ROTATE_COMMAND}")}), timeout) {
            Ok(response) if response.success => None,
            Ok(response) => Some(response.error.unwrap_or_else(|| "the command failed".into())),
            Err(error) => Some(error.to_string()),
        };
        let Some(after) = state(engine) else {
            return fail(answer.unwrap_or_else(|| "no state after the command".into()));
        };
        if !self.is_current(engine) {
            return fail("Pi stopped meanwhile".into());
        }
        apply_state(&mut self.inner.lock().unwrap(), &after);
        let Some(segment) = self.reconcile(&after) else {
            return fail(answer.unwrap_or_else(|| "Pi did not say its session".into()));
        };
        if segment == before {
            return fail(answer.unwrap_or_else(|| "Pi stayed on the same session".into()));
        }
        if let Some(why) = answer {
            crate::note(&format!("tab {}: /{ROTATE_COMMAND} answered badly ({why}) but Pi switched", self.session));
        }
        self.keep_model(engine, &after, model, thinking);
    }

    /// After a rotation: the model and thinking level the tab had are set again where the new session differs, and the
    /// page gets the state and the context's fill.
    fn keep_model(&self, engine: &Engine, after: &Value, model: Option<String>, thinking: Option<String>) {
        let mut changed = false;
        if let Some((provider, model_id)) =
            model.as_deref().filter(|m| model_of(after).as_deref() != Some(*m)).and_then(agent::split_model)
        {
            match engine.request("set_model", json!({"provider": provider, "modelId": model_id}), REQUEST_TIMEOUT) {
                Ok(response) if response.success => changed = true,
                _ => crate::note(&format!("tab {}: the model could not be set again after the rotation", self.session)),
            }
        }
        // After the model: setting a model can change the level Pi allows.
        if let Some(level) = thinking.filter(|level| !level.is_empty()) {
            let now = match changed {
                true => engine.request("get_state", json!({}), REQUEST_TIMEOUT).ok().map(|r| r.data),
                false => Some(after.clone()),
            };
            if now.is_none_or(|now| now["thinkingLevel"].as_str() != Some(&level)) {
                match engine.request("set_thinking_level", json!({"level": level}), REQUEST_TIMEOUT) {
                    Ok(response) if response.success => changed = true,
                    _ => crate::note(&format!(
                        "tab {}: the thinking level could not be set again after the rotation",
                        self.session
                    )),
                }
            }
        }
        if changed {
            if let Some(state) = engine.request("get_state", json!({}), REQUEST_TIMEOUT).ok().filter(|r| r.success) {
                if self.is_current(engine) {
                    apply_state(&mut self.inner.lock().unwrap(), &state.data);
                }
            }
        }
        if self.is_current(engine) {
            self.emit(self.state_message(None));
            self.refresh_context(engine);
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

    /// A command of Pi's own from the `/` menu. Only the ones in `PI_COMMANDS` are sent; the page cannot reach the rest.
    fn pi_command(&self, id: u64, command: String, args: Value) {
        let Some(&(command, timeout)) = PI_COMMANDS.iter().find(|(name, _)| *name == command) else {
            let error = Some("command-unknown".into());
            return self.emit(DockMessage::PiReply { id, ok: false, data: Value::Null, error });
        };
        let args = if args.is_object() { args } else { json!({}) };
        self.pi_request(id, command, args, timeout);
    }

    /// Sends Pi one of its commands and answers the page (or client) with `piReply`.
    fn pi_request(&self, id: u64, command: &'static str, args: Value, timeout: Duration) {
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
        self.report_face();
        self.update_status();
    }

    /// What this session is doing, for Aria's face in the bar.
    fn report_face(&self) {
        let activity = {
            let inner = self.inner.lock().unwrap();
            let mut tools: Vec<String> = inner.tools.values().cloned().collect();
            tools.sort();
            face::Activity {
                running: inner.running,
                tools,
                asking: !inner.dialogs.is_empty(),
                scheduled: inner.scheduled_run,
                outcome: inner.outcome,
            }
        };
        face::report(&self.session, activity);
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
                    self.trace_event(&raw);
                    match raw["type"].as_str() {
                        Some("agent_start") => {
                            {
                                let mut inner = self.inner.lock().unwrap();
                                inner.running = true;
                                inner.watchdog_fired = false;
                                inner.tools.clear();
                                inner.outcome = None;
                                inner.compacting = false;
                            }
                            self.update_status();
                            self.release_soon();
                        }
                        Some("compaction_start") => self.inner.lock().unwrap().compacting = true,
                        // A queued message Pi takes now: the folder before it is kept (once per turn).
                        Some("message_start") if raw["message"]["role"] == "user" => {
                            let content = &raw["message"]["content"];
                            let text = content.as_str().map(str::to_string).unwrap_or_else(|| {
                                content.as_array().map_or(String::new(), |blocks| {
                                    blocks.iter().filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join("")
                                })
                            });
                            let text: String =
                                text.lines().filter(|l| !l.starts_with("[attached")).collect::<Vec<_>>().join("\n");
                            self.take_checkpoint(text.trim_end());
                        }
                        // The router's effort or /effort changed Pi's level: the header and every terminal say so.
                        Some("thinking_level_changed") => {
                            if let Some(level) = raw["level"].as_str() {
                                self.inner.lock().unwrap().thinking = Some(level.to_string());
                                self.emit(self.state_message(None));
                            }
                        }
                        // A compaction (automatic or asked) rotates the tab once it is quiet: now, or when its run
                        // settles.
                        Some("compaction_end") => {
                            {
                                let mut inner = self.inner.lock().unwrap();
                                inner.compacting = false;
                                // Only a compaction that happened (a failed one has no `result`).
                                if raw["result"].is_object() {
                                    inner.rotate_due = true;
                                }
                            }
                            // A rotation that starts now keeps holding what the compaction held.
                            self.when_quiet();
                            self.release_soon();
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
                                inner.tools.insert(
                                    raw["toolCallId"].as_str().unwrap_or_default().to_string(),
                                    raw["toolName"].as_str().unwrap_or_default().to_string(),
                                );
                                inner.watch.clone().filter(|_| raw["toolName"] == "parallel")
                            };
                            if let Some(watch) = watch {
                                watch.expect_tasks(raw["toolCallId"].as_str(), &raw["args"]);
                            }
                        }
                        Some("message_end") if raw["message"]["role"] == "assistant" => {
                            let text = raw["message"]["content"]
                                .as_array()
                                .map(|blocks| {
                                    blocks.iter().filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join("")
                                })
                                .unwrap_or_default();
                            let mut inner = self.inner.lock().unwrap();
                            if !text.trim().is_empty() {
                                inner.last_answer = text;
                            }
                            inner.last_failed = raw["message"]["stopReason"] == "error";
                        }
                        Some("agent_settled") => {
                            let scheduled_run = std::mem::take(&mut self.inner.lock().unwrap().scheduled_run);
                            if scheduled_run {
                                self.finish_scheduled(None, raw["aborted"] == true);
                            }
                            // Pi drops a run's dialogs silently when the run is aborted: they go here too.
                            let (open, restart) = {
                                let mut inner = self.inner.lock().unwrap();
                                inner.running = false;
                                inner.tools.clear();
                                inner.compacting = false;
                                inner.outcome = if raw["aborted"] == true {
                                    None
                                } else if inner.last_failed {
                                    Some((face::Outcome::Failed, Instant::now()))
                                } else {
                                    Some((face::Outcome::Done, Instant::now()))
                                };
                                let restart = std::mem::take(&mut inner.restart_pending);
                                (inner.dialogs.keys().cloned().collect::<Vec<_>>(), restart)
                            };
                            let unseen = !self.seen();
                            self.inner.lock().unwrap().unseen_done = unseen;
                            self.update_status();
                            self.tell_changes();
                            if unseen && !scheduled_run && raw["aborted"] != true {
                                self.notify_settled();
                            }
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
                                self.when_quiet();
                            }
                            self.release_soon();
                        }
                        _ => {}
                    }
                    self.emit(DockMessage::ChatEvent { event: for_page(raw) });
                    self.report_face();
                }
                EngineEvent::UiRequest { id, method, raw } if is_dialog(&method) => self.open_dialog(id, method, raw),
                EngineEvent::UiRequest { raw, .. } => {
                    let event = json!({"type": "ui_notice", "request": raw});
                    if event["request"]["method"] == "notify" {
                        self.shown(event);
                    } else {
                        self.emit(DockMessage::ChatEvent { event });
                    }
                }
                EngineEvent::Response { id, success, command, error } => {
                    // A refused prompt names its text, so the page can take it back out of the thread or the queue.
                    let text = id.and_then(|id| self.inner.lock().unwrap().pending_prompts.remove(&id));
                    if !success && command == "prompt" {
                        self.prompt_failed(error.as_deref().unwrap_or("Pi refused the message"));
                    }
                    if !success {
                        self.emit(DockMessage::ChatEvent {
                            event: json!({"type": "command_failed", "command": command, "error": error, "text": text}),
                        });
                    }
                }
                EngineEvent::Exited { code } => {
                    if current {
                        self.trace_settle(true);
                    }
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
    /// Can the owner see this chat? The dock is open and showing the chat (not the settings), or a terminal attached to
    /// this session has the focus.
    fn seen(&self) -> bool {
        (self.ctx.dock_visible.load(Ordering::SeqCst) && self.ctx.chat_shown.load(Ordering::SeqCst))
            || self.ctx.watchers.sees(&self.session)
    }

    pub fn dock_visibility_changed(&self) {
        if self.seen() {
            return self.looked_at();
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

    /// What the session is doing: `waiting` (a dialog), `working` (a run), `done` (a run nobody saw end), `idle`.
    fn current_status(inner: &Inner) -> &'static str {
        if !inner.dialogs.is_empty() {
            "waiting"
        } else if inner.running {
            "working"
        } else if inner.unseen_done {
            "done"
        } else {
            "idle"
        }
    }

    /// Says the session's status when it changed.
    fn update_status(&self) {
        let changed = {
            let mut inner = self.inner.lock().unwrap();
            let now = Self::current_status(&inner);
            (now != inner.status || inner.status_at == 0).then(|| {
                inner.status = now;
                inner.status_at = now_ms();
                (now, inner.status_at)
            })
        };
        if let Some((status, at)) = changed {
            self.emit(DockMessage::SessionStatus { session: self.session.clone(), status: status.into(), at });
        }
    }

    /// Says the session's status as it is (a client just asked for the list).
    pub fn say_status(&self) {
        let (status, at) = {
            let inner = self.inner.lock().unwrap();
            (Self::current_status(&inner), inner.status_at)
        };
        self.emit(DockMessage::SessionStatus { session: self.session.clone(), status: status.into(), at });
    }

    /// Someone looks at the session: a run that ended unseen is seen now, and its notification goes.
    fn looked_at(&self) {
        let told = {
            let mut inner = self.inner.lock().unwrap();
            inner.unseen_done = false;
            inner.settled_notice.take()
        };
        if let Some(title) = told {
            self.ctx.notifier.dismiss_titled(&title);
        }
        self.update_status();
    }

    /// A turn ended where nobody looks: one notification for the session (its previous one goes first), with the
    /// answer's first line, or why it did not finish.
    fn notify_settled(&self) {
        // The session's own mark ends the title: a notification is taken away by (a part of) its title, and no other
        // session's title holds it.
        let short: String = self.session.chars().take(6).collect();
        let title = sessions::load(&self.ctx.paths)
            .sessions
            .into_iter()
            .find(|entry| entry.id == self.session)
            .map(|entry| entry.title)
            .filter(|title| !title.trim().is_empty())
            .map_or_else(|| format!("Aria · {short}"), |title| format!("Aria · {title} · {short}"));
        let (answer, failed, previous) = {
            let mut inner = self.inner.lock().unwrap();
            (inner.last_answer.clone(), inner.last_failed, inner.settled_notice.replace(title.clone()))
        };
        let first: String =
            answer.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or_default().chars().take(140).collect();
        let body = match (failed, first.is_empty()) {
            (true, _) => {
                format!("It did not finish: {}", if first.is_empty() { "the provider failed" } else { &first })
            }
            (false, true) => "Done".to_string(),
            (false, false) => first,
        };
        if let Some(previous) = previous {
            self.ctx.notifier.dismiss_titled(&previous);
        }
        self.ctx.notifier.send_finished(&self.session, &title, &body);
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
        self.update_status();
        self.report_face();
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

    /// `/debug` in this tab: on answers with the current trace (`null` before a traced turn); off drops it.
    fn trace_set(&self, on: bool) {
        {
            let mut inner = self.inner.lock().unwrap();
            inner.trace_on = on;
            if !on {
                inner.trace = None;
            }
            inner.trace_sent = Some(Instant::now());
        }
        self.emit(DockMessage::ChatTrace { trace: self.trace_tree().unwrap_or(Value::Null) });
    }

    /// One of Pi's events for the trace: `agent_start` starts a turn's, `agent_settled` ends it, the rest feed it.
    fn trace_event(&self, raw: &Value) {
        let changed = {
            let mut inner = self.inner.lock().unwrap();
            if !inner.trace_on {
                return;
            }
            let now = now_ms();
            match raw["type"].as_str() {
                Some("agent_start") => {
                    inner.trace = Some(crate::trace::Trace::new(now));
                    true
                }
                Some("agent_settled") => {
                    let failed = inner.last_failed || raw["aborted"] == true;
                    drop(inner);
                    // `last_failed` is set by this same event's `message_end`, which came before it.
                    self.trace_settle(failed);
                    return;
                }
                _ => inner.trace.as_mut().is_some_and(|t| t.feed(raw, now)),
            }
        };
        if changed {
            self.trace_changed();
        }
    }

    /// The turn ended (or its Pi died): the root ends and the trace goes out at once.
    fn trace_settle(&self, failed: bool) {
        {
            let mut inner = self.inner.lock().unwrap();
            let Some(trace) = inner.trace.as_mut().filter(|t| t.running()) else { return };
            trace.settle(failed, now_ms());
        }
        self.trace_send();
    }

    /// The trace changed: it goes out now, or once the 250 ms since the last send are over (one wait at a time).
    fn trace_changed(&self) {
        let wait = {
            let mut inner = self.inner.lock().unwrap();
            if inner.trace.is_none() || inner.closed {
                return;
            }
            if inner.trace_pending {
                return;
            }
            let since = inner.trace_sent.map(|at| at.elapsed()).unwrap_or(TRACE_EVERY);
            if since >= TRACE_EVERY {
                None
            } else {
                inner.trace_pending = true;
                Some(TRACE_EVERY - since)
            }
        };
        match wait {
            None => self.trace_send(),
            Some(wait) => {
                let chat = self.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(wait);
                    chat.inner.lock().unwrap().trace_pending = false;
                    chat.trace_send();
                });
            }
        }
    }

    /// The current tree with this turn's subagents, if there is a trace.
    fn trace_tree(&self) -> Option<Value> {
        let (watch, has) = {
            let inner = self.inner.lock().unwrap();
            (inner.watch.clone(), inner.trace.is_some())
        };
        if !has {
            return None;
        }
        let agents = watch.map(|w| w.trace_agents()).unwrap_or_default();
        let inner = self.inner.lock().unwrap();
        inner.trace.as_ref().map(|t| t.tree(&agents, now_ms()))
    }

    fn trace_send(&self) {
        {
            let mut inner = self.inner.lock().unwrap();
            if !inner.trace_on || inner.closed {
                return;
            }
            inner.trace_sent = Some(Instant::now());
        }
        if let Some(tree) = self.trace_tree() {
            self.emit(DockMessage::ChatTrace { trace: tree });
        }
    }

    fn exited(&self, generation: u64, code: Option<i32>) {
        let (open, dismiss) = {
            let mut inner = self.inner.lock().unwrap();
            if inner.generation != generation {
                return;
            }
            inner.engine = None;
            if inner.running {
                inner.outcome = Some((face::Outcome::Failed, Instant::now()));
            }
            inner.running = false;
            inner.tools.clear();
            inner.compacting = false;
            let open: Vec<String> = inner.dialogs.drain().map(|(id, _)| id).collect();
            let dismiss = !inner.notified.is_empty();
            inner.notified.clear();
            (open, dismiss)
        };
        // A scheduled run Pi died in ends as failed, so the person's next turn there is never taken for it.
        self.prompt_failed("Pi stopped before the run finished");
        self.update_status();
        if dismiss {
            self.ctx.notifier.dismiss(&self.session);
        }
        for id in open {
            self.emit(DockMessage::ChatDialogClosed { id });
        }
        self.emit(DockMessage::ChatEngine { state: "exited".into() });
        self.emit(DockMessage::ChatEvent { event: json!({"type": "engine_exited", "code": code}) });
        self.report_face();
        // A compaction it held for is over.
        self.release_soon();
    }

    /// The folder this chat's running Pi works in (None when no Pi runs).
    /// The folder this tab was opened in (`sessions.json`), when it has one of its own.
    fn own_folder(&self) -> Option<PathBuf> {
        let list = sessions::load(&self.ctx.paths);
        list.sessions.into_iter().find(|entry| entry.id == self.session)?.workdir.map(PathBuf::from)
    }

    fn workdir(&self) -> Option<PathBuf> {
        let inner = self.inner.lock().unwrap();
        inner.engine.as_ref().and(inner.workdir.clone())
    }

    /// Restart for a changed credential or package: at once when idle, when the current run settles otherwise.
    /// During a rotation the restart waits for it to end, so the segment Pi switched to is recorded first (and behind
    /// what a compaction holds, in order).
    pub fn restart_when_idle(&self) {
        {
            let mut inner = self.inner.lock().unwrap();
            if holding(&inner) {
                inner.held.push_back(PageMessage::EnginesRestart);
                return;
            }
        }
        self.restart_after_run();
    }

    fn restart_after_run(&self) {
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
    /// server), so nothing of a closed tab keeps running; what it held during a rotation is dropped, and no Pi starts
    /// for it again.
    pub fn close_for_good(&self) {
        {
            let mut inner = self.inner.lock().unwrap();
            inner.closed = true;
            inner.held.clear();
        }
        face::forget(&self.session);
        self.stop(true);
    }

    /// A run is going, a dialog waits for an answer, or the session is rotating (or sending what it held).
    /// Makes this the tab of `task`: its Pi (at its next start) runs with the task's fixed permissions. `servers` are the
    /// connectors in `mcp.json`: each one not granted is read-only.
    fn set_scheduled(&self, task: &aria_core::schedules::Task, servers: &[String]) {
        let connectors: serde_json::Map<String, Value> = servers
            .iter()
            .map(|name| (name.clone(), json!(task.grants.connectors.get(name).map_or("read", String::as_str))))
            .collect();
        let grants = json!({
            "connectors": connectors,
            "sandbox": {"askToLeave": false, "extraWritable": self.ctx.schedules.runnable_writable(task)},
            "noAsk": true,
        });
        self.inner.lock().unwrap().scheduled =
            Some(Scheduled { task: task.id.clone(), title: task.title.clone(), grants: grants.to_string() });
    }

    /// A prompt that never became a run (Pi refused it, or could not start): a scheduled run it was ends as failed.
    fn prompt_failed(&self, why: &str) {
        // The message never ran: its checkpoint is not a turn's, and the next message takes its own.
        self.inner.lock().unwrap().run_checkpoint = None;
        if std::mem::take(&mut self.inner.lock().unwrap().scheduled_run) {
            self.finish_scheduled(Some(why), false);
        }
    }

    /// A scheduled run settled (or never started, `failure` says why): its result goes to the task, and a notification
    /// unless the person is looking.
    fn finish_scheduled(&self, failure: Option<&str>, aborted: bool) {
        let (scheduled, answer, failed) = {
            let inner = self.inner.lock().unwrap();
            (inner.scheduled.clone(), inner.last_answer.clone(), inner.last_failed || failure.is_some())
        };
        let Some(scheduled) = scheduled else { return };
        let answer = failure.map_or(answer, str::to_string);
        let first = answer.lines().map(str::trim).find(|line| !line.is_empty()).unwrap_or_default();
        let summary: String =
            if first.is_empty() { "No answer".to_string() } else { first.chars().take(140).collect() };
        let ok = !aborted && !failed;
        let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64);
        self.ctx
            .schedules
            .record(&scheduled.task, aria_core::schedules::RunResult { at, ok, summary: summary.clone() });
        if !self.seen() {
            let body = if ok { summary } else { format!("It did not finish: {summary}") };
            self.ctx.notifier.send(&self.session, &format!("Aria · {}", scheduled.title), &body);
        }
    }

    fn busy(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        // A rotation too: an update's stop must not cut it before its segment is recorded; and the replay of what it held.
        inner.running || inner.rotating || inner.replaying || !inner.dialogs.is_empty()
    }

    /// The returned thread is closing Pi; joining it waits until Pi is gone.
    fn stop(&self, group: bool) -> Option<std::thread::JoinHandle<()>> {
        let (stale, engine) = {
            let mut inner = self.inner.lock().unwrap();
            inner.generation += 1; // whatever the old Pi still says is ignored
            let engine = inner.engine.take();
            inner.running = false;
            inner.tools.clear();
            inner.compacting = false;
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
        // A compaction it held for is over.
        self.release_soon();
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

    /// Feeds one event to the trace as if Pi had sent it (tests).
    #[cfg(test)]
    fn test_trace_event(&self, raw: Value) {
        self.trace_event(&raw);
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
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| watch.expect_tasks(None, &args)));
        }
    }
}

/// Whether `run_dir` is inside `hidden`, the folder the sandbox hides from boxed commands.
pub(crate) fn hidden_from_the_box(run_dir: &std::path::Path, hidden: Option<&std::path::Path>) -> bool {
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
    /// The turn watchdog's limit as last read from the settings.
    turn_limit: Arc<Mutex<Duration>>,
    /// Scheduled runs waiting for their tab to be free: the task's id and the time it was due (the task is read again
    /// when the tab is free, so a pause, a change or a delete meanwhile counts).
    pending: Arc<Mutex<Vec<(String, i64)>>>,
    /// The dock's one dictation, made the first time a tab dictates (its thread and its model only then).
    #[cfg(feature = "gui")]
    dictation: Arc<Mutex<Option<crate::dictation::Dictation>>>,
}

/// What a dictation event says to the page.
#[cfg(feature = "gui")]
fn dictation_message(kind: crate::dictation::EventKind) -> DockMessage {
    use crate::dictation::{EventKind, State};
    let (state, seconds, progress, size, error) = match kind {
        EventKind::Text(text) => return DockMessage::DictationText { text },
        EventKind::State(State::Idle) => ("idle", None, None, None, None),
        EventKind::State(State::Transcribing) => ("transcribing", None, None, None, None),
        EventKind::State(State::Recording { seconds }) => ("recording", Some(seconds), None, None, None),
        EventKind::State(State::NeedsModel { size }) => ("needsModel", None, None, Some(size), None),
        EventKind::State(State::Downloading { progress }) => ("downloading", None, Some(progress), None, None),
        EventKind::State(State::Error(error)) => ("error", None, None, None, Some(error.to_string())),
    };
    DockMessage::DictationState { state: state.into(), seconds, progress, size, error }
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
        // A tab with its own folder stays where it was opened.
        for chat in chats
            .into_iter()
            .filter(|chat| chat.own_folder().is_none() && chat.workdir().is_some_and(|dir| dir != wanted))
        {
            chat.restart_when_idle();
        }
    }

    /// `ctx.notifier` is wrapped so that one session's answer does not remove another's notification.
    /// A tab's dictation message. The language is Aria's (`settings.assistant.language`); only the tab that dictates
    /// can stop or cancel.
    #[cfg(feature = "gui")]
    fn dictate(&self, session: &str, message: PageMessage) {
        let mut slot = self.dictation.lock().unwrap();
        if slot.is_none() {
            let emit = self.emit.clone();
            *slot = Some(crate::dictation::for_dock(&self.ctx.paths, Self::dictation_events(emit)));
        }
        let dictation = slot.as_ref().unwrap();
        let language = settings::load(&self.ctx.paths)
            .ok()
            .flatten()
            .map(|s| s.assistant.language)
            .filter(|l| l == "es")
            .unwrap_or_else(|| "en".into());
        let owns = dictation.owner().as_deref() == Some(session);
        match message {
            PageMessage::DictationStart => dictation.start(session, &language),
            PageMessage::DictationDownload => dictation.download(session, &language),
            PageMessage::DictationStop if owns => dictation.stop(),
            PageMessage::DictationCancel if owns => dictation.cancel(),
            _ => {}
        }
    }

    /// A service with no window has no microphone and no Whisper: starting or downloading answers the page's "no
    /// microphone" error (`dictationState`), the rest has nothing to stop.
    #[cfg(not(feature = "gui"))]
    fn dictate(&self, session: &str, message: PageMessage) {
        if matches!(message, PageMessage::DictationStart | PageMessage::DictationDownload) {
            (self.emit)(DockMessage::Session {
                session: session.to_string(),
                message: Box::new(DockMessage::DictationState {
                    state: "error".into(),
                    seconds: None,
                    progress: None,
                    size: None,
                    error: Some("no-microphone".into()),
                }),
            });
        }
    }

    #[cfg(feature = "gui")]
    fn dictation_events(emit: Emit) -> Box<dyn Fn(crate::dictation::Event) + Send> {
        Box::new(move |event| {
            emit(DockMessage::Session { session: event.owner, message: Box::new(dictation_message(event.kind)) })
        })
    }

    /// Tests: a dictation of their own (made with the same events as the dock's).
    #[cfg(all(test, feature = "gui"))]
    fn use_dictation(
        &self,
        make: impl FnOnce(Box<dyn Fn(crate::dictation::Event) + Send>) -> crate::dictation::Dictation,
    ) {
        *self.dictation.lock().unwrap() = Some(make(Self::dictation_events(self.emit.clone())));
    }

    /// A closed tab leaves `sessions.json` and every client is told; the last regular one is replaced by a fresh tab,
    /// so there is always one to show.
    fn drop_tab(&self, session: &str) {
        let replacement = {
            let _file = SESSIONS_FILE.lock().unwrap_or_else(|e| e.into_inner());
            let mut list = sessions::load(&self.ctx.paths);
            list.sessions.retain(|entry| entry.id != session);
            let mut replacement = None;
            if !list.sessions.iter().any(|entry| entry.schedule.is_none()) {
                let entry = sessions::SessionEntry {
                    id: sessions::new_id(),
                    title: String::new(),
                    segments: Vec::new(),
                    schedule: None,
                    workdir: None,
                };
                list.active = Some(entry.id.clone());
                list.sessions.push(entry.clone());
                replacement = Some(entry);
            }
            if list.active.as_deref() == Some(session) {
                list.active = list.sessions.iter().find(|entry| entry.schedule.is_none()).map(|entry| entry.id.clone());
            }
            if let Err(error) = sessions::save(&self.ctx.paths, &list) {
                crate::note(&format!("the tabs could not be saved after closing {session}: {error}"));
            }
            replacement
        };
        (self.emit)(DockMessage::SessionRemoved { session: session.to_string() });
        if let Some(entry) = replacement {
            (self.emit)(DockMessage::SessionAdded { id: None, session: entry });
        }
    }

    pub fn new(ctx: ChatCtx, emit: Emit) -> Sessions {
        let notifier: Arc<dyn Notifier> = Arc::new(CountingNotifier::new(ctx.notifier.clone()));
        let ctx = Arc::new(ChatCtx { notifier, ..ctx });
        let sessions = Sessions {
            ctx,
            emit,
            chats: Arc::new(Mutex::new(HashMap::new())),
            closed: Arc::new(Mutex::new(HashSet::new())),
            turn_limit: Arc::new(Mutex::new(Duration::from_secs(
                u64::from(aria_core::settings::Models::default().turn_timeout_minutes) * 60,
            ))),
            pending: Arc::default(),
            #[cfg(feature = "gui")]
            dictation: Arc::default(),
        };
        // A deleted task takes its tab with it; a changed one gives its tab's Pi the new grants when its run settles.
        let closer = sessions.clone();
        sessions.ctx.schedules.on_delete(move |id| closer.close_scheduled(id));
        let refresher = sessions.clone();
        sessions.ctx.schedules.on_update(move |id| refresher.refresh_scheduled(id));
        let emit = sessions.emit.clone();
        sessions.ctx.schedules.on_awaiting(move |id| emit(DockMessage::ScheduleAwaiting { schedule: id.to_string() }));
        sessions
    }

    /// Runs a due task in its tab: the tab is made the first time (and the page told), the run shows as the person's
    /// message (`scheduled_run`), and its prompt goes to the tab's Pi. A tab still busy keeps it until `flush_scheduled`.
    pub fn start_scheduled(&self, task: &aria_core::schedules::Task, due_at: i64) {
        let Some(chat) = self.chat(&task.id) else { return };
        if chat.busy() {
            let mut pending = self.pending.lock().unwrap();
            pending.retain(|(waiting, _)| *waiting != task.id);
            pending.push((task.id.clone(), due_at));
            return;
        }
        let now =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64);
        let late = aria_core::schedules::late(due_at, now);
        {
            let _file = SESSIONS_FILE.lock().unwrap_or_else(|e| e.into_inner());
            if let Err(error) = sessions::add_scheduled(&self.ctx.paths, &task.id, &task.title, &task.id) {
                crate::note(&format!("the tab of the scheduled task {} could not be saved: {error}", task.id));
            }
        }
        (self.emit)(DockMessage::ScheduleTab {
            id: task.id.clone(),
            title: task.title.clone(),
            schedule: task.id.clone(),
        });
        chat.set_scheduled(task, &aria_core::mcp::servers(&self.ctx.paths));
        {
            let mut inner = chat.inner.lock().unwrap();
            inner.scheduled_run = true;
            inner.last_answer.clear();
            inner.last_failed = false;
        }
        chat.emit(DockMessage::ChatEvent {
            event: json!({"type": "scheduled_run", "text": task.prompt, "at": task.when.at, "late": late}),
        });
        chat.handle(PageMessage::ChatSend { text: task.prompt.clone(), behavior: None, attachments: Vec::new() });
    }

    /// Starts the scheduled runs whose tab is free now (called once a second), each from its task as it is now: one
    /// deleted, paused or waiting for approval meanwhile does not run.
    pub fn flush_scheduled(&self) {
        let waiting = std::mem::take(&mut *self.pending.lock().unwrap());
        for (id, due_at) in waiting {
            match self.ctx.schedules.get(&id) {
                Some(task) if task.enabled && !task.awaiting => self.start_scheduled(&task, due_at),
                _ => {}
            }
        }
    }

    /// A deleted task's tab: its Pi and everything it started stop, the tab leaves `sessions.json`, and the page is told.
    fn close_scheduled(&self, id: &str) {
        self.closed.lock().unwrap().insert(id.to_string());
        self.pending.lock().unwrap().retain(|(task, _)| task != id);
        let chat = self.chats.lock().unwrap().remove(id);
        if let Some(chat) = chat {
            chat.close_for_good();
            chat.forget_watch();
            chat.forget_jobs();
        }
        self.close_jobs(id.to_string());
        {
            let _file = SESSIONS_FILE.lock().unwrap_or_else(|e| e.into_inner());
            let mut list = sessions::load(&self.ctx.paths);
            list.sessions.retain(|entry| entry.id != id);
            let _ = sessions::save(&self.ctx.paths, &list);
        }
        (self.emit)(DockMessage::ScheduleRemoved { id: id.to_string() });
    }

    /// A changed task: its tab's Pi starts again with the new grants, when its run settles.
    fn refresh_scheduled(&self, id: &str) {
        let chat = self.chats.lock().unwrap().get(id).cloned();
        let (Some(chat), Some(task)) = (chat, self.ctx.schedules.get(id)) else { return };
        chat.set_scheduled(&task, &aria_core::mcp::servers(&self.ctx.paths));
        chat.restart_when_idle();
    }

    fn schedules_answer(&self, id: u64) {
        let now =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64);
        (self.emit)(DockMessage::Schedules { id, tasks: json!(self.ctx.schedules.list(now)) });
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
            let chat = Chat::new(self.ctx.clone(), tagged, session.to_string());
            // A scheduled task's tab keeps its grants whoever starts its Pi (the page's preload, the person, a run).
            if let Some(task) = self.ctx.schedules.get(session) {
                chat.set_scheduled(&task, &aria_core::mcp::servers(&self.ctx.paths));
            }
            chat
        });
        Some(chat.clone())
    }

    pub fn handle(&self, message: PageMessage) {
        match message {
            PageMessage::Session { session, message } => {
                if matches!(*message, PageMessage::Session { .. }) {
                    return;
                }
                if matches!(
                    *message,
                    PageMessage::DictationStart
                        | PageMessage::DictationStop
                        | PageMessage::DictationCancel
                        | PageMessage::DictationDownload
                ) {
                    return self.dictate(&session, *message);
                }
                if let Some(chat) = self.chat(&session) {
                    let first_look = matches!(*message, PageMessage::ChatLoad { .. });
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
                // The terminal's list says what each open session is doing.
                let chats: Vec<Chat> = self.chats.lock().unwrap().values().cloned().collect();
                for chat in chats {
                    chat.say_status();
                }
            }
            PageMessage::SessionsSave { sessions: entries, active } => {
                let _ = self.ctx.paths.ensure_root();
                // The page never owns a tab's chain of segments: each tab keeps the ones stored for it.
                let _file = SESSIONS_FILE.lock().unwrap_or_else(|e| e.into_inner());
                let stored = sessions::load(&self.ctx.paths);
                let entries = entries
                    .into_iter()
                    .map(|mut entry| {
                        let kept = stored.sessions.iter().find(|kept| kept.id == entry.id);
                        entry.segments = kept.map(|kept| kept.segments.clone()).unwrap_or_default();
                        // Nor its folder: the page does not know it.
                        entry.workdir = entry.workdir.or_else(|| kept.and_then(|kept| kept.workdir.clone()));
                        // Nor which tabs are a scheduled task's: the dock made them.
                        entry.schedule = entry.schedule.or_else(|| kept.and_then(|kept| kept.schedule.clone()));
                        entry
                    })
                    .collect();
                // The scheduled tabs the page has not seen yet stay; those of a deleted task go.
                let tasks = self.ctx.schedules.ids();
                let entries = sessions::merge_scheduled(entries, &stored, &tasks);
                let _ = sessions::save(&self.ctx.paths, &SessionList { sessions: entries, active });
                (self.emit)(DockMessage::SessionsChanged);
            }
            PageMessage::SessionNew { id, workdir } => {
                let _ = self.ctx.paths.ensure_root();
                let added = {
                    let _file = SESSIONS_FILE.lock().unwrap_or_else(|e| e.into_inner());
                    sessions::add(&self.ctx.paths, &sessions::new_id(), "", workdir.as_deref())
                };
                (self.emit)(match added {
                    Ok(session) => DockMessage::SessionAdded { id: Some(id), session },
                    Err(sessions::AddError::TooMany) => DockMessage::Error { id, code: "too-many-sessions".into() },
                    Err(sessions::AddError::Io(error)) => {
                        crate::note(&format!("a new tab could not be saved: {error}"));
                        DockMessage::Error { id, code: "session-failed".into() }
                    }
                });
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
                // A tab that goes while it dictates: the recording is dropped, nothing is written anywhere.
                #[cfg(feature = "gui")]
                if let Some(dictation) = self.dictation.lock().unwrap().as_ref() {
                    if dictation.owner().as_deref() == Some(session.as_str()) {
                        dictation.cancel();
                    }
                }
                let chat = self.chats.lock().unwrap().remove(&session);
                if let Some(chat) = chat {
                    chat.close_for_good();
                    chat.forget_watch();
                    chat.forget_jobs();
                }
                self.close_jobs(session.clone());
                crate::checkpoints::remove(&self.ctx.paths.root.join("checkpoints"), &session);
                remove_notices(&self.ctx.paths.root, &session);
                self.drop_tab(&session);
            }
            PageMessage::JobStop { job } => {
                self.ctx.jobs.stop(&job);
            }
            PageMessage::JobDismiss { job } => self.ctx.jobs.dismiss(&job),
            PageMessage::SchedulesList { id } => self.schedules_answer(id),
            // A plain switch: pause or resume. A task that waits for approval stays waiting.
            PageMessage::ScheduleUpdate { id, schedule, enabled } => {
                if let Err(error) = self.ctx.schedules.set_enabled(&schedule, enabled) {
                    crate::note(&format!("a scheduled task could not be changed: {error}"));
                }
                self.schedules_answer(id);
            }
            // The page's Approve button: the only thing that approves a waiting task (no client may send it).
            PageMessage::ScheduleApprove { id, schedule } => {
                if let Err(error) = self.ctx.schedules.approve(&schedule) {
                    crate::note(&format!("a scheduled task could not be approved: {error}"));
                }
                self.schedules_answer(id);
            }
            PageMessage::ScheduleDelete { id, schedule } => {
                if let Err(error) = self.ctx.schedules.delete(&schedule) {
                    crate::note(&format!("a scheduled task could not be deleted: {error}"));
                }
                self.schedules_answer(id);
            }
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
        // Checkpoints of sessions that are not open any more.
        let root = self.ctx.paths.root.join("checkpoints");
        if let Ok(dirs) = std::fs::read_dir(&root) {
            for dir in dirs.flatten() {
                let name = dir.file_name().to_string_lossy().to_string();
                if !open.contains(&name) {
                    crate::checkpoints::remove(&root, &name);
                }
            }
        }
        // And their notices.
        if let Ok(files) = std::fs::read_dir(self.ctx.paths.root.join("notices")) {
            for file in files.flatten() {
                let name = file.file_name().to_string_lossy().trim_end_matches(".jsonl").to_string();
                if !open.contains(&name) {
                    remove_notices(&self.ctx.paths.root, &name);
                }
            }
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

    /// The turn watchdog's limit, read from the settings on every check (a change applies without a restart).
    /// A file that cannot be read (half written, broken) keeps the last limit read, so a `0` (off) is never turned back
    /// into the default; with no file at all the default applies.
    fn read_limit(paths: &AriaPaths, last: &Mutex<Duration>) -> Duration {
        let mut last = last.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(settings) = settings::load(paths) {
            let minutes = settings.unwrap_or_default().models.turn_timeout_minutes;
            *last = Duration::from_secs(u64::from(minutes) * 60);
        }
        *last
    }

    /// Every 30 s: each session's run that got nothing from its Pi for the settings' limit is stopped (see
    /// `Chat::check_stalled`). Returns how many were.
    pub fn check_stalled(&self, now: Instant) -> usize {
        let limit = Self::read_limit(&self.ctx.paths, &self.turn_limit);
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

/// The most notices a tab keeps (the oldest go first).
const NOTICES_KEPT: usize = 500;

/// Where a tab keeps the notices it was shown: `notices/<session>.jsonl`, one `{ at, event }` per line.
fn notices_file(root: &Path, session: &str) -> Option<PathBuf> {
    let plain = !session.is_empty() && !session.contains(['/', '\\']) && !session.starts_with('.');
    plain.then(|| root.join("notices").join(format!("{session}.jsonl")))
}

/// Removes a closed tab's notices.
pub fn remove_notices(root: &Path, session: &str) {
    if let Some(file) = notices_file(root, session) {
        let _ = std::fs::remove_file(file);
    }
}

/// The history with the kept notices in place: each before the first message newer than it (messages without a time
/// keep theirs), the rest at the end, as `{ role: "ariaNotice", event, timestamp }`.
fn with_notices(messages: &Value, notices: &[(i64, Value)]) -> Value {
    let notice = |(at, event): &(i64, Value)| json!({"role": "ariaNotice", "event": event, "timestamp": at});
    let mut rest = notices.iter().peekable();
    let mut out = Vec::new();
    for message in messages.as_array().into_iter().flatten() {
        if let Some(time) = message["timestamp"].as_i64() {
            while let Some(kept) = rest.next_if(|(at, _)| *at < time) {
                out.push(notice(kept));
            }
        }
        out.push(message.clone());
    }
    out.extend(rest.map(notice));
    Value::Array(out)
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
        fn dismiss_titled(&self, title: &str) {
            self.log.lock().unwrap().push(format!("dismiss:{title}"));
        }
    }

    impl TestNotifier {
        fn log(&self) -> Vec<String> {
            self.log.lock().unwrap().clone()
        }

        /// The dialogs' notifications only (not the "it finished" ones of runs that ended unseen).
        fn dialog_log(&self) -> Vec<String> {
            self.log()
                .into_iter()
                .filter(|l| !l.starts_with("send:Aria · ") && !l.starts_with("dismiss:Aria · "))
                .collect()
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
        harness_seen_by(install_pi, crate::clients::Watchers::default())
    }

    fn harness_seen_by(install_pi: bool, watchers: crate::clients::Watchers) -> Harness {
        let turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().join(".aria"));
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
            schedules: crate::schedule_store::ScheduleStore::open(paths.clone()),
            watchers,
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

    /// A harness whose session `watched` is shown by a focused terminal.
    fn harness_watched(watched: &str) -> Harness {
        harness_seen_by(true, crate::clients::Watchers::watching(watched))
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

    fn is_trace(message: &DockMessage) -> bool {
        matches!(message, DockMessage::ChatTrace { .. })
    }

    fn trace_of(message: &DockMessage) -> Value {
        match message {
            DockMessage::ChatTrace { trace } => trace.clone(),
            _ => Value::Null,
        }
    }

    #[test]
    fn trace_a_traced_run_sends_its_trace() {
        let h = harness();
        h.chat.handle(PageMessage::TraceSet { on: true });
        let first = wait_for(&h, "the answer to on", is_trace);
        assert!(trace_of(&first).is_null(), "no turn yet");
        send(&h, "TRACE");
        let done = wait_for(&h, "a settled trace", |m| trace_of(m)["status"] == "done");
        let tree = trace_of(&done);
        let children = tree["children"].as_array().unwrap();
        let llm = children.iter().find(|s| s["id"] == "llm-1").expect("an llm span");
        assert_eq!(llm["cost"], 0.01);
        assert!(children.iter().any(|s| s["id"] == "tool-c1" && s["status"] == "done"));
        assert_eq!(tree["input"], "TRACE");
    }

    #[test]
    fn trace_untraced_runs_send_no_trace() {
        let h = harness();
        send(&h, "TRACE");
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        std::thread::sleep(Duration::from_millis(400));
        assert!(!drain(&h).iter().any(is_trace));
    }

    #[test]
    fn trace_off_drops_the_trace() {
        let h = harness();
        h.chat.handle(PageMessage::TraceSet { on: true });
        send(&h, "TRACE");
        wait_for(&h, "a settled trace", |m| trace_of(m)["status"] == "done");
        h.chat.handle(PageMessage::TraceSet { on: false });
        wait_for(&h, "the null of off", |m| is_trace(m) && trace_of(m).is_null());
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            !drain(&h).iter().any(|m| is_trace(m) && !trace_of(m).is_null()),
            "nothing after off, not even a late send"
        );
        send(&h, "TRACE");
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        std::thread::sleep(Duration::from_millis(400));
        assert!(!drain(&h).iter().any(is_trace));
    }

    #[test]
    fn the_face_follows_the_run() {
        let h = harness();
        silent_run(&h);
        let session = h.chat.session.clone();
        let face_is = |want: face::Face| {
            let deadline = Instant::now() + Duration::from_secs(5);
            while face::session_now(&session) != Some(want) {
                assert!(Instant::now() < deadline, "the face is {:?}, not {want:?}", face::session_now(&session));
                std::thread::sleep(Duration::from_millis(20));
            }
        };
        face_is(face::Face::Thinking);
        h.chat.test_send_raw("tool_start");
        face_is(face::Face::Working);
        h.chat.test_send_raw("tool_end");
        face_is(face::Face::Thinking);
        h.chat.test_send_raw("settle_done");
        face_is(face::Face::Done);
        h.chat.test_send_raw("start");
        face_is(face::Face::Thinking);
        h.chat.test_send_raw("settle");
        face_is(face::Face::Idle); // an aborted run is not done
        h.chat.test_send_raw("start");
        h.chat.test_send_raw("crash");
        face_is(face::Face::Error);
    }

    #[test]
    fn a_question_makes_the_face_ask_until_the_run_ends() {
        let h = harness();
        silent_run(&h);
        let session = h.chat.session.clone();
        h.chat.test_send_raw("ask_card");
        wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { .. }));
        assert_eq!(face::session_now(&session), Some(face::Face::Asking));
        h.chat.test_send_raw("settle");
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        assert_eq!(face::session_now(&session), Some(face::Face::Idle));
    }

    #[test]
    fn trace_on_mid_turn_waits_for_the_next_turn() {
        let h = harness();
        silent_run(&h);
        h.chat.handle(PageMessage::TraceSet { on: true });
        let first = wait_for(&h, "the answer to on", is_trace);
        assert!(trace_of(&first).is_null(), "a turn already under way is not traced");
        h.chat.test_send_raw("start");
        let next = wait_for(&h, "the next turn's trace", |m| trace_of(m)["status"] == "running");
        assert_eq!(trace_of(&next)["children"].as_array().map(Vec::len), Some(0));
    }

    #[test]
    fn trace_a_restart_mid_turn_fails_the_root() {
        let h = harness();
        h.chat.handle(PageMessage::TraceSet { on: true });
        wait_for(&h, "the answer to on", is_trace);
        send(&h, "CRASH");
        wait_for(&h, "a failed trace", |m| trace_of(m)["status"] == "failed");
    }

    #[test]
    fn trace_changes_are_throttled() {
        let h = harness();
        h.chat.handle(PageMessage::TraceSet { on: true });
        silent_run(&h);
        wait_for(&h, "the turn's first trace", |m| trace_of(m)["status"] == "running");
        std::thread::sleep(Duration::from_millis(300));
        drain(&h);
        for i in 0..20 {
            h.chat.test_trace_event(
                json!({"type": "tool_execution_start", "toolCallId": format!("c{i}"), "toolName": "bash", "args": {}}),
            );
        }
        std::thread::sleep(Duration::from_millis(100));
        let early = drain(&h).into_iter().filter(is_trace).count();
        assert!(early <= 1, "{early} traces within 100 ms");
        let last =
            wait_for(&h, "the throttled trace", |m| trace_of(m)["children"].as_array().is_some_and(|c| c.len() == 20));
        assert!(is_trace(&last));
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
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
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
    fn a_fresh_page_gets_the_running_pis_history_again_and_a_tab_switch_does_not() {
        let h = harness();
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        wait_for(&h, "the first history", |m| event_of(m, "history"));
        // A tab switch on the same page: the conversation is already there.
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        std::thread::sleep(Duration::from_millis(300));
        assert!(drain(&h).iter().all(|m| !event_of(m, "history")), "a tab switch replayed the history");
        // A reloaded page: the Pi is still running, so nothing announces it unless the page asks.
        h.chat.handle(PageMessage::ChatLoad { fresh: true });
        let history = wait_for(&h, "the history again", |m| event_of(m, "history"));
        assert!(
            matches!(history, DockMessage::ChatEvent { ref event } if event["messages"][0]["content"] == json!("earlier"))
        );
        std::thread::sleep(Duration::from_millis(300));
        let rest = drain(&h);
        assert!(rest.iter().all(|m| !event_of(m, "history")), "the history came twice");
        assert!(rest.iter().all(|m| !matches!(m, DockMessage::ChatEngine { state } if state == "starting")));
    }

    #[test]
    fn a_fresh_load_that_starts_pi_announces_it_once() {
        let h = harness();
        h.chat.handle(PageMessage::ChatLoad { fresh: true });
        wait_for(&h, "the history", |m| event_of(m, "history"));
        std::thread::sleep(Duration::from_millis(300));
        assert!(drain(&h).iter().all(|m| !event_of(m, "history")), "a new Pi announced itself twice");
    }

    #[test]
    fn chat_load_with_a_missing_pi_only_reports_the_problem() {
        let h = harness_with(false);
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
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
    fn a_terminal_looking_at_the_session_counts_as_seen() {
        let h = harness_watched("s1");
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.visible.store(false, Ordering::SeqCst);
        h.chat.test_send_raw("ask");
        let dialog = wait_for(&h, "the dialog", |m| matches!(m, DockMessage::ChatDialog { .. }));
        assert!(matches!(dialog, DockMessage::ChatDialog { hidden: false, .. }), "a terminal shows it");
        assert!(h.notifier.log().is_empty(), "no notification while a terminal shows the session");
    }

    #[test]
    fn a_new_thinking_level_reaches_the_page() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.test_send_raw("think-high");
        wait_for(&h, "the new level", |m| matches!(m, DockMessage::ChatState { thinking: Some(t), .. } if t == "high"));
    }

    fn status_of(m: &DockMessage) -> Option<String> {
        match m {
            DockMessage::SessionStatus { status, .. } => Some(status.clone()),
            _ => None,
        }
    }

    #[test]
    fn a_session_says_what_it_is_doing() {
        let h = harness();
        h.visible.store(false, Ordering::SeqCst);
        send(&h, "x");
        wait_for(&h, "working", |m| status_of(m).as_deref() == Some("working"));
        wait_for(&h, "done, nobody looked", |m| status_of(m).as_deref() == Some("done"));
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        wait_for(&h, "idle once looked at", |m| status_of(m).as_deref() == Some("idle"));
        h.chat.test_send_raw("ask");
        wait_for(&h, "waiting for an answer", |m| status_of(m).as_deref() == Some("waiting"));
        h.chat.handle(PageMessage::UiAnswer {
            id: "u1".into(),
            value: Some("Allow".into()),
            confirmed: None,
            cancelled: None,
        });
        wait_for(&h, "idle once answered", |m| status_of(m).as_deref() == Some("idle"));
    }

    #[test]
    fn a_pi_that_dies_mid_run_is_not_working() {
        let h = harness();
        send(&h, "CRASH");
        wait_for(&h, "working", |m| status_of(m).as_deref() == Some("working"));
        wait_for(&h, "not working once Pi is gone", |m| status_of(m).as_deref() == Some("idle"));
    }

    #[test]
    fn a_refused_message_leaves_no_checkpoint_behind() {
        let h = harness();
        let own = h._dir.path().join("checkout-widget");
        std::fs::create_dir_all(&own).unwrap();
        with_own_folder(&h, &own);
        send(&h, "FAIL");
        wait_for(&h, "refused", |m| event_of(m, "command_failed"));
        assert!(h.chat.inner.lock().unwrap().run_checkpoint.is_none(), "the next message gets its own");
        send(&h, "/verify off");
        std::thread::sleep(Duration::from_millis(200));
        h.chat.handle(PageMessage::ChatCheckpoints { id: 9 });
        match wait_for(&h, "the list", |m| matches!(m, DockMessage::Checkpoints { id: 9, .. })) {
            DockMessage::Checkpoints { list, .. } => {
                assert!(
                    list.as_array().unwrap().iter().all(|c| !c["text"].as_str().unwrap_or("").starts_with('/')),
                    "{list}"
                )
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn statuses_follow_the_session_list() {
        let s = sessions_harness();
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        s.sessions.handle(PageMessage::SessionsLoad { id: 4 });
        wait_session(&s, "aa's status after the list", "aa", |m| matches!(m, DockMessage::SessionStatus { .. }));
    }

    #[test]
    fn a_turn_that_changed_files_says_how_many() {
        let h = harness();
        let own = h._dir.path().join("checkout-widget");
        std::fs::create_dir_all(&own).unwrap();
        with_own_folder(&h, &own);
        send(&h, "WRITE");
        match wait_for(&h, "the changes", |m| event_of(m, "changes")) {
            DockMessage::ChatEvent { event } => {
                assert_eq!(
                    (event["files"].clone(), event["added"].clone(), event["removed"].clone()),
                    (json!(1), json!(2), json!(0))
                );
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn undo_puts_files_and_conversation_back() {
        let h = harness();
        let own = h._dir.path().join("checkout-widget");
        std::fs::create_dir_all(&own).unwrap();
        with_own_folder(&h, &own);
        send(&h, "WRITE");
        wait_for(&h, "the changes", |m| event_of(m, "changes"));
        let list = rewind_list(&h, 3);
        assert_eq!(list[0]["text"], json!("WRITE"));
        assert_eq!(list[0]["entry"], json!("e1"));
        assert_eq!((list[0]["files"].clone(), list[0]["added"].clone()), (json!(1), json!(2)));
        let checkpoint = list[0]["checkpoint"].as_str().map(str::to_string);
        assert!(checkpoint.is_some(), "{list}");
        h.chat.handle(PageMessage::ChatUndo { id: 4, entry: "e1".into(), checkpoint, files: true, conversation: true });
        wait_for(&h, "the fork", |m| event_of(m, "fork_seen"));
        wait_for(&h, "saved", |m| matches!(m, DockMessage::Saved { id: 4 }));
        assert!(!own.join("made.txt").exists(), "the file the turn made is gone");
    }

    /// The rewind list `id` answers with.
    fn rewind_list(h: &Harness, id: u64) -> Value {
        h.chat.handle(PageMessage::ChatCheckpoints { id });
        match wait_for(h, "the list", |m| matches!(m, DockMessage::Checkpoints { id: i, .. } if *i == id)) {
            DockMessage::Checkpoints { list, .. } => list,
            _ => unreachable!(),
        }
    }

    #[test]
    fn the_conversation_alone_goes_back_and_files_stay() {
        let h = harness();
        let own = h._dir.path().join("checkout-widget");
        std::fs::create_dir_all(&own).unwrap();
        with_own_folder(&h, &own);
        send(&h, "WRITE");
        wait_for(&h, "the changes", |m| event_of(m, "changes"));
        let checkpoint = rewind_list(&h, 3)[0]["checkpoint"].as_str().map(str::to_string);
        h.chat.handle(PageMessage::ChatUndo {
            id: 4,
            entry: "e1".into(),
            checkpoint,
            files: false,
            conversation: true,
        });
        wait_for(&h, "the fork", |m| event_of(m, "fork_seen"));
        wait_for(&h, "saved", |m| matches!(m, DockMessage::Saved { id: 4 }));
        assert!(own.join("made.txt").exists(), "the files stay as they are");
    }

    #[test]
    fn the_code_alone_goes_back_and_the_conversation_stays() {
        let h = harness();
        let own = h._dir.path().join("checkout-widget");
        std::fs::create_dir_all(&own).unwrap();
        with_own_folder(&h, &own);
        send(&h, "WRITE");
        wait_for(&h, "the changes", |m| event_of(m, "changes"));
        let checkpoint = rewind_list(&h, 3)[0]["checkpoint"].as_str().map(str::to_string);
        h.chat.handle(PageMessage::ChatUndo {
            id: 4,
            entry: "e1".into(),
            checkpoint,
            files: true,
            conversation: false,
        });
        wait_for(&h, "saved", |m| matches!(m, DockMessage::Saved { id: 4 }));
        assert!(!own.join("made.txt").exists(), "the file the turn made is gone");
        assert!(!rewind_list(&h, 5).as_array().unwrap().is_empty(), "the conversation keeps its messages");
    }

    #[test]
    fn the_rewind_list_pairs_messages_with_their_checkpoints() {
        let messages = json!([
            {"entryId": "e1", "text": "yes"},
            {"entryId": "e2", "text": "look at this\n[attached image: a.png at /home/ana/.aria/attachments/a.png]"},
            {"entryId": "e3", "text": "yes"},
        ]);
        let kept = |id: &str, text: &str, files: u32| crate::checkpoints::Checkpoint {
            id: id.into(),
            text: text.into(),
            at: 0,
            files,
            added: files * 2,
            removed: 0,
        };
        // Newest first, as the checkpoints list them; the first "yes" has none (sent before checkpoints were kept).
        let list = rewind_entries(&messages, &[kept("c3", "yes", 1), kept("c2", "look at this", 0)]);
        assert_eq!(
            list,
            json!([
                {"entry": "e1", "text": "yes", "checkpoint": null, "files": 0, "added": 0, "removed": 0},
                {"entry": "e2", "text": "look at this", "checkpoint": "c2", "files": 0, "added": 0, "removed": 0},
                {"entry": "e3", "text": "yes", "checkpoint": "c3", "files": 1, "added": 2, "removed": 0},
            ])
        );
    }

    #[test]
    fn the_projects_folder_itself_gets_no_checkpoints() {
        // It holds the projects (each with its own .git, which a checkpoint cannot see into): too big, and useless.
        let h = harness();
        send(&h, "WRITE");
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        h.chat.handle(PageMessage::ChatCheckpoints { id: 6 });
        match wait_for(&h, "the list", |m| matches!(m, DockMessage::Checkpoints { id: 6, .. })) {
            DockMessage::Checkpoints { list, off, .. } => {
                assert_eq!(list[0]["checkpoint"], json!(null), "the conversation can still go back: {list}");
                assert_eq!(off.as_deref(), Some("projects"), "the client says why there is nothing to undo");
            }
            _ => unreachable!(),
        }
        assert!(!h.paths.root.join("checkpoints").join("s1").exists());
    }

    #[test]
    fn undo_leaves_files_alone_when_the_conversation_cannot_go_back() {
        let h = harness();
        let own = h._dir.path().join("checkout-widget");
        std::fs::create_dir_all(&own).unwrap();
        with_own_folder(&h, &own);
        send(&h, "OTHER");
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        std::fs::write(own.join("mine.txt"), "made by the person").unwrap();
        let checkpoint = h.chat.checkpoints().ok().unwrap().list()[0].id.clone();
        let undo = PageMessage::ChatUndo {
            id: 4,
            entry: "e9".into(),
            checkpoint: Some(checkpoint),
            files: true,
            conversation: true,
        };
        h.chat.handle(undo);
        wait_for(&h, "failed", |m| matches!(m, DockMessage::Error { id: 4, code } if code == "failed"));
        assert!(own.join("mine.txt").exists(), "files are only restored once the conversation can go back too");
    }

    #[test]
    fn undo_is_refused_while_a_message_is_on_its_way() {
        let h = harness();
        ready(&h);
        h.chat.inner.lock().unwrap().pending_prompts.insert("p1".into(), "x".into());
        h.chat.handle(PageMessage::ChatUndo {
            id: 7,
            entry: "x".into(),
            checkpoint: None,
            files: false,
            conversation: true,
        });
        wait_for(&h, "busy", |m| matches!(m, DockMessage::Error { id: 7, code } if code == "busy"));
    }

    #[test]
    fn undo_is_refused_while_running() {
        let h = harness();
        ready(&h);
        h.chat.inner.lock().unwrap().running = true;
        h.chat.handle(PageMessage::ChatUndo {
            id: 5,
            entry: "x".into(),
            checkpoint: None,
            files: false,
            conversation: true,
        });
        wait_for(&h, "busy", |m| matches!(m, DockMessage::Error { id: 5, code } if code == "busy"));
    }

    #[test]
    fn a_turn_that_ends_unseen_notifies_once() {
        let h = harness();
        with_own_folder(&h, h._dir.path());
        let mut list = aria_core::sessions::load(&h.paths);
        list.sessions[0].title = "Coupons".into();
        aria_core::sessions::save(&h.paths, &list).unwrap();
        h.visible.store(false, Ordering::SeqCst);
        send(&h, "TASKS");
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        let wanted = "send:Aria · Coupons · s1:Here are 3 tasks for today".to_string();
        for _ in 0..50 {
            if h.notifier.log().contains(&wanted) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(h.notifier.log().iter().filter(|l| l.starts_with("send:")).collect::<Vec<_>>(), vec![&wanted]);
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        assert!(h.notifier.log().contains(&"dismiss:Aria · Coupons · s1".to_string()), "looking at it takes it away");
    }

    #[test]
    fn a_seen_turn_does_not_notify() {
        let h = harness();
        send(&h, "TASKS");
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        std::thread::sleep(Duration::from_millis(100));
        assert!(h.notifier.log().iter().all(|l| !l.starts_with("send:")), "{:?}", h.notifier.log());
    }

    #[test]
    fn each_session_keeps_one_notification() {
        let h = harness();
        h.visible.store(false, Ordering::SeqCst);
        send(&h, "TASKS");
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        send(&h, "TASKS");
        wait_for(&h, "settled again", |m| event_of(m, "agent_settled"));
        std::thread::sleep(Duration::from_millis(100));
        let log = h.notifier.log();
        let second = log.iter().rposition(|l| l.starts_with("send:")).unwrap();
        assert_eq!(log[second - 1], "dismiss:Aria · s1", "the first goes before the second comes: {log:?}");
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
        assert_eq!(h.notifier.dialog_log(), vec!["send:Aria needs an answer:Allow?"]);
        h.chat.handle(PageMessage::UiAnswer {
            id: "u1".into(),
            value: Some("Allow".into()),
            confirmed: None,
            cancelled: None,
        });
        wait_for(&h, "closed", |m| matches!(m, DockMessage::ChatDialogClosed { id } if id == "u1"));
        wait_for(&h, "the answer reaching Pi", |m| event_of(m, "answered"));
        assert_eq!(h.notifier.dialog_log(), vec!["send:Aria needs an answer:Allow?", "dismiss"]);
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
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
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
    fn kept_notices_go_back_in_place_by_their_time() {
        let messages = json!([
            {"role": "user", "content": "a", "timestamp": 100},
            {"role": "assistant", "content": [], "timestamp": 200},
            {"role": "user", "content": "b", "timestamp": 300},
        ]);
        let notice = |text: &str| json!({"type": "ui_notice", "request": {"method": "notify", "message": text}});
        let merged = with_notices(&messages, &[(250, notice("between")), (900, notice("last"))]);
        let roles: Vec<&str> = merged.as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "assistant", "ariaNotice", "user", "ariaNotice"]);
        assert_eq!(merged[2]["event"]["request"]["message"], json!("between"));
        assert_eq!(merged[2]["timestamp"], json!(250));
    }

    #[test]
    fn a_notice_comes_back_with_the_history() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.test_send_raw("notify");
        wait_for(&h, "the notice", |m| event_of(m, "ui_notice"));
        h.chat.handle(PageMessage::ChatLoad { fresh: true });
        match wait_for(&h, "the history", |m| event_of(m, "history")) {
            DockMessage::ChatEvent { event } => {
                let last = event["messages"].as_array().unwrap().last().cloned().unwrap();
                assert_eq!(last["role"], json!("ariaNotice"), "{event}");
                assert_eq!(last["event"]["request"]["message"], json!("careful"));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn closing_a_session_removes_its_notices() {
        let h = harness();
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.test_send_raw("notify");
        wait_for(&h, "the notice", |m| event_of(m, "ui_notice"));
        let file = h.paths.root.join("notices").join("s1.jsonl");
        assert!(file.exists());
        remove_notices(&h.paths.root, "s1");
        assert!(!file.exists());
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
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
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
            schedules: h.chat.ctx.schedules.clone(),
            watchers: Default::default(),
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
    fn list_files_reads_the_sessions_own_folder() {
        let h = harness();
        std::fs::create_dir_all(&h.workdir).unwrap();
        std::fs::write(h.workdir.join("projects-only.rs"), "x").unwrap();
        let own = h._dir.path().join("checkout-widget");
        std::fs::create_dir_all(own.join("src")).unwrap();
        std::fs::write(own.join("src/cart.rs"), "x").unwrap();
        with_own_folder(&h, &own);
        h.chat.handle(PageMessage::ListFiles { id: 6, query: "".into() });
        match wait_for(&h, "files", |m| matches!(m, DockMessage::Files { id: 6, .. })) {
            DockMessage::Files { files, .. } => assert_eq!(files, vec!["src/cart.rs"]),
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
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
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
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        match wait_for(&h, "chatState after the change", |m| matches!(m, DockMessage::ChatState { .. })) {
            DockMessage::ChatState { workdir, .. } => assert_eq!(workdir, h.workdir.display().to_string()),
            _ => unreachable!(),
        }
    }

    /// The tab `s1` of the harness, saved with its own folder.
    fn with_own_folder(h: &Harness, folder: &std::path::Path) {
        let entry = aria_core::sessions::SessionEntry {
            id: "s1".into(),
            title: "".into(),
            segments: vec![],
            schedule: None,
            workdir: Some(folder.display().to_string()),
        };
        aria_core::sessions::save(&h.paths, &SessionList { sessions: vec![entry], active: Some("s1".into()) }).unwrap();
    }

    #[test]
    fn a_session_with_its_own_folder_starts_pi_there() {
        let h = harness();
        let own = h._dir.path().join("checkout-widget");
        std::fs::create_dir_all(&own).unwrap();
        with_own_folder(&h, &own);
        send(&h, "x");
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        match wait_for(&h, "chatState", |m| matches!(m, DockMessage::ChatState { .. })) {
            DockMessage::ChatState { workdir, .. } => assert_eq!(workdir, own.display().to_string()),
            _ => unreachable!(),
        }
    }

    #[test]
    fn a_missing_session_folder_falls_back_with_a_notice() {
        let h = harness();
        let gone = h._dir.path().join("unmounted");
        with_own_folder(&h, &gone);
        send(&h, "x");
        let notice = wait_for(&h, "the notice", |m| event_of(m, "ui_notice"));
        match notice {
            DockMessage::ChatEvent { event } => {
                assert_eq!(event["request"]["key"], json!("chat.folderMissing"));
                assert_eq!(event["request"]["vars"]["path"], json!(gone.display().to_string()));
            }
            _ => unreachable!(),
        }
        wait_for(&h, "ready", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        assert!(!gone.exists(), "a missing folder is never made");
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        match wait_for(&h, "chatState", |m| matches!(m, DockMessage::ChatState { .. })) {
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
        sessions_harness_with(true)
    }

    fn sessions_harness_with(install_pi: bool) -> SessionsHarness {
        let h = harness_with(install_pi);
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
            schedules: h.chat.ctx.schedules.clone(),
            watchers: Default::default(),
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

    fn dictation_state(m: &DockMessage) -> Option<(&str, Option<&str>)> {
        match m {
            DockMessage::DictationState { state, error, .. } => Some((state.as_str(), error.as_deref())),
            _ => None,
        }
    }

    #[cfg(feature = "gui")]
    #[test]
    fn dictation_answers_the_tab_that_asked() {
        let s = sessions_harness();
        s.sessions.use_dictation(|on| crate::dictation::worker::fake("hola Aria", on));
        s.sessions.handle(in_session("s1", PageMessage::DictationStart));
        wait_session(&s, "recording", "s1", |m| dictation_state(m).is_some_and(|(st, _)| st == "recording"));
        s.sessions.handle(in_session("s1", PageMessage::DictationStop));
        let text = wait_session(&s, "the text", "s1", |m| matches!(m, DockMessage::DictationText { .. }));
        assert_eq!(text, DockMessage::DictationText { text: "hola Aria".into() });
    }

    #[cfg(feature = "gui")]
    #[test]
    fn one_dictation_at_a_time() {
        let s = sessions_harness();
        s.sessions.use_dictation(|on| crate::dictation::worker::fake("hola", on));
        s.sessions.handle(in_session("s1", PageMessage::DictationStart));
        wait_session(&s, "recording", "s1", |m| dictation_state(m).is_some_and(|(st, _)| st == "recording"));
        s.sessions.handle(in_session("s2", PageMessage::DictationStart));
        wait_session(&s, "busy", "s2", |m| dictation_state(m) == Some(("error", Some("busy"))));
    }

    #[cfg(feature = "gui")]
    #[test]
    fn a_closed_tab_cancels_its_dictation() {
        let s = sessions_harness();
        s.sessions.use_dictation(|on| crate::dictation::worker::fake("hola", on));
        s.sessions.handle(in_session("s1", PageMessage::DictationStart));
        wait_session(&s, "recording", "s1", |m| dictation_state(m).is_some_and(|(st, _)| st == "recording"));
        s.sessions.handle(PageMessage::SessionClose { session: "s1".into() });
        wait_session(&s, "idle", "s1", |m| dictation_state(m).is_some_and(|(st, _)| st == "idle"));
        std::thread::sleep(Duration::from_millis(200));
        assert!(!s.messages.try_iter().any(|m| matches!(m, DockMessage::Session { message, .. } if matches!(*message, DockMessage::DictationText { .. }))));
    }

    #[cfg(not(feature = "gui"))]
    #[test]
    fn headless_dictation_answers_unavailable() {
        // A service with no window has no microphone: the tab that asks hears the page's "no microphone" error, and
        // nothing starts (no thread, no model, no download).
        let s = sessions_harness();
        for message in [PageMessage::DictationStart, PageMessage::DictationDownload] {
            s.sessions.handle(in_session("s1", message));
            wait_session(&s, "no microphone", "s1", |m| dictation_state(m) == Some(("error", Some("no-microphone"))));
        }
        s.sessions.handle(in_session("s1", PageMessage::DictationStop));
        s.sessions.handle(in_session("s1", PageMessage::DictationCancel));
        std::thread::sleep(Duration::from_millis(100));
        assert!(!s
            .messages
            .try_iter()
            .any(|m| matches!(m, DockMessage::Session { message, .. } if dictation_state(&message).is_some())));
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
        s.sessions.handle(in_session("fresh-1", PageMessage::ChatLoad { fresh: false }));
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
        assert_eq!(s.h.notifier.dialog_log().len(), 2);
        s.sessions.handle(in_session(
            "aa",
            PageMessage::UiAnswer { id: "u1".into(), value: Some("Allow".into()), confirmed: None, cancelled: None },
        ));
        wait_session(&s, "aa's dialog closing", "aa", |m| matches!(m, DockMessage::ChatDialogClosed { .. }));
        assert!(
            !s.h.notifier.dialog_log().contains(&"dismiss".to_string()),
            "bb still waits for an answer: {:?}",
            s.h.notifier.dialog_log()
        );
        s.sessions.handle(in_session(
            "bb",
            PageMessage::UiAnswer { id: "u1".into(), value: Some("Allow".into()), confirmed: None, cancelled: None },
        ));
        wait_session(&s, "bb's dialog closing", "bb", |m| matches!(m, DockMessage::ChatDialogClosed { .. }));
        assert_eq!(s.h.notifier.dialog_log().last().map(String::as_str), Some("dismiss"));
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
            // a run still going would put the restart off until it settles, after the load below
            wait_session(
                &s,
                "the first run settled",
                id,
                |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "agent_settled"),
            );
        }
        s.h.visible.store(false, Ordering::SeqCst);
        s.sessions.test_send_raw("aa", "ask");
        wait_session(&s, "the dialog", "aa", |m| matches!(m, DockMessage::ChatDialog { .. }));
        s.sessions.handle(PageMessage::EnginesRestart);
        wait_session(&s, "the dialog closing", "aa", |m| matches!(m, DockMessage::ChatDialogClosed { .. }));
        // showing a session again wakes a new Pi, which sees the changed credentials
        s.sessions.handle(in_session("bb", PageMessage::ChatLoad { fresh: false }));
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
    fn settings_changes_do_not_restart_a_session_with_its_own_folder() {
        let s = sessions_harness();
        let own = s.h._dir.path().join("checkout-widget");
        std::fs::create_dir_all(&own).unwrap();
        let entry = aria_core::sessions::SessionEntry {
            id: "aa".into(),
            title: "".into(),
            segments: vec![],
            schedule: None,
            workdir: Some(own.display().to_string()),
        };
        aria_core::sessions::save(&s.h.paths, &SessionList { sessions: vec![entry], active: Some("aa".into()) })
            .unwrap();
        s.sessions
            .handle(in_session("aa", PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] }));
        wait_session(&s, "ready", "aa", |m| matches!(m, DockMessage::ChatEngine { state } if state == "ready"));
        let mut settings = settings::load(&s.h.paths).unwrap().unwrap();
        let elsewhere = s.h._dir.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        settings.sandbox.projects_dir = elsewhere.display().to_string();
        settings::save(&s.h.paths, &settings).unwrap();
        s.sessions.settings_changed();
        std::thread::sleep(Duration::from_millis(300));
        let stopped = s
            .messages
            .try_iter()
            .any(|m| matches!(&m, DockMessage::Session { message, .. } if matches!(**message, DockMessage::ChatEngine { ref state } if state == "stopped")));
        assert!(!stopped, "a session with its own folder keeps working where it was opened");
    }

    #[test]
    fn a_page_save_keeps_a_session_folder() {
        let s = sessions_harness();
        let entry = |workdir: Option<&str>| aria_core::sessions::SessionEntry {
            id: "aa".into(),
            title: "Coupons".into(),
            segments: vec![],
            schedule: None,
            workdir: workdir.map(str::to_string),
        };
        let folder = "/home/ana/Projects/checkout-widget";
        aria_core::sessions::save(&s.h.paths, &SessionList { sessions: vec![entry(Some(folder))], active: None })
            .unwrap();
        s.sessions.handle(PageMessage::SessionsSave { sessions: vec![entry(None)], active: Some("aa".into()) });
        assert_eq!(aria_core::sessions::load(&s.h.paths).sessions[0].workdir.as_deref(), Some(folder));
    }

    fn regular(id: &str) -> aria_core::sessions::SessionEntry {
        aria_core::sessions::SessionEntry {
            id: id.into(),
            title: "".into(),
            segments: vec![],
            schedule: None,
            workdir: None,
        }
    }

    #[test]
    fn a_new_session_is_saved_and_announced() {
        let s = sessions_harness();
        aria_core::sessions::save(
            &s.h.paths,
            &SessionList { sessions: vec![regular("aa")], active: Some("aa".into()) },
        )
        .unwrap();
        s.sessions
            .handle(PageMessage::SessionNew { id: 7, workdir: Some("/home/ana/Projects/checkout-widget".into()) });
        let added = loop {
            match s.messages.recv_timeout(Duration::from_secs(5)).expect("sessionAdded") {
                DockMessage::SessionAdded { id, session } => break (id, session),
                _ => {}
            }
        };
        assert_eq!(added.0, Some(7));
        assert_eq!(added.1.workdir.as_deref(), Some("/home/ana/Projects/checkout-widget"));
        let list = aria_core::sessions::load(&s.h.paths);
        assert_eq!(list.sessions.len(), 2);
        assert_eq!(list.sessions[1].id, added.1.id);
        assert_eq!(list.active.as_deref(), Some("aa"));
    }

    #[test]
    fn a_ninth_session_is_refused() {
        let s = sessions_harness();
        let full: Vec<_> = (1..=8).map(|n| regular(&format!("t{n}"))).collect();
        aria_core::sessions::save(&s.h.paths, &SessionList { sessions: full, active: None }).unwrap();
        s.sessions.handle(PageMessage::SessionNew { id: 8, workdir: None });
        let refused = s
            .messages
            .try_iter()
            .any(|m| matches!(m, DockMessage::Error { id: 8, ref code } if code == "too-many-sessions"));
        assert!(refused);
    }

    #[test]
    fn closing_the_last_session_opens_a_new_one() {
        let s = sessions_harness();
        aria_core::sessions::save(
            &s.h.paths,
            &SessionList { sessions: vec![regular("aa")], active: Some("aa".into()) },
        )
        .unwrap();
        s.sessions.handle(PageMessage::SessionClose { session: "aa".into() });
        let got: Vec<DockMessage> = s.messages.try_iter().collect();
        let removed = got.iter().position(|m| matches!(m, DockMessage::SessionRemoved { session } if session == "aa"));
        let added = got.iter().position(|m| matches!(m, DockMessage::SessionAdded { id: None, .. }));
        assert!(removed.is_some() && added.is_some() && removed < added, "removed, then a fresh one: {got:?}");
        let list = aria_core::sessions::load(&s.h.paths);
        assert_eq!(list.sessions.len(), 1);
        assert_ne!(list.sessions[0].id, "aa");
        assert_eq!(list.active.as_deref(), Some(list.sessions[0].id.as_str()));
    }

    #[test]
    fn closing_one_of_two_sessions_removes_it_from_the_list() {
        let s = sessions_harness();
        aria_core::sessions::save(
            &s.h.paths,
            &SessionList { sessions: vec![regular("aa"), regular("bb")], active: Some("aa".into()) },
        )
        .unwrap();
        s.sessions.handle(PageMessage::SessionClose { session: "aa".into() });
        let got: Vec<DockMessage> = s.messages.try_iter().collect();
        assert!(got.iter().any(|m| matches!(m, DockMessage::SessionRemoved { session } if session == "aa")));
        assert!(!got.iter().any(|m| matches!(m, DockMessage::SessionAdded { .. })));
        let list = aria_core::sessions::load(&s.h.paths);
        assert_eq!(list.sessions.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["bb"]);
    }

    #[test]
    fn a_save_is_announced_as_changed() {
        let s = sessions_harness();
        s.sessions.handle(PageMessage::SessionsSave { sessions: vec![regular("aa")], active: Some("aa".into()) });
        assert!(s.messages.try_iter().any(|m| matches!(m, DockMessage::SessionsChanged)));
    }

    #[test]
    fn a_bang_command_runs_in_pi_as_its_bash() {
        let h = harness();
        h.chat.handle(PageMessage::ChatBash { id: 5, command: "pwd".into() });
        let (ok, data, _) = pi_reply(&h, 5);
        assert!(ok);
        assert_eq!(data["output"], json!("ran pwd"));
    }

    #[test]
    fn the_page_cannot_run_a_command() {
        let h = harness();
        h.chat.handle(PageMessage::PiCommand { id: 6, command: "bash".into(), args: json!({"command": "id"}) });
        assert_eq!(pi_reply(&h, 6), (false, Value::Null, Some("command-unknown".into())));
    }

    #[test]
    fn closing_a_session_removes_its_checkpoints() {
        let s = sessions_harness();
        let gone = s.h.paths.root.join("checkpoints").join("aa");
        std::fs::create_dir_all(&gone).unwrap();
        s.sessions.handle(PageMessage::SessionClose { session: "aa".into() });
        assert!(!gone.exists());
        let stale = s.h.paths.root.join("checkpoints").join("zz");
        std::fs::create_dir_all(&stale).unwrap();
        s.sessions.forget_gone_sessions();
        assert!(!stale.exists(), "a session that is not open any more keeps no checkpoints");
    }

    #[test]
    fn finished_notifications_never_touch_another_sessions() {
        let s = sessions_harness();
        s.h.visible.store(false, Ordering::SeqCst);
        let entry = |id: &str, title: &str| aria_core::sessions::SessionEntry {
            id: id.into(),
            title: title.into(),
            segments: vec![],
            schedule: None,
            workdir: None,
        };
        aria_core::sessions::save(
            &s.h.paths,
            &SessionList { sessions: vec![entry("aa", "Coupons"), entry("bb", "")], active: None },
        )
        .unwrap();
        for id in ["aa", "bb"] {
            s.sessions.handle(in_session(
                id,
                PageMessage::ChatSend { text: "TASKS".into(), behavior: None, attachments: vec![] },
            ));
            wait_session(
                &s,
                "settled",
                id,
                |m| matches!(m, DockMessage::ChatEvent { event } if event["type"] == "agent_settled"),
            );
        }
        std::thread::sleep(Duration::from_millis(100));
        let titles: Vec<String> =
            s.h.notifier
                .log()
                .iter()
                .filter_map(|l| l.strip_prefix("send:").map(|r| r.split(':').next().unwrap().to_string()))
                .collect();
        assert_eq!(titles, ["Aria · Coupons · aa", "Aria · bb"], "each one its own, never a prefix of another");
        // A dialog's notification still goes away when it is answered.
        s.sessions.test_send_raw("aa", "ask");
        wait_session(&s, "aa's dialog", "aa", |m| matches!(m, DockMessage::ChatDialog { .. }));
        s.sessions.handle(in_session(
            "aa",
            PageMessage::UiAnswer { id: "u1".into(), value: Some("Allow".into()), confirmed: None, cancelled: None },
        ));
        wait_session(&s, "closed", "aa", |m| matches!(m, DockMessage::ChatDialogClosed { .. }));
        assert_eq!(s.h.notifier.log().last().map(String::as_str), Some("dismiss"), "{:?}", s.h.notifier.log());
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
        let entries = vec![aria_core::sessions::SessionEntry {
            id: "aa".into(),
            title: "Fix the build".into(),
            segments: vec![],
            schedule: None,
            workdir: None,
        }];
        s.sessions.handle(PageMessage::SessionsSave { sessions: entries.clone(), active: Some("aa".into()) });
        assert!(matches!(s.messages.recv_timeout(Duration::from_secs(5)).unwrap(), DockMessage::SessionsChanged));
        s.sessions.handle(PageMessage::SessionsLoad { id: 9 });
        match s.messages.recv_timeout(Duration::from_secs(5)).unwrap() {
            DockMessage::Sessions { id: 9, sessions, active } => {
                assert_eq!(sessions, entries);
                assert_eq!(active.as_deref(), Some("aa"));
            }
            other => panic!("{other:?}"),
        }
    }

    /// A task in the harness's store: every day at 07:30, Notion read-only.
    fn scheduled_task(s: &SessionsHarness) -> aria_core::schedules::Task {
        std::fs::write(s.h.paths.mcp(), r#"{"mcpServers":{"notion":{"url":"https://n"}}}"#).unwrap();
        s.sessions
            .ctx
            .schedules
            .create(&json!({"title": "Notion tasks", "prompt": "Give me my TASKS", "at": "07:30", "connectors": {"notion": "read"}}), 0)
            .unwrap()
    }

    fn engine_state(m: &DockMessage, state: &str) -> bool {
        matches!(m, DockMessage::ChatEngine { state: got } if got == state)
    }

    #[test]
    fn a_due_task_creates_its_tab_and_sends_its_prompt() {
        let s = sessions_harness();
        let task = scheduled_task(&s);
        s.sessions.start_scheduled(&task, now_ms());
        let tab = loop {
            if let DockMessage::ScheduleTab { id, title, schedule } =
                s.messages.recv_timeout(Duration::from_secs(5)).unwrap()
            {
                break (id, title, schedule);
            }
        };
        assert_eq!(tab, (task.id.clone(), "Notion tasks".to_string(), task.id.clone()));
        let run = wait_session(&s, "the scheduled run", &task.id, |m| event_of(m, "scheduled_run"));
        match run {
            DockMessage::ChatEvent { event } => {
                assert_eq!(event["text"], "Give me my TASKS");
                assert_eq!(event["late"], false);
            }
            _ => unreachable!(),
        }
        wait_session(&s, "its run", &task.id, settled);
        let entry = aria_core::sessions::load(&s.h.paths).sessions.into_iter().find(|e| e.id == task.id).unwrap();
        assert_eq!(entry.schedule.as_deref(), Some(task.id.as_str()));
    }

    #[test]
    fn the_settled_run_is_recorded_and_notified() {
        let s = sessions_harness();
        s.h.visible.store(false, Ordering::SeqCst);
        let task = scheduled_task(&s);
        s.sessions.start_scheduled(&task, now_ms());
        wait_session(&s, "its run", &task.id, settled);
        let recorded = || s.sessions.ctx.schedules.get(&task.id).and_then(|t| t.last_result);
        for _ in 0..100 {
            if recorded().is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let result = recorded().expect("the run is recorded");
        assert!(result.ok);
        assert_eq!(result.summary, "Here are 3 tasks for today");
        assert!(
            s.h.notifier.log().contains(&"send:Aria · Notion tasks:Here are 3 tasks for today".to_string()),
            "{:?}",
            s.h.notifier.log()
        );
    }

    #[test]
    fn a_run_pi_refuses_is_recorded_as_failed_and_notified() {
        let s = sessions_harness();
        s.h.visible.store(false, Ordering::SeqCst);
        std::fs::write(s.h.paths.mcp(), r#"{"mcpServers":{}}"#).unwrap();
        let task =
            s.sessions.ctx.schedules.create(&json!({"title": "Refused", "prompt": "FAIL", "at": "07:30"}), 0).unwrap();
        s.sessions.start_scheduled(&task, now_ms());
        wait_session(&s, "the refusal", &task.id, |m| event_of(m, "command_failed"));
        let recorded = || s.sessions.ctx.schedules.get(&task.id).and_then(|t| t.last_result);
        for _ in 0..100 {
            if recorded().is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let result = recorded().expect("a refused run is recorded");
        assert!(!result.ok);
        assert!(result.summary.contains("compaction"), "{}", result.summary);
        assert!(
            s.h.notifier.log().iter().any(|line| line.starts_with("send:Aria · Refused:It did not finish")),
            "{:?}",
            s.h.notifier.log()
        );
    }

    #[test]
    fn a_writing_task_waits_and_the_page_is_asked_to_show_it() {
        let s = sessions_harness();
        std::fs::write(s.h.paths.mcp(), r#"{"mcpServers":{"notion":{"url":"https://n"}}}"#).unwrap();
        let task = s
            .sessions
            .ctx
            .schedules
            .create(&json!({"title": "Write", "prompt": "p", "at": "07:30", "connectors": {"notion": "allow"}}), 0)
            .unwrap();
        let shown = loop {
            if let DockMessage::ScheduleAwaiting { schedule } = s.messages.recv_timeout(Duration::from_secs(5)).unwrap()
            {
                break schedule;
            }
        };
        assert_eq!(shown, task.id);
        s.sessions.handle(PageMessage::ScheduleUpdate { id: 3, schedule: task.id.clone(), enabled: true });
        let still = s.sessions.ctx.schedules.get(&task.id).unwrap();
        assert!(still.awaiting && !still.enabled, "a plain switch does not approve it");
        let answered = loop {
            if let DockMessage::Schedules { id, tasks } = s.messages.recv_timeout(Duration::from_secs(5)).unwrap() {
                break (id, tasks);
            }
        };
        assert_eq!(answered.0, 3, "the failure is answered with the fresh list");
        assert_eq!(answered.1[0]["awaiting"], true);
        s.sessions.handle(PageMessage::ScheduleApprove { id: 4, schedule: task.id.clone() });
        let approved = s.sessions.ctx.schedules.get(&task.id).unwrap();
        assert!(approved.enabled && !approved.awaiting, "the page's approval does");
    }

    fn recorded(s: &SessionsHarness, id: &str) -> aria_core::schedules::RunResult {
        for _ in 0..150 {
            if let Some(result) = s.sessions.ctx.schedules.get(id).and_then(|t| t.last_result) {
                return result;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("the run of {id} was never recorded")
    }

    #[test]
    fn a_run_whose_pi_cannot_start_is_recorded_as_failed() {
        let s = sessions_harness_with(false);
        std::fs::write(s.h.paths.mcp(), r#"{"mcpServers":{}}"#).unwrap();
        let task =
            s.sessions.ctx.schedules.create(&json!({"title": "No Pi", "prompt": "p", "at": "07:30"}), 0).unwrap();
        s.sessions.start_scheduled(&task, 0);
        let result = recorded(&s, &task.id);
        assert!(!result.ok && result.summary.contains("could not start"), "{}", result.summary);
    }

    #[test]
    fn a_run_whose_pi_dies_is_recorded_as_failed_and_the_next_turn_is_not_the_tasks() {
        let s = sessions_harness();
        std::fs::write(s.h.paths.mcp(), r#"{"mcpServers":{}}"#).unwrap();
        let task = s
            .sessions
            .ctx
            .schedules
            .create(&json!({"title": "Crash", "prompt": "CRASH now", "at": "07:30"}), 0)
            .unwrap();
        s.sessions.start_scheduled(&task, 0);
        let result = recorded(&s, &task.id);
        assert!(!result.ok && result.summary.contains("stopped"), "{}", result.summary);
        // The person's next message in that tab is theirs, not the task's.
        s.sessions.handle(in_session(
            &task.id,
            PageMessage::ChatSend { text: "Give me my TASKS".into(), behavior: None, attachments: vec![] },
        ));
        wait_session(&s, "the person's run", &task.id, settled);
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !s.sessions.ctx.schedules.get(&task.id).unwrap().last_result.unwrap().ok,
            "still the task's own failure"
        );
    }

    #[test]
    fn a_waiting_run_reads_its_task_again_when_its_tab_is_free() {
        let s = sessions_harness();
        let task = scheduled_task(&s);
        s.sessions.handle(in_session(
            &task.id,
            PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] },
        ));
        wait_session(&s, "the first run settled", &task.id, settled);
        s.sessions.test_send_raw(&task.id, "start"); // the person's run, still going
        wait_session(&s, "the run", &task.id, |m| event_of(m, "agent_start"));
        s.sessions.start_scheduled(&task, now_ms() - 10 * 60_000);
        s.sessions.ctx.schedules.update(&json!({"id": task.id, "enabled": false})).unwrap(); // paused while it waits
        s.sessions.test_send_raw(&task.id, "settle");
        wait_session(&s, "settled", &task.id, settled);
        s.sessions.flush_scheduled();
        std::thread::sleep(Duration::from_millis(200));
        let ran = std::iter::from_fn(|| s.messages.try_recv().ok())
            .any(|m| matches!(m, DockMessage::Session { ref message, .. } if event_of(message, "scheduled_run")));
        assert!(!ran, "a task paused while it waited does not run");
    }

    #[test]
    fn a_scheduled_tabs_background_agents_get_its_grants() {
        let s = sessions_harness();
        let task = scheduled_task(&s);
        let chat = s.sessions.chat(&task.id).unwrap();
        chat.inner.lock().unwrap().model = Some("opencode-go/glm".into());
        let (_, _, env, _, _) = chat.agent_launch("session").expect("a launch");
        let grants = env.iter().find(|(k, _)| k == "ARIA_GRANTS").map(|(_, v)| v.clone()).expect("ARIA_GRANTS");
        let grants: Value = serde_json::from_str(&grants).unwrap();
        assert_eq!(grants["connectors"], json!({"notion": "read"}));
        let regular = s.sessions.chat("aa").unwrap();
        regular.inner.lock().unwrap().model = Some("opencode-go/glm".into());
        let (_, _, env, _, _) = regular.agent_launch("session").unwrap();
        assert!(!env.iter().any(|(k, _)| k == "ARIA_GRANTS"), "a regular tab's agents keep the person's settings");
    }

    #[test]
    fn a_task_due_mid_run_waits_for_the_run() {
        let s = sessions_harness();
        let task = scheduled_task(&s);
        s.sessions.handle(in_session(
            &task.id,
            PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] },
        ));
        wait_session(&s, "the first run settled", &task.id, settled);
        s.sessions.test_send_raw(&task.id, "start"); // the person's run, still going
        wait_session(&s, "the run", &task.id, |m| event_of(m, "agent_start"));
        s.sessions.start_scheduled(&task, now_ms());
        std::thread::sleep(Duration::from_millis(200));
        let early = std::iter::from_fn(|| s.messages.try_recv().ok())
            .any(|m| matches!(m, DockMessage::Session { ref message, .. } if event_of(message, "scheduled_run")));
        assert!(!early, "it waits while the tab is busy");
        s.sessions.test_send_raw(&task.id, "settle");
        wait_session(&s, "settled", &task.id, settled);
        s.sessions.flush_scheduled();
        wait_session(&s, "the scheduled run, after", &task.id, |m| event_of(m, "scheduled_run"));
    }

    #[test]
    fn deleting_a_running_task_stops_it_and_removes_its_tab() {
        let s = sessions_harness();
        let task = scheduled_task(&s);
        s.sessions.start_scheduled(&task, now_ms());
        wait_session(&s, "its run", &task.id, settled);
        s.sessions.ctx.schedules.delete(&task.id).unwrap();
        let removed = loop {
            if let DockMessage::ScheduleRemoved { id } = s.messages.recv_timeout(Duration::from_secs(5)).unwrap() {
                break id;
            }
        };
        assert_eq!(removed, task.id);
        assert!(!aria_core::sessions::load(&s.h.paths).sessions.iter().any(|e| e.id == task.id));
        assert!(s.sessions.chat(&task.id).is_none(), "the tab is closed for good");
    }

    #[test]
    fn the_tab_pi_gets_its_grants() {
        use base64::Engine as _;
        let s = sessions_harness();
        let task = scheduled_task(&s);
        s.sessions.start_scheduled(&task, now_ms());
        wait_session(&s, "its run", &task.id, settled);
        s.sessions.test_send_raw(&task.id, "env_grants");
        let env = wait_session(&s, "the env", &task.id, |m| event_of(m, "env"));
        let DockMessage::ChatEvent { event } = env else { unreachable!() };
        let raw = base64::engine::general_purpose::STANDARD.decode(event["grants"].as_str().unwrap()).unwrap();
        let grants: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(grants["connectors"], json!({"notion": "read"}));
        assert_eq!(grants["sandbox"], json!({"askToLeave": false, "extraWritable": []}));
        assert_eq!(grants["noAsk"], true, "nothing is ever asked in a scheduled tab");
        let _ = engine_state; // used by the other scheduler tests
    }

    #[test]
    fn a_page_save_keeps_the_scheduled_tab_of_a_live_task() {
        let s = sessions_harness();
        let task = scheduled_task(&s);
        aria_core::sessions::add_scheduled(&s.h.paths, &task.id, "Notion tasks", &task.id).unwrap();
        let regular = aria_core::sessions::SessionEntry {
            id: "aa".into(),
            title: "Fix".into(),
            segments: vec![],
            schedule: None,
            workdir: None,
        };
        s.sessions.handle(PageMessage::SessionsSave { sessions: vec![regular], active: Some("aa".into()) });
        let ids: Vec<String> = aria_core::sessions::load(&s.h.paths).sessions.into_iter().map(|e| e.id).collect();
        assert_eq!(ids, vec!["aa".to_string(), task.id.clone()]);
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
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
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
    fn a_watchdog_stop_keeps_what_the_person_queued() {
        let h = harness();
        silent_run(&h);
        assert!(h.chat.check_stalled(Instant::now() + 2 * LIMIT, LIMIT));
        wait_for(&h, "the abort reaching Pi", |m| event_of(m, "abort_seen"));
        // Pi reads its stdin in order: a `clear_queue` sent before the abort would have answered (and emptied the
        // queue, a `queue_update`) before `abort_seen`. Pi kept its queue, so the queued follow-ups still run.
        let before = drain(&h);
        assert!(!before.iter().any(|m| event_of(m, "queue_update")), "the queue was not cleared: {before:?}");
        // The page's own stop still empties it, as the person asked.
        h.chat.handle(PageMessage::ChatAbort);
        wait_for(&h, "the page's stop clearing the queue", |m| event_of(m, "queue_update"));
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
    fn overlapping_tools_pause_it_until_the_last_ends() {
        let h = harness();
        silent_run(&h);
        h.chat.test_send_raw("tool_start");
        wait_for(&h, "the first tool", |m| event_of(m, "tool_execution_start"));
        h.chat.test_send_raw("tool_start_b");
        wait_for(&h, "the second tool", |m| event_of(m, "tool_execution_start"));
        let hour = Duration::from_secs(3600);
        h.chat.test_send_raw("tool_end");
        wait_for(&h, "the first tool's end", |m| event_of(m, "tool_execution_end"));
        assert!(!h.chat.check_stalled(Instant::now() + hour, LIMIT), "one tool still runs");
        h.chat.test_send_raw("tool_end_b");
        wait_for(&h, "the second tool's end", |m| event_of(m, "tool_execution_end"));
        assert!(!h.chat.check_stalled(Instant::now(), LIMIT), "the clock starts again when the last tool ends");
        assert!(h.chat.check_stalled(Instant::now() + 2 * LIMIT, LIMIT), "then the silence counts");
    }

    #[test]
    fn a_compaction_with_no_run_shows_in_the_state() {
        let h = harness();
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        wait_for(&h, "the first state", |m| matches!(m, DockMessage::ChatState { .. }));
        h.chat.test_send_raw("compaction_start");
        wait_for(&h, "the compaction", |m| event_of(m, "compaction_start"));
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        // The first load can send a second state (once Pi is up) that waits in the backlog: that one is from before the
        // compaction, so the test waits for the state that says it compacts rather than taking whichever comes first.
        wait_for(&h, "a state that says it compacts", |m| {
            matches!(m, DockMessage::ChatState { compacting: true, running: false, .. })
        });
        h.chat.test_send_raw("compaction_end");
        wait_for(&h, "the compaction's end", |m| event_of(m, "compaction_end"));
        // A state already on its way when the compaction ended can still say it is on: the one asked a moment later is over.
        std::thread::sleep(Duration::from_millis(100));
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        wait_for(&h, "a state without the compaction", |m| {
            matches!(m, DockMessage::ChatState { compacting: false, .. })
        });
    }

    #[test]
    fn a_compaction_pauses_the_watchdog() {
        let h = harness();
        silent_run(&h);
        h.chat.test_send_raw("compaction_start");
        wait_for(&h, "the compaction", |m| event_of(m, "compaction_start"));
        let hour = Duration::from_secs(3600);
        assert!(!h.chat.check_stalled(Instant::now() + hour, LIMIT), "a long compaction is not a hang");
        std::thread::sleep(Duration::from_millis(200));
        assert!(!drain(&h).iter().any(|m| event_of(m, "abort_seen") || is_watchdog_notice(m)));
        h.chat.test_send_raw("compaction_end");
        wait_for(&h, "the compaction's end", |m| event_of(m, "compaction_end"));
        assert!(!h.chat.check_stalled(Instant::now(), LIMIT), "the clock starts again when it ends");
        assert!(h.chat.check_stalled(Instant::now() + 2 * LIMIT, LIMIT), "then the silence counts");
        wait_for(&h, "the abort reaching Pi", |m| event_of(m, "abort_seen"));
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
        let limit = Arc::new(Mutex::new(Duration::from_secs(600)));
        assert_eq!(Sessions::read_limit(&h.paths, &limit), Duration::ZERO);
        settings.models.turn_timeout_minutes = 15;
        settings::save(&h.paths, &settings).unwrap();
        assert_eq!(Sessions::read_limit(&h.paths, &limit), Duration::from_secs(900));
        settings.models.turn_timeout_minutes = 0;
        settings::save(&h.paths, &settings).unwrap();
        assert_eq!(Sessions::read_limit(&h.paths, &limit), Duration::ZERO);
        std::fs::write(h.paths.settings(), "{ half written").unwrap();
        assert_eq!(Sessions::read_limit(&h.paths, &limit), Duration::ZERO, "an unreadable file keeps the last limit");
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
        // No hidden folder (the runtime directory is only the /tmp fallback): a boxed command could see the run dir.
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
            schedules: h.chat.ctx.schedules.clone(),
            watchers: Default::default(),
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

        assert!(hidden_from_the_box(
            std::path::Path::new("/run/user/1000/aria"),
            Some(std::path::Path::new("/run/user/1000"))
        ));
        assert!(!hidden_from_the_box(std::path::Path::new("/tmp/aria"), Some(std::path::Path::new("/run/user/1000"))));
        assert!(!hidden_from_the_box(std::path::Path::new("/tmp/aria"), None));
        assert!(
            !hidden_from_the_box(std::path::Path::new("aria"), Some(std::path::Path::new(""))),
            "a relative folder hides nothing"
        );
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
        assert!(
            env.iter().any(|(k, v)| k == "PONYTAIL_QUIET_STARTUP" && v == "1"),
            "subagents stay quiet too: {env:?}"
        );
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
        s.sessions.handle(in_session("bb", PageMessage::ChatLoad { fresh: false }));
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
        let entries = vec![aria_core::sessions::SessionEntry {
            id: "aa".into(),
            title: "Kept".into(),
            segments: vec![],
            schedule: None,
            workdir: None,
        }];
        aria_core::sessions::save(&s.h.paths, &SessionList { sessions: entries, active: None }).unwrap();
        s.sessions.forget_gone_sessions();
        assert_eq!(jobs.events("aa").len(), 1, "an open tab keeps its event for its Pi");
        assert!(jobs.events("gone").is_empty(), "nobody will ever ack a gone tab's event");
    }

    #[test]
    fn sessions_with_background_work_are_told_an_update_waits() {
        let s = sessions_harness();
        for id in ["aa", "bb"] {
            s.sessions.handle(in_session(id, PageMessage::ChatLoad { fresh: false }));
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

    // ---- session rotation: a tab's chain of segments ------------------------------------------------

    /// The harness's tab `s1` in `sessions.json`, with these segments.
    fn save_tab(h: &Harness, segments: &[&str]) {
        let entry = aria_core::sessions::SessionEntry {
            id: "s1".into(),
            title: "Tab".into(),
            segments: segments.iter().map(|s| s.to_string()).collect(),
            schedule: None,
            workdir: None,
        };
        sessions::save(&h.paths, &SessionList { sessions: vec![entry], active: Some("s1".into()) }).unwrap();
    }

    fn stored_segments(h: &Harness) -> Vec<String> {
        sessions::load(&h.paths).sessions.first().map(sessions::segments_of).unwrap_or_default()
    }

    fn session_id_arg(h: &Harness) -> String {
        let args = h.chat.test_spawn_args();
        let at = args.iter().position(|a| a == "--session-id").expect("--session-id is passed");
        args[at + 1].clone()
    }

    /// A session file of `bytes` (sparse) where the fake Pi says its session `id` lives.
    fn segment_file_of_size(h: &Harness, id: &str, bytes: u64) {
        let dir = agent::sessions_dir(&h.paths);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::File::create(dir.join(format!("{id}.jsonl"))).unwrap().set_len(bytes).unwrap();
    }

    #[test]
    fn after_a_compaction_the_settled_tab_rotates_once_and_a_later_pi_starts_on_the_new_segment() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        h.chat.test_send_raw("start");
        wait_for(&h, "the run", |m| event_of(m, "agent_start"));
        h.chat.test_send_raw("compaction_end");
        wait_for(&h, "the compaction", |m| event_of(m, "compaction_end"));
        std::thread::sleep(Duration::from_millis(200));
        assert!(drain(&h).iter().all(|m| !event_of(m, "rotate_seen")), "never during a run");

        h.chat.test_send_raw("settle");
        wait_for(&h, "the rotation", |m| event_of(m, "rotate_seen"));
        eventually("the new segment is recorded", || stored_segments(&h) == ["s1", "rotated-1"]);

        // Another run settles with no compaction: nothing more.
        h.chat.test_send_raw("start");
        h.chat.test_send_raw("settle");
        wait_for(&h, "the second settle", |m| event_of(m, "agent_settled"));
        std::thread::sleep(Duration::from_millis(300));
        assert!(drain(&h).iter().all(|m| !event_of(m, "rotate_seen")), "one rotation per compaction");
        assert_eq!(stored_segments(&h), ["s1", "rotated-1"]);

        h.chat.shutdown();
        ready(&h);
        assert_eq!(session_id_arg(&h), "rotated-1", "Pi starts on the tab's current segment");
    }

    #[test]
    fn a_compaction_while_the_tab_is_quiet_rotates_at_once() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        h.chat.handle(PageMessage::PiCommand { id: 1, command: "compact".into(), args: Value::Null });
        wait_for(&h, "the rotation", |m| event_of(m, "rotate_seen"));
        eventually("the new segment is recorded", || stored_segments(&h) == ["s1", "rotated-1"]);
    }

    #[test]
    fn a_prompt_sent_during_a_rotation_waits_for_it_and_is_answered() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        h.chat.test_send_raw("compaction_end");
        wait_for(&h, "the rotation", |m| event_of(m, "rotate_seen"));
        send(&h, "meanwhile");
        h.chat.handle(PageMessage::PiCommand { id: 7, command: "get_session_stats".into(), args: Value::Null });
        wait_for(&h, "the held prompt's run", |m| event_of(m, "agent_settled"));
        assert!(pi_reply(&h, 7).0, "the held command is answered too");
        assert_eq!(stored_segments(&h), ["s1", "rotated-1"], "recorded before the held prompt ran");
        std::thread::sleep(Duration::from_millis(200));
        assert!(drain(&h).iter().all(|m| !event_of(m, "prompt_during_rotation")), "nothing reached Pi mid-rotation");
    }

    #[test]
    fn a_restart_during_a_rotation_waits_and_the_next_pi_starts_on_the_new_segment() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        h.chat.test_send_raw("compaction_end");
        wait_for(&h, "the rotation", |m| event_of(m, "rotate_seen"));
        assert!(h.chat.busy(), "an update waits for a rotation");
        h.chat.restart_when_idle();
        wait_for(&h, "the held restart", |m| matches!(m, DockMessage::ChatEngine { state } if state == "stopped"));
        assert_eq!(stored_segments(&h), ["s1", "rotated-1"]);
        ready(&h);
        assert_eq!(session_id_arg(&h), "rotated-1");
    }

    #[test]
    fn closing_a_tab_during_a_rotation_drops_what_it_held_and_starts_no_pi() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        drain(&h); // the first run's events
        h.chat.test_send_raw("compaction_end");
        wait_for(&h, "the rotation", |m| event_of(m, "rotate_seen"));
        send(&h, "meanwhile");
        h.chat.close_for_good();
        std::thread::sleep(Duration::from_millis(700));
        let after = drain(&h);
        assert!(
            after.iter().all(|m| !matches!(m, DockMessage::ChatEngine { state } if state == "starting")),
            "no Pi starts for a closed tab"
        );
        assert!(after.iter().all(|m| !event_of(m, "agent_start")), "the held prompt never reached a Pi");
        assert!(h.chat.inner.lock().unwrap().held.is_empty());
    }

    #[test]
    fn stop_during_a_rotation_gives_the_held_prompts_back_and_nothing_starts_after() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        drain(&h); // the first run's events
        h.chat.test_send_raw("compaction_end");
        wait_for(&h, "the rotation", |m| event_of(m, "rotate_seen"));
        send(&h, "meanwhile");
        h.chat.handle(PageMessage::ChatAbort);
        let given = wait_for(&h, "the prompt given back", |m| event_of(m, "command_failed"));
        assert!(matches!(given, DockMessage::ChatEvent { ref event } if event["text"] == "meanwhile"));
        eventually("the rotation still ends", || stored_segments(&h) == ["s1", "rotated-1"]);
        std::thread::sleep(Duration::from_millis(300));
        let after = drain(&h);
        assert!(after.iter().all(|m| !event_of(m, "agent_start")), "nothing starts after a stop");
        assert!(after.iter().all(|m| !event_of(m, "abort_seen")), "Pi was not touched mid-rotation");
    }

    #[test]
    fn a_rotation_that_panics_releases_the_tab() {
        let h = harness();
        {
            let mut inner = h.chat.inner.lock().unwrap();
            inner.rotating = true;
            inner.held.push_back(PageMessage::ChatSend { text: "held".into(), behavior: None, attachments: vec![] });
        }
        let chat = h.chat.clone();
        let thread = std::thread::spawn(move || {
            let _end = RotationEnd { chat };
            panic!("a bug in the rotation");
        });
        assert!(thread.join().is_err());
        let inner = h.chat.inner.lock().unwrap();
        assert!(!inner.rotating && inner.held.is_empty(), "the tab holds nothing any more");
        drop(inner);
        let given = wait_for(&h, "the prompt given back", |m| event_of(m, "command_failed"));
        assert!(matches!(given, DockMessage::ChatEvent { ref event } if event["text"] == "held"));
    }

    #[test]
    fn a_failed_compaction_does_not_rotate() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        h.chat.test_send_raw("compaction_failed");
        wait_for(&h, "the failed compaction", |m| event_of(m, "compaction_end"));
        std::thread::sleep(Duration::from_millis(300));
        assert!(drain(&h).iter().all(|m| !event_of(m, "rotate_seen")));
        assert_eq!(stored_segments(&h), ["s1"]);
    }

    #[test]
    fn pi_starts_on_the_last_stored_segment_or_the_tab_id() {
        let h = harness();
        ready(&h);
        assert_eq!(session_id_arg(&h), "s1", "a tab not in sessions.json runs on its own id");
        h.chat.shutdown();
        save_tab(&h, &["s1", "seg-2"]);
        ready(&h);
        assert_eq!(session_id_arg(&h), "seg-2");
    }

    #[test]
    fn a_segment_over_8_mb_is_compacted_then_rotated_and_compacted_once_per_pi() {
        let h = harness();
        save_tab(&h, &[]);
        segment_file_of_size(&h, "s1", 8 * 1024 * 1024 + 1);
        // The next segment is large too: only the once-per-Pi rule stops a second compaction.
        segment_file_of_size(&h, "rotated-1", 9 * 1024 * 1024);
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        wait_for(&h, "the compaction asked", |m| event_of(m, "compact_seen"));
        wait_for(&h, "the rotation that follows", |m| event_of(m, "rotate_seen"));
        eventually("the new segment is recorded", || stored_segments(&h) == ["s1", "rotated-1"]);
        send(&h, "x");
        wait_for(&h, "a run settles", |m| event_of(m, "agent_settled"));
        std::thread::sleep(Duration::from_millis(300));
        assert!(drain(&h).iter().all(|m| !event_of(m, "compact_seen")), "compact is sent once per Pi");
    }

    #[test]
    fn a_segment_of_8_mb_or_less_is_left_alone() {
        let h = harness();
        save_tab(&h, &[]);
        segment_file_of_size(&h, "s1", 8 * 1024 * 1024);
        ready(&h);
        std::thread::sleep(Duration::from_millis(300));
        assert!(drain(&h).iter().all(|m| !event_of(m, "compact_seen") && !event_of(m, "rotate_seen")));
    }

    #[test]
    fn a_tab_keeps_its_model_and_thinking_level_across_a_rotation() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        h.chat.handle(PageMessage::SetModel { id: 3, model: "opencode-go/kimi-k3".into() });
        wait_for(&h, "the model set", |m| matches!(m, DockMessage::Saved { id: 3 }));
        wait_for(&h, "Pi on the model", |m| event_of(m, "model_set"));
        // `/effort` in Pi: Aria is not told.
        let engine = h.chat.current_engine().unwrap();
        engine.send("set_thinking_level", json!({"level": "high"})).unwrap();
        wait_for(&h, "Pi on the level", |m| event_of(m, "thinking_set"));

        h.chat.test_send_raw("compaction_end");
        wait_for(&h, "the rotation", |m| event_of(m, "rotate_seen"));
        let model = wait_for(&h, "the model set again", |m| event_of(m, "model_set"));
        assert!(matches!(model, DockMessage::ChatEvent { ref event } if event["model"] == "opencode-go/kimi-k3"));
        let level = wait_for(&h, "the level set again", |m| event_of(m, "thinking_set"));
        assert!(matches!(level, DockMessage::ChatEvent { ref event } if event["level"] == "high"));
        wait_for(
            &h,
            "the page told",
            |m| matches!(m, DockMessage::ChatState { model: Some(m), thinking: Some(t), .. } if m == "opencode-go/kimi-k3" && t == "high"),
        );
        eventually("the new segment is recorded", || stored_segments(&h) == ["s1", "rotated-1"]);
    }

    #[test]
    fn a_rotation_answered_after_its_wait_is_still_recorded_and_a_later_pi_uses_it() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        h.chat.inner.lock().unwrap().rotate_timeout = Some(Duration::from_millis(200));
        h.chat.test_send_raw("rotate_late");
        h.chat.test_send_raw("compaction_end");
        wait_for(&h, "the rotation", |m| event_of(m, "rotate_seen"));
        eventually("the new segment is recorded", || stored_segments(&h) == ["s1", "rotated-1"]);
        h.chat.shutdown();
        ready(&h);
        assert_eq!(session_id_arg(&h), "rotated-1");
    }

    #[test]
    fn a_session_pi_moved_to_unannounced_is_recorded_at_the_next_state_read() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        h.chat.test_send_raw("move_session");
        wait_for(&h, "the move", |m| event_of(m, "moved"));
        assert_eq!(stored_segments(&h), ["s1"]);
        // A reloaded page: the running Pi is asked its state again.
        h.chat.handle(PageMessage::ChatLoad { fresh: true });
        eventually("the session Pi is on is recorded", || stored_segments(&h) == ["s1", "moved-1"]);
        h.chat.shutdown();
        ready(&h);
        assert_eq!(session_id_arg(&h), "moved-1");
    }

    #[test]
    fn a_prompt_sent_during_an_aria_started_compaction_is_answered_after_it() {
        let h = harness();
        save_tab(&h, &[]);
        segment_file_of_size(&h, "s1", 9 * 1024 * 1024);
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        wait_for(&h, "the compaction", |m| event_of(m, "compaction_start"));
        send(&h, "meanwhile");
        wait_for(&h, "the held prompt's run", |m| event_of(m, "agent_settled"));
        assert_eq!(stored_segments(&h), ["s1", "rotated-1"], "the rotation went first");
        std::thread::sleep(Duration::from_millis(200));
        let rest = drain(&h);
        assert!(rest.iter().all(|m| !event_of(m, "prompt_during_compaction")), "nothing reached Pi mid-compaction");
        assert!(rest.iter().all(|m| !event_of(m, "command_failed")), "the prompt was not refused");
        assert!(h.chat.inner.lock().unwrap().held.is_empty());
    }

    #[test]
    fn two_prompts_held_during_a_rotation_go_one_after_the_other() {
        let h = harness();
        save_tab(&h, &[]);
        ready(&h);
        h.chat.test_send_raw("compaction_end");
        wait_for(&h, "the rotation", |m| event_of(m, "rotate_seen"));
        send(&h, "one");
        send(&h, "two");
        let queued = wait_for(&h, "the second as a follow-up", |m| event_of(m, "queued_as"));
        assert!(
            matches!(queued, DockMessage::ChatEvent { ref event } if event["behavior"] == "followUp" && event["message"] == "two")
        );
        std::thread::sleep(Duration::from_millis(200));
        assert!(drain(&h).iter().all(|m| !event_of(m, "queued_as") && !event_of(m, "command_failed")));
    }

    #[test]
    fn stop_during_the_replay_stops_pi_and_drops_what_is_still_held() {
        let h = harness();
        ready(&h);
        {
            let mut inner = h.chat.inner.lock().unwrap();
            inner.replaying = true;
            inner.held.push_back(PageMessage::ChatSend { text: "later".into(), behavior: None, attachments: vec![] });
        }
        h.chat.handle(PageMessage::ChatAbort);
        let given = wait_for(&h, "the prompt given back", |m| event_of(m, "command_failed"));
        assert!(matches!(given, DockMessage::ChatEvent { ref event } if event["text"] == "later"));
        wait_for(&h, "Pi stopped", |m| event_of(m, "abort_seen"));
        assert!(h.chat.inner.lock().unwrap().held.is_empty());
    }

    #[test]
    fn a_send_counts_as_waiting_from_the_moment_it_starts() {
        let h = harness();
        // Pi is not running: the send waits on the start, the slow part.
        let starting = h.chat.spawn_lock.lock().unwrap();
        let chat = h.chat.clone();
        let sender = std::thread::spawn(move || {
            chat.handle(PageMessage::ChatSend { text: "x".into(), behavior: None, attachments: vec![] })
        });
        eventually("the prompt is noted before Pi starts", || !h.chat.inner.lock().unwrap().pending_prompts.is_empty());
        drop(starting);
        sender.join().unwrap();
        wait_for(&h, "settled", |m| event_of(m, "agent_settled"));
        eventually("accepted, no longer waiting", || h.chat.inner.lock().unwrap().pending_prompts.is_empty());
    }

    #[test]
    fn a_send_that_cannot_start_pi_leaves_nothing_waiting() {
        let h = harness_with(false);
        send(&h, "x");
        assert!(h.chat.inner.lock().unwrap().pending_prompts.is_empty());
    }

    #[test]
    fn a_rotation_holds_the_page_load_but_not_an_answer() {
        let h = harness();
        h.chat.inner.lock().unwrap().rotating = true;
        h.chat.handle(PageMessage::ChatLoad { fresh: false });
        h.chat.handle(PageMessage::UiAnswer { id: "gone".into(), value: None, confirmed: None, cancelled: Some(true) });
        assert_eq!(h.chat.inner.lock().unwrap().held.len(), 1, "only the load waits");
        assert!(drain(&h).iter().all(|m| !matches!(m, DockMessage::ChatState { .. })));
        h.chat.rotation_ended();
        wait_for(&h, "the load once the rotation ended", |m| matches!(m, DockMessage::ChatState { .. }));
    }

    #[test]
    fn chat_earlier_answers_a_page_read_from_the_tabs_segment_files() {
        let h = harness();
        save_tab(&h, &["s1", "seg-2"]);
        let dir = agent::sessions_dir(&h.paths);
        std::fs::create_dir_all(&dir).unwrap();
        let header = |id: &str| json!({"type": "session", "version": 3, "id": id, "timestamp": "t", "cwd": "/w"});
        let image = json!({"type": "image", "data": "AAAA", "mimeType": "image/png"});
        let old = [
            header("s1"),
            json!({"type": "message", "id": "m1", "parentId": null, "timestamp": "t",
                   "message": {"role": "user", "content": "one", "timestamp": 1}}),
            json!({"type": "message", "id": "m2", "parentId": "m1", "timestamp": "t",
                   "message": {"role": "toolResult", "content": [image], "timestamp": 2}}),
        ];
        let lines = |entries: &[Value]| entries.iter().map(Value::to_string).collect::<Vec<_>>().join("\n") + "\n";
        std::fs::write(dir.join("2026-10-21T00-00-00-000Z_s1.jsonl"), lines(&old)).unwrap();
        std::fs::write(dir.join("2026-10-21T01-00-00-000Z_seg-2.jsonl"), lines(&[header("seg-2")])).unwrap();

        h.chat.handle(PageMessage::ChatEarlier { cursor: None });
        let DockMessage::ChatEvent { event } = wait_for(&h, "the earlier page", |m| event_of(m, "earlier")) else {
            unreachable!()
        };
        assert_eq!(event["messages"][0]["content"], json!("one"));
        assert_eq!(event["messages"][1]["content"][0], json!({"type": "image", "mimeType": "image/png"}));
        assert_eq!(event["cursor"], json!({"segment": 0, "index": 0}));
        assert_eq!(event["more"], json!(false));
        assert!(drain(&h).iter().all(|m| !matches!(m, DockMessage::ChatEngine { .. })), "no Pi is needed");
    }

    #[test]
    fn saving_the_tabs_keeps_each_tabs_stored_segments() {
        let s = sessions_harness();
        let entry = |id: &str, title: &str, segments: &[&str]| aria_core::sessions::SessionEntry {
            id: id.into(),
            title: title.into(),
            segments: segments.iter().map(|s| s.to_string()).collect(),
            schedule: None,
            workdir: None,
        };
        sessions::save(&s.h.paths, &SessionList { sessions: vec![entry("aa", "Old", &["aa", "x-2"])], active: None })
            .unwrap();
        // The page renames a tab, adds one, and never owns the chain (no segments, or stale ones).
        s.sessions.handle(PageMessage::SessionsSave {
            sessions: vec![entry("aa", "Renamed", &[]), entry("bb", "New", &["forged"])],
            active: Some("bb".into()),
        });
        let list = sessions::load(&s.h.paths);
        assert_eq!(list.sessions, vec![entry("aa", "Renamed", &["aa", "x-2"]), entry("bb", "New", &[])]);
        assert_eq!(list.active.as_deref(), Some("bb"));
    }

    #[cfg(all(target_os = "linux", feature = "gui"))]
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
