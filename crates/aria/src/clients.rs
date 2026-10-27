//! Terminal clients of the dock's sessions (docs/superpowers/specs/2026-10-23-terminal-design.md). A client connects to
//! `clients.sock` beside the control socket and speaks the page's own messages, one JSON object per line both ways; the
//! dock's replies and events fan out to the page and to the clients attached to each session.
//!
//! - Ids: a client's request gets an id of the hub's going in (from `CLIENT_BASE`, remembered with the client and its own
//!   id) and its own back coming out, so the reply reaches only the one that asked; any other reply (a numeric id) is the
//!   page's and goes to the page only, whatever its id (the page counts from `Date.now() * 1000`).
//! - A client that does not read is dropped once its queue (`QUEUE` lines) is full: the dock never waits for one.
//! - Client-only lines: `hello { protocol, version, remote? }`, `attach { session }`, `detach`, `focus { focused }`. A session is seen while a client attached
//!   to it has the focus (a client that never said counts as focused).
//! - `hello` only counts as a connection's first line (any other first line means there is none: a local terminal): the
//!   hub answers with its own `hello` (no `remote`), and a different `protocol` is answered `protocol-mismatch` and the
//!   connection closed. `remote: true` marks a client that came in from another machine: it can never send `secretSave`
//!   or a `!command` (`not-for-remote`), and a later `hello` cannot lift the mark. It never chooses a session's folder
//!   either: it talks only to sessions the hub knows, and closes only those (`unknown-session`, so a made-up id never starts a Pi or grows the closed set), its `sessionNew` reaches the chat with no `workdir`, and of Pi's own commands (`piCommand`) it may send only
//!   `get_session_stats`, with none of the arguments it names (`REMOTE_PI_COMMANDS`). The test `the_whole_surface_a_client_may_reach_is_pinned`
//!   states everything a local and a remote client may reach, for every message the page can send.
//! - Sign-ins: a client (local or remote) may start Claude's or Notion's (`startOauth`, `startMcpOauth`) and paste the
//!   address its browser could not load (`oauthCode`); they go to the handlers, not the chat. A sign-in answers twice
//!   (`oauthUrl`, then `oauthResult` or `error`), so its id stays until the last of them. Security: a remote client can
//!   therefore re-bind the hub's Claude or Notion account, so a stolen client key could point them at the attacker's;
//!   accepted, since that key can already chat with the model and use those accounts. Pasted text holds a code: never
//!   logged.
//! - Machines (`machines.rs`): a session that lives on a computer linked to this hub as a machine is listed with the rest
//!   (`machine`, and `offline` while it is not linked), and a line for it is wrapped `{"c": client, "line": ...}` and sent
//!   over that machine's link, with the client's own ids (the machine rewrites them as its own hub does) and after the
//!   same checks a remote client's line gets here; `attach`, `detach` and `focus` follow the session they name, and
//!   `sessionNew { machine: "<name>" }` starts one there (`unknown-machine`, `machine-offline`). A session the hub has
//!   itself is never a machine's. The page's own list is not touched.
//! - `hubUpdate { id, version }` (local or remote): asks this service, when it is a hub, to update itself to that release
//!   (`hub_update.rs` holds what it may and may not do). It goes to the handlers; the answer is `hubUpdateProgress` lines and
//!   then `hubUpdateResult`, so, like a sign-in, its id stays until the last of them. A stolen client key can send it, and all
//!   it can cause is the install of a newer release from the fixed repository, once at a time.

use crate::machines::{self, Machines, Refused};
use aria_core::bridge::{DockMessage, PageMessage, PROTOCOL};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

/// Where the ids the hub gives clients' requests start: far above the page's.
const CLIENT_BASE: u64 = 1 << 62;
const QUEUE: usize = 1024;

/// `clients.sock`, beside the control socket.
pub fn socket_path() -> PathBuf {
    crate::control::control_socket_path().with_file_name("clients.sock")
}

/// What a terminal may ask the dock: its sessions, and inside one what the client's screen does, and the scheduled tasks.
/// Nothing that changes the dock's own state (what it shows, its tabs' order, its engines, dictation).
fn allowed(message: &PageMessage) -> bool {
    match message {
        PageMessage::SessionsLoad { .. }
        | PageMessage::SessionNew { .. }
        | PageMessage::SessionClose { .. }
        | PageMessage::SchedulesList { .. }
        | PageMessage::ScheduleUpdate { .. }
        | PageMessage::ScheduleDelete { .. } => true,
        PageMessage::Session { message, .. } => matches!(
            **message,
            PageMessage::ChatSend { .. }
                | PageMessage::ChatAbort
                | PageMessage::ChatLoad { .. }
                | PageMessage::UiAnswer { .. }
                | PageMessage::ListCommands { .. }
                | PageMessage::ListModels { .. }
                | PageMessage::SetModel { .. }
                | PageMessage::ListFiles { .. }
                | PageMessage::PiCommand { .. }
                | PageMessage::ChatBash { .. }
                | PageMessage::ChatCheckpoints { .. }
                | PageMessage::ChatUndo { .. }
        ),
        _ => false,
    }
}

/// What goes to the handlers and not the chat: a secret to keep, the sign-ins and a hub's update of itself.
fn for_handlers(message: &PageMessage) -> bool {
    matches!(
        message,
        PageMessage::SecretSave { .. }
            | PageMessage::StartOAuth { .. }
            | PageMessage::StartMcpOAuth { .. }
            | PageMessage::OAuthCode { .. }
            | PageMessage::HubUpdate { .. }
    )
}

/// What a client that came in from another machine may not do, whatever a local terminal may: save a secret, or run a
/// `!command` (already turned into `ChatBash`).
fn refused_for_remote(message: &PageMessage) -> bool {
    match message {
        PageMessage::SecretSave { .. } => true,
        PageMessage::Session { message, .. } => matches!(**message, PageMessage::ChatBash { .. }),
        _ => false,
    }
}

/// The Pi commands a remote client may send, each with the argument keys it may carry: only what the terminal sends
/// (`/context` is `get_session_stats` with no arguments). Anything else is `not-for-remote`: `export_html` writes to a
/// path the client names, `compact` is a model call over the whole context.
const REMOTE_PI_COMMANDS: [(&str, &[&str]); 1] = [("get_session_stats", &[])];

/// What a remote client's message becomes on its way to the chat: the same message with what it may not choose taken
/// out, or the code it is refused with. `message` is already one `allowed` lets through.
fn for_remote(message: PageMessage, known: &(dyn Fn(&str) -> bool + Send + Sync)) -> Result<PageMessage, &'static str> {
    match message {
        // Ruling 6 of the spec: no client folder over connect (a session's folder is where its commands may write).
        PageMessage::SessionNew { id, .. } => Ok(PageMessage::SessionNew { id, workdir: None }),
        PageMessage::Session { session, message } => {
            // Any id would start a Pi of its own: only the sessions the service has are talked to (`sessionNew` makes one).
            if !known(&session) {
                return Err("unknown-session");
            }
            let message = match *message {
                PageMessage::PiCommand { id, command, args } => {
                    let Some((_, keys)) = REMOTE_PI_COMMANDS.iter().find(|(name, _)| *name == command) else {
                        return Err("not-for-remote");
                    };
                    let kept: serde_json::Map<String, Value> = args
                        .as_object()
                        .map(|object| {
                            object
                                .iter()
                                .filter(|(key, _)| keys.contains(&key.as_str()))
                                .map(|(key, value)| (key.clone(), value.clone()))
                                .collect()
                        })
                        .unwrap_or_default();
                    PageMessage::PiCommand { id, command, args: Value::Object(kept) }
                }
                other => other,
            };
            Ok(PageMessage::Session { session, message: Box::new(message) })
        }
        // Closing a made-up id would still grow the closed set, rewrite `sessions.json` and tell every client.
        PageMessage::SessionClose { session } if !known(&session) => Err("unknown-session"),
        other => Ok(other),
    }
}

/// Binds the clients socket: its folder `700`, the socket `600`; a file left by a crash is replaced.
pub fn bind(path: &std::path::Path) -> std::io::Result<UnixListener> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    if let Some(dir) = path.parent() {
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
    }
    let _ = std::fs::remove_file(path);
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

struct Client {
    lines: SyncSender<String>,
    stream: UnixStream,
    session: Option<String>,
    focused: bool,
    /// Came in from another machine (its `hello` said so): see `refused_for_remote`.
    remote: bool,
    /// It has sent its first line: a `hello` only counts as that one.
    spoke: bool,
}

/// Which sessions a client is looking at; `ChatCtx` asks it whether anyone sees a session.
#[derive(Clone, Default)]
pub struct Watchers(Arc<Mutex<HashMap<u64, Client>>>);

impl Watchers {
    /// A focused terminal attached to `session` (tests elsewhere in the crate).
    #[cfg(test)]
    pub fn watching(session: &str) -> Watchers {
        let (stream, _) = UnixStream::pair().unwrap();
        let (lines, _) = mpsc::sync_channel(1);
        let client =
            Client { lines, stream, session: Some(session.to_string()), focused: true, remote: false, spoke: true };
        Watchers(Arc::new(Mutex::new(HashMap::from([(1, client)]))))
    }

    pub fn sees(&self, session: &str) -> bool {
        self.0.lock().unwrap().values().any(|c| c.focused && c.session.as_deref() == Some(session))
    }
}

#[derive(Clone)]
pub struct Hub {
    chat: Arc<Mutex<mpsc::Sender<PageMessage>>>,
    handlers: Arc<dyn Fn(PageMessage) + Send + Sync>,
    /// Is this a session the service has? A remote client may talk only to sessions that exist.
    known_session: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    clients: Watchers,
    next: Arc<AtomicU64>,
    /// The ids given to clients' requests. A reply whose id is not here is the page's.
    asked: Arc<Mutex<HashMap<u64, Ask>>>,
    next_ask: Arc<AtomicU64>,
    /// The computers linked to this hub as machines (`machines.rs`): their sessions join the list, and a line for one of
    /// them travels over its link.
    machines: Machines,
}

/// A client's request in flight: who asked, with which id, and what kind of request it was.
struct Ask {
    client: u64,
    own: u64,
    kind: AskKind,
}

#[derive(PartialEq, Clone, Copy)]
enum AskKind {
    Plain,
    /// `startOauth` / `startMcpOauth`: answers `oauthUrl` and then `oauthResult` (or `error`).
    SignIn,
    /// `oauthCode`: answered only when refused; a good one is answered by the sign-in's own result.
    Code,
    /// `hubUpdate`: answers `hubUpdateProgress` as often as the installer prints, and then `hubUpdateResult` (or `error`).
    Update,
}

/// A client's line as the page message it is, with a `!command` (a `piCommand` named `bash`, which the page can never
/// send) turned into its own message so every rule that names it sees it.
fn client_message(value: Value) -> Option<PageMessage> {
    serde_json::from_value::<PageMessage>(value).ok().map(|message| match message {
        PageMessage::Session { session, message } => match *message {
            PageMessage::PiCommand { id, command, args } if command == "bash" => PageMessage::Session {
                session,
                message: Box::new(PageMessage::ChatBash {
                    id,
                    command: args["command"].as_str().unwrap_or_default().to_string(),
                }),
            },
            other => PageMessage::Session { session, message: Box::new(other) },
        },
        other => other,
    })
}

/// The numeric request id of a message (at its top, or inside a `session` wrapper), as a mutable slot.
fn id_slot(value: &mut Value) -> Option<&mut Value> {
    let slot =
        if value["type"] == "session" { value.get_mut("message")?.get_mut("id")? } else { value.get_mut("id")? };
    slot.is_u64().then_some(slot)
}

impl Hub {
    pub fn new(
        chat: mpsc::Sender<PageMessage>,
        handlers: Arc<dyn Fn(PageMessage) + Send + Sync>,
        known_session: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    ) -> Hub {
        Hub {
            chat: Arc::new(Mutex::new(chat)),
            handlers,
            known_session,
            clients: Watchers::default(),
            next: Arc::new(AtomicU64::new(1)),
            asked: Arc::default(),
            next_ask: Arc::new(AtomicU64::new(CLIENT_BASE)),
            machines: Machines::default(),
        }
    }

    /// The registry of linked machines (the tests look at it; the machine side's commands will too).
    #[cfg(test)]
    pub fn machines(&self) -> Machines {
        self.machines.clone()
    }

    /// A hub with other link intervals, before anything is served (the tests need not wait 45 seconds).
    #[cfg(test)]
    pub fn with_timing(mut self, timing: machines::Timing) -> Hub {
        self.machines = Machines::with_timing(timing);
        self
    }

    /// Accepts the machines' links on `listener` (`aria hub-link machine <name>` is the other end).
    pub fn serve_machines(&self, listener: UnixListener) {
        self.machines.serve(self.clone(), listener);
    }

    /// A line to client `number`, if it is still there (the machines' deliveries).
    pub(crate) fn send_line(&self, number: u64, value: &Value) {
        self.send_to(number, value);
    }

    /// A line to every client.
    pub(crate) fn broadcast(&self, value: &Value) {
        let numbers: Vec<u64> = self.clients.0.lock().unwrap().keys().copied().collect();
        for number in numbers {
            self.send_to(number, value);
        }
    }

    pub fn watchers(&self) -> Watchers {
        self.clients.clone()
    }

    #[cfg(test)]
    pub fn count(&self) -> usize {
        self.clients.0.lock().unwrap().len()
    }

    /// Accepts clients until the listener fails (a thread per client).
    pub fn serve(&self, listener: UnixListener) {
        let hub = self.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().map_while(Result::ok) {
                hub.add(stream);
            }
        });
    }

    /// One client: a writer thread drains its queue, a reader thread takes its lines.
    pub fn add(&self, stream: UnixStream) {
        let (Ok(reader), Ok(mut writer), Ok(kept)) = (stream.try_clone(), stream.try_clone(), stream.try_clone())
        else {
            return;
        };
        let number = self.next.fetch_add(1, Ordering::SeqCst);
        let (lines, queue) = mpsc::sync_channel::<String>(QUEUE);
        self.clients
            .0
            .lock()
            .unwrap()
            .insert(number, Client { lines, stream: kept, session: None, focused: true, remote: false, spoke: false });
        let hub = self.clone();
        std::thread::spawn(move || {
            for line in queue {
                if writer.write_all(line.as_bytes()).and_then(|_| writer.write_all(b"\n")).is_err() {
                    break;
                }
            }
            // The queue is done (or the peer is gone): what was queued has been written, so now the end can be shut.
            let _ = writer.shutdown(std::net::Shutdown::Both);
            hub.drop_client(number);
        });
        let hub = self.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(reader).lines().map_while(Result::ok) {
                hub.take(number, &line);
            }
            hub.drop_client(number);
        });
    }

    fn drop_client(&self, number: u64) {
        // The machines it had a route to are told it is gone (nothing happens when it had none).
        self.machines.client_gone(number);
        if let Some(client) = self.clients.0.lock().unwrap().remove(&number) {
            let _ = client.stream.shutdown(std::net::Shutdown::Both);
        }
        // Nobody is left to answer: a request still waiting (a sign-in, a paste) would stay for ever.
        self.asked.lock().unwrap().retain(|_, ask| ask.client != number);
    }

    /// Closes a client after what is queued for it: its queue is closed (the writer thread drains it, then shuts the
    /// socket), so a last line (an error) is not cut off by the shutdown.
    fn drop_after_flush(&self, number: u64) {
        self.clients.0.lock().unwrap().remove(&number);
    }

    /// Notes that client `number` has sent a line; returns whether it was its first and whether it is remote (`None`
    /// when the client is gone: its lines are not taken any more).
    fn spoke(&self, number: u64) -> Option<(bool, bool)> {
        let mut clients = self.clients.0.lock().unwrap();
        let client = clients.get_mut(&number)?;
        let first = !std::mem::replace(&mut client.spoke, true);
        Some((first, client.remote))
    }

    /// A client's first line was a `hello`: marks it remote or not, answers with the hub's own, and refuses another
    /// protocol.
    fn hello(&self, number: u64, value: &Value) {
        self.set(number, |c| c.remote = value["remote"] == true);
        let version = env!("CARGO_PKG_VERSION");
        self.send_to(number, &json!({"type": "hello", "protocol": PROTOCOL, "version": version}));
        if value["protocol"].as_u64() != Some(u64::from(PROTOCOL)) {
            self.send_to(number, &json!({"type": "error", "id": 0, "code": "protocol-mismatch"}));
            self.drop_after_flush(number);
        }
    }

    fn send_to(&self, number: u64, value: &Value) {
        let mut clients = self.clients.0.lock().unwrap();
        let Some(client) = clients.get(&number) else { return };
        if let Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) = client.lines.try_send(value.to_string()) {
            if let Some(client) = clients.remove(&number) {
                let _ = client.stream.shutdown(std::net::Shutdown::Both);
            }
        }
    }

    /// A line from client `number`.
    fn take(&self, number: u64, line: &str) {
        // Any first line counts as the one a `hello` has to be, so garbage first does not make room for a late `hello`.
        let Some((first, remote)) = self.spoke(number) else { return };
        let Ok(mut value) = serde_json::from_str::<Value>(line) else { return };
        match value["type"].as_str() {
            Some("hello") => {
                if first {
                    self.hello(number, &value);
                }
                return;
            }
            Some("attach") => {
                let session = value["session"].as_str().map(str::to_string);
                if let Some(session) = &session {
                    self.follow(number, session, &value);
                }
                return self.set(number, |c| c.session = session);
            }
            Some("detach") => {
                self.follow_current(number, &value);
                return self.set(number, |c| c.session = None);
            }
            Some("focus") => {
                self.follow_current(number, &value);
                return self.set(number, |c| c.focused = value["focused"].as_bool().unwrap_or(true));
            }
            _ => {}
        }
        // A line for a session that lives on a machine goes there, with the client's own ids, and never to the chat.
        if self.machine_route(number, remote, &value) {
            return;
        }
        let kind = match value["type"].as_str() {
            Some("startOauth" | "startMcpOauth") => AskKind::SignIn,
            Some("oauthCode") => AskKind::Code,
            Some("hubUpdate") => AskKind::Update,
            _ => AskKind::Plain,
        };
        // (the client's own id, the hub's)
        let ids = id_slot(&mut value).and_then(|slot| {
            let own = slot.as_u64()?;
            let ours = self.next_ask.fetch_add(1, Ordering::SeqCst);
            self.asked.lock().unwrap().insert(ours, Ask { client: number, own, kind });
            *slot = json!(ours);
            Some((own, ours))
        });
        let own = ids.map(|(own, _)| own);
        // A request that is refused is never asked of anyone, so nothing is waited for.
        let refuse = |code: &str| {
            if let Some((_, ours)) = ids {
                self.asked.lock().unwrap().remove(&ours);
            }
            self.send_to(number, &json!({"type": "error", "id": own.unwrap_or(0), "code": code}));
        };
        // A client's `!command`: its own message, which the page can never send.
        let message = client_message(value);
        match message {
            Some(message) if remote && refused_for_remote(&message) => refuse("not-for-remote"),
            Some(message) if allowed(&message) => {
                match if remote { for_remote(message, &*self.known_session) } else { Ok(message) } {
                    Ok(message) => {
                        let _ = self.chat.lock().unwrap().send(message);
                    }
                    Err(code) => refuse(code),
                }
            }
            Some(message) if for_handlers(&message) => (self.handlers)(message),
            _ => refuse("not-for-clients"),
        }
    }

    /// The machine a session lives on, when it is a machine's and not one of this hub's own.
    fn machine_of(&self, session: &str) -> Option<String> {
        if (self.known_session)(session) {
            return None;
        }
        self.machines.owner_of(session)
    }

    /// `line` (an `attach`, `detach` or `focus`) goes on to the machine that owns `session`, if one does.
    fn follow(&self, number: u64, session: &str, line: &Value) {
        if let Some(machine) = self.machine_of(session) {
            let _ = self.machines.forward(&machine, number, line);
        }
    }

    /// The same for the session the client is attached to now.
    fn follow_current(&self, number: u64, line: &Value) {
        let current = self.clients.0.lock().unwrap().get(&number).and_then(|c| c.session.clone());
        if let Some(session) = current {
            self.follow(number, &session, line);
        }
    }

    /// A client's line addressed to a machine: a message for one of its sessions, closing one, or a new session that
    /// names it. Returns whether it was one (and was handled: forwarded, or refused with the client's own id).
    fn machine_route(&self, number: u64, remote: bool, value: &Value) -> bool {
        let refuse = |own: u64, code: &str| self.send_to(number, &json!({"type": "error", "id": own, "code": code}));
        let offline = |own: u64, outcome: Result<(), Refused>| match outcome {
            Ok(()) => {}
            Err(Refused::Unknown) => refuse(own, "unknown-machine"),
            Err(Refused::Offline) => refuse(own, "machine-offline"),
        };
        match value["type"].as_str() {
            Some("session") => {
                let Some(machine) = value["session"].as_str().and_then(|session| self.machine_of(session)) else {
                    return false;
                };
                let own = value["message"]["id"].as_u64().unwrap_or(0);
                // What the hub lets a client do with a session, and what a remote one may not: the machine checks it again.
                match client_message(value.clone()) {
                    Some(message) if remote && refused_for_remote(&message) => refuse(own, "not-for-remote"),
                    Some(message) if allowed(&message) => offline(own, self.machines.forward(&machine, number, value)),
                    _ => refuse(own, "not-for-clients"),
                }
                true
            }
            Some("sessionClose") => {
                let Some(machine) = value["session"].as_str().and_then(|session| self.machine_of(session)) else {
                    return false;
                };
                offline(0, self.machines.forward(&machine, number, value));
                true
            }
            Some("sessionNew") if !value["machine"].is_null() && value.get("machine").is_some() => {
                let own = value["id"].as_u64().unwrap_or(0);
                let Some(machine) = value["machine"].as_str().filter(|name| machines::valid_name(name)) else {
                    refuse(own, "unknown-machine");
                    return true;
                };
                // The machine is the one asked: its name does not go on.
                let mut line = value.clone();
                if let Some(object) = line.as_object_mut() {
                    object.remove("machine");
                }
                offline(own, self.machines.forward(machine, number, &line));
                true
            }
            _ => false,
        }
    }

    fn set(&self, number: u64, change: impl FnOnce(&mut Client)) {
        if let Some(client) = self.clients.0.lock().unwrap().get_mut(&number) {
            change(client);
        }
    }

    /// Delivers a dock message to the clients it is for; returns whether the page should get it too (false for a reply
    /// to a client's request).
    pub fn route(&self, message: &DockMessage) -> bool {
        let Ok(mut value) = serde_json::to_value(message) else { return true };
        let is_url = value["type"] == "oauthUrl";
        let is_progress = value["type"] == "hubUpdateProgress";
        if let Some(slot) = id_slot(&mut value) {
            let id = slot.as_u64().unwrap_or(0);
            let asked = {
                let mut asked = self.asked.lock().unwrap();
                // A sign-in answers twice: its URL leaves the id in place for the result (or error) that follows; an update
                // answers a line at a time and leaves it for the result. Any other reply is the last: it also ends what the
                // client pasted into that sign-in.
                let partial = asked.get(&id).is_some_and(|a| {
                    (is_url && a.kind == AskKind::SignIn) || (is_progress && a.kind == AskKind::Update)
                });
                let found = if partial {
                    asked.get(&id).map(|a| Ask { client: a.client, own: a.own, kind: a.kind })
                } else {
                    asked.remove(&id)
                };
                if let Some(ask) = found.as_ref().filter(|a| a.kind == AskKind::SignIn && !partial) {
                    asked.retain(|_, other| !(other.client == ask.client && other.kind == AskKind::Code));
                }
                found
            };
            if let Some(Ask { client: number, own, .. }) = asked {
                *slot = json!(own);
                // The list a client asks for has the machines' sessions in it, and their last statuses follow it.
                let is_list = value["type"] == "sessions";
                if is_list {
                    self.machines.merge_into(&mut value);
                }
                self.send_to(number, &value);
                if is_list {
                    for status in self.machines.status_events() {
                        self.send_to(number, &status);
                    }
                }
                // A tab a client opened is everyone's: the page and the other clients get it, without the asker's id.
                if value["type"] == "sessionAdded" {
                    if let Some(object) = value.as_object_mut() {
                        object.remove("id");
                    }
                    let others: Vec<u64> =
                        self.clients.0.lock().unwrap().keys().copied().filter(|n| *n != number).collect();
                    for other in others {
                        self.send_to(other, &value);
                    }
                    return true;
                }
                return false;
            }
            // A reply to the page's own request (0 is no request: a notice for whoever is there).
            if id != 0 {
                return true;
            }
        }
        let numbers: Vec<u64> = {
            let clients = self.clients.0.lock().unwrap();
            match value["type"].as_str() {
                // What a session is doing is for every client's list, attached or not.
                Some("session") if value["message"]["type"] == "sessionStatus" => clients.keys().copied().collect(),
                Some("session") => {
                    let session = value["session"].as_str();
                    clients.iter().filter(|(_, c)| c.session.as_deref() == session).map(|(n, _)| *n).collect()
                }
                _ => clients.keys().copied().collect(),
            }
        };
        for number in numbers {
            self.send_to(number, &value);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::bridge::{DockMessage, PageMessage, PROTOCOL};
    use serde_json::{json, Value};
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    struct Fixture {
        hub: Hub,
        chat: mpsc::Receiver<PageMessage>,
        handled: mpsc::Receiver<PageMessage>,
    }

    fn fixture() -> Fixture {
        fixture_knowing(Arc::new(|_| true))
    }

    fn fixture_knowing(known: Arc<dyn Fn(&str) -> bool + Send + Sync>) -> Fixture {
        let (chat_tx, chat) = mpsc::channel();
        let (handled_tx, handled) = mpsc::channel();
        let handled_tx = std::sync::Mutex::new(handled_tx);
        let hub = Hub::new(
            chat_tx,
            std::sync::Arc::new(move |m| {
                let _ = handled_tx.lock().unwrap().send(m);
            }),
            known,
        );
        Fixture { hub, chat, handled }
    }

    /// A client connected to the hub: the test's end of a socket pair.
    struct Peer {
        out: UnixStream,
        lines: BufReader<UnixStream>,
    }

    impl Peer {
        fn send(&mut self, value: Value) {
            writeln!(self.out, "{value}").unwrap();
        }
        fn next(&mut self) -> Value {
            let mut line = String::new();
            self.lines.read_line(&mut line).unwrap();
            serde_json::from_str(&line).unwrap()
        }
    }

    fn connect(f: &Fixture) -> Peer {
        let (ours, theirs) = UnixStream::pair().unwrap();
        f.hub.add(theirs);
        ours.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        Peer { lines: BufReader::new(ours.try_clone().unwrap()), out: ours }
    }

    fn wait_until(what: &str, done: impl Fn() -> bool) {
        let start = Instant::now();
        while !done() {
            assert!(start.elapsed() < Duration::from_secs(5), "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn id_of(message: &PageMessage) -> u64 {
        match message {
            PageMessage::SessionsLoad { id } => *id,
            PageMessage::Session { message, .. } => match **message {
                PageMessage::ListModels { id } => id,
                ref other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_socket_is_private_and_replaces_a_stale_one() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run").join("clients.sock");
        drop(bind(&path).unwrap());
        let _again = bind(&path).expect("a file nobody listens on is replaced");
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(path.parent().unwrap()).unwrap().permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn the_page_gets_its_replies_whatever_its_ids() {
        // The page numbers its requests from Date.now() * 1000: huge, and never a client's.
        let f = fixture();
        let mut a = connect(&f);
        a.send(json!({"type": "attach", "session": "aa"}));
        wait_until("attached", || f.hub.watchers().sees("aa"));
        let page_id = 1_792_760_400_000_000;
        assert!(f.hub.route(&DockMessage::Settings {
            id: page_id,
            settings: None,
            exists: false,
            problem: None,
            connected: vec![],
            servers: vec![]
        }));
        let reply = DockMessage::Session {
            session: "aa".into(),
            message: Box::new(DockMessage::Models { id: page_id + 1, models: json!([]) }),
        };
        assert!(f.hub.route(&reply));
        assert!(f.hub.route(&DockMessage::SessionsChanged));
        assert_eq!(a.next()["type"], "sessionsChanged", "the page's replies never reached the client");
    }

    #[test]
    fn hub_gives_each_requester_its_own_reply() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(json!({"type": "sessionsLoad", "id": 1}));
        let forwarded = f.chat.recv_timeout(Duration::from_secs(5)).unwrap();
        let theirs = id_of(&forwarded);
        assert_ne!(theirs, 1, "a client's id never reaches the dock as the page's");
        // The page asked with id 1 too: its reply goes to the page only.
        assert!(f.hub.route(&DockMessage::Sessions { id: 1, sessions: vec![], active: None }));
        // The client's reply goes to the client only, with its own id back.
        assert!(!f.hub.route(&DockMessage::Sessions { id: theirs, sessions: vec![], active: Some("aa".into()) }));
        let got = a.next();
        assert_eq!(got["type"], "sessions");
        assert_eq!(got["id"], 1);
        assert_eq!(got["active"], "aa");
        // A session request is remapped inside its wrapper too.
        a.send(json!({"type": "session", "session": "aa", "message": {"type": "listModels", "id": 2}}));
        let inner = id_of(&f.chat.recv_timeout(Duration::from_secs(5)).unwrap());
        assert_ne!(inner, 2);
        let reply = DockMessage::Session {
            session: "aa".into(),
            message: Box::new(DockMessage::Models { id: inner, models: json!([]) }),
        };
        assert!(!f.hub.route(&reply));
        assert_eq!(a.next()["message"]["id"], 2);
    }

    #[test]
    fn a_tab_a_client_opened_reaches_the_page_and_the_other_clients_too() {
        let f = fixture();
        let mut asker = connect(&f);
        let mut other = connect(&f);
        asker.send(json!({"type": "sessionNew", "id": 3, "workdir": "/home/ana"}));
        let theirs = match f.chat.recv_timeout(Duration::from_secs(5)).unwrap() {
            PageMessage::SessionNew { id, .. } => id,
            other => panic!("{other:?}"),
        };
        let entry = aria_core::sessions::SessionEntry {
            id: "b2".into(),
            title: "".into(),
            segments: vec![],
            schedule: None,
            workdir: Some("/home/ana".into()),
        };
        assert!(f.hub.route(&DockMessage::SessionAdded { id: Some(theirs), session: entry }), "the page adds the tab");
        assert_eq!(asker.next()["id"], 3);
        let seen = other.next();
        assert_eq!(seen["type"], "sessionAdded");
        assert!(seen.get("id").is_none(), "another client's id is never shown to the rest");
    }

    #[test]
    fn a_clients_command_reaches_the_chat_as_its_own_message() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(json!({"type": "session", "session": "aa", "message": {"type": "piCommand", "id": 4, "command": "bash", "args": {"command": "pwd"}}}));
        match f.chat.recv_timeout(Duration::from_secs(5)).unwrap() {
            PageMessage::Session { message, .. } => {
                assert!(matches!(*message, PageMessage::ChatBash { ref command, .. } if command == "pwd"))
            }
            other => panic!("{other:?}"),
        }
        assert!(
            serde_json::from_str::<PageMessage>(r#"{"type":"chatBash","id":1,"command":"id"}"#).is_err(),
            "never from JSON"
        );
    }

    #[test]
    fn clients_may_send_only_their_list() {
        let f = fixture();
        let mut a = connect(&f);
        for (n, refused) in [
            json!({"type": "chatShown", "shown": false}),
            json!({"type": "sessionsSave", "sessions": [], "active": null}),
            json!({"type": "enginesRestart"}),
            json!({"type": "session", "session": "aa", "message": {"type": "dictationStart"}}),
        ]
        .into_iter()
        .enumerate()
        {
            a.send(refused);
            assert_eq!(a.next()["code"], "not-for-clients", "refused #{n}");
        }
        assert!(f.chat.try_recv().is_err(), "none of them reached the chat");
        a.send(json!({"type": "session", "session": "aa", "message": {"type": "chatAbort"}}));
        assert!(f.chat.recv_timeout(Duration::from_secs(5)).is_ok());
    }

    fn hello(protocol: u32, remote: Option<bool>) -> Value {
        let mut line = json!({"type": "hello", "protocol": protocol, "version": "9.9.9"});
        if let Some(remote) = remote {
            line["remote"] = json!(remote);
        }
        line
    }

    fn secret_save(id: u64) -> Value {
        json!({"type": "secretSave", "id": id, "name": "GH_TOKEN", "value": "v", "purpose": ""})
    }

    #[test]
    fn hello_is_answered_with_the_protocol() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(hello(PROTOCOL, None));
        assert_eq!(a.next(), json!({"type": "hello", "protocol": PROTOCOL, "version": env!("CARGO_PKG_VERSION")}));
    }

    #[test]
    fn a_different_protocol_is_refused_and_the_client_dropped() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(hello(PROTOCOL + 1, None));
        assert_eq!(a.next()["type"], "hello");
        assert_eq!(a.next(), json!({"type": "error", "id": 0, "code": "protocol-mismatch"}));
        wait_until("dropped", || f.hub.count() == 0);
    }

    #[test]
    fn a_remote_client_cannot_save_secrets_or_run_bash() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(hello(PROTOCOL, Some(true)));
        assert_eq!(a.next()["type"], "hello");
        a.send(secret_save(5));
        assert_eq!(a.next(), json!({"type": "error", "id": 5, "code": "not-for-remote"}));
        a.send(json!({"type": "session", "session": "aa", "message": {"type": "piCommand", "id": 6, "command": "bash", "args": {"command": "pwd"}}}));
        assert_eq!(a.next(), json!({"type": "error", "id": 6, "code": "not-for-remote"}));
        assert!(f.handled.try_recv().is_err(), "no secret reached the handlers");
        assert!(f.chat.try_recv().is_err(), "no command reached the chat");
        a.send(json!({"type": "session", "session": "aa", "message": {"type": "chatAbort"}}));
        match f.chat.recv_timeout(Duration::from_secs(5)).unwrap() {
            PageMessage::Session { session, message } => {
                assert_eq!(session, "aa");
                assert!(matches!(*message, PageMessage::ChatAbort));
            }
            other => panic!("{other:?}"),
        }
    }

    fn new_session_workdir(f: &Fixture) -> Option<String> {
        match f.chat.recv_timeout(Duration::from_secs(5)).unwrap() {
            PageMessage::SessionNew { workdir, .. } => workdir,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_remote_client_never_chooses_a_sessions_folder_but_a_local_one_does() {
        let f = fixture();
        let mut remote = connect(&f);
        remote.send(hello(PROTOCOL, Some(true)));
        remote.next();
        remote.send(json!({"type": "sessionNew", "id": 1, "workdir": "/home/ana/.local/bin"}));
        assert_eq!(new_session_workdir(&f), None, "the hub drops a remote client's folder");
        let mut local = connect(&f);
        local.send(json!({"type": "sessionNew", "id": 2, "workdir": "/home/ana/Projects/checkout-widget"}));
        assert_eq!(new_session_workdir(&f).as_deref(), Some("/home/ana/Projects/checkout-widget"));
    }

    fn pi_command(id: u64, command: &str, args: Value) -> Value {
        json!({"type": "session", "session": "aa", "message": {"type": "piCommand", "id": id, "command": command, "args": args}})
    }

    #[test]
    fn a_remote_clients_pi_commands_are_only_the_ones_the_terminal_sends() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(hello(PROTOCOL, Some(true)));
        a.next();
        // `export_html` writes wherever `outputPath` says; `compact` is a model call: neither is the terminal's
        for (id, command, args) in [
            (1, "export_html", json!({"outputPath": "/home/ana/.ssh/authorized_keys"})),
            (2, "compact", json!({"customInstructions": "x"})),
            (3, "set_model", json!({})),
        ] {
            a.send(pi_command(id, command, args));
            assert_eq!(a.next(), json!({"type": "error", "id": id, "code": "not-for-remote"}), "{command}");
        }
        assert!(f.chat.try_recv().is_err(), "none of them reached the chat");
        // the one it sends goes through with only the arguments the terminal puts in it
        a.send(pi_command(4, "get_session_stats", json!({"outputPath": "/home/ana/x", "extra": 1})));
        match f.chat.recv_timeout(Duration::from_secs(5)).unwrap() {
            PageMessage::Session { message, .. } => match *message {
                PageMessage::PiCommand { command, args, .. } => {
                    assert_eq!(command, "get_session_stats");
                    assert_eq!(args, json!({}), "no argument the terminal does not send");
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_local_clients_pi_commands_are_not_filtered() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(pi_command(1, "export_html", json!({"outputPath": "/home/ana/page.html"})));
        match f.chat.recv_timeout(Duration::from_secs(5)).unwrap() {
            PageMessage::Session { message, .. } => match *message {
                PageMessage::PiCommand { command, args, .. } => {
                    assert_eq!(command, "export_html");
                    assert_eq!(args["outputPath"], "/home/ana/page.html");
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    // ---- The whole surface a client may reach, pinned in one place. ----

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Fate {
        /// Reaches the chat's worker.
        Chat,
        /// Reaches the handlers.
        Handlers,
        Refused(&'static str),
    }
    use Fate::{Chat as C, Handlers as H, Refused as R};
    const NC: Fate = R("not-for-clients");
    const NR: Fate = R("not-for-remote");

    /// Every `PageMessage` variant, with a name. The match has no wildcard: a new variant stops this from compiling until
    /// it is named here, and `every_variant_is_built_and_in_the_table` then asks for its constructor and its row.
    fn label(message: &PageMessage) -> &'static str {
        match message {
            PageMessage::Ready => "ready",
            PageMessage::Opened => "opened",
            PageMessage::CloseRequest => "closeRequest",
            PageMessage::QuitRequest => "quitRequest",
            PageMessage::LoadSettings { .. } => "loadSettings",
            PageMessage::SaveSettings { .. } => "saveSettings",
            PageMessage::SaveKey { .. } => "saveKey",
            PageMessage::SecretSave { .. } => "secretSave",
            PageMessage::SecretsList { .. } => "secretsList",
            PageMessage::SecretDelete { .. } => "secretDelete",
            PageMessage::CheckKey { .. } => "checkKey",
            PageMessage::CredentialDelete { .. } => "credentialDelete",
            PageMessage::StartOAuth { .. } => "startOauth",
            PageMessage::StartMcpOAuth { .. } => "startMcpOauth",
            PageMessage::OAuthCode { .. } => "oauthCode",
            PageMessage::HubUpdate { .. } => "hubUpdate",
            PageMessage::OpenUrl { .. } => "openUrl",
            PageMessage::ChatSend { .. } => "chatSend",
            PageMessage::ChatAbort => "chatAbort",
            PageMessage::ChatLoad { .. } => "chatLoad",
            PageMessage::ChatEarlier { .. } => "chatEarlier",
            PageMessage::ChatClearQueue { .. } => "chatClearQueue",
            PageMessage::UiAnswer { .. } => "uiAnswer",
            PageMessage::ListCommands { .. } => "listCommands",
            PageMessage::ListModels { .. } => "listModels",
            PageMessage::SetModel { .. } => "setModel",
            PageMessage::ListFiles { .. } => "listFiles",
            PageMessage::PiCommand { .. } => "piCommand",
            PageMessage::ChatBash { .. } => "chatBash",
            PageMessage::ChatCheckpoints { .. } => "chatCheckpoints",
            PageMessage::ChatUndo { .. } => "chatUndo",
            PageMessage::InstallClaudeBridge { .. } => "installClaudeBridge",
            PageMessage::EnginesRestart => "enginesRestart",
            PageMessage::McpScan { .. } => "mcpScan",
            PageMessage::McpImport { .. } => "mcpImport",
            PageMessage::ChatShown { .. } => "chatShown",
            PageMessage::Session { .. } => "session",
            PageMessage::SessionsLoad { .. } => "sessionsLoad",
            PageMessage::SessionsSave { .. } => "sessionsSave",
            PageMessage::SessionClose { .. } => "sessionClose",
            PageMessage::SessionNew { .. } => "sessionNew",
            PageMessage::SchedulesList { .. } => "schedulesList",
            PageMessage::ScheduleUpdate { .. } => "scheduleUpdate",
            PageMessage::ScheduleApprove { .. } => "scheduleApprove",
            PageMessage::ScheduleDelete { .. } => "scheduleDelete",
            PageMessage::JobStop { .. } => "jobStop",
            PageMessage::JobOutput { .. } => "jobOutput",
            PageMessage::JobDismiss { .. } => "jobDismiss",
            PageMessage::TraceSet { .. } => "traceSet",
            PageMessage::DictationStart => "dictationStart",
            PageMessage::DictationStop => "dictationStop",
            PageMessage::DictationCancel => "dictationCancel",
            PageMessage::DictationDownload => "dictationDownload",
            PageMessage::Wide { .. } => "wide",
        }
    }

    fn s(text: &str) -> String {
        text.to_string()
    }

    fn every_variant() -> Vec<PageMessage> {
        use PageMessage as P;
        vec![
            P::Ready,
            P::Opened,
            P::CloseRequest,
            P::QuitRequest,
            P::LoadSettings { id: 77 },
            P::SaveSettings { id: 77, settings: json!({}) },
            P::SaveKey { id: 77, provider: s("opencode-go"), key: s("k") },
            P::SecretSave { id: 77, name: s("GH_TOKEN"), value: s("v"), purpose: s("") },
            P::SecretsList { id: 77 },
            P::SecretDelete { id: 77, name: s("GH_TOKEN") },
            P::CheckKey { id: 77, provider: s("opencode-go") },
            P::CredentialDelete { id: 77, provider: s("opencode-go") },
            P::StartOAuth { id: 77 },
            P::StartMcpOAuth { id: 77, service: s("notion") },
            P::OAuthCode { id: 77, service: s("notion"), redirect: s("code=a&state=s") },
            P::HubUpdate { id: 77, version: s("0.22.0") },
            P::OpenUrl { url: s("https://example.com") },
            P::ChatSend { text: s("hi"), behavior: None, attachments: vec![] },
            P::ChatAbort,
            P::ChatLoad { fresh: false },
            P::ChatEarlier { cursor: None },
            P::ChatClearQueue { steer: vec![], keep: vec![] },
            P::UiAnswer { id: s("d1"), value: None, confirmed: Some(true), cancelled: None },
            P::ListCommands { id: 77 },
            P::ListModels { id: 77 },
            P::SetModel { id: 77, model: s("acme/model") },
            P::ListFiles { id: 77, query: s("src") },
            P::PiCommand { id: 77, command: s("get_session_stats"), args: json!({}) },
            P::ChatBash { id: 77, command: s("id") },
            P::ChatCheckpoints { id: 77 },
            P::ChatUndo { id: 77, entry: s("e1"), checkpoint: None, files: false, conversation: true },
            P::InstallClaudeBridge { id: 77 },
            P::EnginesRestart,
            P::McpScan { id: 77 },
            P::McpImport { id: 77, ids: vec![] },
            P::ChatShown { shown: true },
            P::Session { session: s("aa"), message: Box::new(P::ChatAbort) },
            P::SessionsLoad { id: 77 },
            P::SessionsSave { sessions: vec![], active: None },
            P::SessionClose { session: s("aa") },
            P::SessionNew { id: 77, workdir: None },
            P::SchedulesList { id: 77 },
            P::ScheduleUpdate { id: 77, schedule: s("s-1"), enabled: false },
            P::ScheduleApprove { id: 77, schedule: s("s-1") },
            P::ScheduleDelete { id: 77, schedule: s("s-1") },
            P::JobStop { job: s("j1") },
            P::JobOutput { id: 77, job: s("j1"), lines: None },
            P::JobDismiss { job: s("j1") },
            P::TraceSet { on: true },
            P::DictationStart,
            P::DictationStop,
            P::DictationCancel,
            P::DictationDownload,
            P::Wide { on: true },
        ]
    }

    /// Each variant's fate, sent by itself and wrapped in `session { aa }`, from a local client and from a remote one:
    /// `(variant, bare local, bare remote, wrapped local, wrapped remote)`. This is the whole surface.
    #[rustfmt::skip]
    const SURFACE: [(&str, Fate, Fate, Fate, Fate); 54] = [
        ("ready",               NC, NC, NC, NC),
        ("opened",              NC, NC, NC, NC),
        ("closeRequest",        NC, NC, NC, NC),
        ("quitRequest",         NC, NC, NC, NC),
        ("loadSettings",        NC, NC, NC, NC),
        ("saveSettings",        NC, NC, NC, NC),
        ("saveKey",             NC, NC, NC, NC),
        ("secretSave",          H,  NR, NC, NC),
        ("secretsList",         NC, NC, NC, NC),
        ("secretDelete",        NC, NC, NC, NC),
        ("checkKey",            NC, NC, NC, NC),
        ("credentialDelete",    NC, NC, NC, NC),
        ("startOauth",          H,  H,  NC, NC),
        ("startMcpOauth",       H,  H,  NC, NC),
        ("oauthCode",           H,  H,  NC, NC),
        ("hubUpdate",           H,  H,  NC, NC),
        ("openUrl",             NC, NC, NC, NC),
        ("chatSend",            NC, NC, C,  C),
        ("chatAbort",           NC, NC, C,  C),
        ("chatLoad",            NC, NC, C,  C),
        ("chatEarlier",         NC, NC, NC, NC),
        ("chatClearQueue",      NC, NC, NC, NC),
        ("uiAnswer",            NC, NC, C,  C),
        ("listCommands",        NC, NC, C,  C),
        ("listModels",          NC, NC, C,  C),
        ("setModel",            NC, NC, C,  C),
        ("listFiles",           NC, NC, C,  C),
        ("piCommand",           NC, NC, C,  C),
        ("chatBash",            NC, NC, C,  NR),
        ("chatCheckpoints",     NC, NC, C,  C),
        ("chatUndo",            NC, NC, C,  C),
        ("installClaudeBridge", NC, NC, NC, NC),
        ("enginesRestart",      NC, NC, NC, NC),
        ("mcpScan",             NC, NC, NC, NC),
        ("mcpImport",           NC, NC, NC, NC),
        ("chatShown",           NC, NC, NC, NC),
        ("session",             C,  C,  NC, NC),
        ("sessionsLoad",        C,  C,  NC, NC),
        ("sessionsSave",        NC, NC, NC, NC),
        ("sessionClose",        C,  C,  NC, NC),
        ("sessionNew",          C,  C,  NC, NC),
        ("schedulesList",       C,  C,  NC, NC),
        ("scheduleUpdate",      C,  C,  NC, NC),
        ("scheduleApprove",     NC, NC, NC, NC),
        ("scheduleDelete",      C,  C,  NC, NC),
        ("jobStop",             NC, NC, NC, NC),
        ("jobOutput",           NC, NC, NC, NC),
        ("jobDismiss",          NC, NC, NC, NC),
        ("traceSet",            NC, NC, NC, NC),
        ("dictationStart",      NC, NC, NC, NC),
        ("dictationStop",       NC, NC, NC, NC),
        ("dictationCancel",     NC, NC, NC, NC),
        ("dictationDownload",   NC, NC, NC, NC),
        ("wide",                NC, NC, NC, NC),
    ];

    #[test]
    fn every_variant_is_built_and_in_the_table() {
        let built: Vec<&str> = every_variant().iter().map(label).collect();
        let rows: Vec<&str> = SURFACE.iter().map(|row| row.0).collect();
        assert_eq!(built, rows, "one constructor and one row for each PageMessage variant, in the same order");
    }

    /// What became of one line a client sent: it ends with a request that is always refused (`id` 999), so everything the
    /// line caused (an error to the client, a message for the chat or the handlers) happened before the end is seen.
    fn fate(f: &Fixture, peer: &mut Peer, line: Value) -> Fate {
        peer.send(line);
        peer.send(json!({"type": "saveSettings", "id": 999, "settings": {}}));
        let mut refused = None;
        loop {
            let got = peer.next();
            if got["id"] == 999 {
                break;
            }
            assert!(refused.is_none(), "one line, one answer: {got}");
            refused = Some(got["code"].as_str().map(|code| match code {
                "not-for-clients" => NC,
                "not-for-remote" => NR,
                "unknown-session" => R("unknown-session"),
                other => panic!("an unexpected refusal {other}"),
            }));
        }
        let chat = f.chat.try_recv().is_ok();
        let handled = f.handled.try_recv().is_ok();
        assert!(f.chat.try_recv().is_err() && f.handled.try_recv().is_err(), "one line reaches one place");
        match (refused.flatten(), chat, handled) {
            (Some(refusal), false, false) => refusal,
            (None, true, false) => C,
            (None, false, true) => H,
            other => panic!("neither refused nor delivered once: {other:?}"),
        }
    }

    /// The line a client sends for `message`, bare or inside `session { aa }`. `ChatBash` is never read from JSON: a
    /// client's `!command` is a `piCommand` `bash`.
    fn line(message: &PageMessage, wrapped: bool) -> Value {
        let value = match message {
            PageMessage::ChatBash { id, command } if wrapped => {
                json!({"type": "piCommand", "id": id, "command": "bash", "args": {"command": command}})
            }
            other => serde_json::to_value(other).unwrap_or_else(|_| json!({"type": "chatBash", "id": 77})),
        };
        if wrapped {
            json!({"type": "session", "session": "aa", "message": value})
        } else {
            value
        }
    }

    #[test]
    fn the_whole_surface_a_client_may_reach_is_pinned() {
        let f = fixture();
        let mut local = connect(&f);
        let mut remote = connect(&f);
        remote.send(hello(PROTOCOL, Some(true)));
        remote.next();
        for message in every_variant() {
            let name = label(&message);
            let row = SURFACE.iter().find(|row| row.0 == name).expect(name);
            assert_eq!(fate(&f, &mut local, line(&message, false)), row.1, "{name}, bare, local");
            assert_eq!(fate(&f, &mut remote, line(&message, false)), row.2, "{name}, bare, remote");
            assert_eq!(fate(&f, &mut local, line(&message, true)), row.3, "{name}, in a session, local");
            assert_eq!(fate(&f, &mut remote, line(&message, true)), row.4, "{name}, in a session, remote");
        }
    }

    #[test]
    fn a_remote_client_talks_only_to_sessions_that_exist() {
        let known = Arc::new(Mutex::new(vec![s("aa")]));
        let seen = known.clone();
        let f = fixture_knowing(Arc::new(move |id| seen.lock().unwrap().iter().any(|k| k == id)));
        let mut remote = connect(&f);
        remote.send(hello(PROTOCOL, Some(true)));
        remote.next();
        // a made-up id would start a Pi of its own: refused before it reaches the chat, with the request's id
        remote.send(json!({"type": "session", "session": "zz", "message": {"type": "listModels", "id": 5}}));
        assert_eq!(remote.next(), json!({"type": "error", "id": 5, "code": "unknown-session"}));
        remote.send(json!({"type": "session", "session": "zz", "message": {"type": "chatSend", "text": "hi"}}));
        assert_eq!(remote.next(), json!({"type": "error", "id": 0, "code": "unknown-session"}));
        assert!(f.chat.try_recv().is_err());
        // a known one passes
        remote.send(json!({"type": "session", "session": "aa", "message": {"type": "chatAbort"}}));
        assert!(matches!(f.chat.recv_timeout(Duration::from_secs(5)).unwrap(), PageMessage::Session { .. }));
        // and one the client opened is known afterwards (the service reads sessions.json, where `sessionNew` writes it)
        remote.send(json!({"type": "sessionNew", "id": 6}));
        assert!(matches!(f.chat.recv_timeout(Duration::from_secs(5)).unwrap(), PageMessage::SessionNew { .. }));
        known.lock().unwrap().push(s("bb"));
        remote.send(json!({"type": "session", "session": "bb", "message": {"type": "chatAbort"}}));
        assert!(matches!(f.chat.recv_timeout(Duration::from_secs(5)).unwrap(), PageMessage::Session { .. }));
        // a local terminal and the page are not asked
        let mut local = connect(&f);
        local.send(json!({"type": "session", "session": "zz", "message": {"type": "chatAbort"}}));
        assert!(matches!(f.chat.recv_timeout(Duration::from_secs(5)).unwrap(), PageMessage::Session { .. }));
    }

    #[test]
    fn a_remote_client_cannot_close_a_session_that_does_not_exist() {
        let f = fixture_knowing(Arc::new(|id| id == "aa"));
        let mut remote = connect(&f);
        remote.send(hello(PROTOCOL, Some(true)));
        remote.next();
        // a made-up id would grow the closed set, rewrite sessions.json and broadcast: refused before the chat sees it
        remote.send(json!({"type": "sessionClose", "session": "zz"}));
        assert_eq!(remote.next(), json!({"type": "error", "id": 0, "code": "unknown-session"}));
        assert!(f.chat.try_recv().is_err());
        remote.send(json!({"type": "sessionClose", "session": "aa"}));
        assert!(matches!(
            f.chat.recv_timeout(Duration::from_secs(5)).unwrap(),
            PageMessage::SessionClose { session } if session == "aa"
        ));
        // a local terminal is not asked (it closes the tab of a session the dock may not have loaded yet)
        let mut local = connect(&f);
        local.send(json!({"type": "sessionClose", "session": "zz"}));
        assert!(matches!(f.chat.recv_timeout(Duration::from_secs(5)).unwrap(), PageMessage::SessionClose { .. }));
    }

    #[test]
    fn a_second_hello_cannot_lift_the_remote_mark() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(hello(PROTOCOL, Some(true)));
        assert_eq!(a.next()["type"], "hello");
        a.send(hello(PROTOCOL, Some(false)));
        a.send(secret_save(7));
        assert_eq!(a.next(), json!({"type": "error", "id": 7, "code": "not-for-remote"}));
        assert!(f.handled.try_recv().is_err());
    }

    #[test]
    fn a_hello_after_another_line_does_not_mark_the_client() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(json!({"type": "focus", "focused": true}));
        a.send(hello(PROTOCOL, Some(true)));
        a.send(secret_save(8));
        assert!(matches!(f.handled.recv_timeout(Duration::from_secs(5)).unwrap(), PageMessage::SecretSave { .. }));
    }

    #[test]
    fn a_local_client_without_hello_keeps_working() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(secret_save(5));
        assert!(matches!(f.handled.recv_timeout(Duration::from_secs(5)).unwrap(), PageMessage::SecretSave { .. }));
        a.send(json!({"type": "session", "session": "aa", "message": {"type": "piCommand", "id": 6, "command": "bash", "args": {"command": "pwd"}}}));
        match f.chat.recv_timeout(Duration::from_secs(5)).unwrap() {
            PageMessage::Session { message, .. } => assert!(matches!(*message, PageMessage::ChatBash { .. })),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn clients_may_manage_schedules() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(json!({"type": "schedulesList", "id": 1}));
        a.send(json!({"type": "scheduleUpdate", "id": 2, "schedule": "s-1", "enabled": false}));
        a.send(json!({"type": "scheduleDelete", "id": 3, "schedule": "s-1"}));
        let mut ids = vec![];
        for expected in ["list", "update", "delete"] {
            let message = f.chat.recv_timeout(Duration::from_secs(5)).unwrap();
            let id = match (expected, &message) {
                ("list", PageMessage::SchedulesList { id }) => *id,
                ("update", PageMessage::ScheduleUpdate { id, schedule, enabled }) => {
                    assert_eq!((schedule.as_str(), *enabled), ("s-1", false));
                    *id
                }
                ("delete", PageMessage::ScheduleDelete { id, schedule }) => {
                    assert_eq!(schedule, "s-1");
                    *id
                }
                _ => panic!("{expected}: {message:?}"),
            };
            assert!(![1, 2, 3].contains(&id), "ids are remapped");
            ids.push(id);
        }
        ids.dedup();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn a_session_status_reaches_every_client() {
        let f = fixture();
        let mut elsewhere = connect(&f);
        elsewhere.send(json!({"type": "attach", "session": "bb"}));
        wait_until("attached", || f.hub.watchers().sees("bb"));
        let status = DockMessage::Session {
            session: "aa".into(),
            message: Box::new(DockMessage::SessionStatus { session: "aa".into(), status: "working".into(), at: 1 }),
        };
        assert!(f.hub.route(&status));
        assert_eq!(elsewhere.next()["message"]["status"], "working");
    }

    #[test]
    fn session_messages_reach_only_attached_clients() {
        let f = fixture();
        let mut on = connect(&f);
        let mut off = connect(&f);
        on.send(json!({"type": "attach", "session": "aa"}));
        off.send(json!({"type": "attach", "session": "bb"}));
        wait_until("both attached", || f.hub.watchers().sees("aa") && f.hub.watchers().sees("bb"));
        let event = DockMessage::Session {
            session: "aa".into(),
            message: Box::new(DockMessage::ChatEngine { state: "ready".into() }),
        };
        assert!(f.hub.route(&event), "the page gets every session's messages");
        assert!(f.hub.route(&DockMessage::SessionsChanged));
        assert_eq!(on.next()["message"]["state"], "ready");
        assert_eq!(on.next()["type"], "sessionsChanged");
        assert_eq!(off.next()["type"], "sessionsChanged", "bb's client never got aa's event");
    }

    #[test]
    fn a_client_that_stops_reading_is_dropped_not_waited_for() {
        let f = fixture();
        let _asleep = connect(&f);
        let big = "x".repeat(8192);
        let start = Instant::now();
        for _ in 0..4096 {
            f.hub.route(&DockMessage::Error { id: 0, code: big.clone() });
        }
        assert!(start.elapsed() < Duration::from_secs(3), "routing waited for a client that does not read");
        assert_eq!(f.hub.count(), 0, "the client that stopped reading was dropped");
    }

    #[test]
    fn clients_cannot_send_handler_messages() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(json!({"type": "saveSettings", "id": 4, "settings": {}}));
        let got = a.next();
        assert_eq!(got, json!({"type": "error", "id": 4, "code": "not-for-clients"}));
        assert!(f.chat.try_recv().is_err() && f.handled.try_recv().is_err());
        a.send(json!({"type": "secretSave", "id": 5, "name": "GH_TOKEN", "value": "v", "purpose": ""}));
        let handled = f.handled.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(handled, PageMessage::SecretSave { .. }));
    }

    fn sign_in_start(id: u64) -> Value {
        json!({"type": "startOauth", "id": id})
    }

    fn remembered(f: &Fixture) -> usize {
        f.hub.asked.lock().unwrap().len()
    }

    fn handled_id(f: &Fixture) -> u64 {
        match f.handled.recv_timeout(Duration::from_secs(5)).unwrap() {
            PageMessage::StartOAuth { id }
            | PageMessage::StartMcpOAuth { id, .. }
            | PageMessage::OAuthCode { id, .. }
            | PageMessage::HubUpdate { id, .. } => id,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_oauth_sign_in_keeps_the_clients_id_until_its_result() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(sign_in_start(3));
        let ours = handled_id(&f);
        assert_ne!(ours, 3);
        assert!(f.chat.try_recv().is_err(), "a sign-in is for the handlers, not the chat");
        // the URL, then the result: both reach the asker with its own id, and neither is the page's
        assert!(!f.hub.route(&DockMessage::OAuthUrl { id: ours, url: "https://x".into() }));
        assert_eq!(a.next(), json!({"type": "oauthUrl", "id": 3, "url": "https://x"}));
        assert_eq!(remembered(&f), 1, "the id is kept after the URL");
        assert!(!f.hub.route(&DockMessage::OAuthResult { id: ours, ok: true, error: None }));
        assert_eq!(a.next(), json!({"type": "oauthResult", "id": 3, "ok": true, "error": null}));
        assert_eq!(remembered(&f), 0, "and dropped after the result");
        // a sign-in that fails after its URL ends the same way
        a.send(json!({"type": "startMcpOauth", "id": 4, "service": "notion"}));
        let ours = handled_id(&f);
        assert!(!f.hub.route(&DockMessage::OAuthUrl { id: ours, url: "u".into() }));
        a.next();
        assert!(!f.hub.route(&DockMessage::Error { id: ours, code: "oauth-failed".into() }));
        assert_eq!(a.next(), json!({"type": "error", "id": 4, "code": "oauth-failed"}));
        assert_eq!(remembered(&f), 0);
    }

    #[test]
    fn a_hub_update_answers_line_by_line_to_the_asker_alone_and_its_id_stays_until_the_result() {
        let f = fixture();
        let mut a = connect(&f);
        let mut b = connect(&f);
        // A remote client may ask for it (the restricted key is the one that does).
        a.send(json!({"type": "hello", "protocol": PROTOCOL, "version": "x", "remote": true}));
        a.next();
        a.send(json!({"type": "hubUpdate", "id": 5, "version": "0.22.0"}));
        let ours = handled_id(&f);
        assert_ne!(ours, 5, "the client's id is swapped for the hub's");
        assert!(f.chat.try_recv().is_err(), "an update is for the handlers, not the chat");
        for line in ["- Checking this server", "- Installing Aria"] {
            assert!(!f.hub.route(&DockMessage::HubUpdateProgress { id: ours, line: line.into() }), "not the page's");
            assert_eq!(a.next(), json!({"type": "hubUpdateProgress", "id": 5, "line": line}));
            assert_eq!(remembered(&f), 1, "the id is kept after a line");
        }
        assert!(!f.hub.route(&DockMessage::HubUpdateResult {
            id: ours,
            ok: true,
            version: Some("0.22.0".into()),
            reason: None
        }));
        assert_eq!(a.next(), json!({"type": "hubUpdateResult", "id": 5, "ok": true, "version": "0.22.0"}));
        assert_eq!(remembered(&f), 0, "and dropped after the result");
        // The other client heard none of it.
        b.send(json!({"type": "saveSettings", "id": 999, "settings": {}}));
        assert_eq!(b.next()["id"], 999, "nothing else was queued for the other client");
    }

    #[test]
    fn a_hub_update_that_fails_ends_the_same_way() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(json!({"type": "hubUpdate", "id": 6, "version": "0.22.0"}));
        let ours = handled_id(&f);
        assert!(!f.hub.route(&DockMessage::HubUpdateProgress { id: ours, line: "curl: (22) 404".into() }));
        a.next();
        assert!(!f.hub.route(&DockMessage::HubUpdateResult {
            id: ours,
            ok: false,
            version: None,
            reason: Some("curl: (22) 404".into())
        }));
        assert_eq!(a.next(), json!({"type": "hubUpdateResult", "id": 6, "ok": false, "reason": "curl: (22) 404"}));
        assert_eq!(remembered(&f), 0);
        // A message that is not a version string at all is refused before it reaches anyone.
        a.send(json!({"type": "hubUpdate", "id": 7, "version": 22}));
        assert_eq!(a.next(), json!({"type": "error", "id": 7, "code": "not-for-clients"}));
        assert!(f.handled.try_recv().is_err());
    }

    #[test]
    fn a_pasted_address_reaches_the_handlers_and_its_refusal_the_client() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(sign_in_start(3));
        let start = handled_id(&f);
        f.hub.route(&DockMessage::OAuthUrl { id: start, url: "u".into() });
        a.next();
        a.send(json!({"type": "oauthCode", "id": 4, "service": "anthropic", "redirect": "nonsense"}));
        match f.handled.recv_timeout(Duration::from_secs(5)).unwrap() {
            PageMessage::OAuthCode { id, service, redirect } => {
                assert_ne!(id, 4);
                assert_eq!((service.as_str(), redirect.as_str()), ("anthropic", "nonsense"));
                assert!(!f.hub.route(&DockMessage::Error { id, code: "bad-redirect".into() }));
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(a.next(), json!({"type": "error", "id": 4, "code": "bad-redirect"}));
        // the sign-in is still waiting for its result
        assert_eq!(remembered(&f), 1);
        assert!(f.chat.try_recv().is_err());
    }

    #[test]
    fn a_paste_that_the_sign_in_took_leaves_nothing_behind_once_the_sign_in_ends() {
        let f = fixture();
        let mut a = connect(&f);
        let mut b = connect(&f);
        a.send(sign_in_start(3));
        let start = handled_id(&f);
        a.send(json!({"type": "oauthCode", "id": 4, "service": "anthropic", "redirect": "code=a&state=s"}));
        handled_id(&f);
        b.send(json!({"type": "oauthCode", "id": 4, "service": "anthropic", "redirect": "code=a&state=s"}));
        handled_id(&f);
        assert_eq!(remembered(&f), 3);
        // a good paste has no reply of its own: the result of the sign-in ends it, and what its client pasted with it
        f.hub.route(&DockMessage::OAuthResult { id: start, ok: true, error: None });
        a.next();
        assert_eq!(remembered(&f), 1, "only the other client's paste is left");
        drop(b);
        wait_until("the other client's entries are gone", || remembered(&f) == 0);
    }

    #[test]
    fn a_remote_client_may_sign_in_but_still_not_save_a_secret() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(hello(PROTOCOL, Some(true)));
        a.next();
        a.send(sign_in_start(3));
        handled_id(&f);
        a.send(json!({"type": "oauthCode", "id": 4, "service": "notion", "redirect": "code=a&state=s"}));
        handled_id(&f);
        a.send(secret_save(5));
        assert_eq!(a.next(), json!({"type": "error", "id": 5, "code": "not-for-remote"}));
        assert!(f.handled.try_recv().is_err());
    }

    #[test]
    fn a_refused_request_leaves_no_id_behind() {
        let f = fixture();
        let mut a = connect(&f);
        a.send(json!({"type": "saveSettings", "id": 4, "settings": {}}));
        assert_eq!(a.next(), json!({"type": "error", "id": 4, "code": "not-for-clients"}));
        assert_eq!(remembered(&f), 0);
        let mut r = connect(&f);
        r.send(hello(PROTOCOL, Some(true)));
        r.next();
        r.send(secret_save(5));
        assert_eq!(r.next(), json!({"type": "error", "id": 5, "code": "not-for-remote"}));
        assert_eq!(remembered(&f), 0);
    }

    #[test]
    fn a_focused_client_counts_as_seeing_its_session() {
        let f = fixture();
        let mut a = connect(&f);
        assert!(!f.hub.watchers().sees("aa"));
        a.send(json!({"type": "attach", "session": "aa"}));
        wait_until("attached", || f.hub.watchers().sees("aa"));
        a.send(json!({"type": "focus", "focused": false}));
        wait_until("unfocused", || !f.hub.watchers().sees("aa"));
        a.send(json!({"type": "focus", "focused": true}));
        wait_until("focused again", || f.hub.watchers().sees("aa"));
        drop(a);
        wait_until("gone", || !f.hub.watchers().sees("aa"));
    }
}
