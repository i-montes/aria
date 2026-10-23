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
//!   or a `!command` (`not-for-remote`), and a later `hello` cannot lift the mark.
//! - Sign-ins: a client (local or remote) may start Claude's or Notion's (`startOauth`, `startMcpOauth`) and paste the
//!   address its browser could not load (`oauthCode`); they go to the handlers, not the chat. A sign-in answers twice
//!   (`oauthUrl`, then `oauthResult` or `error`), so its id stays until the last of them. Security: a remote client can
//!   therefore re-bind the hub's Claude or Notion account, so a stolen client key could point them at the attacker's;
//!   accepted, since that key can already chat with the model and use those accounts. Pasted text holds a code: never
//!   logged.

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

/// What goes to the handlers and not the chat: a secret to keep, and the sign-ins.
fn for_handlers(message: &PageMessage) -> bool {
    matches!(
        message,
        PageMessage::SecretSave { .. }
            | PageMessage::StartOAuth { .. }
            | PageMessage::StartMcpOAuth { .. }
            | PageMessage::OAuthCode { .. }
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
    clients: Watchers,
    next: Arc<AtomicU64>,
    /// The ids given to clients' requests. A reply whose id is not here is the page's.
    asked: Arc<Mutex<HashMap<u64, Ask>>>,
    next_ask: Arc<AtomicU64>,
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
}

/// The numeric request id of a message (at its top, or inside a `session` wrapper), as a mutable slot.
fn id_slot(value: &mut Value) -> Option<&mut Value> {
    let slot =
        if value["type"] == "session" { value.get_mut("message")?.get_mut("id")? } else { value.get_mut("id")? };
    slot.is_u64().then_some(slot)
}

impl Hub {
    pub fn new(chat: mpsc::Sender<PageMessage>, handlers: Arc<dyn Fn(PageMessage) + Send + Sync>) -> Hub {
        Hub {
            chat: Arc::new(Mutex::new(chat)),
            handlers,
            clients: Watchers::default(),
            next: Arc::new(AtomicU64::new(1)),
            asked: Arc::default(),
            next_ask: Arc::new(AtomicU64::new(CLIENT_BASE)),
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
            Some("attach") => return self.set(number, |c| c.session = value["session"].as_str().map(str::to_string)),
            Some("detach") => return self.set(number, |c| c.session = None),
            Some("focus") => return self.set(number, |c| c.focused = value["focused"].as_bool().unwrap_or(true)),
            _ => {}
        }
        let kind = match value["type"].as_str() {
            Some("startOauth" | "startMcpOauth") => AskKind::SignIn,
            Some("oauthCode") => AskKind::Code,
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
        let message = serde_json::from_value::<PageMessage>(value).ok().map(|message| match message {
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
        });
        match message {
            Some(message) if remote && refused_for_remote(&message) => refuse("not-for-remote"),
            Some(message) if allowed(&message) => {
                let _ = self.chat.lock().unwrap().send(message);
            }
            Some(message) if for_handlers(&message) => (self.handlers)(message),
            _ => refuse("not-for-clients"),
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
        if let Some(slot) = id_slot(&mut value) {
            let id = slot.as_u64().unwrap_or(0);
            let asked = {
                let mut asked = self.asked.lock().unwrap();
                // A sign-in answers twice: its URL leaves the id in place for the result (or error) that follows. Any
                // other reply is the last: it also ends what the client pasted into that sign-in.
                let url_of_a_sign_in = is_url && asked.get(&id).is_some_and(|a| a.kind == AskKind::SignIn);
                let found = if url_of_a_sign_in {
                    asked.get(&id).map(|a| Ask { client: a.client, own: a.own, kind: a.kind })
                } else {
                    asked.remove(&id)
                };
                if let Some(ask) = found.as_ref().filter(|a| a.kind == AskKind::SignIn && !url_of_a_sign_in) {
                    asked.retain(|_, other| !(other.client == ask.client && other.kind == AskKind::Code));
                }
                found
            };
            if let Some(Ask { client: number, own, .. }) = asked {
                *slot = json!(own);
                self.send_to(number, &value);
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
        let (chat_tx, chat) = mpsc::channel();
        let (handled_tx, handled) = mpsc::channel();
        let handled_tx = std::sync::Mutex::new(handled_tx);
        let hub = Hub::new(
            chat_tx,
            std::sync::Arc::new(move |m| {
                let _ = handled_tx.lock().unwrap().send(m);
            }),
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
            | PageMessage::OAuthCode { id, .. } => id,
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
