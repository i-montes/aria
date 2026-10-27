//! `aria link`: this computer as a **machine** of a hub (docs/superpowers/specs/2026-10-27-machine-link-design.md). A thread of
//! the machine's own service (the dock's or `aria serve`'s) keeps one ssh connection open to the hub,
//! `ssh -i <key> <hub> aria hub-link machine <name>`, whose forced command on the hub is the bridge to `machines.sock`
//! (machines.rs is the hub's side), and speaks JSON lines through it:
//!
//! - `link-hello { protocol, name, os, version }` first; the hub answers `link-hello`, or an `error` (`protocol-mismatch`,
//!   `name-mismatch`) that stops the retrying: nothing a reconnect could mend.
//! - `link-sessions { sessions, statuses }` after the hello and whenever this machine's list or statuses change. Nothing of the
//!   service's internals is read: the thread is one more (remote) client of the service's own `clients.sock`, asks
//!   `sessionsLoad` and follows `sessionsChanged` and the status events, and sends what it was told, trimmed to what the hub
//!   keeps (500 entries, well under its 2 MiB line).
//! - `{"c": n, "line": ...}` from the hub is a line of hub client `n`: the first one for an `n` opens **one more local
//!   connection** to `clients.sock` with `hello { remote: true }`, so the machine applies exactly the restrictions of a remote
//!   client (no `secretSave`, no `!command`, no folder chosen by the client, only sessions it has), and what that connection
//!   produces goes back as `{"c": n, "line": ...}`. `{"c": n, "close": true}` closes it, and its end says the same to the hub.
//!   At most 32 at a time (`too-many-clients`).
//! - `link-ping` and `link-pong` are the heartbeat: a ping every `Timing::ping`, a link silent for `Timing::silent` is dropped.
//!   A dropped link is opened again after 2, 4, 8 ... up to 60 seconds (starting over once one has lasted a minute).
//!
//! The thread runs only while `~/.aria/link.json` exists (`{"hub": "user@host", "name": "laptop", "key": "/path/key"}`, mode 600,
//! written by `aria hub init`): it is looked for every ten seconds, so a link starts without a restart and ends when the file
//! goes. What the link is doing is written to `~/.aria/link.state` for `aria hub status`.

use crate::machines::valid_name;
use crate::stdio_bridge::{next_line, Next};
use aria_core::bridge::PROTOCOL;
use aria_core::json_file::write_bytes_atomic;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::{self, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The file that makes this computer a machine, in Aria's home.
pub const CONFIG_FILE: &str = "link.json";
/// What the link is doing, for `aria hub status`.
pub const STATE_FILE: &str = "link.state";

/// The longest line taken from the hub or from a local client: a reply can hold a whole conversation.
const MAX_LINE: usize = 16 * 1024 * 1024;
/// How many hub clients one link serves at a time.
const MAX_CLIENTS: usize = 32;
/// What the hub keeps of this machine's list: its own cap, and a line well under the 2 MiB it refuses to go past.
const MAX_SESSIONS: usize = 500;
const MAX_SESSIONS_BYTES: usize = 1536 * 1024;
/// A local line is written within this long, or its client is closed: a service that does not read is not waited for.
const LOCAL_WRITE: Duration = Duration::from_secs(5);

/// How long things take: injectable, so the tests do not wait.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// A `link-ping` is sent this often.
    pub ping: Duration,
    /// A hub that says nothing for this long is gone.
    pub silent: Duration,
    /// How long the hub has to answer the hello.
    pub handshake: Duration,
    /// The first wait before opening a dropped link again, which doubles up to `backoff_max`.
    pub backoff_first: Duration,
    pub backoff_max: Duration,
    /// A link that lasted this long starts the backoff over.
    pub stable_after: Duration,
    /// At most one `link-sessions` per this long (the first goes at once).
    pub coalesce: Duration,
    /// How often `link.json` is looked for.
    pub poll_config: Duration,
}

impl Default for Timing {
    fn default() -> Timing {
        Timing {
            ping: Duration::from_secs(15),
            silent: Duration::from_secs(45),
            handshake: Duration::from_secs(30),
            backoff_first: Duration::from_secs(2),
            backoff_max: Duration::from_secs(60),
            stable_after: Duration::from_secs(60),
            coalesce: Duration::from_millis(500),
            poll_config: Duration::from_secs(10),
        }
    }
}

/// The wait before opening a link again after it dropped `attempt` times in a row (0 for the first): 2, 4, 8 ... 60 seconds.
pub fn backoff(attempt: u32, timing: &Timing) -> Duration {
    let factor = 1u32.checked_shl(attempt.min(20)).unwrap_or(u32::MAX);
    timing.backoff_first.saturating_mul(factor).min(timing.backoff_max)
}

// ---- link.json ------------------------------------------------------------------------------------------------------

/// Who this computer is to the hub and how it gets there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkConfig {
    /// What `ssh` is given: `user@host`, or an alias of the owner's ssh configuration.
    pub hub: String,
    /// This machine's name on the hub: the one its key's forced command carries.
    pub name: String,
    /// The private key, an absolute path.
    pub key: String,
}

impl LinkConfig {
    pub fn path(root: &Path) -> PathBuf {
        root.join(CONFIG_FILE)
    }

    /// The config in `text`, or what is wrong with it in a plain sentence.
    pub fn parse(text: &str) -> Result<LinkConfig, String> {
        let config: LinkConfig = serde_json::from_str(text).map_err(|error| format!("{CONFIG_FILE}: {error}"))?;
        if config.hub.is_empty()
            || config.hub.chars().any(|c| c.is_control() || c.is_whitespace())
            || crate::connect::check_host(&config.hub).is_err()
        {
            return Err(format!(
                "{CONFIG_FILE}: \"hub\" must be the host ssh is given (user@host), got {:?}",
                config.hub
            ));
        }
        if !valid_name(&config.name) {
            return Err(format!(
                "{CONFIG_FILE}: \"name\" must be lowercase letters, digits and dashes (at most 31), got {:?}",
                config.name
            ));
        }
        if !Path::new(&config.key).is_absolute() {
            return Err(format!("{CONFIG_FILE}: \"key\" must be an absolute path, got {:?}", config.key));
        }
        Ok(config)
    }

    /// The config in `root`: none when there is no file, an error when it is there and wrong (said, and the link does not start).
    pub fn load(root: &Path) -> Result<Option<LinkConfig>, String> {
        match std::fs::read_to_string(LinkConfig::path(root)) {
            Ok(text) => LinkConfig::parse(&text).map(Some),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("{CONFIG_FILE}: {error}")),
        }
    }

    /// Written whole and atomically, readable by its owner alone (it names a key file).
    pub fn save(&self, root: &Path) -> io::Result<()> {
        aria_core::paths::AriaPaths::new(root.to_path_buf()).ensure_root()?;
        let mut text = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        text.push('\n');
        write_bytes_atomic(&LinkConfig::path(root), text.as_bytes(), 0o600)
    }
}

/// A machine's name for this computer: its host name, lowercased, every run of anything but letters and digits turned into one
/// dash, trimmed of dashes and cut to 31 characters; `computer` when nothing usable is left.
pub fn machine_name_from(host: &str) -> String {
    let mut name = String::new();
    for c in host.chars() {
        if c.is_ascii_alphanumeric() {
            name.push(c.to_ascii_lowercase());
        } else if !name.ends_with('-') && !name.is_empty() {
            name.push('-');
        }
    }
    let name: String = name.trim_matches('-').chars().take(31).collect();
    let name = name.trim_end_matches('-').to_string();
    if valid_name(&name) {
        name
    } else {
        "computer".to_string()
    }
}

/// This computer's host name (`hostname`, else `HOSTNAME`), for a default machine name.
pub fn this_computer() -> String {
    Command::new("hostname")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|name| !name.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_default()
}

// ---- link.state -----------------------------------------------------------------------------------------------------

/// What the link is doing, as `aria hub status` shows it.
#[derive(Debug, Clone, PartialEq)]
pub enum State {
    Connecting,
    Connected,
    /// It could not link, or the link dropped: the plain reason, and it is tried again unless it is not a thing to retry.
    Error(String),
}

#[derive(Debug, Clone)]
pub struct StateFile {
    path: PathBuf,
}

impl StateFile {
    pub fn new(root: &Path) -> StateFile {
        StateFile { path: root.join(STATE_FILE) }
    }

    fn write(&self, state: &State) {
        let (kind, reason) = match state {
            State::Connecting => ("connecting", None),
            State::Connected => ("connected", None),
            State::Error(reason) => ("error", Some(reason.as_str())),
        };
        let mut value = json!({"state": kind, "pid": std::process::id()});
        if let Some(reason) = reason {
            value["reason"] = json!(reason);
        }
        let _ = write_bytes_atomic(&self.path, format!("{value}\n").as_bytes(), 0o600);
    }

    fn clear(&self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// What a `link.state` said and whether the service that wrote it still runs.
#[derive(Debug, Clone, PartialEq)]
pub struct Reported {
    pub state: String,
    pub reason: Option<String>,
    pub alive: bool,
}

/// The state this computer's link last wrote, if any (a pid that is gone means a service that is not running: no link).
pub fn reported(root: &Path) -> Option<Reported> {
    let text = std::fs::read_to_string(root.join(STATE_FILE)).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    let pid = value["pid"].as_u64()? as i32;
    // SAFETY: kill(pid, 0) only asks whether the process exists.
    let alive = unsafe { libc::kill(pid, 0) == 0 } || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
    Some(Reported {
        state: value["state"].as_str()?.to_string(),
        reason: value["reason"].as_str().map(str::to_string),
        alive,
    })
}

// ---- the connection to the hub --------------------------------------------------------------------------------------

/// An open way to the hub: lines in, lines out, and a way to end it. The real one is the ssh process; the tests give a socket.
pub struct Connection {
    pub reader: Box<dyn Read + Send>,
    pub writer: Box<dyn Write + Send>,
    /// Ends the connection (kills ssh), which ends the reader.
    pub close: Box<dyn Fn() + Send + Sync>,
    /// What went wrong, in a sentence, when the connection ended without the hub saying why (ssh's last line).
    pub why: Box<dyn Fn() -> String + Send + Sync>,
}

pub trait Connector: Send + Sync {
    fn connect(&self, config: &LinkConfig) -> io::Result<Connection>;
}

/// The arguments of the ssh that opens the link: the key and no other, nothing that could ask a question, the host checked
/// (a host that starts with `-` is an option of ssh), and the hub's forced command named once more for the ssh that sends it.
pub fn ssh_args(config: &LinkConfig) -> Result<Vec<String>, String> {
    crate::connect::check_host(&config.hub).map_err(|_| format!("{:?} is not a host", config.hub))?;
    if !valid_name(&config.name) {
        return Err(format!("{:?} is not a machine name", config.name));
    }
    Ok([
        "-T",
        "-i",
        config.key.as_str(),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        config.hub.as_str(),
        "~/.local/bin/aria",
        "hub-link",
        "machine",
        config.name.as_str(),
    ]
    .iter()
    .map(|arg| arg.to_string())
    .collect())
}

/// ssh, as `aria connect` runs it (`ARIA_SSH` names another program).
pub struct SshConnector;

impl Connector for SshConnector {
    fn connect(&self, config: &LinkConfig) -> io::Result<Connection> {
        let args = ssh_args(config).map_err(io::Error::other)?;
        let mut child = Command::new(crate::connect::ssh_program())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let (stdin, stdout, stderr) = (child.stdin.take(), child.stdout.take(), child.stderr.take());
        let (Some(stdin), Some(stdout), Some(stderr)) = (stdin, stdout, stderr) else {
            let _ = child.kill();
            return Err(io::Error::other("ssh has no pipes"));
        };
        // ssh says why a link could not open on its stderr; the last line of it is what `aria hub status` shows.
        let last = Arc::new(Mutex::new(String::new()));
        {
            let last = last.clone();
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stderr);
                while let Ok(Next::Line(line)) = next_line(&mut reader, 4096) {
                    let text = String::from_utf8_lossy(&line).trim().to_string();
                    if !text.is_empty() {
                        *last.lock().unwrap() = text;
                    }
                }
            });
        }
        let child = Arc::new(Mutex::new(child));
        let closer = child.clone();
        let reaper = child.clone();
        std::thread::spawn(move || {
            // The process is waited for when it ends by itself, so none is left a zombie.
            loop {
                std::thread::sleep(Duration::from_millis(200));
                match reaper.lock().unwrap().try_wait() {
                    Ok(Some(_)) | Err(_) => break,
                    Ok(None) => {}
                }
            }
        });
        Ok(Connection {
            reader: Box::new(stdout),
            writer: Box::new(stdin),
            close: Box::new(move || {
                let mut child = closer.lock().unwrap();
                let _ = child.kill();
                let _ = child.wait();
            }),
            why: Box::new(move || {
                let said = last.lock().unwrap().clone();
                if said.is_empty() {
                    "the server did not answer".to_string()
                } else {
                    said
                }
            }),
        })
    }
}

// ---- one link -------------------------------------------------------------------------------------------------------

/// How one connection ended.
#[derive(Debug, PartialEq)]
pub enum End {
    /// This computer stopped it (the service quits, the config went or changed).
    Stopped,
    /// The hub refused it for a reason a reconnect cannot mend: not tried again.
    Fatal(String),
    /// It dropped, or never opened: tried again after the backoff.
    Dropped { lived: Duration, why: String },
}

/// What the threads of one link tell its loop.
enum Msg {
    Hub(Value),
    HubGone,
    /// A line a local connection (token) produced for hub client `n`.
    Local {
        client: u64,
        token: u64,
        line: Value,
    },
    LocalGone {
        client: u64,
        token: u64,
    },
    /// The observer's view of this machine's sessions, to push.
    Sessions {
        sessions: Vec<Value>,
        statuses: Map<String, Value>,
    },
    ObserverGone,
}

/// A local connection for one hub client.
struct Local {
    stream: UnixStream,
    token: u64,
}

/// `sessions` trimmed to what the hub keeps: the most recent `MAX_SESSIONS`, and, if that is still too long a line, without
/// each entry's `segments` (the tab's own bulk) and then with fewer entries.
pub fn fit_sessions(mut sessions: Vec<Value>) -> Vec<Value> {
    if sessions.len() > MAX_SESSIONS {
        sessions.drain(..sessions.len() - MAX_SESSIONS);
    }
    let size = |sessions: &[Value]| serde_json::to_string(sessions).map(|text| text.len()).unwrap_or(usize::MAX);
    if size(&sessions) > MAX_SESSIONS_BYTES {
        for entry in &mut sessions {
            if let Some(object) = entry.as_object_mut() {
                object.remove("segments");
            }
        }
    }
    while size(&sessions) > MAX_SESSIONS_BYTES && sessions.len() > 1 {
        sessions.remove(0);
    }
    sessions
}

/// The session list and statuses of the service's own `clients.sock`, followed. One local remote client that asks `sessionsLoad`
/// whenever `sessionsChanged` is said, remembers each `sessionStatus`, and gives the loop what it has (the first time at once,
/// then at most one push per `coalesce`).
fn observe(local: &Path, over: Arc<AtomicBool>, tx: Sender<Msg>, coalesce: Duration) {
    let Ok(stream) = UnixStream::connect(local) else {
        let _ = tx.send(Msg::ObserverGone);
        return;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(100)));
    let (Ok(read_half), Ok(mut write)) = (stream.try_clone(), stream.try_clone()) else {
        let _ = tx.send(Msg::ObserverGone);
        return;
    };
    let say = |write: &mut UnixStream, value: Value| write.write_all(format!("{value}\n").as_bytes());
    if say(
        &mut write,
        json!({"type": "hello", "protocol": PROTOCOL, "version": env!("CARGO_PKG_VERSION"), "remote": true}),
    )
    .and_then(|_| say(&mut write, json!({"type": "sessionsLoad", "id": 1})))
    .is_err()
    {
        let _ = tx.send(Msg::ObserverGone);
        return;
    }
    let mut reader = BufReader::new(read_half);
    let mut sessions: Vec<Value> = Vec::new();
    let mut statuses: Map<String, Value> = Map::new();
    let mut have_list = false;
    let mut dirty = false;
    let mut pushed: Option<Instant> = None;
    let mut asked = 1u64;
    while !over.load(Ordering::SeqCst) {
        match next_line(&mut reader, MAX_LINE) {
            Ok(Next::Line(line)) => {
                let Ok(value) = serde_json::from_slice::<Value>(&line) else { continue };
                match value["type"].as_str() {
                    Some("sessions") => {
                        sessions = value["sessions"].as_array().cloned().unwrap_or_default();
                        have_list = true;
                        dirty = true;
                    }
                    Some("sessionsChanged") => {
                        asked += 1;
                        if say(&mut write, json!({"type": "sessionsLoad", "id": asked})).is_err() {
                            break;
                        }
                    }
                    Some("sessionRemoved") => {
                        if let Some(id) = value["session"].as_str() {
                            statuses.remove(id);
                            sessions.retain(|entry| entry["id"] != id);
                            dirty = true;
                        }
                    }
                    Some("session") if value["message"]["type"] == "sessionStatus" => {
                        if let (Some(id), Some(status), Some(at)) = (
                            value["session"].as_str(),
                            value["message"]["status"].as_str(),
                            value["message"]["at"].as_i64(),
                        ) {
                            statuses.insert(id.to_string(), json!({"status": status, "at": at}));
                            dirty = true;
                        }
                    }
                    _ => {}
                }
            }
            Ok(Next::Eof | Next::TooLong) => break,
            // The read timeout: nothing came, so only the push is looked at.
            Err(error) if matches!(error.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
            Err(_) => break,
        }
        if dirty && have_list && pushed.is_none_or(|at| at.elapsed() >= coalesce) {
            let known: std::collections::HashSet<&str> =
                sessions.iter().filter_map(|entry| entry["id"].as_str()).collect();
            let wanted: Map<String, Value> = statuses
                .iter()
                .filter(|(id, _)| known.contains(id.as_str()))
                .map(|(id, s)| (id.clone(), s.clone()))
                .collect();
            if tx.send(Msg::Sessions { sessions: fit_sessions(sessions.clone()), statuses: wanted }).is_err() {
                return;
            }
            dirty = false;
            pushed = Some(Instant::now());
        }
    }
    if !over.load(Ordering::SeqCst) {
        let _ = tx.send(Msg::ObserverGone);
    }
}

/// One link, from the connection to its end.
#[allow(clippy::too_many_arguments)]
pub fn link_once(
    config: &LinkConfig,
    connector: &dyn Connector,
    local: &Path,
    timing: Timing,
    stop: &AtomicBool,
    state: &StateFile,
) -> End {
    let began = Instant::now();
    let dropped = |why: String| End::Dropped { lived: began.elapsed(), why };
    state.write(&State::Connecting);
    let connection = match connector.connect(config) {
        Ok(connection) => connection,
        Err(error) => return dropped(format!("could not start ssh: {error}")),
    };
    let Connection { reader, writer, close, why } = connection;
    let close = Arc::new(close);
    let over = Arc::new(AtomicBool::new(false));
    let (tx, rx): (Sender<Msg>, Receiver<Msg>) = mpsc::channel();

    // The hub's lines, and the lines to the hub through a queue: a slow ssh must not stop the loop that watches the silence.
    {
        let tx = tx.clone();
        let mut reader = BufReader::new(reader);
        std::thread::spawn(move || {
            while let Ok(Next::Line(line)) = next_line(&mut reader, MAX_LINE) {
                let Ok(value) = serde_json::from_slice::<Value>(&line) else { continue };
                if tx.send(Msg::Hub(value)).is_err() {
                    return;
                }
            }
            let _ = tx.send(Msg::HubGone);
        });
    }
    let (out, queue): (SyncSender<String>, Receiver<String>) = mpsc::sync_channel(1024);
    {
        let (close, mut writer) = (close.clone(), writer);
        std::thread::spawn(move || {
            for line in queue {
                if writer
                    .write_all(line.as_bytes())
                    .and_then(|_| writer.write_all(b"\n"))
                    .and_then(|_| writer.flush())
                    .is_err()
                {
                    break;
                }
            }
            // The queue ended or a write failed: ending the connection is what tells the loop.
            close();
        });
    }
    let send = |value: &Value| out.try_send(value.to_string()).is_ok();

    let finish = |end: End, locals: &mut HashMap<u64, Local>| {
        over.store(true, Ordering::SeqCst);
        for (_, local) in locals.drain() {
            let _ = local.stream.shutdown(std::net::Shutdown::Both);
        }
        close();
        end
    };
    let mut locals: HashMap<u64, Local> = HashMap::new();

    // The hello, and the hub's answer.
    let hello = json!({
        "type": "link-hello", "protocol": PROTOCOL, "name": config.name,
        "os": if cfg!(target_os = "macos") { "macos" } else { "linux" },
        "version": env!("CARGO_PKG_VERSION"),
    });
    if !send(&hello) {
        return finish(dropped("could not talk to ssh".to_string()), &mut locals);
    }
    let deadline = Instant::now() + timing.handshake;
    loop {
        if stop.load(Ordering::SeqCst) {
            return finish(End::Stopped, &mut locals);
        }
        let wait = deadline.saturating_duration_since(Instant::now()).min(Duration::from_millis(100));
        match rx.recv_timeout(wait) {
            Ok(Msg::Hub(line)) => match line["type"].as_str() {
                // The hub says its own protocol: another one is refused (it follows with the error, and closes).
                Some("link-hello") if line["protocol"].as_u64().is_some_and(|theirs| theirs != u64::from(PROTOCOL)) => {
                    return finish(
                        End::Fatal(
                            "the server speaks another version of the protocol: update Aria here or on the server"
                                .to_string(),
                        ),
                        &mut locals,
                    );
                }
                Some("link-hello") => break,
                Some("error") => {
                    let code = line["code"].as_str().unwrap_or("unknown");
                    let end = match code {
                        "protocol-mismatch" => End::Fatal(
                            "the server speaks another version of the protocol: update Aria here or on the server"
                                .to_string(),
                        ),
                        "name-mismatch" => End::Fatal(
                            "the server says this key belongs to another machine name: run aria hub init again"
                                .to_string(),
                        ),
                        other => dropped(format!("the server refused the link ({other})")),
                    };
                    return finish(end, &mut locals);
                }
                _ => {}
            },
            Ok(Msg::HubGone) => return finish(dropped(why()), &mut locals),
            Ok(_) => {}
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return finish(dropped(why()), &mut locals),
        }
        if Instant::now() >= deadline {
            return finish(dropped("the server did not answer the hello".to_string()), &mut locals);
        }
    }
    state.write(&State::Connected);

    // This machine's own list, followed, and pushed.
    {
        let (over, tx, local) = (over.clone(), tx.clone(), local.to_path_buf());
        std::thread::spawn(move || observe(&local, over, tx, timing.coalesce));
    }

    let tokens = AtomicU64::new(1);
    let step = (timing.ping.min(timing.silent) / 10).clamp(Duration::from_millis(5), Duration::from_millis(100));
    let mut heard = Instant::now();
    let mut pinged = Instant::now();
    loop {
        if stop.load(Ordering::SeqCst) {
            return finish(End::Stopped, &mut locals);
        }
        match rx.recv_timeout(step) {
            Ok(Msg::Hub(line)) => {
                heard = Instant::now();
                if let Some(client) = line["c"].as_u64() {
                    if line["close"] == true {
                        if let Some(local) = locals.remove(&client) {
                            let _ = local.stream.shutdown(std::net::Shutdown::Both);
                        }
                    } else if let Some(inner) = line.get("line").filter(|inner| inner.is_object()) {
                        if !locals.contains_key(&client) {
                            if locals.len() >= MAX_CLIENTS {
                                send(
                                    &json!({"c": client, "line": {"type": "error", "id": 0, "code": "too-many-clients"}}),
                                );
                                continue;
                            }
                            let token = tokens.fetch_add(1, Ordering::SeqCst);
                            match open_local(local, client, token, tx.clone()) {
                                Ok(stream) => {
                                    locals.insert(client, Local { stream, token });
                                }
                                Err(_) => {
                                    send(
                                        &json!({"c": client, "line": {"type": "error", "id": 0, "code": "machine-unavailable"}}),
                                    );
                                    send(&json!({"c": client, "close": true}));
                                    continue;
                                }
                            }
                        }
                        let failed = locals
                            .get_mut(&client)
                            .is_some_and(|local| local.stream.write_all(format!("{inner}\n").as_bytes()).is_err());
                        if failed {
                            if let Some(local) = locals.remove(&client) {
                                let _ = local.stream.shutdown(std::net::Shutdown::Both);
                            }
                            send(&json!({"c": client, "close": true}));
                        }
                    }
                    continue;
                }
                match line["type"].as_str() {
                    Some("link-ping") => {
                        send(&json!({"type": "link-pong"}));
                    }
                    // A hub that refuses us after its hello says why before it closes.
                    Some("error") if line["code"] == "protocol-mismatch" => {
                        return finish(
                            End::Fatal(
                                "the server speaks another version of the protocol: update Aria here or on the server"
                                    .to_string(),
                            ),
                            &mut locals,
                        );
                    }
                    Some("error") if line["code"] == "name-mismatch" => {
                        return finish(
                            End::Fatal(
                                "the server says this key belongs to another machine name: run aria hub init again"
                                    .to_string(),
                            ),
                            &mut locals,
                        );
                    }
                    _ => {}
                }
            }
            Ok(Msg::HubGone) => return finish(dropped(why()), &mut locals),
            Ok(Msg::Local { client, token, line }) => {
                if locals.get(&client).is_some_and(|local| local.token == token) && line["type"] != "hello" {
                    send(&json!({"c": client, "line": line}));
                }
            }
            Ok(Msg::LocalGone { client, token }) => {
                if locals.get(&client).is_some_and(|local| local.token == token) {
                    locals.remove(&client);
                    send(&json!({"c": client, "close": true}));
                }
            }
            Ok(Msg::Sessions { sessions, statuses }) => {
                if !send(&json!({"type": "link-sessions", "sessions": sessions, "statuses": statuses})) {
                    return finish(dropped("the server does not keep up".to_string()), &mut locals);
                }
            }
            Ok(Msg::ObserverGone) => {
                return finish(dropped("this computer's Aria stopped answering".to_string()), &mut locals);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return finish(dropped(why()), &mut locals),
        }
        if pinged.elapsed() >= timing.ping {
            send(&json!({"type": "link-ping"}));
            pinged = Instant::now();
        }
        if heard.elapsed() >= timing.silent {
            return finish(dropped("the server stopped answering".to_string()), &mut locals);
        }
    }
}

/// One more connection to the service's own `clients.sock` as a remote client, for hub client `client`: its first line is the
/// hello that marks it remote, what it says is handed to the loop with `token` (so an old connection is told from a new one for
/// the same client), and its end is said too.
fn open_local(local: &Path, client: u64, token: u64, tx: Sender<Msg>) -> io::Result<UnixStream> {
    let mut stream = UnixStream::connect(local)?;
    stream.set_write_timeout(Some(LOCAL_WRITE))?;
    let read_half = stream.try_clone()?;
    stream.write_all(
        format!(
            "{}\n",
            json!({"type": "hello", "protocol": PROTOCOL, "version": env!("CARGO_PKG_VERSION"), "remote": true})
        )
        .as_bytes(),
    )?;
    std::thread::spawn(move || {
        let mut reader = BufReader::new(read_half);
        while let Ok(Next::Line(line)) = next_line(&mut reader, MAX_LINE) {
            let Ok(value) = serde_json::from_slice::<Value>(&line) else { continue };
            if tx.send(Msg::Local { client, token, line: value }).is_err() {
                return;
            }
        }
        let _ = tx.send(Msg::LocalGone { client, token });
    });
    Ok(stream)
}

// ---- the thread of the service --------------------------------------------------------------------------------------

/// Sleeps for `total`, in slices, and stops early when `stop` is set.
fn sleep_unless(total: Duration, stop: &AtomicBool) {
    let until = Instant::now() + total;
    while !stop.load(Ordering::SeqCst) && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(20).min(until.saturating_duration_since(Instant::now())));
    }
}

/// The link of one config: opened, and opened again after it drops, until it is stopped or the hub refuses it for good.
fn run_link(
    config: LinkConfig,
    connector: Arc<dyn Connector>,
    local: PathBuf,
    timing: Timing,
    stop: Arc<AtomicBool>,
    state: StateFile,
) {
    let mut attempt = 0u32;
    while !stop.load(Ordering::SeqCst) {
        match link_once(&config, connector.as_ref(), &local, timing, &stop, &state) {
            End::Stopped => break,
            End::Fatal(reason) => {
                crate::note(&format!("not linked to {}: {reason}", config.hub));
                state.write(&State::Error(reason));
                return;
            }
            End::Dropped { lived, why } => {
                if lived >= timing.stable_after {
                    attempt = 0;
                }
                let wait = backoff(attempt, &timing);
                attempt = attempt.saturating_add(1);
                crate::note(&format!(
                    "the link to {} dropped: {why}; trying again in {}s",
                    config.hub,
                    wait.as_secs().max(1)
                ));
                state.write(&State::Error(why));
                sleep_unless(wait, &stop);
            }
        }
    }
}

/// What a running link thread is, to stop it.
struct Running {
    config: LinkConfig,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Running {
    fn end(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The supervisor, to end it when the service quits.
#[derive(Clone)]
pub struct Handle {
    stop: Arc<AtomicBool>,
    done: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl Handle {
    /// Ends the link (if one is up) and the supervisor; returns once they have.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.done.lock().unwrap().take() {
            let _ = thread.join();
        }
    }
}

/// Starts the supervisor for the service whose home is `root` and whose clients socket is `local`: it links this computer
/// while `link.json` is there.
pub fn start(root: PathBuf, local: PathBuf) -> Handle {
    start_with(root, local, Timing::default(), Arc::new(SshConnector))
}

pub fn start_with(root: PathBuf, local: PathBuf, timing: Timing, connector: Arc<dyn Connector>) -> Handle {
    let stop = Arc::new(AtomicBool::new(false));
    let thread = {
        let stop = stop.clone();
        std::thread::spawn(move || supervise(root, local, timing, connector, stop))
    };
    Handle { stop, done: Arc::new(Mutex::new(Some(thread))) }
}

fn supervise(root: PathBuf, local: PathBuf, timing: Timing, connector: Arc<dyn Connector>, stop: Arc<AtomicBool>) {
    let state = StateFile::new(&root);
    let mut running: Option<Running> = None;
    let mut complained: Option<String> = None;
    while !stop.load(Ordering::SeqCst) {
        match LinkConfig::load(&root) {
            Ok(Some(config)) => {
                complained = None;
                if running.as_ref().map(|r| &r.config) != Some(&config) {
                    if let Some(old) = running.take() {
                        old.end();
                    }
                    let link_stop = Arc::new(AtomicBool::new(false));
                    let thread = {
                        let (config, connector, local, link_stop, state) =
                            (config.clone(), connector.clone(), local.clone(), link_stop.clone(), state.clone());
                        std::thread::spawn(move || run_link(config, connector, local, timing, link_stop, state))
                    };
                    crate::note(&format!("linking this computer to {} as {}", config.hub, config.name));
                    running = Some(Running { config, stop: link_stop, thread: Some(thread) });
                }
            }
            Ok(None) => {
                complained = None;
                if let Some(old) = running.take() {
                    old.end();
                    state.clear();
                }
            }
            Err(reason) => {
                if let Some(old) = running.take() {
                    old.end();
                }
                if complained.as_deref() != Some(reason.as_str()) {
                    crate::note(&format!("not linked: {reason}"));
                    state.write(&State::Error(reason.clone()));
                    complained = Some(reason);
                }
            }
        }
        sleep_unless(timing.poll_config, &stop);
    }
    if let Some(old) = running.take() {
        old.end();
    }
    state.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::{self, Hub};
    use aria_core::bridge::{DockMessage, PageMessage};
    use std::io::BufRead;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::AtomicUsize;

    fn fast() -> Timing {
        Timing {
            ping: Duration::from_millis(40),
            silent: Duration::from_millis(600),
            handshake: Duration::from_secs(2),
            backoff_first: Duration::from_millis(20),
            backoff_max: Duration::from_millis(160),
            stable_after: Duration::from_secs(30),
            coalesce: Duration::from_millis(20),
            poll_config: Duration::from_millis(30),
        }
    }

    fn wait_until(what: &str, done: impl Fn() -> bool) {
        let start = Instant::now();
        while !done() {
            assert!(start.elapsed() < Duration::from_secs(8), "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn dock(value: Value) -> DockMessage {
        serde_json::from_value(value).unwrap()
    }

    fn config(name: &str) -> LinkConfig {
        LinkConfig {
            hub: "ana@hub.example".into(),
            name: name.into(),
            key: "/home/ana/.ssh/aria-machine-laptop".into(),
        }
    }

    // ---- link.json, names, backoff, ssh -------------------------------------------------------------------------------

    #[test]
    fn a_good_config_is_read_and_a_bad_one_says_what_is_wrong() {
        let good = r#"{"hub":"ana@hub.example","name":"laptop","key":"/home/ana/.ssh/aria-machine-laptop"}"#;
        assert_eq!(LinkConfig::parse(good).unwrap(), config("laptop"));
        for (text, wanted) in [
            (r#"{"hub":"ana@hub.example","name":"Laptop","key":"/k"}"#, "name"),
            (r#"{"hub":"ana@hub.example","name":"","key":"/k"}"#, "name"),
            (r#"{"hub":"ana@hub.example","name":"a b","key":"/k"}"#, "name"),
            (r#"{"hub":"-oProxyCommand=x","name":"laptop","key":"/k"}"#, "hub"),
            (r#"{"hub":"ana@hub example","name":"laptop","key":"/k"}"#, "hub"),
            (r#"{"hub":"","name":"laptop","key":"/k"}"#, "hub"),
            (r#"{"hub":"ana@hub.example","name":"laptop","key":"relative/key"}"#, "key"),
            (r#"{"hub":"ana@hub.example","name":"laptop"}"#, "link.json"),
            ("not json", "link.json"),
        ] {
            let error = LinkConfig::parse(text).unwrap_err();
            assert!(error.contains(wanted), "{text}: {error}");
        }
    }

    #[test]
    fn the_config_is_saved_for_its_owner_alone_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(LinkConfig::load(dir.path()), Ok(None), "no file is no link");
        config("laptop").save(dir.path()).unwrap();
        let mode = std::fs::metadata(LinkConfig::path(dir.path())).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        assert_eq!(LinkConfig::load(dir.path()), Ok(Some(config("laptop"))));
        std::fs::write(LinkConfig::path(dir.path()), "{").unwrap();
        assert!(LinkConfig::load(dir.path()).is_err(), "a file that is wrong is said, not ignored");
    }

    #[test]
    fn a_machine_name_comes_from_the_host_name() {
        assert_eq!(machine_name_from("Ana-MacBook-Pro.local"), "ana-macbook-pro-local");
        assert_eq!(machine_name_from("  My Laptop!! "), "my-laptop");
        assert_eq!(machine_name_from("---"), "computer");
        assert_eq!(machine_name_from(""), "computer");
        assert_eq!(machine_name_from("ÑÑÑ"), "computer");
        let long = machine_name_from(&"a".repeat(80));
        assert_eq!(long.len(), 31);
        assert!(valid_name(&long));
        // cut where a dash would end the name
        let cut = machine_name_from(&format!("{}-{}", "a".repeat(30), "b"));
        assert!(valid_name(&cut) && !cut.ends_with('-'), "{cut}");
    }

    #[test]
    fn the_wait_before_opening_a_dropped_link_again_doubles_up_to_a_minute() {
        let timing = Timing::default();
        let waits: Vec<u64> = (0..8).map(|attempt| backoff(attempt, &timing).as_secs()).collect();
        assert_eq!(waits, [2, 4, 8, 16, 32, 60, 60, 60]);
        assert_eq!(backoff(u32::MAX, &timing).as_secs(), 60, "never overflows");
    }

    #[test]
    fn ssh_is_given_the_key_and_nothing_that_can_ask_a_question() {
        let args = ssh_args(&config("laptop")).unwrap();
        assert_eq!(
            args,
            [
                "-T",
                "-i",
                "/home/ana/.ssh/aria-machine-laptop",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "ana@hub.example",
                "~/.local/bin/aria",
                "hub-link",
                "machine",
                "laptop"
            ]
        );
        let mut hostile = config("laptop");
        hostile.hub = "-oProxyCommand=touch x".into();
        assert!(ssh_args(&hostile).is_err());
        let mut hostile = config("laptop");
        hostile.name = "laptop; rm -rf ~".into();
        assert!(ssh_args(&hostile).is_err());
    }

    #[test]
    fn a_list_is_trimmed_to_what_the_hub_keeps() {
        let many: Vec<Value> = (0..700).map(|n| json!({"id": format!("s{n}")})).collect();
        let kept = fit_sessions(many);
        assert_eq!(kept.len(), MAX_SESSIONS);
        assert_eq!(kept[0]["id"], "s200", "the most recent are kept");
        let heavy: Vec<Value> =
            (0..40).map(|n| json!({"id": format!("s{n}"), "segments": "x".repeat(100_000)})).collect();
        let kept = fit_sessions(heavy);
        assert!(serde_json::to_string(&kept).unwrap().len() <= MAX_SESSIONS_BYTES);
        assert_eq!(kept.len(), 40, "dropping each tab's bulk was enough, so every tab stays");
        assert!(kept.iter().all(|entry| entry.get("segments").is_none()));
        let huge: Vec<Value> = (0..10).map(|n| json!({"id": format!("s{n}"), "title": "x".repeat(400_000)})).collect();
        let kept = fit_sessions(huge);
        assert!(serde_json::to_string(&kept).unwrap().len() <= MAX_SESSIONS_BYTES && !kept.is_empty());
    }

    // ---- the pieces of a link, with an in-memory hub and an in-memory service ------------------------------------------

    /// This machine's own service, as far as the link sees it: a clients hub on `clients.sock` whose chat is answered here.
    struct Machine {
        hub: Hub,
        socket: PathBuf,
        seen: Arc<Mutex<Vec<Value>>>,
        sessions: Arc<Mutex<Vec<Value>>>,
    }

    impl Machine {
        fn start(dir: &Path, sessions: Vec<Value>) -> Machine {
            let (chat, queue) = mpsc::channel::<PageMessage>();
            let hub = Hub::new(chat, Arc::new(|_| {}), Arc::new(|id| id.starts_with("m-")));
            let socket = dir.join("clients.sock");
            hub.serve(clients::bind(&socket).unwrap());
            let seen = Arc::new(Mutex::new(Vec::new()));
            let sessions = Arc::new(Mutex::new(sessions));
            {
                let (hub, seen, sessions) = (hub.clone(), seen.clone(), sessions.clone());
                std::thread::spawn(move || {
                    for message in queue {
                        seen.lock().unwrap().push(serde_json::to_value(&message).unwrap());
                        if let PageMessage::SessionsLoad { id } = &message {
                            let list = sessions.lock().unwrap().clone();
                            hub.route(&dock(json!({"type": "sessions", "id": id, "sessions": list, "active": null})));
                        }
                    }
                });
            }
            Machine { hub, socket, seen, sessions }
        }
        fn set_sessions(&self, sessions: Vec<Value>) {
            *self.sessions.lock().unwrap() = sessions;
            self.hub.broadcast(&json!({"type": "sessionsChanged"}));
        }
        fn saw(&self, type_: &str) -> usize {
            self.seen.lock().unwrap().iter().filter(|message| message["type"] == type_).count()
        }
    }

    /// The hub's side: a clients hub with machines.sock, as `aria serve` makes it.
    struct HubSide {
        hub: Hub,
        machines_socket: PathBuf,
    }

    impl HubSide {
        fn start(dir: &Path, timing: machines::Timing) -> HubSide {
            let (chat, _) = mpsc::channel();
            let hub = Hub::new(chat, Arc::new(|_| {}), Arc::new(|id| id.starts_with("hub-"))).with_timing(timing);
            let machines_socket = dir.join("machines.sock");
            hub.serve_machines(clients::bind(&machines_socket).unwrap());
            HubSide { hub, machines_socket }
        }
        /// A hub client, as `clients.rs` serves them (as `aria client-stdio` brings it: remote).
        fn client(&self) -> Peer {
            let (ours, theirs) = UnixStream::pair().unwrap();
            self.hub.add(theirs);
            let mut peer = Peer::new(ours);
            peer.send(json!({"type": "hello", "protocol": PROTOCOL, "version": "x", "remote": true}));
            assert_eq!(peer.next()["type"], "hello");
            peer
        }
    }

    use crate::machines;

    struct Peer {
        out: UnixStream,
        lines: BufReader<UnixStream>,
    }

    impl Peer {
        fn new(stream: UnixStream) -> Peer {
            stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            Peer { lines: BufReader::new(stream.try_clone().unwrap()), out: stream }
        }
        fn send(&mut self, value: Value) {
            writeln!(self.out, "{value}").unwrap();
        }
        /// A line to a link that may already have closed on the one before: no error if it has.
        fn send_if_open(&mut self, value: Value) {
            let _ = writeln!(self.out, "{value}");
        }
        fn next(&mut self) -> Value {
            let mut line = String::new();
            assert!(self.lines.read_line(&mut line).unwrap() > 0, "the connection closed");
            serde_json::from_str(&line).unwrap()
        }
        fn next_within(&mut self, wait: Duration) -> Option<Value> {
            self.out.set_read_timeout(Some(wait)).unwrap();
            let mut line = String::new();
            let result = self.lines.read_line(&mut line);
            self.out.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            match result {
                Ok(n) if n > 0 => serde_json::from_str(&line).ok(),
                _ => None,
            }
        }
        /// The next line that is not a heartbeat.
        fn next_real(&mut self) -> Value {
            loop {
                let line = self.next();
                if !matches!(line["type"].as_str(), Some("link-ping" | "link-pong")) {
                    return line;
                }
            }
        }
        fn quiet(&mut self, wait: Duration) -> bool {
            while let Some(line) = self.next_within(wait) {
                if !matches!(line["type"].as_str(), Some("link-ping" | "link-pong")) {
                    return false;
                }
            }
            true
        }
    }

    /// Opens a connection to the hub's machines.sock the way `aria hub-link` does: it writes the preamble itself.
    struct SocketConnector {
        machines_socket: PathBuf,
        opened: Arc<AtomicUsize>,
        last: Arc<Mutex<Option<UnixStream>>>,
        /// The name the preamble carries instead of the config's (the forced command of a key is what says it).
        forced_name: Option<String>,
    }

    impl SocketConnector {
        fn new(side: &HubSide) -> SocketConnector {
            SocketConnector {
                machines_socket: side.machines_socket.clone(),
                opened: Arc::default(),
                last: Arc::default(),
                forced_name: None,
            }
        }
    }

    impl Connector for SocketConnector {
        fn connect(&self, config: &LinkConfig) -> io::Result<Connection> {
            self.opened.fetch_add(1, Ordering::SeqCst);
            let mut stream = UnixStream::connect(&self.machines_socket)?;
            let name = self.forced_name.clone().unwrap_or_else(|| config.name.clone());
            writeln!(stream, "{}", json!({"type": "link-bridge", "name": name}))?;
            *self.last.lock().unwrap() = Some(stream.try_clone()?);
            let closer = stream.try_clone()?;
            Ok(Connection {
                reader: Box::new(stream.try_clone()?),
                writer: Box::new(stream),
                close: Box::new(move || {
                    let _ = closer.shutdown(std::net::Shutdown::Both);
                }),
                why: Box::new(|| "the connection ended".to_string()),
            })
        }
    }

    /// A hub that is a script: the machine's end of a socket pair is given to the link, this end is the test's to read and write.
    struct Scripted {
        peer: Arc<Mutex<Option<Peer>>>,
        opened: Arc<AtomicUsize>,
    }

    impl Connector for Scripted {
        fn connect(&self, _config: &LinkConfig) -> io::Result<Connection> {
            self.opened.fetch_add(1, Ordering::SeqCst);
            let (machine_end, test_end) = UnixStream::pair()?;
            *self.peer.lock().unwrap() = Some(Peer::new(test_end));
            let closer = machine_end.try_clone()?;
            Ok(Connection {
                reader: Box::new(machine_end.try_clone()?),
                writer: Box::new(machine_end),
                close: Box::new(move || {
                    let _ = closer.shutdown(std::net::Shutdown::Both);
                }),
                why: Box::new(|| "the connection ended".to_string()),
            })
        }
    }

    fn scripted() -> (Scripted, Arc<Mutex<Option<Peer>>>, Arc<AtomicUsize>) {
        let peer: Arc<Mutex<Option<Peer>>> = Arc::default();
        let opened: Arc<AtomicUsize> = Arc::default();
        (Scripted { peer: peer.clone(), opened: opened.clone() }, peer, opened)
    }

    /// Runs one link on a thread and gives back what it ended with.
    fn run_once(
        connector: Arc<dyn Connector>,
        local: PathBuf,
        timing: Timing,
        root: &Path,
        stop: Arc<AtomicBool>,
    ) -> JoinHandle<End> {
        let state = StateFile::new(root);
        std::thread::spawn(move || link_once(&config("laptop"), connector.as_ref(), &local, timing, &stop, &state))
    }

    fn take_peer(slot: &Arc<Mutex<Option<Peer>>>) -> Peer {
        wait_until("the link to open", || slot.lock().unwrap().is_some());
        slot.lock().unwrap().take().unwrap()
    }

    // ---- a real hub and a real machine service, linked --------------------------------------------------------------

    struct Linked {
        machine: Machine,
        side: HubSide,
        connector: Arc<SocketConnector>,
        stop: Arc<AtomicBool>,
        thread: Option<JoinHandle<End>>,
        dir: tempfile::TempDir,
    }

    impl Linked {
        fn start(sessions: Vec<Value>) -> Linked {
            let dir = tempfile::tempdir().unwrap();
            let machine_dir = dir.path().join("machine");
            let hub_dir = dir.path().join("hub");
            std::fs::create_dir_all(&machine_dir).unwrap();
            std::fs::create_dir_all(&hub_dir).unwrap();
            let machine = Machine::start(&machine_dir, sessions);
            let side = HubSide::start(&hub_dir, machines::Timing::default());
            let connector = Arc::new(SocketConnector::new(&side));
            let stop = Arc::new(AtomicBool::new(false));
            let thread = run_once(connector.clone(), machine.socket.clone(), fast(), dir.path(), stop.clone());
            Linked { machine, side, connector, stop, thread: Some(thread), dir }
        }
        fn linked(&self) -> bool {
            self.side.hub.machines().list().iter().any(|machine| machine.name == "laptop" && machine.online)
        }
    }

    impl Drop for Linked {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    #[test]
    fn a_machine_links_and_the_hub_sees_its_sessions() {
        let linked = Linked::start(vec![json!({"id": "m-1", "title": "One"}), json!({"id": "m-2", "title": "Two"})]);
        wait_until("the machine to be listed", || linked.linked());
        wait_until("its sessions", || linked.side.hub.machines().owner_of("m-2").is_some());
        let listed = &linked.side.hub.machines().list()[0];
        assert_eq!(listed.os, if cfg!(target_os = "macos") { "macos" } else { "linux" });
        assert_eq!(listed.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(listed.sessions, 2);
        // a change on the machine reaches the hub without anything being asked
        linked.machine.set_sessions(vec![json!({"id": "m-1", "title": "One"}), json!({"id": "m-3", "title": "Three"})]);
        wait_until("the new list", || linked.side.hub.machines().owner_of("m-3").is_some());
        assert!(linked.side.hub.machines().owner_of("m-2").is_none(), "a session that went is gone from the cache");
        assert!(linked.dir.path().exists());
    }

    #[test]
    fn a_hub_clients_line_reaches_the_machines_session_and_the_answer_comes_back_to_that_client_only() {
        let linked = Linked::start(vec![json!({"id": "m-1", "title": "One"})]);
        wait_until("its sessions", || linked.side.hub.machines().owner_of("m-1").is_some());
        let mut ana = linked.side.client();
        let mut bob = linked.side.client();
        ana.send(json!({"type": "attach", "session": "m-1"}));
        ana.send(json!({"type": "session", "session": "m-1", "message": {"type": "chatSend", "text": "hi", "id": 5}}));
        wait_until("the message to reach the machine's chat", || linked.machine.saw("session") >= 1);
        let seen = linked.machine.seen.lock().unwrap().clone();
        let chat = seen.iter().find(|message| message["type"] == "session").unwrap();
        assert_eq!(chat["session"], "m-1");
        assert_eq!(chat["message"]["type"], "chatSend");
        assert_eq!(chat["message"]["text"], "hi");
        // an event of that session goes to the client attached to it, and to nobody else
        linked.machine.hub.route(&dock(json!({
            "type": "session", "session": "m-1", "message": {"type": "chatEvent", "event": {"type": "agent_start"}}
        })));
        let event = ana.next_real();
        assert_eq!(event["session"], "m-1");
        assert_eq!(event["message"]["event"]["type"], "agent_start");
        assert!(bob.quiet(Duration::from_millis(200)), "a client that never asked got the machine's event");
    }

    #[test]
    fn the_machine_applies_the_restrictions_of_a_remote_client() {
        let linked = Linked::start(vec![json!({"id": "m-1", "title": "One"})]);
        wait_until("its sessions", || linked.side.hub.machines().owner_of("m-1").is_some());
        let mut ana = linked.side.client();
        // a session the machine does not have is refused by the machine too, with the client's own id
        ana.send(
            json!({"type": "session", "session": "m-1", "message": {"type": "chatBash", "id": 9, "command": "id"}}),
        );
        let refused = ana.next_real();
        assert_eq!(refused["type"], "error", "{refused}");
        assert_eq!(refused["id"], 9);
    }

    #[test]
    fn a_hub_clients_end_closes_its_local_connection() {
        let linked = Linked::start(vec![json!({"id": "m-1", "title": "One"})]);
        wait_until("its sessions", || linked.side.hub.machines().owner_of("m-1").is_some());
        // the observer is one local connection already
        wait_until("the observer", || linked.machine.hub.count() == 1);
        let mut ana = linked.side.client();
        ana.send(json!({"type": "attach", "session": "m-1"}));
        wait_until("the client's own local connection", || linked.machine.hub.count() == 2);
        drop(ana);
        wait_until("it to close", || linked.machine.hub.count() == 1);
    }

    #[test]
    fn a_dropped_link_is_opened_again_and_the_machine_is_back() {
        let linked = Linked::start(vec![json!({"id": "m-1", "title": "One"})]);
        // the link ends by itself when the connection ends: `link_once` is one link, `run_link` opens the next
        wait_until("the machine to be listed", || linked.linked());
        let state = StateFile::new(linked.dir.path());
        let local = linked.machine.socket.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let connector: Arc<SocketConnector> = linked.connector.clone();
        let opened = connector.opened.clone();
        let before = opened.load(Ordering::SeqCst);
        let runner = {
            let stop = stop.clone();
            let mut other = config("laptop");
            other.name = "tablet".into();
            std::thread::spawn(move || run_link(other, connector, local, fast(), stop, state))
        };
        wait_until("the second machine", || {
            linked.side.hub.machines().list().iter().any(|m| m.name == "tablet" && m.online)
        });
        let opened_now = opened.load(Ordering::SeqCst);
        assert_eq!(opened_now, before + 1);
        // the connection dies: it is opened again
        let dying = linked.connector.last.lock().unwrap().take().unwrap();
        dying.shutdown(std::net::Shutdown::Both).unwrap();
        wait_until("it to be opened again", || opened.load(Ordering::SeqCst) > opened_now);
        wait_until("the machine to be back", || {
            linked.side.hub.machines().list().iter().any(|m| m.name == "tablet" && m.online)
        });
        stop.store(true, Ordering::SeqCst);
        runner.join().unwrap();
    }

    // ---- a script for a hub -------------------------------------------------------------------------------------------

    fn hello_of(peer: &mut Peer) -> Value {
        let hello = peer.next();
        assert_eq!(hello["type"], "link-hello");
        assert_eq!(hello["protocol"], PROTOCOL);
        assert_eq!(hello["name"], "laptop");
        assert!(hello["os"].is_string() && hello["version"].is_string());
        hello
    }

    struct Rig {
        _dir: tempfile::TempDir,
        machine: Machine,
        root: PathBuf,
    }

    fn rig() -> Rig {
        let dir = tempfile::tempdir().unwrap();
        let machine_dir = dir.path().join("machine");
        std::fs::create_dir_all(&machine_dir).unwrap();
        let machine = Machine::start(&machine_dir, vec![json!({"id": "m-1", "title": "One"})]);
        let root = dir.path().to_path_buf();
        Rig { _dir: dir, machine, root }
    }

    #[test]
    fn another_protocol_stops_the_retrying() {
        let rig = rig();
        let (connector, peer, opened) = scripted();
        let stop = Arc::new(AtomicBool::new(false));
        let state = StateFile::new(&rig.root);
        let runner = {
            let (stop, local, state) = (stop.clone(), rig.machine.socket.clone(), state.clone());
            std::thread::spawn(move || run_link(config("laptop"), Arc::new(connector), local, fast(), stop, state))
        };
        let mut hub = take_peer(&peer);
        hello_of(&mut hub);
        // what the hub does for another protocol: its hello (with its own protocol), then the error, then it closes
        hub.send_if_open(json!({"type": "link-hello", "protocol": PROTOCOL + 1, "version": "9"}));
        hub.send_if_open(json!({"type": "error", "id": 0, "code": "protocol-mismatch"}));
        runner.join().unwrap();
        assert_eq!(opened.load(Ordering::SeqCst), 1, "it was opened again");
        let reported = reported(&rig.root).unwrap();
        assert_eq!(reported.state, "error");
        assert!(reported.reason.unwrap().contains("another version of the protocol"));
        stop.store(true, Ordering::SeqCst);
    }

    #[test]
    fn an_error_after_the_hello_stops_the_retrying_too() {
        let rig = rig();
        let (connector, peer, opened) = scripted();
        let stop = Arc::new(AtomicBool::new(false));
        let state = StateFile::new(&rig.root);
        let runner = {
            let (stop, local) = (stop.clone(), rig.machine.socket.clone());
            std::thread::spawn(move || run_link(config("laptop"), Arc::new(connector), local, fast(), stop, state))
        };
        let mut hub = take_peer(&peer);
        hello_of(&mut hub);
        hub.send(json!({"type": "link-hello", "protocol": PROTOCOL, "version": "9"}));
        hub.send_if_open(json!({"type": "error", "id": 0, "code": "protocol-mismatch"}));
        runner.join().unwrap();
        assert_eq!(opened.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_name_the_hub_does_not_know_stops_the_retrying() {
        let rig = rig();
        let (connector, peer, opened) = scripted();
        let stop = Arc::new(AtomicBool::new(false));
        let state = StateFile::new(&rig.root);
        let runner = {
            let (stop, local) = (stop.clone(), rig.machine.socket.clone());
            std::thread::spawn(move || run_link(config("laptop"), Arc::new(connector), local, fast(), stop, state))
        };
        let mut hub = take_peer(&peer);
        hello_of(&mut hub);
        hub.send_if_open(json!({"type": "error", "id": 0, "code": "name-mismatch"}));
        runner.join().unwrap();
        assert_eq!(opened.load(Ordering::SeqCst), 1);
        assert!(reported(&rig.root).unwrap().reason.unwrap().contains("another machine name"));
    }

    #[test]
    fn the_heartbeat_is_answered_and_sent_and_a_silent_hub_is_dropped() {
        let rig = rig();
        let (connector, peer, _) = scripted();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = run_once(Arc::new(connector), rig.machine.socket.clone(), fast(), &rig.root, stop.clone());
        let mut hub = take_peer(&peer);
        hello_of(&mut hub);
        hub.send(json!({"type": "link-hello", "protocol": PROTOCOL, "version": "9"}));
        hub.send(json!({"type": "link-ping"}));
        let mut seen_pong = false;
        let mut seen_ping = false;
        let until = Instant::now() + Duration::from_secs(3);
        while Instant::now() < until && !(seen_pong && seen_ping) {
            match hub.next_within(Duration::from_millis(200)) {
                Some(line) if line["type"] == "link-pong" => seen_pong = true,
                Some(line) if line["type"] == "link-ping" => seen_ping = true,
                _ => {}
            }
        }
        assert!(seen_pong, "a ping is answered");
        assert!(seen_ping, "a ping is sent");
        // and now nothing is heard from the hub: the link is dropped after `silent`
        let end = thread.join().unwrap();
        match end {
            End::Dropped { why, .. } => assert!(why.contains("stopped answering"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_hub_that_does_not_answer_the_hello_is_given_up_on() {
        let rig = rig();
        let (connector, peer, _) = scripted();
        let mut timing = fast();
        timing.handshake = Duration::from_millis(200);
        let stop = Arc::new(AtomicBool::new(false));
        let thread = run_once(Arc::new(connector), rig.machine.socket.clone(), timing, &rig.root, stop);
        let mut hub = take_peer(&peer);
        hello_of(&mut hub);
        match thread.join().unwrap() {
            End::Dropped { why, .. } => assert!(why.contains("did not answer the hello"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn at_most_thirty_two_clients_are_served_at_once() {
        let rig = rig();
        let (connector, peer, _) = scripted();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = run_once(Arc::new(connector), rig.machine.socket.clone(), fast(), &rig.root, stop.clone());
        let mut hub = take_peer(&peer);
        hello_of(&mut hub);
        hub.send(json!({"type": "link-hello", "protocol": PROTOCOL, "version": "9"}));
        for client in 1..=33u64 {
            hub.send(json!({"c": client, "line": {"type": "sessionsLoad", "id": 1}}));
        }
        let mut refused = None;
        let mut answered = std::collections::HashSet::new();
        let until = Instant::now() + Duration::from_secs(5);
        while Instant::now() < until && (refused.is_none() || answered.len() < 32) {
            if let Some(line) = hub.next_within(Duration::from_millis(200)) {
                if let Some(client) = line["c"].as_u64() {
                    if line["line"]["code"] == "too-many-clients" {
                        refused = Some(client);
                    } else if line["line"]["type"] == "sessions" {
                        answered.insert(client);
                    }
                }
            }
        }
        assert_eq!(refused, Some(33), "the 33rd is refused");
        assert_eq!(answered.len(), 32);
        stop.store(true, Ordering::SeqCst);
        assert_eq!(thread.join().unwrap(), End::Stopped);
    }

    #[test]
    fn a_local_clients_own_hello_never_goes_to_the_hub_and_its_end_is_said() {
        let rig = rig();
        let (connector, peer, _) = scripted();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = run_once(Arc::new(connector), rig.machine.socket.clone(), fast(), &rig.root, stop.clone());
        let mut hub = take_peer(&peer);
        hello_of(&mut hub);
        hub.send(json!({"type": "link-hello", "protocol": PROTOCOL, "version": "9"}));
        hub.send(json!({"c": 7, "line": {"type": "sessionsLoad", "id": 3}}));
        let reply = loop {
            let line = hub.next_real();
            if line["c"] == 7 {
                break line;
            }
        };
        assert_eq!(reply["line"]["type"], "sessions", "the service's hello was not relayed: {reply}");
        assert_eq!(reply["line"]["id"], 3, "the client got its own id back");
        // the hub closes the client: the local connection ends, and the link does not say it back
        hub.send(json!({"c": 7, "close": true}));
        wait_until("the local connection to end", || rig.machine.hub.count() == 1);
        // (the session list is pushed as it changes; only a line about client 7 would be wrong)
        while let Some(line) = hub.next_within(Duration::from_millis(200)) {
            assert!(line["c"] != 7, "a close the hub asked for was said back: {line}");
        }
        stop.store(true, Ordering::SeqCst);
        thread.join().unwrap();
    }

    #[test]
    fn a_machine_whose_service_has_no_socket_cannot_link() {
        let dir = tempfile::tempdir().unwrap();
        let (connector, peer, _) = scripted();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = run_once(Arc::new(connector), dir.path().join("none.sock"), fast(), dir.path(), stop);
        let mut hub = take_peer(&peer);
        hello_of(&mut hub);
        hub.send(json!({"type": "link-hello", "protocol": PROTOCOL, "version": "9"}));
        match thread.join().unwrap() {
            End::Dropped { why, .. } => assert!(why.contains("stopped answering"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    // ---- the supervisor and the state file ---------------------------------------------------------------------------

    #[test]
    fn the_link_starts_when_the_file_appears_and_ends_when_it_goes() {
        let dir = tempfile::tempdir().unwrap();
        let machine_dir = dir.path().join("machine");
        let hub_dir = dir.path().join("hub");
        std::fs::create_dir_all(&machine_dir).unwrap();
        std::fs::create_dir_all(&hub_dir).unwrap();
        let machine = Machine::start(&machine_dir, vec![json!({"id": "m-1", "title": "One"})]);
        let side = HubSide::start(&hub_dir, machines::Timing::default());
        let connector = Arc::new(SocketConnector::new(&side));
        let root = dir.path().join("home");
        std::fs::create_dir_all(&root).unwrap();
        let handle = start_with(root.clone(), machine.socket.clone(), fast(), connector.clone());
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(connector.opened.load(Ordering::SeqCst), 0, "no file, no link");
        assert!(reported(&root).is_none());

        config("laptop").save(&root).unwrap();
        wait_until("the link", || side.hub.machines().list().iter().any(|m| m.name == "laptop" && m.online));
        wait_until("the state", || reported(&root).is_some_and(|r| r.state == "connected"));
        assert!(reported(&root).unwrap().alive, "this process wrote it");

        std::fs::remove_file(LinkConfig::path(&root)).unwrap();
        wait_until("the machine to go offline", || {
            side.hub.machines().list().iter().all(|m| m.name != "laptop" || !m.online)
        });
        wait_until("the state to go", || reported(&root).is_none());

        // a wrong file is said, and does not start anything
        let opened = connector.opened.load(Ordering::SeqCst);
        std::fs::write(LinkConfig::path(&root), r#"{"hub":"-x","name":"laptop","key":"/k"}"#).unwrap();
        wait_until("the state to say it", || reported(&root).is_some_and(|r| r.state == "error"));
        assert!(reported(&root).unwrap().reason.unwrap().contains("hub"));
        assert_eq!(connector.opened.load(Ordering::SeqCst), opened);

        handle.stop();
        assert!(reported(&root).is_none(), "a stopped service leaves no state");
    }

    #[test]
    fn a_changed_file_moves_the_link_to_the_new_name() {
        let dir = tempfile::tempdir().unwrap();
        let machine_dir = dir.path().join("machine");
        let hub_dir = dir.path().join("hub");
        std::fs::create_dir_all(&machine_dir).unwrap();
        std::fs::create_dir_all(&hub_dir).unwrap();
        let machine = Machine::start(&machine_dir, vec![]);
        let side = HubSide::start(&hub_dir, machines::Timing::default());
        let connector = Arc::new(SocketConnector::new(&side));
        let root = dir.path().join("home");
        std::fs::create_dir_all(&root).unwrap();
        config("laptop").save(&root).unwrap();
        let handle = start_with(root.clone(), machine.socket.clone(), fast(), connector);
        wait_until("the first name", || side.hub.machines().list().iter().any(|m| m.name == "laptop" && m.online));
        config("desk").save(&root).unwrap();
        wait_until("the second name", || side.hub.machines().list().iter().any(|m| m.name == "desk" && m.online));
        wait_until("the first to go offline", || {
            side.hub.machines().list().iter().any(|m| m.name == "laptop" && !m.online)
        });
        handle.stop();
    }

    #[test]
    fn a_dead_pid_in_the_state_means_no_link() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(STATE_FILE), r#"{"state":"connected","pid":2147483646}"#).unwrap();
        let reported = reported(dir.path()).unwrap();
        assert_eq!(reported.state, "connected");
        assert!(!reported.alive, "a service that is gone is not linked");
        std::fs::write(dir.path().join(STATE_FILE), "not json").unwrap();
        assert!(super::reported(dir.path()).is_none());
    }
}
