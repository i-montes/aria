//! The hub's side of the machine link (docs/superpowers/specs/2026-10-27-machine-link-design.md): a computer linked to this
//! hub as a **machine** keeps one ssh connection open to it, and through it the hub's clients see the machine's sessions
//! and work on them.
//!
//! - `machines.sock` (beside `clients.sock`, same permissions) is where `aria hub-link machine <name>` connects: the forced
//!   command of a machine's key. The bridge's first line is `{"type":"link-bridge","name":"<name>"}`, written by the bridge
//!   from its own argv, so the name comes from the key and never from the machine's input; the machine's own first line is
//!   `link-hello { protocol, name, os, version }` and must carry the same name (`name-mismatch` otherwise).
//! - A second link with the same name replaces the first (the old one is almost always a socket nobody noticed dying).
//! - The registry keeps, per machine, the last `link-sessions { sessions, statuses }` it pushed (memory only: a view, the
//!   machine owns its sessions). A machine whose link drops stays listed, `offline`, with that cache.
//! - A hub client's line for a machine's session travels as `{"c": <client number>, "line": <the line>}`; what the machine
//!   answers for that client comes back as `{"c": n, "line": ...}` and goes to client `n` unchanged, but only while the hub has
//!   sent client `n` something on that link (its route): a machine can never speak to a client it was not asked by.
//!   `{"c": n, "close": true}` ends a route, either way.
//! - The hub reads exactly these types from a machine: `link-hello`, `link-sessions`, `link-ping`, `link-pong` and the `c`
//!   envelopes. Anything else is ignored, so a machine can never make the hub run a client's or a handler's message.
//! - Heartbeat: `link-ping` every `Timing::ping`, and a machine silent for `Timing::silent` is dropped. A machine that does
//!   not read (its queue of `Timing::queue` lines is full) is dropped, never the hub.

use crate::clients::Hub;
use crate::stdio_bridge::{next_line, relay, write_line, Exit, Next};
use aria_core::bridge::PROTOCOL;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::io::{BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest line taken from a machine: a client's reply can hold a whole conversation.
const MAX_LINE: usize = 16 * 1024 * 1024;
/// The longest `link-sessions` line: a list of tabs is small, and a bigger one is a machine that is not behaving.
const MAX_SESSIONS_LINE: usize = 2 * 1024 * 1024;
/// How many sessions of one machine the hub keeps.
const MAX_ENTRIES: usize = 500;
/// The preamble and the hello are short.
const MAX_HANDSHAKE_LINE: usize = 64 * 1024;
/// How long a connection has to say who it is.
const HANDSHAKE: Duration = Duration::from_secs(10);

/// Where `aria hub-link` finds the hub's machines socket.
pub fn socket_path() -> PathBuf {
    crate::clients::socket_path().with_file_name("machines.sock")
}

/// The intervals and the queue of a link: injectable so the tests need not wait.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// How often the hub sends `link-ping`.
    pub ping: Duration,
    /// How long a machine may say nothing before its link is dropped.
    pub silent: Duration,
    /// How many lines may wait for a machine that is slow to read.
    pub queue: usize,
}

impl Default for Timing {
    fn default() -> Timing {
        Timing { ping: Duration::from_secs(15), silent: Duration::from_secs(45), queue: 1024 }
    }
}

/// A machine's name as a key can carry it: lowercase letters, digits and dashes, starting with a letter or a digit, at
/// most 31 characters (`^[a-z0-9][a-z0-9-]{0,30}$`).
pub fn valid_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    (1..=31).contains(&bytes.len())
        && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        && bytes.iter().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
}

/// Why a line could not be queued for a machine.
#[derive(Debug, PartialEq, Eq)]
pub enum Refused {
    /// No machine of that name has ever linked.
    Unknown,
    /// It has, and it is not linked now (or it did not read and was dropped).
    Offline,
}

/// The write end of a link and a handle that can close it.
struct Link {
    lines: SyncSender<String>,
    stream: UnixStream,
}

struct Machine {
    os: String,
    version: String,
    since: Instant,
    /// Which connection owns this entry: a connection that ends marks the machine offline only if it is still the one.
    conn: u64,
    link: Option<Link>,
    sessions: Vec<Value>,
    statuses: Map<String, Value>,
    /// The hub clients that have been sent something on this link.
    routes: HashSet<u64>,
}

#[derive(Default)]
struct Registry {
    machines: HashMap<String, Machine>,
    next_conn: u64,
}

impl Registry {
    /// The names in the order lists and lookups use.
    fn sorted(&self) -> Vec<&String> {
        let mut names: Vec<&String> = self.machines.keys().collect();
        names.sort();
        names
    }
}

/// What a machine said about itself, for `machinesList` (`aria hub machines`) and the tests.
#[derive(Debug, Clone, PartialEq)]
pub struct MachineInfo {
    pub name: String,
    pub os: String,
    pub version: String,
    pub online: bool,
    pub since_secs: u64,
    pub sessions: usize,
}

impl MachineInfo {
    /// The shape `machinesList` answers with: `since` is when its link began, as seconds since 1970 (a client writes the
    /// time of day itself), and means nothing while the machine is offline.
    pub fn to_json(&self) -> Value {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or_default();
        json!({
            "name": self.name,
            "os": self.os,
            "version": self.version,
            "online": self.online,
            "since": now.saturating_sub(self.since_secs),
            "sessions": self.sessions,
        })
    }
}

#[derive(Clone)]
pub struct Machines {
    inner: Arc<Mutex<Registry>>,
    timing: Timing,
}

impl Default for Machines {
    fn default() -> Machines {
        Machines::with_timing(Timing::default())
    }
}

impl Machines {
    pub fn with_timing(timing: Timing) -> Machines {
        Machines { inner: Arc::default(), timing }
    }

    pub fn list(&self) -> Vec<MachineInfo> {
        let registry = self.inner.lock().unwrap();
        registry
            .sorted()
            .into_iter()
            .map(|name| {
                let machine = &registry.machines[name];
                MachineInfo {
                    name: name.clone(),
                    os: machine.os.clone(),
                    version: machine.version.clone(),
                    online: machine.link.is_some(),
                    since_secs: machine.since.elapsed().as_secs(),
                    sessions: machine.sessions.len(),
                }
            })
            .collect()
    }

    /// The machine whose cached sessions hold `session` (the first by name when two would).
    pub fn owner_of(&self, session: &str) -> Option<String> {
        let registry = self.inner.lock().unwrap();
        registry
            .sorted()
            .into_iter()
            .find(|name| registry.machines[*name].sessions.iter().any(|entry| entry["id"] == session))
            .cloned()
    }

    /// Queues `line` for machine `name` on behalf of hub client `client`, which from now on has a route there. A machine
    /// that does not keep up (its queue is full) has its link closed: the hub never waits for one.
    pub fn forward(&self, name: &str, client: u64, line: &Value) -> Result<(), Refused> {
        let mut registry = self.inner.lock().unwrap();
        let machine = registry.machines.get_mut(name).ok_or(Refused::Unknown)?;
        let link = machine.link.as_ref().ok_or(Refused::Offline)?;
        let envelope = json!({"c": client, "line": line}).to_string();
        match link.lines.try_send(envelope) {
            Ok(()) => {
                machine.routes.insert(client);
                Ok(())
            }
            Err(_) => {
                // The reader of that link then ends it and the machine goes offline; the registry is not touched here.
                let _ = link.stream.shutdown(std::net::Shutdown::Both);
                Err(Refused::Offline)
            }
        }
    }

    /// Adds every machine's cached sessions to a `sessions` reply: each entry gains `machine`, and `offline` while its
    /// machine is not linked. An id the hub already lists (or an earlier machine did) is not listed twice.
    pub fn merge_into(&self, reply: &mut Value) {
        let registry = self.inner.lock().unwrap();
        let Some(list) = reply.get_mut("sessions").and_then(Value::as_array_mut) else { return };
        let mut seen: HashSet<String> =
            list.iter().filter_map(|entry| entry["id"].as_str().map(str::to_string)).collect();
        for name in registry.sorted() {
            let machine = &registry.machines[name];
            for entry in &machine.sessions {
                let Some(id) = entry["id"].as_str() else { continue };
                if !seen.insert(id.to_string()) {
                    continue;
                }
                let mut entry = entry.clone();
                entry["machine"] = json!(name);
                if machine.link.is_none() {
                    entry["offline"] = json!(true);
                }
                list.push(entry);
            }
        }
    }

    /// The last status of every cached machine session, as the `session` lines a client's list takes.
    pub fn status_events(&self) -> Vec<Value> {
        let registry = self.inner.lock().unwrap();
        let mut events = Vec::new();
        for name in registry.sorted() {
            let machine = &registry.machines[name];
            for (id, status) in &machine.statuses {
                if machine.sessions.iter().any(|entry| entry["id"] == id.as_str()) {
                    events.push(status_event(id, status));
                }
            }
        }
        events
    }

    /// A hub client is gone: every machine it had a route to is told, and the route ends.
    pub fn client_gone(&self, client: u64) {
        let mut registry = self.inner.lock().unwrap();
        for machine in registry.machines.values_mut() {
            if machine.routes.remove(&client) {
                if let Some(link) = &machine.link {
                    let _ = link.lines.try_send(json!({"c": client, "close": true}).to_string());
                }
            }
        }
    }

    /// Puts a new link in place of the machine's old one, if any, and returns the connection's number.
    fn register(&self, name: &str, os: String, version: String, link: Link) -> u64 {
        let mut registry = self.inner.lock().unwrap();
        registry.next_conn += 1;
        let conn = registry.next_conn;
        let machine = registry.machines.entry(name.to_string()).or_insert_with(|| Machine {
            os: String::new(),
            version: String::new(),
            since: Instant::now(),
            conn,
            link: None,
            sessions: Vec::new(),
            statuses: Map::new(),
            routes: HashSet::new(),
        });
        if let Some(old) = machine.link.take() {
            let _ = old.stream.shutdown(std::net::Shutdown::Both);
        }
        machine.os = os;
        machine.version = version;
        machine.since = Instant::now();
        machine.conn = conn;
        machine.link = Some(link);
        machine.routes.clear();
        conn
    }

    /// The connection `conn` of `name` ended: the machine is offline, unless a newer link already took its place.
    fn dropped(&self, name: &str, conn: u64) -> bool {
        let mut registry = self.inner.lock().unwrap();
        match registry.machines.get_mut(name) {
            Some(machine) if machine.conn == conn && machine.link.is_some() => {
                machine.link = None;
                machine.routes.clear();
                true
            }
            _ => false,
        }
    }

    /// Takes a `link-sessions` line into the cache of the connection `conn`; returns the statuses that changed.
    fn store(&self, name: &str, conn: u64, line: &Value) -> Option<Vec<Value>> {
        let sessions: Vec<Value> = line["sessions"]
            .as_array()?
            .iter()
            .filter(|entry| {
                entry.is_object() && entry["id"].as_str().is_some_and(|id| !id.is_empty() && id.len() <= 200)
            })
            .take(MAX_ENTRIES)
            .cloned()
            .map(|mut entry| {
                // `machine` and `offline` are the hub's to say.
                if let Some(object) = entry.as_object_mut() {
                    object.remove("machine");
                    object.remove("offline");
                }
                entry
            })
            .collect();
        let mut statuses = Map::new();
        if let Some(given) = line["statuses"].as_object() {
            for (id, status) in given.iter().take(MAX_ENTRIES) {
                if let (Some(text), Some(at)) = (status["status"].as_str(), status["at"].as_i64()) {
                    statuses.insert(id.clone(), json!({"status": text, "at": at}));
                }
            }
        }
        let mut registry = self.inner.lock().unwrap();
        let machine = registry.machines.get_mut(name)?;
        if machine.conn != conn {
            return None;
        }
        let changed: Vec<Value> = statuses
            .iter()
            .filter(|(id, status)| machine.statuses.get(*id) != Some(status))
            .filter(|(id, _)| sessions.iter().any(|entry| entry["id"] == id.as_str()))
            .map(|(id, status)| status_event(id, status))
            .collect();
        machine.sessions = sessions;
        machine.statuses = statuses;
        Some(changed)
    }

    /// May a line from `name` go to client `client`? Only one the hub has sent something on this link.
    fn has_route(&self, name: &str, client: u64) -> bool {
        self.inner.lock().unwrap().machines.get(name).is_some_and(|machine| machine.routes.contains(&client))
    }

    fn end_route(&self, name: &str, client: u64) {
        if let Some(machine) = self.inner.lock().unwrap().machines.get_mut(name) {
            machine.routes.remove(&client);
        }
    }

    /// Accepts machines' links on `listener` (a thread per link).
    pub fn serve(&self, hub: Hub, listener: UnixListener) {
        let machines = self.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().map_while(Result::ok) {
                let (machines, hub) = (machines.clone(), hub.clone());
                std::thread::spawn(move || machines.connection(&hub, stream));
            }
        });
    }

    /// One machine's link, from its first byte to its last.
    pub(crate) fn connection(&self, hub: &Hub, stream: UnixStream) {
        let _ = stream.set_read_timeout(Some(HANDSHAKE));
        let (Ok(read_half), Ok(mut write_half)) = (stream.try_clone(), stream.try_clone()) else { return };
        let mut reader = BufReader::new(read_half);
        let close = || {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        };
        // The bridge says whose key this is.
        let Some(bridge) = read_json(&mut reader) else { return close() };
        let name = match (bridge["type"].as_str(), bridge["name"].as_str()) {
            (Some("link-bridge"), Some(name)) if valid_name(name) => name.to_string(),
            _ => return close(),
        };
        let Some(hello) = read_json(&mut reader) else { return close() };
        if hello["type"] != "link-hello" {
            return close();
        }
        if hello["protocol"].as_u64() != Some(u64::from(PROTOCOL)) {
            let _ = write_line(&mut write_half, &hub_hello());
            let _ = write_line(&mut write_half, &json!({"type": "error", "id": 0, "code": "protocol-mismatch"}));
            return close();
        }
        if hello["name"].as_str() != Some(name.as_str()) {
            let _ = write_line(&mut write_half, &json!({"type": "error", "id": 0, "code": "name-mismatch"}));
            return close();
        }
        let os = plain(&hello["os"], 64);
        let version = plain(&hello["version"], 64);
        let _ = stream.set_read_timeout(None);

        let (lines, queue) = mpsc::sync_channel::<String>(self.timing.queue.max(1));
        let Ok(kept) = stream.try_clone() else { return };
        let conn = self.register(&name, os, version, Link { lines: lines.clone(), stream: kept });
        let _ = lines.try_send(hub_hello().to_string());
        std::thread::spawn(move || {
            for line in queue {
                if write_half.write_all(line.as_bytes()).and_then(|_| write_half.write_all(b"\n")).is_err() {
                    break;
                }
            }
            let _ = write_half.shutdown(std::net::Shutdown::Both);
        });
        hub.broadcast(&json!({"type": "sessionsChanged"}));

        // Silence is checked and pinged by a thread of its own; it ends with the link.
        let heard = Arc::new(Mutex::new(Instant::now()));
        let over = Arc::new(AtomicBool::new(false));
        {
            let (heard, over, lines, timing) = (heard.clone(), over.clone(), lines.clone(), self.timing);
            let Ok(stream) = stream.try_clone() else { return };
            std::thread::spawn(move || {
                let step =
                    (timing.ping.min(timing.silent) / 10).clamp(Duration::from_millis(5), Duration::from_millis(100));
                let mut pinged = Instant::now();
                while !over.load(Ordering::SeqCst) {
                    std::thread::sleep(step);
                    if pinged.elapsed() >= timing.ping {
                        let _ = lines.try_send(json!({"type": "link-ping"}).to_string());
                        pinged = Instant::now();
                    }
                    if heard.lock().unwrap().elapsed() >= timing.silent {
                        let _ = stream.shutdown(std::net::Shutdown::Both);
                        break;
                    }
                }
            });
        }

        // A line too long, the end, or an error ends the link as a line that asks for it does.
        while let Ok(Next::Line(line)) = next_line(&mut reader, MAX_LINE) {
            *heard.lock().unwrap() = Instant::now();
            let Ok(value) = serde_json::from_slice::<Value>(&line) else { continue };
            if !self.take(hub, &name, conn, &lines, &value, line.len()) {
                break;
            }
        }
        over.store(true, Ordering::SeqCst);
        close();
        if self.dropped(&name, conn) {
            hub.broadcast(&json!({"type": "sessionsChanged"}));
        }
    }

    /// One line from machine `name`; false ends its link.
    fn take(&self, hub: &Hub, name: &str, conn: u64, lines: &SyncSender<String>, value: &Value, size: usize) -> bool {
        if let Some(client) = value["c"].as_u64() {
            if value["close"] == true {
                self.end_route(name, client);
            } else if let Some(line) = value.get("line").filter(|line| line.is_object()) {
                self.deliver(hub, name, client, line.clone());
            }
            return true;
        }
        match value["type"].as_str() {
            Some("link-sessions") => {
                if size > MAX_SESSIONS_LINE {
                    return false;
                }
                if let Some(changed) = self.store(name, conn, value) {
                    for event in changed {
                        hub.broadcast(&event);
                    }
                    hub.broadcast(&json!({"type": "sessionsChanged"}));
                }
            }
            Some("link-ping") => {
                let _ = lines.try_send(json!({"type": "link-pong"}).to_string());
            }
            // Heard, and that is all it is for; every other type is ignored.
            _ => {}
        }
        true
    }

    /// A line of machine `name` for hub client `client`, if the hub asked that client's way.
    fn deliver(&self, hub: &Hub, name: &str, client: u64, mut line: Value) {
        // The hub's own `hello` is the one a client has; a second is none of a machine's business.
        if line["type"] == "hello" || !self.has_route(name, client) {
            return;
        }
        if line["type"] == "sessionAdded" && line["session"].is_object() {
            line["session"]["machine"] = json!(name);
        }
        hub.send_line(client, &line);
    }
}

fn hub_hello() -> Value {
    json!({"type": "link-hello", "protocol": PROTOCOL, "version": env!("CARGO_PKG_VERSION")})
}

fn status_event(id: &str, status: &Value) -> Value {
    json!({"type": "session", "session": id, "message": {
        "type": "sessionStatus", "session": id, "status": status["status"], "at": status["at"]}})
}

/// A short printable text from a machine's hello (its OS, its version): anything else is dropped.
fn plain(value: &Value, max: usize) -> String {
    value.as_str().unwrap_or_default().chars().filter(|c| !c.is_control()).take(max).collect()
}

/// One short JSON object from a connection that has not said who it is yet.
fn read_json(reader: &mut BufReader<UnixStream>) -> Option<Value> {
    match next_line(reader, MAX_HANDSHAKE_LINE) {
        Ok(Next::Line(line)) => serde_json::from_slice::<Value>(&line).ok().filter(Value::is_object),
        _ => None,
    }
}

// ---- `aria hub-link machine <name>`: the forced command of a machine's key ------------------------------------------

/// The machine name `aria hub-link` was asked for, or why it was refused: exactly `machine <name>`, with a valid name.
/// Nothing else is read: the command a key forces comes from the hub's `authorized_keys`, and `SSH_ORIGINAL_COMMAND`
/// (what the connecting side asked to run) is never consulted.
pub fn link_args(args: &[String]) -> Result<String, &'static str> {
    match args {
        [kind, name] if kind == "machine" && valid_name(name) => Ok(name.clone()),
        _ => Err("usage: aria hub-link machine <name> (lowercase letters, digits and dashes, at most 31)"),
    }
}

/// `aria hub-link <args>`: the bridge between a machine's ssh and the hub's machines socket. Prints nothing to stdout
/// unless the hub does.
pub fn hub_link_command(args: &[String]) -> i32 {
    run_link(args, std::io::stdin(), std::io::stdout(), &socket_path()) as i32
}

fn run_link(
    args: &[String],
    input: impl Read + Send + 'static,
    output: impl Write + Send + 'static,
    socket: &std::path::Path,
) -> Exit {
    let name = match link_args(args) {
        Ok(name) => name,
        Err(usage) => {
            eprintln!("{usage}");
            return Exit::Usage;
        }
    };
    let Ok(hub) = UnixStream::connect(socket) else {
        eprintln!("aria hub-link: no hub is running here");
        return Exit::HubDown;
    };
    let (Ok(mut to_hub), Ok(from_hub), Ok(closer)) = (hub.try_clone(), hub.try_clone(), hub.try_clone()) else {
        return Exit::HubDown;
    };
    if write_line(&mut to_hub, &json!({"type": "link-bridge", "name": name})).is_err() {
        return Exit::HubDown;
    }
    relay(BufReader::new(input), output, to_hub, from_hub, closer, |_| true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::Hub;
    use std::io::BufRead;
    use std::time::Instant;

    fn wait_until(what: &str, done: impl Fn() -> bool) {
        let start = Instant::now();
        while !done() {
            assert!(start.elapsed() < Duration::from_secs(5), "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// One end of a connection, for a hub client or a machine: lines out, lines in (with a way to see that none came).
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
        fn next(&mut self) -> Value {
            let mut line = String::new();
            assert!(self.lines.read_line(&mut line).unwrap() > 0, "the connection closed");
            serde_json::from_str(&line).unwrap()
        }
        /// The next line, or None when the connection ends or stays quiet for `wait`.
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
        /// Everything that is not a heartbeat line, until `wait` of quiet.
        fn quiet(&mut self, wait: Duration) -> bool {
            while let Some(line) = self.next_within(wait) {
                if !matches!(line["type"].as_str(), Some("link-ping" | "link-pong")) {
                    return false;
                }
            }
            true
        }
        fn closed(&mut self) -> bool {
            let mut line = String::new();
            matches!(self.lines.read_line(&mut line), Ok(0) | Err(_))
        }
    }

    struct World {
        hub: Hub,
        socket: PathBuf,
        _dir: tempfile::TempDir,
    }

    fn world_with(timing: Timing) -> World {
        let (chat, _) = mpsc::channel();
        let hub = Hub::new(chat, Arc::new(|_| {}), Arc::new(|id| id.starts_with("hub-"))).with_timing(timing);
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("run").join("machines.sock");
        hub.serve_machines(crate::clients::bind(&socket).unwrap());
        World { hub, socket, _dir: dir }
    }

    fn world() -> World {
        world_with(Timing::default())
    }

    impl World {
        /// A machine that has linked: the bridge's preamble, its hello and the hub's answer read.
        fn machine(&self, name: &str) -> Peer {
            let mut peer = self.machine_raw();
            peer.send(json!({"type": "link-bridge", "name": name}));
            peer.send(
                json!({"type": "link-hello", "protocol": PROTOCOL, "name": name, "os": "linux", "version": "9.9.9"}),
            );
            let hello = peer.next();
            assert_eq!(hello["type"], "link-hello");
            assert_eq!(hello["protocol"], PROTOCOL);
            peer
        }
        fn machine_raw(&self) -> Peer {
            Peer::new(UnixStream::connect(&self.socket).unwrap())
        }
        /// A hub client, as `clients.rs` serves them; `remote` marks it as one that came in over ssh.
        fn client(&self, remote: bool) -> Peer {
            let (ours, theirs) = UnixStream::pair().unwrap();
            self.hub.add(theirs);
            let mut peer = Peer::new(ours);
            peer.send(json!({"type": "hello", "protocol": PROTOCOL, "version": "x", "remote": remote}));
            assert_eq!(peer.next()["type"], "hello");
            peer
        }
        fn sessions_of(&self, machine: &mut Peer, ids: &[&str]) {
            let sessions: Vec<Value> = ids.iter().map(|id| json!({"id": id, "title": format!("title {id}")})).collect();
            machine.send(json!({"type": "link-sessions", "sessions": sessions, "statuses": {}}));
            wait_until("the cache", || ids.iter().all(|id| self.hub.machines().owner_of(id).is_some()));
        }
    }

    fn chat_send(session: &str, text: &str) -> Value {
        json!({"type": "session", "session": session, "message": {"type": "chatSend", "text": text}})
    }

    // ---- the handshake ------------------------------------------------------------------------------------------

    #[test]
    fn a_machine_that_says_hello_is_answered_with_the_hubs_own() {
        let world = world();
        let _machine = world.machine("laptop");
        wait_until("the machine to be listed", || world.hub.machines().list().len() == 1);
        let listed = &world.hub.machines().list()[0];
        assert_eq!((listed.name.as_str(), listed.os.as_str(), listed.version.as_str()), ("laptop", "linux", "9.9.9"));
        assert!(listed.online);
    }

    #[test]
    fn another_protocol_is_answered_and_refused() {
        let world = world();
        let mut peer = world.machine_raw();
        peer.send(json!({"type": "link-bridge", "name": "laptop"}));
        peer.send(json!({"type": "link-hello", "protocol": PROTOCOL + 1, "name": "laptop"}));
        assert_eq!(peer.next()["type"], "link-hello");
        assert_eq!(peer.next()["code"], "protocol-mismatch");
        assert!(peer.closed());
        assert!(world.hub.machines().list().is_empty(), "a refused machine is not listed");
    }

    #[test]
    fn the_name_in_the_hello_must_be_the_one_the_key_gave() {
        let world = world();
        let mut peer = world.machine_raw();
        peer.send(json!({"type": "link-bridge", "name": "laptop"}));
        peer.send(json!({"type": "link-hello", "protocol": PROTOCOL, "name": "someone-else"}));
        assert_eq!(peer.next()["code"], "name-mismatch");
        assert!(peer.closed());
        assert!(world.hub.machines().list().is_empty());
    }

    #[test]
    fn a_connection_that_does_not_start_with_the_bridges_preamble_is_closed() {
        let world = world();
        for first in [
            json!({"type": "link-hello", "protocol": PROTOCOL, "name": "laptop"}),
            json!({"type": "link-bridge", "name": "Bad Name"}),
            json!({"type": "link-bridge", "name": ""}),
            json!({"type": "hello", "protocol": PROTOCOL}),
        ] {
            let mut peer = world.machine_raw();
            peer.send(first);
            assert!(peer.closed());
        }
        assert!(world.hub.machines().list().is_empty());
    }

    #[test]
    fn a_second_link_with_the_same_name_replaces_the_first() {
        let world = world();
        let mut old = world.machine("laptop");
        world.sessions_of(&mut old, &["m-1"]);
        let mut new = world.machine("laptop");
        assert!(old.closed(), "the old link is closed");
        wait_until("the new link", || world.hub.machines().list()[0].online);
        // The old link ending must not mark the machine offline: the new one is the machine now.
        std::thread::sleep(Duration::from_millis(100));
        assert!(world.hub.machines().list()[0].online);
        assert_eq!(world.hub.machines().list().len(), 1);
        world.sessions_of(&mut new, &["m-2"]);
        assert_eq!(world.hub.machines().owner_of("m-2").as_deref(), Some("laptop"));
    }

    // ---- the list -----------------------------------------------------------------------------------------------

    #[test]
    fn the_list_a_client_gets_has_the_machines_sessions_tagged() {
        let world = world();
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["m-1", "m-2"]);
        let mut reply =
            json!({"type": "sessions", "id": 3, "sessions": [{"id": "hub-1", "title": "mine"}], "active": "hub-1"});
        world.hub.machines().merge_into(&mut reply);
        let ids: Vec<&str> = reply["sessions"].as_array().unwrap().iter().map(|e| e["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["hub-1", "m-1", "m-2"]);
        assert!(reply["sessions"][0].get("machine").is_none(), "the hub's own carry no machine");
        assert_eq!(reply["sessions"][1]["machine"], "laptop");
        assert!(reply["sessions"][1].get("offline").is_none());
        assert_eq!(reply["active"], "hub-1");
    }

    #[test]
    fn an_offline_machines_sessions_stay_listed_as_offline() {
        let world = world();
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["m-1"]);
        drop(machine);
        wait_until("the machine to go offline", || !world.hub.machines().list()[0].online);
        let mut reply = json!({"type": "sessions", "id": 1, "sessions": [], "active": null});
        world.hub.machines().merge_into(&mut reply);
        assert_eq!(reply["sessions"][0]["id"], "m-1");
        assert_eq!(reply["sessions"][0]["offline"], true);
        assert_eq!(reply["sessions"][0]["machine"], "laptop");
    }

    #[test]
    fn a_session_id_the_hub_has_is_not_listed_twice_and_a_machine_cannot_claim_a_hub_session() {
        let world = world();
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["hub-1", "m-1"]);
        let mut reply = json!({"type": "sessions", "id": 1, "sessions": [{"id": "hub-1"}], "active": null});
        world.hub.machines().merge_into(&mut reply);
        assert_eq!(reply["sessions"].as_array().unwrap().len(), 2);
        // `hub-1` is the hub's: a client's line for it never travels to the machine.
        let mut client = world.client(true);
        client.send(chat_send("hub-1", "hi"));
        let mut quiet = machine;
        assert!(quiet.quiet(Duration::from_millis(150)));
    }

    #[test]
    fn a_machine_cannot_set_the_tags_the_hub_adds() {
        let world = world();
        let mut machine = world.machine("laptop");
        machine.send(json!({"type": "link-sessions", "sessions": [{"id": "m-1", "machine": "other", "offline": true}], "statuses": {}}));
        wait_until("the cache", || world.hub.machines().owner_of("m-1").is_some());
        let mut reply = json!({"sessions": []});
        world.hub.machines().merge_into(&mut reply);
        assert_eq!(reply["sessions"][0]["machine"], "laptop");
        assert!(reply["sessions"][0].get("offline").is_none());
    }

    #[test]
    fn a_change_of_the_cache_tells_every_client_to_read_the_list_again_and_statuses_follow() {
        let world = world();
        let mut client = world.client(false);
        let mut machine = world.machine("laptop");
        assert_eq!(client.next()["type"], "sessionsChanged", "a machine linking changes the list");
        machine.send(json!({"type": "link-sessions", "sessions": [{"id": "m-1"}], "statuses": {"m-1": {"status": "working", "at": 7}}}));
        let status = client.next();
        assert_eq!(status["type"], "session");
        assert_eq!(status["session"], "m-1");
        assert_eq!(status["message"], json!({"type": "sessionStatus", "session": "m-1", "status": "working", "at": 7}));
        assert_eq!(client.next()["type"], "sessionsChanged");
        // The same statuses again are not news; the list still changed.
        machine.send(json!({"type": "link-sessions", "sessions": [{"id": "m-1"}], "statuses": {"m-1": {"status": "working", "at": 7}}}));
        assert_eq!(client.next()["type"], "sessionsChanged");
        drop(machine);
        assert_eq!(client.next()["type"], "sessionsChanged", "a machine dropping changes the list");
        let events = world.hub.machines().status_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["message"]["status"], "working");
    }

    // ---- routing ------------------------------------------------------------------------------------------------

    #[test]
    fn two_clients_one_machine_each_gets_only_its_own_replies_with_its_own_ids() {
        let world = world();
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["m-1"]);
        let mut a = world.client(true);
        let mut b = world.client(true);
        a.send(json!({"type": "session", "session": "m-1", "message": {"type": "listFiles", "id": 41, "query": "x"}}));
        b.send(json!({"type": "session", "session": "m-1", "message": {"type": "listFiles", "id": 41, "query": "y"}}));
        let first = machine.next();
        let second = machine.next();
        let (ca, cb) = (first["c"].as_u64().unwrap(), second["c"].as_u64().unwrap());
        assert_ne!(ca, cb, "each client has its own number");
        // The ids are the clients' own: the machine rewrites them, as its own hub does for a client.
        assert_eq!(first["line"]["message"]["id"], 41);
        assert_eq!(second["line"]["message"]["id"], 41);
        assert_eq!(first["line"]["message"]["query"], "x");
        machine.send(json!({"c": ca, "line": {"type": "files", "id": 41, "files": ["for-a"]}}));
        machine.send(json!({"c": cb, "line": {"type": "files", "id": 41, "files": ["for-b"]}}));
        assert_eq!(a.next()["files"], json!(["for-a"]));
        assert_eq!(b.next()["files"], json!(["for-b"]));
        assert!(a.quiet(Duration::from_millis(100)) && b.quiet(Duration::from_millis(100)));
    }

    #[test]
    fn attach_detach_and_focus_follow_the_session_they_name() {
        let world = world();
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["m-1"]);
        let mut client = world.client(false);
        client.send(json!({"type": "attach", "session": "m-1"}));
        let attach = machine.next();
        assert_eq!(attach["line"], json!({"type": "attach", "session": "m-1"}));
        client.send(json!({"type": "focus", "focused": false}));
        assert_eq!(machine.next()["line"], json!({"type": "focus", "focused": false}));
        client.send(json!({"type": "detach"}));
        assert_eq!(machine.next()["line"], json!({"type": "detach"}));
        // Once detached, a focus names nothing of the machine's.
        client.send(json!({"type": "focus", "focused": true}));
        assert!(machine.quiet(Duration::from_millis(100)));
        // The hub's own session is attached without the machine hearing of it.
        client.send(json!({"type": "attach", "session": "hub-1"}));
        assert!(machine.quiet(Duration::from_millis(100)));
    }

    #[test]
    fn what_a_remote_client_may_not_send_is_refused_before_it_reaches_a_machine() {
        let world = world();
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["m-1"]);
        let mut client = world.client(true);
        client.send(json!({"type": "session", "session": "m-1", "message": {"type": "piCommand", "id": 5, "command": "bash", "args": {"command": "ls"}}}));
        let refused = client.next();
        assert_eq!((refused["code"].as_str(), refused["id"].as_u64()), (Some("not-for-remote"), Some(5)));
        client.send(json!({"type": "session", "session": "m-1", "message": {"type": "jobStop", "job": "x"}}));
        assert_eq!(client.next()["code"], "not-for-clients");
        assert!(machine.quiet(Duration::from_millis(100)));
    }

    #[test]
    fn a_line_for_a_machine_that_is_offline_is_answered_with_machine_offline() {
        let world = world();
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["m-1"]);
        drop(machine);
        wait_until("offline", || !world.hub.machines().list()[0].online);
        let mut client = world.client(false);
        client
            .send(json!({"type": "session", "session": "m-1", "message": {"type": "listFiles", "id": 9, "query": ""}}));
        let refused = client.next();
        assert_eq!((refused["code"].as_str(), refused["id"].as_u64()), (Some("machine-offline"), Some(9)));
    }

    #[test]
    fn a_session_new_goes_to_the_machine_it_names() {
        let world = world();
        let mut machine = world.machine("laptop");
        let mut client = world.client(true);
        client.send(json!({"type": "sessionNew", "id": 12, "machine": "laptop", "workdir": "/etc"}));
        let forwarded = machine.next();
        assert_eq!(forwarded["line"]["type"], "sessionNew");
        assert_eq!(forwarded["line"]["id"], 12);
        assert!(forwarded["line"].get("machine").is_none(), "the machine is the one asked: its name is not sent on");
        let c = forwarded["c"].as_u64().unwrap();
        // The machine's `sessionAdded` for that client comes back tagged with where it lives.
        machine.send(
            json!({"c": c, "line": {"type": "sessionAdded", "id": 12, "session": {"id": "m-9", "title": "new"}}}),
        );
        let added = client.next();
        assert_eq!(added["id"], 12);
        assert_eq!(added["session"]["machine"], "laptop");
    }

    #[test]
    fn a_session_new_for_a_machine_that_is_offline_or_unknown_says_which() {
        let world = world();
        let machine = world.machine("laptop");
        drop(machine);
        wait_until("offline", || !world.hub.machines().list()[0].online);
        let mut client = world.client(false);
        client.send(json!({"type": "sessionNew", "id": 4, "machine": "laptop"}));
        let offline = client.next();
        assert_eq!((offline["code"].as_str(), offline["id"].as_u64()), (Some("machine-offline"), Some(4)));
        client.send(json!({"type": "sessionNew", "id": 5, "machine": "never-seen"}));
        assert_eq!(client.next()["code"], "unknown-machine");
        client.send(json!({"type": "sessionNew", "id": 6, "machine": "Bad Name"}));
        assert_eq!(client.next()["code"], "unknown-machine");
        client.send(json!({"type": "sessionNew", "id": 7, "machine": ""}));
        assert_eq!(client.next()["code"], "unknown-machine");
    }

    #[test]
    fn a_session_new_without_a_machine_goes_to_the_hub_as_before() {
        let (chat, handled) = mpsc::channel();
        let hub = Hub::new(chat, Arc::new(|_| {}), Arc::new(|_| true));
        let (ours, theirs) = UnixStream::pair().unwrap();
        hub.add(theirs);
        let mut client = Peer::new(ours);
        client.send(json!({"type": "sessionNew", "id": 2}));
        let message = handled.recv_timeout(Duration::from_secs(5)).unwrap();
        // The hub gives the request an id of its own on the way in, as for any client: it is not the client's 2.
        assert!(matches!(message, aria_core::bridge::PageMessage::SessionNew { .. }), "{message:?}");
    }

    #[test]
    fn a_machine_can_only_speak_to_a_client_it_was_asked_by() {
        let world = world();
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["m-1"]);
        let mut asked = world.client(true);
        let mut stranger = world.client(true);
        asked.send(chat_send("m-1", "hi"));
        let c = machine.next()["c"].as_u64().unwrap();
        // A client number nobody has, and one that exists but never used this link.
        machine.send(json!({"c": 999_999, "line": {"type": "notice", "text": "to nobody"}}));
        let stranger_number = c + 1;
        machine.send(json!({"c": stranger_number, "line": {"type": "notice", "text": "to a stranger"}}));
        machine.send(json!({"c": c, "line": {"type": "hello", "protocol": 1}}));
        machine.send(json!({"c": c, "line": {"type": "notice", "text": "to the asker"}}));
        assert_eq!(asked.next()["text"], "to the asker", "a hello from a machine is not passed on either");
        assert!(stranger.quiet(Duration::from_millis(100)));
        // After the route is ended by the machine, nothing more goes to that client.
        machine.send(json!({"c": c, "close": true}));
        machine.send(json!({"c": c, "line": {"type": "notice", "text": "after the end"}}));
        assert!(asked.quiet(Duration::from_millis(150)));
    }

    #[test]
    fn a_machine_cannot_make_the_hub_run_what_a_client_would() {
        let (chat, handled) = mpsc::channel();
        let (handlers_tx, handlers) = mpsc::channel();
        let handlers_tx = Mutex::new(handlers_tx);
        let hub = Hub::new(
            chat,
            Arc::new(move |message| {
                let _ = handlers_tx.lock().unwrap().send(message);
            }),
            Arc::new(|_| true),
        );
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("run").join("machines.sock");
        hub.serve_machines(crate::clients::bind(&socket).unwrap());
        let mut machine = Peer::new(UnixStream::connect(&socket).unwrap());
        machine.send(json!({"type": "link-bridge", "name": "laptop"}));
        machine.send(json!({"type": "link-hello", "protocol": PROTOCOL, "name": "laptop"}));
        assert_eq!(machine.next()["type"], "link-hello");
        for line in [
            json!({"type": "sessionNew", "id": 1}),
            json!({"type": "hubUpdate", "id": 1, "version": "99.0.0"}),
            json!({"type": "secretSave", "id": 1, "account": "a", "value": "b"}),
            json!({"type": "chatSend", "text": "x"}),
            json!({"type": "session", "session": "x", "message": {"type": "chatSend", "text": "x"}}),
            json!({"type": "attach", "session": "x"}),
            json!({"type": "hello", "protocol": 1}),
        ] {
            machine.send(line);
        }
        // A ping after them, answered: the hub read every line and nothing of it reached the chat or the handlers.
        machine.send(json!({"type": "link-ping"}));
        assert_eq!(machine.next()["type"], "link-pong");
        assert!(handled.recv_timeout(Duration::from_millis(200)).is_err(), "the chat saw something a machine sent");
        assert!(handlers.recv_timeout(Duration::from_millis(50)).is_err(), "a handler saw something a machine sent");
    }

    #[test]
    fn a_client_that_goes_away_tells_every_machine_it_had_a_route_to() {
        let world = world();
        let mut laptop = world.machine("laptop");
        let mut mac = world.machine("mac");
        world.sessions_of(&mut laptop, &["m-1"]);
        world.sessions_of(&mut mac, &["x-1"]);
        let mut client = world.client(true);
        client.send(chat_send("m-1", "hi"));
        let c = laptop.next()["c"].as_u64().unwrap();
        drop(client);
        assert_eq!(laptop.next(), json!({"c": c, "close": true}));
        assert!(mac.quiet(Duration::from_millis(150)), "a machine it never used hears nothing");
    }

    // ---- the link's health --------------------------------------------------------------------------------------

    #[test]
    fn the_hub_pings_and_a_silent_machine_is_dropped() {
        let world =
            world_with(Timing { ping: Duration::from_millis(30), silent: Duration::from_millis(200), queue: 64 });
        let mut machine = world.machine("laptop");
        assert_eq!(machine.next()["type"], "link-ping");
        // Answering keeps it alive past the silence limit.
        for _ in 0..12 {
            std::thread::sleep(Duration::from_millis(30));
            machine.send(json!({"type": "link-pong"}));
        }
        assert!(world.hub.machines().list()[0].online);
        // Saying nothing does not.
        wait_until("the silent machine to be dropped", || !world.hub.machines().list()[0].online);
    }

    #[test]
    fn a_machines_ping_is_answered_with_a_pong() {
        let world = world();
        let mut machine = world.machine("laptop");
        machine.send(json!({"type": "link-ping"}));
        assert_eq!(machine.next()["type"], "link-pong");
    }

    #[test]
    fn a_cache_over_the_cap_drops_the_link_and_keeps_the_hub() {
        let world = world();
        let mut machine = world.machine("laptop");
        let big = "x".repeat(MAX_SESSIONS_LINE + 10);
        machine.send(json!({"type": "link-sessions", "sessions": [{"id": "m-1", "title": big}], "statuses": {}}));
        assert!(machine.closed());
        wait_until("offline", || !world.hub.machines().list()[0].online);
        assert!(world.hub.machines().owner_of("m-1").is_none(), "nothing of the oversized line was kept");
    }

    #[test]
    fn a_machine_that_does_not_read_is_dropped_and_the_hub_goes_on() {
        let world = world_with(Timing { ping: Duration::from_secs(60), silent: Duration::from_secs(60), queue: 4 });
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["m-1"]);
        let mut client = world.client(true);
        let big = "y".repeat(256 * 1024);
        // The machine never reads: its socket fills, then its queue, and the next line finds it full.
        for _ in 0..200 {
            client.send(chat_send("m-1", &big));
            if !world.hub.machines().list()[0].online {
                break;
            }
        }
        wait_until("the slow machine to be dropped", || !world.hub.machines().list()[0].online);
        // The hub still answers its other clients.
        let mut other = world.client(false);
        other.send(json!({"type": "focus", "focused": true}));
        let mut third = world.machine("mac");
        third.send(json!({"type": "link-ping"}));
        assert_eq!(third.next()["type"], "link-pong");
    }

    // ---- aria hub-link ------------------------------------------------------------------------------------------

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|text| text.to_string()).collect()
    }

    #[test]
    fn names_are_lowercase_letters_digits_and_dashes() {
        for good in ["laptop", "a", "0", "my-mac-2", "a-b", &"a".repeat(31)] {
            assert!(valid_name(good), "{good}");
        }
        for bad in
            ["", "-a", "A", "my_mac", "my mac", "a.b", "a/b", "ñ", &"a".repeat(32), "a\n", "a;ls", "$(x)", "../x"]
        {
            assert!(!valid_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn hub_link_takes_exactly_machine_and_a_valid_name() {
        assert_eq!(link_args(&args(&["machine", "laptop"])), Ok("laptop".to_string()));
        for bad in [
            args(&[]),
            args(&["machine"]),
            args(&["machine", "Laptop"]),
            args(&["machine", "a", "b"]),
            args(&["client", "laptop"]),
            args(&["laptop"]),
            args(&["machine", "a;rm -rf ~"]),
            args(&["machine", ""]),
        ] {
            assert!(link_args(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn hub_link_never_reads_what_the_connecting_side_asked_to_run() {
        // Whatever ssh offers as the original command, the name is the one on the key's own command line.
        std::env::set_var("SSH_ORIGINAL_COMMAND", "aria hub-link machine somebody-else");
        assert_eq!(link_args(&args(&["machine", "laptop"])), Ok("laptop".to_string()));
        assert!(link_args(&args(&[])).is_err());
        std::env::remove_var("SSH_ORIGINAL_COMMAND");
    }

    #[test]
    fn the_bridge_writes_the_preamble_first_and_relays_both_ways() {
        let world = world();
        let (mut machine_in, bridge_in) = UnixStream::pair().unwrap();
        let (bridge_out, machine_out) = UnixStream::pair().unwrap();
        let socket = world.socket.clone();
        let link = std::thread::spawn(move || run_link(&args(&["machine", "laptop"]), bridge_in, bridge_out, &socket));
        // The machine's side of ssh: its hello, from stdin; the hub's answer on stdout.
        writeln!(
            machine_in,
            "{}",
            json!({"type": "link-hello", "protocol": PROTOCOL, "name": "laptop", "os": "linux", "version": "1"})
        )
        .unwrap();
        let mut answer = BufReader::new(machine_out);
        let mut line = String::new();
        answer.read_line(&mut line).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&line).unwrap()["type"], "link-hello");
        wait_until("the machine to be listed", || world.hub.machines().list().len() == 1);
        // A line that is not a JSON object never reaches the hub.
        writeln!(machine_in, "not json").unwrap();
        writeln!(machine_in, "{}", json!({"type": "link-ping"})).unwrap();
        line.clear();
        answer.read_line(&mut line).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&line).unwrap()["type"], "link-pong");
        drop(machine_in);
        assert_eq!(link.join().unwrap(), Exit::Done);
    }

    #[test]
    fn the_name_on_the_key_beats_the_name_a_machine_claims() {
        let world = world();
        let (mut machine_in, bridge_in) = UnixStream::pair().unwrap();
        let (bridge_out, machine_out) = UnixStream::pair().unwrap();
        let socket = world.socket.clone();
        let link = std::thread::spawn(move || run_link(&args(&["machine", "laptop"]), bridge_in, bridge_out, &socket));
        // A machine whose key says "laptop" cannot become "mac" by saying so.
        writeln!(machine_in, "{}", json!({"type": "link-hello", "protocol": PROTOCOL, "name": "mac"})).unwrap();
        let mut answer = BufReader::new(machine_out);
        let mut line = String::new();
        answer.read_line(&mut line).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&line).unwrap()["code"], "name-mismatch");
        let _ = link.join();
        assert!(world.hub.machines().list().is_empty());
    }

    #[test]
    fn hub_link_with_no_hub_says_so_on_stderr_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let (_in, bridge_in) = UnixStream::pair().unwrap();
        let (bridge_out, mut seen) = UnixStream::pair().unwrap();
        let exit = run_link(&args(&["machine", "laptop"]), bridge_in, bridge_out, &dir.path().join("none.sock"));
        assert_eq!(exit, Exit::HubDown);
        let mut text = String::new();
        seen.set_read_timeout(Some(Duration::from_millis(100))).unwrap();
        let _ = seen.read_to_string(&mut text);
        assert_eq!(text, "");
    }

    // ---- who is linked ------------------------------------------------------------------------------------------------

    #[test]
    fn any_client_can_ask_who_is_linked_and_gets_its_own_id_back() {
        let world = world();
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["m-1", "m-2"]);
        for remote in [false, true] {
            let mut client = world.client(remote);
            client.send(json!({"type": "machinesList", "id": 11}));
            let reply = client.next();
            assert_eq!(reply["type"], "machines", "{reply}");
            assert_eq!(reply["id"], 11, "the client's own id: {reply}");
            let listed = &reply["machines"][0];
            assert_eq!(listed["name"], "laptop");
            assert_eq!((listed["os"].as_str(), listed["version"].as_str()), (Some("linux"), Some("9.9.9")));
            assert_eq!((listed["online"].as_bool(), listed["sessions"].as_u64()), (Some(true), Some(2)));
            assert!(listed["since"].as_u64().is_some_and(|since| since > 1_600_000_000), "{listed}");
            assert_eq!(reply["machines"].as_array().unwrap().len(), 1);
        }
    }

    #[test]
    fn a_machine_that_dropped_is_listed_offline_and_nobody_is_listed_before_any_links() {
        let world = world();
        let mut client = world.client(true);
        client.send(json!({"type": "machinesList", "id": 1}));
        assert_eq!(client.next()["machines"], json!([]));
        let mut machine = world.machine("laptop");
        world.sessions_of(&mut machine, &["m-1"]);
        drop(machine);
        wait_until("the machine to go offline", || world.hub.machines().list().iter().all(|machine| !machine.online));
        client.send(json!({"type": "machinesList", "id": 2}));
        let reply = loop {
            let line = client.next();
            if line["type"] == "machines" {
                break line;
            }
        };
        assert_eq!(reply["machines"][0]["online"], false, "{reply}");
        assert_eq!(reply["machines"][0]["sessions"], 1, "its cache stays");
    }
}
