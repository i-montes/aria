//! The socket a session's Pi reaches its background work through: the session's `background` extension starts, lists,
//! reads and stops jobs here, and hears how they ended. One socket per session, beside the session's watch socket,
//! mode 0600; JSON lines both ways.
//!
//! Requests carry an `id` and are answered with the same `re`: `{"re":n,"ok":true,…}` or
//! `{"re":n,"ok":false,"error":"…"}`. `ack` is not answered. After `hello` the connection also gets the session's job
//! events (`{"event":"ended"|"cut","seq":n,…}`), first those not yet acknowledged, then new ones as they happen. One
//! connection per session gets them: a later `hello` takes them over, and the earlier connection keeps answering
//! requests.

use crate::jobs::{AgentStart, Job, JobEvent, Jobs, ProcessStart};
use crate::lines::read_bounded_line;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

/// Bytes read of one request line before the rest is skipped.
const MAX_LINE: u64 = 1024 * 1024;
/// Output lines `output` gives when the request names no number, and the most it gives.
const OUTPUT_DEFAULT: usize = 60;
const OUTPUT_MAX: usize = 400;
/// How long a write to a Pi that does not read may block before the connection is given up.
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
/// How often a connection's event forwarder checks that its connection is still there.
const FORWARD_POLL: Duration = Duration::from_millis(200);

/// Where the session's jobs socket lives: `jobs-<session>.sock`, or `j-<16 hex>.sock` when that would pass the Unix
/// socket path limit (the same rule as `watch::socket_path`).
pub fn socket_path(dir: &Path, session: &str) -> PathBuf {
    const LIMIT: usize = 100;
    let path = dir.join(format!("jobs-{session}.sock"));
    if path.as_os_str().len() <= LIMIT {
        return path;
    }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    session.hash(&mut hasher);
    dir.join(format!("j-{:016x}.sock", hasher.finish()))
}

/// How a subagent is run: the program (the session's `pi` shim), its extra arguments (`--model`/`--thinking`), the env
/// to add and the names to remove.
pub type AgentLaunch = (PathBuf, Vec<String>, Vec<(String, String)>, Vec<String>);

/// How the dock starts a subagent for this session.
pub trait Launcher: Send + Sync {
    /// How to run one for `model` `"session"` or `"heavy"`; `None` when a subagent cannot be started now.
    fn agent(&self, model: &str) -> Option<AgentLaunch>;
    /// Runs right before the subagent's child is spawned.
    fn before_spawn(&self, job: &Job);
}

/// The session's jobs socket. Dropping it removes the file and closes its connections.
pub struct JobsSocket {
    path: PathBuf,
    stop: Arc<AtomicBool>,
    clients: Arc<Mutex<HashMap<u64, UnixStream>>>,
}

/// What a connection needs to answer.
struct Session {
    name: String,
    jobs: Jobs,
    launcher: Arc<dyn Launcher>,
}

/// A connection's writing half, shared by answers and pushed events. `written` is the highest event `seq` this
/// connection was given since its last `hello`, so an event both queued and pushed goes out once.
struct Writer {
    stream: UnixStream,
    written: u64,
}

impl Writer {
    fn line(&mut self, value: &Value) -> std::io::Result<()> {
        let mut bytes = serde_json::to_vec(value)?;
        bytes.push(b'\n');
        self.stream.write_all(&bytes)
    }
}

impl JobsSocket {
    /// Listens on `path` (mode 0600, replacing a stale file) for `session`'s Pi.
    pub fn bind(path: &Path, session: String, jobs: Jobs, launcher: Arc<dyn Launcher>) -> std::io::Result<JobsSocket> {
        if let Some(dir) = path.parent().filter(|dir| !dir.exists()) {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
        }
        let _ = std::fs::remove_file(path);
        let listener = UnixListener::bind(path)?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        listener.set_nonblocking(true)?;
        let stop = Arc::new(AtomicBool::new(false));
        let clients: Arc<Mutex<HashMap<u64, UnixStream>>> = Arc::default();
        let session = Arc::new(Session { name: session, jobs, launcher });
        {
            let (stop, clients) = (stop.clone(), clients.clone());
            let next = AtomicU64::new(0);
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            // BSD (macOS) inherits the listener's non-blocking flag when accepting; Linux does not.
                            let _ = stream.set_nonblocking(false);
                            let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
                            let id = next.fetch_add(1, Ordering::SeqCst);
                            let Ok(kept) = stream.try_clone() else { continue };
                            clients.lock().unwrap().insert(id, kept);
                            // Dropped while this was accepting: close what was just taken.
                            if stop.load(Ordering::SeqCst) {
                                let _ = stream.shutdown(std::net::Shutdown::Both);
                            }
                            let (session, clients) = (session.clone(), clients.clone());
                            std::thread::spawn(move || {
                                serve(stream, &session);
                                clients.lock().unwrap().remove(&id);
                            });
                        }
                        Err(_) => std::thread::sleep(Duration::from_millis(50)),
                    }
                }
            });
        }
        Ok(JobsSocket { path: path.to_path_buf(), stop, clients })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for JobsSocket {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = std::fs::remove_file(&self.path);
        for stream in self.clients.lock().unwrap().values() {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
    }
}

/// One connection: its requests until it closes.
fn serve(stream: UnixStream, session: &Session) {
    let Ok(write_half) = stream.try_clone() else { return };
    let writer = Arc::new(Mutex::new(Writer { stream: write_half, written: 0 }));
    let closed = Arc::new(AtomicBool::new(false));
    let mut reader = BufReader::new(stream);
    loop {
        // One line at a time, at most MAX_LINE bytes: a client that never sends a newline cannot grow the dock's memory.
        let bytes = match read_bounded_line(&mut reader, MAX_LINE) {
            None => break,
            Some(None) => continue,
            Some(Some(bytes)) => bytes,
        };
        let Ok(request) = serde_json::from_slice::<Value>(&bytes) else { continue };
        if !request.is_object() {
            continue;
        }
        let re = request.get("id").cloned().unwrap_or(Value::Null);
        let written = match request["op"].as_str() {
            Some("ack") => {
                if let Some(seq) = request["event"].as_u64() {
                    session.jobs.ack(&session.name, seq);
                }
                Ok(())
            }
            Some("hello") => hello(session, re, &writer, &closed),
            op => {
                let answer = match answer(session, op, &request) {
                    Ok(mut fields) => {
                        fields.insert("re".into(), re);
                        fields.insert("ok".into(), json!(true));
                        Value::Object(fields)
                    }
                    Err(error) => json!({"re": re, "ok": false, "error": error}),
                };
                writer.lock().unwrap().line(&answer)
            }
        };
        if written.is_err() {
            break;
        }
    }
    closed.store(true, Ordering::SeqCst);
    let _ = writer.lock().unwrap().stream.shutdown(std::net::Shutdown::Both);
}

/// Subscribes the connection to the session's events, answers, then gives every event not yet acknowledged. The
/// writer stays locked throughout, so a pushed event waits for them and goes out after, once.
fn hello(session: &Session, re: Value, writer: &Arc<Mutex<Writer>>, closed: &Arc<AtomicBool>) -> std::io::Result<()> {
    let (tx, rx) = mpsc::channel::<JobEvent>();
    let mut w = writer.lock().unwrap();
    // First subscribed, then read: an event that ends in between is both queued and pushed, never neither.
    session.jobs.subscribe(&session.name, tx);
    // Never lowered: an event the old forwarder still holds was given (maybe acked) and stays given.
    w.line(&json!({"re": re, "ok": true}))?;
    for event in session.jobs.events(&session.name) {
        w.line(&serde_json::to_value(&event)?)?;
        w.written = w.written.max(event.seq());
    }
    drop(w);
    let (writer, closed) = (writer.clone(), closed.clone());
    std::thread::spawn(move || forward(rx, &writer, &closed));
    Ok(())
}

/// Writes the events pushed to this connection until it closes or another `hello` takes them over.
fn forward(rx: mpsc::Receiver<JobEvent>, writer: &Mutex<Writer>, closed: &AtomicBool) {
    while !closed.load(Ordering::SeqCst) {
        let event = match rx.recv_timeout(FORWARD_POLL) {
            Ok(event) => event,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let mut w = writer.lock().unwrap();
        if event.seq() <= w.written {
            continue;
        }
        let Ok(value) = serde_json::to_value(&event) else { continue };
        if w.line(&value).is_err() {
            break;
        }
        w.written = event.seq();
    }
}

type Fields = serde_json::Map<String, Value>;

fn fields(value: Value) -> Fields {
    match value {
        Value::Object(map) => map,
        _ => Fields::new(),
    }
}

/// The answer to every request but `hello` and `ack`: its fields beside `re` and `ok`, or the error.
fn answer(session: &Session, op: Option<&str>, request: &Value) -> Result<Fields, String> {
    let jobs = &session.jobs;
    match op {
        Some("start") => {
            let job = match request["kind"].as_str() {
                Some("process") => start_process(session, request)?,
                Some("agent") => start_agent(session, request)?,
                _ => return Err("kind must be \"process\" or \"agent\"".into()),
            };
            Ok(fields(json!({"job": job})))
        }
        Some("list") => Ok(fields(json!({"jobs": jobs.list(Some(&session.name))}))),
        Some("output") => {
            let id = own_job(session, request)?;
            let lines = request["lines"].as_u64().map_or(OUTPUT_DEFAULT, |n| n.min(OUTPUT_MAX as u64) as usize);
            let lines = jobs.output(&id, lines).ok_or_else(|| format!("no job {id} in this session"))?;
            Ok(fields(json!({"lines": lines})))
        }
        Some("stop") => {
            let id = own_job(session, request)?;
            // A job that already ended needs no stop; the client hears so rather than an error.
            if jobs.stop(&id) {
                Ok(Fields::new())
            } else {
                Ok(fields(json!({"already": true})))
            }
        }
        _ => Err("unknown op".into()),
    }
}

/// The `job` the request names, when it is one of this session's.
fn own_job(session: &Session, request: &Value) -> Result<String, String> {
    let id = request["job"].as_str().ok_or("job is missing")?;
    match session.jobs.get(id) {
        Some(job) if job.session == session.name => Ok(id.to_string()),
        _ => Err(format!("no job {id} in this session")),
    }
}

/// The request's `folder` (or `cwd`): absolute, `~` expanded, an existing folder.
fn folder(request: &Value) -> Result<PathBuf, String> {
    let given = request["folder"].as_str().or_else(|| request["cwd"].as_str()).ok_or("folder is missing")?;
    let home = || std::env::var_os("HOME").filter(|h| !h.is_empty()).map(PathBuf::from);
    let path = match given.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with('/') => {
            let home = home().ok_or("HOME is not set, so ~ cannot be expanded")?;
            home.join(rest.trim_start_matches('/'))
        }
        _ => PathBuf::from(given),
    };
    if !path.is_absolute() {
        return Err(format!("folder must be an absolute path: {given}"));
    }
    if !path.is_dir() {
        return Err(format!("no such folder: {given}"));
    }
    Ok(path)
}

/// A string field, or `default` when it is absent; an error when it is there but not a string.
fn text<'a>(request: &'a Value, key: &str, default: Option<&'a str>) -> Result<&'a str, String> {
    match &request[key] {
        Value::Null => default.ok_or_else(|| format!("{key} is missing")),
        Value::String(s) => Ok(s),
        _ => Err(format!("{key} must be a string")),
    }
}

fn start_process(session: &Session, request: &Value) -> Result<Job, String> {
    let command = text(request, "command", None)?;
    // What actually runs: the line the sandbox decided. Never the command itself, which would run it unboxed.
    let line = match &request["line"] {
        Value::String(line) => line.as_str(),
        _ => return Err("line is required".into()),
    };
    let title = text(request, "title", Some(command))?;
    let folder = folder(request)?;
    let env = match &request["env"] {
        Value::Null => Vec::new(),
        Value::Object(map) => map
            .iter()
            // No background work inside background work.
            .filter(|(name, _)| name.as_str() != "ARIA_JOBS_SOCK")
            .map(|(name, value)| match value {
                Value::String(v) => Ok((name.clone(), v.clone())),
                _ => Err(format!("env {name} must be a string")),
            })
            .collect::<Result<_, _>>()?,
        _ => return Err("env must be an object of strings".into()),
    };
    session.jobs.start_process(ProcessStart {
        session: session.name.clone(),
        title: title.to_string(),
        command: command.to_string(),
        line: line.to_string(),
        folder,
        env,
    })
}

fn start_agent(session: &Session, request: &Value) -> Result<Job, String> {
    let task = text(request, "task", None)?;
    let title = text(request, "title", Some(task.lines().next().unwrap_or(task)))?;
    let folder = folder(request)?;
    let writes = match &request["writes"] {
        Value::Null => true,
        Value::Bool(b) => *b,
        _ => return Err("writes must be true or false".into()),
    };
    let isolation = text(request, "isolation", Some("auto"))?;
    if !matches!(isolation, "auto" | "folder" | "worktree") {
        return Err("isolation must be \"auto\", \"folder\" or \"worktree\"".into());
    }
    let model = text(request, "model", Some("session"))?;
    if !matches!(model, "session" | "heavy") {
        return Err("model must be \"session\" or \"heavy\"".into());
    }
    let Some((program, args, env, unset)) = session.launcher.agent(model) else {
        return Err("cannot start a subagent now: the session's Pi is not ready".into());
    };
    let launcher = session.launcher.clone();
    session.jobs.start_agent(AgentStart {
        session: session.name.clone(),
        title: title.to_string(),
        task: task.to_string(),
        folder,
        writes,
        isolation: isolation.to_string(),
        program,
        args,
        env,
        unset,
        before_spawn: Some(Arc::new(move |job: &Job| launcher.before_spawn(job))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::ProcessStart;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// A launcher that hands out a fake `pi`, or nothing.
    struct FakeLauncher {
        pi: Option<PathBuf>,
        spawned: AtomicUsize,
    }

    impl Launcher for FakeLauncher {
        fn agent(&self, _model: &str) -> Option<AgentLaunch> {
            self.pi.clone().map(|pi| (pi, Vec::new(), Vec::new(), Vec::new()))
        }
        fn before_spawn(&self, _job: &Job) {
            self.spawned.fetch_add(1, Ordering::SeqCst);
        }
    }

    const FAKE_PI: &str = r#"#!/bin/sh
echo '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"all green"}],"stopReason":"stop"}}'
"#;

    struct Setup {
        dir: tempfile::TempDir,
        jobs: Jobs,
        launcher: Arc<FakeLauncher>,
    }

    fn setup(with_pi: bool) -> Setup {
        let dir = tempfile::tempdir().unwrap();
        let pi = with_pi.then(|| {
            let path = dir.path().join("pi");
            std::fs::write(&path, FAKE_PI).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            path
        });
        let jobs = Jobs::new(dir.path().join("jobs.json"), Arc::new(|| {}));
        Setup { dir, jobs, launcher: Arc::new(FakeLauncher { pi, spawned: AtomicUsize::new(0) }) }
    }

    fn bind(setup: &Setup, session: &str) -> JobsSocket {
        let path = setup.dir.path().join(format!("{session}.sock"));
        JobsSocket::bind(&path, session.to_string(), setup.jobs.clone(), setup.launcher.clone()).unwrap()
    }

    struct Client {
        reader: BufReader<UnixStream>,
        writer: UnixStream,
    }

    impl Client {
        fn connect(socket: &JobsSocket) -> Client {
            let stream = UnixStream::connect(socket.path()).unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            Client { reader: BufReader::new(stream.try_clone().unwrap()), writer: stream }
        }

        fn send(&mut self, request: Value) {
            writeln!(self.writer, "{request}").unwrap();
        }

        fn read(&mut self) -> Value {
            let mut line = String::new();
            self.reader.read_line(&mut line).unwrap();
            serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line:?}"))
        }

        /// Sends a request and reads lines until its answer, keeping pushed events aside.
        fn ask(&mut self, request: Value) -> Value {
            let id = request["id"].clone();
            self.send(request);
            loop {
                let line = self.read();
                if line["re"] == id {
                    return line;
                }
            }
        }
    }

    fn eventually(what: &str, test: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if test() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("{what}");
    }

    fn start_process(jobs: &Jobs, session: &str, folder: &Path, line: &str) -> Job {
        jobs.start_process(ProcessStart {
            session: session.into(),
            title: line.into(),
            command: line.into(),
            line: line.into(),
            folder: folder.to_path_buf(),
            env: Vec::new(),
        })
        .unwrap()
    }

    #[test]
    fn a_socket_path_fits_the_unix_limit_even_in_a_deep_folder() {
        let short = socket_path(Path::new("/run/user/1000/aria"), "s1");
        assert_eq!(short, PathBuf::from("/run/user/1000/aria/jobs-s1.sock"));
        let deep = PathBuf::from(format!("/Users/someone/{}/.aria/run", "x".repeat(40)));
        let path = socket_path(&deep, &"0".repeat(36));
        assert!(path.as_os_str().len() <= 100, "{}", path.display());
        assert!(path.file_name().unwrap().to_str().unwrap().starts_with("j-"));
        assert_ne!(path, socket_path(&deep, &"1".repeat(36)), "two sessions never share a socket");
    }

    #[test]
    fn hello_then_pending_events() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(false);
        let job = start_process(&s.jobs, "s1", s.dir.path(), "echo bye");
        eventually("the end is an event", || !s.jobs.events("s1").is_empty());
        let seq = s.jobs.events("s1")[0].seq();
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        c.send(json!({"id": 1, "op": "hello"}));
        assert_eq!(c.read(), json!({"re": 1, "ok": true}));
        let event = c.read();
        assert_eq!(event["event"], "ended");
        assert_eq!(event["seq"], seq);
        assert_eq!(event["job"]["id"], job.id.as_str());
        assert_eq!(event["by"], "self");
        assert_eq!(event["lastLines"], json!(["bye"]));
        // Acknowledged: no answer, and the event is not kept.
        c.send(json!({"op": "ack", "event": seq}));
        eventually("the ack lands", || s.jobs.events("s1").is_empty());
    }

    #[test]
    fn start_list_output_stop_a_process() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        assert_eq!(c.ask(json!({"id": 1, "op": "hello"}))["ok"], true);
        let folder = s.dir.path().to_str().unwrap();
        let line = "echo hi; sleep 30";
        let started = c.ask(json!({"id": 2, "op": "start", "kind": "process", "command": line, "line": line,
            "title": "a long one", "folder": folder, "env": {"FOO": "bar"}}));
        assert_eq!(started["ok"], true, "{started}");
        let id = started["job"]["id"].as_str().unwrap().to_string();
        assert_eq!(started["job"]["kind"], "process");
        assert_eq!(started["job"]["state"], "running");
        assert_eq!(started["job"]["command"], line);
        for n in 3.. {
            let output = c.ask(json!({"id": n, "op": "output", "job": id}));
            assert_eq!(output["ok"], true, "{output}");
            if output["lines"] == json!(["hi"]) {
                break;
            }
            assert!(n < 200, "no output: {output}");
            std::thread::sleep(Duration::from_millis(20));
        }
        let list = c.ask(json!({"id": 300, "op": "list"}));
        assert_eq!(list["ok"], true);
        assert_eq!(list["jobs"].as_array().unwrap().len(), 1);
        assert_eq!(list["jobs"][0]["id"], id.as_str());
        assert_eq!(c.ask(json!({"id": 301, "op": "stop", "job": id})), json!({"re": 301, "ok": true}));
        let event = c.read();
        assert_eq!(event["event"], "ended", "{event}");
        assert_eq!(event["job"]["id"], id.as_str());
        assert_eq!(event["by"], "stop");
        assert_eq!(event["job"]["state"], "stopped");
    }

    #[test]
    fn a_process_needs_an_absolute_folder_and_string_env() {
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let relative = c.ask(json!({"id": 1, "op": "start", "kind": "process", "command": "true", "line": "true",
            "title": "t", "folder": "some/where"}));
        assert_eq!(relative["ok"], false);
        assert!(relative["error"].as_str().unwrap().contains("absolute"), "{relative}");
        let folder = s.dir.path().to_str().unwrap();
        let env = c.ask(json!({"id": 2, "op": "start", "kind": "process", "command": "true", "line": "true",
            "title": "t", "folder": folder, "env": {"N": 1}}));
        assert_eq!(env["ok"], false, "{env}");
        assert!(s.jobs.list(None).is_empty(), "nothing started");
    }

    #[test]
    fn the_process_runs_line_and_shows_command() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let started = c.ask(json!({"id": 1, "op": "start", "kind": "process", "command": "say it",
            "line": "echo \"ran $FOO\"", "title": "t", "folder": "~", "env": {"FOO": "bar"}}));
        assert_eq!(started["ok"], true, "{started}");
        assert_eq!(started["job"]["command"], "say it");
        assert_eq!(started["job"]["folder"], "~");
        let id = started["job"]["id"].as_str().unwrap().to_string();
        eventually("it ran the line with the env", || {
            s.jobs.output(&id, 10).unwrap_or_default() == vec!["ran bar".to_string()]
        });
    }

    #[test]
    fn an_agent_start_uses_the_launcher() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(true);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let folder = s.dir.path().to_str().unwrap();
        let started = c.ask(json!({"id": 1, "op": "start", "kind": "agent", "task": "check the tests",
            "title": "check", "folder": folder, "writes": false, "isolation": "auto", "model": "session"}));
        assert_eq!(started["ok"], true, "{started}");
        assert_eq!(started["job"]["kind"], "agent");
        assert_eq!(started["job"]["task"], "check the tests");
        let id = started["job"]["id"].as_str().unwrap().to_string();
        eventually("the agent ends", || s.jobs.get(&id).is_some_and(|j| j.result.is_some()));
        assert_eq!(s.jobs.get(&id).unwrap().result.as_deref(), Some("all green"));
        assert_eq!(s.launcher.spawned.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn without_a_launcher_answer_an_agent_is_refused() {
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let folder = s.dir.path().to_str().unwrap();
        let refused = c.ask(json!({"id": 1, "op": "start", "kind": "agent", "task": "t", "title": "t",
            "folder": folder}));
        assert_eq!(refused["ok"], false, "{refused}");
        assert!(refused["error"].as_str().unwrap().contains("cannot start"), "{refused}");
        assert!(s.jobs.list(None).is_empty());
    }

    #[test]
    fn a_session_sees_only_its_own_jobs() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(false);
        let theirs = start_process(&s.jobs, "s2", s.dir.path(), "sleep 30");
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let list = c.ask(json!({"id": 1, "op": "list"}));
        assert_eq!(list, json!({"re": 1, "ok": true, "jobs": []}));
        let output = c.ask(json!({"id": 2, "op": "output", "job": theirs.id}));
        assert_eq!(output["ok"], false, "{output}");
        let stop = c.ask(json!({"id": 3, "op": "stop", "job": theirs.id}));
        assert_eq!(stop["ok"], false, "{stop}");
        assert!(s.jobs.get(&theirs.id).unwrap().state == crate::jobs::JobState::Running);
        s.jobs.stop(&theirs.id);
    }

    #[test]
    fn the_socket_is_private() {
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mode = std::fs::metadata(socket.path()).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        let path = socket.path().to_path_buf();
        drop(socket);
        assert!(!path.exists(), "dropping the socket removes its file");
    }

    #[test]
    fn a_folder_it_makes_for_the_socket_is_private_too() {
        let s = setup(false);
        let path = s.dir.path().join("run").join("aria").join("j.sock");
        let socket = JobsSocket::bind(&path, "s1".into(), s.jobs.clone(), s.launcher.clone()).unwrap();
        for dir in [s.dir.path().join("run"), s.dir.path().join("run").join("aria")] {
            let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o700, "{}", dir.display());
        }
        drop(socket);
    }

    #[test]
    fn a_line_past_the_limit_is_skipped_and_the_next_request_answered() {
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        // 3 MiB without a newline, written from another thread (the dock reads while it is written), then a newline.
        let mut writer = c.writer.try_clone().unwrap();
        let sent = std::thread::spawn(move || {
            let chunk = vec![b'x'; 64 * 1024];
            for _ in 0..48 {
                writer.write_all(&chunk).unwrap();
            }
            writer.write_all(b"\n").unwrap();
        });
        sent.join().unwrap();
        assert_eq!(c.ask(json!({"id": 1, "op": "list"})), json!({"re": 1, "ok": true, "jobs": []}));
    }

    #[test]
    fn a_process_without_its_line_is_refused() {
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let folder = s.dir.path().to_str().unwrap();
        let refused = c.ask(json!({"id": 1, "op": "start", "kind": "process", "command": "true", "title": "t",
            "folder": folder}));
        assert_eq!(refused, json!({"re": 1, "ok": false, "error": "line is required"}));
        assert!(s.jobs.list(None).is_empty(), "nothing ran unsandboxed");
    }

    #[test]
    fn stopping_a_job_that_ended_says_so() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(false);
        let job = start_process(&s.jobs, "s1", s.dir.path(), "true");
        eventually("it ends", || s.jobs.get(&job.id).is_some_and(|j| j.ended_at.is_some()));
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        assert_eq!(c.ask(json!({"id": 1, "op": "stop", "job": job.id})), json!({"re": 1, "ok": true, "already": true}));
    }

    #[test]
    fn an_ack_only_reaches_its_own_session() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(false);
        start_process(&s.jobs, "s2", s.dir.path(), "true");
        eventually("s2 has an end", || !s.jobs.events("s2").is_empty());
        let seq = s.jobs.events("s2")[0].seq();
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        c.send(json!({"op": "ack", "event": seq}));
        // Answered in order: once list is answered, the ack has been handled.
        assert_eq!(c.ask(json!({"id": 1, "op": "list"}))["ok"], true);
        assert_eq!(s.jobs.events("s2").len(), 1, "s1 cannot ack s2's event");
    }

    fn cut(seq: u64) -> JobEvent {
        JobEvent::Cut { seq, jobs: Vec::new() }
    }

    #[test]
    fn a_pushed_event_already_given_goes_out_once() {
        // An event that ends between `subscribe` and `events` is both queued (given by hello) and pushed.
        let (dock, peer) = UnixStream::pair().unwrap();
        let writer = Mutex::new(Writer { stream: dock, written: 5 });
        let (tx, rx) = mpsc::channel();
        for seq in [4, 5, 6] {
            tx.send(cut(seq)).unwrap();
        }
        drop(tx);
        forward(rx, &writer, &AtomicBool::new(false));
        drop(writer);
        let mut got = String::new();
        BufReader::new(peer).read_to_string(&mut got).unwrap();
        let seqs: Vec<u64> =
            got.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()["seq"].as_u64().unwrap()).collect();
        assert_eq!(seqs, vec![6]);
    }

    #[test]
    fn every_end_reaches_a_hello_exactly_once() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(false);
        let socket = bind(&s, "s1");
        // Ends spread around the hello, so some land while it subscribes and reads the queue.
        let ids: Vec<String> =
            (0..8).map(|i| start_process(&s.jobs, "s1", s.dir.path(), &format!("sleep 0.{i:02}")).id).collect();
        let mut c = Client::connect(&socket);
        assert_eq!(c.ask(json!({"id": 1, "op": "hello"}))["ok"], true);
        let mut seen = Vec::new();
        while seen.len() < ids.len() {
            let event = c.read();
            assert_eq!(event["event"], "ended", "{event}");
            seen.push(event["job"]["id"].as_str().unwrap().to_string());
        }
        seen.sort();
        let mut want = ids.clone();
        want.sort();
        assert_eq!(seen, want, "each end once");
        c.reader.get_ref().set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        let mut more = String::new();
        assert!(c.reader.read_line(&mut more).is_err() || more.is_empty(), "no repeat: {more}");
    }

    #[test]
    fn a_second_hello_on_one_connection_keeps_what_it_already_gave() {
        // An acked event still buffered for the old forwarder must not go out again after the new hello.
        let s = setup(false);
        let session = Session { name: "s1".into(), jobs: s.jobs.clone(), launcher: s.launcher.clone() };
        let (dock, _peer) = UnixStream::pair().unwrap();
        let writer = Arc::new(Mutex::new(Writer { stream: dock, written: 9 }));
        let closed = Arc::new(AtomicBool::new(false));
        hello(&session, json!(2), &writer, &closed).unwrap();
        assert_eq!(writer.lock().unwrap().written, 9);
        closed.store(true, Ordering::SeqCst);
    }

    #[test]
    fn a_bad_line_is_ignored() {
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        c.send(json!("not json"));
        writeln!(c.writer, "not json").unwrap();
        assert_eq!(c.ask(json!({"id": 1, "op": "list"})), json!({"re": 1, "ok": true, "jobs": []}));
        let unknown = c.ask(json!({"id": 2, "op": "dance"}));
        assert_eq!(unknown, json!({"re": 2, "ok": false, "error": "unknown op"}));
    }

    #[test]
    fn output_lines_default_to_60_and_stop_at_400() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(false);
        let job = start_process(&s.jobs, "s1", s.dir.path(), "seq 500");
        eventually("it prints all", || s.jobs.output(&job.id, 1000).unwrap_or_default().len() == 500);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let default = c.ask(json!({"id": 1, "op": "output", "job": job.id}));
        assert_eq!(default["lines"].as_array().unwrap().len(), 60);
        assert_eq!(default["lines"][59], "500");
        let many = c.ask(json!({"id": 2, "op": "output", "job": job.id, "lines": 1000}));
        assert_eq!(many["lines"].as_array().unwrap().len(), 400);
    }

    #[test]
    fn a_new_hello_takes_the_events_and_the_old_connection_still_answers() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut old = Client::connect(&socket);
        assert_eq!(old.ask(json!({"id": 1, "op": "hello"}))["ok"], true);
        let mut new = Client::connect(&socket);
        assert_eq!(new.ask(json!({"id": 1, "op": "hello"}))["ok"], true);
        let job = start_process(&s.jobs, "s1", s.dir.path(), "true");
        let event = new.read();
        assert_eq!(event["job"]["id"], job.id.as_str(), "{event}");
        // The old one got no event: its next line is the answer it asks for.
        old.send(json!({"id": 2, "op": "list"}));
        let answer = old.read();
        assert_eq!(answer["re"], 2, "{answer}");
        assert_eq!(answer["jobs"][0]["id"], job.id.as_str());
    }
}
