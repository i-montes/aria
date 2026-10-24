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
use crate::schedule_store::ScheduleStore;
use aria_core::paths::AriaPaths;
use aria_core::secrets;
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
/// to add, the names to remove, and where its worktree goes if it gets one.
pub type AgentLaunch = (PathBuf, Vec<String>, Vec<(String, String)>, Vec<String>, PathBuf);

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
    schedules: ScheduleStore,
    /// Aria's home and keyring, for the person's secrets (`secret_import`, `secrets_list`).
    paths: AriaPaths,
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
    pub fn bind(
        path: &Path,
        session: String,
        jobs: Jobs,
        launcher: Arc<dyn Launcher>,
        schedules: ScheduleStore,
        paths: AriaPaths,
    ) -> std::io::Result<JobsSocket> {
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
        let session = Arc::new(Session { name: session, jobs, launcher, schedules, paths });
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
        // Scheduled tasks are the dock's, not the session's: a tab the person talks in can make or change them (the task
        // id is `task`); a scheduled task's own tab, which runs unattended, never can.
        Some("schedule_create" | "schedule_update" | "schedule_delete")
            if session.schedules.get(&session.name).is_some() =>
        {
            Err("a scheduled task cannot make or change scheduled tasks".into())
        }
        Some("schedule_create") => {
            let task = session.schedules.create(request, now_ms())?;
            Ok(fields(json!({"task": task})))
        }
        Some("schedule_list") => Ok(fields(json!({"tasks": session.schedules.list(now_ms())}))),
        Some("schedule_update") => {
            let mut input = request.clone();
            input["id"] = request["task"].clone();
            let task = session.schedules.update(&input)?;
            Ok(fields(json!({"task": task})))
        }
        Some("schedule_delete") => {
            session.schedules.delete(request["task"].as_str().ok_or("task is missing")?)?;
            Ok(Fields::new())
        }
        // The value never comes back: only the name it was kept under and the variable a command reads it from.
        Some("secret_import") => {
            let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
            let projects = aria_core::agent::workdir(aria_core::settings::load(&session.paths).ok().flatten().as_ref());
            let path = secret_file(
                request["path"].as_str().ok_or("path is missing")?,
                request["cwd"].as_str(),
                &projects,
                &home,
                &session.paths.root,
            )?;
            let text = std::fs::read_to_string(&path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
            let value = secrets::extract(&text, request["key"].as_str())?;
            let purpose = request["purpose"].as_str().unwrap_or_default();
            let entry = secrets::store(
                &session.paths,
                request["name"].as_str().ok_or("name is missing")?,
                &value,
                purpose,
                now_ms(),
            )?;
            Ok(fields(json!({"name": entry.name, "env": secrets::env_name(&entry.name)})))
        }
        Some("secrets_list") => {
            let list: Vec<Value> = secrets::list(&session.paths)
                .into_iter()
                .map(|e| json!({"name": e.name, "purpose": e.purpose, "env": secrets::env_name(&e.name)}))
                .collect();
            Ok(fields(json!({"secrets": list})))
        }
        _ => Err("unknown op".into()),
    }
}

/// The only thing `secret_import` says when a file is out of bounds: it never tells whether the file or the key exists.
const SECRET_FILE_REFUSED: &str = "Aria keeps a secret only from a file inside this session's project or the projects folder, and never from Aria's own folder or a hidden folder of your home folder.";

/// The file `secret_import` may read: `raw` (absolute, `~/…` or relative to `cwd`, the session's project root) with no `..`
/// in it, inside `cwd` or the projects folder, and not inside Aria's own folder or a dot entry of the home folder
/// (`~/.ssh`, `~/.aws`, `~/.config`, `~/.aria`, any `~/.<name>`), judged where it is written and, when it exists, where
/// it really is, which must also be inside the real `cwd` or projects folder (a link cannot lead out). Every refusal is the same sentence, and nothing is read before the check.
fn secret_file(
    raw: &str,
    cwd: Option<&str>,
    projects: &Path,
    home: &Path,
    aria_root: &Path,
) -> Result<PathBuf, String> {
    use std::path::Component;
    let refused = || SECRET_FILE_REFUSED.to_string();
    // Absolute, `.` dropped, `..` refused.
    let clean = |path: &Path| -> Option<PathBuf> {
        let mut out = PathBuf::new();
        for part in path.components() {
            match part {
                Component::RootDir => out.push("/"),
                Component::Normal(name) => out.push(name),
                Component::CurDir => {}
                Component::ParentDir | Component::Prefix(_) => return None,
            }
        }
        out.is_absolute().then_some(out)
    };
    if raw.trim().is_empty() {
        return Err(refused());
    }
    let cwd = cwd.and_then(|cwd| clean(Path::new(cwd)));
    let path = match raw.trim().strip_prefix("~/") {
        Some(rest) => home.join(rest),
        None if Path::new(raw.trim()).is_relative() => cwd.as_ref().ok_or_else(refused)?.join(raw.trim()),
        None => PathBuf::from(raw.trim()),
    };
    let path = clean(&path).ok_or_else(refused)?;
    let roots: Vec<PathBuf> = cwd.into_iter().chain(clean(projects)).filter(|root| root != Path::new("/")).collect();
    if !roots.iter().any(|root| path.starts_with(root)) {
        return Err(refused());
    }
    let homes: Vec<PathBuf> = clean(home).into_iter().chain(std::fs::canonicalize(home).ok()).collect();
    let aria_roots: Vec<PathBuf> = clean(aria_root).into_iter().chain(std::fs::canonicalize(aria_root).ok()).collect();
    let forbidden = |path: &Path| {
        aria_roots.iter().any(|root| path.starts_with(root))
            || homes.iter().any(|home| {
                path.strip_prefix(home)
                    .ok()
                    .and_then(|rest| rest.components().next())
                    .is_some_and(|first| first.as_os_str().to_string_lossy().starts_with('.'))
            })
    };
    if forbidden(&path) {
        return Err(refused());
    }
    // A file that exists is judged where it really is: inside the real `cwd` or projects folder, and not forbidden. A link
    // in the project cannot lead out of them (to `/proc/self/environ`, `~/Documents`, `/etc`) or into a dot entry.
    if let Ok(real) = std::fs::canonicalize(&path) {
        let real_roots: Vec<PathBuf> = roots
            .iter()
            .filter_map(|root| std::fs::canonicalize(root).ok())
            .filter(|root| root != Path::new("/"))
            .collect();
        if forbidden(&real) || !real_roots.iter().any(|root| real.starts_with(root)) {
            return Err(refused());
        }
    }
    Ok(path)
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
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
    let commits = match &request["commits"] {
        Value::Null => false,
        Value::Bool(b) => *b,
        _ => return Err("commits must be true or false".into()),
    };
    let isolation = text(request, "isolation", Some("auto"))?;
    if !matches!(isolation, "auto" | "folder" | "worktree") {
        return Err("isolation must be \"auto\", \"folder\" or \"worktree\"".into());
    }
    if commits && !writes {
        return Err("commits needs writes".into());
    }
    if commits && isolation == "worktree" {
        return Err("a subagent that commits works in its folder: isolation must be \"folder\" or \"auto\"".into());
    }
    // Its commits must land in the person's repository: a worktree's work comes back as one uncommitted patch.
    let isolation = if commits { "folder" } else { isolation };
    let model = text(request, "model", Some("session"))?;
    if !matches!(model, "session" | "heavy") {
        return Err("model must be \"session\" or \"heavy\"".into());
    }
    let Some((program, args, env, unset, worktrees)) = session.launcher.agent(model) else {
        return Err("cannot start a subagent now: the session's Pi is not ready".into());
    };
    let launcher = session.launcher.clone();
    session.jobs.start_agent(AgentStart {
        session: session.name.clone(),
        title: title.to_string(),
        task: task.to_string(),
        folder,
        writes,
        commits,
        isolation: isolation.to_string(),
        worktrees,
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
            self.pi.clone().map(|pi| {
                let worktrees = pi.with_file_name(".aria-worktrees");
                (pi, Vec::new(), Vec::new(), Vec::new(), worktrees)
            })
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
        schedules: ScheduleStore,
        paths: AriaPaths,
        keyring: Arc<aria_core::keyring::MemoryBackend>,
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
        let paths = aria_core::paths::AriaPaths::new(dir.path().to_path_buf());
        std::fs::write(paths.mcp(), r#"{"mcpServers":{"notion":{"url":"https://n"}}}"#).unwrap();
        let schedules = ScheduleStore::open(paths);
        let (secret_paths, keyring) = aria_core::testing::vault_paths(&dir.path().join("home"));
        Setup {
            dir,
            jobs,
            launcher: Arc::new(FakeLauncher { pi, spawned: AtomicUsize::new(0) }),
            schedules,
            paths: secret_paths,
            keyring,
        }
    }

    fn bind(setup: &Setup, session: &str) -> JobsSocket {
        let path = setup.dir.path().join(format!("{session}.sock"));
        JobsSocket::bind(
            &path,
            session.to_string(),
            setup.jobs.clone(),
            setup.launcher.clone(),
            setup.schedules.clone(),
            setup.paths.clone(),
        )
        .unwrap()
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

    #[test]
    fn a_secret_from_a_file_goes_to_the_keyring_and_its_value_never_comes_back() {
        use aria_core::keyring::Backend;
        let s = setup(false);
        let work = s.dir.path().join("work");
        std::fs::create_dir_all(&work).unwrap();
        std::fs::write(work.join(".env"), "PORT=3000\nOPENAI_API_KEY=sk-proj-0123456789abcdef\n").unwrap();
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let saved = c.ask(json!({"id": 1, "op": "secret_import", "path": ".env", "cwd": work.display().to_string(),
            "key": "OPENAI_API_KEY", "name": "OpenAI key", "purpose": "the widget's chat"}));
        assert_eq!(saved["ok"], true, "{saved}");
        assert_eq!(saved["name"], "openai-key");
        assert_eq!(saved["env"], "ARIA_SECRET_OPENAI_KEY");
        assert!(!saved.to_string().contains("sk-proj"), "the value never comes back: {saved}");
        assert_eq!(s.keyring.get("user:openai-key").unwrap().as_deref(), Some("sk-proj-0123456789abcdef"));
        let list = c.ask(json!({"id": 2, "op": "secrets_list"}));
        assert_eq!(
            list["secrets"],
            json!([{"name": "openai-key", "purpose": "the widget's chat", "env": "ARIA_SECRET_OPENAI_KEY"}])
        );
        let missing = c.ask(json!({"id": 3, "op": "secret_import", "path": ".env", "cwd": work.display().to_string(), "key": "NOPE", "name": "x"}));
        assert_eq!(missing["ok"], false);
        assert!(!missing.to_string().contains("sk-proj"), "nor in an error: {missing}");
    }

    fn secret_file(raw: &str, cwd: Option<&str>) -> Result<PathBuf, String> {
        super::secret_file(
            raw,
            cwd,
            Path::new("/home/ana/Projects"),
            Path::new("/home/ana"),
            Path::new("/home/ana/.aria"),
        )
    }

    #[test]
    fn a_secret_is_read_only_from_the_project_or_the_projects_folder() {
        let cwd = Some("/home/ana/Projects/acme/checkout-widget");
        assert_eq!(secret_file(".env", cwd).unwrap(), PathBuf::from("/home/ana/Projects/acme/checkout-widget/.env"));
        assert!(secret_file("/home/ana/Projects/acme/checkout-widget/config/.env.local", cwd).is_ok());
        assert!(secret_file("/home/ana/Projects/other/.env", cwd).is_ok(), "the projects folder");
        assert!(secret_file("~/Projects/other/.env", None).is_ok(), "`~` is expanded before the check");
        assert!(secret_file("/home/ana/Projects/./other/.env", None).is_ok());
        // A project outside the projects folder: its own root counts.
        assert!(secret_file(".env", Some("/srv/code/app")).is_ok());
        assert!(secret_file("/srv/code/app/.env", Some("/srv/code/app")).is_ok());
    }

    #[test]
    fn a_link_in_the_project_cannot_lead_out_of_the_allowed_area() {
        use std::os::unix::fs::symlink;
        let base = tempfile::tempdir().unwrap();
        let (project, outside, aria) = (base.path().join("proj"), base.path().join("docs"), base.path().join("aria"));
        for dir in [&project, &outside, &aria] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(project.join("real.env"), "TOKEN=1\n").unwrap();
        std::fs::write(outside.join("notes.txt"), "TOKEN=2\n").unwrap();
        std::fs::write(aria.join("auth.json"), "TOKEN=3\n").unwrap();
        symlink(outside.join("notes.txt"), project.join("to-notes")).unwrap();
        symlink(&outside, project.join("to-docs")).unwrap();
        symlink("/etc/hostname", project.join("to-etc")).unwrap();
        symlink(aria.join("auth.json"), project.join("to-aria")).unwrap();
        symlink(project.join("real.env"), project.join("inside")).unwrap();
        #[cfg(target_os = "linux")]
        symlink("/proc/self/environ", project.join("to-environ")).unwrap();
        let cwd = project.display().to_string();
        let check = |raw: &str| {
            super::secret_file(raw, Some(&cwd), &base.path().join("projects"), Path::new("/home/ana"), &aria)
        };
        assert!(check("real.env").is_ok());
        assert!(check("inside").is_ok(), "a link to a file in the project is fine");
        let mut said = std::collections::HashSet::new();
        for raw in ["to-notes", "to-docs/notes.txt", "to-etc", "to-aria", "to-environ"] {
            if raw == "to-environ" && !cfg!(target_os = "linux") {
                continue;
            }
            said.insert(check(raw).unwrap_err());
        }
        assert_eq!(said.len(), 1, "one sentence for all: {said:?}");
    }

    #[test]
    fn any_other_file_is_refused_with_one_sentence_whatever_it_is() {
        let cwd = Some("/home/ana/Projects/acme/checkout-widget");
        let mut said = std::collections::HashSet::new();
        for raw in [
            "/etc/passwd",
            "/home/ana/.aws/credentials",
            "~/.aws/credentials",
            "/home/ana/.ssh/id_ed25519",
            "/home/ana/.gnupg/pubring.kbx",
            "/home/ana/.config/gh/hosts.yml",
            "/home/ana/.local/share/keyrings/login.keyring",
            "/home/ana/.kube/config",
            "/home/ana/.aria/auth.json",
            "/home/ana/.anything/x",
            "/home/ana/.env",
            "/home/ana/Documents/notes.txt",
            "../../../.ssh/id_ed25519",
            "/home/ana/Projects/acme/../../.aws/credentials",
            "/home/ana/Projects/../.ssh/id_ed25519",
            "/home/ana/Projectsx/.env",
            "/tmp/.env",
            "",
        ] {
            let why = secret_file(raw, cwd).unwrap_err();
            said.insert(why);
        }
        assert_eq!(said.len(), 1, "the same sentence for every refusal: {said:?}");
        let sentence = said.into_iter().next().unwrap();
        for word in ["exist", "found", "could not read", "no such", "key"] {
            assert!(!sentence.to_lowercase().contains(word), "{sentence}");
        }
        assert!(sentence.contains("project"), "{sentence}");
        assert!(secret_file("relative/.env", None).is_err(), "a relative path with no working folder");
    }

    #[test]
    fn a_dot_folder_of_the_home_is_refused_even_inside_a_project_that_is_the_home_folder() {
        let at_home = |raw: &str| {
            super::secret_file(
                raw,
                Some("/home/ana"),
                Path::new("/home/ana/Projects"),
                Path::new("/home/ana"),
                Path::new("/home/ana/.aria"),
            )
        };
        assert!(at_home("notes/.env").is_ok());
        for raw in [".ssh/id_ed25519", ".aws/credentials", ".config/gh/hosts.yml", ".aria/auth.json", ".env"] {
            assert!(at_home(raw).is_err(), "{raw}");
        }
        // Aria's own folder when it is not under a dot folder.
        let moved = super::secret_file(
            "/srv/aria-home/auth.json",
            Some("/srv"),
            Path::new("/home/ana/Projects"),
            Path::new("/home/ana"),
            Path::new("/srv/aria-home"),
        );
        assert!(moved.is_err());
    }

    #[test]
    fn the_service_refuses_a_secret_outside_the_project_and_says_nothing_about_it() {
        use aria_core::keyring::Backend;
        let s = setup(false);
        let work = s.dir.path().join("work");
        std::fs::create_dir_all(&work).unwrap();
        std::fs::write(s.dir.path().join("home/auth.json"), "SECRET_TOKEN=abc-not-for-the-model\n").unwrap();
        std::os::unix::fs::symlink(s.dir.path().join("home/auth.json"), work.join("link")).unwrap();
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let cwd = work.display().to_string();
        let home = std::env::var("HOME").unwrap();
        let mut errors = std::collections::HashSet::new();
        for (n, path) in [
            format!("{home}/.aws/credentials"),
            "/etc/passwd".to_string(),
            "/etc/definitely-not-there".to_string(),
            "../home/auth.json".to_string(),
            "link".to_string(), // a link in the project to Aria's own folder
        ]
        .into_iter()
        .enumerate()
        {
            let answer = c.ask(json!({"id": n + 1, "op": "secret_import", "path": path, "cwd": cwd, "key": "SECRET_TOKEN", "name": "x"}));
            assert_eq!(answer["ok"], false, "{path}: {answer}");
            assert!(!answer.to_string().contains("abc-not-for-the-model"), "{answer}");
            errors.insert(answer["error"].as_str().unwrap().to_string());
        }
        assert_eq!(errors.len(), 1, "one sentence for all: {errors:?}");
        assert!(s.keyring.get("user:x").unwrap().is_none(), "nothing was kept");
    }

    #[test]
    fn a_secret_in_the_projects_folder_is_kept_when_the_session_works_elsewhere() {
        let s = setup(false);
        let projects = s.dir.path().join("projects");
        std::fs::create_dir_all(projects.join("app")).unwrap();
        std::fs::write(projects.join("app/.env"), "TOKEN=tok-1234567890abcdef\n").unwrap();
        let mut settings = aria_core::settings::Settings::defaults("en");
        settings.sandbox.projects_dir = projects.display().to_string();
        aria_core::settings::save(&s.paths, &settings).unwrap();
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let elsewhere = s.dir.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        let saved = c
            .ask(json!({"id": 1, "op": "secret_import", "path": projects.join("app/.env").display().to_string(),
            "cwd": elsewhere.display().to_string(), "key": "TOKEN", "name": "app-token"}));
        assert_eq!(saved["ok"], true, "{saved}");
    }

    #[test]
    fn create_lists_and_deletes_a_task() {
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let created = c.ask(json!({"id": 1, "op": "schedule_create", "title": "Notion tasks", "prompt": "My tasks", "at": "07:30", "days": ["mon", "fri"]}));
        assert_eq!(created["ok"], true, "{created}");
        let id = created["task"]["id"].as_str().unwrap().to_string();
        let list = c.ask(json!({"id": 2, "op": "schedule_list"}));
        assert_eq!(list["tasks"][0]["id"], id.as_str());
        assert!(list["tasks"][0]["next"].is_i64());
        assert_eq!(c.ask(json!({"id": 3, "op": "schedule_delete", "id_": 0, "task": id}))["ok"], true);
        assert_eq!(c.ask(json!({"id": 4, "op": "schedule_list"}))["tasks"], json!([]));
    }

    #[test]
    fn create_refuses_an_unknown_connector() {
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let refused = c.ask(json!({"id": 1, "op": "schedule_create", "title": "t", "prompt": "p", "at": "07:30", "connectors": {"gmail": "read"}}));
        assert_eq!(refused["ok"], false);
        assert!(refused["error"].as_str().unwrap().contains("gmail"), "{refused}");
    }

    #[test]
    fn create_and_update_refuse_a_writable_folder_outside_the_projects_folder_and_tmp() {
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let refused = c.ask(json!({"id": 1, "op": "schedule_create", "title": "t", "prompt": "p", "at": "07:30", "writable": ["/home/ana/.config"]}));
        assert_eq!(refused["ok"], false, "{refused}");
        let error = refused["error"].as_str().unwrap();
        assert!(error.contains("projects folder") && error.contains("/tmp"), "{error}");
        let id = c.ask(json!({"id": 2, "op": "schedule_create", "title": "t", "prompt": "p", "at": "07:30"}))["task"]
            ["id"]
            .clone();
        let refused = c.ask(json!({"id": 3, "op": "schedule_update", "task": id, "writable": ["/etc"]}));
        assert_eq!(refused["ok"], false, "{refused}");
        assert!(refused["error"].as_str().unwrap().contains("projects folder"), "{refused}");
    }

    #[test]
    fn update_pauses_a_task() {
        let s = setup(false);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let id = c.ask(json!({"id": 1, "op": "schedule_create", "title": "t", "prompt": "p", "at": "07:30"}))["task"]
            ["id"]
            .clone();
        let paused = c.ask(json!({"id": 2, "op": "schedule_update", "task": id, "enabled": false}));
        assert_eq!(paused["task"]["enabled"], false, "{paused}");
    }

    #[test]
    fn a_scheduled_tab_cannot_change_tasks() {
        let s = setup(false);
        let task = s.schedules.create(&json!({"title": "t", "prompt": "p", "at": "07:30"}), 0).unwrap();
        let socket = bind(&s, &task.id);
        let mut c = Client::connect(&socket);
        for op in ["schedule_create", "schedule_update", "schedule_delete"] {
            let answer = c.ask(json!({"id": 1, "op": op, "task": task.id, "title": "t", "prompt": "p", "at": "07:30"}));
            assert_eq!(answer["ok"], false, "{op}: {answer}");
        }
        assert_eq!(c.ask(json!({"id": 2, "op": "schedule_list"}))["ok"], true, "it may still list");
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
    fn a_subagent_that_commits_works_in_its_folder() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(true);
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let folder = s.dir.path().to_str().unwrap();
        let ask = |c: &mut Client, id: u64, extra: Value| {
            let mut request = json!({"id": id, "op": "start", "kind": "agent", "task": "slice it",
                "title": "slice", "folder": folder, "commits": true});
            for (key, value) in extra.as_object().unwrap() {
                request[key] = value.clone();
            }
            c.ask(request)
        };
        let worktree = ask(&mut c, 1, json!({"isolation": "worktree"}));
        let message = "a subagent that commits works in its folder: isolation must be \"folder\" or \"auto\"";
        assert_eq!(worktree, json!({"re": 1, "ok": false, "error": message}));
        let read_only = ask(&mut c, 2, json!({"writes": false}));
        assert_eq!(read_only, json!({"re": 2, "ok": false, "error": "commits needs writes"}));
        let text = ask(&mut c, 3, json!({"commits": "yes"}));
        assert_eq!(text, json!({"re": 3, "ok": false, "error": "commits must be true or false"}));
        assert!(s.jobs.list(None).is_empty(), "nothing started");
        assert_eq!(s.launcher.spawned.load(Ordering::SeqCst), 0);

        for (id, extra) in [(4, json!({})), (5, json!({"isolation": "auto"})), (6, json!({"isolation": "folder"}))] {
            let started = ask(&mut c, id, extra);
            assert_eq!(started["ok"], true, "{started}");
            let job = started["job"]["id"].as_str().unwrap().to_string();
            eventually("the agent ends", || s.jobs.get(&job).is_some_and(|j| j.result.is_some()));
            assert_eq!(s.jobs.get(&job).unwrap().isolation.as_deref(), Some("folder"));
        }
    }

    #[test]
    fn a_subagent_that_commits_waits_for_its_folder_instead_of_taking_a_worktree() {
        use crate::jobs::JobState;
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = setup(true);
        // A fake `pi` that stays running, so the first writer holds the repository.
        let pi = s.dir.path().join("pi");
        std::fs::write(&pi, "#!/bin/sh\nsleep 30\n").unwrap();
        let (_repo_dir, root) = crate::worktree::tests::repo();
        let socket = bind(&s, "s1");
        let mut c = Client::connect(&socket);
        let folder = root.to_str().unwrap();
        let mut start = |id: u64, extra: Value| {
            let mut request = json!({"id": id, "op": "start", "kind": "agent", "task": "slice it",
                "title": "slice", "folder": folder, "writes": true});
            for (key, value) in extra.as_object().unwrap() {
                request[key] = value.clone();
            }
            let started = c.ask(request);
            assert_eq!(started["ok"], true, "{started}");
            started["job"]["id"].as_str().unwrap().to_string()
        };
        let holder = start(1, json!({"isolation": "folder"}));
        assert_eq!(s.jobs.get(&holder).unwrap().isolation.as_deref(), Some("folder"));
        // The contrast: without `commits`, "auto" gives way to a worktree when the repository is taken.
        let plain = start(2, json!({"isolation": "auto"}));
        let plain = s.jobs.get(&plain).unwrap();
        assert_eq!(plain.isolation.as_deref(), Some("worktree"));
        assert!(plain.worktree.is_some());
        // With `commits`, the same request waits for the folder, whether `isolation` is "auto" or missing.
        let auto = start(3, json!({"isolation": "auto", "commits": true}));
        let missing = start(4, json!({"commits": true}));
        for id in [&auto, &missing] {
            let job = s.jobs.get(id).unwrap();
            assert_eq!(job.state, JobState::Queued, "{job:?}");
            assert_eq!(job.worktree, None);
        }
        s.jobs.stop(&holder);
        eventually("the first to commit gets the folder", || {
            s.jobs.get(&auto).is_some_and(|j| j.state == JobState::Running)
        });
        let job = s.jobs.get(&auto).unwrap();
        assert_eq!((job.isolation.as_deref(), job.worktree), (Some("folder"), None));
        assert_eq!(s.jobs.get(&missing).unwrap().state, JobState::Queued, "the folder has one writer at a time");
        s.jobs.stop_all();
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
        let socket = JobsSocket::bind(
            &path,
            "s1".into(),
            s.jobs.clone(),
            s.launcher.clone(),
            s.schedules.clone(),
            s.paths.clone(),
        )
        .unwrap();
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
        let session = Session {
            name: "s1".into(),
            jobs: s.jobs.clone(),
            launcher: s.launcher.clone(),
            schedules: s.schedules.clone(),
            paths: s.paths.clone(),
        };
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
