//! Background work the dock owns: processes a session's Pi starts (dev servers, watchers, long builds) that keep
//! running while the conversation goes on. The dock, not Pi, is their parent, so a Pi that restarts never takes them
//! down. Every child is spawned from one long-lived thread and leads its own process group, so Stop reaches whatever it
//! forked, and the page is told about changes at most four times a second.
//!
//! How a job ended becomes an event for its session, kept until Pi acknowledges it, so a Pi that restarts in between
//! still hears about it exactly once. The jobs (never their output) are kept in a store; a dock that went away while
//! jobs ran finds them there and reports them as cut.
//!
//! Subagents (`agent`) are jobs too: a `pi` run in JSON mode whose last answer is the job's result.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::os::fd::{FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

mod agent;

// The dock starts subagents from Task 6's wiring on; until then only the tests do.
#[allow(unused_imports)]
pub use agent::AgentStart;

/// Processes running at once, across every session.
const MAX_PROCESSES: usize = 10;
/// Subagents running at once, across every session; more wait, queued, and start in order.
const MAX_AGENTS_RUNNING: usize = 3;
/// Characters kept of a subagent's last answer, its result.
const RESULT_CHARS: usize = 8000;
/// Lines of output kept per job (the latest ones).
const OUTPUT_LINES: usize = 1000;
/// Characters kept of one line of output.
const LINE_CHARS: usize = 2000;
/// Finished jobs kept with their output. An older one loses its output, and is forgotten once its end has been
/// acknowledged.
const FINISHED_KEPT: usize = 20;
/// Lines of output an end event carries.
const LAST_LINES: usize = 20;
/// How long the end of a job waits, after its exit, for the rest of its output (a fork may keep the pipe open).
const OUTPUT_DRAIN: Duration = Duration::from_millis(500);
/// How long a stopped group has after SIGTERM before SIGKILL.
const STOP_GRACE: Duration = Duration::from_secs(3);
const THROTTLE: Duration = Duration::from_millis(250);
/// Bytes read of one line before the rest is skipped: a child that never prints a newline cannot grow the dock's memory.
const LINE_BYTES: u64 = (LINE_CHARS * 4) as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Process,
    Agent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Queued,
    Running,
    /// The process ended by itself; `exit` holds its code.
    Exited,
    Failed,
    /// Ended by Stop.
    Stopped,
    Done,
    Cut,
}

impl JobState {
    fn active(self) -> bool {
        matches!(self, JobState::Queued | JobState::Running)
    }
}

/// One job as Pi and the page see it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Job {
    pub id: String,
    pub number: u64,
    pub session: String,
    pub kind: Kind,
    pub title: String,
    /// What the model asked to run, never the line actually run (which may be wrapped in a sandbox).
    pub command: Option<String>,
    pub task: Option<String>,
    /// Shown with `~` for the home folder.
    pub folder: String,
    pub state: JobState,
    pub started_at: i64,
    pub ended_at: Option<i64>,
    pub url: Option<String>,
    pub exit: Option<i32>,
    pub isolation: Option<String>,
    pub worktree: Option<String>,
    pub writes: bool,
    pub result: Option<String>,
    pub conflict: bool,
    pub dismissed: bool,
}

/// Who ended a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndedBy {
    #[serde(rename = "self")]
    Itself,
    #[serde(rename = "stop")]
    Stopped,
}

/// Something a session's Pi is told about its jobs, kept until it acknowledges `seq`.
// Events are few and short-lived; an unboxed `Job` keeps them simple to build and match.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum JobEvent {
    /// A job ended; `last_lines` is the end of its output (processes only).
    Ended { seq: u64, job: Job, by: EndedBy, last_lines: Vec<String> },
    /// The dock went away while these jobs ran.
    Cut { seq: u64, jobs: Vec<Job> },
}

impl JobEvent {
    pub fn seq(&self) -> u64 {
        match self {
            JobEvent::Ended { seq, .. } | JobEvent::Cut { seq, .. } => *seq,
        }
    }

    /// Whether this event is about the job `id`.
    fn about(&self, id: &str) -> bool {
        match self {
            JobEvent::Ended { job, .. } => job.id == id,
            JobEvent::Cut { jobs, .. } => jobs.iter().any(|j| j.id == id),
        }
    }
}

/// What `jobs.json` holds: the jobs as they were and each session's unacknowledged events. No output beyond an end
/// event's own `last_lines`.
#[derive(Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct Store {
    next_seq: u64,
    next_number: u64,
    jobs: Vec<Job>,
    events: BTreeMap<String, Vec<JobEvent>>,
}

/// A process to start: `line` is what `bash -c` runs, `command` what the model asked for.
pub struct ProcessStart {
    pub session: String,
    pub title: String,
    pub command: String,
    pub line: String,
    pub folder: PathBuf,
    pub env: Vec<(String, String)>,
}

struct Entry {
    job: Job,
    output: VecDeque<String>,
    /// The process group the child leads (its pid), 0 before it is spawned.
    group: i32,
    /// Stop was asked: the exit that follows is `Stopped`, not `Exited`.
    stopping: bool,
    /// A subagent's own part while it is queued or running.
    agent: Option<agent::AgentRun>,
}

#[derive(Default)]
struct State {
    entries: Vec<Entry>,
    next_number: u64,
    next_seq: u64,
    /// Per session, the events not yet acknowledged, in `seq` order.
    events: HashMap<String, Vec<JobEvent>>,
    /// Per session, where new events are pushed while its Pi listens.
    subscribers: HashMap<String, mpsc::Sender<JobEvent>>,
    dirty: bool,
    store: PathBuf,
}

impl State {
    fn entry(&mut self, id: &str) -> Option<&mut Entry> {
        self.entries.iter_mut().find(|e| e.job.id == id)
    }

    /// Lets go of the oldest finished jobs beyond `FINISHED_KEPT`, so a long day of jobs keeps the dock's memory
    /// bounded: each is forgotten, or, while its end still waits for its Pi, keeps only its `Job` (the event has its last
    /// lines). True when a job was forgotten.
    fn prune(&mut self) -> bool {
        let mut finished: Vec<(i64, u64)> = self
            .entries
            .iter()
            .filter(|e| !e.job.state.active())
            .map(|e| (e.job.ended_at.unwrap_or(0), e.job.number))
            .collect();
        if finished.len() <= FINISHED_KEPT {
            return false;
        }
        finished.sort();
        let old: Vec<u64> = finished[..finished.len() - FINISHED_KEPT].iter().map(|(_, n)| *n).collect();
        let events = &self.events;
        let held = |job: &Job| events.get(&job.session).is_some_and(|q| q.iter().any(|ev| ev.about(&job.id)));
        let before = self.entries.len();
        self.entries.retain_mut(|e| {
            if e.job.state.active() || !old.contains(&e.job.number) {
                return true;
            }
            if held(&e.job) {
                e.output = VecDeque::new();
                return true;
            }
            false
        });
        self.entries.len() < before
    }

    /// Queues an event for its session and pushes it to the session's subscriber, if one listens.
    fn emit(&mut self, session: &str, event: impl FnOnce(u64) -> JobEvent) {
        let event = event(self.next_seq);
        self.next_seq += 1;
        if let Some(tx) = self.subscribers.get(session) {
            if tx.send(event.clone()).is_err() {
                self.subscribers.remove(session);
            }
        }
        self.events.entry(session.to_string()).or_default().push(event);
        self.dirty = true;
    }

    /// Rewrites the store. Called on every change of a job's state or of the events, never for a line of output.
    fn save(&self) {
        let store = Store {
            next_seq: self.next_seq,
            next_number: self.next_number,
            jobs: self.entries.iter().map(|e| e.job.clone()).collect(),
            events: self.events.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        };
        let written = serde_json::to_value(&store)
            .map_err(std::io::Error::other)
            .and_then(|value| aria_core::json_file::write_json_atomic(&self.store, &value, 0o600));
        if let Err(e) = written {
            eprintln!("aria: could not save the jobs to {}: {e}", self.store.display());
        }
    }

    /// The jobs a previous dock left in the store. Those still running or queued are cut: their session hears it once.
    fn load(&mut self) {
        let stored: Store = match aria_core::json_file::read_json(&self.store) {
            Ok(Some(value)) => serde_json::from_value(value).unwrap_or_else(|e| {
                eprintln!("aria: ignoring the jobs in {}: {e}", self.store.display());
                Store::default()
            }),
            Ok(None) => return,
            Err(e) => {
                eprintln!("aria: ignoring the jobs: {e}");
                Store::default()
            }
        };
        let highest = stored.jobs.iter().map(|j| j.number).max().unwrap_or(0);
        self.next_number = stored.next_number.max(highest + 1).max(self.next_number);
        let last_seq = stored.events.values().flatten().map(JobEvent::seq).max().unwrap_or(0);
        self.next_seq = stored.next_seq.max(last_seq + 1).max(self.next_seq);
        // Events the previous dock's Pis had not acknowledged come first; any cut follows them.
        for (session, mut queue) in stored.events {
            queue.sort_by_key(JobEvent::seq);
            if !queue.is_empty() {
                self.events.insert(session, queue);
            }
        }
        let ended_at = now_ms();
        let mut cut: BTreeMap<String, Vec<Job>> = BTreeMap::new();
        for mut job in stored.jobs {
            if job.state.active() {
                job.state = JobState::Cut;
                job.ended_at = Some(ended_at);
                cut.entry(job.session.clone()).or_default().push(job.clone());
            }
            self.entries.push(Entry { job, output: VecDeque::new(), group: 0, stopping: false, agent: None });
        }
        let changed = !cut.is_empty();
        for (session, jobs) in cut {
            self.emit(&session, |seq| JobEvent::Cut { seq, jobs });
        }
        if self.prune() || changed {
            self.save();
        }
    }
}

/// Work for the spawner thread: a closure that spawns and sends its own reply.
type Spawn = Box<dyn FnOnce() + Send>;

/// The dock's background jobs. Cheap to clone; every clone is the same jobs.
#[derive(Clone)]
pub struct Jobs {
    state: Arc<Mutex<State>>,
    /// Job threads hold it weakly, so the spawner thread (and with it every child) ends with the last clone of `Jobs`.
    spawner: Arc<mpsc::Sender<Spawn>>,
}

/// Runs `spawn` on the spawner thread and waits for what it returns.
fn on_spawner<T: Send + 'static>(
    spawner: &mpsc::Sender<Spawn>,
    spawn: impl FnOnce() -> std::io::Result<T> + Send + 'static,
) -> std::io::Result<T> {
    let gone = || std::io::Error::other("the spawner is gone");
    let (reply, spawned) = mpsc::channel();
    match spawner.send(Box::new(move || {
        let _ = reply.send(spawn());
    })) {
        Ok(()) => spawned.recv().unwrap_or_else(|_| Err(gone())),
        Err(_) => Err(gone()),
    }
}

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// The folder as the page shows it: `~` for the home folder.
fn shown(folder: &Path) -> String {
    if let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        if let Ok(rest) = folder.strip_prefix(&home) {
            return if rest.as_os_str().is_empty() { "~".into() } else { format!("~/{}", rest.display()) };
        }
    }
    folder.display().to_string()
}

/// Removes terminal escapes (colours, cursor moves, OSC links) so the page and Pi see plain text.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        match chars.next() {
            // CSI: parameters, then one final byte in @..~
            Some('[') => {
                for c in chars.by_ref() {
                    if ('@'..='~').contains(&c) {
                        break;
                    }
                }
            }
            // OSC: up to BEL or ESC \
            Some(']') => {
                while let Some(c) = chars.next() {
                    if c == '\u{7}' {
                        break;
                    }
                    if c == '\u{1b}' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// A URL on this machine, as the person opens it: `0.0.0.0` and `[::]` (every address) become `localhost`.
fn localize(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, path) = rest.split_at(authority_end);
    let (host, port) = if authority.starts_with('[') {
        let close = authority.find(']')?;
        authority.split_at(close + 1)
    } else {
        authority.split_at(authority.find(':').unwrap_or(authority.len()))
    };
    let host = host.to_ascii_lowercase();
    let host = match host.as_str() {
        "0.0.0.0" | "[::]" => "localhost",
        "localhost" | "[::1]" => host.as_str(),
        h if h.starts_with("127.") && h.split('.').count() == 4 => h,
        _ => return None,
    };
    Some(format!("{scheme}://{host}{port}{path}"))
}

/// The first local `http(s)` URL in a line of output: a dev server's address, the job's link.
fn detect_url(line: &str) -> Option<String> {
    let mut from = 0;
    while let Some(at) = line[from..].find("http").map(|i| from + i) {
        let candidate = &line[at..];
        from = at + 4;
        if !(candidate.starts_with("http://") || candidate.starts_with("https://")) {
            continue;
        }
        let end = candidate.find(|c: char| c.is_whitespace() || "\"'<>`".contains(c)).unwrap_or(candidate.len());
        // A sentence's punctuation after the address is not part of it.
        let url = candidate[..end].trim_end_matches(['.', ',', ';', ':', ')', '!', '?']);
        if let Some(local) = localize(url) {
            return Some(local);
        }
    }
    None
}

/// A pipe both ends close-on-exec: the child only gets the write end through its stdout and stderr. On Linux both ends
/// are close-on-exec from the start, so a fork on another thread never inherits them.
fn pipe() -> std::io::Result<(File, OwnedFd)> {
    let mut fds = [0; 2];
    #[cfg(target_os = "linux")]
    let made = unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) };
    #[cfg(not(target_os = "linux"))]
    let made = unsafe { libc::pipe(fds.as_mut_ptr()) };
    if made != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let (read, write) = unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) };
    #[cfg(not(target_os = "linux"))]
    for fd in fds {
        unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
    }
    Ok((File::from(read), write))
}

/// Makes the child the leader of its own group, so Stop reaches everything it forks, and, on Linux, tells it to end if
/// the spawner thread dies (the dock exits or crashes) instead of living on unseen.
fn own_group(command: &mut Command) {
    std::os::unix::process::CommandExt::process_group(command, 0);
    #[cfg(target_os = "linux")]
    unsafe {
        std::os::unix::process::CommandExt::pre_exec(command, || {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }
}

/// Runs `line` with `bash -c` in `folder`, stdout and stderr into one pipe so their lines interleave as printed.
/// Called on the spawner thread only.
fn spawn_process(line: &str, folder: &Path, env: &[(String, String)]) -> std::io::Result<(Child, File)> {
    let (read, write) = pipe()?;
    let mut command = Command::new("bash");
    command
        .arg("-c")
        .arg(line)
        .current_dir(folder)
        .envs(env.iter().map(|(k, v)| (k, v)))
        .stdin(Stdio::null())
        .stdout(Stdio::from(write.try_clone()?))
        .stderr(Stdio::from(write));
    own_group(&mut command);
    let child = command.spawn()?;
    // `command` holds the parent's copies of the write end; it drops here, so the reader sees EOF when the group ends.
    Ok((child, read))
}

/// Ends process groups: SIGTERM, then SIGKILL for those still there after `STOP_GRACE`. Returns once all are gone, or a
/// second after SIGKILL.
fn end_groups(groups: &[i32]) {
    let alive = |g: i32| unsafe { libc::killpg(g, 0) == 0 };
    let wait = |until: Instant| {
        while Instant::now() < until && groups.iter().any(|g| alive(*g)) {
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    for g in groups {
        unsafe { libc::killpg(*g, libc::SIGTERM) };
    }
    wait(Instant::now() + STOP_GRACE);
    for g in groups.iter().filter(|g| alive(**g)) {
        unsafe { libc::killpg(*g, libc::SIGKILL) };
    }
    wait(Instant::now() + Duration::from_secs(1));
}

/// One line of output as kept: the text after the last carriage return (a progress bar's latest state), no escapes,
/// at most `LINE_CHARS` characters.
fn clean(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let text = text.trim_end_matches(['\n', '\r']);
    let text = text.rsplit('\r').next().unwrap_or_default();
    strip_ansi(text).chars().take(LINE_CHARS).collect()
}

/// Reads one line, at most `limit` bytes of it, into `bytes`; the rest of a longer line is skipped, a bounded chunk at a
/// time. False at the end of the input.
fn read_line(reader: &mut impl BufRead, limit: u64, bytes: &mut Vec<u8>) -> bool {
    match reader.by_ref().take(limit).read_until(b'\n', bytes) {
        Ok(0) | Err(_) => return false,
        Ok(_) => {}
    }
    if bytes.last() != Some(&b'\n') && bytes.len() as u64 >= limit {
        loop {
            let mut rest = Vec::new();
            match reader.by_ref().take(limit).read_until(b'\n', &mut rest) {
                Ok(0) | Err(_) => break,
                Ok(_) if rest.last() == Some(&b'\n') => break,
                Ok(_) => {}
            }
        }
    }
    true
}

/// Reads a job's output until every process holding the pipe has closed it. `_drained` drops at the end, which tells the
/// waiter the output is complete.
fn read_output(id: String, pipe: File, state: Arc<Mutex<State>>, _drained: mpsc::Sender<()>) {
    let mut reader = BufReader::new(pipe);
    loop {
        let mut bytes = Vec::new();
        if !read_line(&mut reader, LINE_BYTES, &mut bytes) {
            break;
        }
        let line = clean(&bytes);
        let mut s = state.lock().unwrap();
        let Some(entry) = s.entry(&id) else { break };
        if entry.job.url.is_none() {
            entry.job.url = detect_url(&line);
        }
        entry.output.push_back(line);
        while entry.output.len() > OUTPUT_LINES {
            entry.output.pop_front();
        }
        s.dirty = true;
    }
}

/// Waits for the child, then for the rest of its output (at most `OUTPUT_DRAIN`), and records how it ended: the job's
/// new state in the store and an end event for its session.
fn wait_exit(id: String, mut child: Child, state: Arc<Mutex<State>>, drained: mpsc::Receiver<()>) {
    let status = child.wait();
    let _ = drained.recv_timeout(OUTPUT_DRAIN);
    let mut s = state.lock().unwrap();
    let Some(entry) = s.entry(&id) else { return };
    let code = status.ok().and_then(|st| st.code());
    entry.job.exit = code;
    entry.job.ended_at = Some(now_ms());
    entry.job.state = match (entry.stopping, code) {
        (true, _) => JobState::Stopped,
        (false, Some(_)) => JobState::Exited,
        // killed by a signal nobody here sent
        (false, None) => JobState::Failed,
    };
    let by = if entry.stopping { EndedBy::Stopped } else { EndedBy::Itself };
    let skip = entry.output.len().saturating_sub(LAST_LINES);
    let last_lines: Vec<String> = entry.output.iter().skip(skip).cloned().collect();
    let job = entry.job.clone();
    let session = job.session.clone();
    s.emit(&session, |seq| JobEvent::Ended { seq, job, by, last_lines });
    s.prune();
    s.save();
}

impl Jobs {
    /// The jobs, kept in `store`. `on_change` runs at most every 250 ms while something changed.
    ///
    /// Jobs a previous dock left running or queued in `store` come back cut, with one `Cut` event per session.
    ///
    /// The spawner thread lives as long as a clone of these jobs does: children are told to end (`PR_SET_PDEATHSIG`)
    /// when the thread that spawned them goes, so it must be one thread that outlives them all.
    pub fn new(store: PathBuf, on_change: Arc<dyn Fn() + Send + Sync>) -> Jobs {
        let mut loaded = State { next_number: 1, next_seq: 1, store, ..State::default() };
        loaded.load();
        let state = Arc::new(Mutex::new(loaded));
        let (spawner, requests) = mpsc::channel::<Spawn>();
        std::thread::Builder::new()
            .name("aria-jobs-spawner".into())
            .spawn(move || {
                for work in requests {
                    work();
                }
            })
            .expect("the jobs spawner thread");
        {
            // Weak: the throttle ends with the last job thread or clone of `Jobs`.
            let state = Arc::downgrade(&state);
            std::thread::spawn(move || loop {
                std::thread::sleep(THROTTLE);
                let Some(state) = state.upgrade() else { break };
                let changed = std::mem::take(&mut state.lock().unwrap().dirty);
                drop(state);
                if changed {
                    on_change();
                }
            });
        }
        Jobs { state, spawner: Arc::new(spawner) }
    }

    /// Starts a process in the background and returns at once: its output and exit are recorded as they come.
    pub fn start_process(&self, start: ProcessStart) -> Result<Job, String> {
        let job = {
            let mut s = self.state.lock().unwrap();
            let running = s.entries.iter().filter(|e| e.job.kind == Kind::Process && e.job.state.active()).count();
            if running >= MAX_PROCESSES {
                return Err(format!(
                    "There are already {MAX_PROCESSES} background processes running; stop one before starting another."
                ));
            }
            // Reserved before spawning, so two starts at once cannot pass the limit together.
            let number = s.next_number;
            s.next_number += 1;
            let job = Job {
                id: format!("j{number}"),
                number,
                session: start.session.clone(),
                kind: Kind::Process,
                title: start.title.clone(),
                command: Some(start.command.clone()),
                task: None,
                folder: shown(&start.folder),
                state: JobState::Running,
                started_at: now_ms(),
                ended_at: None,
                url: None,
                exit: None,
                isolation: None,
                worktree: None,
                writes: false,
                result: None,
                conflict: false,
                dismissed: false,
            };
            s.entries.push(Entry { job: job.clone(), output: VecDeque::new(), group: 0, stopping: false, agent: None });
            job
        };
        let ProcessStart { line, folder, env, .. } = start;
        let result = on_spawner(&self.spawner, move || spawn_process(&line, &folder, &env));
        let mut s = self.state.lock().unwrap();
        match result {
            Ok((child, pipe)) => {
                if let Some(entry) = s.entry(&job.id) {
                    entry.group = child.id() as i32;
                }
                s.dirty = true;
                s.save();
                drop(s);
                let (drained_tx, drained) = mpsc::channel();
                let (id, state) = (job.id.clone(), self.state.clone());
                std::thread::spawn(move || read_output(id, pipe, state, drained_tx));
                let (id, state) = (job.id.clone(), self.state.clone());
                std::thread::spawn(move || wait_exit(id, child, state, drained));
                Ok(job)
            }
            Err(e) => {
                s.entries.retain(|entry| entry.job.id != job.id);
                Err(format!("Could not start the process: {e}"))
            }
        }
    }

    /// The latest `lines` lines of a job's output.
    pub fn output(&self, id: &str, lines: usize) -> Option<Vec<String>> {
        let mut s = self.state.lock().unwrap();
        let entry = s.entry(id)?;
        let skip = entry.output.len().saturating_sub(lines);
        Some(entry.output.iter().skip(skip).cloned().collect())
    }

    /// Marks running jobs as stopping and returns their groups, and whether any job was stopped. A queued subagent is
    /// stopped here and never starts; one still being started is stopped as soon as it has a group.
    fn stopping(&self, which: impl Fn(&Job) -> bool) -> (Vec<i32>, bool) {
        let mut s = self.state.lock().unwrap();
        let (mut groups, mut any, mut dropped) = (Vec::new(), false, Vec::new());
        let now = now_ms();
        for e in s.entries.iter_mut().filter(|e| which(&e.job)) {
            match e.job.state {
                JobState::Running if e.group > 0 => {
                    e.stopping = true;
                    groups.push(e.group);
                    any = true;
                }
                JobState::Running if e.job.kind == Kind::Agent => {
                    e.stopping = true;
                    any = true;
                }
                JobState::Queued => {
                    e.job.state = JobState::Stopped;
                    e.job.ended_at = Some(now);
                    e.agent = None;
                    dropped.push(e.job.clone());
                    any = true;
                }
                _ => {}
            }
        }
        let save = !dropped.is_empty();
        for job in dropped {
            let session = job.session.clone();
            s.emit(&session, |seq| JobEvent::Ended { seq, job, by: EndedBy::Stopped, last_lines: Vec::new() });
        }
        if save {
            s.prune();
            s.save();
        }
        if any {
            s.dirty = true;
        }
        (groups, any)
    }

    /// Stops a running or queued job and everything it forked, without waiting. False when it is neither.
    pub fn stop(&self, id: &str) -> bool {
        let (groups, any) = self.stopping(|j| j.id == id);
        if !groups.is_empty() {
            std::thread::spawn(move || end_groups(&groups));
        }
        any
    }

    pub fn get(&self, id: &str) -> Option<Job> {
        self.state.lock().unwrap().entry(id).map(|e| e.job.clone())
    }

    /// The jobs (of one session, or all): running and queued first, oldest first, then finished, newest first.
    pub fn list(&self, session: Option<&str>) -> Vec<Job> {
        let s = self.state.lock().unwrap();
        let mine = |j: &&Job| session.is_none_or(|session| j.session == session);
        let jobs: Vec<&Job> = s.entries.iter().map(|e| &e.job).filter(mine).collect();
        let mut active: Vec<Job> = jobs.iter().filter(|j| j.state.active()).map(|j| (*j).clone()).collect();
        let mut finished: Vec<Job> = jobs.iter().filter(|j| !j.state.active()).map(|j| (*j).clone()).collect();
        active.sort_by_key(|j| j.number);
        finished.sort_by_key(|j| std::cmp::Reverse((j.ended_at, j.number)));
        active.extend(finished);
        active
    }

    /// Jobs running or queued.
    pub fn active(&self) -> usize {
        self.state.lock().unwrap().entries.iter().filter(|e| e.job.state.active()).count()
    }

    /// Stops every running job of a session (its tab closed), without waiting; its queued subagents never start.
    pub fn stop_session(&self, session: &str) {
        let (groups, _) = self.stopping(|j| j.session == session);
        if !groups.is_empty() {
            std::thread::spawn(move || end_groups(&groups));
        }
    }

    /// Stops every running job and returns once their groups are gone (the dock is quitting), at most `STOP_GRACE`
    /// and a second later.
    pub fn stop_all(&self) {
        let (groups, _) = self.stopping(|_| true);
        end_groups(&groups);
        // The waiters record each exit as the groups go; give them a moment to catch up.
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline && self.active() > 0 {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Hides a finished job from the page's list; it stays until it is forgotten.
    pub fn dismiss(&self, id: &str) {
        let mut s = self.state.lock().unwrap();
        if let Some(entry) = s.entry(id) {
            entry.job.dismissed = true;
            s.dirty = true;
            s.save();
        }
    }

    /// The session's events not yet acknowledged, in `seq` order.
    pub fn events(&self, session: &str) -> Vec<JobEvent> {
        self.state.lock().unwrap().events.get(session).cloned().unwrap_or_default()
    }

    /// The session's Pi has this event: it is not given again (not even after a restart), and its jobs may now be
    /// forgotten.
    pub fn ack(&self, session: &str, seq: u64) {
        let mut s = self.state.lock().unwrap();
        let Some(queue) = s.events.get_mut(session) else { return };
        queue.retain(|e| e.seq() != seq);
        if queue.is_empty() {
            s.events.remove(session);
        }
        s.prune();
        s.dirty = true;
        s.save();
    }

    /// The session is gone (its tab closed): its events are dropped unread, its subscriber with them, and the jobs they
    /// held may be forgotten.
    pub fn forget_session(&self, session: &str) {
        let mut s = self.state.lock().unwrap();
        s.subscribers.remove(session);
        if s.events.remove(session).is_some() {
            s.prune();
            s.dirty = true;
            s.save();
        }
    }

    /// Pushes the session's new events to `tx` from now on. One subscriber per session: a new one replaces the old.
    /// Events already queued are not pushed; `events` gives them.
    pub fn subscribe(&self, session: &str, tx: mpsc::Sender<JobEvent>) {
        self.state.lock().unwrap().subscribers.insert(session.to_string(), tx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) fn jobs() -> (tempfile::TempDir, Jobs, Arc<AtomicUsize>) {
        let dir = tempfile::tempdir().unwrap();
        let changes = Arc::new(AtomicUsize::new(0));
        let counter = changes.clone();
        let jobs = Jobs::new(
            dir.path().join("jobs.json"),
            Arc::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
        );
        (dir, jobs, changes)
    }

    fn start(jobs: &Jobs, folder: &Path, line: &str) -> Job {
        jobs.start_process(ProcessStart {
            session: "s1".into(),
            title: line.into(),
            command: line.into(),
            line: line.into(),
            folder: folder.to_path_buf(),
            env: Vec::new(),
        })
        .unwrap()
    }

    pub(super) fn eventually(what: &str, within: Duration, test: impl Fn() -> bool) {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            if test() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("{what}");
    }

    fn group(jobs: &Jobs, id: &str) -> i32 {
        jobs.state.lock().unwrap().entries.iter().find(|e| e.job.id == id).unwrap().group
    }

    #[test]
    fn a_process_runs_and_its_output_is_kept() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let job = start(&jobs, dir.path(), "echo hi; echo there; sleep 5");
        assert_eq!(job.id, format!("j{}", job.number));
        assert_eq!(job.command.as_deref(), Some("echo hi; echo there; sleep 5"));
        eventually("two lines", Duration::from_secs(2), || {
            jobs.output(&job.id, 10) == Some(vec!["hi".to_string(), "there".to_string()])
        });
        assert_eq!(jobs.get(&job.id).unwrap().state, JobState::Running);
        assert_eq!(jobs.active(), 1);
    }

    #[test]
    fn stdout_and_stderr_share_one_output_in_order() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let job = start(&jobs, dir.path(), "echo one; echo two >&2; echo three; sleep 5");
        eventually("three lines in order", Duration::from_secs(2), || {
            jobs.output(&job.id, 10) == Some(vec!["one".into(), "two".into(), "three".into()])
        });
    }

    #[test]
    fn exit_code_and_state_are_recorded() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let failing = start(&jobs, dir.path(), "exit 3");
        let fine = start(&jobs, dir.path(), "true");
        for (job, code) in [(&failing, 3), (&fine, 0)] {
            eventually("exited", Duration::from_secs(2), || {
                jobs.get(&job.id).is_some_and(|j| j.state == JobState::Exited && j.exit == Some(code))
            });
            assert!(jobs.get(&job.id).unwrap().ended_at.is_some());
        }
        assert_eq!(jobs.active(), 0);
    }

    #[test]
    fn stop_kills_the_whole_group() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let job = start(&jobs, dir.path(), r#"sh -c 'trap "" TERM; sleep 60' & sleep 60"#);
        let group = group(&jobs, &job.id);
        std::thread::sleep(Duration::from_millis(200));
        assert!(jobs.stop(&job.id));
        eventually("stopped and the group gone", Duration::from_secs(5), || {
            let gone = unsafe { libc::kill(-group, 0) } != 0
                && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
            gone && jobs.get(&job.id).unwrap().state == JobState::Stopped
        });
        assert!(!jobs.stop(&job.id), "a finished job is not stopped twice");
    }

    #[test]
    fn the_first_local_url_is_the_jobs_url() {
        assert_eq!(detect_url("  ▲ ready on http://0.0.0.0:3000/x"), Some("http://localhost:3000/x".into()));
        assert_eq!(detect_url("listening at https://127.0.0.1:8443"), Some("https://127.0.0.1:8443".into()));
        assert_eq!(detect_url("docs at http://example.com"), None);
        assert_eq!(detect_url("Local: http://[::1]:5173/"), Some("http://[::1]:5173/".into()));
        assert_eq!(
            detect_url("see http://example.com, then http://localhost:8080."),
            Some("http://localhost:8080".into()),
            "the first local one, without the sentence's period"
        );
    }

    #[test]
    fn ansi_escapes_are_removed() {
        assert_eq!(strip_ansi("\u{1b}[32mready\u{1b}[0m"), "ready");
        assert_eq!(strip_ansi("\u{1b}]8;;http://x\u{7}link\u{1b}]8;;\u{7} ok"), "link ok");
    }

    #[test]
    fn a_url_is_found_in_the_output() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let job = start(
            &jobs,
            dir.path(),
            r"printf '\033[1mready on http://0.0.0.0:4000\033[0m\nhttp://localhost:9\n'; sleep 5",
        );
        eventually("the url", Duration::from_secs(2), || {
            jobs.get(&job.id).unwrap().url.as_deref() == Some("http://localhost:4000")
        });
    }

    #[test]
    fn output_is_bounded_by_lines_and_chars() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let many = start(&jobs, dir.path(), "yes | head -n 5000; sleep 5");
        eventually("the last line kept", Duration::from_secs(3), || {
            jobs.output(&many.id, 5000).is_some_and(|o| o.len() == OUTPUT_LINES)
        });
        let long = start(&jobs, dir.path(), "head -c 5000 /dev/zero | tr '\\0' a; echo; echo after; sleep 5");
        eventually("the long line and the next", Duration::from_secs(3), || {
            jobs.output(&long.id, 10).is_some_and(|o| o.len() == 2)
        });
        let out = jobs.output(&long.id, 10).unwrap();
        assert_eq!(out[0], "a".repeat(LINE_CHARS));
        assert_eq!(out[1], "after");
    }

    #[test]
    fn an_eleventh_process_is_refused() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        for _ in 0..MAX_PROCESSES {
            start(&jobs, dir.path(), "sleep 30");
        }
        let refused = jobs.start_process(ProcessStart {
            session: "s1".into(),
            title: "one more".into(),
            command: "sleep 30".into(),
            line: "sleep 30".into(),
            folder: dir.path().to_path_buf(),
            env: Vec::new(),
        });
        assert!(refused.unwrap_err().contains("10 background processes"));
        assert_eq!(jobs.active(), MAX_PROCESSES);
        jobs.stop_all();
        assert_eq!(jobs.active(), 0, "stop_all returns once every group is gone");
    }

    #[test]
    fn the_folder_and_env_are_the_ones_given() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let folder = dir.path().canonicalize().unwrap();
        let job = jobs
            .start_process(ProcessStart {
                session: "s1".into(),
                title: "where".into(),
                command: "pwd; echo $X".into(),
                line: "pwd; echo $X".into(),
                folder: folder.clone(),
                env: vec![("X".into(), "1".into())],
            })
            .unwrap();
        eventually("pwd and X", Duration::from_secs(2), || {
            jobs.output(&job.id, 10) == Some(vec![folder.display().to_string(), "1".to_string()])
        });
    }

    #[test]
    fn changes_are_reported_at_most_four_times_a_second() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, changes) = jobs();
        let job = start(&jobs, dir.path(), "for i in $(seq 100); do echo $i; done; sleep 5");
        eventually("a hundred lines", Duration::from_secs(1), || {
            jobs.output(&job.id, 200).is_some_and(|o| o.len() == 100)
        });
        std::thread::sleep(Duration::from_millis(300));
        let seen = changes.load(Ordering::SeqCst);
        assert!((1..=5).contains(&seen), "{seen} changes");
    }

    #[test]
    fn a_session_is_listed_running_first_and_dismissed_jobs_are_marked() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let done = start(&jobs, dir.path(), "true");
        eventually("done", Duration::from_secs(2), || jobs.active() == 0);
        let running = start(&jobs, dir.path(), "sleep 30");
        let ids: Vec<String> = jobs.list(Some("s1")).into_iter().map(|j| j.id).collect();
        assert_eq!(ids, vec![running.id.clone(), done.id.clone()]);
        assert!(jobs.list(Some("other")).is_empty());
        jobs.dismiss(&done.id);
        assert!(jobs.get(&done.id).unwrap().dismissed);
        jobs.stop_session("s1");
        eventually("stopped", Duration::from_secs(5), || jobs.get(&running.id).unwrap().state == JobState::Stopped);
    }

    fn start_in(jobs: &Jobs, session: &str, folder: &Path, line: &str) -> Job {
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

    pub(super) fn ended(event: &JobEvent) -> (&Job, EndedBy, &[String]) {
        match event {
            JobEvent::Ended { job, by, last_lines, .. } => (job, *by, last_lines),
            other => panic!("not an end: {other:?}"),
        }
    }

    #[test]
    fn a_process_that_ends_by_itself_leaves_an_event() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let job = start(&jobs, dir.path(), "echo bye; exit 1");
        eventually("an event", Duration::from_secs(2), || !jobs.events("s1").is_empty());
        let events = jobs.events("s1");
        assert_eq!(events.len(), 1);
        let (ended_job, by, last_lines) = ended(&events[0]);
        assert_eq!(ended_job.id, job.id);
        assert_eq!(ended_job.state, JobState::Exited);
        assert_eq!(ended_job.exit, Some(1));
        assert_eq!(by, EndedBy::Itself);
        assert_eq!(last_lines, ["bye".to_string()]);
        assert!(jobs.events("s2").is_empty());
    }

    #[test]
    fn the_last_lines_are_all_there_even_when_output_ends_with_the_exit() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        for _ in 0..5 {
            start(&jobs, dir.path(), "seq 30");
        }
        eventually("five events", Duration::from_secs(3), || jobs.events("s1").len() == 5);
        let expected: Vec<String> = (11..=30).map(|n| n.to_string()).collect();
        for event in jobs.events("s1") {
            assert_eq!(ended(&event).2, expected.as_slice());
        }
    }

    #[test]
    fn a_stopped_process_says_it_was_stopped() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let job = start(&jobs, dir.path(), "sleep 30");
        assert!(jobs.stop(&job.id));
        eventually("an event", Duration::from_secs(5), || !jobs.events("s1").is_empty());
        let events = jobs.events("s1");
        let (ended_job, by, _) = ended(&events[0]);
        assert_eq!(ended_job.state, JobState::Stopped);
        assert_eq!(by, EndedBy::Stopped);
    }

    #[test]
    fn events_stay_until_acked_and_survive_reconnect() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let first = start(&jobs, dir.path(), "true");
        eventually("first end", Duration::from_secs(2), || jobs.events("s1").len() == 1);
        let second = start(&jobs, dir.path(), "true");
        eventually("second end", Duration::from_secs(2), || jobs.events("s1").len() == 2);
        let events = jobs.events("s1");
        assert_eq!(ended(&events[0]).0.id, first.id);
        assert_eq!(ended(&events[1]).0.id, second.id);
        let (seq1, seq2) = (events[0].seq(), events[1].seq());
        assert!(seq1 < seq2);
        jobs.ack("s1", seq1);
        assert_eq!(jobs.events("s1").iter().map(JobEvent::seq).collect::<Vec<_>>(), vec![seq2]);

        // A Pi that restarts subscribes again: what ends now reaches it and stays until acknowledged.
        let (old_tx, old_rx) = mpsc::channel();
        jobs.subscribe("s1", old_tx);
        let (tx, rx) = mpsc::channel();
        jobs.subscribe("s1", tx);
        let third = start(&jobs, dir.path(), "true");
        let pushed = rx.recv_timeout(Duration::from_secs(2)).expect("the third end is pushed");
        assert_eq!(ended(&pushed).0.id, third.id);
        assert!(old_rx.try_recv().is_err(), "the replaced subscriber hears nothing");
        let pending: Vec<u64> = jobs.events("s1").iter().map(JobEvent::seq).collect();
        assert_eq!(pending, vec![seq2, pushed.seq()]);
        jobs.ack("s1", seq2);
        jobs.ack("s1", pushed.seq());
        assert!(jobs.events("s1").is_empty());
    }

    fn stored_job(number: u64, session: &str, state: JobState) -> Job {
        Job {
            id: format!("j{number}"),
            number,
            session: session.into(),
            kind: Kind::Process,
            title: "dev".into(),
            command: Some("npm run dev".into()),
            task: None,
            folder: "~/p".into(),
            state,
            started_at: 1,
            ended_at: (state != JobState::Running).then_some(2),
            url: None,
            exit: (state == JobState::Exited).then_some(0),
            isolation: None,
            worktree: None,
            writes: false,
            result: None,
            conflict: false,
            dismissed: false,
        }
    }

    fn jobs_at(store: &Path) -> Jobs {
        Jobs::new(store.to_path_buf(), Arc::new(|| {}))
    }

    #[test]
    fn running_jobs_in_the_store_come_back_cut() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("jobs.json");
        let running = stored_job(4, "s1", JobState::Running);
        let exited = stored_job(3, "s1", JobState::Exited);
        let value = serde_json::json!({ "nextSeq": 7, "nextNumber": 5, "jobs": [running, exited] });
        aria_core::json_file::write_json_atomic(&store, &value, 0o600).unwrap();

        let before = now_ms();
        let jobs = jobs_at(&store);
        let listed = jobs.list(Some("s1"));
        assert_eq!(listed.len(), 2);
        let cut = jobs.get("j4").unwrap();
        assert_eq!(cut.state, JobState::Cut);
        assert!(cut.ended_at.is_some_and(|t| t >= before));
        assert_eq!(jobs.get("j3").unwrap(), exited);
        let events = jobs.events("s1");
        assert_eq!(events, vec![JobEvent::Cut { seq: 7, jobs: vec![cut.clone()] }]);

        // The cut is written back: a second load cuts nothing again (the unacknowledged cut is still the only event),
        // and the numbers go on.
        drop(jobs);
        let again = jobs_at(&store);
        assert_eq!(again.events("s1"), events);
        assert_eq!(again.get("j4").unwrap().state, JobState::Cut);
        let text = std::fs::read_to_string(&store).unwrap();
        let saved: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(saved["nextSeq"], 8);
        assert_eq!(saved["nextNumber"], 5);
    }

    #[test]
    fn events_serialize_as_pi_reads_them() {
        let job = stored_job(1, "s1", JobState::Stopped);
        let ended = JobEvent::Ended { seq: 2, job: job.clone(), by: EndedBy::Stopped, last_lines: vec!["x".into()] };
        let value = serde_json::to_value(&ended).unwrap();
        assert_eq!(value["event"], "ended");
        assert_eq!(value["by"], "stop");
        assert_eq!(value["lastLines"][0], "x");
        assert_eq!(value["job"]["startedAt"], 1);
        assert_eq!(serde_json::to_value(EndedBy::Itself).unwrap(), "self");
        let cut = serde_json::to_value(JobEvent::Cut { seq: 3, jobs: vec![job] }).unwrap();
        assert_eq!(cut["event"], "cut");
        assert_eq!(cut["seq"], 3);
    }

    #[test]
    fn the_store_has_no_output() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        // The output says what the command line does not, so only the output could put it in the store.
        let job = start(&jobs, dir.path(), "echo secret-$((6 * 7))-ish");
        eventually("ended", Duration::from_secs(2), || !jobs.events("s1").is_empty());
        assert_eq!(jobs.output(&job.id, 1), Some(vec!["secret-42-ish".to_string()]));
        // An unacknowledged end keeps its own last lines in the store; once acknowledged, nothing of the output is left.
        let seq = jobs.events("s1")[0].seq();
        jobs.ack("s1", seq);
        let store = std::fs::read_to_string(dir.path().join("jobs.json")).unwrap();
        assert!(store.contains(&job.id), "the job is stored");
        assert!(!store.contains("secret-42-ish"), "{store}");
        let mode = std::fs::metadata(dir.path().join("jobs.json")).unwrap().permissions();
        assert_eq!(std::os::unix::fs::PermissionsExt::mode(&mode) & 0o777, 0o600);
    }

    #[test]
    fn stop_session_stops_only_that_session() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let mine = start_in(&jobs, "s1", dir.path(), "sleep 30");
        let theirs = start_in(&jobs, "s2", dir.path(), "sleep 30");
        jobs.stop_session("s1");
        eventually("s1 stopped", Duration::from_secs(5), || jobs.get(&mine.id).unwrap().state == JobState::Stopped);
        assert_eq!(jobs.get(&theirs.id).unwrap().state, JobState::Running);
        assert!(jobs.events("s2").is_empty());
    }

    #[test]
    fn finished_jobs_are_forgotten_only_once_their_end_is_acked() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        for _ in 0..FINISHED_KEPT + 3 {
            start(&jobs, dir.path(), "true");
        }
        eventually("every end", Duration::from_secs(5), || jobs.events("s1").len() == FINISHED_KEPT + 3);
        assert_eq!(jobs.list(None).len(), FINISHED_KEPT + 3, "unacknowledged ends keep their jobs");
        for event in jobs.events("s1") {
            jobs.ack("s1", event.seq());
        }
        assert_eq!(jobs.list(None).len(), FINISHED_KEPT);
    }

    /// Runs `line` in `session` and waits for its end event.
    fn run_to_end(jobs: &Jobs, session: &str, folder: &Path, line: &str) -> Job {
        let job = start_in(jobs, session, folder, line);
        eventually("its end", Duration::from_secs(3), || {
            jobs.events(session).iter().any(|e| matches!(e, JobEvent::Ended { job: j, .. } if j.id == job.id))
        });
        job
    }

    #[test]
    fn ends_held_for_their_pi_keep_their_job_but_not_their_output() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let ran: Vec<Job> = (0..FINISHED_KEPT + 3).map(|_| run_to_end(&jobs, "s1", dir.path(), "echo x")).collect();
        assert_eq!(jobs.list(None).len(), FINISHED_KEPT + 3);
        for (i, job) in ran.iter().enumerate() {
            let expected: Vec<String> = if i < 3 { Vec::new() } else { vec!["x".into()] };
            assert_eq!(jobs.output(&job.id, 10), Some(expected), "{}", job.id);
        }
        let first = &jobs.events("s1")[0];
        assert_eq!(ended(first).2, ["x".to_string()], "the event keeps its own last lines");
    }

    #[test]
    fn a_forgotten_session_loses_its_events_and_subscriber() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let (tx, rx) = mpsc::channel();
        jobs.subscribe("s1", tx);
        for _ in 0..FINISHED_KEPT + 3 {
            run_to_end(&jobs, "s1", dir.path(), "true");
        }
        let other = run_to_end(&jobs, "s2", dir.path(), "true");
        while rx.try_recv().is_ok() {}
        jobs.forget_session("s1");
        assert!(jobs.events("s1").is_empty());
        assert_eq!(rx.try_recv(), Err(mpsc::TryRecvError::Disconnected), "the subscriber is dropped");
        assert_eq!(jobs.list(None).len(), FINISHED_KEPT, "the jobs it held may be forgotten");
        assert!(jobs.get(&other.id).is_some(), "the other session's held job stays");
        assert_eq!(jobs.events("s2").len(), 1);
    }

    #[test]
    fn unacked_events_survive_a_restart() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("jobs.json");
        let jobs = jobs_at(&store);
        let done = run_to_end(&jobs, "s1", dir.path(), "echo bye");
        let running = start_in(&jobs, "s1", dir.path(), "sleep 30");
        let before = jobs.events("s1");
        // The store as a dock that crashed now leaves it; this dock's own waiters would record the sleep's end.
        let crashed = std::fs::read(&store).unwrap();
        jobs.stop_all();
        drop(jobs);
        std::fs::write(&store, crashed).unwrap();

        // The old dock went away with `running` still going: its end is kept, and the cut comes after it.
        let again = jobs_at(&store);
        let events = again.events("s1");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], before[0]);
        assert_eq!(ended(&events[0]).0.id, done.id);
        assert_eq!(ended(&events[0]).2, ["bye".to_string()]);
        match &events[1] {
            JobEvent::Cut { seq, jobs } => {
                assert!(*seq > events[0].seq());
                assert_eq!(jobs.iter().map(|j| j.id.clone()).collect::<Vec<_>>(), vec![running.id.clone()]);
            }
            other => panic!("not a cut: {other:?}"),
        }
        again.ack("s1", events[0].seq());
        drop(again);

        let third = jobs_at(&store);
        let left: Vec<u64> = third.events("s1").iter().map(JobEvent::seq).collect();
        assert_eq!(left, vec![events[1].seq()], "only the unacknowledged cut is left");
    }
}
