//! Background work the dock owns: processes a session's Pi starts (dev servers, watchers, long builds) that keep
//! running while the conversation goes on. The dock, not Pi, is their parent, so a Pi that restarts never takes them
//! down. Every child is spawned from one long-lived thread and leads its own process group, so Stop reaches whatever it
//! forked, and the page is told about changes at most four times a second.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::os::fd::{FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Processes running at once, across every session.
const MAX_PROCESSES: usize = 10;
/// Lines of output kept per job (the latest ones).
const OUTPUT_LINES: usize = 1000;
/// Characters kept of one line of output.
const LINE_CHARS: usize = 2000;
/// Finished jobs kept in memory; older ones are forgotten.
const FINISHED_KEPT: usize = 20;
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
}

#[derive(Default)]
struct State {
    entries: Vec<Entry>,
    next_number: u64,
    dirty: bool,
}

impl State {
    fn entry(&mut self, id: &str) -> Option<&mut Entry> {
        self.entries.iter_mut().find(|e| e.job.id == id)
    }

    /// Forgets the oldest finished jobs beyond `FINISHED_KEPT`, so a long day of jobs keeps the dock's memory bounded.
    fn prune(&mut self) {
        let mut finished: Vec<(i64, u64)> = self
            .entries
            .iter()
            .filter(|e| !e.job.state.active())
            .map(|e| (e.job.ended_at.unwrap_or(0), e.job.number))
            .collect();
        if finished.len() <= FINISHED_KEPT {
            return;
        }
        finished.sort();
        let gone: Vec<u64> = finished[..finished.len() - FINISHED_KEPT].iter().map(|(_, n)| *n).collect();
        self.entries.retain(|e| e.job.state.active() || !gone.contains(&e.job.number));
    }
}

/// Work for the spawner thread: a closure that spawns and sends its own reply.
type Spawn = Box<dyn FnOnce() + Send>;

/// The dock's background jobs. Cheap to clone; every clone is the same jobs.
#[derive(Clone)]
pub struct Jobs {
    state: Arc<Mutex<State>>,
    spawner: mpsc::Sender<Spawn>,
    /// Where the jobs are kept across restarts.
    store: PathBuf,
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

/// A pipe both ends close-on-exec: the child only gets the write end through its stdout and stderr.
fn pipe() -> std::io::Result<(File, OwnedFd)> {
    let mut fds = [0; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let (read, write) = unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) };
    for fd in fds {
        unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
    }
    Ok((File::from(read), write))
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
    // Its own group, so Stop reaches everything it forks.
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    // If the spawner thread dies (the dock exits or crashes), the child is told to end instead of living on unseen.
    #[cfg(target_os = "linux")]
    unsafe {
        std::os::unix::process::CommandExt::pre_exec(&mut command, || {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }
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

/// Reads a job's output until every process holding the pipe has closed it.
fn read_output(id: String, pipe: File, state: Arc<Mutex<State>>) {
    let mut reader = BufReader::new(pipe);
    loop {
        let mut bytes = Vec::new();
        match (&mut reader).take(LINE_BYTES).read_until(b'\n', &mut bytes) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if bytes.last() != Some(&b'\n') && bytes.len() as u64 >= LINE_BYTES {
            // too long: skip the rest of it, a bounded chunk at a time
            loop {
                let mut rest = Vec::new();
                match (&mut reader).take(LINE_BYTES).read_until(b'\n', &mut rest) {
                    Ok(0) | Err(_) => break,
                    Ok(_) if rest.last() == Some(&b'\n') => break,
                    Ok(_) => {}
                }
            }
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

/// Waits for the child and records how it ended.
fn wait_exit(id: String, mut child: Child, state: Arc<Mutex<State>>) {
    let status = child.wait();
    let mut s = state.lock().unwrap();
    if let Some(entry) = s.entry(&id) {
        let code = status.ok().and_then(|st| st.code());
        entry.job.exit = code;
        entry.job.ended_at = Some(now_ms());
        entry.job.state = match (entry.stopping, code) {
            (true, _) => JobState::Stopped,
            (false, Some(_)) => JobState::Exited,
            // killed by a signal nobody here sent
            (false, None) => JobState::Failed,
        };
    }
    s.prune();
    s.dirty = true;
}

impl Jobs {
    /// The jobs, kept in `store`. `on_change` runs at most every 250 ms while something changed.
    ///
    /// The spawner thread lives as long as a clone of these jobs does: children are told to end (`PR_SET_PDEATHSIG`)
    /// when the thread that spawned them goes, so it must be one thread that outlives them all.
    pub fn new(store: PathBuf, on_change: Arc<dyn Fn() + Send + Sync>) -> Jobs {
        let state = Arc::new(Mutex::new(State { next_number: 1, ..State::default() }));
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
        Jobs { state, spawner, store }
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
            s.entries.push(Entry { job: job.clone(), output: VecDeque::new(), group: 0, stopping: false });
            job
        };
        let (reply, spawned) = mpsc::channel();
        let ProcessStart { line, folder, env, .. } = start;
        let sent = self.spawner.send(Box::new(move || {
            let _ = reply.send(spawn_process(&line, &folder, &env));
        }));
        let result = match sent {
            Ok(()) => spawned.recv().unwrap_or_else(|_| Err(std::io::Error::other("the spawner is gone"))),
            Err(_) => Err(std::io::Error::other("the spawner is gone")),
        };
        let mut s = self.state.lock().unwrap();
        match result {
            Ok((child, pipe)) => {
                if let Some(entry) = s.entry(&job.id) {
                    entry.group = child.id() as i32;
                }
                s.dirty = true;
                drop(s);
                let (id, state) = (job.id.clone(), self.state.clone());
                std::thread::spawn(move || read_output(id, pipe, state));
                let (id, state) = (job.id.clone(), self.state.clone());
                std::thread::spawn(move || wait_exit(id, child, state));
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

    /// Marks running jobs as stopping and returns their groups.
    fn stopping(&self, which: impl Fn(&Job) -> bool) -> Vec<i32> {
        let mut s = self.state.lock().unwrap();
        let groups: Vec<i32> = s
            .entries
            .iter_mut()
            .filter(|e| e.job.state == JobState::Running && e.group > 0 && which(&e.job))
            .map(|e| {
                e.stopping = true;
                e.group
            })
            .collect();
        if !groups.is_empty() {
            s.dirty = true;
        }
        groups
    }

    /// Stops a running job and everything it forked, without waiting. False when it is not running.
    pub fn stop(&self, id: &str) -> bool {
        let groups = self.stopping(|j| j.id == id);
        if groups.is_empty() {
            return false;
        }
        std::thread::spawn(move || end_groups(&groups));
        true
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

    /// Stops every running job of a session (its tab closed), without waiting.
    pub fn stop_session(&self, session: &str) {
        let groups = self.stopping(|j| j.session == session);
        if !groups.is_empty() {
            std::thread::spawn(move || end_groups(&groups));
        }
    }

    /// Stops every running job and returns once their groups are gone (the dock is quitting), at most `STOP_GRACE`
    /// and a second later.
    pub fn stop_all(&self) {
        let groups = self.stopping(|_| true);
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn jobs() -> (tempfile::TempDir, Jobs, Arc<AtomicUsize>) {
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

    fn eventually(what: &str, within: Duration, test: impl Fn() -> bool) {
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
}
