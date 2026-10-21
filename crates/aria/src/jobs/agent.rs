//! Background subagents: `pi --mode json -p --no-session … "Task: …"` runs the dock starts for a session. At most
//! `MAX_AGENTS_RUNNING` run at once and the rest wait, queued, in order. Two writers never share a folder: a second one
//! for the same folder waits, or, in a git repository, works in a worktree whose changes come back when it is done.

use super::*;
use crate::worktree::{self, Back, Worktree};

/// Runs with a subagent's job right before its child is spawned (the dock names it in the session's watch).
pub type BeforeSpawn = Arc<dyn Fn(&Job) + Send + Sync>;

/// A subagent to start.
#[derive(Clone)]
pub struct AgentStart {
    pub session: String,
    pub title: String,
    pub task: String,
    pub folder: PathBuf,
    pub writes: bool,
    /// `auto`, `folder` or `worktree`.
    pub isolation: String,
    /// Where its worktree goes, if it gets one: `.aria-worktrees` in the projects folder.
    pub worktrees: PathBuf,
    /// The `pi` shim the session uses.
    pub program: PathBuf,
    /// `--model`/`--thinking` pairs.
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub unset: Vec<String>,
    pub before_spawn: Option<BeforeSpawn>,
}

/// What the dock keeps of a subagent while it is queued or running.
pub(super) struct AgentRun {
    start: AgentStart,
    /// The git root its folder is in, for a writer.
    root: Option<PathBuf>,
    /// For a writer, what one folder-mode writer at a time holds: the git root, or the canonical folder.
    key: Option<PathBuf>,
    holds_key: bool,
    worktree: Option<Worktree>,
    answer: Option<Answer>,
    /// The last lines of its stderr, said when it fails without an answer.
    stderr: VecDeque<String>,
}

/// The last assistant message on a subagent's stdout.
struct Answer {
    text: String,
    /// `stopReason: "error"`: the API failed.
    failed: bool,
    error: Option<String>,
}

/// Bytes read of one line of a subagent's JSON before the rest is skipped. A message carries a whole answer, so this is
/// far more than a line of a process's output, and still bounded.
const AGENT_LINE_BYTES: u64 = 1 << 20;
/// Lines of a subagent's stderr kept to explain a failure.
const STDERR_LINES: usize = 5;

fn clip(text: &str) -> String {
    text.chars().take(RESULT_CHARS).collect()
}

/// The assistant's answer in one line of `pi --mode json`, if the line ends an assistant message.
fn answer(line: &[u8]) -> Option<Answer> {
    // Most lines are something else; only these are parsed.
    if !line.windows(11).any(|w| w == b"message_end") {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(line).ok()?;
    let message = &value["message"];
    if value["type"] != "message_end" || message["role"] != "assistant" {
        return None;
    }
    let text = match &message["content"] {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b["type"] == "text")
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };
    Some(Answer {
        text: clip(&text),
        failed: message["stopReason"] == "error",
        error: message["errorMessage"].as_str().map(clip),
    })
}

/// The task as the subagent reads it: where to work and what it may change.
fn brief(task: &str, cwd: &Path, writes: bool) -> String {
    let rule = if writes { "Do not commit." } else { "Do not change any file." };
    format!("Task: {task}\n\nWork in {}.\n{rule}", cwd.display())
}

/// Runs `pi` in JSON mode in `cwd`. Called on the spawner thread only.
fn spawn_agent(
    program: &Path,
    args: &[String],
    brief: &str,
    cwd: &Path,
    env: &[(String, String)],
    unset: &[String],
) -> std::io::Result<Child> {
    let mut command = Command::new(program);
    command
        .args(["--mode", "json", "-p", "--no-session"])
        .args(args)
        .arg(brief)
        .current_dir(cwd)
        .envs(env.iter().map(|(k, v)| (k, v)))
        // A subagent never starts background work of its own.
        .env_remove("ARIA_JOBS_SOCK")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for name in unset {
        command.env_remove(name);
    }
    own_group(&mut command);
    command.spawn()
}

/// Whether the queued subagent `id` can start now, and if so in a worktree. `None` while its folder is taken.
fn decide(s: &State, id: &str) -> Option<bool> {
    let run = s.entries.iter().find(|e| e.job.id == id)?.agent.as_ref()?;
    if !run.start.writes {
        return Some(false);
    }
    let running = |e: &&Entry| e.job.id != id && e.job.state == JobState::Running;
    let others = || s.entries.iter().filter(running).filter_map(|e| e.agent.as_ref());
    let key_free = || others().all(|r| !r.holds_key || r.key != run.key);
    match &run.root {
        Some(root) => match run.start.isolation.as_str() {
            "worktree" => Some(true),
            "auto" => Some(others().any(|r| r.start.writes && r.root.as_ref() == Some(root))),
            _ => key_free().then_some(false),
        },
        None => key_free().then_some(false),
    }
}

/// Marks the queued subagents that can start now as running, oldest first, and returns them.
fn pick(s: &mut State) -> Vec<String> {
    let mut running =
        s.entries.iter().filter(|e| e.job.kind == Kind::Agent && e.job.state == JobState::Running).count();
    let mut queued: Vec<(u64, String)> = s
        .entries
        .iter()
        .filter(|e| e.job.state == JobState::Queued && e.agent.is_some())
        .map(|e| (e.job.number, e.job.id.clone()))
        .collect();
    queued.sort();
    let mut picked = Vec::new();
    for (_, id) in queued {
        if running >= MAX_AGENTS_RUNNING {
            break;
        }
        let Some(worktree) = decide(s, &id) else { continue };
        let entry = s.entry(&id).expect("a queued agent");
        entry.job.state = JobState::Running;
        entry.job.started_at = now_ms();
        entry.job.isolation = Some(if worktree { "worktree" } else { "folder" }.into());
        let run = entry.agent.as_mut().expect("a queued agent");
        run.holds_key = !worktree && run.key.is_some();
        running += 1;
        picked.push(id);
    }
    if !picked.is_empty() {
        s.dirty = true;
    }
    picked
}

/// What a subagent's threads share: the jobs, and the spawner while the dock keeps it.
#[derive(Clone)]
struct Core {
    state: Arc<Mutex<State>>,
    spawner: std::sync::Weak<mpsc::Sender<Spawn>>,
}

impl Core {
    /// Starts every queued subagent that can start now. One that cannot be started fails, which may free a slot.
    fn schedule(&self) {
        loop {
            let picked = pick(&mut self.state.lock().unwrap());
            if picked.is_empty() {
                return;
            }
            for id in picked {
                if let Err(e) = self.launch(&id) {
                    self.fail(&id, e);
                }
            }
        }
    }

    /// Spawns a subagent `pick` marked running: its worktree first when it gets one.
    fn launch(&self, id: &str) -> Result<Job, String> {
        let (job, start, root) = {
            let mut s = self.state.lock().unwrap();
            let entry = s.entry(id).ok_or("The subagent is gone.")?;
            let run = entry.agent.as_ref().ok_or("The subagent is gone.")?;
            let start = run.start.clone();
            (entry.job.clone(), start, run.root.clone())
        };
        let mut cwd = start.folder.clone();
        let mut made = None;
        if let (Some("worktree"), Some(root)) = (job.isolation.as_deref(), &root) {
            let wt =
                worktree::create(root, &start.worktrees, id).map_err(|e| format!("Could not make a worktree: {e}"))?;
            // The same subfolder of the repository, in the worktree.
            let inside =
                start.folder.canonicalize().ok().and_then(|f| f.strip_prefix(root).ok().map(Path::to_path_buf));
            cwd = wt.path.join(inside.unwrap_or_default());
            made = Some(wt);
        }
        let job = {
            let mut s = self.state.lock().unwrap();
            let entry = s.entry(id).ok_or("The subagent is gone.")?;
            entry.job.worktree = made.as_ref().map(|wt| wt.path.display().to_string());
            if let Some(run) = entry.agent.as_mut() {
                run.worktree = made.clone();
            }
            let job = entry.job.clone();
            // Recorded before the spawn: a dock that dies now leaves a cut job that names its worktree, not an orphan.
            if made.is_some() {
                s.save();
            }
            job
        };
        if let Some(hook) = &start.before_spawn {
            hook(&job);
        }
        let brief = brief(&start.task, &cwd, start.writes);
        let spawned = match self.spawner.upgrade() {
            Some(spawner) => {
                let AgentStart { program, args, env, unset, .. } = start;
                on_spawner(&spawner, move || spawn_agent(&program, &args, &brief, &cwd, &env, &unset))
            }
            None => Err(std::io::Error::other("the dock is going away")),
        };
        let mut child = match spawned {
            Ok(child) => child,
            Err(e) => {
                // It never ran: its worktree is as it was made.
                if let (Some(root), Some(wt)) = (&root, &made) {
                    let _ = worktree::remove(root, &wt.path);
                    let mut s = self.state.lock().unwrap();
                    if let Some(entry) = s.entry(id) {
                        entry.job.worktree = None;
                        if let Some(run) = entry.agent.as_mut() {
                            run.worktree = None;
                        }
                    }
                }
                return Err(format!("Could not start the subagent: {e}"));
            }
        };
        let group = child.id() as i32;
        let (stdout, stderr) = (child.stdout.take(), child.stderr.take());
        let stop = {
            let mut s = self.state.lock().unwrap();
            let stop = s.entry(id).map(|entry| {
                entry.group = group;
                entry.stopping
            });
            s.dirty = true;
            s.save();
            stop.unwrap_or(true)
        };
        if stop {
            // Stopped while it was being started.
            std::thread::spawn(move || end_groups(&[group]));
        }
        let (drained_tx, drained) = mpsc::channel();
        if let Some(stdout) = stdout {
            let (id, state, tx) = (id.to_string(), self.state.clone(), drained_tx.clone());
            std::thread::spawn(move || read_answers(id, stdout, state, tx));
        }
        if let Some(stderr) = stderr {
            let (id, state, tx) = (id.to_string(), self.state.clone(), drained_tx);
            std::thread::spawn(move || read_stderr(id, stderr, state, tx));
        }
        let (id, core) = (id.to_string(), self.clone());
        std::thread::spawn(move || core.wait(id, child, drained));
        Ok(job)
    }

    /// A subagent that could not be started ends `Failed`, saying why.
    fn fail(&self, id: &str, why: String) {
        let mut s = self.state.lock().unwrap();
        let Some(entry) = s.entry(id) else { return };
        entry.job.state = JobState::Failed;
        entry.job.ended_at = Some(now_ms());
        entry.job.result = Some(clip(&why));
        entry.agent = None;
        let job = entry.job.clone();
        let session = job.session.clone();
        s.emit(&session, |seq| JobEvent::Ended { seq, job, by: EndedBy::Itself, last_lines: Vec::new() });
        s.prune();
        s.save();
    }

    /// Waits for the subagent and records how it ended: its answer as the result and, for a worktree agent that is
    /// done, its changes back in the person's folder. Then whatever waited for its slot or its folder starts.
    fn wait(&self, id: String, mut child: Child, drained: mpsc::Receiver<()>) {
        let status = child.wait();
        let _ = drained.recv_timeout(OUTPUT_DRAIN);
        let code = status.ok().and_then(|st| st.code());
        let ran = {
            let mut s = self.state.lock().unwrap();
            s.entry(&id).and_then(|entry| {
                // Gone: a stop that comes now must not signal a group id the system may reuse.
                entry.group = 0;
                let stopping = entry.stopping;
                let run = entry.agent.as_mut()?;
                let stderr: Vec<String> = run.stderr.drain(..).collect();
                Some((stopping, run.answer.take(), stderr, run.root.clone(), run.worktree.clone()))
            })
        };
        let Some((stopping, answer, stderr, root, worktree)) = ran else { return };
        let state = if stopping {
            JobState::Stopped
        } else if code != Some(0) || answer.as_ref().is_some_and(|a| a.failed) {
            JobState::Failed
        } else {
            JobState::Done
        };
        let text = answer.as_ref().map(|a| a.text.clone()).filter(|t| !t.is_empty());
        let result = match state {
            JobState::Failed => text
                .or_else(|| answer.and_then(|a| a.error))
                .or_else(|| (!stderr.is_empty()).then(|| clip(&stderr.join("\n")))),
            _ => text,
        };
        // Stopped or failed, it brings nothing back: its worktree stays for the person.
        let mut kept = worktree.as_ref().map(|wt| wt.path.display().to_string());
        let mut conflict = false;
        if let (JobState::Done, Some(root), Some(wt)) = (state, &root, &worktree) {
            match worktree::bring_back(root, wt) {
                Ok(Back::Applied | Back::Empty) => kept = None,
                Ok(Back::Conflict(_)) => conflict = true,
                Err(e) => {
                    eprintln!("aria: could not bring back {}: {e}", wt.path.display());
                    conflict = true;
                }
            }
        }
        {
            let mut s = self.state.lock().unwrap();
            let Some(entry) = s.entry(&id) else { return };
            entry.job.state = state;
            entry.job.exit = code;
            entry.job.ended_at = Some(now_ms());
            entry.job.result = result;
            entry.job.worktree = kept;
            entry.job.conflict = conflict;
            // Lets go of its folder.
            entry.agent = None;
            let by = if stopping { EndedBy::Stopped } else { EndedBy::Itself };
            let job = entry.job.clone();
            let session = job.session.clone();
            s.emit(&session, |seq| JobEvent::Ended { seq, job, by, last_lines: Vec::new() });
            s.prune();
            s.save();
        }
        self.schedule();
    }
}

/// Reads a subagent's JSON events and keeps its latest answer. `_drained` drops at the end.
fn read_answers(id: String, stdout: std::process::ChildStdout, state: Arc<Mutex<State>>, _drained: mpsc::Sender<()>) {
    let mut reader = BufReader::new(stdout);
    loop {
        let mut bytes = Vec::new();
        if !read_line(&mut reader, AGENT_LINE_BYTES, &mut bytes) {
            break;
        }
        let Some(answer) = answer(&bytes) else { continue };
        let mut s = state.lock().unwrap();
        let Some(run) = s.entry(&id).and_then(|e| e.agent.as_mut()) else { break };
        run.answer = Some(answer);
    }
}

/// Drains a subagent's stderr, keeping its last lines. `_drained` drops at the end.
fn read_stderr(id: String, stderr: std::process::ChildStderr, state: Arc<Mutex<State>>, _drained: mpsc::Sender<()>) {
    let mut reader = BufReader::new(stderr);
    loop {
        let mut bytes = Vec::new();
        if !read_line(&mut reader, LINE_BYTES, &mut bytes) {
            break;
        }
        let line = clean(&bytes);
        let mut s = state.lock().unwrap();
        let Some(run) = s.entry(&id).and_then(|e| e.agent.as_mut()) else { break };
        run.stderr.push_back(line);
        while run.stderr.len() > STDERR_LINES {
            run.stderr.pop_front();
        }
    }
}

impl Jobs {
    fn core(&self) -> Core {
        Core { state: self.state.clone(), spawner: Arc::downgrade(&self.spawner) }
    }

    /// Starts a subagent, or queues it when `MAX_AGENTS_RUNNING` run already or its folder is taken by another writer.
    /// Returns it `Running` or `Queued`; an error when it could not be started at all.
    pub fn start_agent(&self, start: AgentStart) -> Result<Job, String> {
        // Asked before the lock: git runs here.
        let root = if start.writes { worktree::git_root(&start.folder) } else { None };
        let key = start
            .writes
            .then(|| root.clone().unwrap_or_else(|| start.folder.canonicalize().unwrap_or(start.folder.clone())));
        let (queued, picked) = {
            let mut s = self.state.lock().unwrap();
            let number = s.next_number;
            s.next_number += 1;
            let job = Job {
                id: format!("j{number}"),
                number,
                session: start.session.clone(),
                kind: Kind::Agent,
                title: start.title.clone(),
                command: None,
                task: Some(start.task.clone()),
                folder: shown(&start.folder),
                state: JobState::Queued,
                started_at: now_ms(),
                ended_at: None,
                url: None,
                exit: None,
                isolation: None,
                worktree: None,
                writes: start.writes,
                result: None,
                conflict: false,
                dismissed: false,
            };
            let run =
                AgentRun { start, root, key, holds_key: false, worktree: None, answer: None, stderr: VecDeque::new() };
            s.entries.push(Entry {
                job: job.clone(),
                output: VecDeque::new(),
                group: 0,
                stopping: false,
                agent: Some(run),
            });
            let picked = pick(&mut s);
            s.dirty = true;
            s.save();
            (job, picked)
        };
        let core = self.core();
        let mut started = Ok(queued.clone());
        for id in picked {
            match core.launch(&id) {
                Ok(job) if id == queued.id => started = Ok(job),
                Ok(_) => {}
                Err(e) if id == queued.id => started = Err(e),
                Err(e) => core.fail(&id, e),
            }
        }
        if started.is_err() {
            let mut s = self.state.lock().unwrap();
            s.entries.retain(|e| e.job.id != queued.id);
            s.dirty = true;
            s.save();
        }
        // A start that failed may have freed a slot or a folder.
        core.schedule();
        started
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{ended, eventually, jobs};
    use super::*;
    use crate::worktree::tests::{git, repo};
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    const FAKE_PI: &str = r#"#!/bin/sh
for last; do :; done
if [ -n "$FAKE_MARKS" ]; then touch "$FAKE_MARKS/$$"; fi
if [ -n "$FAKE_ARGS" ]; then printf '%s %s %s %s %s %s' "$1" "$2" "$3" "$4" "$5" "$6" > "$FAKE_ARGS"; fi
if [ -n "$FAKE_BRIEF" ]; then printf '%s' "$last" > "$FAKE_BRIEF"; fi
if [ -n "$FAKE_SOCK" ]; then printf '%s' "${ARIA_JOBS_SOCK:-none}" > "$FAKE_SOCK"; fi
if [ -n "$FAKE_PWD" ]; then pwd -P > "$FAKE_PWD"; fi
if [ -n "$FAKE_WRITE" ]; then echo new > "$FAKE_WRITE"; fi
echo '{"type":"message_end","message":{"role":"user","content":"hi"}}'
if [ -n "$FAKE_ERROR" ]; then
  echo '{"type":"message_end","message":{"role":"assistant","content":[],"stopReason":"error","errorMessage":"boom"}}'
else
  echo '{"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":"all"}]}}'
  echo '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"all green"}],"stopReason":"stop"}}'
fi
echo noise >&2
sleep "${FAKE_SLEEP:-0}"
exit "${FAKE_EXIT:-0}"
"#;

    /// A fake `pi` in its own folder: what it does is told by its `FAKE_*` env.
    fn fake_pi() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pi");
        fs::write(&path, FAKE_PI).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        (dir, path)
    }

    fn agent(program: &Path, folder: &Path, env: &[(&str, &str)]) -> AgentStart {
        AgentStart {
            session: "s1".into(),
            title: "check".into(),
            task: "fix it".into(),
            folder: folder.to_path_buf(),
            writes: false,
            isolation: "auto".into(),
            // Beside the fake `pi`: outside any repository a test makes, as the projects folder is.
            worktrees: program.parent().unwrap().canonicalize().unwrap().join(".aria-worktrees"),
            program: program.to_path_buf(),
            args: Vec::new(),
            env: env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            unset: Vec::new(),
            before_spawn: None,
        }
    }

    fn writer(program: &Path, folder: &Path, isolation: &str, env: &[(&str, &str)]) -> AgentStart {
        AgentStart { writes: true, isolation: isolation.into(), ..agent(program, folder, env) }
    }

    fn state(jobs: &Jobs, id: &str) -> JobState {
        jobs.get(id).unwrap().state
    }

    fn ends(jobs: &Jobs, id: &str, within: Duration) -> Job {
        eventually("the agent ends", within, || !state(jobs, id).active());
        jobs.get(id).unwrap()
    }

    #[test]
    fn an_agent_ends_with_its_last_answer() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let (_pi_dir, pi) = fake_pi();
        let job = jobs.start_agent(agent(&pi, dir.path(), &[])).unwrap();
        assert_eq!(job.kind, Kind::Agent);
        assert_eq!(job.state, JobState::Running);
        assert_eq!(job.task.as_deref(), Some("fix it"));
        assert_eq!(job.isolation.as_deref(), Some("folder"));
        let done = ends(&jobs, &job.id, Duration::from_secs(3));
        assert_eq!(done.state, JobState::Done);
        assert_eq!(done.result.as_deref(), Some("all green"));
        assert_eq!(jobs.active(), 0);
        let events = jobs.events("s1");
        assert_eq!(events.len(), 1);
        let (ended_job, by, last_lines) = ended(&events[0]);
        assert_eq!(ended_job.id, job.id);
        assert_eq!(ended_job.state, JobState::Done);
        assert_eq!(by, EndedBy::Itself);
        assert!(last_lines.is_empty());
    }

    #[test]
    fn an_error_answer_or_exit_fails() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let (_pi_dir, pi) = fake_pi();
        let error = jobs.start_agent(agent(&pi, dir.path(), &[("FAKE_ERROR", "1")])).unwrap();
        let exit = jobs.start_agent(agent(&pi, dir.path(), &[("FAKE_EXIT", "2")])).unwrap();
        let error = ends(&jobs, &error.id, Duration::from_secs(3));
        assert_eq!(error.state, JobState::Failed);
        assert_eq!(error.result.as_deref(), Some("boom"), "an error with no text says the API's error");
        let exit = ends(&jobs, &exit.id, Duration::from_secs(3));
        assert_eq!(exit.state, JobState::Failed);
        assert_eq!(exit.exit, Some(2));
    }

    #[test]
    fn the_brief_says_where_and_how() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let (pi_dir, pi) = fake_pi();
        let folder = dir.path().canonicalize().unwrap();
        let (brief, args) = (pi_dir.path().join("brief"), pi_dir.path().join("args"));
        let env = [("FAKE_BRIEF", brief.to_str().unwrap()), ("FAKE_ARGS", args.to_str().unwrap())];
        let start = AgentStart { args: vec!["--model".into(), "m".into()], ..writer(&pi, &folder, "auto", &env) };
        let job = jobs.start_agent(start).unwrap();
        ends(&jobs, &job.id, Duration::from_secs(3));
        assert_eq!(fs::read_to_string(&args).unwrap(), "--mode json -p --no-session --model m");
        let text = fs::read_to_string(&brief).unwrap();
        assert_eq!(text, format!("Task: fix it\n\nWork in {}.\nDo not commit.", folder.display()));

        let job = jobs.start_agent(agent(&pi, &folder, &env)).unwrap();
        ends(&jobs, &job.id, Duration::from_secs(3));
        let text = fs::read_to_string(&brief).unwrap();
        assert!(text.ends_with("\nDo not change any file."), "{text}");
        assert!(!text.contains("Do not commit."), "{text}");
    }

    #[test]
    fn at_most_three_run_and_the_rest_queue() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let (_pi_dir, pi) = fake_pi();
        let first = jobs.start_agent(agent(&pi, dir.path(), &[("FAKE_SLEEP", "1")])).unwrap();
        let more: Vec<Job> =
            (0..3).map(|_| jobs.start_agent(agent(&pi, dir.path(), &[("FAKE_SLEEP", "30")])).unwrap()).collect();
        assert_eq!(first.state, JobState::Running);
        assert_eq!(more[0].state, JobState::Running);
        assert_eq!(more[1].state, JobState::Running);
        assert_eq!(more[2].state, JobState::Queued);
        assert_eq!(more[2].isolation, None, "decided when it starts");
        assert_eq!(jobs.active(), 4, "a queued agent is active");
        eventually("the fourth runs once the first is done", Duration::from_secs(4), || {
            state(&jobs, &first.id) == JobState::Done && state(&jobs, &more[2].id) == JobState::Running
        });
        jobs.stop_all();
        assert_eq!(jobs.active(), 0);
    }

    #[test]
    fn second_writer_in_a_plain_folder_waits() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let (_pi_dir, pi) = fake_pi();
        let first = jobs.start_agent(writer(&pi, dir.path(), "auto", &[("FAKE_SLEEP", "1")])).unwrap();
        let second = jobs.start_agent(writer(&pi, dir.path(), "worktree", &[])).unwrap();
        assert_eq!(first.state, JobState::Running);
        assert_eq!(second.state, JobState::Queued, "no worktree outside git: it waits");
        let reader = jobs.start_agent(agent(&pi, dir.path(), &[])).unwrap();
        assert_eq!(reader.state, JobState::Running, "a reader takes no key");
        let first = ends(&jobs, &first.id, Duration::from_secs(4));
        let second = ends(&jobs, &second.id, Duration::from_secs(4));
        assert_eq!(second.state, JobState::Done);
        assert_eq!(second.isolation.as_deref(), Some("folder"));
        assert!(second.started_at >= first.ended_at.unwrap(), "it started once the first ended");
    }

    #[test]
    fn auto_uses_a_worktree_when_the_repo_is_taken() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_repo_dir, root) = repo();
        fs::create_dir(root.join("sub")).unwrap();
        fs::write(root.join("sub/c.txt"), "c\n").unwrap();
        git(&root, &["add", "sub"]);
        git(&root, &["commit", "-q", "-m", "sub"]);
        let (_dir, jobs, _) = jobs();
        let (pi_dir, pi) = fake_pi();
        let a_pwd = pi_dir.path().join("a");
        let b_pwd = pi_dir.path().join("b");
        let a = writer(&pi, &root, "folder", &[("FAKE_SLEEP", "30"), ("FAKE_PWD", a_pwd.to_str().unwrap())]);
        let a = jobs.start_agent(a).unwrap();
        assert_eq!(a.isolation.as_deref(), Some("folder"));
        let b = writer(&pi, &root.join("sub"), "auto", &[("FAKE_SLEEP", "30"), ("FAKE_PWD", b_pwd.to_str().unwrap())]);
        let b = AgentStart { session: "s2".into(), ..b };
        let b = jobs.start_agent(b).unwrap();
        assert_eq!(b.state, JobState::Running);
        assert_eq!(b.isolation.as_deref(), Some("worktree"));
        let name = root.file_name().unwrap().to_string_lossy().into_owned();
        let worktree = pi_dir.path().canonicalize().unwrap().join(".aria-worktrees").join(format!("{name}-{}", b.id));
        assert_eq!(b.worktree.as_deref(), Some(worktree.to_str().unwrap()));
        eventually("both say where they are", Duration::from_secs(3), || a_pwd.exists() && b_pwd.exists());
        assert_eq!(fs::read_to_string(&a_pwd).unwrap().trim(), root.to_str().unwrap());
        assert_eq!(fs::read_to_string(&b_pwd).unwrap().trim(), worktree.join("sub").to_str().unwrap());
        jobs.stop_all();
    }

    #[test]
    fn a_done_worktree_agent_brings_its_changes_back() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_repo_dir, root) = repo();
        let (_dir, jobs, _) = jobs();
        let (pi_dir, pi) = fake_pi();
        let job = jobs.start_agent(writer(&pi, &root, "worktree", &[("FAKE_WRITE", "b.txt")])).unwrap();
        assert_eq!(job.isolation.as_deref(), Some("worktree"));
        let done = ends(&jobs, &job.id, Duration::from_secs(5));
        assert_eq!(done.state, JobState::Done);
        assert_eq!(fs::read_to_string(root.join("b.txt")).unwrap(), "new\n");
        assert_eq!(git(&root, &["diff", "--cached"]), "", "nothing staged");
        assert_eq!(done.worktree, None);
        assert!(!done.conflict);
        let name = root.file_name().unwrap().to_string_lossy().into_owned();
        assert!(!pi_dir
            .path()
            .canonicalize()
            .unwrap()
            .join(".aria-worktrees")
            .join(format!("{name}-{}", job.id))
            .exists());
    }

    #[test]
    fn the_store_names_the_worktree_before_the_subagent_starts() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_repo_dir, root) = repo();
        let (dir, jobs, _) = jobs();
        let (_pi_dir, pi) = fake_pi();
        let store = dir.path().join("jobs.json");
        let seen: Arc<Mutex<Option<String>>> = Arc::default();
        let (into, at) = (seen.clone(), store.clone());
        let hook: BeforeSpawn = Arc::new(move |_: &Job| {
            *into.lock().unwrap() = Some(fs::read_to_string(&at).unwrap_or_default());
        });
        let start = AgentStart { before_spawn: Some(hook), ..writer(&pi, &root, "worktree", &[]) };
        let job = jobs.start_agent(start).unwrap();
        let stored = seen.lock().unwrap().clone().expect("the hook ran");
        let stored: serde_json::Value = serde_json::from_str(&stored).unwrap();
        let path = job.worktree.clone().expect("a worktree");
        assert_eq!(stored["jobs"][0]["worktree"], serde_json::json!(path), "saved before the spawn");
        ends(&jobs, &job.id, Duration::from_secs(5));
    }

    #[test]
    fn a_stopped_worktree_agent_keeps_its_worktree() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_repo_dir, root) = repo();
        let (_dir, jobs, _) = jobs();
        let (_pi_dir, pi) = fake_pi();
        let env = [("FAKE_WRITE", "b.txt"), ("FAKE_SLEEP", "30")];
        let job = jobs.start_agent(writer(&pi, &root, "worktree", &env)).unwrap();
        assert!(jobs.stop(&job.id));
        let stopped = ends(&jobs, &job.id, Duration::from_secs(5));
        assert_eq!(stopped.state, JobState::Stopped);
        let path = stopped.worktree.expect("the worktree is kept");
        assert!(Path::new(&path).exists());
        assert!(!root.join("b.txt").exists(), "nothing is brought back");
        let (_, by, _) = ended(&jobs.events("s1")[0]);
        assert_eq!(by, EndedBy::Stopped);
    }

    #[test]
    fn closing_a_session_drops_its_queue() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let (pi_dir, pi) = fake_pi();
        let marks = pi_dir.path().join("marks");
        fs::create_dir(&marks).unwrap();
        let env = [("FAKE_SLEEP", "30"), ("FAKE_MARKS", marks.to_str().unwrap())];
        let started: Vec<Job> = (0..4).map(|_| jobs.start_agent(agent(&pi, dir.path(), &env)).unwrap()).collect();
        assert_eq!(started[3].state, JobState::Queued);
        eventually("the three run", Duration::from_secs(3), || fs::read_dir(&marks).unwrap().count() == 3);
        jobs.stop_session("s1");
        assert_eq!(state(&jobs, &started[3].id), JobState::Stopped, "the queued one at once");
        eventually("all stopped", Duration::from_secs(5), || jobs.active() == 0);
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(fs::read_dir(&marks).unwrap().count(), 3, "the queued one never ran");
        assert_eq!(jobs.events("s1").len(), 4);
    }

    #[test]
    fn before_spawn_runs_with_the_job() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let (pi_dir, pi) = fake_pi();
        let marks = pi_dir.path().join("marks");
        fs::create_dir(&marks).unwrap();
        let seen: Arc<Mutex<Option<(Job, usize)>>> = Arc::default();
        let (into, at) = (seen.clone(), marks.clone());
        let hook: BeforeSpawn = Arc::new(move |job: &Job| {
            *into.lock().unwrap() = Some((job.clone(), fs::read_dir(&at).unwrap().count()));
        });
        let start = AgentStart {
            before_spawn: Some(hook),
            ..agent(&pi, dir.path(), &[("FAKE_MARKS", marks.to_str().unwrap())])
        };
        let job = jobs.start_agent(start).unwrap();
        let (hooked, runs_before) = seen.lock().unwrap().clone().expect("the hook ran");
        assert_eq!(hooked.id, job.id);
        assert_eq!(hooked.title, "check");
        assert_eq!(hooked.task.as_deref(), Some("fix it"));
        assert_eq!(runs_before, 0, "before the child started");
        ends(&jobs, &job.id, Duration::from_secs(3));
    }

    #[test]
    fn the_child_does_not_get_the_jobs_socket() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let (pi_dir, pi) = fake_pi();
        let sock = pi_dir.path().join("sock");
        std::env::set_var("ARIA_JOBS_SOCK", "/run/dock.sock");
        let env = [("FAKE_SOCK", sock.to_str().unwrap()), ("ARIA_JOBS_SOCK", "/run/session.sock")];
        let start = AgentStart { unset: vec!["ARIA_JOBS_SOCK".into()], ..agent(&pi, dir.path(), &env) };
        let job = jobs.start_agent(start);
        std::env::remove_var("ARIA_JOBS_SOCK");
        ends(&jobs, &job.unwrap().id, Duration::from_secs(3));
        assert_eq!(fs::read_to_string(&sock).unwrap(), "none");
    }

    #[test]
    fn a_program_that_cannot_start_is_an_error() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, jobs, _) = jobs();
        let missing = dir.path().join("no-pi");
        let refused = jobs.start_agent(agent(&missing, dir.path(), &[]));
        assert!(refused.unwrap_err().contains("Could not start the subagent"));
        assert!(jobs.list(None).is_empty());
    }
}
