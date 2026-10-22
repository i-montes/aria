//! The subagents a session's Pi starts (`parallel`), watched live. Each child `pi --mode json` finds the socket in
//! `PI_WATCH_SOCK` and streams one JSON line per thing it does (the owner's `subagent-watch` extension, child side):
//! `{"t":"prompt"|"text"|"thinking"|"llm_start"|"turn"|"tool"|"result"|"end", "at", …}`. Aria hosts the socket (one per
//! session), keeps the agents and tells the page, at most four times a second. Each agent also keeps its LLM and tool
//! calls as spans for the `/debug` trace (crates/aria/src/trace.rs).

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::BufReader;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MAX_AGENTS: usize = 20;
const MAX_STEPS: usize = 400;
/// The spans kept per agent for the `/debug` trace (crates/aria/src/trace.rs); the oldest finished go first.
const MAX_SPANS: usize = 200;
const MAX_TEXT: usize = 4000;
const MAX_LINE: u64 = 64 * 1024;
const THROTTLE: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, PartialEq)]
enum Status {
    Running,
    Done,
    Failed,
}

#[derive(Debug, Clone)]
struct Agent {
    id: u64,
    name: String,
    task: String,
    status: Status,
    steps: Vec<Value>,
    started_ms: i64,
    /// One word for what it is doing now (the design's status: reading, writing, running…).
    doing: &'static str,
    /// The `parallel` call that named it, when one did.
    call: Option<String>,
    ended_ms: Option<i64>,
    /// Its LLM and tool calls for the trace, with epoch times (`at` from its lines, or when they arrived).
    spans: Vec<Value>,
    llm_count: u32,
    tool_count: u32,
    open_llm: Option<String>,
}

#[derive(Default)]
struct State {
    agents: Vec<Agent>,
    /// Names, tasks and call ids of the latest `parallel` calls, waiting for their children to connect.
    pending: VecDeque<(String, String, Option<String>)>,
    next_id: u64,
    dirty: bool,
}

pub struct Watch {
    path: PathBuf,
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
}

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        text.chars().take(max).collect::<String>() + "…"
    }
}

/// The design's one-word status for a tool the agent runs.
fn doing_for(tool: &str) -> &'static str {
    match tool {
        "read" => "reading",
        "write" | "edit" => "writing",
        "bash" => "running",
        "grep" | "find" | "ls" => "searching",
        _ => "working",
    }
}

/// The first line of a tool's target, short enough for one row.
fn one_line(text: &str) -> String {
    clip(text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or(""), 120)
}

/// A Unix socket path holds at most 104 bytes on macOS (108 on Linux). The usual name is `watch-<session>.sock`; in a
/// folder deep enough to pass the limit, the session id is replaced by a short hash of it.
pub fn socket_path(dir: &Path, session: &str) -> PathBuf {
    const LIMIT: usize = 100;
    let path = dir.join(format!("watch-{session}.sock"));
    if path.as_os_str().len() <= LIMIT {
        return path;
    }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    session.hash(&mut hasher);
    dir.join(format!("w-{:016x}.sock", hasher.finish()))
}

impl Watch {
    /// Listens on `path` (mode 0600, replacing a stale file). `on_change` runs at most every 250 ms while something changed.
    pub fn bind(path: &Path, on_change: Arc<dyn Fn() + Send + Sync>) -> std::io::Result<Watch> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let _ = std::fs::remove_file(path);
        let listener = UnixListener::bind(path)?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        listener.set_nonblocking(true)?;
        let state = Arc::new(Mutex::new(State::default()));
        let stop = Arc::new(AtomicBool::new(false));
        {
            let (state, stop) = (state.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            // BSD (macOS) inherits the listener's non-blocking flag when accepting; Linux does not.
                            // The child's reader below blocks on purpose, so make the accepted stream blocking again.
                            let _ = stream.set_nonblocking(false);
                            let state = state.clone();
                            std::thread::spawn(move || serve(stream, state));
                        }
                        Err(_) => std::thread::sleep(Duration::from_millis(50)),
                    }
                }
            });
        }
        {
            let (state, stop) = (state.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    std::thread::sleep(THROTTLE);
                    let changed = std::mem::take(&mut state.lock().unwrap().dirty);
                    if changed {
                        on_change();
                    }
                }
            });
        }
        Ok(Watch { path: path.to_path_buf(), state, stop })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The arguments of a `parallel` call (and its call id): its tasks name the children that will connect (the
    /// extension's rule, in English), and the trace puts them under that call.
    pub fn expect_tasks(&self, call: Option<&str>, args: &Value) {
        let mut list: Vec<Value> = Vec::new();
        for key in ["tasks", "chain"] {
            if let Some(items) = args[key].as_array() {
                list.extend(items.iter().cloned());
            }
        }
        if let Some(items) = args["team"]["tasks"].as_array() {
            for t in items {
                let mut t = t.clone();
                if t["name"].is_null() {
                    t["name"] = t.get("id").or_else(|| t.get("assignee")).cloned().unwrap_or(Value::Null);
                }
                list.push(t);
            }
        }
        if let Some(task) = args["task"].as_str() {
            list.push(json!({"name": args["agent"].as_str().unwrap_or("subagent"), "task": task}));
        }
        if let (Some(task), Some(models)) = (args["race"]["task"].as_str(), args["race"]["models"].as_array()) {
            for m in models {
                list.push(json!({"name": format!("race {}", m.as_str().unwrap_or("?")), "task": task}));
            }
        }
        let mut state = self.state.lock().unwrap();
        // Appended, not replaced: two `parallel` calls in one turn both name their children.
        let more: Vec<(String, String, Option<String>)> = list
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let name = t["name"]
                    .as_str()
                    .or_else(|| t["agent"].as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("task {}", i + 1));
                let shown = match t["agent"].as_str() {
                    Some(agent) if agent != name => format!("{name} ({agent})"),
                    _ => name,
                };
                (shown, t["task"].as_str().unwrap_or_default().to_string(), call.map(str::to_string))
            })
            .collect();
        state.pending.extend(more);
        while state.pending.len() > MAX_AGENTS {
            state.pending.pop_front();
        }
    }

    /// The agents as the trace reads them (crates/aria/src/trace.rs).
    pub fn trace_agents(&self) -> Vec<crate::trace::TraceAgent> {
        let state = self.state.lock().unwrap();
        state
            .agents
            .iter()
            .map(|a| crate::trace::TraceAgent {
                id: a.id,
                name: a.name.clone(),
                task: a.task.clone(),
                started_ms: a.started_ms,
                ended_ms: a.ended_ms,
                failed: a.status == Status::Failed,
                call: a.call.clone(),
                spans: a.spans.clone(),
            })
            .collect()
    }

    /// The agents as the page's subagents view reads them (see ui/chat/parts/subagents.js and session.js).
    pub fn agents(&self) -> Value {
        let state = self.state.lock().unwrap();
        Value::Array(
            state
                .agents
                .iter()
                .map(|a| {
                    let (status, kind) = match a.status {
                        Status::Running => (a.doing, "active"),
                        Status::Done => ("done", "done"),
                        Status::Failed => ("failed", "done"),
                    };
                    let mut steps = a.steps.clone();
                    if a.status == Status::Running {
                        steps.push(json!({"kind": "live", "text": "working", "since": a.started_ms}));
                    }
                    json!({"id": format!("a{}", a.id), "role": a.name, "task": a.task, "status": status, "kind": kind, "steps": steps})
                })
                .collect(),
        )
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = std::fs::remove_file(&self.path);
    }
}

/// One child: its lines until it closes.
fn serve(stream: UnixStream, state: Arc<Mutex<State>>) {
    let id = {
        let mut s = state.lock().unwrap();
        s.next_id += 1;
        let id = s.next_id;
        s.agents.push(Agent {
            id,
            name: format!("#{id}"),
            task: String::new(),
            status: Status::Running,
            steps: Vec::new(),
            started_ms: now_ms(),
            doing: "starting",
            call: None,
            ended_ms: None,
            spans: Vec::new(),
            llm_count: 0,
            tool_count: 0,
            open_llm: None,
        });
        if s.agents.len() > MAX_AGENTS {
            let extra = s.agents.len() - MAX_AGENTS;
            s.agents.drain(0..extra);
        }
        s.dirty = true;
        id
    };
    let mut reader = BufReader::new(stream);
    loop {
        // One line at a time, at most MAX_LINE bytes: a child that never sends a newline cannot grow the dock's memory.
        let bytes = match crate::lines::read_bounded_line(&mut reader, MAX_LINE) {
            None => break,
            Some(None) => continue,
            Some(Some(bytes)) => bytes,
        };
        let line = String::from_utf8_lossy(&bytes);
        let Ok(message) = serde_json::from_str::<Value>(line.trim_end()) else { continue };
        let mut s = state.lock().unwrap();
        let named = if message["t"] == "prompt" {
            Some(name_for(&mut s.pending, message["text"].as_str().unwrap_or_default()))
        } else {
            None
        };
        let Some(agent) = s.agents.iter_mut().find(|a| a.id == id) else { break };
        if let Some((name, task, call)) = named {
            agent.name = name;
            agent.task = task;
            agent.call = call;
        }
        apply(agent, &message);
        s.dirty = true;
    }
    let mut s = state.lock().unwrap();
    if let Some(agent) = s.agents.iter_mut().find(|a| a.id == id) {
        if agent.status == Status::Running {
            agent.status = if agent.steps.is_empty() { Status::Failed } else { Status::Done };
        }
        if agent.ended_ms.is_none() {
            agent.ended_ms = Some(now_ms());
        }
        let end = agent.ended_ms.unwrap_or_default();
        for span in agent.spans.iter_mut().filter(|s| s["status"] == "running") {
            span["status"] = json!("failed");
            span["end"] = json!(end);
        }
    }
    s.dirty = true;
}

fn push_step(agent: &mut Agent, step: Value) {
    agent.steps.push(step);
    if agent.steps.len() > MAX_STEPS {
        let extra = agent.steps.len() - MAX_STEPS;
        agent.steps.drain(0..extra);
    }
}

/// A span for the trace; over `MAX_SPANS` the oldest finished one goes.
fn push_span(agent: &mut Agent, span: Value) {
    agent.spans.push(span);
    if agent.spans.len() > MAX_SPANS {
        if let Some(at) = agent.spans.iter().position(|s| s["status"] != "running") {
            agent.spans.remove(at);
        }
    }
}

fn span_mut<'a>(agent: &'a mut Agent, id: &str) -> Option<&'a mut Value> {
    agent.spans.iter_mut().find(|s| s["id"] == id)
}

/// What a line adds to the agent's trace spans (`at`: the child's clock, or now for an older extension).
fn trace_line(agent: &mut Agent, m: &Value) {
    use crate::trace::{cut, finish_llm, finish_tool, PREVIEW};
    let at = m["at"].as_i64().unwrap_or_else(now_ms);
    match m["t"].as_str() {
        Some("llm_start") => {
            agent.llm_count += 1;
            let id = format!("llm-{}", agent.llm_count);
            agent.open_llm = Some(id.clone());
            push_span(agent, json!({"id": id, "kind": "llm", "name": "LLM call", "status": "running", "start": at}));
        }
        // An assistant message ended; an older extension sends `turn` with no `llm_start` and no data: nothing to show.
        Some("turn") if m.get("at").is_some() => {
            let id = match agent.open_llm.take() {
                Some(id) => id,
                None => {
                    agent.llm_count += 1;
                    let id = format!("llm-{}", agent.llm_count);
                    push_span(
                        agent,
                        json!({"id": id, "kind": "llm", "name": "LLM call", "status": "running", "start": at}),
                    );
                    id
                }
            };
            let message =
                json!({"model": m["model"], "usage": m["usage"], "stopReason": m["stop"], "errorMessage": m["error"]});
            if let Some(span) = span_mut(agent, &id) {
                finish_llm(span, &message, at);
            }
        }
        Some("tool") => {
            agent.tool_count += 1;
            let id = match m["id"].as_str() {
                Some(id) => format!("tool-{id}"),
                None => format!("tool-{}", agent.tool_count),
            };
            let args = m["args"].as_str().unwrap_or_default();
            let span = json!({
                "id": id, "kind": "tool", "name": m["name"].as_str().unwrap_or("tool"), "status": "running", "start": at,
                "detail": cut(&one_line(args), 80), "input": cut(args, PREVIEW),
            });
            push_span(agent, span);
        }
        Some("result") => {
            let span = match m["id"].as_str() {
                Some(id) => span_mut(agent, &format!("tool-{id}")),
                None => agent.spans.iter_mut().rev().find(|s| s["kind"] == "tool" && s["status"] == "running"),
            };
            if let Some(span) = span {
                finish_tool(span, m["text"].as_str().unwrap_or_default(), m["error"] == json!(true), at);
            }
        }
        Some("end") => agent.ended_ms = Some(at),
        _ => {}
    }
}

fn apply(agent: &mut Agent, m: &Value) {
    trace_line(agent, m);
    let text = |key: &str| m[key].as_str().unwrap_or_default();
    match m["t"].as_str() {
        // The prompt names the agent and its task (the header); it is not a step.
        Some("prompt") => {}
        Some("thinking") => agent.doing = "thinking",
        Some("text") => {
            agent.doing = "answering";
            let delta = text("delta");
            match agent.steps.last_mut() {
                Some(last) if last["kind"] == "msg" => {
                    let joined = format!("{}{}", last["text"].as_str().unwrap_or_default(), delta);
                    last["text"] = json!(clip(&joined, MAX_TEXT));
                }
                _ => push_step(agent, json!({"kind": "msg", "text": delta})),
            }
        }
        // The page names the tool and picks its icon (ui/chat/parts/session.js), in the reader's language.
        Some("tool") => {
            let name = text("name");
            agent.doing = doing_for(name);
            push_step(agent, json!({"kind": "tool", "tool": name, "target": one_line(text("args"))}));
        }
        Some("result") if m["error"] == json!(true) => {
            push_step(agent, json!({"kind": "note", "label": "error", "text": clip(text("text"), 600)}))
        }
        Some("end") => agent.status = Status::Done,
        _ => {}
    }
}

/// Names a child from its prompt: the pending `parallel` task whose text it contains, or the line after "Task:".
fn name_for(
    pending: &mut VecDeque<(String, String, Option<String>)>,
    prompt: &str,
) -> (String, String, Option<String>) {
    if let Some(at) = pending
        .iter()
        .position(|(_, task, _)| !task.is_empty() && prompt.contains(&task.chars().take(160).collect::<String>()))
    {
        return pending.remove(at).unwrap();
    }
    let after = prompt.split_once("Task:").map(|(_, rest)| rest).unwrap_or(prompt);
    let line = after.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("subagent");
    (clip(line, 30), clip(line, 200), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::AtomicUsize;

    fn watch() -> (tempfile::TempDir, Watch, Arc<AtomicUsize>) {
        let dir = tempfile::tempdir().unwrap();
        let changes = Arc::new(AtomicUsize::new(0));
        let counter = changes.clone();
        let w = Watch::bind(
            &dir.path().join("w.sock"),
            Arc::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
        )
        .unwrap();
        (dir, w, changes)
    }

    fn child(w: &Watch, lines: &[Value]) -> UnixStream {
        let mut s = UnixStream::connect(w.path()).unwrap();
        for line in lines {
            writeln!(s, "{line}").unwrap();
        }
        s.flush().unwrap();
        s
    }

    fn eventually(w: &Watch, what: &str, test: impl Fn(&Value) -> bool) -> Value {
        for _ in 0..100 {
            let agents = w.agents();
            if test(&agents) {
                return agents;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("{what}: {}", w.agents());
    }

    #[test]
    fn lines_with_ids_and_times_build_spans() {
        let (_d, w, _) = watch();
        let usage = json!({"input": 10, "output": 4, "cost": {"total": 0.02}});
        let _c = child(
            &w,
            &[
                json!({"t": "prompt", "text": "Task: find X", "at": 900}),
                json!({"t": "llm_start", "at": 1000}),
                json!({"t": "turn", "at": 1500, "model": "m", "usage": usage, "stop": "stop"}),
                json!({"t": "tool", "id": "c1", "name": "bash", "args": "ls -la", "at": 1600}),
                json!({"t": "result", "id": "c1", "text": "out", "error": false, "at": 1900}),
                json!({"t": "end", "at": 2000}),
            ],
        );
        let agents = eventually_traced(&w, |a| a.first().is_some_and(|a| a.ended_ms.is_some()));
        let a = &agents[0];
        assert_eq!(a.ended_ms, Some(2000));
        assert_eq!(a.spans.len(), 2);
        let llm = &a.spans[0];
        assert_eq!(
            (llm["id"].as_str(), llm["start"].as_i64(), llm["end"].as_i64()),
            (Some("llm-1"), Some(1000), Some(1500))
        );
        assert_eq!(llm["detail"], "m");
        assert_eq!(llm["cost"], 0.02);
        assert_eq!(llm["tokensIn"], 10);
        let tool = &a.spans[1];
        assert_eq!(
            (tool["id"].as_str(), tool["start"].as_i64(), tool["end"].as_i64()),
            (Some("tool-c1"), Some(1600), Some(1900))
        );
        assert_eq!(tool["detail"], "ls -la");
        assert_eq!(tool["output"], "out");
        assert_eq!(tool["status"], "done");
    }

    #[test]
    fn old_lines_still_work() {
        let (_d, w, _) = watch();
        let _c = child(
            &w,
            &[json!({"t": "tool", "name": "read", "args": "/a"}), json!({"t": "result", "text": "x", "error": true})],
        );
        let agents = eventually_traced(&w, |a| {
            a.first().is_some_and(|a| a.spans.first().is_some_and(|s| s["status"] == "failed"))
        });
        assert_eq!(agents[0].spans[0]["id"], "tool-1");
        assert!(agents[0].spans[0]["start"].as_i64().unwrap() > 0);
        assert_eq!(agents[0].spans[0]["error"], "x");
    }

    #[test]
    fn a_child_remembers_the_call_that_named_it() {
        let (_d, w, _) = watch();
        w.expect_tasks(Some("c9"), &json!({"tasks": [{"name": "researcher", "task": "find X"}]}));
        let _c = child(&w, &[json!({"t": "prompt", "text": "Task: find X"})]);
        let agents = eventually_traced(&w, |a| a.first().is_some_and(|a| a.name == "researcher"));
        assert_eq!(agents[0].call.as_deref(), Some("c9"));
        assert_eq!(agents[0].task, "find X");
    }

    fn eventually_traced(
        w: &Watch,
        test: impl Fn(&[crate::trace::TraceAgent]) -> bool,
    ) -> Vec<crate::trace::TraceAgent> {
        for _ in 0..100 {
            let agents = w.trace_agents();
            if test(&agents) {
                return agents;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("never: {:?}", w.trace_agents());
    }

    #[test]
    fn a_child_that_connects_is_an_active_agent_and_the_page_is_told() {
        let (_d, w, changes) = watch();
        let _c = child(&w, &[]);
        eventually(&w, "an active agent", |a| a[0]["kind"] == "active");
        std::thread::sleep(Duration::from_millis(400));
        assert!(changes.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn a_socket_path_fits_the_unix_limit_even_in_a_deep_folder() {
        let short = socket_path(Path::new("/run/user/1000/aria"), "s1");
        assert_eq!(short, PathBuf::from("/run/user/1000/aria/watch-s1.sock"), "the usual name when it fits");
        let deep = PathBuf::from(format!("/Users/someone/{}/.aria/run", "x".repeat(40)));
        let session = "0".repeat(36);
        let path = socket_path(&deep, &session);
        assert!(path.as_os_str().len() <= 100, "{}", path.display());
        assert_ne!(path, socket_path(&deep, &"1".repeat(36)), "two sessions never share a socket");
    }

    #[test]
    fn the_socket_is_private_and_removed_with_the_watch() {
        let (_d, w, _) = watch();
        let path = w.path().to_path_buf();
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        drop(w);
        assert!(!path.exists());
    }

    #[test]
    fn the_prompt_names_it_from_the_parallel_task() {
        let (_d, w, _) = watch();
        w.expect_tasks(None, &json!({"tasks": [{"agent": "security", "task": "Review the sandbox policy"}, {"agent": "frontend", "task": "Fix the tabs"}]}));
        let _c = child(&w, &[json!({"t": "prompt", "text": "You are a subagent.\nTask: Fix the tabs\nBe brief."})]);
        let agents = eventually(&w, "named", |a| a[0]["role"] == "frontend");
        assert_eq!(agents[0]["task"], "Fix the tabs");
    }

    #[test]
    fn without_a_matching_task_the_line_after_task_names_it() {
        let (_d, w, _) = watch();
        let _c = child(&w, &[json!({"t": "prompt", "text": "Intro\nTask: Count the files in src\nmore"})]);
        eventually(&w, "named from the prompt", |a| a[0]["role"] == "Count the files in src");
    }

    #[test]
    fn text_deltas_join_into_one_message_and_tools_and_errors_are_steps() {
        let (_d, w, _) = watch();
        let _c = child(
            &w,
            &[
                json!({"t": "prompt", "text": "Task: x"}),
                json!({"t": "text", "delta": "Look"}),
                json!({"t": "text", "delta": "ing."}),
                json!({"t": "tool", "name": "read", "args": "src/main.rs"}),
                json!({"t": "result", "text": "boom", "error": true}),
            ],
        );
        // Wait for the error note itself: the live step a running agent shows can make three steps before it arrives.
        let agents = eventually(&w, "steps", |a| {
            a[0]["steps"].as_array().is_some_and(|s| s.iter().any(|step| step["kind"] == "note"))
        });
        let steps = agents[0]["steps"].as_array().unwrap();
        assert_eq!(
            steps[0],
            json!({"kind": "msg", "text": "Looking."}),
            "the prompt is not a step: the task is in the header"
        );
        assert_eq!(steps[1], json!({"kind": "tool", "tool": "read", "target": "src/main.rs"}));
        assert_eq!(steps[2], json!({"kind": "note", "label": "error", "text": "boom"}));
        assert_eq!(steps.last().unwrap()["kind"], "live", "a running agent shows what it is doing now");
    }

    #[test]
    fn a_long_or_multiline_target_is_one_short_line() {
        let (_d, w, _) = watch();
        let code = format!("const a = 1;\n{}", "x".repeat(300));
        let _c = child(&w, &[json!({"t": "tool", "name": "codemode", "args": code})]);
        // Wait for the tool step itself: a running agent may show its "live" step first.
        let agents = eventually(&w, "the tool", |a| {
            a[0]["steps"].as_array().is_some_and(|s| s.iter().any(|step| step["kind"] == "tool"))
        });
        let steps = agents[0]["steps"].as_array().unwrap();
        let tool = steps.iter().find(|step| step["kind"] == "tool").unwrap();
        assert_eq!(tool["target"], "const a = 1;");
    }

    #[test]
    fn the_status_is_one_word_for_what_it_does_now() {
        let (_d, w, _) = watch();
        let _a = child(&w, &[]);
        eventually(&w, "starting", |a| a[0]["status"] == "starting");
        let _b = child(&w, &[json!({"t": "tool", "name": "read", "args": "a"})]);
        eventually(&w, "reading", |a| a.as_array().unwrap().iter().any(|x| x["status"] == "reading"));
        let _c = child(&w, &[json!({"t": "tool", "name": "edit", "args": "a"})]);
        eventually(&w, "writing", |a| a.as_array().unwrap().iter().any(|x| x["status"] == "writing"));
        let _d2 = child(&w, &[json!({"t": "tool", "name": "bash", "args": "ls"})]);
        eventually(&w, "running", |a| a.as_array().unwrap().iter().any(|x| x["status"] == "running"));
    }

    #[test]
    fn end_marks_it_done() {
        let (_d, w, _) = watch();
        let _c = child(&w, &[json!({"t": "prompt", "text": "Task: x"}), json!({"t": "end"})]);
        let agents = eventually(&w, "done", |a| a[0]["kind"] == "done");
        assert_eq!(agents[0]["status"], "done");
        assert!(agents[0]["steps"].as_array().unwrap().iter().all(|s| s["kind"] != "live"));
    }

    #[test]
    fn a_child_that_closes_without_end_is_done_with_work_and_failed_without() {
        let (_d, w, _) = watch();
        drop(child(&w, &[json!({"t": "prompt", "text": "Task: a"}), json!({"t": "text", "delta": "partial"})]));
        drop(child(&w, &[]));
        let agents = eventually(&w, "two closed", |a| {
            a.as_array().is_some_and(|l| l.len() == 2 && l.iter().all(|x| x["kind"] == "done"))
        });
        // The two connections are served by their own threads, so either may be listed first: tell them by their work.
        let list = agents.as_array().unwrap();
        let worked = |a: &&Value| a["steps"].as_array().is_some_and(|s| s.iter().any(|step| step["kind"] == "msg"));
        assert_eq!(list.iter().find(worked).unwrap()["status"], "done");
        assert_eq!(list.iter().find(|a| !worked(a)).unwrap()["status"], "failed");
    }

    #[test]
    fn two_parallel_calls_in_a_row_both_name_their_children() {
        let (_d, w, _) = watch();
        w.expect_tasks(None, &json!({"tasks": [{"agent": "security", "task": "first job"}]}));
        w.expect_tasks(None, &json!({"tasks": [{"agent": "frontend", "task": "second job"}]}));
        let _a = child(&w, &[json!({"t": "prompt", "text": "Task: first job"})]);
        eventually(&w, "the first call's child keeps its name", |a| a[0]["role"] == "security");
    }

    #[test]
    fn a_child_that_never_sends_a_newline_is_cut_and_the_next_line_still_counts() {
        let (_d, w, _) = watch();
        let mut c = child(&w, &[]);
        c.write_all(&vec![b'x'; 200 * 1024]).unwrap();
        writeln!(c).unwrap();
        writeln!(c, "{}", json!({"t": "tool", "name": "ls", "args": "."})).unwrap();
        eventually(&w, "the line after the long one", |a| {
            a[0]["steps"].as_array().is_some_and(|s| s.iter().any(|x| x["tool"] == "ls"))
        });
    }

    #[test]
    fn more_than_twenty_agents_keep_the_latest_twenty_and_garbage_is_skipped() {
        let (_d, w, _) = watch();
        let mut streams = Vec::new();
        for _ in 0..22 {
            streams.push(child(&w, &[]));
        }
        let mut bad = child(&w, &[]);
        writeln!(bad, "not json").unwrap();
        writeln!(bad, "{}", json!({"t": "tool", "name": "bash", "args": "ls"})).unwrap();
        eventually(&w, "twenty", |a| a.as_array().is_some_and(|l| l.len() == 20));
        eventually(&w, "the good line after the garbage", |a| {
            a.as_array().unwrap().iter().any(|x| x["status"] == "running")
        });
    }
}
