//! A session's conversation as data: the dock's messages for one session (`chatEvent`, `chatState`, `chatQuota`,
//! `chatDialog`…) go in, the entries the view draws come out. A port of `ui/chat/state.js` (same meaning), without what
//! only the page needs (attachments' chips, paging back, the router's line, the answers card).

use aria_core::bridge::DockMessage;
use aria_core::quota::Window;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const RESULT_LIMIT: usize = 4000;

/// A text the client says in the assistant's language (`texts::t`), or one it shows as it came.
#[derive(Debug, Clone, PartialEq)]
pub enum Text {
    Key { key: String, vars: Value },
    Raw(String),
}

impl Text {
    fn key(key: &str, vars: Value) -> Text {
        Text::Key { key: key.to_string(), vars }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tone {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Status {
    Running,
    Done,
    Error,
}

/// One step of a turn: a tool call, or (`think`) a piece of reasoning in `output`.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub id: String,
    pub name: String,
    pub summary: String,
    pub status: Status,
    pub output: String,
    pub think: bool,
    pub added: u32,
    pub removed: u32,
    /// What a command does, in the model's words (the sandbox asks for it): the step's title, as in the dock.
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Entry {
    User {
        text: String,
    },
    Assistant {
        text: String,
        thinking: String,
        done: bool,
    },
    Tools {
        calls: Vec<Call>,
    },
    Notice {
        tone: Tone,
        text: Text,
    },
    Note {
        text: Text,
    },
    Divider {
        text: Text,
    },
    /// What an instruction changed in the folder.
    Changes {
        files: u32,
        added: u32,
        removed: u32,
    },
}

/// A task of the model's list (`tasks_update`).
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    pub id: String,
    pub title: String,
    /// `todo`, `active` or `done`.
    pub status: String,
    pub agent: Option<String>,
}

/// The tasks in a `tasks_update` result's details.
fn tasks_of(details: &Value) -> Vec<Task> {
    details["tasks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| {
            Some(Task {
                id: t["id"].as_str()?.to_string(),
                title: t["title"].as_str()?.to_string(),
                status: t["status"].as_str().unwrap_or("todo").to_string(),
                agent: t["agent"].as_str().map(str::to_string),
            })
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dialog {
    pub id: String,
    pub method: String,
    pub request: Value,
}

#[derive(Debug, Default)]
pub struct Session {
    pub entries: Vec<Entry>,
    pub running: bool,
    pub compacting: bool,
    pub engine: String,
    pub model: Option<String>,
    pub thinking: Option<String>,
    pub context: Option<f64>,
    pub workdir: String,
    pub problem: Option<String>,
    pub quota: Vec<Window>,
    pub dialogs: Vec<Dialog>,
    pub queue: Vec<String>,
    pub agents: Value,
    /// What the run is doing (`ui/chat/parts/working.js`'s phases): `working`, `thinking`, `reading files`…
    pub phase: String,
    /// The model's task list, from its last `tasks_update`.
    pub tasks: Vec<Task>,
    /// What the extensions publish (`aria-context`, `aria-verify`…), by key.
    pub statuses: std::collections::HashMap<String, String>,
    pub loaded: bool,
    /// A refused message's text, back to the editor.
    pub give_back: Option<String>,
    turn_from: usize,
    stopped_by_watchdog: bool,
    texts: BTreeMap<u64, String>,
    thoughts: BTreeMap<u64, String>,
    /// Messages this client showed when it sent them: their echo is not shown again.
    own: Vec<String>,
    steps: u64,
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => {
            blocks.iter().filter(|b| b["type"] == "text").filter_map(|b| b["text"].as_str()).collect()
        }
        _ => String::new(),
    }
}

fn thinking_of(content: &Value) -> String {
    content.as_array().map_or(String::new(), |blocks| {
        blocks.iter().filter(|b| b["type"] == "thinking").filter_map(|b| b["thinking"].as_str()).collect()
    })
}

/// The person's text without the dock's attachment lines (`[attached: …]`).
fn without_attached(text: &str) -> String {
    text.lines().filter(|line| !line.starts_with("[attached")).collect::<Vec<_>>().join("\n").trim_end().to_string()
}

/// What the run is doing while this tool runs (as `phaseFor` in `ui/chat/state.js`).
pub fn phase_for(name: &str) -> &'static str {
    if name.contains("engram") || name.contains("mem_") {
        "searching memory"
    } else if name.contains("tavily") {
        "searching the web"
    } else if name.contains("notion") {
        "checking Notion"
    } else if name.starts_with("parallel") || name.starts_with("subagent") {
        "delegating"
    } else {
        match name {
            "bash" => "running command",
            "read" | "grep" | "find" | "ls" => "reading files",
            "write" | "edit" => "writing files",
            _ => "working",
        }
    }
}

/// A short label for a call: the command's first line, the path, or the tool's name.
pub fn tool_summary(name: &str, args: &Value) -> String {
    let field = |key: &str| args[key].as_str().filter(|s| !s.is_empty());
    match name {
        "bash" => field("command").map(|c| c.lines().next().unwrap_or_default().to_string()),
        "read" | "write" | "edit" => field("path").or_else(|| field("file_path")).map(str::to_string),
        "grep" => field("pattern").map(str::to_string),
        "find" => field("pattern").or_else(|| field("path")).map(str::to_string),
        "ls" => Some(field("path").unwrap_or(".").to_string()),
        _ => None,
    }
    .unwrap_or_else(|| name.to_string())
}

fn count_lines(text: &str) -> u32 {
    text.lines().count() as u32
}

/// Lines a call adds and removes: an edit's new and old texts, a write's content.
fn changed_lines(name: &str, args: &Value) -> (u32, u32) {
    match name {
        "edit" => {
            let edits = args["edits"].as_array().cloned().unwrap_or_else(|| vec![args.clone()]);
            edits.iter().fold((0, 0), |(added, removed), edit| {
                (
                    added + count_lines(edit["newText"].as_str().unwrap_or_default()),
                    removed + count_lines(edit["oldText"].as_str().unwrap_or_default()),
                )
            })
        }
        "write" => (count_lines(args["content"].as_str().unwrap_or_default()), 0),
        _ => (0, 0),
    }
}

fn new_call(id: &str, name: &str, args: &Value, status: Status) -> Call {
    let (added, removed) = changed_lines(name, args);
    Call {
        id: id.to_string(),
        name: name.to_string(),
        summary: tool_summary(name, args),
        status,
        output: String::new(),
        think: false,
        added,
        removed,
        reason: args["reason"].as_str().unwrap_or_default().trim().to_string(),
    }
}

/// The group a turn's steps go into: the last `Tools` entry after the last user message and the last thing said
/// (and at or after `from`).
fn turn_group(entries: &[Entry], from: usize) -> Option<usize> {
    for i in (from..entries.len()).rev() {
        match &entries[i] {
            Entry::User { .. } => return None,
            Entry::Assistant { text, .. } if !text.is_empty() => return None,
            Entry::Tools { .. } => return Some(i),
            _ => {}
        }
    }
    None
}

fn add_steps(entries: &mut Vec<Entry>, steps: Vec<Call>, from: usize) {
    if steps.is_empty() {
        return;
    }
    match turn_group(entries, from) {
        Some(at) => {
            if let Entry::Tools { calls } = &mut entries[at] {
                calls.extend(steps);
            }
        }
        None => entries.push(Entry::Tools { calls: steps }),
    }
}

fn bang_entry(command: &str, output: &str, exit_code: i64) -> Entry {
    let status = if exit_code == 0 { Status::Done } else { Status::Error };
    let mut call = new_call("bang", "bash", &json!({"command": command}), status);
    call.summary = format!("!{command}");
    call.output = output.chars().take(RESULT_LIMIT).collect();
    Entry::Tools { calls: vec![call] }
}

impl Session {
    fn push(&mut self, entry: Entry) {
        self.entries.push(entry);
    }

    fn think_steps(&mut self, thinking: &str) -> Vec<Call> {
        let text = thinking.trim();
        if text.is_empty() {
            return Vec::new();
        }
        self.steps += 1;
        vec![Call {
            id: format!("think-{}", self.steps),
            name: "thinking".into(),
            summary: String::new(),
            status: Status::Done,
            output: text.to_string(),
            think: true,
            added: 0,
            removed: 0,
            reason: String::new(),
        }]
    }

    /// A message this client sent while nothing ran: shown at once, its echo skipped.
    pub fn add_user(&mut self, text: &str) {
        self.own.push(text.to_string());
        self.push(Entry::User { text: text.to_string() });
    }

    /// A `!command` the person ran (Pi's `bash`) with its output.
    pub fn add_bang(&mut self, command: &str, output: &str, exit_code: i64) {
        self.push(bang_entry(command, output, exit_code));
    }

    pub fn apply(&mut self, message: &DockMessage) {
        match message {
            DockMessage::ChatEvent { event } => self.event(event),
            DockMessage::ChatState { workdir, running, compacting, model, thinking, context, problem } => {
                let changed = !self.entries.is_empty()
                    && self.model.is_some()
                    && model.is_some()
                    && self.model.as_deref() != model.as_deref();
                self.workdir = workdir.clone();
                self.running = *running || *compacting;
                self.compacting = *compacting;
                self.model = model.clone();
                self.thinking = thinking.clone();
                self.context = *context;
                self.problem = problem.clone();
                if changed {
                    let short = model.as_deref().unwrap_or_default().rsplit('/').next().unwrap_or_default();
                    self.push(Entry::Note { text: Text::Raw(short.to_string()) });
                }
            }
            DockMessage::ChatQuota { windows } => self.quota = windows.clone(),
            DockMessage::ChatAgents { agents } => self.agents = agents.clone(),
            DockMessage::ChatEngine { state } => {
                self.engine = state.clone();
                if state == "stopped" || state == "exited" {
                    self.running = false;
                    self.loaded = true;
                }
            }
            DockMessage::ChatDialog { id, method, request, .. } => {
                self.dialogs.retain(|d| &d.id != id);
                self.dialogs.push(Dialog { id: id.clone(), method: method.clone(), request: request.clone() });
            }
            DockMessage::ChatDialogClosed { id } => self.dialogs.retain(|d| &d.id != id),
            _ => {}
        }
    }

    fn event(&mut self, event: &Value) {
        match event["type"].as_str().unwrap_or_default() {
            "agent_start" => {
                self.running = true;
                self.phase = "working".into();
                self.turn_from = self.entries.len();
                self.stopped_by_watchdog = false;
            }
            "agent_settled" => {
                self.running = false;
                if event["aborted"] == json!(true) && !self.stopped_by_watchdog {
                    self.push(Entry::Note { text: Text::key("chat.stoppedByYou", json!({})) });
                }
                self.stopped_by_watchdog = false;
            }
            "scheduled_run" => self.push(Entry::User { text: event["text"].as_str().unwrap_or_default().to_string() }),
            "message_start" => match event["message"]["role"].as_str() {
                Some("assistant") => {
                    self.texts.clear();
                    self.thoughts.clear();
                    self.push(Entry::Assistant { text: String::new(), thinking: String::new(), done: false });
                }
                Some("user") => {
                    let text = without_attached(&text_of(&event["message"]["content"]));
                    if let Some(at) = self.own.iter().position(|own| *own == text) {
                        self.own.remove(at);
                        return;
                    }
                    if let Some(at) = self.queue.iter().position(|q| *q == text) {
                        self.queue.remove(at);
                    }
                    self.push(Entry::User { text });
                }
                _ => {}
            },
            "queue_update" => {
                let texts = |key: &str| -> Vec<String> {
                    event[key].as_array().map_or(Vec::new(), |list| {
                        list.iter().filter_map(Value::as_str).map(without_attached).collect()
                    })
                };
                self.queue = [texts("steering"), texts("followUp")].concat();
            }
            "message_update" => {
                let kind = event["assistantMessageEvent"]["type"].as_str().unwrap_or_default();
                if kind.starts_with("thinking") {
                    self.phase = "thinking".into();
                } else if kind == "text_delta" && self.phase == "thinking" {
                    self.phase = "working".into();
                }
                self.stream(&event["assistantMessageEvent"]);
            }
            "message_end" if event["message"]["role"] == "assistant" => self.end_assistant(&event["message"]),
            // The task list is drawn above the editor, never as a step of the conversation.
            "tool_execution_start" if event["toolName"] == "tasks_update" => {}
            "tool_execution_end" if event["toolName"] == "tasks_update" => {
                if event["isError"] != json!(true) {
                    self.tasks = tasks_of(&event["result"]["details"]);
                }
            }
            "tool_execution_start" => {
                self.phase = phase_for(event["toolName"].as_str().unwrap_or_default()).into();
                let call = new_call(
                    event["toolCallId"].as_str().unwrap_or_default(),
                    event["toolName"].as_str().unwrap_or_default(),
                    &event["args"],
                    Status::Running,
                );
                let from = self.turn_from;
                add_steps(&mut self.entries, vec![call], from);
            }
            "tool_execution_end" => {
                let id = event["toolCallId"].as_str().unwrap_or_default();
                let output: String = text_of(&event["result"]["content"]).chars().take(RESULT_LIMIT).collect();
                let status = if event["isError"] == json!(true) { Status::Error } else { Status::Done };
                for entry in &mut self.entries {
                    if let Entry::Tools { calls } = entry {
                        for call in calls.iter_mut().filter(|c| c.id == id) {
                            call.status = status;
                            call.output = output.clone();
                        }
                    }
                }
            }
            "auto_retry_start" => self.push(Entry::Notice {
                tone: Tone::Warn,
                text: Text::key("chat.retry", json!({"attempt": event["attempt"], "max": event["maxAttempts"]})),
            }),
            "compaction_start" => {
                self.phase = "compacting context".into();
                if !self.running {
                    self.running = true;
                    self.compacting = true;
                }
            }
            "compaction_end" => {
                if self.compacting {
                    self.running = false;
                    self.compacting = false;
                }
                if event["aborted"] != json!(true) && !event["result"].is_null() {
                    self.push(Entry::Divider { text: Text::key("chat.compacted", json!({})) });
                }
            }
            "engine_exited" => {
                self.running = false;
                self.push(Entry::Notice { tone: Tone::Error, text: Text::key("chat.engine.exited", json!({})) });
            }
            "command_failed" => {
                if let Some(text) = event["text"].as_str() {
                    // Never reached Pi: its echo will not come, and the same text sent later from elsewhere shows.
                    if let Some(at) = self.own.iter().position(|own| own == text) {
                        self.own.remove(at);
                    }
                    if let Some(at) = self.queue.iter().position(|q| q == text) {
                        self.queue.remove(at);
                    } else if let Some(at) =
                        self.entries.iter().rposition(|e| matches!(e, Entry::User { text: t } if t == text))
                    {
                        self.entries.remove(at);
                    }
                    self.give_back = Some(text.to_string());
                }
                let said = event["error"].as_str().or(event["command"].as_str()).unwrap_or_default();
                self.push(Entry::Notice { tone: Tone::Error, text: Text::Raw(said.to_string()) });
            }
            "ui_notice" => {
                let request = &event["request"];
                if request["method"] == "setStatus" {
                    let key = request["statusKey"].as_str().unwrap_or_default().to_string();
                    match request["statusText"].as_str() {
                        Some(text) => self.statuses.insert(key, text.to_string()),
                        None => self.statuses.remove(&key),
                    };
                    return;
                }
                if request["method"] != "notify" {
                    return;
                }
                if request["key"] == "chat.watchdog.stopped" {
                    self.stopped_by_watchdog = true;
                }
                self.push(notice_entry(request));
            }
            "history" => {
                self.loaded = true;
                let past = self.read_history(&event["messages"]);
                // A notice shown live before the history came is in it too: once.
                self.entries
                    .retain(|e| !matches!(e, Entry::Notice { .. } | Entry::Changes { .. }) || !past.contains(e));
                if self.entries.is_empty() {
                    self.entries = past;
                } else if let Some(first) = self.entries.iter().find_map(|e| match e {
                    Entry::User { text } => Some(text.clone()),
                    _ => None,
                }) {
                    let at = past.iter().rposition(|e| matches!(e, Entry::User { text } if *text == first));
                    let mut merged: Vec<Entry> = past[..at.unwrap_or(past.len())].to_vec();
                    merged.append(&mut self.entries);
                    self.entries = merged;
                } else if !self.running {
                    let mut merged = past;
                    merged.append(&mut self.entries);
                    self.entries = merged;
                }
                self.turn_from = self.entries.len();
            }
            "changes" => self.push(changes_entry(event)),
            "session_reset" => {
                self.entries.clear();
                self.queue.clear();
                self.dialogs.clear();
            }
            _ => {}
        }
    }

    fn stream(&mut self, update: &Value) {
        let Some(Entry::Assistant { done: false, .. }) = self.entries.last() else { return };
        let kind = update["type"].as_str().unwrap_or_default();
        let blocks = if kind.starts_with("thinking") {
            &mut self.thoughts
        } else if kind.starts_with("text") {
            &mut self.texts
        } else {
            return;
        };
        let index = update["contentIndex"].as_u64().unwrap_or(0);
        match kind.split('_').nth(1) {
            Some("delta") => blocks.entry(index).or_default().push_str(update["delta"].as_str().unwrap_or_default()),
            Some("start") => {
                blocks.entry(index).or_default();
            }
            Some("end") => {
                if let Some(content) = update["content"].as_str() {
                    blocks.insert(index, content.to_string());
                }
            }
            _ => return,
        }
        let (text, thinking) = (self.texts.values().cloned().collect(), self.thoughts.values().cloned().collect());
        if let Some(last) = self.entries.last_mut() {
            *last = Entry::Assistant { text, thinking, done: false };
        }
    }

    fn end_assistant(&mut self, message: &Value) {
        let text = text_of(&message["content"]);
        let thinking = thinking_of(&message["content"]);
        if matches!(self.entries.last(), Some(Entry::Assistant { done: false, .. })) {
            self.entries.pop();
        }
        let steps = self.think_steps(&thinking);
        let from = self.turn_from;
        add_steps(&mut self.entries, steps, from);
        if !text.is_empty() {
            self.push(Entry::Assistant { text, thinking: String::new(), done: true });
        }
    }

    fn read_history(&mut self, messages: &Value) -> Vec<Entry> {
        let mut entries = Vec::new();
        for message in messages.as_array().into_iter().flatten() {
            match message["role"].as_str() {
                Some("user") => entries.push(Entry::User { text: without_attached(&text_of(&message["content"])) }),
                Some("assistant") => {
                    let text = text_of(&message["content"]);
                    let steps = self.think_steps(&thinking_of(&message["content"]));
                    add_steps(&mut entries, steps, 0);
                    if !text.is_empty() {
                        entries.push(Entry::Assistant { text, thinking: String::new(), done: true });
                    }
                    let calls: Vec<Call> = message["content"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|b| b["type"] == "toolCall" && b["name"] != "tasks_update")
                        .map(|b| {
                            new_call(
                                b["id"].as_str().unwrap_or_default(),
                                b["name"].as_str().unwrap_or_default(),
                                &b["arguments"],
                                Status::Done,
                            )
                        })
                        .collect();
                    add_steps(&mut entries, calls, 0);
                }
                // What the chat showed that Pi does not keep (a notice, an instruction's changes), kept by the dock.
                Some("ariaNotice") => match message["event"]["type"].as_str() {
                    Some("ui_notice") => entries.push(notice_entry(&message["event"]["request"])),
                    Some("changes") => entries.push(changes_entry(&message["event"])),
                    _ => {}
                },
                // A command the person ran (`!pwd`, Pi's `bash`).
                Some("bashExecution") => entries.push(bang_entry(
                    message["command"].as_str().unwrap_or_default(),
                    message["output"].as_str().unwrap_or_default(),
                    message["exitCode"].as_i64().unwrap_or(0),
                )),
                Some("compactionSummary") => {
                    entries.push(Entry::Divider { text: Text::key("chat.compacted", json!({})) })
                }
                Some("toolResult") if message["toolName"] == "tasks_update" => {
                    if message["isError"] != json!(true) {
                        self.tasks = tasks_of(&message["details"]);
                    }
                }
                Some("toolResult") => {
                    let id = message["toolCallId"].as_str().unwrap_or_default();
                    let output: String = text_of(&message["content"]).chars().take(RESULT_LIMIT).collect();
                    let status = if message["isError"] == json!(true) { Status::Error } else { Status::Done };
                    for entry in entries.iter_mut().rev() {
                        if let Entry::Tools { calls } = entry {
                            if let Some(call) = calls.iter_mut().find(|c| c.id == id) {
                                call.status = status;
                                call.output = output.clone();
                                break;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        entries
    }
}

/// A `notify` request as the chat shows it (the dock's own name a key, in the person's language).
fn notice_entry(request: &Value) -> Entry {
    let tone = match request["notifyType"].as_str() {
        Some("warning") => Tone::Warn,
        Some("error") => Tone::Error,
        _ => Tone::Info,
    };
    let text = match request["key"].as_str() {
        Some(key) => {
            let key = if key == "chat.watchdog.stopped" {
                let many = request["vars"]["minutes"] != json!(1);
                format!("{key}.{}", if many { "many" } else { "one" })
            } else {
                key.to_string()
            };
            Text::Key { key, vars: request["vars"].clone() }
        }
        None => Text::Raw(request["message"].as_str().unwrap_or_default().to_string()),
    };
    Entry::Notice { tone, text }
}

/// An instruction's `changes` event as the chat shows it.
fn changes_entry(event: &Value) -> Entry {
    let n = |key: &str| event[key].as_u64().unwrap_or(0) as u32;
    Entry::Changes { files: n("files"), added: n("added"), removed: n("removed") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::bridge::DockMessage;
    use serde_json::json;

    fn feed(session: &mut Session, lines: &[serde_json::Value]) {
        for line in lines {
            session.apply(&serde_json::from_value::<DockMessage>(line.clone()).unwrap());
        }
    }

    fn event(e: serde_json::Value) -> serde_json::Value {
        json!({"type": "chatEvent", "event": e})
    }

    fn plain_answer() -> Vec<serde_json::Value> {
        vec![
            event(json!({"type": "agent_start"})),
            event(json!({"type": "message_start", "message": {"role": "assistant", "content": []}})),
            event(
                json!({"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "contentIndex": 0, "delta": "Let me "}}),
            ),
            event(
                json!({"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "contentIndex": 0, "delta": "think"}}),
            ),
            event(
                json!({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "contentIndex": 1, "delta": "Hel"}}),
            ),
            event(
                json!({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "contentIndex": 1, "delta": "lo **there**"}}),
            ),
            event(json!({"type": "message_end", "message": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Let me think"}, {"type": "text", "text": "Hello **there**"}]}})),
            event(json!({"type": "agent_settled"})),
        ]
    }

    fn turn_with_tools() -> Vec<serde_json::Value> {
        vec![
            event(json!({"type": "agent_start"})),
            event(json!({"type": "message_start", "message": {"role": "user", "content": "fix the coupon"}})),
            event(json!({"type": "message_start", "message": {"role": "assistant", "content": []}})),
            event(json!({"type": "message_end", "message": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Read first."},
                {"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "src/coupon.ts"}}]}})),
            event(
                json!({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "read", "args": {"path": "src/coupon.ts"}}),
            ),
            event(json!({"type": "tool_execution_end", "toolCallId": "c1", "toolName": "read", "isError": false,
                "result": {"content": [{"type": "text", "text": "export const x = 1"}]}})),
            event(json!({"type": "tool_execution_start", "toolCallId": "c2", "toolName": "edit",
                "args": {"path": "src/coupon.ts", "edits": [{"oldText": "a\nb", "newText": "a\nb\nc\nd"}]}})),
            event(
                json!({"type": "tool_execution_end", "toolCallId": "c2", "toolName": "edit", "isError": false, "result": {"content": []}}),
            ),
            event(
                json!({"type": "tool_execution_start", "toolCallId": "c3", "toolName": "bash", "args": {"command": "bun test coupon"}}),
            ),
            event(json!({"type": "tool_execution_end", "toolCallId": "c3", "toolName": "bash", "isError": false,
                "result": {"content": [{"type": "text", "text": " 43 pass\n 0 fail\n"}]}})),
            event(json!({"type": "message_start", "message": {"role": "assistant", "content": []}})),
            event(
                json!({"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}}),
            ),
            event(json!({"type": "agent_settled"})),
        ]
    }

    #[test]
    fn a_plain_answer_streams_into_one_assistant_entry() {
        let mut s = Session::default();
        feed(&mut s, &plain_answer()[..6]);
        assert!(s.running);
        match s.entries.last().unwrap() {
            Entry::Assistant { text, thinking, done } => {
                assert_eq!((text.as_str(), thinking.as_str(), *done), ("Hello **there**", "Let me think", false));
            }
            other => panic!("{other:?}"),
        }
        feed(&mut s, &plain_answer()[6..]);
        assert!(!s.running);
        // The reasoning becomes a step of the turn, the text the answer.
        assert!(matches!(&s.entries[0], Entry::Tools { calls } if calls[0].think && calls[0].output == "Let me think"));
        assert!(matches!(&s.entries[1], Entry::Assistant { text, done: true, .. } if text == "Hello **there**"));
        assert_eq!(s.entries.len(), 2);
    }

    #[test]
    fn a_turn_with_tools_keeps_calls_in_order_with_their_output() {
        let mut s = Session::default();
        feed(&mut s, &turn_with_tools());
        assert!(matches!(&s.entries[0], Entry::User { text } if text == "fix the coupon"), "{:?}", s.entries);
        let Entry::Tools { calls } = &s.entries[1] else { panic!("{:?}", s.entries) };
        let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["thinking", "read", "edit", "bash"]);
        assert_eq!(calls[1].summary, "src/coupon.ts");
        assert_eq!(calls[1].output, "export const x = 1");
        assert_eq!(calls[1].status, Status::Done);
        assert_eq!((calls[2].added, calls[2].removed), (4, 2));
        assert_eq!(calls[3].summary, "bun test coupon");
        assert!(matches!(&s.entries[2], Entry::Assistant { text, .. } if text == "Done."));
    }

    #[test]
    fn an_own_message_is_not_shown_twice() {
        let mut s = Session::default();
        s.add_user("fix the coupon");
        feed(&mut s, &turn_with_tools()[..2]);
        let users = s.entries.iter().filter(|e| matches!(e, Entry::User { .. })).count();
        assert_eq!(users, 1);
    }

    #[test]
    fn history_goes_before_notices_that_came_first() {
        let mut s = Session::default();
        s.entries.push(Entry::Notice { tone: Tone::Warn, text: Text::Raw("folder gone".into()) });
        feed(
            &mut s,
            &[event(json!({"type": "history", "messages": [
                {"role": "user", "content": "fix it"},
                {"role": "assistant", "content": [{"type": "text", "text": "Reading."},
                    {"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "src/a.ts"}}]},
                {"role": "toolResult", "toolCallId": "c1", "content": [{"type": "text", "text": "body"}], "isError": false},
                {"role": "assistant", "content": [{"type": "text", "text": "Fixed."}]},
            ]}))],
        );
        assert!(s.loaded);
        assert_eq!(s.entries.len(), 5, "{:?}", s.entries);
        assert!(matches!(&s.entries[4], Entry::Notice { .. }), "the notice that came first stays, after the past");
        assert!(matches!(&s.entries[0], Entry::User { text } if text == "fix it"));
        assert!(matches!(&s.entries[2], Entry::Tools { calls } if calls[0].output == "body"));
        assert!(matches!(&s.entries[3], Entry::Assistant { text, .. } if text == "Fixed."));
    }

    #[test]
    fn notices_and_changes_come_back_in_place() {
        let mut s = Session::default();
        feed(
            &mut s,
            &[event(json!({"type": "history", "messages": [
                {"role": "user", "content": "fix it", "timestamp": 1},
                {"role": "ariaNotice", "event": {"type": "changes", "files": 2, "added": 5, "removed": 1}, "timestamp": 2},
                {"role": "ariaNotice", "event": {"type": "ui_notice", "request": {"method": "notify", "message": "Jev: off", "notifyType": "info"}}, "timestamp": 3},
                {"role": "user", "content": "next", "timestamp": 4},
            ]}))],
        );
        assert!(matches!(&s.entries[1], Entry::Changes { files: 2, added: 5, removed: 1 }), "{:?}", s.entries);
        assert!(
            matches!(&s.entries[2], Entry::Notice { tone: Tone::Info, text: Text::Raw(t) } if t == "Jev: off"),
            "{:?}",
            s.entries
        );
        assert!(matches!(&s.entries[3], Entry::User { text } if text == "next"));
    }

    #[test]
    fn a_notice_shown_before_the_history_is_not_shown_twice() {
        let mut s = Session::default();
        let notice = json!({"type": "ui_notice", "request": {"method": "notify", "message": "folder gone", "notifyType": "warning"}});
        feed(&mut s, &[event(notice.clone())]);
        feed(
            &mut s,
            &[event(json!({"type": "history", "messages": [
                {"role": "user", "content": "fix it", "timestamp": 1},
                {"role": "ariaNotice", "event": notice, "timestamp": 2},
            ]}))],
        );
        let notices = s.entries.iter().filter(|e| matches!(e, Entry::Notice { .. })).count();
        assert_eq!(notices, 1, "{:?}", s.entries);
    }

    #[test]
    fn a_command_the_person_ran_comes_back_with_the_history() {
        let mut s = Session::default();
        feed(
            &mut s,
            &[event(json!({"type": "history", "messages": [
                {"role": "bashExecution", "command": "pwd", "output": "/home/ana\n", "exitCode": 0, "cancelled": false, "truncated": false, "timestamp": 1},
            ]}))],
        );
        assert!(
            matches!(&s.entries[0], Entry::Tools { calls } if calls[0].summary == "!pwd" && calls[0].output == "/home/ana\n")
        );
    }

    #[test]
    fn the_phase_follows_what_the_run_does() {
        let mut s = Session::default();
        feed(&mut s, &[event(json!({"type": "agent_start"}))]);
        assert_eq!(s.phase, "working");
        feed(&mut s, &turn_with_tools()[2..5]);
        assert_eq!(s.phase, "reading files");
        feed(
            &mut s,
            &[
                event(json!({"type": "message_start", "message": {"role": "assistant", "content": []}})),
                event(
                    json!({"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "contentIndex": 0, "delta": "hm"}}),
                ),
            ],
        );
        assert_eq!(s.phase, "thinking");
        feed(
            &mut s,
            &[event(
                json!({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "contentIndex": 1, "delta": "ok"}}),
            )],
        );
        assert_eq!(s.phase, "working");
        assert_eq!(phase_for("tavily_search"), "searching the web");
        assert_eq!(phase_for("bash"), "running command");
        assert_eq!(phase_for("edit"), "writing files");
    }

    #[test]
    fn a_refused_message_leaves_the_own_list() {
        let mut s = Session::default();
        s.add_user("x");
        feed(&mut s, &[event(json!({"type": "command_failed", "command": "prompt", "error": "busy", "text": "x"}))]);
        feed(&mut s, &[event(json!({"type": "message_start", "message": {"role": "user", "content": "x"}}))]);
        assert!(
            s.entries.iter().any(|e| matches!(e, Entry::User { text } if text == "x")),
            "sent again from elsewhere, it shows"
        );
    }

    #[test]
    fn tasks_come_from_the_last_update_and_the_history() {
        let list = |status: &str| {
            json!({"tasks": [{"id": "a", "title": "Write the test", "status": "done"},
            {"id": "b", "title": "Make it pass", "status": status, "agent": "backend"}]})
        };
        let mut s = Session::default();
        feed(
            &mut s,
            &[
                event(
                    json!({"type": "tool_execution_start", "toolCallId": "t1", "toolName": "tasks_update", "args": list("todo")}),
                ),
                event(
                    json!({"type": "tool_execution_end", "toolCallId": "t1", "toolName": "tasks_update", "isError": false,
                "result": {"content": [{"type": "text", "text": "1/2 done"}], "details": list("active")}}),
                ),
            ],
        );
        assert_eq!(s.tasks.len(), 2);
        assert_eq!((s.tasks[1].status.as_str(), s.tasks[1].agent.as_deref()), ("active", Some("backend")));
        assert!(
            !s.entries
                .iter()
                .any(|e| matches!(e, Entry::Tools { calls } if calls.iter().any(|c| c.name == "tasks_update"))),
            "the strip shows them, not the conversation"
        );
        let mut back = Session::default();
        feed(
            &mut back,
            &[event(json!({"type": "history", "messages": [
                {"role": "assistant", "content": [{"type": "toolCall", "id": "t1", "name": "tasks_update", "arguments": list("todo")}]},
                {"role": "toolResult", "toolCallId": "t1", "toolName": "tasks_update", "content": [], "details": list("done"), "isError": false},
            ]}))],
        );
        assert_eq!(back.tasks[1].status, "done", "they come back with the history");
    }

    #[test]
    fn the_extensions_statuses_are_kept() {
        let mut s = Session::default();
        feed(
            &mut s,
            &[event(
                json!({"type": "ui_notice", "request": {"method": "setStatus", "statusKey": "aria-verify", "statusText": "{\"ok\":true}"}}),
            )],
        );
        assert_eq!(s.statuses.get("aria-verify").map(String::as_str), Some("{\"ok\":true}"));
        feed(
            &mut s,
            &[event(json!({"type": "ui_notice", "request": {"method": "setStatus", "statusKey": "aria-verify"}}))],
        );
        assert!(s.statuses.get("aria-verify").is_none(), "cleared");
    }

    #[test]
    fn a_dialog_is_open_until_closed_by_anyone() {
        let mut s = Session::default();
        feed(
            &mut s,
            &[
                json!({"type": "chatDialog", "id": "u1", "method": "select", "request": {"title": "Allow?", "options": ["Allow", "Deny"]}, "hidden": false}),
            ],
        );
        assert_eq!(s.dialogs.len(), 1);
        assert_eq!(s.dialogs[0].id, "u1");
        feed(&mut s, &[json!({"type": "chatDialogClosed", "id": "u1"})]);
        assert!(s.dialogs.is_empty());
    }

    #[test]
    fn a_compaction_leaves_a_divider() {
        let mut s = Session::default();
        feed(
            &mut s,
            &[
                event(json!({"type": "compaction_start", "reason": "threshold"})),
                event(json!({"type": "compaction_end", "result": {"summary": "S"}, "aborted": false})),
            ],
        );
        assert!(matches!(&s.entries[0], Entry::Divider { text: Text::Key { key, .. } } if key == "chat.compacted"));
        assert!(!s.running);
    }

    #[test]
    fn the_state_and_quota_are_kept() {
        let mut s = Session::default();
        feed(
            &mut s,
            &[
                json!({"type": "chatState", "workdir": "/home/ana/Projects/checkout-widget", "running": false, "compacting": false,
                    "model": "claude-bridge/claude-opus-5-5", "thinking": "high", "context": 31.0, "problem": null}),
                json!({"type": "chatQuota", "windows": [{"id": "5h", "label": "5h", "percent": 62.0, "resetsAt": null}]}),
            ],
        );
        assert_eq!(s.model.as_deref(), Some("claude-bridge/claude-opus-5-5"));
        assert_eq!(s.context, Some(31.0));
        assert_eq!(s.quota[0].percent, 62.0);
        assert_eq!(s.workdir, "/home/ana/Projects/checkout-widget");
    }

    #[test]
    fn a_bang_command_shows_its_output() {
        let mut s = Session::default();
        s.add_bang("pwd", "/home/ana\n", 0);
        assert!(
            matches!(&s.entries[0], Entry::Tools { calls } if calls[0].name == "bash" && calls[0].summary == "!pwd" && calls[0].output == "/home/ana\n")
        );
    }

    #[test]
    fn texts_come_from_the_dock_and_the_client() {
        assert_eq!(
            crate::term::texts::t("en", "chat.engine.exited", &json!({})),
            "Aria's engine stopped. Your next message starts it again."
        );
        assert_eq!(crate::term::texts::t("es", "chat.retry", &json!({"attempt": 1, "max": 3})).contains("1/3"), true);
        assert_eq!(crate::term::texts::t("es", "chat.stoppedByYou", &json!({})), "detenido por ti");
        assert_eq!(crate::term::texts::t("en", "nope.nothing", &json!({})), "nope.nothing");
    }
}
