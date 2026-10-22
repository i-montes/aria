//! A turn's trace for `/debug`: every LLM call, tool call and subagent of the current turn as a span on one timeline.
//! Pure: Pi's events (and the times they arrived) in, the tree the page draws (ui/chat/parts/trace.js) out.
//! Times in the tree are ms from the turn's start; a running span has no `end`; cost and tokens are each span's own.

use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// At most this many spans in a tree: beyond it the oldest finished calls go, and `dropped` counts them.
pub const MAX_SPANS: usize = 500;
/// Previews (input, output, error) are cut at this many characters.
pub const PREVIEW: usize = 2000;

/// A subagent as the watch knows it (crates/aria/src/watch.rs): its spans carry epoch times, rebased by `Trace::tree`.
#[derive(Debug, Clone, Default)]
pub struct TraceAgent {
    pub id: u64,
    pub name: String,
    pub task: String,
    pub started_ms: i64,
    pub ended_ms: Option<i64>,
    pub failed: bool,
    /// The `parallel` call that started it, when the watch could tell.
    pub call: Option<String>,
    pub spans: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct Trace {
    started_ms: i64,
    root: Map<String, Value>,
    spans: Vec<Value>,
    index: HashMap<String, usize>,
    open_llm: Option<String>,
    llm_count: u32,
    compactions: u32,
    dropped: usize,
}

/// `text` cut to `max` characters, on a character boundary.
pub(crate) fn cut(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((at, _)) => text[..at].to_string(),
        None => text.to_string(),
    }
}

/// The text blocks of a message's content (a string, or blocks of which the `text` ones count).
fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks.iter().filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

/// What a tool works on, in one short line: the first of its usual arguments, else its first string argument.
pub(crate) fn tool_detail(args: &Value) -> String {
    let keys = ["command", "path", "file_path", "query", "url", "pattern"];
    let found = keys
        .iter()
        .find_map(|k| args[*k].as_str())
        .or_else(|| args.as_object().and_then(|m| m.values().find_map(Value::as_str)))
        .unwrap_or_default();
    cut(found.lines().next().unwrap_or_default().trim(), 80)
}

/// An LLM span's fields from an assistant message: model, usage, output, and how it ended.
pub(crate) fn finish_llm(span: &mut Value, message: &Value, end: i64) {
    span["end"] = json!(end);
    if let Some(model) = message["model"].as_str() {
        span["detail"] = json!(model);
    }
    let usage = &message["usage"];
    for (from, to) in
        [("input", "tokensIn"), ("output", "tokensOut"), ("cacheRead", "cacheRead"), ("reasoning", "reasoning")]
    {
        if let Some(n) = usage[from].as_u64() {
            span[to] = json!(n);
        }
    }
    if let Some(cost) = usage["cost"]["total"].as_f64() {
        span["cost"] = json!(cost);
    }
    let output = text_of(&message["content"]);
    if !output.is_empty() {
        span["output"] = json!(cut(&output, PREVIEW));
    }
    match message["stopReason"].as_str() {
        Some("error") => {
            span["status"] = json!("failed");
            span["error"] = json!(cut(message["errorMessage"].as_str().unwrap_or("error"), PREVIEW));
        }
        Some("aborted") => {
            span["status"] = json!("failed");
            span["error"] = json!("stopped");
        }
        _ => span["status"] = json!("done"),
    }
}

/// A tool span's start: its name, target and arguments.
pub(crate) fn tool_span(id: String, name: &str, args: &Value, start: i64) -> Value {
    let input = if args.is_null() { String::new() } else { cut(&args.to_string(), PREVIEW) };
    json!({"id": id, "kind": "tool", "name": name, "detail": tool_detail(args), "status": "running", "start": start, "input": input})
}

/// A tool span's end: its output, or its error.
pub(crate) fn finish_tool(span: &mut Value, output: &str, failed: bool, end: i64) {
    span["end"] = json!(end);
    let text = cut(output, PREVIEW);
    if failed {
        span["status"] = json!("failed");
        span["error"] = json!(text);
    } else {
        span["status"] = json!("done");
        span["output"] = json!(text);
    }
}

impl Trace {
    pub fn new(started_ms: i64) -> Trace {
        let mut root = Map::new();
        root.insert("id".into(), json!("root"));
        root.insert("kind".into(), json!("agent"));
        root.insert("name".into(), json!("Aria"));
        root.insert("status".into(), json!("running"));
        root.insert("start".into(), json!(0));
        root.insert("input".into(), json!(""));
        Trace {
            started_ms,
            root,
            spans: Vec::new(),
            index: HashMap::new(),
            open_llm: None,
            llm_count: 0,
            compactions: 0,
            dropped: 0,
        }
    }

    pub fn started_ms(&self) -> i64 {
        self.started_ms
    }

    fn rel(&self, at_ms: i64) -> i64 {
        (at_ms - self.started_ms).max(0)
    }

    fn push(&mut self, span: Value) {
        let id = span["id"].as_str().unwrap_or_default().to_string();
        self.index.insert(id, self.spans.len());
        self.spans.push(span);
        self.trim();
    }

    /// Over `MAX_SPANS`, the oldest finished spans go (a running one never does).
    fn trim(&mut self) {
        if self.spans.len() <= MAX_SPANS {
            return;
        }
        let before = self.spans.len();
        let mut extra = before - MAX_SPANS;
        self.spans.retain(|s| {
            if extra > 0 && s["status"] != "running" {
                extra -= 1;
                false
            } else {
                true
            }
        });
        self.dropped += before - self.spans.len();
        self.reindex();
    }

    fn reindex(&mut self) {
        self.index =
            self.spans.iter().enumerate().map(|(i, s)| (s["id"].as_str().unwrap_or_default().to_string(), i)).collect();
    }

    fn span_mut(&mut self, id: &str) -> Option<&mut Value> {
        self.index.get(id).copied().and_then(move |i| self.spans.get_mut(i))
    }

    /// One of Pi's events; true when the tree changed.
    pub fn feed(&mut self, event: &Value, at_ms: i64) -> bool {
        let at = self.rel(at_ms);
        match event["type"].as_str() {
            Some("message_start") if event["message"]["role"] == "user" => {
                let text = text_of(&event["message"]["content"]);
                self.root.insert("input".into(), json!(cut(&text, PREVIEW)));
                true
            }
            Some("message_start") if event["message"]["role"] == "assistant" => {
                self.llm_count += 1;
                let id = format!("llm-{}", self.llm_count);
                self.open_llm = Some(id.clone());
                self.push(json!({"id": id, "kind": "llm", "name": "LLM call", "status": "running", "start": at}));
                true
            }
            Some("message_end") if event["message"]["role"] == "assistant" => {
                let id = match self.open_llm.take() {
                    Some(id) => id,
                    None => {
                        // An end with no start (a trace turned on mid-message): a span from now.
                        self.llm_count += 1;
                        let id = format!("llm-{}", self.llm_count);
                        self.push(
                            json!({"id": id, "kind": "llm", "name": "LLM call", "status": "running", "start": at}),
                        );
                        id
                    }
                };
                if let Some(model) = event["message"]["model"].as_str() {
                    self.root.insert("detail".into(), json!(model));
                }
                let message = event["message"].clone();
                if let Some(span) = self.span_mut(&id) {
                    finish_llm(span, &message, at);
                }
                true
            }
            Some("tool_execution_start") => {
                let id = format!("tool-{}", event["toolCallId"].as_str().unwrap_or_default());
                let span = tool_span(id, event["toolName"].as_str().unwrap_or("tool"), &event["args"], at);
                self.push(span);
                true
            }
            Some("tool_execution_end") => {
                let id = format!("tool-{}", event["toolCallId"].as_str().unwrap_or_default());
                let output = text_of(&event["result"]["content"]);
                let failed = event["isError"] == true;
                match self.span_mut(&id) {
                    Some(span) => {
                        finish_tool(span, &output, failed, at);
                        true
                    }
                    None => false,
                }
            }
            Some("compaction_start") => {
                self.compactions += 1;
                let id = format!("compaction-{}", self.compactions);
                self.push(json!({"id": id, "kind": "tool", "name": "compaction", "status": "running", "start": at}));
                true
            }
            Some("compaction_end") => {
                let id = format!("compaction-{}", self.compactions);
                let failed = !event["result"].is_object();
                match self.span_mut(&id) {
                    Some(span) => {
                        span["end"] = json!(at);
                        span["status"] = json!(if failed { "failed" } else { "done" });
                        true
                    }
                    None => false,
                }
            }
            _ => false,
        }
    }

    /// The turn ended: the root ends, and anything still running ends with it, failed.
    pub fn settle(&mut self, failed: bool, at_ms: i64) {
        let at = self.rel(at_ms);
        self.root.insert("end".into(), json!(at));
        self.root.insert("status".into(), json!(if failed { "failed" } else { "done" }));
        self.open_llm = None;
        for span in &mut self.spans {
            if span["status"] == "running" {
                span["status"] = json!("failed");
                span["end"] = json!(at);
            }
        }
    }

    /// The tree the page draws: the root, its spans in order of start, and the subagents of this turn under the call
    /// that started them (or the root). `now` is ms from the turn's start.
    pub fn tree(&self, agents: &[TraceAgent], now_ms: i64) -> Value {
        let mut children: Vec<Value> = self.spans.clone();
        let mut root_agents = Vec::new();
        for agent in agents.iter().filter(|a| a.started_ms >= self.started_ms) {
            let node = self.agent_node(agent);
            let parent =
                agent.call.as_ref().and_then(|call| children.iter_mut().find(|s| s["id"] == format!("tool-{call}")));
            match parent {
                Some(parent) => {
                    if !parent["children"].is_array() {
                        parent["children"] = json!([]);
                    }
                    parent["children"].as_array_mut().unwrap().push(node);
                }
                None => root_agents.push(node),
            }
        }
        children.extend(root_agents);
        children.sort_by_key(|s| s["start"].as_i64().unwrap_or(0));
        let mut root = self.root.clone();
        root.insert("now".into(), json!(self.rel(now_ms)));
        root.insert("dropped".into(), json!(self.dropped));
        root.insert("children".into(), Value::Array(children));
        Value::Object(root)
    }

    fn agent_node(&self, agent: &TraceAgent) -> Value {
        let rebase = |v: &Value| v.as_i64().map(|t| json!(self.rel(t)));
        let mut spans: Vec<Value> = agent
            .spans
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s["id"] = json!(format!("agent-{}-{}", agent.id, s["id"].as_str().unwrap_or_default()));
                if let Some(start) = rebase(&s["start"]) {
                    s["start"] = start;
                }
                if let Some(end) = rebase(&s["end"]) {
                    s["end"] = end;
                }
                s
            })
            .collect();
        spans.sort_by_key(|s| s["start"].as_i64().unwrap_or(0));
        let status = match (agent.ended_ms, agent.failed) {
            (None, _) => "running",
            (Some(_), true) => "failed",
            (Some(_), false) => "done",
        };
        let mut node = json!({
            "id": format!("agent-{}", agent.id), "kind": "agent", "name": agent.name, "detail": cut(&agent.task, 80),
            "status": status, "start": self.rel(agent.started_ms), "input": cut(&agent.task, PREVIEW), "children": spans,
        });
        if let Some(end) = agent.ended_ms {
            node["end"] = json!(self.rel(end));
        }
        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn child<'a>(tree: &'a Value, id: &str) -> &'a Value {
        tree["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["id"] == id)
            .unwrap_or_else(|| panic!("no {id} in {tree}"))
    }

    fn assistant_end(extra: Value) -> Value {
        let mut message =
            json!({"role": "assistant", "content": [{"type": "text", "text": "ok"}], "stopReason": "stop"});
        for (k, v) in extra.as_object().unwrap() {
            message[k] = v.clone();
        }
        json!({"type": "message_end", "message": message})
    }

    #[test]
    fn an_llm_call_spans_message_start_to_end_with_usage() {
        let mut t = Trace::new(1000);
        assert!(t.feed(&json!({"type": "message_start", "message": {"role": "assistant"}}), 1100));
        let usage = json!({"input": 10, "output": 5, "cacheRead": 3, "reasoning": 2, "cost": {"total": 0.01}});
        assert!(t.feed(&assistant_end(json!({"model": "m", "usage": usage})), 1900));
        let tree = t.tree(&[], 2000);
        let llm = child(&tree, "llm-1");
        assert_eq!(llm["start"], 100);
        assert_eq!(llm["end"], 900);
        assert_eq!(llm["detail"], "m");
        assert_eq!(llm["tokensIn"], 10);
        assert_eq!(llm["tokensOut"], 5);
        assert_eq!(llm["cacheRead"], 3);
        assert_eq!(llm["reasoning"], 2);
        assert_eq!(llm["cost"], 0.01);
        assert_eq!(llm["output"], "ok");
        assert_eq!(llm["status"], "done");
        assert_eq!(llm["kind"], "llm");
        assert_eq!(tree["now"], 1000);
    }

    #[test]
    fn a_failed_or_aborted_message_fails_its_span() {
        let mut t = Trace::new(0);
        t.feed(&json!({"type": "message_start", "message": {"role": "assistant"}}), 1);
        t.feed(&assistant_end(json!({"stopReason": "error", "errorMessage": "boom"})), 2);
        t.feed(&json!({"type": "message_start", "message": {"role": "assistant"}}), 3);
        t.feed(&assistant_end(json!({"stopReason": "aborted"})), 4);
        let tree = t.tree(&[], 5);
        assert_eq!(child(&tree, "llm-1")["status"], "failed");
        assert_eq!(child(&tree, "llm-1")["error"], "boom");
        assert_eq!(child(&tree, "llm-2")["status"], "failed");
        assert_eq!(child(&tree, "llm-2")["error"], "stopped");
    }

    #[test]
    fn a_tool_spans_its_execution_by_call_id() {
        let mut t = Trace::new(0);
        t.feed(&json!({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "bash", "args": {"command": "ls -la"}}), 200);
        t.feed(&json!({"type": "tool_execution_start", "toolCallId": "c2", "toolName": "bash", "args": {"command": "false"}}), 210);
        t.feed(&json!({"type": "tool_execution_end", "toolCallId": "c1", "isError": false, "result": {"content": [{"type": "text", "text": "out"}]}}), 500);
        t.feed(&json!({"type": "tool_execution_end", "toolCallId": "c2", "isError": true, "result": {"content": [{"type": "text", "text": "bad"}]}}), 600);
        let tree = t.tree(&[], 700);
        let tool = child(&tree, "tool-c1");
        assert_eq!(tool["name"], "bash");
        assert_eq!(tool["detail"], "ls -la");
        assert_eq!(tool["input"], json!({"command": "ls -la"}).to_string());
        assert_eq!(tool["output"], "out");
        assert_eq!((tool["start"].as_i64(), tool["end"].as_i64()), (Some(200), Some(500)));
        assert_eq!(tool["status"], "done");
        let failed = child(&tree, "tool-c2");
        assert_eq!(failed["status"], "failed");
        assert_eq!(failed["error"], "bad");
    }

    #[test]
    fn the_users_message_is_the_roots_input() {
        let mut t = Trace::new(0);
        t.feed(&json!({"type": "message_start", "message": {"role": "user", "content": [{"type": "text", "text": "launch it"}]}}), 1);
        assert_eq!(t.tree(&[], 2)["input"], "launch it");
    }

    #[test]
    fn a_turn_without_a_user_message_has_no_input() {
        let mut t = Trace::new(0);
        t.feed(&json!({"type": "message_start", "message": {"role": "assistant"}}), 1);
        let tree = t.tree(&[], 2);
        assert_eq!(tree["input"], "");
        assert_eq!(tree["children"].as_array().unwrap().len(), 1);
        assert_eq!(tree["status"], "running");
    }

    #[test]
    fn compaction_is_a_span() {
        let mut t = Trace::new(0);
        t.feed(&json!({"type": "compaction_start", "reason": "threshold"}), 10);
        t.feed(&json!({"type": "compaction_end", "result": {"summary": "S"}}), 40);
        let tree = t.tree(&[], 50);
        let c = child(&tree, "compaction-1");
        assert_eq!(
            (c["kind"].as_str(), c["name"].as_str(), c["status"].as_str()),
            (Some("tool"), Some("compaction"), Some("done"))
        );
        assert_eq!(c["end"], 40);
    }

    #[test]
    fn settle_ends_the_root() {
        let mut t = Trace::new(0);
        t.feed(&json!({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "bash", "args": {}}), 100);
        t.settle(false, 1200);
        let tree = t.tree(&[], 1300);
        assert_eq!(tree["status"], "done");
        assert_eq!(tree["end"], 1200);
        assert_eq!(child(&tree, "tool-c1")["status"], "failed");
        assert_eq!(child(&tree, "tool-c1")["end"], 1200);
        let mut t = Trace::new(0);
        t.settle(true, 5);
        assert_eq!(t.tree(&[], 5)["status"], "failed");
    }

    #[test]
    fn previews_are_cut() {
        let mut t = Trace::new(0);
        t.feed(&json!({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "read", "args": {}}), 1);
        let long = "é".repeat(5000);
        t.feed(&json!({"type": "tool_execution_end", "toolCallId": "c1", "result": {"content": [{"type": "text", "text": long}]}}), 2);
        assert_eq!(child(&t.tree(&[], 3), "tool-c1")["output"].as_str().unwrap().chars().count(), PREVIEW);
    }

    #[test]
    fn over_500_spans_the_oldest_finished_go() {
        let mut t = Trace::new(0);
        t.feed(&json!({"type": "tool_execution_start", "toolCallId": "live", "toolName": "bash", "args": {}}), 0);
        for i in 0..600 {
            let id = format!("c{i}");
            t.feed(&json!({"type": "tool_execution_start", "toolCallId": id, "toolName": "bash", "args": {}}), i + 1);
            t.feed(&json!({"type": "tool_execution_end", "toolCallId": id, "result": {"content": []}}), i + 1);
        }
        let tree = t.tree(&[], 700);
        let children = tree["children"].as_array().unwrap();
        assert_eq!(children.len(), MAX_SPANS);
        assert_eq!(tree["dropped"], 101);
        assert!(children.iter().any(|s| s["id"] == "tool-live"));
        assert!(children.iter().any(|s| s["id"] == "tool-c599"));
        assert!(!children.iter().any(|s| s["id"] == "tool-c0"));
    }

    #[test]
    fn agents_go_under_their_call_or_the_root() {
        let mut t = Trace::new(10_000);
        t.feed(
            &json!({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "parallel", "args": {}}),
            10_100,
        );
        let spans = vec![
            json!({"id": "llm-1", "kind": "llm", "name": "LLM call", "status": "done", "start": 10_200, "end": 10_400}),
        ];
        let agents = vec![
            TraceAgent {
                id: 1,
                name: "researcher".into(),
                task: "find X".into(),
                started_ms: 10_150,
                ended_ms: None,
                call: Some("c1".into()),
                spans,
                ..Default::default()
            },
            TraceAgent {
                id: 2,
                name: "loose".into(),
                started_ms: 10_160,
                ended_ms: Some(10_900),
                call: None,
                ..Default::default()
            },
            TraceAgent {
                id: 3,
                name: "unknown call".into(),
                started_ms: 10_170,
                ended_ms: Some(10_900),
                failed: true,
                call: Some("zz".into()),
                ..Default::default()
            },
            TraceAgent { id: 4, name: "before".into(), started_ms: 9_000, ..Default::default() },
        ];
        let tree = t.tree(&agents, 11_000);
        let under = &child(&tree, "tool-c1")["children"][0];
        assert_eq!(under["id"], "agent-1");
        assert_eq!(under["status"], "running");
        assert_eq!(under["start"], 150);
        assert_eq!(under["children"][0]["id"], "agent-1-llm-1");
        assert_eq!(under["children"][0]["start"], 200);
        assert_eq!(under["children"][0]["end"], 400);
        assert_eq!(child(&tree, "agent-2")["status"], "done");
        assert_eq!(child(&tree, "agent-2")["end"], 900);
        assert_eq!(child(&tree, "agent-3")["status"], "failed");
        assert!(!tree["children"].as_array().unwrap().iter().any(|s| s["id"] == "agent-4"));
    }

    #[test]
    fn the_detail_of_a_tool_is_its_target() {
        assert_eq!(tool_detail(&json!({"path": "/a/b"})), "/a/b");
        assert_eq!(tool_detail(&json!({"query": "q"})), "q");
        assert_eq!(tool_detail(&json!({"other": "first\nsecond"})), "first");
        assert_eq!(tool_detail(&json!({"other": "x".repeat(200)})).chars().count(), 80);
        assert_eq!(tool_detail(&json!({"n": 3})), "");
    }
}
