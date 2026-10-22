//! The messages between the dock (Rust) and its page (the web UI), one JSON object per message with
//! a `type` tag. The page sends `PageMessage`s; the dock answers with `DockMessage`s. Requests that
//! expect an answer carry an `id` that the answer repeats.

use crate::history::EarlierCursor;
use crate::quota::Window as QuotaWindow;
use crate::sessions::SessionEntry;
use crate::shortcut::{InUse, Platform, Shortcut};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type")]
pub enum PageMessage {
    /// The page has loaded.
    #[serde(rename = "ready")]
    Ready,
    /// The open transition has finished.
    #[serde(rename = "opened")]
    Opened,
    /// The page asks to close the dock (Esc).
    #[serde(rename = "closeRequest")]
    CloseRequest,
    /// The person pressed Cmd+Q twice: Aria stops its background work and its Pis, and exits.
    #[serde(rename = "quitRequest")]
    QuitRequest,
    #[serde(rename = "loadSettings")]
    LoadSettings { id: u64 },
    #[serde(rename = "saveSettings")]
    SaveSettings { id: u64, settings: Value },
    /// `provider` is `opencode-go`, `anthropic-api`, `typesafe` or `tavily`.
    #[serde(rename = "saveKey")]
    SaveKey { id: u64, provider: String, key: String },
    /// Validate the key already stored for a provider (`auth.json`), without the page ever seeing it.
    #[serde(rename = "checkKey")]
    CheckKey { id: u64, provider: String },
    /// Start the Anthropic OAuth sign-in.
    #[serde(rename = "startOauth")]
    StartOAuth { id: u64 },
    /// Start an MCP server's OAuth sign-in (`service` is `notion`).
    #[serde(rename = "startMcpOauth")]
    StartMcpOAuth { id: u64, service: String },
    #[serde(rename = "openUrl")]
    OpenUrl { url: String },
    /// A chat message. `behavior` is `steer` or `followUp` when a run is already going. `attachments` are the files the
    /// person attached, read by the page as base64; the dock saves them and tells Pi where they are.
    #[serde(rename = "chatSend")]
    ChatSend {
        text: String,
        behavior: Option<String>,
        #[serde(default)]
        attachments: Vec<Attachment>,
    },
    #[serde(rename = "chatAbort")]
    ChatAbort,
    /// The dock opened: tell the page where Aria works and how it is doing. `fresh` is a page that just booted (its
    /// first look at each tab): a Pi that is already running replays its conversation to it.
    #[serde(rename = "chatLoad")]
    ChatLoad {
        #[serde(default)]
        fresh: bool,
    },
    /// The page of earlier messages before `cursor` (none: the newest earlier page), read from the tab's segment files;
    /// answered with an `earlier` chat event.
    #[serde(rename = "chatEarlier")]
    ChatEarlier {
        #[serde(default)]
        cursor: Option<EarlierCursor>,
    },
    /// Cancel a queued message: Pi can only empty its queue, so the others are queued again, in order: those in `steer`
    /// for the next gap between steps, those in `keep` for when the run ends.
    #[serde(rename = "chatClearQueue")]
    ChatClearQueue {
        #[serde(default)]
        steer: Vec<String>,
        #[serde(default)]
        keep: Vec<String>,
    },
    /// The answer to a dialog (`id` is Pi's request id): a `value`, a `confirmed` flag or `cancelled`.
    #[serde(rename = "uiAnswer")]
    UiAnswer { id: String, value: Option<String>, confirmed: Option<bool>, cancelled: Option<bool> },
    #[serde(rename = "listCommands")]
    ListCommands { id: u64 },
    #[serde(rename = "listModels")]
    ListModels { id: u64 },
    #[serde(rename = "setModel")]
    SetModel { id: u64, model: String },
    #[serde(rename = "listFiles")]
    ListFiles { id: u64, query: String },
    /// One of Pi's own commands the dock offers in its `/` menu (`compact`, `get_session_stats`, `export_html`), with its
    /// fields; replies `piReply`.
    #[serde(rename = "piCommand")]
    PiCommand {
        id: u64,
        command: String,
        #[serde(default)]
        args: Value,
    },
    /// Download and list the Claude provider (`claude-bridge`) in Aria's Pi; replies `saved`, or an `error` whose `code` is
    /// `pi-missing`, `claude-code-missing` or `claude-bridge-install-failed`.
    #[serde(rename = "installClaudeBridge")]
    InstallClaudeBridge { id: u64 },
    /// A credential or package changed: every session's Pi stops and starts again when next used, so it sees the change.
    #[serde(rename = "enginesRestart")]
    EnginesRestart,
    /// Find the MCP servers other apps configure (Claude Code, Claude Desktop, Cursor, VS Code); replies `mcpFound`.
    #[serde(rename = "mcpScan")]
    McpScan { id: u64 },
    /// Copy the found servers with these ids (`source:name`) into Aria's `mcp.json`; replies `saved` (and every Pi
    /// restarts when its run settles), or an `error` whose `code` is `mcp-file-broken`, `mcp-write-failed` or
    /// `nothing-to-import` (none of the ids found, or only SSE ones).
    #[serde(rename = "mcpImport")]
    McpImport { id: u64, ids: Vec<String> },
    /// Is the chat on screen (false while the settings are open)? A prompt nobody sees gets a notification.
    #[serde(rename = "chatShown")]
    ChatShown { shown: bool },
    /// A chat message for one session (each session is its own tab and its own Pi).
    #[serde(rename = "session")]
    Session { session: String, message: Box<PageMessage> },
    #[serde(rename = "sessionsLoad")]
    SessionsLoad { id: u64 },
    /// The page owns the tabs; this is the list to keep.
    #[serde(rename = "sessionsSave")]
    SessionsSave { sessions: Vec<SessionEntry>, active: Option<String> },
    /// Close a session's Pi (its conversation stays in Pi's session file).
    #[serde(rename = "sessionClose")]
    SessionClose { session: String },
    /// The scheduled tasks, for the `/schedules` panel; replies `schedules`.
    #[serde(rename = "schedulesList")]
    SchedulesList { id: u64 },
    /// Pause or resume a scheduled task; replies `schedules`.
    #[serde(rename = "scheduleUpdate")]
    ScheduleUpdate { id: u64, schedule: String, enabled: bool },
    /// Delete a scheduled task and its tab; replies `schedules`.
    #[serde(rename = "scheduleDelete")]
    ScheduleDelete { id: u64, schedule: String },
    /// Stop a background job (any session's: the page lists them all) and whatever it forked.
    #[serde(rename = "jobStop")]
    JobStop { job: String },
    /// The latest `lines` lines of a job's output (80 when absent); replies `jobOutput` with the same `id`.
    #[serde(rename = "jobOutput")]
    JobOutput { id: u64, job: String, lines: Option<usize> },
    /// Hide a finished job from the background list.
    #[serde(rename = "jobDismiss")]
    JobDismiss { job: String },
    /// `/debug true|false` in a tab (wrapped in `session`): trace its turns, or stop and drop the trace.
    #[serde(rename = "traceSet")]
    TraceSet { on: bool },
    /// The active tab's trace panel is shown: the dock's surface widens to the left (or narrows back).
    #[serde(rename = "wide")]
    Wide { on: bool },
}

/// A file attached to a chat message: its name as the person had it, its media type and its bytes in base64.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Attachment {
    pub name: String,
    #[serde(default)]
    pub mime: String,
    pub data: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum DockMessage {
    #[serde(rename = "open")]
    Open { monitor: String },
    #[serde(rename = "close")]
    Close,
    /// Show the corner notice ("Aria is installed — press Super A to set it up").
    #[serde(rename = "notice")]
    Notice,
    /// The host's shortcut in use, for the notice, the hints and the settings' Shortcut row (`Super + A`,
    /// `Option + Space`); `taken` the ones tried before it that something else holds (the chosen one, the default), in
    /// the same form; `os` (`linux`, `mac`) which modifier names the settings row records. An empty `hotkey` means none
    /// could be registered (macOS, every candidate taken; `taken` lists them all).
    #[serde(rename = "platform")]
    Platform { hotkey: String, taken: Vec<String>, os: String },
    /// Aria's version (`0.2.0`, from the workspace's `Cargo.toml`), sent once when the dock starts; the settings show it.
    #[serde(rename = "version")]
    Version { version: String },
    /// `connected` names what is really set up, never a secret: `provider:type` for each credential in
    /// `auth.json` (`anthropic:oauth`) and `mcp:<service>` for each configured MCP service. `servers` names every server
    /// in Aria's `mcp.json` except `engram` (connected or not): the connectors whose access level the settings show.
    #[serde(rename = "settings")]
    Settings {
        id: u64,
        settings: Option<Value>,
        exists: bool,
        problem: Option<String>,
        connected: Vec<String>,
        servers: Vec<String>,
    },
    #[serde(rename = "saved")]
    Saved { id: u64 },
    /// The answer to `mcpScan`: `[{ id, name, importAs, source, kind, target, secrets, warnings, importable, disabled }]`
    /// (`mcp_import::Found`), never a secret's value, only the names of the variables and headers. `kind: "sse"` is
    /// `importable: false`; `warnings` name what may not work the same in Aria.
    #[serde(rename = "mcpFound")]
    McpFound { id: u64, servers: Value },
    /// `error` is a UI string key: `key-rejected`, `network-unavailable`, `auth-file-broken`.
    #[serde(rename = "keyResult")]
    KeyResult { id: u64, ok: bool, error: Option<String> },
    #[serde(rename = "oauthUrl")]
    OAuthUrl { id: u64, url: String },
    /// `error` is a UI string key: `oauth-denied`, `oauth-timeout`, `oauth-state`, `network-unavailable`.
    #[serde(rename = "oauthResult")]
    OAuthResult { id: u64, ok: bool, error: Option<String> },
    /// Something prevents Aria from working (`hyprland-classic-config`, `hyprland-not-running`, …).
    #[serde(rename = "problem")]
    Problem { code: String },
    #[serde(rename = "error")]
    Error { id: u64, code: String },
    /// Where Aria works and how the engine is; `problem` is a UI string key (`pi-missing`, `node-missing`).
    #[serde(rename = "chatState")]
    ChatState {
        workdir: String,
        running: bool,
        /// Pi is compacting with no run of its own (a compaction Aria asked for): the page shows it as working.
        compacting: bool,
        model: Option<String>,
        thinking: Option<String>,
        context: Option<f64>,
        problem: Option<String>,
    },
    /// A Pi session event, verbatim.
    #[serde(rename = "chatEvent")]
    ChatEvent { event: Value },
    /// How much of the provider's quota is spent, for the model in use (empty when the provider has none to read).
    /// The subagents this session's Pi started (see crates/aria/src/watch.rs and ui/chat/parts/subagents.js).
    /// `settings.json` changed outside the page (`/effort`, `/autocompact`, an editor): the page reads it again.
    #[serde(rename = "settingsChanged")]
    SettingsChanged,
    #[serde(rename = "chatAgents")]
    ChatAgents { agents: Value },
    /// A traced tab's current turn as a tree (crates/aria/src/trace.rs), or `null` before its first turn and when
    /// tracing stops. At most four times a second, and once when the turn settles.
    #[serde(rename = "chatTrace")]
    ChatTrace { trace: Value },
    #[serde(rename = "chatQuota")]
    ChatQuota { windows: Vec<QuotaWindow> },
    /// An extension dialog; `hidden` says the owner could not see the chat when it arrived.
    #[serde(rename = "chatDialog")]
    ChatDialog { id: String, method: String, request: Value, hidden: bool },
    #[serde(rename = "chatDialogClosed")]
    ChatDialogClosed { id: String },
    /// `starting` | `ready` | `exited`.
    #[serde(rename = "chatEngine")]
    ChatEngine { state: String },
    #[serde(rename = "commands")]
    Commands { id: u64, commands: Value },
    #[serde(rename = "models")]
    Models { id: u64, models: Value },
    #[serde(rename = "files")]
    Files { id: u64, files: Vec<String> },
    /// Pi's answer to a `piCommand`: its `data`, or its `error` when it refused (or `command-unknown`, never sent).
    #[serde(rename = "piReply")]
    PiReply { id: u64, ok: bool, data: Value, error: Option<String> },
    /// What a session says (its chat messages and the replies to its requests).
    #[serde(rename = "session")]
    Session { session: String, message: Box<DockMessage> },
    #[serde(rename = "sessions")]
    Sessions { id: u64, sessions: Vec<SessionEntry>, active: Option<String> },
    /// Every scheduled task (aria-core `schedules::Task`, camelCase, with `next`), the answer to a schedules request.
    #[serde(rename = "schedules")]
    Schedules { id: u64, tasks: Value },
    /// A scheduled task's tab, made by its first run: the page adds it (not active) if it does not have it.
    #[serde(rename = "scheduleTab")]
    ScheduleTab { id: String, title: String, schedule: String },
    /// A deleted task's tab is gone.
    #[serde(rename = "scheduleRemoved")]
    ScheduleRemoved { id: String },
    /// A task the model made or widened to write waits for the person: the page shows it in `/schedules`, to approve.
    #[serde(rename = "scheduleAwaiting")]
    ScheduleAwaiting { schedule: String },
    /// Every session's background jobs (crates/aria/src/jobs.rs `Job`, camelCase): running and queued first, then
    /// finished, newest first. Sent whenever they change (at most four times a second) and on any session's `chatLoad`.
    #[serde(rename = "jobs")]
    Jobs { jobs: Value },
    /// The answer to a `jobOutput`: empty when the job is unknown or keeps no output (a subagent).
    #[serde(rename = "jobOutput")]
    JobOutput { id: u64, lines: Vec<String> },
}

impl DockMessage {
    /// The `platform` message for the shortcut in use (`taken`: the ones tried before it), in the page's form.
    pub fn platform(in_use: &InUse, platform: Platform) -> DockMessage {
        DockMessage::Platform {
            hotkey: in_use.shortcut.display(),
            taken: in_use.taken.iter().map(Shortcut::display).collect(),
            os: platform.name().into(),
        }
    }
}

impl PageMessage {
    /// The variant's name, for logs: a message can carry a key (`SaveKey`), so its fields are never printed.
    pub fn log_name(&self) -> String {
        let debug = format!("{self:?}");
        debug.split(|c: char| !c.is_alphanumeric()).next().unwrap_or_default().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn page(text: &str) -> PageMessage {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn a_page_message_is_logged_by_its_name_only_never_with_a_secret() {
        let message = PageMessage::SaveKey { id: 1, provider: "opencode-go".into(), key: "sk-secret".into() };
        assert_eq!(message.log_name(), "SaveKey");
        assert_eq!(PageMessage::Ready.log_name(), "Ready");
    }

    #[test]
    fn page_messages_parse() {
        assert_eq!(page(r#"{"type":"ready"}"#), PageMessage::Ready);
        assert_eq!(page(r#"{"type":"opened"}"#), PageMessage::Opened);
        assert_eq!(page(r#"{"type":"closeRequest"}"#), PageMessage::CloseRequest);
        assert_eq!(page(r#"{"type":"quitRequest"}"#), PageMessage::QuitRequest);
        assert_eq!(page(r#"{"type":"loadSettings","id":1}"#), PageMessage::LoadSettings { id: 1 });
        assert_eq!(
            page(r#"{"type":"saveSettings","id":2,"settings":{"a":1}}"#),
            PageMessage::SaveSettings { id: 2, settings: json!({"a": 1}) }
        );
        assert_eq!(
            page(r#"{"type":"saveKey","id":3,"provider":"typesafe","key":"k"}"#),
            PageMessage::SaveKey { id: 3, provider: "typesafe".into(), key: "k".into() }
        );
        assert_eq!(
            page(r#"{"type":"checkKey","id":6,"provider":"opencode-go"}"#),
            PageMessage::CheckKey { id: 6, provider: "opencode-go".into() }
        );
        assert_eq!(page(r#"{"type":"startOauth","id":4}"#), PageMessage::StartOAuth { id: 4 });
        assert_eq!(
            page(r#"{"type":"startMcpOauth","id":5,"service":"notion"}"#),
            PageMessage::StartMcpOAuth { id: 5, service: "notion".into() }
        );
        assert_eq!(page(r#"{"type":"openUrl","url":"https://x"}"#), PageMessage::OpenUrl { url: "https://x".into() });
    }

    #[test]
    fn trace_messages_have_their_names() {
        assert_eq!(page(r#"{"type":"traceSet","on":true}"#), PageMessage::TraceSet { on: true });
        assert_eq!(page(r#"{"type":"wide","on":false}"#), PageMessage::Wide { on: false });
        let value = serde_json::to_value(DockMessage::ChatTrace { trace: Value::Null }).unwrap();
        assert_eq!(value, json!({"type": "chatTrace", "trace": null}));
    }

    #[test]
    fn chat_send_without_attachments_still_parses() {
        assert_eq!(
            page(r#"{"type":"chatSend","text":"hi"}"#),
            PageMessage::ChatSend { text: "hi".into(), behavior: None, attachments: vec![] }
        );
        assert_eq!(
            page(
                r#"{"type":"chatSend","text":"look","attachments":[{"name":"a.png","mime":"image/png","data":"AAAA"}]}"#
            ),
            PageMessage::ChatSend {
                text: "look".into(),
                behavior: None,
                attachments: vec![Attachment { name: "a.png".into(), mime: "image/png".into(), data: "AAAA".into() }]
            }
        );
    }

    #[test]
    fn chat_page_messages_parse() {
        assert_eq!(
            page(r#"{"type":"chatSend","text":"hi","behavior":"steer"}"#),
            PageMessage::ChatSend { text: "hi".into(), behavior: Some("steer".into()), attachments: vec![] }
        );
        assert_eq!(
            page(r#"{"type":"chatSend","text":"hi"}"#),
            PageMessage::ChatSend { text: "hi".into(), behavior: None, attachments: vec![] }
        );
        assert_eq!(page(r#"{"type":"chatAbort"}"#), PageMessage::ChatAbort);
        assert_eq!(page(r#"{"type":"chatLoad"}"#), PageMessage::ChatLoad { fresh: false });
        assert_eq!(page(r#"{"type":"chatLoad","fresh":true}"#), PageMessage::ChatLoad { fresh: true });
        assert_eq!(page(r#"{"type":"chatEarlier"}"#), PageMessage::ChatEarlier { cursor: None });
        assert_eq!(page(r#"{"type":"chatEarlier","cursor":null}"#), PageMessage::ChatEarlier { cursor: None });
        assert_eq!(
            page(r#"{"type":"chatEarlier","cursor":{"segment":1,"index":40}}"#),
            PageMessage::ChatEarlier { cursor: Some(EarlierCursor { segment: 1, index: 40 }) }
        );
        assert_eq!(page(r#"{"type":"installClaudeBridge","id":12}"#), PageMessage::InstallClaudeBridge { id: 12 });
        assert_eq!(page(r#"{"type":"enginesRestart"}"#), PageMessage::EnginesRestart);
        assert_eq!(page(r#"{"type":"mcpScan","id":7}"#), PageMessage::McpScan { id: 7 });
        assert_eq!(
            page(r#"{"type":"mcpImport","id":8,"ids":["cursor:docs","vscode:tracker"]}"#),
            PageMessage::McpImport { id: 8, ids: vec!["cursor:docs".into(), "vscode:tracker".into()] }
        );
        assert_eq!(page(r#"{"type":"chatShown","shown":false}"#), PageMessage::ChatShown { shown: false });
        assert_eq!(page(r#"{"type":"chatClearQueue"}"#), PageMessage::ChatClearQueue { steer: vec![], keep: vec![] });
        assert_eq!(
            page(r#"{"type":"chatClearQueue","keep":["a","b"]}"#),
            PageMessage::ChatClearQueue { steer: vec![], keep: vec!["a".into(), "b".into()] }
        );
        assert_eq!(
            page(r#"{"type":"chatClearQueue","steer":["now"],"keep":["later"]}"#),
            PageMessage::ChatClearQueue { steer: vec!["now".into()], keep: vec!["later".into()] }
        );
        assert_eq!(
            page(r#"{"type":"uiAnswer","id":"u1","value":"Allow"}"#),
            PageMessage::UiAnswer { id: "u1".into(), value: Some("Allow".into()), confirmed: None, cancelled: None }
        );
        assert_eq!(
            page(r#"{"type":"uiAnswer","id":"u2","confirmed":true}"#),
            PageMessage::UiAnswer { id: "u2".into(), value: None, confirmed: Some(true), cancelled: None }
        );
        assert_eq!(
            page(r#"{"type":"uiAnswer","id":"u3","cancelled":true}"#),
            PageMessage::UiAnswer { id: "u3".into(), value: None, confirmed: None, cancelled: Some(true) }
        );
        assert_eq!(page(r#"{"type":"listCommands","id":7}"#), PageMessage::ListCommands { id: 7 });
        assert_eq!(page(r#"{"type":"listModels","id":8}"#), PageMessage::ListModels { id: 8 });
        assert_eq!(
            page(r#"{"type":"piCommand","id":5,"command":"compact","args":{"customInstructions":"code"}}"#),
            PageMessage::PiCommand { id: 5, command: "compact".into(), args: json!({"customInstructions": "code"}) }
        );
        assert_eq!(
            page(r#"{"type":"piCommand","id":6,"command":"get_session_stats"}"#),
            PageMessage::PiCommand { id: 6, command: "get_session_stats".into(), args: Value::Null }
        );
        assert_eq!(
            page(r#"{"type":"setModel","id":9,"model":"opencode-go/glm-5.3"}"#),
            PageMessage::SetModel { id: 9, model: "opencode-go/glm-5.3".into() }
        );
        assert_eq!(
            page(r#"{"type":"listFiles","id":10,"query":"src"}"#),
            PageMessage::ListFiles { id: 10, query: "src".into() }
        );
    }

    #[test]
    fn session_messages_parse_and_serialize() {
        assert_eq!(
            page(r#"{"type":"session","session":"a1","message":{"type":"chatSend","text":"hi"}}"#),
            PageMessage::Session {
                session: "a1".into(),
                message: Box::new(PageMessage::ChatSend { text: "hi".into(), behavior: None, attachments: vec![] })
            }
        );
        assert_eq!(page(r#"{"type":"sessionsLoad","id":3}"#), PageMessage::SessionsLoad { id: 3 });
        assert_eq!(
            page(r#"{"type":"sessionsSave","sessions":[{"id":"a1","title":"Fix"}],"active":"a1"}"#),
            PageMessage::SessionsSave {
                sessions: vec![SessionEntry { id: "a1".into(), title: "Fix".into(), segments: vec![], schedule: None }],
                active: Some("a1".into())
            }
        );
        assert_eq!(
            page(r#"{"type":"sessionClose","session":"a1"}"#),
            PageMessage::SessionClose { session: "a1".into() }
        );
        let value = |m: DockMessage| serde_json::to_value(m).unwrap();
        assert_eq!(
            value(DockMessage::Session {
                session: "a1".into(),
                message: Box::new(DockMessage::ChatEngine { state: "ready".into() })
            }),
            json!({"type": "session", "session": "a1", "message": {"type": "chatEngine", "state": "ready"}})
        );
        assert_eq!(
            value(DockMessage::Sessions {
                id: 3,
                sessions: vec![SessionEntry { id: "a1".into(), title: "Fix".into(), segments: vec![], schedule: None }],
                active: None
            }),
            json!({"type": "sessions", "id": 3, "sessions": [{"id": "a1", "title": "Fix", "segments": []}], "active": null})
        );
    }

    #[test]
    fn chat_dock_messages_serialize() {
        let value = |m: DockMessage| serde_json::to_value(m).unwrap();
        assert_eq!(
            value(DockMessage::ChatState {
                workdir: "/home/u/Projects".into(),
                running: false,
                compacting: false,
                model: Some("opencode-go/glm-5.3".into()),
                thinking: None,
                context: Some(30.0),
                problem: None
            }),
            json!({"type": "chatState", "workdir": "/home/u/Projects", "running": false, "compacting": false, "model": "opencode-go/glm-5.3", "thinking": null, "context": 30.0, "problem": null})
        );
        assert_eq!(
            value(DockMessage::ChatEvent { event: json!({"type": "agent_start"}) }),
            json!({"type": "chatEvent", "event": {"type": "agent_start"}})
        );
        assert_eq!(
            value(DockMessage::ChatDialog {
                id: "u1".into(),
                method: "select".into(),
                request: json!({"title": "Allow?"}),
                hidden: true
            }),
            json!({"type": "chatDialog", "id": "u1", "method": "select", "request": {"title": "Allow?"}, "hidden": true})
        );
        assert_eq!(
            value(DockMessage::ChatQuota {
                windows: vec![QuotaWindow {
                    id: "5h".into(),
                    label: "5h".into(),
                    percent: 42.0,
                    resets_at: Some("2026-10-19T03:00:00Z".into())
                }]
            }),
            json!({"type": "chatQuota", "windows": [{"id": "5h", "label": "5h", "percent": 42.0, "resetsAt": "2026-10-19T03:00:00Z"}]})
        );
        assert_eq!(value(DockMessage::ChatAgents { agents: json!([]) }), json!({"type": "chatAgents", "agents": []}));
        assert_eq!(value(DockMessage::SettingsChanged), json!({"type": "settingsChanged"}));
        assert_eq!(
            value(DockMessage::ChatDialogClosed { id: "u1".into() }),
            json!({"type": "chatDialogClosed", "id": "u1"})
        );
        assert_eq!(
            value(DockMessage::ChatEngine { state: "ready".into() }),
            json!({"type": "chatEngine", "state": "ready"})
        );
        assert_eq!(
            value(DockMessage::Commands { id: 7, commands: json!([]) }),
            json!({"type": "commands", "id": 7, "commands": []})
        );
        assert_eq!(
            value(DockMessage::Models { id: 8, models: json!([]) }),
            json!({"type": "models", "id": 8, "models": []})
        );
        assert_eq!(
            value(DockMessage::Files { id: 10, files: vec!["a.rs".into()] }),
            json!({"type": "files", "id": 10, "files": ["a.rs"]})
        );
    }

    #[test]
    fn job_messages_round_trip() {
        assert_eq!(page(r#"{"type":"jobStop","job":"j3"}"#), PageMessage::JobStop { job: "j3".into() });
        assert_eq!(
            page(r#"{"type":"jobOutput","id":4,"job":"j3"}"#),
            PageMessage::JobOutput { id: 4, job: "j3".into(), lines: None }
        );
        assert_eq!(
            page(r#"{"type":"jobOutput","id":5,"job":"j3","lines":200}"#),
            PageMessage::JobOutput { id: 5, job: "j3".into(), lines: Some(200) }
        );
        assert_eq!(page(r#"{"type":"jobDismiss","job":"j3"}"#), PageMessage::JobDismiss { job: "j3".into() });
        let value = |m: DockMessage| serde_json::to_value(m).unwrap();
        assert_eq!(value(DockMessage::Jobs { jobs: json!([]) }), json!({"type": "jobs", "jobs": []}));
        assert_eq!(
            value(DockMessage::JobOutput { id: 4, lines: vec!["ready on :3000".into()] }),
            json!({"type": "jobOutput", "id": 4, "lines": ["ready on :3000"]})
        );
    }

    #[test]
    fn unknown_page_message_is_an_error() {
        assert!(serde_json::from_str::<PageMessage>(r#"{"type":"formatDisk"}"#).is_err());
    }

    #[test]
    fn dock_messages_serialize() {
        let value = |m: DockMessage| serde_json::to_value(m).unwrap();
        assert_eq!(
            value(DockMessage::KeyResult { id: 3, ok: false, error: Some("key-rejected".into()) }),
            json!({"type": "keyResult", "id": 3, "ok": false, "error": "key-rejected"})
        );
        assert_eq!(value(DockMessage::Open { monitor: "DP-2".into() }), json!({"type": "open", "monitor": "DP-2"}));
        assert_eq!(value(DockMessage::Close), json!({"type": "close"}));
        assert_eq!(value(DockMessage::Notice), json!({"type": "notice"}));
        assert_eq!(
            value(DockMessage::Version { version: "0.2.0".into() }),
            json!({"type": "version", "version": "0.2.0"})
        );
        assert_eq!(
            value(DockMessage::Platform {
                hotkey: "Control + Option + Space".into(),
                taken: vec!["Option + Space".into()],
                os: "mac".into()
            }),
            json!({"type": "platform", "hotkey": "Control + Option + Space", "taken": ["Option + Space"], "os": "mac"})
        );
        let linux = |text| crate::shortcut::parse(text, Platform::Linux).unwrap();
        let in_use =
            InUse { shortcut: linux("SUPER + SHIFT + A"), taken: vec![linux("CTRL + ALT + K"), linux("SUPER + A")] };
        assert_eq!(
            value(DockMessage::platform(&in_use, Platform::Linux)),
            json!({"type": "platform", "hotkey": "Super + Shift + A", "taken": ["Ctrl + Alt + K", "Super + A"], "os": "linux"}),
            "the shortcut in use and the taken ones, in the page's form"
        );
        assert_eq!(
            value(DockMessage::Settings {
                id: 1,
                settings: None,
                exists: false,
                problem: None,
                connected: vec!["mcp:tavily".into()],
                servers: vec!["rea".into(), "tavily".into()]
            }),
            json!({"type": "settings", "id": 1, "settings": null, "exists": false, "problem": null, "connected": ["mcp:tavily"], "servers": ["rea", "tavily"]})
        );
        assert_eq!(value(DockMessage::Saved { id: 2 }), json!({"type": "saved", "id": 2}));
        assert_eq!(
            value(DockMessage::McpFound { id: 7, servers: json!([{"id": "cursor:docs"}]) }),
            json!({"type": "mcpFound", "id": 7, "servers": [{"id": "cursor:docs"}]})
        );
        assert_eq!(
            value(DockMessage::OAuthUrl { id: 4, url: "https://x".into() }),
            json!({"type": "oauthUrl", "id": 4, "url": "https://x"})
        );
        assert_eq!(
            value(DockMessage::OAuthResult { id: 4, ok: true, error: None }),
            json!({"type": "oauthResult", "id": 4, "ok": true, "error": null})
        );
        assert_eq!(
            value(DockMessage::Problem { code: "hyprland-classic-config".into() }),
            json!({"type": "problem", "code": "hyprland-classic-config"})
        );
        assert_eq!(
            value(DockMessage::Error { id: 9, code: "unknown".into() }),
            json!({"type": "error", "id": 9, "code": "unknown"})
        );
    }
}
