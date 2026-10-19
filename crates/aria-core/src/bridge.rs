//! The messages between the dock (Rust) and its page (the web UI), one JSON object per message with
//! a `type` tag. The page sends `PageMessage`s; the dock answers with `DockMessage`s. Requests that
//! expect an answer carry an `id` that the answer repeats.

use serde::{Deserialize, Serialize};
use crate::sessions::SessionEntry;
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
    #[serde(rename = "loadSettings")]
    LoadSettings { id: u64 },
    #[serde(rename = "saveSettings")]
    SaveSettings { id: u64, settings: Value },
    /// `provider` is `opencode-go`, `anthropic-api`, `typesafe` or `tavily`.
    #[serde(rename = "saveKey")]
    SaveKey { id: u64, provider: String, key: String },
    /// Start the Anthropic OAuth sign-in.
    #[serde(rename = "startOauth")]
    StartOAuth { id: u64 },
    /// Start an MCP server's OAuth sign-in (`service` is `notion`).
    #[serde(rename = "startMcpOauth")]
    StartMcpOAuth { id: u64, service: String },
    #[serde(rename = "openUrl")]
    OpenUrl { url: String },
    /// A chat message. `behavior` is `steer` or `followUp` when a run is already going.
    #[serde(rename = "chatSend")]
    ChatSend { text: String, behavior: Option<String> },
    #[serde(rename = "chatAbort")]
    ChatAbort,
    /// The dock opened: tell the page where Aria works and how it is doing.
    #[serde(rename = "chatLoad")]
    ChatLoad,
    /// Cancel a queued message: Pi can only empty its queue, so the messages in `keep` are queued again, in order.
    #[serde(rename = "chatClearQueue")]
    ChatClearQueue {
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
    /// `connected` names what is really set up, never a secret: `provider:type` for each credential in
    /// `auth.json` (`anthropic:oauth`) and `mcp:<service>` for each configured MCP service.
    #[serde(rename = "settings")]
    Settings { id: u64, settings: Option<Value>, exists: bool, problem: Option<String>, connected: Vec<String> },
    #[serde(rename = "saved")]
    Saved { id: u64 },
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
    ChatState { workdir: String, running: bool, model: Option<String>, thinking: Option<String>, context: Option<f64>, problem: Option<String> },
    /// A Pi session event, verbatim.
    #[serde(rename = "chatEvent")]
    ChatEvent { event: Value },
    /// An extension dialog; `hidden` says the dock was not visible when it arrived.
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
    /// What a session says (its chat messages and the replies to its requests).
    #[serde(rename = "session")]
    Session { session: String, message: Box<DockMessage> },
    #[serde(rename = "sessions")]
    Sessions { id: u64, sessions: Vec<SessionEntry>, active: Option<String> },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn page(text: &str) -> PageMessage {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn page_messages_parse() {
        assert_eq!(page(r#"{"type":"ready"}"#), PageMessage::Ready);
        assert_eq!(page(r#"{"type":"opened"}"#), PageMessage::Opened);
        assert_eq!(page(r#"{"type":"closeRequest"}"#), PageMessage::CloseRequest);
        assert_eq!(page(r#"{"type":"loadSettings","id":1}"#), PageMessage::LoadSettings { id: 1 });
        assert_eq!(
            page(r#"{"type":"saveSettings","id":2,"settings":{"a":1}}"#),
            PageMessage::SaveSettings { id: 2, settings: json!({"a": 1}) }
        );
        assert_eq!(
            page(r#"{"type":"saveKey","id":3,"provider":"typesafe","key":"k"}"#),
            PageMessage::SaveKey { id: 3, provider: "typesafe".into(), key: "k".into() }
        );
        assert_eq!(page(r#"{"type":"startOauth","id":4}"#), PageMessage::StartOAuth { id: 4 });
        assert_eq!(
            page(r#"{"type":"startMcpOauth","id":5,"service":"notion"}"#),
            PageMessage::StartMcpOAuth { id: 5, service: "notion".into() }
        );
        assert_eq!(page(r#"{"type":"openUrl","url":"https://x"}"#), PageMessage::OpenUrl { url: "https://x".into() });
    }

    #[test]
    fn chat_page_messages_parse() {
        assert_eq!(
            page(r#"{"type":"chatSend","text":"hi","behavior":"steer"}"#),
            PageMessage::ChatSend { text: "hi".into(), behavior: Some("steer".into()) }
        );
        assert_eq!(page(r#"{"type":"chatSend","text":"hi"}"#), PageMessage::ChatSend { text: "hi".into(), behavior: None });
        assert_eq!(page(r#"{"type":"chatAbort"}"#), PageMessage::ChatAbort);
        assert_eq!(page(r#"{"type":"chatLoad"}"#), PageMessage::ChatLoad);
        assert_eq!(page(r#"{"type":"chatClearQueue"}"#), PageMessage::ChatClearQueue { keep: vec![] });
        assert_eq!(
            page(r#"{"type":"chatClearQueue","keep":["a","b"]}"#),
            PageMessage::ChatClearQueue { keep: vec!["a".into(), "b".into()] }
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
            PageMessage::Session { session: "a1".into(), message: Box::new(PageMessage::ChatSend { text: "hi".into(), behavior: None }) }
        );
        assert_eq!(page(r#"{"type":"sessionsLoad","id":3}"#), PageMessage::SessionsLoad { id: 3 });
        assert_eq!(
            page(r#"{"type":"sessionsSave","sessions":[{"id":"a1","title":"Fix"}],"active":"a1"}"#),
            PageMessage::SessionsSave { sessions: vec![SessionEntry { id: "a1".into(), title: "Fix".into() }], active: Some("a1".into()) }
        );
        assert_eq!(page(r#"{"type":"sessionClose","session":"a1"}"#), PageMessage::SessionClose { session: "a1".into() });
        let value = |m: DockMessage| serde_json::to_value(m).unwrap();
        assert_eq!(
            value(DockMessage::Session { session: "a1".into(), message: Box::new(DockMessage::ChatEngine { state: "ready".into() }) }),
            json!({"type": "session", "session": "a1", "message": {"type": "chatEngine", "state": "ready"}})
        );
        assert_eq!(
            value(DockMessage::Sessions { id: 3, sessions: vec![SessionEntry { id: "a1".into(), title: "Fix".into() }], active: None }),
            json!({"type": "sessions", "id": 3, "sessions": [{"id": "a1", "title": "Fix"}], "active": null})
        );
    }

    #[test]
    fn chat_dock_messages_serialize() {
        let value = |m: DockMessage| serde_json::to_value(m).unwrap();
        assert_eq!(
            value(DockMessage::ChatState { workdir: "/home/u/Projects".into(), running: false, model: Some("opencode-go/glm-5.3".into()), thinking: None, context: Some(30.0), problem: None }),
            json!({"type": "chatState", "workdir": "/home/u/Projects", "running": false, "model": "opencode-go/glm-5.3", "thinking": null, "context": 30.0, "problem": null})
        );
        assert_eq!(
            value(DockMessage::ChatEvent { event: json!({"type": "agent_start"}) }),
            json!({"type": "chatEvent", "event": {"type": "agent_start"}})
        );
        assert_eq!(
            value(DockMessage::ChatDialog { id: "u1".into(), method: "select".into(), request: json!({"title": "Allow?"}), hidden: true }),
            json!({"type": "chatDialog", "id": "u1", "method": "select", "request": {"title": "Allow?"}, "hidden": true})
        );
        assert_eq!(value(DockMessage::ChatDialogClosed { id: "u1".into() }), json!({"type": "chatDialogClosed", "id": "u1"}));
        assert_eq!(value(DockMessage::ChatEngine { state: "ready".into() }), json!({"type": "chatEngine", "state": "ready"}));
        assert_eq!(value(DockMessage::Commands { id: 7, commands: json!([]) }), json!({"type": "commands", "id": 7, "commands": []}));
        assert_eq!(value(DockMessage::Models { id: 8, models: json!([]) }), json!({"type": "models", "id": 8, "models": []}));
        assert_eq!(value(DockMessage::Files { id: 10, files: vec!["a.rs".into()] }), json!({"type": "files", "id": 10, "files": ["a.rs"]}));
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
            value(DockMessage::Settings { id: 1, settings: None, exists: false, problem: None, connected: vec!["mcp:tavily".into()] }),
            json!({"type": "settings", "id": 1, "settings": null, "exists": false, "problem": null, "connected": ["mcp:tavily"]})
        );
        assert_eq!(value(DockMessage::Saved { id: 2 }), json!({"type": "saved", "id": 2}));
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
        assert_eq!(value(DockMessage::Error { id: 9, code: "unknown".into() }), json!({"type": "error", "id": 9, "code": "unknown"}));
    }
}
