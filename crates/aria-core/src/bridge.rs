//! The messages between the dock (Rust) and its page (the web UI), one JSON object per message with
//! a `type` tag. The page sends `PageMessage`s; the dock answers with `DockMessage`s. Requests that
//! expect an answer carry an `id` that the answer repeats.

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
    #[serde(rename = "loadSettings")]
    LoadSettings { id: u64 },
    #[serde(rename = "saveSettings")]
    SaveSettings { id: u64, settings: Value },
    /// `provider` is `opencode-go`, `anthropic-api` or `typesafe`.
    #[serde(rename = "saveKey")]
    SaveKey { id: u64, provider: String, key: String },
    /// Start the Anthropic OAuth sign-in.
    #[serde(rename = "startOauth")]
    StartOAuth { id: u64 },
    #[serde(rename = "openUrl")]
    OpenUrl { url: String },
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
    #[serde(rename = "settings")]
    Settings { id: u64, settings: Option<Value>, exists: bool, problem: Option<String> },
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
        assert_eq!(page(r#"{"type":"openUrl","url":"https://x"}"#), PageMessage::OpenUrl { url: "https://x".into() });
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
            value(DockMessage::Settings { id: 1, settings: None, exists: false, problem: None }),
            json!({"type": "settings", "id": 1, "settings": null, "exists": false, "problem": null})
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
