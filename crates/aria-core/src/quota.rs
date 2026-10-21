//! How much of the provider's quota is spent, for the model in use: OpenCode Go's 5-hour, weekly and monthly windows
//! (`GET /zen/go/v1/usage` with the Go key) and a Claude plan's 5-hour and weekly windows, plus any per-model weekly one
//! such as Fable (`GET /api/oauth/usage` with the sign-in's token). An API key has no quota to read; nothing here ever
//! fails loudly: no source or a bad answer is just "no bars".

use crate::auth::{AuthStore, Credential};
use crate::validate::ProbeRequest;
use serde::Serialize;
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Window {
    /// `5h`, `week`, `month`, or the model's name for a per-model weekly window (`fable`, `opus`).
    pub id: String,
    pub label: String,
    /// 0 to 100 (not clamped here).
    pub percent: f64,
    /// When the window resets (an ISO time as the provider sent it), when it said.
    #[serde(rename = "resetsAt")]
    pub resets_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    OpenCodeGo { key: String },
    Claude { token: String },
}

/// The model is `provider/id`. A Claude plan is read through the sign-in's token (`anthropic` credential of type OAuth,
/// not expired); an API key, OpenCode Zen and anything unknown have no quota to read.
pub fn source_for(model: &str, auth: &AuthStore, now_ms: i64) -> Option<Source> {
    let provider = model.split('/').next().unwrap_or_default();
    match provider {
        "opencode-go" => match auth.get("opencode-go").ok().flatten()? {
            // A key the keyring could not give back is still its command: no reading then.
            Credential::ApiKey { key } if !key.trim().is_empty() && !key.starts_with('!') => {
                Some(Source::OpenCodeGo { key: key.trim().to_string() })
            }
            _ => None,
        },
        "anthropic" | "claude-bridge" => match auth.get("anthropic").ok().flatten()? {
            Credential::OAuth { access, expires, .. } if expires > now_ms => Some(Source::Claude { token: access }),
            _ => None,
        },
        _ => None,
    }
}

pub fn request_for(source: &Source) -> ProbeRequest {
    match source {
        Source::OpenCodeGo { key } => ProbeRequest {
            method: "GET",
            url: "https://opencode.ai/zen/go/v1/usage".into(),
            headers: vec![("authorization".into(), format!("Bearer {key}"))],
            body: None,
        },
        Source::Claude { token } => ProbeRequest {
            method: "GET",
            url: "https://api.anthropic.com/api/oauth/usage".into(),
            headers: vec![
                ("authorization".into(), format!("Bearer {token}")),
                ("anthropic-beta".into(), "oauth-2025-04-20".into()),
            ],
            body: None,
        },
    }
}

fn capitalised(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map(|c| c.to_uppercase().collect::<String>() + chars.as_str()).unwrap_or_default()
}

/// `{ usage: { rolling|weekly|monthly: { status, percent, resetsAt } } }`.
pub fn parse_go_usage(body: &Value) -> Option<Vec<Window>> {
    let usage = body.get("usage")?.as_object()?;
    let mut windows = Vec::new();
    for (key, id, label) in [("rolling", "5h", "5h"), ("weekly", "week", "Week"), ("monthly", "month", "Month")] {
        let Some(window) = usage.get(key) else { continue };
        let percent = window.get("percent").and_then(Value::as_f64)?;
        windows.push(Window {
            id: id.into(),
            label: label.into(),
            percent,
            resets_at: window.get("resetsAt").and_then(Value::as_str).map(str::to_string),
        });
    }
    (!windows.is_empty()).then_some(windows)
}

/// `{ five_hour: { utilization, resets_at }, seven_day: {…}, seven_day_<model>: {…} | null }`; every non-null
/// `seven_day_<model>` becomes a window named after the model.
pub fn parse_claude_usage(body: &Value) -> Option<Vec<Window>> {
    let object = body.as_object()?;
    let read = |id: &str, label: &str, value: &Value| -> Option<Window> {
        let percent = value.get("utilization").and_then(Value::as_f64)?;
        Some(Window {
            id: id.into(),
            label: label.into(),
            percent,
            resets_at: value.get("resets_at").and_then(Value::as_str).map(str::to_string),
        })
    };
    let mut windows = Vec::new();
    windows.extend(object.get("five_hour").and_then(|v| read("5h", "5h", v)));
    windows.extend(object.get("seven_day").and_then(|v| read("week", "Week", v)));
    let mut models: Vec<_> =
        object.iter().filter_map(|(key, value)| key.strip_prefix("seven_day_").map(|name| (name, value))).collect();
    models.sort_by_key(|(name, _)| *name);
    for (name, value) in models {
        if value.is_null() {
            continue;
        }
        windows.extend(read(name, &capitalised(name), value));
    }
    (!windows.is_empty()).then_some(windows)
}

pub fn parse(source: &Source, body: &Value) -> Option<Vec<Window>> {
    match source {
        Source::OpenCodeGo { .. } => parse_go_usage(body),
        Source::Claude { .. } => parse_claude_usage(body),
    }
}

/// Sends a request and returns the JSON answer; `None` for anything but a good answer.
pub type Fetcher = dyn Fn(&ProbeRequest) -> Option<Value> + Send + Sync;

pub fn fetch(source: &Source, fetcher: &Fetcher) -> Option<Vec<Window>> {
    parse(source, &fetcher(&request_for(source))?)
}

/// The real thing: two seconds at most, and the token is never logged.
pub fn real_fetcher(request: &ProbeRequest) -> Option<Value> {
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(3)).build();
    let mut call = agent.get(&request.url);
    for (name, value) in &request.headers {
        call = call.set(name, value);
    }
    call.call().ok()?.into_json().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::AriaPaths;
    use serde_json::json;

    fn auth_with(entries: Value) -> (tempfile::TempDir, AuthStore) {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().to_path_buf());
        std::fs::write(paths.auth(), entries.to_string()).unwrap();
        let store = AuthStore::new(paths.auth());
        (dir, store)
    }

    #[test]
    fn the_go_source_reads_the_real_key_through_the_keyring() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, _keyring) = crate::testing::vault_paths(dir.path());
        let store = AuthStore::for_paths(&paths);
        store.set("opencode-go", &Credential::ApiKey { key: "real-key".into() }).unwrap();
        assert_eq!(source_for("opencode-go/glm", &store, 0), Some(Source::OpenCodeGo { key: "real-key".into() }));
    }

    #[test]
    fn go_usage_gives_the_three_windows() {
        let body = json!({"usage": {
            "rolling": {"status": "ok", "percent": 42, "resetsAt": "2026-10-19T03:00:00Z"},
            "weekly": {"status": "ok", "percent": 17.5},
            "monthly": {"status": "ok", "percent": 6}
        }});
        let windows = parse_go_usage(&body).unwrap();
        assert_eq!(
            windows.iter().map(|w| (w.id.as_str(), w.label.as_str(), w.percent)).collect::<Vec<_>>(),
            vec![("5h", "5h", 42.0), ("week", "Week", 17.5), ("month", "Month", 6.0)]
        );
        assert_eq!(windows[0].resets_at.as_deref(), Some("2026-10-19T03:00:00Z"));
        assert_eq!(windows[1].resets_at, None);
    }

    #[test]
    fn go_usage_with_a_missing_window_shows_the_others_and_a_broken_one_shows_nothing() {
        assert_eq!(parse_go_usage(&json!({"usage": {"rolling": {"percent": 1}}})).unwrap().len(), 1);
        assert_eq!(parse_go_usage(&json!({"usage": {"rolling": {"percent": "lots"}}})), None);
        assert_eq!(parse_go_usage(&json!({"usage": {}})), None);
        assert_eq!(parse_go_usage(&json!({"nope": 1})), None);
        assert_eq!(parse_go_usage(&json!(null)), None);
    }

    #[test]
    fn claude_usage_has_5h_and_week_and_a_window_per_model_when_there_is_one() {
        let body = json!({
            "five_hour": {"utilization": 61.0, "resets_at": "2026-10-19T03:00:00Z"},
            "seven_day": {"utilization": 23.0, "resets_at": "2026-10-24T00:00:00Z"},
            "seven_day_opus": null,
            "seven_day_fable": {"utilization": 88.0, "resets_at": "2026-10-24T00:00:00Z"}
        });
        let windows = parse_claude_usage(&body).unwrap();
        assert_eq!(
            windows.iter().map(|w| (w.id.as_str(), w.label.as_str(), w.percent)).collect::<Vec<_>>(),
            vec![("5h", "5h", 61.0), ("week", "Week", 23.0), ("fable", "Fable", 88.0)]
        );
    }

    #[test]
    fn claude_usage_without_per_model_windows_and_garbage() {
        let windows = parse_claude_usage(
            &json!({"five_hour": {"utilization": 5}, "seven_day": {"utilization": 1}, "seven_day_opus": null}),
        )
        .unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(parse_claude_usage(&json!({"five_hour": {"utilization": "x"}})), None);
        assert_eq!(parse_claude_usage(&json!([1])), None);
    }

    #[test]
    fn the_source_follows_the_model_and_the_credential() {
        let (_d, auth) = auth_with(json!({
            "opencode-go": {"type": "api_key", "key": " go-key "},
            "anthropic": {"type": "oauth", "access": "tok", "refresh": "r", "expires": 2_000}
        }));
        assert_eq!(source_for("opencode-go/glm-5.3", &auth, 1_000), Some(Source::OpenCodeGo { key: "go-key".into() }));
        assert_eq!(
            source_for("anthropic/claude-sonnet-5-5", &auth, 1_000),
            Some(Source::Claude { token: "tok".into() })
        );
        assert_eq!(
            source_for("claude-bridge/claude-opus-5-5", &auth, 1_000),
            Some(Source::Claude { token: "tok".into() })
        );
        assert_eq!(source_for("anthropic/claude-sonnet-5-5", &auth, 3_000), None, "an expired token is not used");
        assert_eq!(source_for("opencode/big-pickle", &auth, 1_000), None);
        assert_eq!(source_for("nonsense", &auth, 1_000), None);
    }

    #[test]
    fn an_api_key_for_anthropic_or_no_credential_has_no_quota_to_read() {
        let (_d, auth) = auth_with(json!({"anthropic": {"type": "api_key", "key": "sk-ant-x"}}));
        assert_eq!(source_for("anthropic/claude-sonnet-5-5", &auth, 0), None);
        assert_eq!(source_for("opencode-go/glm-5.3", &auth, 0), None);
        let (_d2, empty) = auth_with(json!({}));
        assert_eq!(source_for("anthropic/x", &empty, 0), None);
    }

    #[test]
    fn the_requests_carry_the_token_and_the_beta_header() {
        let go = request_for(&Source::OpenCodeGo { key: "k".into() });
        assert_eq!((go.method, go.url.as_str()), ("GET", "https://opencode.ai/zen/go/v1/usage"));
        assert!(go.headers.contains(&("authorization".into(), "Bearer k".into())));
        let claude = request_for(&Source::Claude { token: "t".into() });
        assert_eq!(claude.url, "https://api.anthropic.com/api/oauth/usage");
        assert!(claude.headers.contains(&("anthropic-beta".into(), "oauth-2025-04-20".into())));
    }

    #[test]
    fn fetch_goes_through_the_fetcher_and_any_failure_is_none() {
        let source = Source::OpenCodeGo { key: "k".into() };
        let ok: Box<Fetcher> = Box::new(|_| Some(json!({"usage": {"rolling": {"percent": 9}}})));
        assert_eq!(fetch(&source, &*ok).unwrap()[0].percent, 9.0);
        let down: Box<Fetcher> = Box::new(|_| None);
        assert_eq!(fetch(&source, &*down), None);
        let junk: Box<Fetcher> = Box::new(|_| Some(json!("<html>")));
        assert_eq!(fetch(&source, &*junk), None);
    }
}
