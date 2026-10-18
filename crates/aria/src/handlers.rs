//! What the dock does with each message from the page: settings, keys, sign-in. `handle` is blocking
//! and replies through `emit`; the caller runs `SaveKey` and `StartOAuth` on a worker thread
//! (`is_blocking`) and moves the replies back to the GTK thread.

use aria_core::auth::AuthStore;
use aria_core::bridge::{DockMessage, PageMessage};
use aria_core::json_file::JsonFileError;
use aria_core::oauth::{self, OAuthError};
use aria_core::paths::AriaPaths;
use aria_core::providers;
use aria_core::settings::{self, Settings, SettingsError};
use aria_core::validate::{check_key, KeyCheck, KeyKind};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

pub type Validator = Arc<dyn Fn(KeyKind, &str) -> KeyCheck + Send + Sync>;
pub type Opener = Arc<dyn Fn(&str) + Send + Sync>;

pub struct Ctx {
    pub paths: AriaPaths,
    pub validator: Validator,
    /// Opens a URL in the user's browser.
    pub opener: Opener,
    /// How long the browser sign-in may take.
    pub oauth_timeout: Duration,
}

impl Ctx {
    pub fn real(paths: AriaPaths) -> Ctx {
        Ctx {
            paths,
            validator: Arc::new(check_key),
            opener: Arc::new(|url| {
                let _ = Command::new("xdg-open").arg(url).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
            }),
            oauth_timeout: Duration::from_secs(300),
        }
    }
}

/// Messages that wait on the network or the browser: run them off the GTK thread.
pub fn is_blocking(message: &PageMessage) -> bool {
    matches!(message, PageMessage::SaveKey { .. } | PageMessage::StartOAuth { .. })
}

pub fn handle(message: PageMessage, ctx: &Ctx, emit: &dyn Fn(DockMessage)) {
    match message {
        PageMessage::Ready | PageMessage::Opened | PageMessage::CloseRequest => {}
        PageMessage::LoadSettings { id } => emit(load_settings(id, ctx)),
        PageMessage::SaveSettings { id, settings } => {
            let _ = ctx.paths.ensure_root();
            emit(save_settings(id, settings, ctx));
        }
        PageMessage::SaveKey { id, provider, key } => {
            let _ = ctx.paths.ensure_root();
            emit(save_key(id, &provider, &key, ctx));
        }
        PageMessage::StartOAuth { id } => {
            let _ = ctx.paths.ensure_root();
            oauth_flow(id, ctx, emit);
        }
        PageMessage::OpenUrl { url } => {
            if url.starts_with("https://") {
                (ctx.opener)(&url);
            }
        }
    }
}

fn load_settings(id: u64, ctx: &Ctx) -> DockMessage {
    match settings::load(&ctx.paths) {
        Ok(Some(settings)) => DockMessage::Settings { id, settings: serde_json::to_value(settings).ok(), exists: true, problem: None },
        Ok(None) => DockMessage::Settings { id, settings: None, exists: false, problem: None },
        Err(error) => {
            let problem = match error {
                SettingsError::File(JsonFileError::Io { .. }) => "settings-unreadable",
                _ => "settings-invalid",
            };
            DockMessage::Settings { id, settings: None, exists: true, problem: Some(problem.into()) }
        }
    }
}

fn save_settings(id: u64, value: serde_json::Value, ctx: &Ctx) -> DockMessage {
    let Ok(settings) = serde_json::from_value::<Settings>(value) else {
        return DockMessage::Error { id, code: "settings-invalid".into() };
    };
    match settings::save(&ctx.paths, &settings) {
        Ok(()) => DockMessage::Saved { id },
        Err(SettingsError::File(JsonFileError::Invalid { .. })) | Err(SettingsError::Invalid(_)) => {
            DockMessage::Error { id, code: "settings-invalid".into() }
        }
        Err(_) => DockMessage::Error { id, code: "settings-unwritable".into() },
    }
}

/// `opencode-go` | `anthropic-api` | `typesafe` -> how to check the key and where `auth.json` keeps it.
fn key_target(provider: &str) -> Option<(KeyKind, Vec<&'static str>)> {
    match provider {
        "opencode-go" | "anthropic-api" => {
            let kind = if provider == "opencode-go" { KeyKind::OpenCodeGo } else { KeyKind::AnthropicApi };
            providers::provider(provider).map(|module| (kind, module.auth_entries.to_vec()))
        }
        "typesafe" => Some((KeyKind::TypeSafe, vec!["typesafe"])),
        _ => None,
    }
}

fn key_result(id: u64, error: Option<&str>) -> DockMessage {
    DockMessage::KeyResult { id, ok: error.is_none(), error: error.map(str::to_string) }
}

/// Checks the key against its provider and only then stores it; a key that is rejected or cannot be
/// checked is never written.
fn save_key(id: u64, provider: &str, key: &str, ctx: &Ctx) -> DockMessage {
    let Some((kind, entries)) = key_target(provider) else {
        return DockMessage::Error { id, code: "unknown-provider".into() };
    };
    let key = key.trim();
    if key.is_empty() {
        return key_result(id, Some("key-rejected"));
    }
    match (ctx.validator)(kind, key) {
        KeyCheck::Valid => {}
        KeyCheck::Rejected => return key_result(id, Some("key-rejected")),
        KeyCheck::Unavailable(_) => return key_result(id, Some("network-unavailable")),
    }
    let store = AuthStore::new(ctx.paths.auth());
    for entry in entries {
        if store.set(entry, &aria_core::auth::Credential::ApiKey { key: key.to_string() }).is_err() {
            return key_result(id, Some("auth-file-broken"));
        }
    }
    key_result(id, None)
}

fn oauth_error_key(error: &OAuthError) -> &'static str {
    match error {
        OAuthError::StateMismatch => "oauth-state",
        OAuthError::Denied(_) => "oauth-denied",
        OAuthError::Timeout => "oauth-timeout",
        OAuthError::Network(_) => "network-unavailable",
        OAuthError::NoCode | OAuthError::BadToken(_) | OAuthError::Io(_) => "oauth-failed",
    }
}

fn oauth_flow(id: u64, ctx: &Ctx, emit: &dyn Fn(DockMessage)) {
    let result = |error: Option<&str>| DockMessage::OAuthResult { id, ok: error.is_none(), error: error.map(str::to_string) };
    let sign_in = match oauth::anthropic::start() {
        Ok(sign_in) => sign_in,
        Err(error) => return emit(result(Some(oauth_error_key(&error)))),
    };
    emit(DockMessage::OAuthUrl { id, url: sign_in.url.clone() });
    (ctx.opener)(&sign_in.url);
    match sign_in.wait(ctx.oauth_timeout) {
        Ok(credential) => match AuthStore::new(ctx.paths.auth()).set("anthropic", &credential) {
            Ok(()) => emit(result(None)),
            Err(_) => emit(result(Some("auth-file-broken"))),
        },
        Err(error) => emit(result(Some(oauth_error_key(&error)))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::auth::{AuthStore, Credential};
    use aria_core::validate::{KeyCheck, KeyKind};
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct Harness {
        dir: tempfile::TempDir,
        ctx: Ctx,
        opened: Arc<Mutex<Vec<String>>>,
        seen_keys: Arc<Mutex<Vec<(KeyKind, String)>>>,
    }

    fn harness(verdict: KeyCheck) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let opened = Arc::new(Mutex::new(Vec::new()));
        let seen_keys = Arc::new(Mutex::new(Vec::new()));
        let (o, s) = (opened.clone(), seen_keys.clone());
        let ctx = Ctx {
            paths: AriaPaths { root: dir.path().to_path_buf() },
            validator: Arc::new(move |kind, key| {
                s.lock().unwrap().push((kind, key.to_string()));
                verdict.clone()
            }),
            opener: Arc::new(move |url| o.lock().unwrap().push(url.to_string())),
            oauth_timeout: Duration::from_millis(150),
        };
        Harness { dir, ctx, opened, seen_keys }
    }

    fn run(h: &Harness, message: PageMessage) -> Vec<DockMessage> {
        let out = Mutex::new(Vec::new());
        handle(message, &h.ctx, &|m| out.lock().unwrap().push(m));
        out.into_inner().unwrap()
    }

    fn auth(h: &Harness) -> AuthStore {
        AuthStore::new(h.ctx.paths.auth())
    }

    #[test]
    fn load_settings_when_missing() {
        let h = harness(KeyCheck::Valid);
        assert_eq!(
            run(&h, PageMessage::LoadSettings { id: 1 }),
            vec![DockMessage::Settings { id: 1, settings: None, exists: false, problem: None }]
        );
    }

    #[test]
    fn load_settings_broken() {
        let h = harness(KeyCheck::Valid);
        std::fs::write(h.ctx.paths.settings(), "{ nope").unwrap();
        assert_eq!(
            run(&h, PageMessage::LoadSettings { id: 1 }),
            vec![DockMessage::Settings { id: 1, settings: None, exists: true, problem: Some("settings-invalid".into()) }]
        );
    }

    #[test]
    fn save_settings_round_trip() {
        let h = harness(KeyCheck::Valid);
        let mut settings = aria_core::settings::Settings::defaults("es");
        settings.assistant.user_name = "Ivan".into();
        let value = serde_json::to_value(&settings).unwrap();
        assert_eq!(run(&h, PageMessage::SaveSettings { id: 2, settings: value.clone() }), vec![DockMessage::Saved { id: 2 }]);
        assert_eq!(
            run(&h, PageMessage::LoadSettings { id: 3 }),
            vec![DockMessage::Settings { id: 3, settings: Some(value), exists: true, problem: None }]
        );
    }

    #[test]
    fn save_settings_with_the_wrong_shape_is_an_error_and_writes_nothing() {
        let h = harness(KeyCheck::Valid);
        let replies = run(&h, PageMessage::SaveSettings { id: 4, settings: json!({"compaction": {"percent": "lots"}}) });
        assert_eq!(replies, vec![DockMessage::Error { id: 4, code: "settings-invalid".into() }]);
        assert!(!h.ctx.paths.settings().exists());
    }

    #[test]
    fn save_settings_does_not_replace_a_broken_file() {
        let h = harness(KeyCheck::Valid);
        std::fs::write(h.ctx.paths.settings(), "{ nope").unwrap();
        let value = serde_json::to_value(aria_core::settings::Settings::defaults("en")).unwrap();
        let replies = run(&h, PageMessage::SaveSettings { id: 5, settings: value });
        assert_eq!(replies, vec![DockMessage::Error { id: 5, code: "settings-invalid".into() }]);
        assert_eq!(std::fs::read_to_string(h.ctx.paths.settings()).unwrap(), "{ nope");
    }

    #[test]
    fn opencode_key_writes_both_entries() {
        let h = harness(KeyCheck::Valid);
        let replies = run(&h, PageMessage::SaveKey { id: 6, provider: "opencode-go".into(), key: "k1".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 6, ok: true, error: None }]);
        assert_eq!(h.seen_keys.lock().unwrap().as_slice(), [(KeyKind::OpenCodeGo, "k1".to_string())]);
        let store = auth(&h);
        assert_eq!(store.get("opencode-go").unwrap(), Some(Credential::ApiKey { key: "k1".into() }));
        assert_eq!(store.get("opencode").unwrap(), Some(Credential::ApiKey { key: "k1".into() }));
    }

    #[test]
    fn other_keys_land_under_their_own_entries() {
        let h = harness(KeyCheck::Valid);
        run(&h, PageMessage::SaveKey { id: 1, provider: "anthropic-api".into(), key: "a".into() });
        run(&h, PageMessage::SaveKey { id: 2, provider: "typesafe".into(), key: "t".into() });
        let store = auth(&h);
        assert_eq!(store.get("anthropic").unwrap(), Some(Credential::ApiKey { key: "a".into() }));
        assert_eq!(store.get("typesafe").unwrap(), Some(Credential::ApiKey { key: "t".into() }));
        assert!(!store.has("opencode-go"));
    }

    #[test]
    fn empty_key_is_rejected_without_a_request() {
        let h = harness(KeyCheck::Valid);
        let replies = run(&h, PageMessage::SaveKey { id: 12, provider: "opencode-go".into(), key: "   ".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 12, ok: false, error: Some("key-rejected".into()) }]);
        assert!(h.seen_keys.lock().unwrap().is_empty());
    }

    #[test]
    fn rejected_key_writes_nothing() {
        let h = harness(KeyCheck::Rejected);
        let replies = run(&h, PageMessage::SaveKey { id: 7, provider: "opencode-go".into(), key: "bad".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 7, ok: false, error: Some("key-rejected".into()) }]);
        assert!(!h.ctx.paths.auth().exists());
    }

    #[test]
    fn unreachable_provider_writes_nothing() {
        let h = harness(KeyCheck::Unavailable("timed out".into()));
        let replies = run(&h, PageMessage::SaveKey { id: 8, provider: "typesafe".into(), key: "k".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 8, ok: false, error: Some("network-unavailable".into()) }]);
        assert!(!h.ctx.paths.auth().exists());
    }

    #[test]
    fn broken_auth_file_is_reported_and_kept() {
        let h = harness(KeyCheck::Valid);
        std::fs::write(h.ctx.paths.auth(), "{ nope").unwrap();
        let replies = run(&h, PageMessage::SaveKey { id: 9, provider: "opencode-go".into(), key: "k".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 9, ok: false, error: Some("auth-file-broken".into()) }]);
        assert_eq!(std::fs::read_to_string(h.ctx.paths.auth()).unwrap(), "{ nope");
    }

    #[test]
    fn unknown_provider_is_an_error() {
        let h = harness(KeyCheck::Valid);
        let replies = run(&h, PageMessage::SaveKey { id: 10, provider: "mystery".into(), key: "k".into() });
        assert_eq!(replies, vec![DockMessage::Error { id: 10, code: "unknown-provider".into() }]);
        assert!(h.seen_keys.lock().unwrap().is_empty());
    }

    #[test]
    fn abandoned_oauth_times_out_and_writes_nothing() {
        let h = harness(KeyCheck::Valid);
        let replies = run(&h, PageMessage::StartOAuth { id: 11 });
        assert_eq!(replies.len(), 2);
        match &replies[0] {
            DockMessage::OAuthUrl { id: 11, url } => {
                assert!(url.starts_with("https://claude.ai/oauth/authorize?"));
                assert_eq!(h.opened.lock().unwrap().as_slice(), [url.clone()]);
            }
            other => panic!("expected the sign-in URL first, got {other:?}"),
        }
        assert_eq!(replies[1], DockMessage::OAuthResult { id: 11, ok: false, error: Some("oauth-timeout".into()) });
        assert!(!h.ctx.paths.auth().exists());
    }

    #[test]
    fn only_https_urls_are_opened() {
        let h = harness(KeyCheck::Valid);
        run(&h, PageMessage::OpenUrl { url: "file:///etc/passwd".into() });
        run(&h, PageMessage::OpenUrl { url: "javascript:alert(1)".into() });
        assert!(h.opened.lock().unwrap().is_empty());
        run(&h, PageMessage::OpenUrl { url: "https://console.typesafe.ai/keys".into() });
        assert_eq!(h.opened.lock().unwrap().as_slice(), ["https://console.typesafe.ai/keys".to_string()]);
    }

    #[test]
    fn page_messages_the_dock_owns_are_ignored_here() {
        let h = harness(KeyCheck::Valid);
        assert!(run(&h, PageMessage::Ready).is_empty());
        assert!(!h.dir.path().join("settings.json").exists());
    }
}
