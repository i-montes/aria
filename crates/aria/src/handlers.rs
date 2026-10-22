//! What the dock does with each message from the page: settings, keys, sign-in. `handle` is blocking
//! and replies through `emit`; the caller runs `SaveKey` and `StartOAuth` on a worker thread
//! (`is_blocking`) and moves the replies back to the GTK thread.

use aria_core::agent::{self, InstallCommand};
use aria_core::auth::{AuthStore, Credential};
use aria_core::bridge::{DockMessage, PageMessage};
use aria_core::json_file::JsonFileError;
use aria_core::mcp::{self, McpError};
use aria_core::mcp_import::{self, ImportError};
use aria_core::oauth::{self, OAuthError};
use aria_core::paths::AriaPaths;
use aria_core::providers;
use aria_core::settings::{self, Settings, SettingsError};
use aria_core::validate::{check_key, KeyCheck, KeyKind, ProbeRequest};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

/// The program that opens a link in the user's browser.
#[cfg(target_os = "linux")]
const OPENER: &str = "xdg-open";
#[cfg(target_os = "macos")]
const OPENER: &str = "open";

pub type Validator = Arc<dyn Fn(KeyKind, &str) -> KeyCheck + Send + Sync>;
pub type Opener = Arc<dyn Fn(&str) + Send + Sync>;
/// Checks a stored OAuth credential's token against its provider (the Claude usage endpoint).
pub type OAuthChecker = Arc<dyn Fn(&ProbeRequest) -> KeyCheck + Send + Sync>;
/// Runs `pi install` for the Claude provider; true when it exited well.
pub type Installer = Arc<dyn Fn(&InstallCommand) -> bool + Send + Sync>;

pub struct Ctx {
    pub paths: AriaPaths,
    pub validator: Validator,
    /// Checks a stored OAuth token (Claude) without a browser.
    pub oauth_checker: OAuthChecker,
    /// Opens a URL in the user's browser.
    pub opener: Opener,
    /// How long the browser sign-in may take.
    pub oauth_timeout: Duration,
    /// Notion's MCP address (a parameter so tests can point it at a fake server).
    pub notion_url: String,
    pub installer: Installer,
    /// The `PATH` Pi would inherit (to find Claude Code and Node).
    pub path_var: String,
    /// The person's home, where the other apps keep their MCP servers (empty without `$HOME`: nothing is found).
    pub home: PathBuf,
    /// Where Claude Desktop and VS Code keep theirs on Linux (`$XDG_CONFIG_HOME` or `~/.config`).
    pub config_dir: PathBuf,
    /// Every session's Pi starts again when its run settles (`enginesRestart`); the dock wires it to the chat.
    pub restart_engines: Arc<dyn Fn() + Send + Sync>,
}

impl Ctx {
    pub fn real(paths: AriaPaths) -> Ctx {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        Ctx {
            paths,
            validator: Arc::new(check_key),
            oauth_checker: Arc::new(aria_core::validate::check),
            opener: Arc::new(|url| {
                let _ = Command::new(OPENER)
                    .arg(url)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn();
            }),
            oauth_timeout: Duration::from_secs(300),
            notion_url: mcp::NOTION_URL.to_string(),
            installer: Arc::new(|command| {
                let mut run = Command::new(&command.program);
                for name in &command.unset {
                    run.env_remove(name);
                }
                run.args(&command.args)
                    .envs(command.env.iter().map(|(k, v)| (k, v)))
                    .current_dir(&command.cwd)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success())
            }),
            path_var: std::env::var("PATH").unwrap_or_default(),
            home: home.clone(),
            config_dir: mcp_import::config_dir(&home, std::env::var_os("XDG_CONFIG_HOME").as_deref()),
            restart_engines: Arc::new(|| {}),
        }
    }
}

/// Messages that wait on the network or the browser: run them off the GTK thread.
pub fn is_blocking(message: &PageMessage) -> bool {
    matches!(
        message,
        PageMessage::SaveKey { .. }
            | PageMessage::CheckKey { .. }
            | PageMessage::StartOAuth { .. }
            | PageMessage::StartMcpOAuth { .. }
            | PageMessage::InstallClaudeBridge { .. }
            | PageMessage::McpScan { .. }
            | PageMessage::McpImport { .. }
    )
}

/// Messages for the chat (`chat::Chat`), which has its own worker.
pub fn is_chat(message: &PageMessage) -> bool {
    matches!(
        message,
        PageMessage::Session { .. }
            | PageMessage::SessionsLoad { .. }
            | PageMessage::SessionsSave { .. }
            | PageMessage::SessionClose { .. }
            | PageMessage::SchedulesList { .. }
            | PageMessage::ScheduleUpdate { .. }
            | PageMessage::ScheduleDelete { .. }
            | PageMessage::EnginesRestart
            | PageMessage::ChatShown { .. }
            | PageMessage::JobStop { .. }
            | PageMessage::JobOutput { .. }
            | PageMessage::JobDismiss { .. }
    )
}

pub fn handle(message: PageMessage, ctx: &Ctx, emit: &dyn Fn(DockMessage)) {
    match message {
        PageMessage::Ready | PageMessage::Opened | PageMessage::CloseRequest => {}
        PageMessage::QuitRequest => crate::request_quit(),
        PageMessage::LoadSettings { id } => emit(load_settings(id, ctx)),
        PageMessage::SaveSettings { id, settings } => {
            let _ = ctx.paths.ensure_root();
            emit(save_settings(id, settings, ctx));
        }
        PageMessage::SaveKey { id, provider, key } => {
            let _ = ctx.paths.ensure_root();
            emit(save_key(id, &provider, &key, ctx));
        }
        PageMessage::CheckKey { id, provider } => {
            let _ = ctx.paths.ensure_root();
            emit(check_stored_key(id, &provider, ctx));
        }
        PageMessage::StartOAuth { id } => {
            let _ = ctx.paths.ensure_root();
            oauth_flow(id, ctx, emit);
        }
        PageMessage::StartMcpOAuth { id, service } => {
            let _ = ctx.paths.ensure_root();
            if service == "notion" {
                notion_flow(id, ctx, emit);
            } else {
                emit(DockMessage::Error { id, code: "unknown-service".into() });
            }
        }
        PageMessage::InstallClaudeBridge { id } => {
            let _ = ctx.paths.ensure_root();
            emit(install_claude_bridge(id, ctx));
        }
        PageMessage::McpScan { id } => emit(mcp_scan(id, ctx)),
        PageMessage::McpImport { id, ids } => {
            let _ = ctx.paths.ensure_root();
            emit(mcp_import(id, &ids, ctx));
        }
        PageMessage::OpenUrl { url } => {
            if url.starts_with("https://") {
                (ctx.opener)(&url);
            }
        }
        // The chat messages are handled by `chat::Sessions`, not here (they arrive wrapped in a session, or are about
        // background jobs, which the dock owns for every session).
        PageMessage::ChatSend { .. }
        | PageMessage::ChatAbort
        | PageMessage::ChatLoad { .. }
        | PageMessage::ChatEarlier { .. }
        | PageMessage::EnginesRestart
        | PageMessage::ChatShown { .. }
        | PageMessage::ChatClearQueue { .. }
        | PageMessage::UiAnswer { .. }
        | PageMessage::ListCommands { .. }
        | PageMessage::ListModels { .. }
        | PageMessage::SetModel { .. }
        | PageMessage::ListFiles { .. }
        | PageMessage::PiCommand { .. }
        | PageMessage::Session { .. }
        | PageMessage::SessionsLoad { .. }
        | PageMessage::SessionsSave { .. }
        | PageMessage::SessionClose { .. }
        | PageMessage::SchedulesList { .. }
        | PageMessage::ScheduleUpdate { .. }
        | PageMessage::ScheduleDelete { .. }
        | PageMessage::JobStop { .. }
        | PageMessage::JobOutput { .. }
        | PageMessage::JobDismiss { .. } => {}
    }
}

/// The servers the other apps configure that Aria's `mcp.json` does not have yet (names, kinds, targets and the names of
/// their secrets; never a value).
fn mcp_scan(id: u64, ctx: &Ctx) -> DockMessage {
    let found = mcp_import::find_in(&ctx.home, &ctx.config_dir, &mcp::servers(&ctx.paths));
    DockMessage::McpFound { id, servers: serde_json::to_value(found).unwrap_or_default() }
}

/// Finds the servers again (the page never holds their configuration) and copies the chosen ones; on success every Pi
/// restarts when its run settles, so it sees them.
fn mcp_import(id: u64, ids: &[String], ctx: &Ctx) -> DockMessage {
    let found = mcp_import::find_in(&ctx.home, &ctx.config_dir, &mcp::servers(&ctx.paths));
    match mcp_import::import(&ctx.paths, &found, ids) {
        Ok(_) => {
            (ctx.restart_engines)();
            DockMessage::Saved { id }
        }
        Err(ImportError::NothingToImport) => DockMessage::Error { id, code: "nothing-to-import".into() },
        Err(ImportError::File(_)) => DockMessage::Error { id, code: "mcp-file-broken".into() },
        Err(ImportError::Write(_)) => DockMessage::Error { id, code: "mcp-write-failed".into() },
        Err(ImportError::Vault(_)) => DockMessage::Error { id, code: "keyring-failed".into() },
    }
}

/// What is really set up on disk (names only): the credentials in `auth.json` and the MCP services.
fn connected(ctx: &Ctx) -> Vec<String> {
    let mut found = AuthStore::for_paths(&ctx.paths).summary();
    found.extend(mcp::connected(&ctx.paths));
    found.sort();
    found
}

fn load_settings(id: u64, ctx: &Ctx) -> DockMessage {
    let connected = connected(ctx);
    let servers = mcp::servers(&ctx.paths);
    match settings::load(&ctx.paths) {
        Ok(Some(settings)) => DockMessage::Settings {
            id,
            settings: serde_json::to_value(settings).ok(),
            exists: true,
            problem: None,
            connected,
            servers,
        },
        Ok(None) => DockMessage::Settings { id, settings: None, exists: false, problem: None, connected, servers },
        Err(error) => {
            let problem = match error {
                SettingsError::File(JsonFileError::Io { .. }) => "settings-unreadable",
                _ => "settings-invalid",
            };
            DockMessage::Settings {
                id,
                settings: None,
                exists: true,
                problem: Some(problem.into()),
                connected,
                servers,
            }
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
        // Tavily's key is not a credential for the agent's provider: it goes in mcp.json (see save_key).
        "tavily" => Some((KeyKind::Tavily, vec![])),
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
    if provider == "tavily" {
        return match mcp::set_tavily_key(&ctx.paths, key) {
            Ok(()) => key_result(id, None),
            Err(McpError::File(_) | McpError::Write(_)) => key_result(id, Some("mcp-file-broken")),
            Err(McpError::Vault(_)) => key_result(id, Some("keyring-failed")),
        };
    }
    let store = AuthStore::for_paths(&ctx.paths);
    for entry in entries {
        match store.set(entry, &Credential::ApiKey { key: key.to_string() }) {
            Ok(()) => {}
            Err(aria_core::auth::AuthError::Vault(_)) => return key_result(id, Some("keyring-failed")),
            Err(_) => return key_result(id, Some("auth-file-broken")),
        }
    }
    key_result(id, None)
}

/// Checks the credential already in `auth.json` for a provider. The page never has the key: it asks, and
/// the dock reads it and validates it. Used by the setup to go on when the credential it finds still works.
fn check_stored_key(id: u64, provider: &str, ctx: &Ctx) -> DockMessage {
    if provider == "anthropic-oauth" {
        return check_stored_oauth(id, ctx);
    }
    let Some((kind, entries)) = key_target(provider) else {
        return DockMessage::Error { id, code: "unknown-provider".into() };
    };
    if entries.is_empty() {
        // Tavily's key lives in mcp.json and is checked when it is saved, not here.
        return DockMessage::Error { id, code: "unknown-provider".into() };
    }
    let store = AuthStore::for_paths(&ctx.paths);
    let mut stored = None;
    for entry in entries {
        match store.get(entry) {
            Ok(Some(credential)) => {
                stored = Some(credential);
                break;
            }
            Ok(None) => {}
            Err(_) => return key_result(id, Some("auth-file-broken")),
        }
    }
    match stored {
        Some(Credential::ApiKey { key }) => match (ctx.validator)(kind, &key) {
            KeyCheck::Valid => key_result(id, None),
            KeyCheck::Rejected => key_result(id, Some("key-rejected")),
            KeyCheck::Unavailable(_) => key_result(id, Some("network-unavailable")),
        },
        // An OAuth credential is not an API key and cannot be checked this way.
        _ => key_result(id, Some("key-missing")),
    }
}

/// A Claude sign-in is a token, not a key: it cannot be probed with `x-api-key`. An expired one is fine
/// when it can be refreshed (Pi does that when it runs); a live one is confirmed against the same cheap
/// usage endpoint the quota bars read.
fn check_stored_oauth(id: u64, ctx: &Ctx) -> DockMessage {
    let store = AuthStore::for_paths(&ctx.paths);
    let credential = match store.get("anthropic") {
        Ok(Some(credential)) => credential,
        Ok(None) => return key_result(id, Some("key-missing")),
        Err(_) => return key_result(id, Some("auth-file-broken")),
    };
    let Credential::OAuth { access, refresh, expires } = credential else {
        // An API key for the same provider is checked by the key path, not here.
        return key_result(id, Some("key-missing"));
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or(0);
    if expires <= now {
        return if refresh.trim().is_empty() { key_result(id, Some("key-rejected")) } else { key_result(id, None) };
    }
    let request = aria_core::quota::request_for(&aria_core::quota::Source::Claude { token: access });
    match (ctx.oauth_checker)(&request) {
        KeyCheck::Valid => key_result(id, None),
        KeyCheck::Rejected => key_result(id, Some("key-rejected")),
        KeyCheck::Unavailable(_) => key_result(id, Some("network-unavailable")),
    }
}

/// Makes the `claude-bridge` provider available to Aria's Pi: it needs Claude Code on the machine, and the package (about
/// 300 MB, once). Claude Code is checked first so nothing is downloaded for a provider that could not run.
fn install_claude_bridge(id: u64, ctx: &Ctx) -> DockMessage {
    let error = |code: &str| DockMessage::Error { id, code: code.into() };
    let Ok(install) = aria_core::pi::locate(&ctx.paths, aria_core::pi::pinned_version()) else {
        return error("pi-missing");
    };
    if aria_core::pi::locate_tool(&ctx.paths, "claude", &ctx.path_var).is_none() {
        return error("claude-code-missing");
    }
    if agent::claude_bridge_installed(&ctx.paths) {
        return DockMessage::Saved { id };
    }
    if agent::prepare(&ctx.paths, None).is_err() {
        return error("claude-bridge-install-failed");
    }
    let command = agent::claude_bridge_install_command(&ctx.paths, &install, &ctx.path_var);
    if (ctx.installer)(&command) && agent::claude_bridge_installed(&ctx.paths) {
        DockMessage::Saved { id }
    } else {
        error("claude-bridge-install-failed")
    }
}

fn oauth_error_key(error: &OAuthError) -> &'static str {
    match error {
        OAuthError::StateMismatch => "oauth-state",
        OAuthError::Denied(_) => "oauth-denied",
        OAuthError::Timeout => "oauth-timeout",
        OAuthError::Network(_) => "network-unavailable",
        OAuthError::NoCode | OAuthError::BadToken(_) | OAuthError::Io(_) | OAuthError::Setup(_) => "oauth-failed",
    }
}

/// Notion's sign-in: discover, register, open the browser, wait, then store the record in Pi's
/// `mcp-auth.json` and list Notion in `mcp.json`. Nothing is written unless it all succeeds.
fn notion_flow(id: u64, ctx: &Ctx, emit: &dyn Fn(DockMessage)) {
    let result =
        |error: Option<&str>| DockMessage::OAuthResult { id, ok: error.is_none(), error: error.map(str::to_string) };
    let sign_in = match oauth::mcp::start(&ctx.notion_url) {
        Ok(sign_in) => sign_in,
        Err(error) => return emit(result(Some(oauth_error_key(&error)))),
    };
    emit(DockMessage::OAuthUrl { id, url: sign_in.url.clone() });
    (ctx.opener)(&sign_in.url);
    let record = match sign_in.wait(ctx.oauth_timeout) {
        Ok(record) => record,
        Err(error) => return emit(result(Some(oauth_error_key(&error)))),
    };
    let stored = mcp::store_oauth_record(&ctx.paths, "notion", &ctx.notion_url, record)
        .and_then(|()| mcp::add_notion_server(&ctx.paths));
    emit(result(stored.err().map(|_| "mcp-file-broken")));
}

fn oauth_flow(id: u64, ctx: &Ctx, emit: &dyn Fn(DockMessage)) {
    let result =
        |error: Option<&str>| DockMessage::OAuthResult { id, ok: error.is_none(), error: error.map(str::to_string) };
    let sign_in = match oauth::anthropic::start() {
        Ok(sign_in) => sign_in,
        Err(error) => return emit(result(Some(oauth_error_key(&error)))),
    };
    emit(DockMessage::OAuthUrl { id, url: sign_in.url.clone() });
    (ctx.opener)(&sign_in.url);
    match sign_in.wait(ctx.oauth_timeout) {
        Ok(credential) => match AuthStore::for_paths(&ctx.paths).set("anthropic", &credential) {
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
        checked_oauth: Arc<Mutex<Vec<String>>>,
        restarts: Arc<std::sync::atomic::AtomicUsize>,
    }

    fn harness(verdict: KeyCheck) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let opened = Arc::new(Mutex::new(Vec::new()));
        let seen_keys = Arc::new(Mutex::new(Vec::new()));
        let checked_oauth = Arc::new(Mutex::new(Vec::new()));
        let (o, s, c) = (opened.clone(), seen_keys.clone(), checked_oauth.clone());
        let restarts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let r = restarts.clone();
        let key_verdict = verdict.clone();
        let ctx = Ctx {
            paths: AriaPaths::new(dir.path().to_path_buf()),
            validator: Arc::new(move |kind, key| {
                s.lock().unwrap().push((kind, key.to_string()));
                key_verdict.clone()
            }),
            oauth_checker: Arc::new(move |request| {
                c.lock().unwrap().push(request.url.clone());
                verdict.clone()
            }),
            opener: Arc::new(move |url| o.lock().unwrap().push(url.to_string())),
            oauth_timeout: Duration::from_millis(150),
            notion_url: "http://127.0.0.1:9/mcp".into(),
            installer: Arc::new(|_| false),
            path_var: String::new(),
            home: dir.path().join("home"),
            config_dir: dir.path().join("home/.config"),
            restart_engines: Arc::new(move || {
                r.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }),
        };
        Harness { dir, ctx, opened, seen_keys, checked_oauth, restarts }
    }

    fn run(h: &Harness, message: PageMessage) -> Vec<DockMessage> {
        let out = Mutex::new(Vec::new());
        handle(message, &h.ctx, &|m| out.lock().unwrap().push(m));
        out.into_inner().unwrap()
    }

    fn auth(h: &Harness) -> AuthStore {
        AuthStore::for_paths(&h.ctx.paths)
    }

    // ---- importing other apps' MCP servers -------------------------------------------------------------

    #[test]
    fn scan_then_import_writes_mcp_json() {
        use std::sync::atomic::Ordering;
        let h = harness(KeyCheck::Valid);
        let cursor = h.ctx.home.join(".cursor/mcp.json");
        std::fs::create_dir_all(cursor.parent().unwrap()).unwrap();
        std::fs::write(
            &cursor,
            r#"{"mcpServers": {"docs": {"url": "https://docs.example.com/mcp", "headers": {"X-Api-Key": "docs-secret-value"}}, "tavily": {"command": "tavily-mcp"}, "feed": {"type": "sse", "url": "https://feed.example.com/sse"}}}"#,
        )
        .unwrap();
        mcp::add_notion_server(&h.ctx.paths).unwrap();
        mcp::set_tavily_key(&h.ctx.paths, "tvly-key").unwrap();

        let replies = run(&h, PageMessage::McpScan { id: 1 });
        assert_eq!(
            replies,
            vec![DockMessage::McpFound {
                id: 1,
                servers: serde_json::json!([
                    {"id": "cursor:docs", "name": "docs", "importAs": "docs", "source": "cursor", "kind": "http", "target": "docs.example.com", "secrets": ["X-Api-Key"], "warnings": [], "importable": true, "disabled": false},
                    {"id": "cursor:feed", "name": "feed", "importAs": "feed", "source": "cursor", "kind": "sse", "target": "feed.example.com", "secrets": [], "warnings": [], "importable": false, "disabled": false}
                ]),
            }],
            "tavily is already in Aria's mcp.json, and no secret value reaches the page"
        );

        assert_eq!(
            run(&h, PageMessage::McpImport { id: 2, ids: vec![] }),
            vec![DockMessage::Error { id: 2, code: "nothing-to-import".into() }]
        );
        assert_eq!(h.restarts.load(Ordering::SeqCst), 0);

        assert_eq!(
            run(&h, PageMessage::McpImport { id: 3, ids: vec!["cursor:docs".into()] }),
            vec![DockMessage::Saved { id: 3 }]
        );
        assert_eq!(h.restarts.load(Ordering::SeqCst), 1, "every Pi restarts to see the new server");
        assert_eq!(mcp::servers(&h.ctx.paths), vec!["docs", "notion", "tavily"]);
        let file: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(h.ctx.paths.mcp()).unwrap()).unwrap();
        assert_eq!(file["mcpServers"]["docs"]["headers"]["X-Api-Key"], "docs-secret-value");
        assert_eq!(std::fs::metadata(h.ctx.paths.mcp()).unwrap().permissions().mode() & 0o777, 0o600);

        // Imported: no longer offered.
        assert_eq!(
            run(&h, PageMessage::McpScan { id: 4 }),
            vec![DockMessage::McpFound {
                id: 4,
                servers: serde_json::json!([{"id": "cursor:feed", "name": "feed", "importAs": "feed", "source": "cursor", "kind": "sse", "target": "feed.example.com", "secrets": [], "warnings": [], "importable": false, "disabled": false}])
            }]
        );
        // SSE is listed but never imported.
        assert_eq!(
            run(&h, PageMessage::McpImport { id: 6, ids: vec!["cursor:feed".into()] }),
            vec![DockMessage::Error { id: 6, code: "nothing-to-import".into() }]
        );

        std::fs::write(h.ctx.paths.mcp(), "{ broken").unwrap();
        assert_eq!(
            run(&h, PageMessage::McpImport { id: 5, ids: vec!["cursor:docs".into()] }),
            vec![DockMessage::Error { id: 5, code: "mcp-file-broken".into() }]
        );
        assert_eq!(h.restarts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn an_import_that_cannot_write_says_so() {
        let h = harness(KeyCheck::Valid);
        let cursor = h.ctx.home.join(".cursor/mcp.json");
        std::fs::create_dir_all(cursor.parent().unwrap()).unwrap();
        std::fs::write(&cursor, r#"{"mcpServers": {"docs": {"url": "https://docs.example.com/mcp"}}}"#).unwrap();
        let root = &h.ctx.paths.root;
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o500)).unwrap();
        let reply = run(&h, PageMessage::McpImport { id: 1, ids: vec!["cursor:docs".into()] });
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(reply, vec![DockMessage::Error { id: 1, code: "mcp-write-failed".into() }]);
        assert_eq!(h.restarts.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn scan_and_import_run_off_the_ui_thread() {
        assert!(is_blocking(&PageMessage::McpScan { id: 1 }));
        assert!(is_blocking(&PageMessage::McpImport { id: 1, ids: vec![] }));
    }

    // ---- the Claude provider -----------------------------------------------------------------------

    use aria_core::agent::{claude_bridge_installed, InstallCommand, CLAUDE_BRIDGE_PACKAGE};
    use std::os::unix::fs::PermissionsExt;

    /// A pinned Pi and a Claude Code that are really there (empty scripts), and an installer that records what it was asked.
    fn with_pi_and_claude(h: &mut Harness, claude: bool) -> Arc<Mutex<Vec<InstallCommand>>> {
        let binary = h.ctx.paths.pi_install(aria_core::pi::pinned_version()).join("node_modules/.bin/pi");
        std::fs::create_dir_all(binary.parent().unwrap()).unwrap();
        std::fs::write(&binary, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        if claude {
            std::fs::create_dir_all(h.ctx.paths.bin()).unwrap();
            std::fs::write(h.ctx.paths.bin().join("claude"), "#!/bin/sh\n").unwrap();
        }
        let ran = Arc::new(Mutex::new(Vec::new()));
        let (ran_in, agent) = (ran.clone(), h.ctx.paths.agent());
        h.ctx.installer = Arc::new(move |command: &InstallCommand| {
            ran_in.lock().unwrap().push(command.clone());
            // what `pi install` leaves behind
            std::fs::create_dir_all(agent.join("npm/node_modules/pi-claude-agent-sdk")).unwrap();
            std::fs::write(agent.join("settings.json"), format!(r#"{{"packages":["{CLAUDE_BRIDGE_PACKAGE}"]}}"#))
                .unwrap();
            true
        });
        ran
    }

    #[test]
    fn background_job_messages_go_to_the_chat() {
        assert!(is_chat(&PageMessage::JobStop { job: "j1".into() }));
        assert!(is_chat(&PageMessage::JobOutput { id: 1, job: "j1".into(), lines: None }));
        assert!(is_chat(&PageMessage::JobDismiss { job: "j1".into() }));
        assert!(!is_chat(&PageMessage::LoadSettings { id: 1 }));
    }

    #[test]
    fn installing_the_claude_provider_runs_pi_install_and_says_saved() {
        let mut h = harness(KeyCheck::Valid);
        let ran = with_pi_and_claude(&mut h, true);
        assert_eq!(run(&h, PageMessage::InstallClaudeBridge { id: 5 }), vec![DockMessage::Saved { id: 5 }]);
        let ran = ran.lock().unwrap();
        assert_eq!(ran.len(), 1);
        assert_eq!(ran[0].args, vec!["install".to_string(), CLAUDE_BRIDGE_PACKAGE.to_string()]);
        assert!(claude_bridge_installed(&h.ctx.paths));
    }

    #[test]
    fn an_already_installed_claude_provider_is_not_downloaded_again() {
        let mut h = harness(KeyCheck::Valid);
        let ran = with_pi_and_claude(&mut h, true);
        run(&h, PageMessage::InstallClaudeBridge { id: 1 });
        assert_eq!(run(&h, PageMessage::InstallClaudeBridge { id: 2 }), vec![DockMessage::Saved { id: 2 }]);
        assert_eq!(ran.lock().unwrap().len(), 1);
    }

    #[test]
    fn without_claude_code_nothing_is_downloaded() {
        let mut h = harness(KeyCheck::Valid);
        let ran = with_pi_and_claude(&mut h, false);
        h.ctx.path_var = "/nonexistent".into();
        assert_eq!(
            run(&h, PageMessage::InstallClaudeBridge { id: 3 }),
            vec![DockMessage::Error { id: 3, code: "claude-code-missing".into() }]
        );
        assert!(ran.lock().unwrap().is_empty());
    }

    #[test]
    fn without_the_pinned_pi_it_says_so() {
        let mut h = harness(KeyCheck::Valid);
        h.ctx.path_var = "/nonexistent".into();
        assert_eq!(
            run(&h, PageMessage::InstallClaudeBridge { id: 4 }),
            vec![DockMessage::Error { id: 4, code: "pi-missing".into() }]
        );
    }

    #[test]
    fn a_failed_install_is_reported_and_leaves_nothing_listed() {
        let mut h = harness(KeyCheck::Valid);
        with_pi_and_claude(&mut h, true);
        h.ctx.installer = Arc::new(|_| false);
        assert_eq!(
            run(&h, PageMessage::InstallClaudeBridge { id: 6 }),
            vec![DockMessage::Error { id: 6, code: "claude-bridge-install-failed".into() }]
        );
        assert!(!claude_bridge_installed(&h.ctx.paths));
    }

    #[test]
    fn an_install_that_exits_well_but_left_no_package_is_a_failure() {
        let mut h = harness(KeyCheck::Valid);
        with_pi_and_claude(&mut h, true);
        h.ctx.installer = Arc::new(|_| true); // lies: installs nothing
        assert_eq!(
            run(&h, PageMessage::InstallClaudeBridge { id: 7 }),
            vec![DockMessage::Error { id: 7, code: "claude-bridge-install-failed".into() }]
        );
    }

    #[test]
    fn load_settings_when_missing() {
        let h = harness(KeyCheck::Valid);
        assert_eq!(
            run(&h, PageMessage::LoadSettings { id: 1 }),
            vec![DockMessage::Settings {
                id: 1,
                settings: None,
                exists: false,
                problem: None,
                connected: vec![],
                servers: vec![]
            }]
        );
    }

    #[test]
    fn load_settings_broken() {
        let h = harness(KeyCheck::Valid);
        std::fs::write(h.ctx.paths.settings(), "{ nope").unwrap();
        assert_eq!(
            run(&h, PageMessage::LoadSettings { id: 1 }),
            vec![DockMessage::Settings {
                id: 1,
                settings: None,
                exists: true,
                problem: Some("settings-invalid".into()),
                connected: vec![],
                servers: vec![]
            }]
        );
    }

    #[test]
    fn save_settings_round_trip() {
        let h = harness(KeyCheck::Valid);
        let mut settings = aria_core::settings::Settings::defaults("es");
        settings.assistant.user_name = "Ivan".into();
        let value = serde_json::to_value(&settings).unwrap();
        assert_eq!(
            run(&h, PageMessage::SaveSettings { id: 2, settings: value.clone() }),
            vec![DockMessage::Saved { id: 2 }]
        );
        assert_eq!(
            run(&h, PageMessage::LoadSettings { id: 3 }),
            vec![DockMessage::Settings {
                id: 3,
                settings: Some(value),
                exists: true,
                problem: None,
                connected: vec![],
                servers: vec![]
            }]
        );
    }

    #[test]
    fn save_settings_with_the_wrong_shape_is_an_error_and_writes_nothing() {
        let h = harness(KeyCheck::Valid);
        let replies =
            run(&h, PageMessage::SaveSettings { id: 4, settings: json!({"compaction": {"percent": "lots"}}) });
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

    /// A keyring that refuses everything (locked, or the Secret Service gone).
    #[derive(Debug)]
    struct Refusing;

    impl aria_core::keyring::Backend for Refusing {
        fn get(&self, _: &str) -> Result<Option<String>, aria_core::keyring::KeyringError> {
            Ok(None)
        }
        fn set(&self, _: &str, _: &str) -> Result<(), aria_core::keyring::KeyringError> {
            Err(aria_core::keyring::KeyringError::Failed("locked".into()))
        }
        fn delete(&self, _: &str) -> Result<(), aria_core::keyring::KeyringError> {
            Ok(())
        }
    }

    #[test]
    fn a_keyring_that_refuses_a_key_says_so_and_writes_nothing() {
        let mut h = harness(KeyCheck::Valid);
        let live = h.dir.path().join("live");
        std::fs::create_dir_all(&live).unwrap();
        h.ctx.paths.vault = Some(aria_core::vault::Vault {
            keyring: Arc::new(Refusing),
            live,
            exe: std::path::PathBuf::from("/bin/aria"),
        });
        let replies = run(&h, PageMessage::SaveKey { id: 7, provider: "opencode-go".into(), key: "k1".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 7, ok: false, error: Some("keyring-failed".into()) }]);
        assert!(!h.ctx.paths.auth().exists());
        let replies = run(&h, PageMessage::SaveKey { id: 8, provider: "tavily".into(), key: "t".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 8, ok: false, error: Some("keyring-failed".into()) }]);
        assert!(!h.ctx.paths.mcp().exists());
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
        assert_eq!(
            replies,
            vec![DockMessage::KeyResult { id: 8, ok: false, error: Some("network-unavailable".into()) }]
        );
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
    fn a_stored_key_is_checked_without_the_page_seeing_it() {
        let h = harness(KeyCheck::Valid);
        run(&h, PageMessage::SaveKey { id: 1, provider: "opencode-go".into(), key: "k1".into() });
        h.seen_keys.lock().unwrap().clear();
        let replies = run(&h, PageMessage::CheckKey { id: 2, provider: "opencode-go".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 2, ok: true, error: None }]);
        assert_eq!(h.seen_keys.lock().unwrap().as_slice(), [(KeyKind::OpenCodeGo, "k1".to_string())]);
    }

    #[test]
    fn checking_a_key_that_is_not_there_says_so() {
        let h = harness(KeyCheck::Valid);
        let replies = run(&h, PageMessage::CheckKey { id: 3, provider: "anthropic-api".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 3, ok: false, error: Some("key-missing".into()) }]);
        assert!(h.seen_keys.lock().unwrap().is_empty());
    }

    #[test]
    fn a_stored_key_that_was_revoked_is_reported_and_kept() {
        let h = harness(KeyCheck::Rejected);
        AuthStore::for_paths(&h.ctx.paths).set("opencode-go", &Credential::ApiKey { key: "old".into() }).unwrap();
        let before = std::fs::read_to_string(h.ctx.paths.auth()).unwrap();
        let replies = run(&h, PageMessage::CheckKey { id: 4, provider: "opencode-go".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 4, ok: false, error: Some("key-rejected".into()) }]);
        assert_eq!(std::fs::read_to_string(h.ctx.paths.auth()).unwrap(), before);
    }

    #[test]
    fn checking_with_a_broken_auth_file_is_reported_and_the_file_is_kept() {
        let h = harness(KeyCheck::Valid);
        std::fs::write(h.ctx.paths.auth(), "{ nope").unwrap();
        let replies = run(&h, PageMessage::CheckKey { id: 5, provider: "opencode-go".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 5, ok: false, error: Some("auth-file-broken".into()) }]);
        assert_eq!(std::fs::read_to_string(h.ctx.paths.auth()).unwrap(), "{ nope");
    }

    #[test]
    fn a_stored_anthropic_api_key_is_checked_like_any_other() {
        let h = harness(KeyCheck::Valid);
        run(&h, PageMessage::SaveKey { id: 6, provider: "anthropic-api".into(), key: "sk-ant-1".into() });
        h.seen_keys.lock().unwrap().clear();
        let replies = run(&h, PageMessage::CheckKey { id: 7, provider: "anthropic-api".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 7, ok: true, error: None }]);
        assert_eq!(h.seen_keys.lock().unwrap().as_slice(), [(KeyKind::AnthropicApi, "sk-ant-1".to_string())]);
    }

    #[test]
    fn a_stored_claude_sign_in_is_checked_against_the_usage_endpoint() {
        let h = harness(KeyCheck::Valid);
        AuthStore::for_paths(&h.ctx.paths)
            .set("anthropic", &Credential::OAuth { access: "tok".into(), refresh: "ref".into(), expires: i64::MAX })
            .unwrap();
        let replies = run(&h, PageMessage::CheckKey { id: 8, provider: "anthropic-oauth".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 8, ok: true, error: None }]);
        assert_eq!(
            h.checked_oauth.lock().unwrap().as_slice(),
            ["https://api.anthropic.com/api/oauth/usage".to_string()]
        );
    }

    #[test]
    fn an_expired_claude_sign_in_with_a_refresh_token_still_works_without_a_request() {
        let h = harness(KeyCheck::Rejected);
        AuthStore::for_paths(&h.ctx.paths)
            .set("anthropic", &Credential::OAuth { access: "old".into(), refresh: "ref".into(), expires: 1 })
            .unwrap();
        let replies = run(&h, PageMessage::CheckKey { id: 9, provider: "anthropic-oauth".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 9, ok: true, error: None }]);
        assert!(h.checked_oauth.lock().unwrap().is_empty());
    }

    #[test]
    fn an_expired_claude_sign_in_that_cannot_be_refreshed_asks_to_sign_in_again() {
        let h = harness(KeyCheck::Valid);
        AuthStore::for_paths(&h.ctx.paths)
            .set("anthropic", &Credential::OAuth { access: "old".into(), refresh: String::new(), expires: 1 })
            .unwrap();
        let replies = run(&h, PageMessage::CheckKey { id: 10, provider: "anthropic-oauth".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 10, ok: false, error: Some("key-rejected".into()) }]);
    }

    #[test]
    fn a_revoked_claude_token_is_reported() {
        let h = harness(KeyCheck::Rejected);
        AuthStore::for_paths(&h.ctx.paths)
            .set("anthropic", &Credential::OAuth { access: "tok".into(), refresh: "ref".into(), expires: i64::MAX })
            .unwrap();
        let replies = run(&h, PageMessage::CheckKey { id: 11, provider: "anthropic-oauth".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 11, ok: false, error: Some("key-rejected".into()) }]);
        assert_eq!(h.checked_oauth.lock().unwrap().len(), 1);
    }

    #[test]
    fn checking_a_claude_sign_in_that_is_not_there_says_so() {
        let h = harness(KeyCheck::Valid);
        let replies = run(&h, PageMessage::CheckKey { id: 12, provider: "anthropic-oauth".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 12, ok: false, error: Some("key-missing".into()) }]);
    }

    #[test]
    fn abandoned_oauth_times_out_and_writes_nothing() {
        let h = harness(KeyCheck::Valid);
        let replies = run(&h, PageMessage::StartOAuth { id: 11 });
        assert_eq!(replies.len(), 2);
        match &replies[0] {
            DockMessage::OAuthUrl { id: 11, url } => {
                assert!(url.starts_with("https://claude.ai/oauth/authorize?"));
                assert_eq!(h.opened.lock().unwrap().as_slice(), std::slice::from_ref(url));
            }
            other => panic!("expected the sign-in URL first, got {other:?}"),
        }
        assert_eq!(replies[1], DockMessage::OAuthResult { id: 11, ok: false, error: Some("oauth-timeout".into()) });
        assert!(!h.ctx.paths.auth().exists());
    }

    #[test]
    fn a_tavily_key_is_checked_and_written_to_mcp_json_not_to_auth_json() {
        let h = harness(KeyCheck::Valid);
        let replies = run(&h, PageMessage::SaveKey { id: 20, provider: "tavily".into(), key: " tvly-1 ".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 20, ok: true, error: None }]);
        assert_eq!(h.seen_keys.lock().unwrap().as_slice(), [(KeyKind::Tavily, "tvly-1".to_string())]);
        let mcp: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(h.ctx.paths.mcp()).unwrap()).unwrap();
        assert_eq!(mcp["mcpServers"]["tavily"]["headers"]["Authorization"], "Bearer tvly-1");
        assert!(!h.ctx.paths.auth().exists());
    }

    #[test]
    fn a_rejected_tavily_key_writes_nothing() {
        let h = harness(KeyCheck::Rejected);
        let replies = run(&h, PageMessage::SaveKey { id: 21, provider: "tavily".into(), key: "bad".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 21, ok: false, error: Some("key-rejected".into()) }]);
        assert!(!h.ctx.paths.mcp().exists());
    }

    #[test]
    fn a_broken_mcp_file_is_reported_and_kept() {
        let h = harness(KeyCheck::Valid);
        std::fs::write(h.ctx.paths.mcp(), "{ nope").unwrap();
        let replies = run(&h, PageMessage::SaveKey { id: 22, provider: "tavily".into(), key: "k".into() });
        assert_eq!(replies, vec![DockMessage::KeyResult { id: 22, ok: false, error: Some("mcp-file-broken".into()) }]);
        assert_eq!(std::fs::read_to_string(h.ctx.paths.mcp()).unwrap(), "{ nope");
    }

    #[test]
    fn the_settings_reply_lists_every_configured_server_but_engram() {
        let h = harness(KeyCheck::Valid);
        std::fs::write(h.ctx.paths.mcp(), "{ nope").unwrap();
        match &run(&h, PageMessage::LoadSettings { id: 1 })[0] {
            DockMessage::Settings { servers, .. } => assert!(servers.is_empty()),
            other => panic!("expected settings, got {other:?}"),
        }
        std::fs::write(h.ctx.paths.mcp(), r#"{"mcpServers":{"rea":{"command":"rea"},"engram":{"command":"engram"}}}"#)
            .unwrap();
        run(&h, PageMessage::SaveKey { id: 2, provider: "tavily".into(), key: "t".into() });
        match &run(&h, PageMessage::LoadSettings { id: 3 })[0] {
            DockMessage::Settings { servers, connected, .. } => {
                assert_eq!(servers, &vec!["rea".to_string(), "tavily".to_string()]);
                assert!(!connected.iter().any(|c| c == "mcp:rea")); // a server is not a credential
            }
            other => panic!("expected settings, got {other:?}"),
        }
    }

    #[test]
    fn the_settings_reply_says_what_is_really_connected() {
        let h = harness(KeyCheck::Valid);
        run(&h, PageMessage::SaveKey { id: 1, provider: "opencode-go".into(), key: "k".into() });
        run(&h, PageMessage::SaveKey { id: 2, provider: "tavily".into(), key: "t".into() });
        AuthStore::for_paths(&h.ctx.paths)
            .set("anthropic", &Credential::OAuth { access: "a".into(), refresh: "r".into(), expires: 1 })
            .unwrap();
        match &run(&h, PageMessage::LoadSettings { id: 3 })[0] {
            DockMessage::Settings { connected, .. } => {
                let expected: Vec<String> =
                    ["anthropic:oauth", "mcp:tavily", "opencode-go:api_key", "opencode:api_key"]
                        .map(String::from)
                        .to_vec();
                assert_eq!(connected, &expected);
            }
            other => panic!("expected settings, got {other:?}"),
        }
    }

    #[test]
    fn notion_signs_in_through_the_browser_and_is_stored_in_pis_files() {
        use aria_core::testing::fake;
        let server = fake(true, 201);
        let mut h = harness(KeyCheck::Valid);
        h.ctx.notion_url = format!("{}/mcp", server.base);
        h.ctx.oauth_timeout = Duration::from_secs(5);
        let out = Mutex::new(Vec::new());
        let browser = Mutex::new(None);
        handle(PageMessage::StartMcpOAuth { id: 30, service: "notion".into() }, &h.ctx, &|m| {
            if let DockMessage::OAuthUrl { url, .. } = &m {
                // the user finishes in the browser: it comes back to the loopback address with the state
                let q: std::collections::HashMap<String, String> =
                    url::Url::parse(url).unwrap().query_pairs().into_owned().collect();
                let (redirect, state) = (q["redirect_uri"].clone(), q["state"].clone());
                *browser.lock().unwrap() = Some(std::thread::spawn(move || {
                    use std::io::{Read, Write};
                    let port: u16 = redirect.rsplit(':').next().unwrap().split('/').next().unwrap().parse().unwrap();
                    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
                    write!(stream, "GET /callback?code=abc&state={state} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
                    let mut sink = String::new();
                    let _ = stream.read_to_string(&mut sink);
                }));
            }
            out.lock().unwrap().push(m);
        });
        browser.lock().unwrap().take().unwrap().join().unwrap();
        let replies = out.into_inner().unwrap();
        assert!(matches!(&replies[0], DockMessage::OAuthUrl { id: 30, .. }));
        assert_eq!(replies[1], DockMessage::OAuthResult { id: 30, ok: true, error: None });
        assert_eq!(h.opened.lock().unwrap().len(), 1);
        // Pi's files: the record under Pi's key, and Notion listed as a server (no secret in mcp.json)
        let auth: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(h.ctx.paths.mcp_auth()).unwrap()).unwrap();
        let key = format!("mcp__notion|{}/mcp", server.base);
        assert_eq!(auth[&key]["tokens"]["access_token"], "at");
        let mcp: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(h.ctx.paths.mcp()).unwrap()).unwrap();
        assert!(mcp["mcpServers"]["notion"]["url"].as_str().unwrap().ends_with("/mcp"));
        assert!(mcp["mcpServers"]["notion"].get("headers").is_none());
    }

    #[test]
    fn an_unreachable_notion_is_a_network_error_and_writes_nothing() {
        let h = harness(KeyCheck::Valid); // notion_url points at a closed port
        let replies = run(&h, PageMessage::StartMcpOAuth { id: 31, service: "notion".into() });
        assert_eq!(
            replies,
            vec![DockMessage::OAuthResult { id: 31, ok: false, error: Some("network-unavailable".into()) }]
        );
        assert!(!h.ctx.paths.mcp_auth().exists());
        assert!(!h.ctx.paths.mcp().exists());
    }

    #[test]
    fn only_notion_signs_in_this_way() {
        let h = harness(KeyCheck::Valid);
        let replies = run(&h, PageMessage::StartMcpOAuth { id: 32, service: "figma".into() });
        assert_eq!(replies, vec![DockMessage::Error { id: 32, code: "unknown-service".into() }]);
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
