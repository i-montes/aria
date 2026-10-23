//! `aria` in a terminal: a client of the dock's sessions (docs/superpowers/specs/2026-10-23-terminal-design.md).

pub mod app;
pub mod dialog;
pub mod editor;
pub mod git;
pub mod link;
pub mod markdown;
pub mod picker;
pub mod rewind;
mod screen;
pub mod select;
pub mod state;
pub mod texts;
pub mod theme;
pub mod view;

/// A key the pickers, dialogs and editor understand (crossterm's events become these in `app`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Key {
    Char(char),
    Enter,
    Esc,
    Up,
    Down,
    Left,
    Right,
    Backspace,
    Tab,
    PageUp,
    PageDown,
}

/// Runs the client; the process's exit code.
pub fn run() -> i32 {
    let paths = aria_core::paths::AriaPaths::from_env();
    let settings = aria_core::settings::load(&paths).ok().flatten().unwrap_or_default();
    let lang = if settings.assistant.language == "es" { "es" } else { "en" }.to_string();
    let exe = std::env::current_exe().unwrap_or_else(|_| "aria".into());
    let clients = crate::clients::socket_path();
    match crate::control::ensure_running(&exe, &crate::control::control_socket_path(), &clients) {
        Ok(()) => {}
        Err(crate::control::ControlError::OldDock) => {
            eprintln!("aria: {}", texts::t(&lang, "term.oldDock", &serde_json::json!({})));
            return 1;
        }
        Err(error) => {
            eprintln!(
                "aria: {}",
                texts::t(&lang, "term.cannotStart", &serde_json::json!({ "error": error.to_string() }))
            );
            return 1;
        }
    }
    let link = match link::Link::connect(&clients) {
        Ok(link) => link,
        Err(error) => {
            eprintln!(
                "aria: {}",
                texts::t(&lang, "term.cannotStart", &serde_json::json!({ "error": error.to_string() }))
            );
            return 1;
        }
    };
    let config = app::Config {
        cwd: std::env::current_dir().map(|d| d.display().to_string()).unwrap_or_default(),
        lang,
        verbose: settings.display.verbose,
        connected: aria_core::auth::AuthStore::for_paths(&paths).summary(),
        allow_pay_per_use: settings.models.allow_pay_per_use,
        compact_at: settings.compaction.percent,
        remote: None,
    };
    finish(screen::run(link, config, &paths, &|| link::Link::connect(&clients)))
}

/// Runs the client on the service of another computer, over ssh (`aria connect host`); the process's exit code: 1 when
/// ssh is missing, the hub is down or too old, or the link failed, 3 when the protocols differ.
pub fn run_remote(host: &str) -> i32 {
    use crate::connect::{self, ConnectError};
    let paths = aria_core::paths::AriaPaths::from_env();
    let settings = aria_core::settings::load(&paths).ok().flatten().unwrap_or_default();
    let lang = if settings.assistant.language == "es" { "es" } else { "en" }.to_string();
    let none = serde_json::json!({});
    let link = match connect::open(host) {
        Ok(link) => link,
        Err(error) => {
            let (text, code) = match error {
                ConnectError::OptionLikeHost => (format!("{host}: not a host"), 2),
                ConnectError::NoSsh => (texts::t(&lang, "term.noSsh", &none), 1),
                ConnectError::Failed(error) => match link::LinkError::of(&error) {
                    Some(link::LinkError::HubDown) => (texts::t(&lang, "term.hubDown", &none), 1),
                    Some(link::LinkError::HubTooOld) => (texts::t(&lang, "term.hubTooOld", &none), 1),
                    Some(link::LinkError::ProtocolMismatch { theirs }) => {
                        let theirs = theirs.map_or("?".to_string(), |n| n.to_string());
                        let vars = serde_json::json!({ "ours": aria_core::bridge::PROTOCOL, "theirs": theirs });
                        (texts::t(&lang, "term.protocolMismatch", &vars), 3)
                    }
                    None => {
                        (texts::t(&lang, "term.cannotStart", &serde_json::json!({ "error": error.to_string() })), 1)
                    }
                },
            };
            eprintln!("aria: {text}");
            return code;
        }
    };
    let config = app::Config {
        cwd: String::new(),
        lang,
        verbose: settings.display.verbose,
        connected: Vec::new(),
        allow_pay_per_use: settings.models.allow_pay_per_use,
        compact_at: settings.compaction.percent,
        remote: Some(host.to_string()),
    };
    finish(screen::run(link, config, &paths, &|| {
        connect::open(host).map_err(|e| match e {
            ConnectError::Failed(error) => error,
            ConnectError::NoSsh => std::io::Error::new(std::io::ErrorKind::NotFound, "ssh is not installed"),
            ConnectError::OptionLikeHost => std::io::Error::new(std::io::ErrorKind::InvalidInput, "not a host"),
        })
    }))
}

/// What to print and return when the screen is over.
fn finish(result: std::io::Result<Option<String>>) -> i32 {
    match result {
        Ok(note) => {
            if let Some(note) = note {
                println!("{note}");
            }
            0
        }
        Err(error) => {
            eprintln!("aria: {error}");
            1
        }
    }
}
