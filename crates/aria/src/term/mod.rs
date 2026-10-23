//! `aria` in a terminal: a client of the dock's sessions (docs/superpowers/specs/2026-10-23-terminal-design.md).

pub mod app;
pub mod dialog;
pub mod editor;
pub mod git;
pub mod link;
pub mod markdown;
pub mod picker;
mod screen;
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
    if let Err(error) = crate::control::ensure_running(&exe, &clients) {
        eprintln!("aria: {}", texts::t(&lang, "term.cannotStart", &serde_json::json!({ "error": error.to_string() })));
        return 1;
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
    let hint_file = paths.root.join("term.json");
    let hint_seen = std::fs::read_to_string(&hint_file).is_ok_and(|text| text.contains("\"selectHint\":true"));
    let config = app::Config {
        cwd: std::env::current_dir().map(|d| d.display().to_string()).unwrap_or_default(),
        lang,
        verbose: settings.display.verbose,
        connected: aria_core::auth::AuthStore::for_paths(&paths).summary(),
        allow_pay_per_use: settings.models.allow_pay_per_use,
        compact_at: settings.compaction.percent,
        hint_seen,
    };
    match screen::run(link, config, &paths, &hint_file) {
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
