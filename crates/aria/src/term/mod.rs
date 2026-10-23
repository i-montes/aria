//! `aria` in a terminal: a client of the dock's sessions (docs/superpowers/specs/2026-10-23-terminal-design.md).

pub mod git;
pub mod link;
pub mod markdown;
pub mod state;
pub mod texts;
pub mod theme;
pub mod view;

/// Runs the client; the process's exit code.
pub fn run() -> i32 {
    eprintln!("aria: the terminal client is not ready yet; the dock runs as `aria dock`");
    1
}
