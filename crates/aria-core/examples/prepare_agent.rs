//! Prepares Aria's agent directory under `ARIA_HOME` from its `settings.json`, exactly as a session start does, and
//! prints the files Aria kept because they were edited. For live checks against a copy of `~/.aria`:
//!
//!   ARIA_HOME=/tmp/aria-copy cargo run -p aria-core --example prepare_agent
fn main() {
    let paths = aria_core::paths::AriaPaths::from_env();
    let settings = aria_core::settings::load(&paths).ok().flatten();
    match aria_core::agent::prepare_reporting(&paths, settings.as_ref()) {
        Ok(kept) => println!("prepared {} (kept: {kept:?})", paths.agent().display()),
        Err(error) => {
            eprintln!("could not prepare: {error}");
            std::process::exit(1);
        }
    }
}
