//! Prepares the agent directory of the Aria home in `ARIA_HOME` as a session start does, writes its `pi` shim, and
//! prints the shell lines that give a command Pi's exact session environment, with `ARIA_PARITY_WORKDIR` as the working
//! folder. `tools/parity/scratch-home.sh` runs it against a scratch copy of `~/.aria`.
use aria_core::{agent, paths::AriaPaths, pi, settings};
use std::path::PathBuf;

fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn main() {
    let paths = AriaPaths::from_env();
    let workdir = PathBuf::from(std::env::var("ARIA_PARITY_WORKDIR").expect("ARIA_PARITY_WORKDIR"));
    let fail = |what: &str, error: &dyn std::fmt::Display| -> ! {
        eprintln!("parity_env: {what}: {error}");
        std::process::exit(1)
    };
    let settings = settings::load(&paths).unwrap_or_else(|e| fail("settings", &e));
    let install = pi::locate(&paths, pi::pinned_version()).unwrap_or_else(|e| fail("the pinned Pi", &e));
    agent::prepare_reporting(&paths, settings.as_ref()).unwrap_or_else(|e| fail("prepare", &e));
    let save_sessions = settings.as_ref().is_some_and(|s| s.subagents.save_sessions);
    pi::write_shim(&paths, &install, save_sessions).unwrap_or_else(|e| fail("shim", &e));
    let path_var = std::env::var("PATH").unwrap_or_default();
    for name in pi::unset_for_pi(std::env::vars()) {
        println!("unset {name}");
    }
    for (key, value) in pi::session_env(&paths, &path_var, &workdir) {
        println!("export {key}={}", quote(&value));
    }
    println!("export ARIA_PI={}", quote(&install.binary.display().to_string()));
    let dist = install.dir.join("node_modules/@earendil-works/pi-coding-agent/dist");
    println!("export ARIA_PI_DIST={}", quote(&dist.display().to_string()));
}
