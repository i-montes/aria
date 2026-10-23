//! The pinned Pi Aria runs: which version, where it is installed, whether Node can run it, and the
//! `pi` shim that Pi's own children (subagents) find on their `PATH`.

use crate::paths::AriaPaths;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The version recorded in the repo's `pi-version` file; changing it is a deliberate upgrade.
pub fn pinned_version() -> &'static str {
    include_str!("../../../pi-version").trim()
}

pub const MIN_NODE: (u32, u32, u32) = (22, 19, 0);

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PiError {
    #[error("the pinned Pi is not installed")]
    NotInstalled,
    #[error("Node {}.{} or newer is needed to run Pi", MIN_NODE.0, MIN_NODE.1)]
    NodeMissing,
}

impl PiError {
    /// The UI string key the dock shows for this problem.
    pub fn problem_code(&self) -> &'static str {
        match self {
            PiError::NotInstalled => "pi-missing",
            PiError::NodeMissing => "node-missing",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PiInstall {
    pub dir: PathBuf,
    pub binary: PathBuf,
}

pub fn locate(paths: &AriaPaths, version: &str) -> Result<PiInstall, PiError> {
    let dir = paths.pi_install(version);
    let binary = dir.join("node_modules/.bin/pi");
    match std::fs::metadata(&binary) {
        Ok(meta) if meta.is_file() && meta.permissions().mode() & 0o111 != 0 => Ok(PiInstall { dir, binary }),
        _ => Err(PiError::NotInstalled),
    }
}

/// `v22.23.2` or `22.23.2` -> `(22, 23, 2)`.
pub fn parse_node_version(text: &str) -> Option<(u32, u32, u32)> {
    let mut parts = text.trim().trim_start_matches('v').split('.');
    let mut next = || parts.next()?.split(|c: char| !c.is_ascii_digit()).next()?.parse::<u32>().ok();
    Some((next()?, next()?, next()?))
}

pub fn meets_minimum(version: (u32, u32, u32)) -> bool {
    version >= MIN_NODE
}

/// A tool Pi's children need (`node`, `claude`) as they will find it: the installer's link in `~/.aria/bin` first (a
/// service has no `mise` on its `PATH`), then the given `PATH` string.
pub fn locate_tool(paths: &AriaPaths, name: &str, path_var: &str) -> Option<PathBuf> {
    let linked = paths.bin().join(name);
    if linked.exists() {
        return Some(linked);
    }
    std::env::split_paths(path_var).map(|dir| dir.join(name)).find(|candidate| candidate.is_file())
}

/// Node as Pi will find it: the installer's link in `~/.aria/bin` first (a service has no `mise` on its
/// `PATH`), then the given `PATH` string.
pub fn locate_node(paths: &AriaPaths, path_var: &str) -> Option<PathBuf> {
    locate_tool(paths, "node", path_var)
}

/// Is there a Node that can run Pi?
pub fn node_ok(paths: &AriaPaths, path_var: &str) -> bool {
    let Some(node) = locate_node(paths, path_var) else { return false };
    let Ok(output) = Command::new(node).arg("--version").output() else { return false };
    parse_node_version(&String::from_utf8_lossy(&output.stdout)).is_some_and(meets_minimum)
}

/// `~/.aria/bin/pi`: runs the pinned binary, so a child that starts `pi` by name gets the same version. With
/// `save_sessions` (settings: `subagents.saveSessions`) a subagent's `--no-session` becomes a session folder of its own.
pub fn write_shim(paths: &AriaPaths, install: &PiInstall, save_sessions: bool) -> io::Result<PathBuf> {
    std::fs::create_dir_all(paths.bin())?;
    let shim = paths.bin().join("pi");
    let script = if save_sessions {
        let dir = paths.agent().join("sessions").join("subagents");
        format!(
            "#!/bin/sh\n# Subagents keep their sessions (settings: subagents.saveSessions).\nfor a do\n  shift\n  if [ \"$a\" = --no-session ]; then set -- \"$@\" --session-dir '{}'; else set -- \"$@\" \"$a\"; fi\ndone\nexec '{}' \"$@\"\n",
            dir.display(),
            install.binary.display()
        )
    } else {
        format!("#!/bin/sh\nexec '{}' \"$@\"\n", install.binary.display())
    };
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = paths.bin().join(format!(".pi.{}.{n}.tmp", std::process::id()));
    std::fs::write(&tmp, script)?;
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    std::fs::rename(&tmp, &shim)?;
    Ok(shim)
}

/// The environment a session's Pi runs with: its agent directory, Aria's `bin` first on `PATH`, no version check, the
/// extensions' projects folder (the working folder) and Aria's settings, which `/effort` and `/autocompact` write
/// (extensions/lib/config.ts).
pub fn session_env(paths: &AriaPaths, path_var: &str, workdir: &Path) -> Vec<(String, String)> {
    vec![
        ("PI_CODING_AGENT_DIR".into(), paths.agent().display().to_string()),
        ("PATH".into(), path_with_bin(paths, path_var)),
        ("PI_SKIP_VERSION_CHECK".into(), "1".into()),
        ("ARIA_PROJECTS_DIR".into(), workdir.display().to_string()),
        ("ARIA_SETTINGS".into(), paths.settings().display().to_string()),
        // Ponytail's own switches: no "loaded" notice on every session and no status line (the dock has its own).
        ("PONYTAIL_QUIET_STARTUP".into(), "1".into()),
        ("PONYTAIL_HIDE_STATUS".into(), "1".into()),
    ]
}

/// The inherited variables Pi must not see: any `ANTHROPIC_*` breaks the Claude provider, which signs in with OAuth.
pub fn unset_for_pi(vars: impl Iterator<Item = (String, String)>) -> Vec<String> {
    vars.map(|(name, _)| name).filter(|name| name.starts_with("ANTHROPIC_")).collect()
}

/// The directory to put first on a Pi process's `PATH`.
pub fn path_with_bin(paths: &AriaPaths, path_var: &str) -> String {
    let bin: &Path = &paths.bin();
    format!("{}:{}", bin.display(), path_var)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Executing a script another test thread just wrote can fail with "text file busy" (a concurrent
    // fork inherits the open write handle), so the tests that run a fake node take turns.
    static NODE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn paths() -> (tempfile::TempDir, AriaPaths) {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().to_path_buf());
        (dir, paths)
    }

    #[test]
    fn anthropic_variables_are_unset_for_pi() {
        let vars = [("ANTHROPIC_API_KEY", "k"), ("ANTHROPIC_BASE_URL", "u"), ("PATH", "/bin"), ("MY_ANTHROPIC", "z")];
        let unset = unset_for_pi(vars.iter().map(|(k, v)| (k.to_string(), v.to_string())));
        assert_eq!(unset, vec!["ANTHROPIC_API_KEY".to_string(), "ANTHROPIC_BASE_URL".to_string()]);
    }

    #[test]
    fn session_env_lists_what_pi_needs() {
        let (_dir, paths) = paths();
        let env = session_env(&paths, "/usr/bin", Path::new("/work"));
        let keys: Vec<&str> = env.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            keys,
            [
                "PI_CODING_AGENT_DIR",
                "PATH",
                "PI_SKIP_VERSION_CHECK",
                "ARIA_PROJECTS_DIR",
                "ARIA_SETTINGS",
                "PONYTAIL_QUIET_STARTUP",
                "PONYTAIL_HIDE_STATUS"
            ]
        );
        assert_eq!(env[3].1, "/work");
        assert_eq!(env[4].1, paths.settings().display().to_string());
        assert_eq!((env[5].1.as_str(), env[6].1.as_str()), ("1", "1"));
    }

    fn fake_install(paths: &AriaPaths, version: &str, mode: u32) -> PathBuf {
        let binary = paths.pi_install(version).join("node_modules/.bin/pi");
        std::fs::create_dir_all(binary.parent().unwrap()).unwrap();
        std::fs::write(&binary, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(mode)).unwrap();
        binary
    }

    #[test]
    fn the_pinned_version_comes_from_the_repo_file() {
        assert_eq!(pinned_version(), "1.3.0");
    }

    #[test]
    fn locates_an_installed_binary() {
        let (_dir, paths) = paths();
        let binary = fake_install(&paths, "1.3.0", 0o755);
        let install = locate(&paths, "1.3.0").unwrap();
        assert_eq!(install.binary, binary);
        assert_eq!(install.dir, paths.pi_install("1.3.0"));
    }

    #[test]
    fn missing_or_unexecutable_binary_is_not_installed() {
        let (_dir, paths) = paths();
        assert_eq!(locate(&paths, "1.3.0"), Err(PiError::NotInstalled));
        fake_install(&paths, "1.3.0", 0o644);
        assert_eq!(locate(&paths, "1.3.0"), Err(PiError::NotInstalled));
        assert_eq!(locate(&paths, "9.9.9"), Err(PiError::NotInstalled));
    }

    #[test]
    fn problem_codes() {
        assert_eq!(PiError::NotInstalled.problem_code(), "pi-missing");
        assert_eq!(PiError::NodeMissing.problem_code(), "node-missing");
    }

    #[test]
    fn node_versions_parse_and_compare() {
        assert_eq!(parse_node_version("v22.23.2\n"), Some((22, 23, 2)));
        assert_eq!(parse_node_version("22.19.0"), Some((22, 19, 0)));
        assert_eq!(parse_node_version("nonsense"), None);
        assert!(meets_minimum((22, 19, 0)));
        assert!(!meets_minimum((22, 18, 9)));
        assert!(meets_minimum((24, 0, 0)));
        assert!(!meets_minimum((20, 99, 0)));
    }

    fn fake_node(dir: &Path, version: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let node = dir.join("node");
        std::fs::write(&node, format!("#!/bin/sh\necho {version}\n")).unwrap();
        std::fs::set_permissions(&node, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn node_is_found_on_the_path_and_its_version_checked() {
        let _turn = NODE.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, paths) = paths();
        let bin = dir.path().join("elsewhere");
        fake_node(&bin, "v22.23.2");
        assert!(node_ok(&paths, bin.to_str().unwrap()));
        fake_node(&bin, "v20.1.0");
        assert!(!node_ok(&paths, bin.to_str().unwrap()));
        assert!(!node_ok(&paths, "/nonexistent"));
    }

    #[test]
    fn the_installers_node_link_wins_over_the_path() {
        let _turn = NODE.lock().unwrap_or_else(|e| e.into_inner());
        let (dir, paths) = paths();
        fake_node(&paths.bin(), "v22.23.2");
        let old = dir.path().join("old");
        fake_node(&old, "v18.0.0");
        assert_eq!(locate_node(&paths, old.to_str().unwrap()), Some(paths.bin().join("node")));
        assert!(node_ok(&paths, old.to_str().unwrap()));
    }

    #[test]
    fn the_claude_code_link_wins_over_the_path_and_a_missing_one_is_none() {
        let (dir, paths) = paths();
        assert_eq!(locate_tool(&paths, "claude", "/nonexistent"), None);
        let elsewhere = dir.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::fs::write(elsewhere.join("claude"), "#!/bin/sh\n").unwrap();
        assert_eq!(locate_tool(&paths, "claude", elsewhere.to_str().unwrap()), Some(elsewhere.join("claude")));
        std::fs::create_dir_all(paths.bin()).unwrap();
        std::fs::write(paths.bin().join("claude"), "#!/bin/sh\n").unwrap();
        assert_eq!(locate_tool(&paths, "claude", elsewhere.to_str().unwrap()), Some(paths.bin().join("claude")));
    }

    #[test]
    fn the_shim_is_executable_and_execs_the_binary() {
        let (_dir, paths) = paths();
        let binary = fake_install(&paths, "1.3.0", 0o755);
        let install = locate(&paths, "1.3.0").unwrap();
        let shim = write_shim(&paths, &install, false).unwrap();
        assert_eq!(shim, paths.bin().join("pi"));
        assert_eq!(std::fs::metadata(&shim).unwrap().permissions().mode() & 0o777, 0o755);
        let text = std::fs::read_to_string(&shim).unwrap();
        assert!(text.starts_with("#!/bin/sh\n"));
        assert!(text.contains(&format!("exec '{}' \"$@\"", binary.display())));
    }

    fn run_shim(paths: &AriaPaths, save_sessions: bool, args: &[&str]) -> String {
        let bin = paths.root.join("echo-args");
        std::fs::write(&bin, "#!/bin/sh\nprintf '%s|' \"$@\"\n").unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let install = PiInstall { dir: paths.root.clone(), binary: bin };
        let shim = write_shim(paths, &install, save_sessions).unwrap();
        String::from_utf8(std::process::Command::new(&shim).args(args).output().unwrap().stdout).unwrap()
    }

    #[test]
    fn the_shim_keeps_no_session_by_default() {
        let (_dir, paths) = paths();
        assert_eq!(
            run_shim(&paths, false, &["--mode", "json", "-p", "--no-session", "Task: x y"]),
            "--mode|json|-p|--no-session|Task: x y|"
        );
    }

    #[test]
    fn with_saved_sessions_the_shim_gives_a_child_a_session_dir() {
        let (_dir, paths) = paths();
        let dir = paths.agent().join("sessions").join("subagents");
        assert_eq!(
            run_shim(&paths, true, &["--mode", "json", "-p", "--no-session", "Task: x y"]),
            format!("--mode|json|-p|--session-dir|{}|Task: x y|", dir.display())
        );
    }

    #[test]
    fn bin_goes_first_on_the_path() {
        let (_dir, paths) = paths();
        assert_eq!(path_with_bin(&paths, "/usr/bin"), format!("{}:/usr/bin", paths.bin().display()));
    }
}
