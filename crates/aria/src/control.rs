//! One dock at a time, reached through a Unix socket. On Omarchy it lives in `$XDG_RUNTIME_DIR/aria/`
//! (or the temp directory); on macOS it lives in `~/.aria/run/`, so a dock started by the LaunchAgent
//! and a dock started from a terminal agree on the same path. `aria toggle` sends a line to the running
//! dock, and starts the dock first when nothing answers.

use std::io::Write;
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

#[derive(Debug)]
pub enum ControlError {
    AlreadyRunning,
    Io(String),
}

impl std::fmt::Display for ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ControlError::AlreadyRunning => write!(f, "the dock is already running"),
            ControlError::Io(message) => write!(f, "{message}"),
        }
    }
}

pub fn control_socket_path() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        let runtime = std::env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        runtime.join("aria").join("control.sock")
    }
    #[cfg(target_os = "macos")]
    {
        // A LaunchAgent and a terminal do not always share TMPDIR; ~/.aria/run is deterministic. ARIA_HOME
        // keeps a development instance separate, the same way the rest of Aria's data works.
        let root = std::env::var_os("ARIA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".aria"));
        root.join("run").join("control.sock")
    }
}

/// Binds the control socket. A live dock on it means `AlreadyRunning`; a file nobody listens on
/// (left by a crash) is replaced.
pub fn bind_single_instance(path: &Path) -> Result<UnixListener, ControlError> {
    if UnixStream::connect(path).is_ok() {
        return Err(ControlError::AlreadyRunning);
    }
    if let Some(dir) = path.parent() {
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir).map_err(|e| ControlError::Io(e.to_string()))?;
    }
    let _ = std::fs::remove_file(path);
    UnixListener::bind(path).map_err(|e| ControlError::Io(e.to_string()))
}

pub fn send(path: &Path, command: &str) -> std::io::Result<()> {
    let mut stream = UnixStream::connect(path)?;
    stream.write_all(format!("{command}\n").as_bytes())
}

/// `aria toggle` / `aria show`: send `command`, starting `exe` (detached) first when no dock is running.
pub fn command_or_start(path: &Path, exe: &Path, command: &str) -> Result<(), ControlError> {
    if send(path, command).is_ok() {
        return Ok(());
    }
    Command::new(exe)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| ControlError::Io(e.to_string()))?;
    for _ in 0..60 {
        std::thread::sleep(Duration::from_millis(50));
        if send(path, command).is_ok() {
            return Ok(());
        }
    }
    Err(ControlError::Io("the dock did not start".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;

    #[test]
    fn second_bind_is_already_running() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("aria").join("control.sock");
        let _first = bind_single_instance(&path).unwrap();
        assert!(matches!(bind_single_instance(&path), Err(ControlError::AlreadyRunning)));
    }

    #[test]
    fn stale_socket_is_replaced() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("control.sock");
        drop(UnixListener::bind(&path).unwrap()); // the file stays behind, nobody listens (a crash)
        assert!(path.exists());
        assert!(bind_single_instance(&path).is_ok());
    }

    #[test]
    fn send_reaches_the_running_instance() {
        use std::io::{BufRead, BufReader};
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("control.sock");
        let listener = bind_single_instance(&path).unwrap();
        send(&path, "toggle").unwrap();
        let (stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).unwrap();
        assert_eq!(line, "toggle\n");
    }

    #[test]
    fn send_without_an_instance_fails() {
        let dir = tempfile::tempdir().unwrap();
        assert!(send(&dir.path().join("control.sock"), "toggle").is_err());
    }
}
