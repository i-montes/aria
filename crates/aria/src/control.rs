//! One dock at a time, reached through a Unix socket. On Omarchy it lives in `$XDG_RUNTIME_DIR/aria/`
//! (or `/tmp/aria/`); on macOS it lives in `~/.aria/run/`, so a dock started by the LaunchAgent
//! and a dock started from a terminal agree on the same path. `aria toggle` sends a line to the running
//! dock, and starts the dock first when nothing answers. `aria update-ready` (the installer's) asks the running dock to
//! restart into the new binary once nothing is running, and waits for its answer.

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
        let runtime = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp"));
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
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
            .map_err(|e| ControlError::Io(e.to_string()))?;
    }
    let _ = std::fs::remove_file(path);
    UnixListener::bind(path).map_err(|e| ControlError::Io(e.to_string()))
}

pub fn send(path: &Path, command: &str) -> std::io::Result<()> {
    let mut stream = UnixStream::connect(path)?;
    stream.write_all(format!("{command}\n").as_bytes())
}

/// Sends `command` and waits up to `timeout` for the dock's one-line answer. A dock that does not know the command
/// (an older version) closes the connection without one, which is an error.
pub fn ask(path: &Path, command: &str, timeout: Duration) -> std::io::Result<String> {
    use std::io::{BufRead, BufReader};
    let mut stream = UnixStream::connect(path)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.write_all(format!("{command}\n").as_bytes())?;
    let mut answer = String::new();
    BufReader::new(stream).read_line(&mut answer)?;
    if answer.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "the dock did not answer"));
    }
    Ok(answer.trim_end().to_string())
}

/// The path to run the installed binary from. On Linux the running binary's path reads "… (deleted)" once the
/// installer has replaced the file; the new one is at the same path.
pub fn binary_path(current: PathBuf) -> PathBuf {
    match current.to_str().and_then(|path| path.strip_suffix(" (deleted)")) {
        Some(path) => PathBuf::from(path),
        None => current,
    }
}

/// Replaces this process with the binary now installed at its path, with the same arguments and environment. The
/// process id stays, so systemd and launchd keep supervising it. Returns only if the exec failed.
pub fn reexec() -> std::io::Error {
    use std::os::unix::process::CommandExt;
    let exe = match std::env::current_exe() {
        Ok(exe) => binary_path(exe),
        Err(error) => return error,
    };
    Command::new(exe).args(std::env::args_os().skip(1)).exec()
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
    fn ask_returns_the_docks_answer() {
        use std::io::{BufRead, BufReader};
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("control.sock");
        let listener = bind_single_instance(&path).unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            assert_eq!(line, "update-ready\n");
            stream.write_all(b"ok\n").unwrap();
        });
        assert_eq!(ask(&path, "update-ready", Duration::from_secs(2)).unwrap(), "ok");
    }

    #[test]
    fn ask_fails_when_the_dock_does_not_answer() {
        let _turn = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("control.sock");
        // An older dock reads the line and drops it without a word.
        let listener = bind_single_instance(&path).unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            drop(stream);
        });
        assert!(ask(&path, "update-ready", Duration::from_millis(500)).is_err());
    }

    #[test]
    fn the_replaced_binary_is_run_by_its_path() {
        assert_eq!(
            binary_path(PathBuf::from("/home/me/.local/bin/aria (deleted)")),
            PathBuf::from("/home/me/.local/bin/aria")
        );
        assert_eq!(binary_path(PathBuf::from("/Users/me/.local/bin/aria")), PathBuf::from("/Users/me/.local/bin/aria"));
    }

    #[test]
    fn send_without_an_instance_fails() {
        let dir = tempfile::tempdir().unwrap();
        assert!(send(&dir.path().join("control.sock"), "toggle").is_err());
    }
}
