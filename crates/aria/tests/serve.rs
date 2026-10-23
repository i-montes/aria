//! `aria serve` as a process: the service with no window, started the way a unit on a computer with no desktop starts
//! it. Only the build without the `gui` feature is the one this is for (`cargo test -p aria --no-default-features`).
#![cfg(all(unix, not(feature = "gui")))]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// A computer of its own: Aria's home, its runtime directory and the `PATH` the service sees, all in one temporary folder
/// (the `PATH` holds nothing, so there is no `secret-tool` and no `omarchy-*`; no `DISPLAY` or `WAYLAND_DISPLAY` either).
struct Machine {
    dir: tempfile::TempDir,
}

impl Machine {
    fn new() -> Machine {
        let dir = tempfile::tempdir().unwrap();
        for sub in ["home", "run", "bin"] {
            std::fs::create_dir(dir.path().join(sub)).unwrap();
        }
        Machine { dir }
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_aria"));
        command
            .args(args)
            .env_clear()
            .env("HOME", self.dir.path().join("home"))
            .env("ARIA_HOME", self.dir.path().join("home/.aria"))
            .env("XDG_RUNTIME_DIR", self.dir.path().join("run"))
            .env("PATH", self.dir.path().join("bin"))
            .stdin(Stdio::null());
        command
    }

    /// `aria serve`, its log (what it says on stderr) in a file of the machine.
    fn serve(&self) -> Serve {
        let log = self.dir.path().join(format!("serve-{}.log", self.dir.path().read_dir().unwrap().count()));
        let stderr = std::fs::File::create(&log).unwrap();
        let child = self.command(&["serve"]).stdout(Stdio::null()).stderr(stderr).spawn().unwrap();
        Serve { child, log }
    }

    fn clients_socket(&self) -> PathBuf {
        self.dir.path().join("run/aria/clients.sock")
    }
}

struct Serve {
    child: Child,
    log: PathBuf,
}

impl Serve {
    fn log(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }

    /// Waits for the service to say it is serving (its sockets are bound by then).
    fn ready(&self, socket: &Path) {
        let start = Instant::now();
        while !socket.exists() || !self.log().contains("serving with no window") {
            assert!(start.elapsed() < Duration::from_secs(20), "the service never came up:\n{}", self.log());
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn exit_within(&mut self, patience: Duration) -> std::process::ExitStatus {
        let start = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(start.elapsed() < patience, "the service did not exit:\n{}", self.log());
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Serve {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn line_of(reader: &mut impl BufRead) -> serde_json::Value {
    let mut line = String::new();
    reader.read_line(&mut line).expect("a line from the service");
    serde_json::from_str(&line).unwrap_or_else(|e| panic!("not a JSON line ({e}): {line:?}"))
}

#[test]
fn serve_answers_a_terminal_without_a_display() {
    let machine = Machine::new();
    let serve = machine.serve();
    serve.ready(&machine.clients_socket());

    let stream = UnixStream::connect(machine.clients_socket()).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = stream;
    writeln!(writer, r#"{{"type": "hello", "protocol": 1, "version": "0.0.0"}}"#).unwrap();
    let hello = line_of(&mut reader);
    assert_eq!(hello["type"], "hello", "{hello}");
    writeln!(writer, r#"{{"type": "sessionsLoad", "id": 1}}"#).unwrap();
    loop {
        let reply = line_of(&mut reader);
        if reply["type"] == "sessions" {
            assert_eq!(reply["id"], 1, "{reply}");
            break;
        }
    }
}

#[test]
fn aria_quit_stops_serve() {
    let machine = Machine::new();
    let mut serve = machine.serve();
    serve.ready(&machine.clients_socket());

    let quit = machine.command(&["quit"]).output().unwrap();
    assert!(quit.status.success(), "{quit:?}");
    assert!(String::from_utf8_lossy(&quit.stdout).contains("Aria is stopping"), "{quit:?}");
    let status = serve.exit_within(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "{}", serve.log());
}

#[test]
fn a_second_serve_says_already_running() {
    let machine = Machine::new();
    let first = machine.serve();
    first.ready(&machine.clients_socket());

    let second = machine.command(&["serve"]).output().unwrap();
    assert!(second.status.success(), "{second:?}");
    assert!(String::from_utf8_lossy(&second.stderr).contains("already running"), "{second:?}");
}

#[test]
fn serve_keeps_keys_in_plain_files_when_there_is_no_keyring() {
    // The machine's PATH holds nothing: no `secret-tool`, so there is no keyring to move the keys into.
    let machine = Machine::new();
    let serve = machine.serve();
    serve.ready(&machine.clients_socket());
    assert!(serve.log().contains("keys stay in plain files"), "{}", serve.log());
}

#[test]
fn serve_exits_1_and_says_why_when_terminals_cannot_connect() {
    // A folder where the clients socket must go: nothing can bind there, and a service nobody can reach is no service.
    let machine = Machine::new();
    std::fs::create_dir_all(machine.clients_socket()).unwrap();
    let mut serve = machine.serve();
    let status = serve.exit_within(Duration::from_secs(20));
    assert_eq!(status.code(), Some(1), "{}", serve.log());
    let log = serve.log();
    assert!(log.contains("aria: terminals cannot connect"), "{log}");
    assert!(log.contains("clients.sock"), "the message names the socket:\n{log}");
    assert!(!log.contains("serving with no window"), "{log}");
}
