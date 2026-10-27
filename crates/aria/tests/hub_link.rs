//! `aria hub-link machine <name>` as the binary runs it: the forced command of a machine's key. It takes exactly
//! `machine <name>` with a valid name from its own argv, never anything the connecting side asked to run, and prints
//! nothing on stdout of its own.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

/// Where the binary looks for the hub's machines socket with `XDG_RUNTIME_DIR` (Linux) or `ARIA_HOME` (macOS) set to `dir`.
fn machines_socket(dir: &Path) -> PathBuf {
    if cfg!(target_os = "macos") {
        dir.join("run").join("machines.sock")
    } else {
        dir.join("aria").join("machines.sock")
    }
}

fn link(dir: &Path, args: &[&str], original_command: Option<&str>) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_aria"));
    command
        .arg("hub-link")
        .args(args)
        .env("XDG_RUNTIME_DIR", dir)
        .env("ARIA_HOME", dir)
        .env("HOME", dir)
        .env_remove("SSH_ORIGINAL_COMMAND")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(original) = original_command {
        command.env("SSH_ORIGINAL_COMMAND", original);
    }
    command
}

fn run(mut command: Command) -> Output {
    let mut child = command.spawn().unwrap();
    drop(child.stdin.take());
    child.wait_with_output().unwrap()
}

#[test]
fn anything_but_machine_and_a_valid_name_is_refused_with_nothing_on_stdout() {
    let dir = tempfile::tempdir().unwrap();
    for args in [
        vec![],
        vec!["machine"],
        vec!["machine", "Bad Name"],
        vec!["machine", "a", "b"],
        vec!["client", "laptop"],
        vec!["machine", "a;rm -rf ~"],
        vec!["machine", ""],
    ] {
        let output = run(link(dir.path(), &args, None));
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?} wrote to stdout: {:?}", String::from_utf8_lossy(&output.stdout));
        assert!(String::from_utf8_lossy(&output.stderr).contains("usage: aria hub-link machine"), "{args:?}");
    }
}

#[test]
fn with_no_hub_it_exits_1_and_says_so_on_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let output = run(link(dir.path(), &["machine", "laptop"], Some("aria hub-link machine somebody-else")));
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no hub is running"));
}

#[test]
fn the_name_the_hub_hears_is_the_one_on_the_command_line_whatever_ssh_offered() {
    let dir = tempfile::tempdir().unwrap();
    let socket = machines_socket(dir.path());
    std::fs::create_dir_all(socket.parent().unwrap()).unwrap();
    let listener = UnixListener::bind(&socket).unwrap();
    // The connecting side asked for another command and another name: neither is read.
    let mut child =
        link(dir.path(), &["machine", "laptop"], Some("aria hub-link machine mac; cat /etc/passwd")).spawn().unwrap();
    let (mut hub, _) = listener.accept().unwrap();
    hub.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut first = String::new();
    BufReader::new(hub.try_clone().unwrap()).read_line(&mut first).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&first).unwrap(),
        serde_json::json!({"type": "link-bridge", "name": "laptop"})
    );
    // What the hub says reaches stdout, and what a machine says reaches the hub (an object only).
    writeln!(hub, "{}", serde_json::json!({"type": "link-hello", "protocol": 1})).unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    assert!(line.contains("link-hello"), "{line}");
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "not an object").unwrap();
    writeln!(stdin, "{}", serde_json::json!({"type": "link-ping"})).unwrap();
    let mut next = String::new();
    BufReader::new(hub.try_clone().unwrap()).read_line(&mut next).unwrap();
    assert!(next.contains("link-ping"), "garbage was passed on: {next}");
    // The machine's side ends first (its ssh closed): a clean exit, with the hub still there.
    drop(stdin);
    assert!(child.wait().unwrap().success());
    drop(hub);
}
