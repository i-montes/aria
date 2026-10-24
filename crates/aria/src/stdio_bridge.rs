//! `aria client-stdio`: the bridge `ssh <host> aria client-stdio` runs on the machine that has the hub. Its stdin and
//! stdout are the client's lines; its other end is the hub's `clients.sock`. Nothing of Aria listens on a port: the
//! transport is ssh.
//!
//! The hub marks a connection remote because the first line says so, so it is only safe because this bridge is what
//! every remote line goes through: it reads the client's own `hello` (and checks its protocol itself), writes its own
//! `hello` with `remote: true` to the hub, and from then on relays only lines it parsed as JSON, never a `hello`. A key
//! that can only run this command cannot reach the hub in any other way. A line is read up to `MAX_LINE` bytes and no
//! further: nothing longer is ever buffered.
//!
//! Exit codes (`Exit`): 0 the client closed, 1 the hub is down (or went away), 3 another protocol, 4 a line over the cap.

use aria_core::bridge::PROTOCOL;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::mpsc;

/// The longest line the bridge takes from the client (a pasted file in a message, with room to spare).
pub const MAX_LINE: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// The client closed its side.
    Done = 0,
    /// No hub on the socket, or it went away.
    HubDown = 1,
    /// The client or the hub speaks another protocol (or the client's first line was no `hello`).
    Mismatch = 3,
    /// A line over `MAX_LINE`.
    Garbage = 4,
}

/// What reading one line gave.
enum Next {
    Line(Vec<u8>),
    Eof,
    /// The line went past the cap before its end: what came after is not read.
    TooLong,
}

/// One line (without its newline), never holding more than `max` bytes of it.
fn next_line(reader: &mut impl BufRead, max: usize) -> std::io::Result<Next> {
    let mut line = Vec::new();
    loop {
        let buffered = match reader.fill_buf() {
            Ok(buffered) => buffered,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if buffered.is_empty() {
            return Ok(if line.is_empty() { Next::Eof } else { Next::Line(line) });
        }
        let end = buffered.iter().position(|byte| *byte == b'\n');
        let take = end.unwrap_or(buffered.len());
        if line.len() + take > max {
            return Ok(Next::TooLong);
        }
        line.extend_from_slice(&buffered[..take]);
        reader.consume(take + usize::from(end.is_some()));
        if end.is_some() {
            return Ok(Next::Line(line));
        }
    }
}

fn write_line(out: &mut impl Write, value: &Value) -> std::io::Result<()> {
    out.write_all(format!("{value}\n").as_bytes())?;
    out.flush()
}

fn error(code: &str) -> Value {
    json!({"type": "error", "id": 0, "code": code})
}

/// Runs the bridge until one side ends. `input` and `output` are the client's; the hub is on `socket`.
pub fn run(input: impl Read + Send + 'static, output: impl Write + Send + 'static, socket: &Path) -> Exit {
    let mut input = BufReader::new(input);
    let mut output = output;
    let first = match next_line(&mut input, MAX_LINE) {
        Ok(Next::Line(line)) => line,
        Ok(Next::TooLong) => return Exit::Garbage,
        Ok(Next::Eof) | Err(_) => return Exit::Done,
    };
    match serde_json::from_slice::<Value>(&first) {
        Ok(hello) if hello["type"] == "hello" => {
            if hello["protocol"].as_u64() != Some(u64::from(PROTOCOL)) {
                let version = env!("CARGO_PKG_VERSION");
                let _ = write_line(&mut output, &json!({"type": "hello", "protocol": PROTOCOL, "version": version}));
                let _ = write_line(&mut output, &error("protocol-mismatch"));
                return Exit::Mismatch;
            }
        }
        _ => {
            let _ = write_line(&mut output, &error("protocol-mismatch"));
            return Exit::Mismatch;
        }
    }
    let Ok(hub) = UnixStream::connect(socket) else {
        let _ = write_line(&mut output, &error("hub-down"));
        return Exit::HubDown;
    };
    let (Ok(mut to_hub), Ok(from_hub), Ok(closer)) = (hub.try_clone(), hub.try_clone(), hub.try_clone()) else {
        let _ = write_line(&mut output, &error("hub-down"));
        return Exit::HubDown;
    };
    let hello = json!({"type": "hello", "protocol": PROTOCOL, "version": env!("CARGO_PKG_VERSION"), "remote": true});
    if write_line(&mut to_hub, &hello).is_err() {
        let _ = write_line(&mut output, &error("hub-down"));
        return Exit::HubDown;
    }

    let (ended, end) = mpsc::channel::<Exit>();
    // The hub's lines go to the client as they are, until the hub closes.
    let hub_ended = ended.clone();
    std::thread::spawn(move || {
        let mut lines = BufReader::new(from_hub);
        let mut line = Vec::new();
        let exit = loop {
            line.clear();
            match lines.read_until(b'\n', &mut line) {
                Ok(0) | Err(_) => break Exit::HubDown,
                Ok(_) => {}
            }
            if output.write_all(&line).and_then(|_| output.flush()).is_err() {
                break Exit::Done;
            }
            // A short line only: an error of the hub's is small, and nothing else is worth parsing.
            if line.len() < 256
                && serde_json::from_slice::<Value>(&line).is_ok_and(|v| v["code"] == "protocol-mismatch")
            {
                break Exit::Mismatch;
            }
        };
        let _ = hub_ended.send(exit);
    });
    // The client's lines go to the hub when they are JSON objects and not a `hello`.
    std::thread::spawn(move || {
        let exit = loop {
            let line = match next_line(&mut input, MAX_LINE) {
                Ok(Next::Line(line)) => line,
                Ok(Next::TooLong) => break Exit::Garbage,
                Ok(Next::Eof) | Err(_) => break Exit::Done,
            };
            let Ok(value) = serde_json::from_slice::<Value>(&line) else { continue };
            if !value.is_object() || value["type"] == "hello" {
                continue;
            }
            if write_line(&mut to_hub, &value).is_err() {
                break Exit::HubDown;
            }
        };
        let _ = ended.send(exit);
    });
    let exit = end.recv().unwrap_or(Exit::Done);
    // One side is over: closing the hub's end ends the hub's client, and the thread reading it.
    let _ = closer.shutdown(std::net::Shutdown::Both);
    exit
}

/// `aria client-stdio`: the bridge on this process's stdin and stdout, to the hub of this machine. Returns the exit code.
pub fn command() -> i32 {
    run(std::io::stdin(), std::io::stdout(), &crate::clients::socket_path()) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::{self, Hub};
    use aria_core::bridge::{PageMessage, PROTOCOL};
    use serde_json::{json, Value};
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    fn wait_until(what: &str, done: impl Fn() -> bool) {
        let start = Instant::now();
        while !done() {
            assert!(start.elapsed() < Duration::from_secs(5), "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// A real hub on a socket in a temp folder.
    struct Live {
        hub: Hub,
        chat: mpsc::Receiver<PageMessage>,
        handled: mpsc::Receiver<PageMessage>,
        socket: PathBuf,
        _dir: tempfile::TempDir,
    }

    fn live() -> Live {
        let (chat_tx, chat) = mpsc::channel();
        let (handled_tx, handled) = mpsc::channel();
        let handled_tx = std::sync::Mutex::new(handled_tx);
        let hub = Hub::new(
            chat_tx,
            std::sync::Arc::new(move |m| {
                let _ = handled_tx.lock().unwrap().send(m);
            }),
            std::sync::Arc::new(|_| true),
        );
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("run").join("clients.sock");
        hub.serve(clients::bind(&socket).unwrap());
        Live { hub, chat, handled, socket, _dir: dir }
    }

    /// A stand-in for the hub that records every line the bridge writes to it.
    fn fake_hub() -> (PathBuf, mpsc::Receiver<String>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("clients.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let (seen, lines) = mpsc::channel();
        std::thread::spawn(move || {
            let Ok((stream, _)) = listener.accept() else { return };
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                let _ = seen.send(line);
            }
        });
        (socket, lines, dir)
    }

    /// The bridge running on a thread, and the test's end of the client side (what ssh's stdin and stdout are).
    struct Bridge {
        out: UnixStream,
        lines: BufReader<UnixStream>,
        exit: std::thread::JoinHandle<Exit>,
    }

    fn start(socket: &Path) -> Bridge {
        let (ours, theirs) = UnixStream::pair().unwrap();
        ours.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let input = theirs.try_clone().unwrap();
        let socket = socket.to_path_buf();
        let exit = std::thread::spawn(move || run(input, theirs, &socket));
        Bridge { lines: BufReader::new(ours.try_clone().unwrap()), out: ours, exit }
    }

    impl Bridge {
        fn send(&mut self, value: Value) {
            writeln!(self.out, "{value}").unwrap();
        }
        fn raw(&mut self, bytes: &[u8]) {
            self.out.write_all(bytes).unwrap();
        }
        fn next(&mut self) -> Value {
            let mut line = String::new();
            self.lines.read_line(&mut line).unwrap();
            serde_json::from_str(&line).unwrap_or_else(|_| panic!("not a line of JSON: {line:?}"))
        }
        fn exit(self) -> Exit {
            wait_until("the bridge to end", || self.exit.is_finished());
            self.exit.join().unwrap()
        }
    }

    fn hello(protocol: u32, remote: Option<bool>) -> Value {
        let mut line = json!({"type": "hello", "protocol": protocol, "version": "9.9.9"});
        if let Some(remote) = remote {
            line["remote"] = json!(remote);
        }
        line
    }

    fn focus() -> Value {
        json!({"type": "focus", "focused": false})
    }

    #[test]
    fn the_client_gets_the_hubs_hello_and_the_bridge_marks_it_remote() {
        let hub = live();
        let mut bridge = start(&hub.socket);
        // The client says it is not remote: the bridge is what the hub trusts, and it says remote.
        bridge.send(hello(PROTOCOL, Some(false)));
        assert_eq!(bridge.next(), json!({"type": "hello", "protocol": PROTOCOL, "version": env!("CARGO_PKG_VERSION")}));
        assert_eq!(hub.hub.count(), 1);
        bridge.send(json!({"type": "secretSave", "id": 5, "name": "GH_TOKEN", "value": "v", "purpose": ""}));
        assert_eq!(bridge.next(), json!({"type": "error", "id": 5, "code": "not-for-remote"}));
        bridge.send(json!({"type": "session", "session": "aa", "message": {"type": "piCommand", "id": 6, "command": "bash", "args": {"command": "pwd"}}}));
        assert_eq!(bridge.next(), json!({"type": "error", "id": 6, "code": "not-for-remote"}));
        assert!(hub.handled.try_recv().is_err(), "no secret reached the handlers");
        assert!(hub.chat.try_recv().is_err(), "no command reached the chat");
        // What a remote client may do passes through.
        bridge.send(json!({"type": "session", "session": "aa", "message": {"type": "chatAbort"}}));
        assert!(matches!(hub.chat.recv_timeout(Duration::from_secs(5)).unwrap(), PageMessage::Session { .. }));
    }

    #[test]
    fn a_clients_own_hello_is_not_forwarded() {
        let (socket, seen, _dir) = fake_hub();
        let mut bridge = start(&socket);
        bridge.send(hello(PROTOCOL, Some(false)));
        bridge.send(focus());
        let first: Value = serde_json::from_str(&seen.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap();
        assert_eq!(
            first,
            json!({"type": "hello", "protocol": PROTOCOL, "version": env!("CARGO_PKG_VERSION"), "remote": true})
        );
        let second: Value = serde_json::from_str(&seen.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap();
        assert_eq!(second, focus(), "the client's hello went nowhere");
    }

    #[test]
    fn a_later_hello_from_the_client_is_dropped() {
        let (socket, seen, _dir) = fake_hub();
        let mut bridge = start(&socket);
        bridge.send(hello(PROTOCOL, None));
        bridge.send(hello(PROTOCOL, Some(false)));
        bridge.send(focus());
        let lines: Vec<Value> = (0..2)
            .map(|_| serde_json::from_str(&seen.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap())
            .collect();
        assert_eq!(lines[0]["remote"], true);
        assert_eq!(lines[1], focus(), "the second hello was not relayed");
        assert!(seen.recv_timeout(Duration::from_millis(200)).is_err());
    }

    #[test]
    fn lines_that_are_not_json_are_dropped_not_relayed() {
        let (socket, seen, _dir) = fake_hub();
        let mut bridge = start(&socket);
        bridge.send(hello(PROTOCOL, None));
        bridge.raw(b"this is not json\n[1, 2\n\n{\"type\": \n");
        bridge.send(focus());
        let first = seen.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(first.contains("\"remote\":true"));
        let second: Value = serde_json::from_str(&seen.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap();
        assert_eq!(second, focus(), "only the line that is JSON got through");
        assert!(seen.recv_timeout(Duration::from_millis(200)).is_err());
    }

    #[test]
    fn a_first_line_that_is_not_a_hello_ends_with_mismatch() {
        // No hub is running: a bridge that tried to connect would end with HubDown instead.
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = start(&dir.path().join("clients.sock"));
        bridge.send(focus());
        assert_eq!(bridge.next(), json!({"type": "error", "id": 0, "code": "protocol-mismatch"}));
        assert_eq!(bridge.exit(), Exit::Mismatch);
        let mut bridge = start(&dir.path().join("clients.sock"));
        bridge.raw(b"not even json\n");
        assert_eq!(bridge.next()["code"], "protocol-mismatch");
        assert_eq!(bridge.exit(), Exit::Mismatch);
    }

    #[test]
    fn a_different_protocol_ends_with_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = start(&dir.path().join("clients.sock"));
        bridge.send(hello(PROTOCOL + 1, None));
        assert_eq!(bridge.next()["type"], "hello");
        assert_eq!(bridge.next(), json!({"type": "error", "id": 0, "code": "protocol-mismatch"}));
        assert_eq!(bridge.exit(), Exit::Mismatch);
    }

    #[test]
    fn a_hub_with_another_protocol_ends_with_mismatch() {
        // The hub answers an error of its own; the client sees it, and the exit code says why.
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("clients.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(stream.try_clone().unwrap()).read_line(&mut line).unwrap();
            writeln!(stream, "{}", json!({"type": "hello", "protocol": 2, "version": "9.9.9"})).unwrap();
            writeln!(stream, "{}", json!({"type": "error", "id": 0, "code": "protocol-mismatch"})).unwrap();
        });
        let mut bridge = start(&socket);
        bridge.send(hello(PROTOCOL, None));
        assert_eq!(bridge.next()["type"], "hello");
        assert_eq!(bridge.next()["code"], "protocol-mismatch");
        assert_eq!(bridge.exit(), Exit::Mismatch);
    }

    #[test]
    fn a_hub_that_is_down_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = start(&dir.path().join("missing.sock"));
        bridge.send(hello(PROTOCOL, None));
        assert_eq!(bridge.next(), json!({"type": "error", "id": 0, "code": "hub-down"}));
        assert_eq!(bridge.exit(), Exit::HubDown);
    }

    #[test]
    fn a_client_that_closes_ends_the_bridge_and_the_hub_side() {
        let hub = live();
        let mut bridge = start(&hub.socket);
        bridge.send(hello(PROTOCOL, None));
        bridge.next();
        wait_until("connected", || hub.hub.count() == 1);
        bridge.out.shutdown(std::net::Shutdown::Write).unwrap();
        let Bridge { exit, .. } = bridge;
        wait_until("the bridge to end", || exit.is_finished());
        assert_eq!(exit.join().unwrap(), Exit::Done);
        wait_until("the hub side to close", || hub.hub.count() == 0);
    }

    #[test]
    fn a_hub_that_goes_away_ends_the_bridge() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("clients.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        std::thread::spawn(move || drop(listener.accept()));
        let mut bridge = start(&socket);
        bridge.send(hello(PROTOCOL, None));
        assert_eq!(bridge.exit(), Exit::HubDown);
    }

    #[test]
    fn a_line_over_the_cap_ends_the_bridge_and_drops_the_hub_side() {
        let hub = live();
        let mut bridge = start(&hub.socket);
        bridge.send(hello(PROTOCOL, None));
        bridge.next();
        wait_until("connected", || hub.hub.count() == 1);
        let mut out = bridge.out.try_clone().unwrap();
        std::thread::spawn(move || {
            let chunk = vec![b'x'; 64 * 1024];
            let mut sent = 0;
            while sent <= MAX_LINE {
                if out.write_all(&chunk).is_err() {
                    break;
                }
                sent += chunk.len();
            }
        });
        assert_eq!(bridge.exit(), Exit::Garbage);
        wait_until("the hub side to close", || hub.hub.count() == 0);
    }

    #[test]
    fn a_first_line_over_the_cap_ends_the_bridge_before_any_hub() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = start(&dir.path().join("clients.sock"));
        let mut out = bridge.out.try_clone().unwrap();
        std::thread::spawn(move || {
            let chunk = vec![b'x'; 64 * 1024];
            let mut sent = 0;
            while sent <= MAX_LINE {
                if out.write_all(&chunk).is_err() {
                    break;
                }
                sent += chunk.len();
            }
        });
        assert_eq!(bridge.exit(), Exit::Garbage);
    }

    #[test]
    fn a_line_of_exactly_the_cap_is_still_read() {
        let mut reader = std::io::Cursor::new([vec![b'x'; 10], b"\nrest".to_vec()].concat());
        assert!(matches!(next_line(&mut reader, 10).unwrap(), Next::Line(l) if l.len() == 10));
        assert!(matches!(next_line(&mut reader, 10).unwrap(), Next::Line(l) if l == b"rest"));
        assert!(matches!(next_line(&mut reader, 10).unwrap(), Next::Eof));
        let mut reader = std::io::Cursor::new(vec![b'x'; 11]);
        assert!(matches!(next_line(&mut reader, 10).unwrap(), Next::TooLong));
    }
}
