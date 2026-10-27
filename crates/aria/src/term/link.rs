//! The client's connection to the dock (`clients.sock`, or `aria client-stdio` over ssh): the page's messages out, the
//! dock's in, one JSON object per line. A connection starts with a `hello` each way (`Link::over`). A message the
//! client does not know (a newer dock) is skipped; a dock that goes away is said once, as
//! `error { id: 0, code: "gone" }`.

use aria_core::bridge::{DockMessage, PageMessage, PROTOCOL};
use serde_json::{json, Value};
use std::fmt;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;
use std::time::Duration;

/// How long the hub on the local socket has to answer the `hello` before the client gives up.
const SOCKET_HELLO_TIMEOUT: Duration = Duration::from_secs(10);

/// The same over ssh: its clock starts before ssh has logged in, so a first-contact host-key prompt, a passphrase or a
/// slow login profile has to fit in it.
pub const SSH_HELLO_TIMEOUT: Duration = Duration::from_secs(60);

/// Why a hub did not open the link (the `io::Error` of `Link::over` carries one of these, as `InvalidData`; its text is
/// the code).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkError {
    /// The hub speaks another protocol (`theirs` when its `hello` said which).
    ProtocolMismatch { theirs: Option<u64> },
    /// The bridge could not reach the hub (it is not running).
    HubDown,
    /// The hub does not know the `hello`, or answered something that is not one.
    HubTooOld,
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LinkError::ProtocolMismatch { .. } => "protocol-mismatch",
            LinkError::HubDown => "hub-down",
            LinkError::HubTooOld => "hub-too-old",
        })
    }
}

impl std::error::Error for LinkError {}

impl LinkError {
    /// The reason inside an error `Link` gave, if it is one.
    pub fn of(error: &io::Error) -> Option<&LinkError> {
        error.get_ref().and_then(|inner| inner.downcast_ref::<LinkError>())
    }

    fn io(self) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, self)
    }
}

pub struct Link {
    out: Mutex<Box<dyn Write + Send>>,
    events: Mutex<Option<Receiver<DockMessage>>>,
    /// The ssh a link over `spawn` runs on: killed and waited on when the link goes.
    child: Option<Child>,
}

impl Link {
    /// The dock on a unix socket (a terminal on the same machine).
    pub fn connect(path: &Path) -> io::Result<Link> {
        let stream = UnixStream::connect(path)?;
        let reader = stream.try_clone()?;
        let closer = stream.try_clone()?;
        Link::over(reader, stream, false).inspect_err(|_| {
            // The thread reading the dock ends with its socket.
            let _ = closer.shutdown(std::net::Shutdown::Both);
        })
    }

    /// The hub behind `command`'s stdin and stdout (`ssh host aria client-stdio`); the child is killed and waited on
    /// when the link is dropped, or when it does not open.
    pub fn spawn(command: &mut Command) -> io::Result<Link> {
        Link::spawn_within(command, SSH_HELLO_TIMEOUT)
    }

    pub(crate) fn spawn_within(command: &mut Command, within: Duration) -> io::Result<Link> {
        let mut child = command.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(io::Error::other("the child has no pipes"));
        };
        match Link::over_within(stdout, stdin, true, within) {
            Ok(mut link) => {
                link.child = Some(child);
                Ok(link)
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(error)
            }
        }
    }

    /// A link on any pair of streams: sends the `hello`, and waits up to 10 s (`SOCKET_HELLO_TIMEOUT`) for the hub's.
    pub fn over(
        reader: impl Read + Send + 'static,
        writer: impl Write + Send + 'static,
        remote: bool,
    ) -> io::Result<Link> {
        Link::over_within(reader, writer, remote, SOCKET_HELLO_TIMEOUT)
    }

    fn over_within(
        reader: impl Read + Send + 'static,
        writer: impl Write + Send + 'static,
        remote: bool,
        within: Duration,
    ) -> io::Result<Link> {
        let mut writer: Box<dyn Write + Send> = Box::new(writer);
        let hello =
            json!({"type": "hello", "protocol": PROTOCOL, "version": env!("CARGO_PKG_VERSION"), "remote": remote});
        writeln!(writer, "{hello}")?;
        writer.flush()?;
        let (first, hello_reply) = mpsc::channel::<io::Result<String>>();
        let (sender, events) = mpsc::channel();
        // The reply is waited for here, with a timeout; a pipe has none, so the reading is a thread's.
        std::thread::spawn(move || {
            let mut lines = BufReader::new(reader).lines();
            // The account's shell may print a banner or a motd before `aria client-stdio` starts: a line that is not
            // JSON is not the hub's, and is skipped (the timeout still bounds the wait).
            let answer = loop {
                match lines.next() {
                    Some(Ok(line)) if serde_json::from_str::<Value>(&line).is_err() => continue,
                    Some(line) => break line,
                    None => {
                        break Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "the connection closed before the hub answered",
                        ))
                    }
                }
            };
            let opened = answer.is_ok();
            if first.send(answer).is_err() || !opened {
                return;
            }
            for line in lines.map_while(Result::ok) {
                if let Ok(message) = serde_json::from_str::<DockMessage>(&line) {
                    if sender.send(message).is_err() {
                        return;
                    }
                }
            }
            let _ = sender.send(DockMessage::Error { id: 0, code: "gone".into() });
        });
        let line = match hello_reply.recv_timeout(within) {
            Ok(line) => line?,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "the hub did not answer"))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "the connection closed before the hub answered",
                ))
            }
        };
        match check_reply(&line) {
            None => Ok(Link { out: Mutex::new(writer), events: Mutex::new(Some(events)), child: None }),
            Some(error) => Err(error.io()),
        }
    }

    fn line(&self, value: serde_json::Value) {
        let mut out = self.out.lock().unwrap();
        let _ = writeln!(out, "{value}").and_then(|_| out.flush());
    }

    pub fn send(&self, message: &PageMessage) {
        if let Ok(value) = serde_json::to_value(message) {
            self.line(value);
        }
    }

    pub fn attach(&self, session: &str) {
        self.line(json!({"type": "attach", "session": session}));
    }

    pub fn detach(&self) {
        self.line(json!({"type": "detach"}));
    }

    pub fn focus(&self, focused: bool) {
        self.line(json!({"type": "focus", "focused": focused}));
    }

    /// What the dock says; taken once (later calls get a receiver that is already closed).
    pub fn events(&self) -> Receiver<DockMessage> {
        self.events.lock().unwrap().take().unwrap_or_else(|| mpsc::channel().1)
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// What the hub's first line says: `None` when it is a `hello` of this protocol.
fn check_reply(line: &str) -> Option<LinkError> {
    let value: Value = serde_json::from_str(line).unwrap_or(Value::Null);
    match value["type"].as_str() {
        Some("hello") => match value["protocol"].as_u64() {
            Some(theirs) if theirs == u64::from(PROTOCOL) => None,
            theirs => Some(LinkError::ProtocolMismatch { theirs }),
        },
        Some("error") => Some(match value["code"].as_str() {
            Some("protocol-mismatch") => LinkError::ProtocolMismatch { theirs: None },
            Some("hub-down") => LinkError::HubDown,
            // `not-for-clients` is an old hub that does not know the `hello`.
            _ => LinkError::HubTooOld,
        }),
        _ => Some(LinkError::HubTooOld),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::bridge::{DockMessage, PageMessage, PROTOCOL};
    use serde_json::{json, Value};
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::time::Duration;

    /// What a hub says first when it is the same protocol.
    fn hello_reply() -> Value {
        json!({"type": "hello", "protocol": PROTOCOL, "version": "9.9.9"})
    }

    /// A dock that accepts one client: the test reads what it sent and writes what the dock says.
    struct FakeDock {
        _dir: tempfile::TempDir,
        link: Link,
        from_client: BufReader<UnixStream>,
        to_client: UnixStream,
    }

    fn fake_dock() -> FakeDock {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clients.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let connecting = std::thread::spawn({
            let path = path.clone();
            move || Link::connect(&path).unwrap()
        });
        let (stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut from_client = BufReader::new(stream.try_clone().unwrap());
        let mut to_client = stream;
        let mut hello = String::new();
        from_client.read_line(&mut hello).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&hello).unwrap()["type"], "hello");
        writeln!(to_client, "{}", hello_reply()).unwrap();
        FakeDock { _dir: dir, link: connecting.join().unwrap(), from_client, to_client }
    }

    impl FakeDock {
        fn read(&mut self) -> Value {
            let mut line = String::new();
            self.from_client.read_line(&mut line).unwrap();
            serde_json::from_str(&line).unwrap()
        }
        fn say(&mut self, value: Value) {
            writeln!(self.to_client, "{value}").unwrap();
        }
    }

    /// `Link::over` on a pair of sockets whose other end reads the client's first line, then says `replies`. What the
    /// client's first line was, and the dock's end (kept open) come back with the result.
    fn open_with(replies: &[Value], remote: bool) -> (io::Result<Link>, Value, UnixStream) {
        open_lines(replies.iter().map(Value::to_string).collect(), remote)
    }

    /// The same with the dock's lines as they are on the wire (a banner is not JSON).
    fn open_lines(replies: Vec<String>, remote: bool) -> (io::Result<Link>, Value, UnixStream) {
        let (ours, theirs) = UnixStream::pair().unwrap();
        let dock = std::thread::spawn(move || {
            let mut reader = BufReader::new(theirs.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let mut theirs = theirs;
            for reply in replies {
                writeln!(theirs, "{reply}").unwrap();
            }
            (serde_json::from_str::<Value>(&line).unwrap(), theirs)
        });
        let link = Link::over(ours.try_clone().unwrap(), ours, remote);
        let (hello, theirs) = dock.join().unwrap();
        (link, hello, theirs)
    }

    fn refused(replies: &[Value]) -> io::Error {
        match open_with(replies, true).0 {
            Ok(_) => panic!("the link opened"),
            Err(error) => error,
        }
    }

    #[test]
    fn the_hello_is_sent_first_and_a_matching_reply_opens_the_link() {
        let (link, hello, mut dock) = open_with(&[hello_reply()], true);
        assert_eq!(
            hello,
            json!({"type": "hello", "protocol": PROTOCOL, "version": env!("CARGO_PKG_VERSION"), "remote": true})
        );
        let link = link.unwrap();
        let events = link.events();
        writeln!(dock, "{}", json!({"type": "sessionsChanged"})).unwrap();
        assert_eq!(events.recv_timeout(Duration::from_secs(5)).unwrap(), DockMessage::SessionsChanged);
    }

    #[test]
    fn text_before_the_hello_is_skipped_not_taken_for_an_old_hub() {
        // The hub account's shell may print a banner or a motd before `aria client-stdio` starts.
        let banner = vec![
            "Welcome to the hub".to_string(),
            "".to_string(),
            "Last login: Fri Oct 23 09:00:00 2026 from 192.0.2.1".to_string(),
            hello_reply().to_string(),
            json!({"type": "sessionsChanged"}).to_string(),
        ];
        let (link, _, _dock) = open_lines(banner, true);
        let link = link.expect("the banner is skipped");
        assert_eq!(link.events().recv_timeout(Duration::from_secs(5)).unwrap(), DockMessage::SessionsChanged);
        // a banner and then nothing is still a hub that never answers, and a banner then a refusal is that refusal
        let (link, _, _dock) = open_lines(
            vec!["Welcome".to_string(), json!({"type": "error", "id": 0, "code": "protocol-mismatch"}).to_string()],
            true,
        );
        assert_eq!(link.err().unwrap().to_string(), "protocol-mismatch");
    }

    #[test]
    fn a_local_link_says_it_is_not_remote() {
        let (link, hello, _dock) = open_with(&[hello_reply()], false);
        assert!(link.is_ok());
        assert_eq!(hello["remote"], false);
    }

    #[test]
    fn an_old_hub_that_answers_not_for_clients_is_hub_too_old() {
        let error = refused(&[json!({"type": "error", "id": 0, "code": "not-for-clients"})]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "hub-too-old");
    }

    #[test]
    fn a_hello_with_another_protocol_is_a_mismatch() {
        let error = refused(&[json!({"type": "hello", "protocol": PROTOCOL + 1, "version": "9.9.9"})]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "protocol-mismatch");
        assert_eq!(
            error.get_ref().and_then(|e| e.downcast_ref::<LinkError>()),
            Some(&LinkError::ProtocolMismatch { theirs: Some(u64::from(PROTOCOL) + 1) })
        );
    }

    #[test]
    fn a_bridge_that_says_protocol_mismatch_is_a_mismatch() {
        let error = refused(&[json!({"type": "error", "id": 0, "code": "protocol-mismatch"})]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "protocol-mismatch");
        assert_eq!(
            error.get_ref().and_then(|e| e.downcast_ref::<LinkError>()),
            Some(&LinkError::ProtocolMismatch { theirs: None })
        );
    }

    #[test]
    fn a_bridge_that_says_hub_down_is_hub_down() {
        let error = refused(&[json!({"type": "error", "id": 0, "code": "hub-down"})]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "hub-down");
    }

    #[test]
    fn any_other_first_line_is_hub_too_old() {
        for first in [
            json!({"type": "sessionsChanged"}),
            json!({"type": "error", "id": 0, "code": "something-else"}),
            json!("a string"),
        ] {
            let error = refused(std::slice::from_ref(&first));
            assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{first}");
            assert_eq!(error.to_string(), "hub-too-old", "{first}");
        }
    }

    #[test]
    fn a_hub_that_never_answers_times_out() {
        let (ours, theirs) = UnixStream::pair().unwrap();
        let started = std::time::Instant::now();
        let error = Link::over_within(ours.try_clone().unwrap(), ours, false, Duration::from_millis(200))
            .err()
            .expect("no link");
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(5));
        drop(theirs);
    }

    #[test]
    fn a_hub_that_closes_before_answering_is_an_error() {
        let (ours, theirs) = UnixStream::pair().unwrap();
        drop(theirs);
        let error = Link::over(ours.try_clone().unwrap(), ours, false).err().expect("no link");
        assert_ne!(error.kind(), io::ErrorKind::TimedOut);
    }

    fn alive(pid: &str) -> bool {
        std::process::Command::new("kill")
            .args(["-0", pid])
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[test]
    fn a_spawned_child_is_killed_and_waited_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("pid");
        let script = format!(
            "echo $$ > {}; read line; echo '{{\"type\":\"hello\",\"protocol\":{PROTOCOL},\"version\":\"x\"}}'; exec sleep 60",
            pid_file.display()
        );
        let link = Link::spawn(std::process::Command::new("sh").args(["-c", &script])).unwrap();
        let pid = std::fs::read_to_string(&pid_file).unwrap().trim().to_string();
        assert!(alive(&pid));
        drop(link);
        assert!(!alive(&pid), "the child is gone once the link is");
    }

    #[test]
    fn a_spawned_child_that_never_answers_times_out_and_is_killed() {
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("pid");
        let script = format!("echo $$ > {}; exec sleep 60", pid_file.display());
        let started = std::time::Instant::now();
        let error =
            Link::spawn_within(std::process::Command::new("sh").args(["-c", &script]), Duration::from_millis(500))
                .err()
                .expect("no link");
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(10));
        let pid = std::fs::read_to_string(&pid_file).unwrap().trim().to_string();
        assert!(!alive(&pid), "the child is not left running");
    }

    #[test]
    fn spawn_waits_for_ssh_longer_than_the_socket() {
        assert_eq!(SSH_HELLO_TIMEOUT, Duration::from_secs(60));
        assert!(SSH_HELLO_TIMEOUT > SOCKET_HELLO_TIMEOUT);
    }

    #[test]
    fn a_child_that_is_not_there_is_an_error_not_a_panic() {
        let error = Link::spawn(&mut std::process::Command::new("/nonexistent/ssh-for-aria")).err().expect("no link");
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn a_child_that_says_hub_down_is_hub_down_and_is_not_left_running() {
        let script = r#"read line; echo '{"type":"error","id":0,"code":"hub-down"}'; exec sleep 60"#;
        let error = Link::spawn(std::process::Command::new("sh").args(["-c", script])).err().expect("no link");
        assert_eq!(error.to_string(), "hub-down");
    }

    #[test]
    fn messages_go_out_as_the_page_writes_them() {
        let mut dock = fake_dock();
        dock.link.send(&PageMessage::SessionsLoad { id: 1 });
        dock.link.send(&PageMessage::SessionNew { id: 2, workdir: Some("/home/ana".into()) });
        assert_eq!(dock.read(), json!({"type": "sessionsLoad", "id": 1}));
        assert_eq!(dock.read(), json!({"type": "sessionNew", "id": 2, "workdir": "/home/ana"}));
    }

    #[test]
    fn attach_is_sent_as_a_client_line() {
        let mut dock = fake_dock();
        dock.link.attach("aa");
        dock.link.focus(false);
        assert_eq!(dock.read(), json!({"type": "attach", "session": "aa"}));
        assert_eq!(dock.read(), json!({"type": "focus", "focused": false}));
    }

    #[test]
    fn unknown_messages_are_skipped() {
        let mut dock = fake_dock();
        let events = dock.link.events();
        dock.say(json!({"type": "somethingNewer", "x": 1}));
        dock.say(json!({"type": "sessionsChanged"}));
        let got = events.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(got, DockMessage::SessionsChanged);
    }

    #[test]
    fn a_closed_dock_is_reported_once_as_gone() {
        let dock = fake_dock();
        let events = dock.link.events();
        drop(dock.to_client);
        drop(dock.from_client);
        let got = events.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(got, DockMessage::Error { id: 0, code: "gone".into() });
        assert!(events.recv_timeout(Duration::from_millis(200)).is_err(), "said once");
    }
}
