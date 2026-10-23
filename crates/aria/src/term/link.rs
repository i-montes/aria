//! The client's connection to the dock (`clients.sock`): the page's messages out, the dock's in, one JSON object per
//! line. A message the client does not know (a newer dock) is skipped; a dock that goes away is said once, as
//! `error { id: 0, code: "gone" }`.

use aria_core::bridge::{DockMessage, PageMessage};
use serde_json::json;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;

pub struct Link {
    out: Mutex<UnixStream>,
    next: AtomicU64,
    events: Mutex<Option<Receiver<DockMessage>>>,
}

impl Link {
    pub fn connect(path: &Path) -> io::Result<Link> {
        let stream = UnixStream::connect(path)?;
        let reader = stream.try_clone()?;
        let (sender, events) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(reader).lines().map_while(Result::ok) {
                if let Ok(message) = serde_json::from_str::<DockMessage>(&line) {
                    if sender.send(message).is_err() {
                        return;
                    }
                }
            }
            let _ = sender.send(DockMessage::Error { id: 0, code: "gone".into() });
        });
        Ok(Link { out: Mutex::new(stream), next: AtomicU64::new(1), events: Mutex::new(Some(events)) })
    }

    fn line(&self, value: serde_json::Value) {
        let mut out = self.out.lock().unwrap();
        let _ = writeln!(out, "{value}");
    }

    pub fn send(&self, message: &PageMessage) {
        if let Ok(value) = serde_json::to_value(message) {
            self.line(value);
        }
    }

    /// Sends the message `make` builds with a fresh id, and returns that id.
    pub fn request(&self, make: impl FnOnce(u64) -> PageMessage) -> u64 {
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        self.send(&make(id));
        id
    }

    pub fn attach(&self, session: &str) {
        self.line(json!({"type": "attach", "session": session}));
    }

    pub fn focus(&self, focused: bool) {
        self.line(json!({"type": "focus", "focused": focused}));
    }

    /// What the dock says; taken once (later calls get a receiver that is already closed).
    pub fn events(&self) -> Receiver<DockMessage> {
        self.events.lock().unwrap().take().unwrap_or_else(|| mpsc::channel().1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::bridge::{DockMessage, PageMessage};
    use serde_json::{json, Value};
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::time::Duration;

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
        let link = Link::connect(&path).unwrap();
        let (stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        FakeDock { _dir: dir, link, from_client: BufReader::new(stream.try_clone().unwrap()), to_client: stream }
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

    #[test]
    fn requests_get_increasing_ids() {
        let mut dock = fake_dock();
        let first = dock.link.request(|id| PageMessage::SessionsLoad { id });
        let second = dock.link.request(|id| PageMessage::SessionNew { id, workdir: Some("/home/ana".into()) });
        assert_eq!((first, second), (1, 2));
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
