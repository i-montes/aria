//! The browser comes back to a loopback address: this waits for that one request, answers it with a
//! small page, and hands back the authorization code. Shared by every OAuth sign-in.

use super::OAuthError;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

/// The authorization code from the browser's request line target (`/callback?code=…&state=…`).
pub fn parse_callback(request_target: &str, expected_state: &str) -> Result<String, OAuthError> {
    let url = url::Url::parse(&format!("http://localhost{request_target}")).map_err(|_| OAuthError::NoCode)?;
    let param = |name: &str| url.query_pairs().find(|(k, _)| k == name).map(|(_, v)| v.into_owned());
    if let Some(error) = param("error") {
        return Err(OAuthError::Denied(error));
    }
    if param("state").as_deref() != Some(expected_state) {
        return Err(OAuthError::StateMismatch);
    }
    param("code").filter(|c| !c.is_empty()).ok_or(OAuthError::NoCode)
}

/// The address a person pastes when the browser could not reach the loopback listener (a terminal on another computer),
/// or only its query. `text` is kept apart from any `Debug` output on purpose: it holds the authorization code.
/// `tag` is the caller's own, handed back to `Pastes::refused` so it can answer whoever pasted.
pub struct Paste {
    pub text: String,
    pub tag: u64,
}

/// What `wait_for_code` also listens to: pasted redirects, and where a refused one is reported. When every sender of
/// `inbox` is gone the wait is cancelled (`OAuthError::Cancelled`): that is how a newer sign-in replaces an older one.
pub struct Pastes<'a> {
    pub inbox: &'a Receiver<Paste>,
    /// Called with the `tag` of a paste that is not this sign-in's redirect; the wait goes on (a typo cancels nothing).
    pub refused: &'a dyn Fn(u64),
}

/// The authorization code from what a person pasted: a full redirect address (`http://localhost:PORT/callback?code=…&state=…`,
/// http or https, any host and port, the path exactly `callback_path`) or only its query (`code=…&state=…`, with or
/// without the `?`). Whitespace anywhere is dropped (a terminal may have wrapped a long address). The `state` must be
/// ours: it is what stops a code somebody else planted. No error carries the pasted text.
pub fn parse_pasted(text: &str, callback_path: &str, expected_state: &str) -> Result<String, OAuthError> {
    let text: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let query = if has_scheme(&text) {
        let url = url::Url::parse(&text).map_err(|_| OAuthError::NoCode)?;
        if !matches!(url.scheme(), "http" | "https") || url.path() != callback_path {
            return Err(OAuthError::NoCode);
        }
        url.query().unwrap_or_default().to_string()
    } else {
        text.strip_prefix('?').unwrap_or(&text).to_string()
    };
    parse_callback(&format!("{callback_path}?{query}"), expected_state).map_err(|error| match error {
        // the provider's own words came out of the pasted text
        OAuthError::Denied(_) => OAuthError::Denied("denied".into()),
        other => other,
    })
}

/// `scheme://…` (a letter, then letters, digits, `+`, `-` or `.`).
fn has_scheme(text: &str) -> bool {
    let Some((scheme, _)) = text.split_once("://") else { return false };
    scheme.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && scheme.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

/// Waits (up to `timeout`) for the browser's request to `callback_path` on a non-blocking listener
/// and returns its code. Other requests (a favicon, say) get a 404 and are ignored; a callback that is
/// denied, has the wrong `state` or has no code is answered with a 400 and reported. With `paste`, an address
/// pasted meanwhile that parses (`parse_pasted`) ends the wait with its code too.
pub fn wait_for_code(
    listener: &TcpListener,
    callback_path: &str,
    expected_state: &str,
    timeout: Duration,
    paste: Option<&Pastes>,
) -> Result<String, OAuthError> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(pastes) = paste {
            loop {
                match pastes.inbox.try_recv() {
                    Ok(pasted) => match parse_pasted(&pasted.text, callback_path, expected_state) {
                        Ok(code) => return Ok(code),
                        Err(_) => (pastes.refused)(pasted.tag),
                    },
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return Err(OAuthError::Cancelled),
                }
            }
        }
        match listener.accept() {
            Ok((mut stream, _)) => {
                if let Some(result) = handle(&mut stream, callback_path, expected_state) {
                    return result;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(OAuthError::Timeout);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(OAuthError::Io(e.to_string())),
        }
    }
}

/// One browser request. `None` means it was not the callback: keep waiting.
fn handle(stream: &mut TcpStream, callback_path: &str, expected_state: &str) -> Option<Result<String, OAuthError>> {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut buffer = [0u8; 4096];
    let read = stream.read(&mut buffer).unwrap_or(0);
    let request = String::from_utf8_lossy(&buffer[..read]);
    let target = request.lines().next()?.split_whitespace().nth(1)?.to_string();
    if !target.starts_with(callback_path) {
        respond(stream, "404 Not Found", "Not found.");
        return None;
    }
    match parse_callback(&target, expected_state) {
        Ok(code) => {
            respond(stream, "200 OK", "Signed in. You can close this tab and go back to Aria.");
            Some(Ok(code))
        }
        Err(error) => {
            respond(stream, "400 Bad Request", "Sign-in failed. You can close this tab and try again in Aria.");
            Some(Err(error))
        }
    }
}

fn respond(stream: &mut TcpStream, status: &str, message: &str) {
    let html = format!("<!doctype html><meta charset=utf-8><title>Aria</title><body style=\"font:16px system-ui;padding:3rem\">{message}</body>");
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html}",
        html.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    const PATH: &str = "/callback";

    #[test]
    fn a_pasted_redirect_gives_its_code() {
        let pasted = "  http://localhost:53692/callback?code=abc&state=st\n";
        assert_eq!(parse_pasted(pasted, PATH, "st"), Ok("abc".to_string()));
        // any host, port and scheme: the address bar of a browser on another computer
        assert_eq!(parse_pasted("https://hub.example:8443/callback?code=abc&state=st", PATH, "st"), Ok("abc".into()));
        // a long address a terminal wrapped onto several lines
        assert_eq!(parse_pasted("http://localhost:1/callback?code=a\nbc&sta\r\nte=st", PATH, "st"), Ok("abc".into()));
    }

    #[test]
    fn a_pasted_query_alone_gives_its_code() {
        assert_eq!(parse_pasted("code=abc&state=st", PATH, "st"), Ok("abc".to_string()));
        assert_eq!(parse_pasted("?code=abc&state=st\n", PATH, "st"), Ok("abc".to_string()));
    }

    #[test]
    fn a_pasted_address_with_the_wrong_state_is_refused() {
        assert_eq!(
            parse_pasted("http://localhost:1/callback?code=abc&state=planted", PATH, "st"),
            Err(OAuthError::StateMismatch)
        );
        assert_eq!(parse_pasted("code=abc", PATH, "st"), Err(OAuthError::StateMismatch));
        assert_eq!(parse_pasted("code=abc&state=stX", PATH, "st"), Err(OAuthError::StateMismatch));
    }

    #[test]
    fn a_pasted_address_for_another_path_is_refused() {
        assert!(parse_pasted("http://localhost:1/other?code=abc&state=st", PATH, "st").is_err());
        assert!(parse_pasted("http://localhost:1/callback/x?code=abc&state=st", PATH, "st").is_err());
        assert!(parse_pasted("http://localhost:1/?code=abc&state=st", PATH, "st").is_err());
    }

    #[test]
    fn anything_that_is_not_a_redirect_is_refused_without_echoing_it() {
        for text in [
            "",
            "   ",
            "hello world",
            "ftp://localhost:1/callback?code=abc&state=st",
            "file:///callback?code=abc&state=st",
            "javascript:alert(1)",
            "http://localhost:1/callback?state=st",
            "http://localhost:1/callback?code=&state=st",
            "localhost:1/callback?code=abc&state=st",
        ] {
            let error = parse_pasted(text, PATH, "st").expect_err(text);
            assert!(!error.to_string().contains("abc") && !error.to_string().contains("hello"), "{error}");
        }
        let error = parse_pasted("code=SECRETCODE&state=wrong", PATH, "st").unwrap_err();
        assert!(!format!("{error:?} {error}").contains("SECRETCODE"));
        let error = parse_pasted("error=oops-pasted&state=st", PATH, "st").unwrap_err();
        assert!(!format!("{error:?} {error}").contains("oops-pasted"));
    }

    fn listener() -> TcpListener {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        listener
    }

    #[test]
    fn a_paste_while_waiting_ends_the_wait_before_the_timeout() {
        let (sender, inbox) = mpsc::channel();
        sender.send(Paste { text: "code=abc&state=st".into(), tag: 7 }).unwrap();
        let refused = |_: u64| panic!("a good paste is not refused");
        let start = Instant::now();
        let code = wait_for_code(
            &listener(),
            PATH,
            "st",
            Duration::from_secs(30),
            Some(&Pastes { inbox: &inbox, refused: &refused }),
        );
        assert_eq!(code, Ok("abc".to_string()));
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn a_paste_that_does_not_parse_is_reported_and_the_wait_goes_on() {
        let (sender, inbox) = mpsc::channel();
        sender.send(Paste { text: "nonsense".into(), tag: 8 }).unwrap();
        let refused_tags = std::sync::Mutex::new(Vec::new());
        let refused = |tag: u64| refused_tags.lock().unwrap().push(tag);
        let later = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            sender.send(Paste { text: "code=abc&state=st".into(), tag: 9 }).unwrap();
            sender
        });
        let code = wait_for_code(
            &listener(),
            PATH,
            "st",
            Duration::from_secs(30),
            Some(&Pastes { inbox: &inbox, refused: &refused }),
        );
        assert_eq!(code, Ok("abc".to_string()));
        assert_eq!(refused_tags.into_inner().unwrap(), vec![8]);
        drop(later.join().unwrap());
    }

    #[test]
    fn a_wait_whose_paste_channel_is_closed_is_cancelled() {
        let (sender, inbox) = mpsc::channel::<Paste>();
        drop(sender);
        let refused = |_: u64| {};
        let result = wait_for_code(
            &listener(),
            PATH,
            "st",
            Duration::from_secs(30),
            Some(&Pastes { inbox: &inbox, refused: &refused }),
        );
        assert_eq!(result, Err(OAuthError::Cancelled));
    }

    #[test]
    fn without_a_paste_channel_the_wait_times_out_as_before() {
        assert_eq!(wait_for_code(&listener(), PATH, "st", Duration::from_millis(100), None), Err(OAuthError::Timeout));
    }
}
