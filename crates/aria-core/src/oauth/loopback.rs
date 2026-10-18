//! The browser comes back to a loopback address: this waits for that one request, answers it with a
//! small page, and hands back the authorization code. Shared by every OAuth sign-in.

use super::OAuthError;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
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

/// Waits (up to `timeout`) for the browser's request to `callback_path` on a non-blocking listener
/// and returns its code. Other requests (a favicon, say) get a 404 and are ignored; a callback that is
/// denied, has the wrong `state` or has no code is answered with a 400 and reported.
pub fn wait_for_code(listener: &TcpListener, callback_path: &str, expected_state: &str, timeout: Duration) -> Result<String, OAuthError> {
    let deadline = Instant::now() + timeout;
    loop {
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
