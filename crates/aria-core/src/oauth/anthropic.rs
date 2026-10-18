//! Anthropic (Claude Pro/Max) sign-in: OAuth authorization code + PKCE with a localhost callback.
//! The values are the ones Pi's own `/login` uses, which is what the Claude Max route runs on today.

use super::OAuthError;
use crate::auth::Credential;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::RngCore;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const SCOPES: &str =
    "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
const PREFERRED_PORT: u16 = 53692;
const CALLBACK_PATH: &str = "/callback";
/// Tokens are treated as expired five minutes early.
const EXPIRY_MARGIN_MS: i64 = 300_000;

pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
}

pub fn challenge_for(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

pub fn pkce() -> Pkce {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    let verifier = URL_SAFE_NO_PAD.encode(bytes);
    Pkce { challenge: challenge_for(&verifier), verifier }
}

pub fn authorize_url(pkce: &Pkce, redirect_uri: &str) -> String {
    let params = [
        ("code", "true"),
        ("client_id", CLIENT_ID),
        ("response_type", "code"),
        ("redirect_uri", redirect_uri),
        ("scope", SCOPES),
        ("code_challenge", pkce.challenge.as_str()),
        ("code_challenge_method", "S256"),
        ("state", pkce.verifier.as_str()),
    ];
    url::Url::parse_with_params(AUTHORIZE_URL, params).expect("static URL").into()
}

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

pub fn token_request_body(code: &str, state: &str, pkce: &Pkce, redirect_uri: &str) -> Value {
    json!({
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "state": state,
        "redirect_uri": redirect_uri,
        "code_verifier": pkce.verifier,
    })
}

pub fn credential_from_token(token: &Value, now_ms: i64) -> Result<Credential, OAuthError> {
    let text = |key: &str| token.get(key).and_then(Value::as_str).map(str::to_string);
    let (Some(access), Some(refresh), Some(expires_in)) =
        (text("access_token"), text("refresh_token"), token.get("expires_in").and_then(Value::as_i64))
    else {
        return Err(OAuthError::BadToken("missing access_token, refresh_token or expires_in".into()));
    };
    Ok(Credential::OAuth { access, refresh, expires: now_ms + expires_in * 1000 - EXPIRY_MARGIN_MS })
}

/// A sign-in in progress: the listener is already bound, `url` is what the browser must open.
pub struct SignIn {
    pub url: String,
    listener: TcpListener,
    pkce: Pkce,
    redirect_uri: String,
}

/// Binds the callback listener (port 53692, or any free port when it is busy) and builds the URL.
pub fn start() -> Result<SignIn, OAuthError> {
    let listener = TcpListener::bind(("127.0.0.1", PREFERRED_PORT))
        .or_else(|_| TcpListener::bind(("127.0.0.1", 0)))
        .map_err(|e| OAuthError::Io(e.to_string()))?;
    listener.set_nonblocking(true).map_err(|e| OAuthError::Io(e.to_string()))?;
    let port = listener.local_addr().map_err(|e| OAuthError::Io(e.to_string()))?.port();
    let redirect_uri = format!("http://localhost:{port}{CALLBACK_PATH}");
    let pkce = pkce();
    Ok(SignIn { url: authorize_url(&pkce, &redirect_uri), listener, pkce, redirect_uri })
}

impl SignIn {
    /// Waits for the browser to come back (up to `timeout`), then exchanges the code for tokens.
    pub fn wait(self, timeout: Duration) -> Result<Credential, OAuthError> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if let Some(result) = self.handle(stream) {
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

    /// One browser request. `None` means it was not the callback (a favicon, say): keep waiting.
    fn handle(&self, mut stream: TcpStream) -> Option<Result<Credential, OAuthError>> {
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let mut buffer = [0u8; 4096];
        let read = stream.read(&mut buffer).unwrap_or(0);
        let request = String::from_utf8_lossy(&buffer[..read]);
        let target = request.lines().next()?.split_whitespace().nth(1)?.to_string();
        if !target.starts_with(CALLBACK_PATH) {
            respond(&mut stream, "404 Not Found", "Not found.");
            return None;
        }
        match parse_callback(&target, &self.pkce.verifier) {
            Ok(code) => {
                respond(&mut stream, "200 OK", "Signed in. You can close this tab and go back to Aria.");
                Some(self.exchange(&code))
            }
            Err(error) => {
                respond(&mut stream, "400 Bad Request", "Sign-in failed. You can close this tab and try again in Aria.");
                Some(Err(error))
            }
        }
    }

    fn exchange(&self, code: &str) -> Result<Credential, OAuthError> {
        let body = token_request_body(code, &self.pkce.verifier, &self.pkce, &self.redirect_uri);
        let response = ureq::post(TOKEN_URL)
            .timeout(Duration::from_secs(30))
            .set("Accept", "application/json")
            .send_json(body)
            .map_err(|e| match e {
                ureq::Error::Status(status, _) => OAuthError::BadToken(format!("HTTP {status}")),
                ureq::Error::Transport(t) => OAuthError::Network(t.to_string()),
            })?;
        let token: Value = response.into_json().map_err(|e| OAuthError::BadToken(e.to_string()))?;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or_default();
        credential_from_token(&token, now_ms)
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
    use serde_json::json;
    use std::net::TcpListener;
    use std::sync::Mutex;
    use std::time::Duration;

    // `start()` binds a fixed port first; keep the two tests that use it from racing each other.
    static PORT: Mutex<()> = Mutex::new(());

    fn query(url: &str) -> std::collections::HashMap<String, String> {
        url::Url::parse(url).unwrap().query_pairs().map(|(k, v)| (k.into_owned(), v.into_owned())).collect()
    }

    #[test]
    fn pkce_challenge_is_s256() {
        // RFC 7636 appendix B
        assert_eq!(
            challenge_for("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
        let pkce = pkce();
        assert_eq!(pkce.challenge, challenge_for(&pkce.verifier));
        assert!(pkce.verifier.len() >= 43);
    }

    #[test]
    fn authorize_url_has_every_param() {
        let pkce = Pkce { verifier: "v".into(), challenge: "c".into() };
        let url = authorize_url(&pkce, "http://localhost:53692/callback");
        assert!(url.starts_with("https://claude.ai/oauth/authorize?"));
        let q = query(&url);
        assert_eq!(q.len(), 8);
        assert_eq!(q["code"], "true");
        assert_eq!(q["client_id"], "9d1c250a-e61b-44d9-88ed-5944d1962f5e");
        assert_eq!(q["response_type"], "code");
        assert_eq!(q["redirect_uri"], "http://localhost:53692/callback");
        assert_eq!(
            q["scope"],
            "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
        );
        assert_eq!(q["code_challenge"], "c");
        assert_eq!(q["code_challenge_method"], "S256");
        assert_eq!(q["state"], "v");
    }

    #[test]
    fn parse_callback_ok() {
        assert_eq!(parse_callback("/callback?code=abc&state=v", "v"), Ok("abc".to_string()));
    }

    #[test]
    fn parse_callback_state_mismatch() {
        assert_eq!(parse_callback("/callback?code=abc&state=other", "v"), Err(OAuthError::StateMismatch));
        assert_eq!(parse_callback("/callback?code=abc", "v"), Err(OAuthError::StateMismatch));
    }

    #[test]
    fn parse_callback_denied() {
        assert_eq!(
            parse_callback("/callback?error=access_denied&state=v", "v"),
            Err(OAuthError::Denied("access_denied".into()))
        );
    }

    #[test]
    fn parse_callback_without_code() {
        assert_eq!(parse_callback("/callback?state=v", "v"), Err(OAuthError::NoCode));
    }

    #[test]
    fn token_body_shape() {
        let pkce = Pkce { verifier: "v".into(), challenge: "c".into() };
        assert_eq!(
            token_request_body("code1", "v", &pkce, "http://localhost:53692/callback"),
            json!({
                "grant_type": "authorization_code",
                "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                "code": "code1",
                "state": "v",
                "redirect_uri": "http://localhost:53692/callback",
                "code_verifier": "v"
            })
        );
    }

    #[test]
    fn credential_expiry() {
        let cred = credential_from_token(&json!({"access_token": "a", "refresh_token": "r", "expires_in": 3600}), 1_000_000).unwrap();
        assert_eq!(
            cred,
            crate::auth::Credential::OAuth { access: "a".into(), refresh: "r".into(), expires: 1_000_000 + 3_600_000 - 300_000 }
        );
        assert!(matches!(credential_from_token(&json!({"access_token": "a"}), 0), Err(OAuthError::BadToken(_))));
    }

    #[test]
    fn busy_port_falls_back() {
        let _guard = PORT.lock().unwrap();
        let _holder = TcpListener::bind(("127.0.0.1", 53692));
        let sign_in = start().unwrap();
        let redirect = &query(&sign_in.url)["redirect_uri"];
        assert!(redirect.starts_with("http://localhost:"));
        assert!(!redirect.contains(":53692/"), "expected another port, got {redirect}");
    }

    fn browser_request(redirect_uri: &str, target: &str) -> String {
        use std::io::{Read, Write};
        let port: u16 = redirect_uri.rsplit(':').next().unwrap().split('/').next().unwrap().parse().unwrap();
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(stream, "GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    #[test]
    fn callback_with_wrong_state_fails_and_a_favicon_is_ignored() {
        let _guard = PORT.lock().unwrap();
        let sign_in = start().unwrap();
        let redirect = sign_in.redirect_uri.clone();
        let browser = std::thread::spawn(move || {
            let favicon = browser_request(&redirect, "/favicon.ico");
            let callback = browser_request(&redirect, "/callback?code=abc&state=not-the-verifier");
            (favicon, callback)
        });
        assert_eq!(sign_in.wait(Duration::from_secs(5)), Err(OAuthError::StateMismatch));
        let (favicon, callback) = browser.join().unwrap();
        assert!(favicon.starts_with("HTTP/1.1 404"));
        assert!(callback.starts_with("HTTP/1.1 400"));
    }

    #[test]
    fn wait_times_out() {
        let _guard = PORT.lock().unwrap();
        let result = start().unwrap().wait(Duration::from_millis(200));
        assert_eq!(result, Err(OAuthError::Timeout));
    }
}
