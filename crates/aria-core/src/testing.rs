//! Test support shared by this crate's tests and the dock's (`features = ["testing"]`): a tiny server
//! that behaves like Notion's MCP, for exercising the OAuth flow without the network.

use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct Req {
    pub method: String,
    pub path: String,
    pub body: String,
}

pub struct Fake {
    pub base: String,
    seen: Arc<Mutex<Vec<Req>>>,
}

impl Fake {
    pub fn requests(&self, method: &str, path: &str) -> Vec<Req> {
        self.seen.lock().unwrap().iter().filter(|r| r.method == method && r.path == path).cloned().collect()
    }
}

/// A tiny server that behaves like Notion's MCP: a 401 that points at its metadata, an OAuth server
/// with dynamic client registration, and a token endpoint. `protected: false` makes /mcp answer 200.
pub fn fake(protected: bool, register_status: u16) -> Fake {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (log, root) = (seen.clone(), base.clone());
    std::thread::spawn(move || {
        for stream in listener.incoming().map_while(Result::ok) {
            let (log, root) = (log.clone(), root.clone());
            std::thread::spawn(move || serve_one(stream, &log, &root, protected, register_status));
        }
    });
    Fake { base, seen }
}

fn serve_one(
    mut stream: std::net::TcpStream,
    log: &Mutex<Vec<Req>>,
    base: &str,
    protected: bool,
    register_status: u16,
) {
    let mut data = Vec::new();
    let mut buffer = [0u8; 4096];
    let (head_end, length) = loop {
        let n = stream.read(&mut buffer).unwrap_or(0);
        if n == 0 {
            return;
        }
        data.extend_from_slice(&buffer[..n]);
        if let Some(at) = data.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&data[..at]).to_lowercase();
            let length = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0usize);
            break (at + 4, length);
        }
    };
    while data.len() < head_end + length {
        let n = stream.read(&mut buffer).unwrap_or(0);
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..n]);
    }
    let text = String::from_utf8_lossy(&data).into_owned();
    let first = text.lines().next().unwrap_or("");
    let (method, path) = (
        first.split_whitespace().next().unwrap_or("").to_string(),
        first.split_whitespace().nth(1).unwrap_or("").to_string(),
    );
    let body = text[head_end.min(text.len())..].to_string();
    log.lock().unwrap().push(Req { method: method.clone(), path: path.clone(), body: body.clone() });
    let json_response = |status: u16, value: serde_json::Value| (status, vec![], value.to_string());
    let (status, headers, payload) = match (method.as_str(), path.as_str()) {
        ("POST", "/mcp") if protected => (
            401,
            vec![format!(
                "WWW-Authenticate: Bearer realm=\"OAuth\", resource_metadata=\"{base}/.well-known/oauth-protected-resource/mcp\", scope=\"default\""
            )],
            String::new(),
        ),
        ("POST", "/mcp") => json_response(200, json!({"jsonrpc": "2.0", "id": 1, "result": {}})),
        ("GET", "/.well-known/oauth-protected-resource/mcp") => {
            json_response(200, json!({"resource": format!("{base}/mcp"), "authorization_servers": [base], "scopes_supported": ["default"]}))
        }
        ("GET", "/.well-known/oauth-authorization-server") => json_response(
            200,
            json!({
                "issuer": base, "authorization_endpoint": format!("{base}/authorize"), "token_endpoint": format!("{base}/token"),
                "registration_endpoint": format!("{base}/register"), "code_challenge_methods_supported": ["S256"],
            }),
        ),
        ("POST", "/register") if register_status == 201 => {
            let request: serde_json::Value = serde_json::from_str(&body).unwrap_or(json!({}));
            json_response(
                201,
                json!({
                    "client_id": "cid-123", "redirect_uris": request["redirect_uris"], "client_name": request["client_name"],
                    "grant_types": request["grant_types"], "response_types": request["response_types"],
                    "token_endpoint_auth_method": "none", "client_id_issued_at": 1,
                }),
            )
        }
        ("POST", "/register") => json_response(register_status, json!({"error": "invalid_redirect_uri"})),
        ("POST", "/token") => json_response(200, json!({"access_token": "at", "refresh_token": "rt", "token_type": "bearer", "expires_in": 3600, "scope": "default"})),
        _ => json_response(404, json!({})),
    };
    let mut response = format!(
        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        payload.len()
    );
    for h in headers {
        response.push_str(&h);
        response.push_str("\r\n");
    }
    response.push_str("\r\n");
    response.push_str(&payload);
    let _ = stream.write_all(response.as_bytes());
}

/// A home in `dir` whose credentials go through a vault backed by memory (never the real keyring): its live copies in
/// `dir/live`, its commands naming `/bin/aria`.
pub fn vault_paths(dir: &std::path::Path) -> (crate::paths::AriaPaths, Arc<crate::keyring::MemoryBackend>) {
    let keyring = Arc::new(crate::keyring::MemoryBackend::default());
    let live = dir.join("live");
    std::fs::create_dir_all(&live).unwrap();
    let vault = crate::vault::Vault { keyring: keyring.clone(), live, exe: std::path::PathBuf::from("/bin/aria") };
    let paths = crate::paths::AriaPaths { root: dir.to_path_buf(), vault: Some(vault) };
    (paths, keyring)
}
