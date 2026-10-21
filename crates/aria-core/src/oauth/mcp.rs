//! OAuth sign-in for a remote MCP server (Notion), the way the MCP authorization spec and Pi do it:
//! the server's 401 points at its resource metadata, which names an authorization server; the client
//! registers itself there (a public client, dynamic registration), sends the user to the browser with
//! PKCE and the `resource` it wants, and trades the code that comes back to a loopback address for
//! tokens. The result is a record in the shape Pi stores in `mcp-auth.json`.

use super::loopback::wait_for_code;
use super::pkce::{pkce, Pkce};
use super::OAuthError;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::RngCore;
use serde_json::{json, Value};
use std::net::TcpListener;
use std::time::Duration;

const CALLBACK_PATH: &str = "/callback";
const CLIENT_NAME: &str = "Aria";
const HTTP_TIMEOUT: Duration = Duration::from_secs(20);

fn setup(message: impl Into<String>) -> OAuthError {
    OAuthError::Setup(message.into())
}

fn transport(error: ureq::Error) -> OAuthError {
    match error {
        ureq::Error::Status(status, _) => setup(format!("HTTP {status}")),
        ureq::Error::Transport(t) => OAuthError::Network(t.to_string()),
    }
}

fn get_json(url: &str) -> Result<Value, OAuthError> {
    ureq::get(url)
        .timeout(HTTP_TIMEOUT)
        .set("Accept", "application/json")
        .call()
        .map_err(transport)?
        .into_json()
        .map_err(|e| setup(e.to_string()))
}

/// The value of `name="..."` in a `WWW-Authenticate` header.
fn quoted_param(header: &str, name: &str) -> Option<String> {
    let start = header.find(&format!("{name}=\""))? + name.len() + 2;
    let end = header[start..].find('"')?;
    Some(header[start..start + end].to_string())
}

/// Where a server's resource metadata lives when its 401 does not say (RFC 9728).
fn well_known_resource_metadata(server_url: &str) -> Option<String> {
    let url = url::Url::parse(server_url).ok()?;
    let path = if url.path() == "/" { "" } else { url.path() };
    Some(format!("{}/.well-known/oauth-protected-resource{path}", url.origin().ascii_serialization()))
}

pub struct Discovery {
    /// The `resource` the tokens are for (sent to the authorization and token endpoints).
    pub resource: String,
    pub resource_metadata: Value,
    pub authorization_server_url: String,
    pub authorization_server: Value,
    pub scope: Option<String>,
}

/// Asks the server to start a session without credentials, follows its 401 to the resource metadata,
/// and from there to the authorization server's metadata.
pub fn discover(server_url: &str) -> Result<Discovery, OAuthError> {
    let probe = ureq::post(server_url)
        .timeout(HTTP_TIMEOUT)
        .set("Accept", "application/json, text/event-stream")
        .send_json(json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": { "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": { "name": "aria", "version": env!("CARGO_PKG_VERSION") } },
        }));
    let challenge = match probe {
        Err(ureq::Error::Status(401, response)) => response.header("www-authenticate").unwrap_or_default().to_string(),
        Ok(_) => return Err(setup("this server does not ask for a sign-in")),
        Err(other) => return Err(transport(other)),
    };
    let metadata_url = quoted_param(&challenge, "resource_metadata")
        .or_else(|| well_known_resource_metadata(server_url))
        .ok_or_else(|| setup("the server did not say where its sign-in is described"))?;
    let resource_metadata = get_json(&metadata_url)?;
    let authorization_server_url = resource_metadata["authorization_servers"][0]
        .as_str()
        .ok_or_else(|| setup("the server names no authorization server"))?
        .trim_end_matches('/')
        .to_string();
    let authorization_server = get_json(&format!("{authorization_server_url}/.well-known/oauth-authorization-server"))
        .or_else(|_| get_json(&format!("{authorization_server_url}/.well-known/openid-configuration")))?;
    let scope = quoted_param(&challenge, "scope").or_else(|| {
        let scopes: Vec<&str> =
            resource_metadata["scopes_supported"].as_array()?.iter().filter_map(Value::as_str).collect();
        (!scopes.is_empty()).then(|| scopes.join(" "))
    });
    Ok(Discovery {
        resource: resource_metadata["resource"].as_str().unwrap_or(server_url).to_string(),
        resource_metadata,
        authorization_server_url,
        authorization_server,
        scope,
    })
}

/// A sign-in in progress: the client is registered and the listener is bound; `url` is where the
/// browser must go.
pub struct McpSignIn {
    pub url: String,
    server_url: String,
    discovery: Discovery,
    client_information: Value,
    listener: TcpListener,
    pkce: Pkce,
    state: String,
    redirect_uri: String,
}

fn endpoint(metadata: &Value, key: &str) -> Result<String, OAuthError> {
    metadata[key].as_str().map(str::to_string).ok_or_else(|| setup(format!("the authorization server has no {key}")))
}

/// Discovers the server's sign-in, registers a public client for a loopback callback, and builds the
/// authorization URL.
pub fn start(server_url: &str) -> Result<McpSignIn, OAuthError> {
    let discovery = discover(server_url)?;
    let metadata = &discovery.authorization_server;
    let (authorize, register) =
        (endpoint(metadata, "authorization_endpoint")?, endpoint(metadata, "registration_endpoint")?);
    endpoint(metadata, "token_endpoint")?;
    if let Some(methods) = metadata["code_challenge_methods_supported"].as_array() {
        if !methods.iter().any(|m| m == "S256") {
            return Err(setup("the authorization server does not support PKCE (S256)"));
        }
    }
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| OAuthError::Io(e.to_string()))?;
    listener.set_nonblocking(true).map_err(|e| OAuthError::Io(e.to_string()))?;
    let port = listener.local_addr().map_err(|e| OAuthError::Io(e.to_string()))?.port();
    let redirect_uri = format!("http://127.0.0.1:{port}{CALLBACK_PATH}");

    let client_information: Value = ureq::post(&register)
        .timeout(HTTP_TIMEOUT)
        .send_json(json!({
            "client_name": CLIENT_NAME,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }))
        .map_err(transport)?
        .into_json()
        .map_err(|e| setup(e.to_string()))?;
    let client_id =
        client_information["client_id"].as_str().ok_or_else(|| setup("the server registered no client"))?.to_string();

    let pkce = pkce();
    let mut random = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut random);
    let state = URL_SAFE_NO_PAD.encode(random);
    let mut params = vec![
        ("response_type", "code".to_string()),
        ("client_id", client_id),
        ("redirect_uri", redirect_uri.clone()),
        ("code_challenge", pkce.challenge.clone()),
        ("code_challenge_method", "S256".to_string()),
        ("state", state.clone()),
        ("resource", discovery.resource.clone()),
    ];
    if let Some(scope) = &discovery.scope {
        params.push(("scope", scope.clone()));
    }
    let url = url::Url::parse_with_params(&authorize, params).map_err(|e| setup(e.to_string()))?.into();
    Ok(McpSignIn {
        url,
        server_url: server_url.to_string(),
        discovery,
        client_information,
        listener,
        pkce,
        state,
        redirect_uri,
    })
}

impl McpSignIn {
    /// Waits for the browser (up to `timeout`), trades the code for tokens, and returns the record to
    /// store in `mcp-auth.json`.
    pub fn wait(self, timeout: Duration) -> Result<Value, OAuthError> {
        let code = wait_for_code(&self.listener, CALLBACK_PATH, &self.state, timeout)?;
        let token_endpoint = endpoint(&self.discovery.authorization_server, "token_endpoint")?;
        let client_id = self.client_information["client_id"].as_str().unwrap_or_default();
        let tokens: Value = ureq::post(&token_endpoint)
            .timeout(HTTP_TIMEOUT)
            .set("Accept", "application/json")
            .send_form(&[
                ("grant_type", "authorization_code"),
                ("code", &code),
                ("redirect_uri", &self.redirect_uri),
                ("client_id", client_id),
                ("code_verifier", &self.pkce.verifier),
                ("resource", &self.discovery.resource),
            ])
            .map_err(|e| match e {
                ureq::Error::Status(status, _) => OAuthError::BadToken(format!("HTTP {status}")),
                ureq::Error::Transport(t) => OAuthError::Network(t.to_string()),
            })?
            .into_json()
            .map_err(|e| OAuthError::BadToken(e.to_string()))?;
        if tokens["access_token"].as_str().is_none_or(str::is_empty) {
            return Err(OAuthError::BadToken("the response has no access_token".into()));
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or_default();
        let expires_in = tokens["expires_in"].as_i64().unwrap_or(3600);
        Ok(json!({
            "serverUrl": self.server_url,
            "discovery": {
                "authorizationServerUrl": self.discovery.authorization_server_url,
                "authorizationServerMetadata": self.discovery.authorization_server,
                "resourceMetadata": self.discovery.resource_metadata,
            },
            "clientInformation": self.client_information,
            "oauthState": self.state,
            "codeVerifier": self.pkce.verifier,
            "tokens": tokens,
            "tokensExpireAt": now_ms + expires_in * 1000,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::fake;
    use std::io::{Read, Write};
    use std::time::Duration;

    fn query(url: &str) -> std::collections::HashMap<String, String> {
        url::Url::parse(url).unwrap().query_pairs().map(|(k, v)| (k.into_owned(), v.into_owned())).collect()
    }

    fn browser(redirect_uri: &str, query_string: &str) {
        let port: u16 = redirect_uri.rsplit(':').next().unwrap().split('/').next().unwrap().parse().unwrap();
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(stream, "GET /callback?{query_string} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut sink = String::new();
        let _ = stream.read_to_string(&mut sink);
    }

    #[test]
    fn the_challenge_leads_to_the_resource_and_the_authorization_server() {
        let server = fake(true, 201);
        let found = discover(&format!("{}/mcp", server.base)).unwrap();
        assert_eq!(found.authorization_server_url, server.base);
        assert_eq!(found.resource, format!("{}/mcp", server.base));
        assert_eq!(found.scope.as_deref(), Some("default"));
        assert_eq!(found.authorization_server["token_endpoint"], format!("{}/token", server.base));
    }

    #[test]
    fn a_server_that_asks_for_no_sign_in_is_reported() {
        let server = fake(false, 201);
        assert!(matches!(discover(&format!("{}/mcp", server.base)), Err(OAuthError::Setup(_))));
    }

    #[test]
    fn it_registers_a_public_loopback_client_named_aria() {
        let server = fake(true, 201);
        let _sign_in = start(&format!("{}/mcp", server.base)).unwrap();
        let sent: serde_json::Value = serde_json::from_str(&server.requests("POST", "/register")[0].body).unwrap();
        assert_eq!(sent["client_name"], "Aria");
        assert_eq!(sent["token_endpoint_auth_method"], "none");
        assert!(sent["grant_types"].as_array().unwrap().iter().any(|g| g == "refresh_token"));
        let uri = sent["redirect_uris"][0].as_str().unwrap();
        assert!(uri.starts_with("http://127.0.0.1:") && uri.ends_with("/callback"), "{uri}");
    }

    #[test]
    fn a_refused_registration_is_reported() {
        let server = fake(true, 400);
        assert!(matches!(start(&format!("{}/mcp", server.base)), Err(OAuthError::Setup(_))));
    }

    #[test]
    fn the_authorize_url_carries_pkce_state_resource_and_scope() {
        let server = fake(true, 201);
        let sign_in = start(&format!("{}/mcp", server.base)).unwrap();
        assert!(sign_in.url.starts_with(&format!("{}/authorize?", server.base)));
        let q = query(&sign_in.url);
        let registered: serde_json::Value =
            serde_json::from_str(&server.requests("POST", "/register")[0].body).unwrap();
        assert_eq!(q["response_type"], "code");
        assert_eq!(q["client_id"], "cid-123");
        assert_eq!(q["redirect_uri"], registered["redirect_uris"][0]);
        assert_eq!(q["code_challenge_method"], "S256");
        assert_eq!(q["resource"], format!("{}/mcp", server.base));
        assert_eq!(q["scope"], "default");
        assert!(!q["state"].is_empty() && q["state"] != q["code_challenge"]);
    }

    #[test]
    fn the_whole_sign_in_ends_with_a_record_in_pis_shape() {
        let server = fake(true, 201);
        let url = format!("{}/mcp", server.base);
        let sign_in = start(&url).unwrap();
        let q = query(&sign_in.url);
        let (redirect, state) = (q["redirect_uri"].clone(), q["state"].clone());
        let go = std::thread::spawn(move || browser(&redirect, &format!("code=abc&state={state}")));
        let record = sign_in.wait(Duration::from_secs(5)).unwrap();
        go.join().unwrap();

        let token_request = &server.requests("POST", "/token")[0].body;
        let form: std::collections::HashMap<String, String> =
            url::form_urlencoded::parse(token_request.as_bytes()).into_owned().collect();
        assert_eq!(form["grant_type"], "authorization_code");
        assert_eq!(form["code"], "abc");
        assert_eq!(form["client_id"], "cid-123");
        assert_eq!(form["redirect_uri"], q["redirect_uri"]);
        assert_eq!(form["resource"], url);
        assert!(form["code_verifier"].len() >= 43);

        assert_eq!(record["serverUrl"], url);
        assert_eq!(record["discovery"]["authorizationServerUrl"], server.base);
        assert_eq!(record["discovery"]["resourceMetadata"]["resource"], url);
        assert_eq!(record["discovery"]["authorizationServerMetadata"]["issuer"], server.base);
        assert_eq!(record["clientInformation"]["client_id"], "cid-123");
        assert_eq!(record["oauthState"], q["state"]);
        assert_eq!(record["codeVerifier"], form["code_verifier"]);
        assert_eq!(record["tokens"]["access_token"], "at");
        assert_eq!(record["tokens"]["refresh_token"], "rt");
        assert_eq!(record["tokens"]["token_type"], "bearer");
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
        let expires = record["tokensExpireAt"].as_i64().unwrap();
        assert!(expires > now + 3_000_000 && expires <= now + 3_600_000 + 5_000, "{expires} vs {now}");
    }

    #[test]
    fn a_callback_with_the_wrong_state_asks_for_no_tokens() {
        let server = fake(true, 201);
        let sign_in = start(&format!("{}/mcp", server.base)).unwrap();
        let redirect = query(&sign_in.url)["redirect_uri"].clone();
        let go = std::thread::spawn(move || browser(&redirect, "code=abc&state=not-ours"));
        assert_eq!(sign_in.wait(Duration::from_secs(5)).unwrap_err(), OAuthError::StateMismatch);
        go.join().unwrap();
        assert!(server.requests("POST", "/token").is_empty());
    }

    #[test]
    fn a_denied_sign_in_is_reported() {
        let server = fake(true, 201);
        let sign_in = start(&format!("{}/mcp", server.base)).unwrap();
        let q = query(&sign_in.url);
        let (redirect, state) = (q["redirect_uri"].clone(), q["state"].clone());
        let go = std::thread::spawn(move || browser(&redirect, &format!("error=access_denied&state={state}")));
        assert_eq!(sign_in.wait(Duration::from_secs(5)).unwrap_err(), OAuthError::Denied("access_denied".into()));
        go.join().unwrap();
    }

    #[test]
    fn an_abandoned_sign_in_times_out() {
        let server = fake(true, 201);
        let sign_in = start(&format!("{}/mcp", server.base)).unwrap();
        assert_eq!(sign_in.wait(Duration::from_millis(200)).unwrap_err(), OAuthError::Timeout);
    }
}
