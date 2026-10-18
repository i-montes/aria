//! Is this key accepted by its provider? One cheap request per provider; the key is never logged.

use serde_json::{json, Value};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyKind {
    OpenCodeGo,
    AnthropicApi,
    TypeSafe,
    Tavily,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeRequest {
    pub method: &'static str,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyCheck {
    Valid,
    Rejected,
    Unavailable(String),
}

fn header(name: &str, value: String) -> (String, String) {
    (name.to_string(), value)
}

pub fn probe_for(kind: KeyKind, key: &str) -> ProbeRequest {
    match kind {
        KeyKind::OpenCodeGo => ProbeRequest {
            method: "GET",
            url: "https://opencode.ai/zen/go/v1/usage".into(),
            headers: vec![header("authorization", format!("Bearer {key}"))],
            body: None,
        },
        KeyKind::AnthropicApi => ProbeRequest {
            method: "GET",
            url: "https://api.anthropic.com/v1/models".into(),
            headers: vec![header("x-api-key", key.to_string()), header("anthropic-version", "2023-06-01".into())],
            body: None,
        },
        // Tavily's MCP endpoint accepts any key when a session starts, so it cannot validate one.
        // `/usage` answers 401 for a bad key and costs no credits.
        KeyKind::Tavily => ProbeRequest {
            method: "GET",
            url: "https://api.tavily.com/usage".into(),
            headers: vec![header("Authorization", format!("Bearer {key}"))],
            body: None,
        },
        KeyKind::TypeSafe => ProbeRequest {
            method: "POST",
            url: "https://api.typesafe.ai/v1/systemone".into(),
            headers: vec![header("Authorization", format!("Bearer {key}")), header("Content-Type", "application/json".into())],
            body: Some(json!({
                "model": "jev-latest",
                "state": "ping",
                "questions": { "ok": { "type": "noul", "instructions": "The state is a short test message" } }
            })),
        },
    }
}

pub fn classify(status: u16) -> KeyCheck {
    match status {
        200..=299 => KeyCheck::Valid,
        401 | 403 => KeyCheck::Rejected,
        other => KeyCheck::Unavailable(format!("HTTP {other}")),
    }
}

/// Runs the probe (10 s timeout). Network and TLS failures are `Unavailable`, never `Rejected`.
pub fn check(probe: &ProbeRequest) -> KeyCheck {
    let mut request = ureq::request(probe.method, &probe.url).timeout(Duration::from_secs(10));
    for (name, value) in &probe.headers {
        request = request.set(name, value);
    }
    let result = match &probe.body {
        Some(body) => request.send_json(body.clone()),
        None => request.call(),
    };
    match result {
        Ok(response) => classify(response.status()),
        Err(ureq::Error::Status(code, _)) => classify(code),
        Err(ureq::Error::Transport(error)) => KeyCheck::Unavailable(error.to_string()),
    }
}

pub fn check_key(kind: KeyKind, key: &str) -> KeyCheck {
    check(&probe_for(kind, key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn h(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn probe_opencode_go() {
        let p = probe_for(KeyKind::OpenCodeGo, "k1");
        assert_eq!((p.method, p.url.as_str()), ("GET", "https://opencode.ai/zen/go/v1/usage"));
        assert_eq!(p.headers, h(&[("authorization", "Bearer k1")]));
        assert!(p.body.is_none());
    }

    #[test]
    fn probe_anthropic_api() {
        let p = probe_for(KeyKind::AnthropicApi, "k2");
        assert_eq!((p.method, p.url.as_str()), ("GET", "https://api.anthropic.com/v1/models"));
        assert_eq!(p.headers, h(&[("x-api-key", "k2"), ("anthropic-version", "2023-06-01")]));
    }

    #[test]
    fn probe_typesafe() {
        let p = probe_for(KeyKind::TypeSafe, "k3");
        assert_eq!((p.method, p.url.as_str()), ("POST", "https://api.typesafe.ai/v1/systemone"));
        assert_eq!(p.headers, h(&[("Authorization", "Bearer k3"), ("Content-Type", "application/json")]));
        assert_eq!(
            p.body.unwrap(),
            json!({
                "model": "jev-latest",
                "state": "ping",
                "questions": { "ok": { "type": "noul", "instructions": "The state is a short test message" } }
            })
        );
    }

    #[test]
    fn probe_tavily_uses_the_free_usage_endpoint() {
        // The MCP endpoint accepts any key at initialize; /usage is the call that checks it, for free.
        let p = probe_for(KeyKind::Tavily, "tvly-1");
        assert_eq!((p.method, p.url.as_str()), ("GET", "https://api.tavily.com/usage"));
        assert_eq!(p.headers, h(&[("Authorization", "Bearer tvly-1")]));
        assert!(p.body.is_none());
    }

    #[test]
    fn classify_statuses() {
        assert_eq!(classify(200), KeyCheck::Valid);
        assert_eq!(classify(204), KeyCheck::Valid);
        assert_eq!(classify(401), KeyCheck::Rejected);
        assert_eq!(classify(403), KeyCheck::Rejected);
        assert_eq!(classify(500), KeyCheck::Unavailable("HTTP 500".into()));
    }

    #[test]
    fn check_key_unreachable() {
        let probe = ProbeRequest { method: "GET", url: "http://127.0.0.1:9".into(), headers: vec![], body: None };
        assert!(matches!(check(&probe), KeyCheck::Unavailable(_)));
    }
}
