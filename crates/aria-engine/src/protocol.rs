//! Pi's RPC records: what comes out of stdout, and how to write what goes into stdin.

use serde_json::{json, Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Response {
        id: Option<String>,
        command: String,
        success: bool,
        data: Value,
        error: Option<String>,
    },
    /// A dialog (`select`, `confirm`, `input`, `editor`) or a fire-and-forget request (`notify`, …).
    UiRequest {
        id: String,
        method: String,
        raw: Value,
    },
    /// Any session event (`agent_start`, `message_update`, …), verbatim.
    Event {
        kind: String,
        raw: Value,
    },
    Unparsable(String),
}

pub fn parse_line(line: &str) -> Incoming {
    let Ok(value) = serde_json::from_str::<Value>(line) else { return Incoming::Unparsable(line.to_string()) };
    let Some(kind) = value.get("type").and_then(Value::as_str).map(str::to_string) else {
        return Incoming::Unparsable(line.to_string());
    };
    let text = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
    match kind.as_str() {
        "response" => Incoming::Response {
            id: text("id"),
            command: text("command").unwrap_or_default(),
            success: value.get("success").and_then(Value::as_bool).unwrap_or(false),
            data: value.get("data").cloned().unwrap_or(Value::Null),
            error: text("error"),
        },
        "extension_ui_request" => Incoming::UiRequest {
            id: text("id").unwrap_or_default(),
            method: text("method").unwrap_or_default(),
            raw: value,
        },
        _ => Incoming::Event { kind, raw: value },
    }
}

/// One command as a JSON line (without the LF): `type` and `id` merged with `fields`.
pub fn command(kind: &str, id: &str, fields: Value) -> String {
    let mut object = match fields {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    object.insert("type".into(), json!(kind));
    object.insert("id".into(), json!(id));
    Value::Object(object).to_string()
}

#[derive(Debug, Clone, PartialEq)]
pub enum UiAnswer {
    Value(String),
    Confirmed(bool),
    Cancelled,
}

pub fn ui_response(id: &str, answer: &UiAnswer) -> String {
    let mut object = json!({"type": "extension_ui_response", "id": id});
    match answer {
        UiAnswer::Value(value) => object["value"] = json!(value),
        UiAnswer::Confirmed(confirmed) => object["confirmed"] = json!(confirmed),
        UiAnswer::Cancelled => object["cancelled"] = json!(true),
    }
    object.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_response() {
        let line = r#"{"id":"3","type":"response","command":"get_state","success":true,"data":{"isStreaming":false}}"#;
        assert_eq!(
            parse_line(line),
            Incoming::Response {
                id: Some("3".into()),
                command: "get_state".into(),
                success: true,
                data: json!({"isStreaming": false}),
                error: None
            }
        );
        let failed = r#"{"type":"response","command":"parse","success":false,"error":"bad"}"#;
        assert_eq!(
            parse_line(failed),
            Incoming::Response {
                id: None,
                command: "parse".into(),
                success: false,
                data: Value::Null,
                error: Some("bad".into())
            }
        );
    }

    #[test]
    fn parses_an_event() {
        match parse_line(r#"{"type":"agent_start"}"#) {
            Incoming::Event { kind, .. } => assert_eq!(kind, "agent_start"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parses_a_ui_request_with_its_method() {
        let line = r#"{"type":"extension_ui_request","id":"u1","method":"select","title":"Allow?","options":["Allow","Block"]}"#;
        match parse_line(line) {
            Incoming::UiRequest { id, method, raw } => {
                assert_eq!((id.as_str(), method.as_str()), ("u1", "select"));
                assert_eq!(raw["options"], json!(["Allow", "Block"]));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn garbage_is_unparsable() {
        assert!(matches!(parse_line("not json"), Incoming::Unparsable(_)));
        assert!(matches!(parse_line("[1,2]"), Incoming::Unparsable(_)));
        assert!(matches!(parse_line(r#"{"no":"type"}"#), Incoming::Unparsable(_)));
    }

    #[test]
    fn command_merges_type_id_and_fields() {
        let line = command("prompt", "1", json!({"message": "hi"}));
        assert_eq!(
            serde_json::from_str::<Value>(&line).unwrap(),
            json!({"type": "prompt", "id": "1", "message": "hi"})
        );
        assert!(!line.contains('\n'));
    }

    #[test]
    fn ui_response_shapes() {
        let value = |answer| serde_json::from_str::<Value>(&ui_response("u", &answer)).unwrap();
        assert_eq!(
            value(UiAnswer::Value("Allow".into())),
            json!({"type": "extension_ui_response", "id": "u", "value": "Allow"})
        );
        assert_eq!(
            value(UiAnswer::Confirmed(true)),
            json!({"type": "extension_ui_response", "id": "u", "confirmed": true})
        );
        assert_eq!(value(UiAnswer::Cancelled), json!({"type": "extension_ui_response", "id": "u", "cancelled": true}));
    }
}
