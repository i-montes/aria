use aria_engine::engine::{Engine, EngineError, EngineEvent, SpawnSpec};
use aria_engine::protocol::UiAnswer;
use serde_json::json;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::Duration;

fn spec(program: &str, args: Vec<String>) -> SpawnSpec {
    SpawnSpec { program: PathBuf::from(program), args, env: vec![], unset: vec![], cwd: std::env::temp_dir() }
}

fn fake() -> (Engine, Receiver<EngineEvent>) {
    let script = format!("{}/tests/fake_pi.sh", env!("CARGO_MANIFEST_DIR"));
    Engine::spawn(spec("/bin/sh", vec![script])).expect("spawn the fake pi")
}

fn next(events: &Receiver<EngineEvent>) -> EngineEvent {
    events.recv_timeout(Duration::from_secs(5)).expect("an event within 5 s")
}

fn kind(event: &EngineEvent) -> String {
    match event {
        EngineEvent::Event(raw) => raw["type"].as_str().unwrap_or_default().to_string(),
        other => format!("{other:?}"),
    }
}

#[test]
fn request_gets_its_matching_response() {
    let (engine, _events) = fake();
    let response = engine.request("get_state", json!({}), Duration::from_secs(5)).unwrap();
    assert!(response.success);
    assert_eq!(response.data["isStreaming"], json!(false));
}

#[test]
fn prompt_streams_events_in_order() {
    let (engine, events) = fake();
    engine.send("prompt", json!({"message": "hello"})).unwrap();
    // The prompt's own response arrives first, as an event nobody was waiting for.
    let mut kinds = Vec::new();
    while kinds.len() < 4 {
        kinds.push(kind(&next(&events)));
    }
    assert!(kinds[0].starts_with("Response"), "{kinds:?}");
    assert_eq!(&kinds[1..], ["agent_start", "message_update", "agent_settled"]);
}

#[test]
fn a_failed_command_nobody_waited_for_still_reaches_the_events() {
    let (engine, events) = fake();
    engine.send("reject", json!({})).unwrap();
    match next(&events) {
        EngineEvent::Response { success, error, command, .. } => {
            assert!(!success);
            assert_eq!(error.as_deref(), Some("nope"));
            assert_eq!(command, "reject");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn ui_request_arrives_and_answer_reaches_pi() {
    let (engine, events) = fake();
    engine.send("ask", json!({})).unwrap();
    match next(&events) {
        EngineEvent::UiRequest { id, method, raw } => {
            assert_eq!((id.as_str(), method.as_str()), ("u1", "select"));
            assert_eq!(raw["title"], json!("Allow?"));
        }
        other => panic!("{other:?}"),
    }
    engine.answer_ui("u1", UiAnswer::Value("Allow".into())).unwrap();
    match next(&events) {
        EngineEvent::Event(raw) => assert_eq!(raw, json!({"type": "answered", "value": "Allow"})),
        other => panic!("{other:?}"),
    }
}

#[test]
fn exit_is_reported() {
    let (engine, events) = fake();
    engine.send("crash", json!({})).unwrap();
    assert_eq!(next(&events), EngineEvent::Exited { code: Some(3) });
    assert!(!engine.is_alive());
}

#[test]
fn request_after_exit_is_gone() {
    let (engine, events) = fake();
    engine.send("crash", json!({})).unwrap();
    assert!(matches!(next(&events), EngineEvent::Exited { .. }));
    assert!(matches!(engine.request("get_state", json!({}), Duration::from_secs(1)), Err(EngineError::Gone)));
}

#[test]
fn request_times_out_when_pi_stays_silent() {
    let (engine, _events) = fake();
    let result = engine.request("silent", json!({}), Duration::from_millis(200));
    assert!(matches!(result, Err(EngineError::Timeout)));
}

#[test]
fn a_missing_program_is_a_spawn_error() {
    let result = Engine::spawn(spec("/nonexistent/pi", vec![]));
    assert!(matches!(result, Err(EngineError::Spawn(_))));
}

#[test]
fn shutdown_stops_a_running_child() {
    let (engine, events) = fake();
    engine.shutdown();
    assert!(matches!(next(&events), EngineEvent::Exited { .. }));
}

#[test]
fn close_works_through_a_shared_engine_and_kills_a_child_that_ignores_eof() {
    // `sleep` never reads stdin: closing it does not stop it, so `close` must kill it after its grace time.
    let (engine, events) = Engine::spawn(spec("/bin/sh", vec!["-c".into(), "exec sleep 30".into()])).unwrap();
    let engine = std::sync::Arc::new(engine);
    let started = std::time::Instant::now();
    engine.close(Duration::from_millis(300));
    assert!(matches!(events.recv_timeout(Duration::from_secs(5)).unwrap(), EngineEvent::Exited { .. }));
    assert!(started.elapsed() < Duration::from_secs(3), "killed after the grace time, not left running");
    assert!(!engine.is_alive());
}

#[test]
fn close_lets_a_child_that_stops_on_eof_end_by_itself() {
    let (engine, events) = fake();
    engine.close(Duration::from_secs(3));
    assert_eq!(next(&events), EngineEvent::Exited { code: Some(0) });
}

#[test]
fn an_unset_variable_does_not_reach_the_child() {
    std::env::set_var("ARIA_ENGINE_TEST_SECRET", "x");
    let probe = r#"printf '{"type":"probe","value":"%s"}\n' "${ARIA_ENGINE_TEST_SECRET:-absent}"; sleep 1"#;
    let mut spec = spec("/bin/sh", vec!["-c".into(), probe.into()]);
    spec.unset = vec!["ARIA_ENGINE_TEST_SECRET".into()];
    let (_engine, events) = Engine::spawn(spec).expect("spawn the probe");
    match next(&events) {
        EngineEvent::Event(raw) => assert_eq!(raw["value"], "absent"),
        other => panic!("{other:?}"),
    }
}
