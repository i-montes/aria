//! Session rotation against the real pinned Pi with the owner's real credentials, in a scratch Aria home: two short
//! prompts, a compaction, `/aria-rotate`, and the new session starting from the compaction's summary (a few model calls).
//! Run by hand: `cargo test -p aria --test live_rotation -- --ignored --nocapture`.
mod common;

use aria_core::agent;
use aria_core::paths::AriaPaths;
use aria_core::pi;
use aria_core::settings;
use aria_engine::engine::{Engine, EngineEvent, SpawnSpec};
use serde_json::{json, Value};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

/// Sends a prompt and waits until its run settles.
fn run(engine: &Engine, events: &Receiver<EngineEvent>, text: &str) {
    engine.send("prompt", json!({"message": text})).unwrap();
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match events.recv_timeout(left).expect("Pi settled within 120 s") {
            EngineEvent::Event(raw) if raw["type"] == "agent_settled" => return,
            EngineEvent::Response { success: false, error, .. } => panic!("a command failed: {error:?}"),
            EngineEvent::Exited { code } => panic!("Pi exited: {code:?}"),
            _ => {}
        }
    }
}

fn request(engine: &Engine, command: &str, fields: Value, timeout: Duration) -> Value {
    let response = engine.request(command, fields, timeout).unwrap_or_else(|e| panic!("{command}: {e}"));
    assert!(response.success, "{command}: {:?}", response.error);
    response.data
}

#[test]
#[ignore = "needs the installed Pi and real credentials"]
fn a_compacted_session_rotates_into_one_that_starts_from_the_summary() {
    let real = AriaPaths::from_env();
    // A scratch Aria home that shares the real Pi install and borrows the real credentials.
    let dir = tempfile::tempdir().unwrap();
    let paths = AriaPaths::new(dir.path().to_path_buf());
    std::os::unix::fs::symlink(real.pi_root(), paths.pi_root()).unwrap();
    common::borrow_auth(&real, &paths);
    let loaded = settings::load(&real).ok().flatten();

    let install = pi::locate(&paths, pi::pinned_version()).expect("the pinned Pi is installed");
    agent::prepare(&paths, loaded.as_ref()).unwrap();
    pi::write_shim(&paths, &install, false).unwrap();
    // Keep almost nothing, so two short prompts are enough to compact.
    let pi_settings = paths.agent().join("settings.json");
    let mut value: Value = serde_json::from_str(&std::fs::read_to_string(&pi_settings).unwrap()).unwrap();
    value["compaction"]["keepRecentTokens"] = json!(1);
    std::fs::write(&pi_settings, value.to_string()).unwrap();
    let path_var = std::env::var("PATH").unwrap_or_default();
    assert!(pi::node_ok(&real, &path_var), "a Node that can run Pi");

    let sessions = agent::sessions_dir(&paths);
    let workdir = dir.path().join("work");
    std::fs::create_dir_all(&workdir).unwrap();
    let spec = SpawnSpec {
        program: install.binary,
        args: vec![
            "--mode".into(),
            "rpc".into(),
            "--session-dir".into(),
            sessions.display().to_string(),
            "--session-id".into(),
            "live-rotation".into(),
        ],
        env: vec![
            ("PI_CODING_AGENT_DIR".into(), paths.agent().display().to_string()),
            ("PATH".into(), pi::path_with_bin(&real, &path_var)),
            ("PI_SKIP_VERSION_CHECK".into(), "1".into()),
        ],
        unset: pi::unset_for_pi(std::env::vars()),
        cwd: workdir,
    };
    let (engine, events) = Engine::spawn(spec).expect("start Pi");
    let state = request(&engine, "get_state", json!({}), Duration::from_secs(30));
    assert_eq!(state["sessionId"], json!("live-rotation"));

    run(&engine, &events, "Remember the word: lantern. Reply with OK.");
    run(&engine, &events, "Reply with the single word: pong");
    let compacted = request(&engine, "compact", json!({}), Duration::from_secs(600));
    println!("compacted: {compacted}");

    request(&engine, "prompt", json!({"message": "/aria-rotate"}), Duration::from_secs(60));
    let state = request(&engine, "get_state", json!({}), Duration::from_secs(30));
    let rotated = state["sessionId"].as_str().unwrap_or_default().to_string();
    println!("new session: {rotated} at {}", state["sessionFile"]);
    assert!(!rotated.is_empty() && rotated != "live-rotation", "Pi runs on a new session");

    let messages = request(&engine, "get_messages", json!({}), Duration::from_secs(30))["messages"].clone();
    let first = &messages[0];
    assert_eq!(first["role"], json!("custom"), "{first}");
    assert_eq!(first["customType"], json!("aria-earlier"), "{first}");
    assert_eq!(first["details"]["from"], json!("live-rotation"));
    assert!(messages.as_array().is_some_and(|all| all.len() > 1), "the kept context follows the summary");
    engine.shutdown();
}
