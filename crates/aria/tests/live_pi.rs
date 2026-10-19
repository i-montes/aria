//! Talks to the real pinned Pi with the owner's real credentials (a one-word prompt, a few tokens).
//! Run by hand: `cargo test -p aria --test live_pi -- --ignored --nocapture`.
use aria_core::agent;
use aria_core::paths::AriaPaths;
use aria_core::pi;
use aria_core::settings;
use aria_engine::engine::{Engine, EngineEvent, SpawnSpec};
use serde_json::json;
use std::time::{Duration, Instant};

#[test]
#[ignore = "needs the installed Pi and real credentials"]
fn the_pinned_pi_answers_a_prompt_through_the_engine() {
    let real = AriaPaths::from_env();
    // A scratch Aria home that shares the real Pi install and borrows the real credentials.
    let dir = tempfile::tempdir().unwrap();
    let paths = AriaPaths { root: dir.path().to_path_buf() };
    std::os::unix::fs::symlink(real.pi_root(), paths.pi_root()).unwrap();
    std::fs::copy(real.auth(), paths.auth()).expect("run the Aria setup first (auth.json)");
    let loaded = settings::load(&real).ok().flatten();

    let install = pi::locate(&paths, pi::pinned_version()).expect("the pinned Pi is installed");
    agent::prepare(&paths, loaded.as_ref()).unwrap();
    pi::write_shim(&paths, &install).unwrap();
    let path_var = std::env::var("PATH").unwrap_or_default();
    assert!(pi::node_ok(&real, &path_var), "a Node that can run Pi");

    let spec = SpawnSpec {
        program: install.binary,
        args: vec!["--mode".into(), "rpc".into(), "--no-session".into()],
        env: vec![
            ("PI_CODING_AGENT_DIR".into(), paths.agent().display().to_string()),
            ("PATH".into(), pi::path_with_bin(&real, &path_var)),
            ("PI_SKIP_VERSION_CHECK".into(), "1".into()),
        ],
        cwd: dir.path().to_path_buf(),
    };
    let (engine, events) = Engine::spawn(spec).expect("start Pi");
    let state = engine.request("get_state", json!({}), Duration::from_secs(30)).expect("get_state");
    assert!(state.success, "{state:?}");
    println!("model: {}", state.data["model"]["id"]);

    engine.send("prompt", json!({"message": "Reply with the single word: pong"})).unwrap();
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut text = String::new();
    loop {
        let event = events.recv_timeout(deadline.saturating_duration_since(Instant::now())).expect("Pi settled within 120 s");
        match event {
            EngineEvent::Event(raw) if raw["type"] == "message_update" => {
                if let Some(delta) = raw["assistantMessageEvent"]["delta"].as_str() {
                    if raw["assistantMessageEvent"]["type"] == "text_delta" {
                        text.push_str(delta);
                    }
                }
            }
            EngineEvent::Event(raw) if raw["type"] == "agent_settled" => break,
            EngineEvent::Response { success: false, error, .. } => panic!("a command failed: {error:?}"),
            EngineEvent::Exited { code } => panic!("Pi exited: {code:?}"),
            _ => {}
        }
    }
    println!("assistant: {text}");
    assert!(!text.trim().is_empty(), "the assistant said something");
    engine.shutdown();
}

/// Installs the Claude provider with the command Aria builds, into a scratch agent directory, and checks that Pi then
/// offers its models. Downloads about 300 MB. Run by hand: `cargo test -p aria --test live_pi -- --ignored claude`.
#[test]
#[ignore = "downloads the Claude provider"]
fn the_claude_provider_installs_with_arias_command_and_its_models_show_up() {
    use aria_core::agent::{claude_bridge_installed, claude_bridge_install_command};
    let real = AriaPaths::from_env();
    let dir = tempfile::tempdir().unwrap();
    let paths = AriaPaths { root: dir.path().to_path_buf() };
    std::os::unix::fs::symlink(real.pi_root(), paths.pi_root()).unwrap();
    // a stand-in credential: Pi lists a provider's models once it has one (the key is never used)
    std::fs::write(paths.auth(), r#"{"anthropic":{"type":"api_key","key":"sk-ant-listing-only"}}"#).unwrap();
    let path_var = std::env::var("PATH").unwrap_or_default();
    let install = pi::locate(&paths, pi::pinned_version()).expect("the pinned Pi is installed");
    agent::prepare(&paths, None).unwrap();
    assert!(!claude_bridge_installed(&paths));

    let command = claude_bridge_install_command(&paths, &install, &path_var);
    let status = std::process::Command::new(&command.program)
        .args(&command.args)
        .envs(command.env.iter().map(|(k, v)| (k, v)))
        .current_dir(&command.cwd)
        .status()
        .expect("run pi install");
    assert!(status.success());
    assert!(claude_bridge_installed(&paths), "Pi listed the package and downloaded it");

    let spec = SpawnSpec {
        program: install.binary,
        args: vec!["--mode".into(), "rpc".into(), "--no-session".into()],
        env: vec![
            ("PI_CODING_AGENT_DIR".into(), paths.agent().display().to_string()),
            ("PATH".into(), pi::path_with_bin(&paths, &path_var)),
            ("PI_SKIP_VERSION_CHECK".into(), "1".into()),
        ],
        cwd: dir.path().to_path_buf(),
    };
    let (engine, _events) = Engine::spawn(spec).expect("start Pi");
    let models = engine.request("get_available_models", json!({}), Duration::from_secs(60)).expect("models");
    let providers: Vec<String> = models.data["models"].as_array().unwrap().iter().map(|m| m["provider"].as_str().unwrap_or("").to_string()).collect();
    assert!(providers.iter().any(|p| p == "claude-bridge"), "the Claude provider's models show up: {providers:?}");
    assert!(providers.iter().any(|p| p == "anthropic"), "and the API-key provider's too");
    engine.shutdown();
}
