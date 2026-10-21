//! Everything Aria needs from Hyprland, with no edits to `~/.config/hypr/`: the hotkey is registered
//! at run time through `hyprctl eval` (Hyprland 0.56+ with the Lua config), the focused monitor comes
//! from Hyprland's IPC socket, and `configreloaded>>` events tell us when to register again.

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum HyprError {
    #[error("Hyprland is not running (no live instance found)")]
    NotRunning,
    #[error("hyprctl eval failed: {0}")]
    Eval(String),
    #[error("could not talk to Hyprland: {0}")]
    Io(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigKind {
    /// Hyprland 0.56+ with the Lua config: `hyprctl eval` works.
    Lua,
    /// The classic hyprlang config: not supported in phase 1.
    Classic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Monitor {
    pub name: String,
    pub x: i32,
    pub y: i32,
}

fn lua_string(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// The Lua that binds `SUPER + A` to `<exe> toggle` and adds the dock's layer rule. It unbinds first,
/// so registering twice never leaves two binds that would toggle twice.
pub fn registration_lua(exe: &Path) -> String {
    format!(
        r#"hl.unbind("SUPER + A"); hl.bind("SUPER + A", hl.dsp.exec_cmd("{} toggle"), {{ description = "Toggle Aria" }}); hl.layer_rule({{ name = "aria-dock", match = {{ namespace = "^aria-dock$" }}, blur = false, no_anim = true }})"#,
        lua_string(&exe.to_string_lossy())
    )
}

fn runtime_dir() -> PathBuf {
    PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into()))
}

/// The newest Hyprland instance under `<runtime>/hypr/` that still has its request socket.
pub fn discover_signature(runtime: &Path) -> Option<String> {
    std::fs::read_dir(runtime.join("hypr"))
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let modified = std::fs::metadata(entry.path().join(".socket.sock")).ok()?.modified().ok()?;
            Some((modified, entry.file_name().to_string_lossy().into_owned()))
        })
        .max()
        .map(|(_, name)| name)
}

/// `$HYPRLAND_INSTANCE_SIGNATURE` while that instance is alive. A service that outlives its Hyprland
/// (or starts without the variable) finds the live instance instead.
fn instance_signature() -> Result<String, HyprError> {
    let runtime = runtime_dir();
    if let Ok(signature) = std::env::var("HYPRLAND_INSTANCE_SIGNATURE") {
        if runtime.join("hypr").join(&signature).join(".socket.sock").exists() {
            return Ok(signature);
        }
    }
    discover_signature(&runtime).ok_or(HyprError::NotRunning)
}

fn hyprctl_eval(lua: &str) -> Result<String, HyprError> {
    let signature = instance_signature()?;
    let output = Command::new("hyprctl")
        .env("HYPRLAND_INSTANCE_SIGNATURE", signature)
        .args(["eval", lua])
        .output()
        .map_err(|e| HyprError::Io(e.to_string()))?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn register_hotkey(exe: &Path) -> Result<(), HyprError> {
    match hyprctl_eval(&registration_lua(exe))?.as_str() {
        "ok" => Ok(()),
        other => Err(HyprError::Eval(other.to_string())),
    }
}

pub fn config_kind_from_eval(output: &str) -> ConfigKind {
    if output.trim() == "ok" {
        ConfigKind::Lua
    } else {
        ConfigKind::Classic
    }
}

pub fn detect_config() -> Result<ConfigKind, HyprError> {
    Ok(config_kind_from_eval(&hyprctl_eval("return 1")?))
}

/// One of Hyprland's two sockets: `.socket.sock` (requests) or `.socket2.sock` (events).
pub fn socket_path(name: &str) -> Result<PathBuf, HyprError> {
    Ok(runtime_dir().join("hypr").join(instance_signature()?).join(name))
}

/// The monitor marked `focused` in the output of `hyprctl monitors -j`.
pub fn parse_focused(monitors_json: &str) -> Option<Monitor> {
    let monitors: serde_json::Value = serde_json::from_str(monitors_json).ok()?;
    let focused = monitors.as_array()?.iter().find(|m| m["focused"].as_bool() == Some(true))?;
    Some(Monitor {
        name: focused["name"].as_str()?.to_string(),
        x: focused["x"].as_i64()? as i32,
        y: focused["y"].as_i64()? as i32,
    })
}

pub fn focused_monitor_at(socket: &Path) -> Option<Monitor> {
    let mut stream = UnixStream::connect(socket).ok()?;
    stream.write_all(b"j/monitors").ok()?;
    let mut body = String::new();
    stream.read_to_string(&mut body).ok()?;
    parse_focused(&body)
}

pub fn focused_monitor() -> Option<Monitor> {
    focused_monitor_at(&socket_path(".socket.sock").ok()?)
}

pub fn is_config_reloaded(event_line: &str) -> bool {
    event_line.starts_with("configreloaded>>")
}

/// Calls `on_reload` for every `configreloaded>>` event. The socket path is asked for again on every
/// attempt, so a Hyprland that was not up yet, or restarted with a new instance, is followed; a
/// dropped connection is retried after `retry`.
pub fn watch_events_with(
    resolve: impl Fn() -> Option<PathBuf> + Send + 'static,
    retry: Duration,
    on_reload: impl Fn() + Send + 'static,
) -> JoinHandle<()> {
    std::thread::spawn(move || loop {
        if let Some(path) = resolve() {
            if let Ok(stream) = UnixStream::connect(&path) {
                for line in BufReader::new(stream).lines() {
                    match line {
                        Ok(line) if is_config_reloaded(&line) => on_reload(),
                        Ok(_) => {}
                        Err(_) => break,
                    }
                }
            }
        }
        std::thread::sleep(retry);
    })
}

/// The same for a known socket path.
pub fn watch_events_at(path: PathBuf, retry: Duration, on_reload: impl Fn() + Send + 'static) -> JoinHandle<()> {
    watch_events_with(move || Some(path.clone()), retry, on_reload)
}

pub fn watch_events(on_reload: impl Fn() + Send + 'static) -> JoinHandle<()> {
    watch_events_with(|| socket_path(".socket2.sock").ok(), Duration::from_secs(1), on_reload)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Registration {
    /// Hyprland is not reachable yet (or refused the bind): trying again.
    WaitingForHyprland,
    /// The classic configuration: the hotkey cannot be registered; not retried.
    ClassicConfig,
    Registered,
}

/// Registers the hotkey, retrying every `retry` until Hyprland answers. Returns how it settled:
/// `Registered` or `ClassicConfig`. `on_state` hears about every attempt.
pub fn keep_registered_with(
    detect: impl Fn() -> Result<ConfigKind, HyprError>,
    register: impl Fn() -> Result<(), HyprError>,
    retry: Duration,
    on_state: impl Fn(Registration),
) -> Registration {
    loop {
        match detect() {
            Ok(ConfigKind::Classic) => {
                on_state(Registration::ClassicConfig);
                return Registration::ClassicConfig;
            }
            Ok(ConfigKind::Lua) if register().is_ok() => {
                on_state(Registration::Registered);
                return Registration::Registered;
            }
            _ => {
                on_state(Registration::WaitingForHyprland);
                std::thread::sleep(retry);
            }
        }
    }
}

/// In the background: register the hotkey as soon as Hyprland can take it, then register it again
/// after every config reload (a reload drops runtime binds).
pub fn keep_hotkey_registered(
    exe: PathBuf,
    retry: Duration,
    on_state: impl Fn(Registration) + Send + 'static,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let settled = keep_registered_with(detect_config, || register_hotkey(&exe), retry, on_state);
        if settled == Registration::Registered {
            let _ = watch_events(move || {
                if let Err(error) = register_hotkey(&exe) {
                    eprintln!("aria: could not register SUPER + A after a reload: {error}");
                }
            })
            .join();
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::os::unix::net::UnixListener;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const MONITORS: &str = r#"[
        {"id":0,"name":"DP-1","width":1920,"height":1080,"x":0,"y":0,"scale":1.25,"focused":false},
        {"id":1,"name":"DP-2","width":1920,"height":1080,"x":1536,"y":0,"scale":1.25,"focused":true}
    ]"#;

    #[test]
    fn discovers_the_newest_live_instance() {
        let runtime = tempfile::tempdir().unwrap();
        let hypr = runtime.path().join("hypr");
        let make = |name: &str, with_socket: bool| {
            std::fs::create_dir_all(hypr.join(name)).unwrap();
            if with_socket {
                std::fs::write(hypr.join(name).join(".socket.sock"), "").unwrap();
                std::thread::sleep(Duration::from_millis(20)); // distinct modification times
            }
        };
        make("old_instance", true);
        make("dead_instance", false);
        make("new_instance", true);
        assert_eq!(discover_signature(runtime.path()), Some("new_instance".to_string()));
    }

    #[test]
    fn discovers_nothing_without_instances() {
        let runtime = tempfile::tempdir().unwrap();
        assert_eq!(discover_signature(runtime.path()), None);
        std::fs::create_dir_all(runtime.path().join("hypr").join("dead")).unwrap();
        assert_eq!(discover_signature(runtime.path()), None);
    }

    #[test]
    fn watcher_follows_a_socket_path_that_appears_or_changes() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join(".socket2.sock");
        let gone = dir.path().join("gone.sock");
        let listener = UnixListener::bind(&real).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.write_all(b"configreloaded>>\n").unwrap();
            let mut hold = String::new();
            let _ = BufReader::new(stream).read_line(&mut hold);
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let resolve = {
            let (calls, real) = (calls.clone(), real.clone());
            // Hyprland not up yet, then an old instance's dead path, then the live one.
            move || match calls.fetch_add(1, Ordering::SeqCst) {
                0 => None,
                1 => Some(gone.clone()),
                _ => Some(real.clone()),
            }
        };
        let fired = Arc::new(AtomicUsize::new(0));
        let seen = fired.clone();
        watch_events_with(resolve, Duration::from_millis(20), move || {
            seen.fetch_add(1, Ordering::SeqCst);
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while fired.load(Ordering::SeqCst) < 1 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(fired.load(Ordering::SeqCst), 1);
        assert!(calls.load(Ordering::SeqCst) >= 3);
        drop(server);
    }

    fn states_until_settled(
        detect: impl Fn() -> Result<ConfigKind, HyprError>,
        register: impl Fn() -> Result<(), HyprError>,
    ) -> (Vec<Registration>, Registration) {
        let seen = std::cell::RefCell::new(Vec::new());
        let last = keep_registered_with(detect, register, Duration::from_millis(1), |s| seen.borrow_mut().push(s));
        (seen.into_inner(), last)
    }

    #[test]
    fn registration_waits_for_hyprland_and_then_registers() {
        let attempts = std::cell::Cell::new(0);
        let (seen, last) = states_until_settled(
            || {
                attempts.set(attempts.get() + 1);
                if attempts.get() <= 2 {
                    Err(HyprError::NotRunning)
                } else {
                    Ok(ConfigKind::Lua)
                }
            },
            || Ok(()),
        );
        assert_eq!(
            seen,
            vec![Registration::WaitingForHyprland, Registration::WaitingForHyprland, Registration::Registered]
        );
        assert_eq!(last, Registration::Registered);
    }

    #[test]
    fn registration_retries_when_the_bind_is_refused_once() {
        let refused = std::cell::Cell::new(true);
        let (seen, last) = states_until_settled(
            || Ok(ConfigKind::Lua),
            || if refused.replace(false) { Err(HyprError::Eval("not ready".into())) } else { Ok(()) },
        );
        assert_eq!(seen, vec![Registration::WaitingForHyprland, Registration::Registered]);
        assert_eq!(last, Registration::Registered);
    }

    #[test]
    fn a_classic_config_is_reported_and_not_retried() {
        let (seen, last) =
            states_until_settled(|| Ok(ConfigKind::Classic), || panic!("must not register on a classic config"));
        assert_eq!(seen, vec![Registration::ClassicConfig]);
        assert_eq!(last, Registration::ClassicConfig);
    }

    #[test]
    fn lua_is_exact() {
        assert_eq!(
            registration_lua(Path::new("/home/u/.local/bin/aria")),
            r#"hl.unbind("SUPER + A"); hl.bind("SUPER + A", hl.dsp.exec_cmd("/home/u/.local/bin/aria toggle"), { description = "Toggle Aria" }); hl.layer_rule({ name = "aria-dock", match = { namespace = "^aria-dock$" }, blur = false, no_anim = true })"#
        );
    }

    #[test]
    fn lua_escapes_quotes_and_backslashes() {
        let lua = registration_lua(Path::new(r#"/tmp/a"b\c/aria"#));
        assert!(lua.contains(r#"exec_cmd("/tmp/a\"b\\c/aria toggle")"#), "{lua}");
    }

    #[test]
    fn eval_output_decides_the_config_kind() {
        assert_eq!(config_kind_from_eval("ok"), ConfigKind::Lua);
        assert_eq!(config_kind_from_eval("ok\n"), ConfigKind::Lua);
        assert_eq!(config_kind_from_eval("unknown request"), ConfigKind::Classic);
        assert_eq!(config_kind_from_eval(""), ConfigKind::Classic);
    }

    #[test]
    fn parse_focused_picks_focused() {
        assert_eq!(parse_focused(MONITORS), Some(Monitor { name: "DP-2".into(), x: 1536, y: 0 }));
    }

    #[test]
    fn parse_focused_none() {
        assert_eq!(parse_focused("[]"), None);
        assert_eq!(parse_focused("not json"), None);
        assert_eq!(parse_focused(r#"[{"name":"DP-1","x":0,"y":0,"focused":false}]"#), None);
    }

    #[test]
    fn reload_event() {
        assert!(is_config_reloaded("configreloaded>>"));
        assert!(!is_config_reloaded("workspace>>2"));
    }

    #[test]
    fn focused_monitor_asks_the_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".socket.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 64];
            let n = stream.read(&mut request).unwrap();
            assert_eq!(&request[..n], b"j/monitors");
            stream.write_all(MONITORS.as_bytes()).unwrap();
        });
        assert_eq!(focused_monitor_at(&path), Some(Monitor { name: "DP-2".into(), x: 1536, y: 0 }));
        server.join().unwrap();
    }

    #[test]
    fn watcher_reconnects_and_fires_on_every_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".socket2.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let server = std::thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            first.write_all(b"workspace>>1\nconfigreloaded>>\n").unwrap();
            drop(first); // Hyprland restarted: the watcher has to reconnect
            let (mut second, _) = listener.accept().unwrap();
            second.write_all(b"configreloaded>>\n").unwrap();
            let mut hold = String::new();
            let _ = BufReader::new(second).read_line(&mut hold); // keep it open until the test ends
        });
        let count = Arc::new(AtomicUsize::new(0));
        let seen = count.clone();
        watch_events_at(path, Duration::from_millis(50), move || {
            seen.fetch_add(1, Ordering::SeqCst);
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while count.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(count.load(Ordering::SeqCst), 2);
        drop(server);
    }
}
