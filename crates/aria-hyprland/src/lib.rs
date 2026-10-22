//! Everything Aria needs from Hyprland, with no edits to `~/.config/hypr/`: the hotkey is registered
//! at run time through `hyprctl eval` (Hyprland 0.56+ with the Lua config), the focused monitor comes
//! from Hyprland's IPC socket, and `configreloaded>>` events tell us when to register again.

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use aria_core::shortcut::{self, Platform, Shortcut};
use serde_json::Value;

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
    /// The height a layer surface anchored to the top and the bottom gets there, in logical pixels: the monitor's
    /// less what is reserved above and below it (the bar). None when `hyprctl monitors` does not say.
    pub usable_height: Option<i32>,
}

fn lua_string(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// What Aria's bind says in `hyprctl binds` (`description`), which is how its own bind is told from someone else's:
/// with the Lua config every bind's dispatcher is `__lua`.
pub const BIND_DESCRIPTION: &str = "Toggle Aria";

/// A shortcut as Hyprland's Lua `hl.bind` takes it: `SUPER + SHIFT + A`, `SUPER + SPACE`.
pub fn hypr_combo(shortcut: &Shortcut) -> String {
    shortcut.mods.iter().cloned().chain([shortcut.key.to_ascii_uppercase()]).collect::<Vec<_>>().join(" + ")
}

/// The Lua that binds `shortcut` to `<exe> toggle` and adds the dock's layer rule. It unbinds `previous` (the
/// shortcuts Aria bound before) and the new one first, so a change leaves no old bind behind and registering twice
/// never leaves two binds that would toggle twice. The layer rule turns blur off and Hyprland's layer animation off: the
/// dock appears and goes at once (`aria::dock::gtk` says why nothing animates it). `no_anim` is said outright because a
/// rule declared again under the same name keeps the fields it does not name.
pub fn registration_lua(exe: &Path, shortcut: &Shortcut, previous: &[Shortcut]) -> String {
    let combo = hypr_combo(shortcut);
    let mut unbind: Vec<String> = Vec::new();
    for old in previous.iter().map(hypr_combo).filter(|old| *old != combo).chain([combo.clone()]) {
        if !unbind.contains(&old) {
            unbind.push(old);
        }
    }
    let unbinds: String = unbind.iter().map(|c| format!(r#"hl.unbind("{}"); "#, lua_string(c))).collect();
    format!(
        r#"{unbinds}hl.bind("{}", hl.dsp.exec_cmd("{} toggle"), {{ description = "{BIND_DESCRIPTION}" }}); hl.layer_rule({{ name = "aria-dock", match = {{ namespace = "^aria-dock$" }}, blur = false, no_anim = true }})"#,
        lua_string(&combo),
        lua_string(&exe.to_string_lossy())
    )
}

/// Hyprland's modifier mask for a shortcut (SHIFT 1, CTRL 4, ALT 8, SUPER 64).
pub fn modmask(shortcut: &Shortcut) -> u64 {
    shortcut
        .mods
        .iter()
        .map(|m| match m.as_str() {
            "SHIFT" => 1,
            "CTRL" => 4,
            "ALT" => 8,
            "SUPER" => 64,
            _ => 0,
        })
        .sum()
}

/// Whether a bind may be on a digit key. Omarchy binds digits by keycode (`code:10` … `code:19`), and `hyprctl binds`
/// lists those with an empty key and keycode 0, so which digit is unknown: any such bind counts (conservative), as
/// does a keycode in 10..=19 when one is listed.
fn maybe_a_digit(bind: &Value) -> bool {
    let key = bind["key"].as_str().unwrap_or("");
    let code = bind["keycode"].as_u64().unwrap_or(0);
    (key.is_empty() && code == 0) || (10..=19).contains(&code)
}

/// Whether one entry of `hyprctl -j binds` is Aria's own: its description, or (a bind made outside the Lua config)
/// `exec <exe> toggle`.
fn is_own(bind: &Value, exe: &Path) -> bool {
    bind["description"].as_str() == Some(BIND_DESCRIPTION)
        || (bind["dispatcher"].as_str() == Some("exec")
            && bind["arg"].as_str() == Some(format!("{} toggle", exe.to_string_lossy()).as_str()))
}

fn same_keys(bind: &Value, shortcut: &Shortcut) -> bool {
    let key_matches = bind["key"].as_str().is_some_and(|key| key.eq_ignore_ascii_case(&shortcut.key))
        || (shortcut.key.chars().all(|c| c.is_ascii_digit()) && maybe_a_digit(bind));
    bind["modmask"].as_u64() == Some(modmask(shortcut))
        && key_matches
        && bind["submap"].as_str().unwrap_or("").is_empty()
        && bind["mouse"].as_bool() != Some(true)
}

/// Whether something other than Aria binds `shortcut`, from the output of `hyprctl -j binds`: a bind outside any
/// submap with the same modifiers and key that is not Aria's own.
pub fn taken(binds_json: &Value, shortcut: &Shortcut, exe: &Path) -> bool {
    binds_json.as_array().is_some_and(|binds| binds.iter().any(|b| same_keys(b, shortcut) && !is_own(b, exe)))
}

/// The shortcuts Aria's own binds use right now (left by an earlier dock, say), from `hyprctl -j binds`.
pub fn own_binds(binds_json: &Value, exe: &Path) -> Vec<Shortcut> {
    let Some(binds) = binds_json.as_array() else { return Vec::new() };
    binds
        .iter()
        .filter(|b| is_own(b, exe) && b["submap"].as_str().unwrap_or("").is_empty())
        .filter_map(|b| {
            let mask = b["modmask"].as_u64()?;
            let mods: Vec<&str> = [(64, "SUPER"), (4, "CTRL"), (8, "ALT"), (1, "SHIFT")]
                .iter()
                .filter(|(bit, _)| mask & bit != 0)
                .map(|(_, m)| *m)
                .collect();
            shortcut::parse(&format!("{} + {}", mods.join(" + "), b["key"].as_str()?), Platform::Linux)
        })
        .collect()
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

/// The shortcut the dock uses and the ones it tried first, which something else binds (shared with macOS).
pub use aria_core::shortcut::InUse;

/// Aria's bind, as it stands: what it bound last (so a change unbinds it).
#[derive(Debug, Clone)]
pub struct Hotkey {
    exe: PathBuf,
    bound: Option<Shortcut>,
}

impl Hotkey {
    pub fn new(exe: PathBuf) -> Hotkey {
        Hotkey { exe, bound: None }
    }

    /// The Lua for the first free one of `candidates` (by `binds_json`, the output of `hyprctl -j binds`) and what it
    /// would use. It unbinds Aria's previous shortcut and Aria's own binds listed in `binds_json`. When every candidate
    /// is taken, the first is bound anyway, as before shortcuts could be chosen (and reported in use, the others taken).
    pub fn plan(&self, binds_json: &Value, candidates: &[Shortcut]) -> (String, InUse) {
        let (chosen, skipped) = shortcut::first_free(candidates, |s| taken(binds_json, s, &self.exe));
        // What Hyprland lists as Aria's is unbound; what Aria bound last only when Hyprland gave no list (a reload drops
        // runtime binds, and a bind someone added to their config since must stay).
        let previous: Vec<Shortcut> =
            if binds_json.is_array() { own_binds(binds_json, &self.exe) } else { self.bound.iter().cloned().collect() };
        (registration_lua(&self.exe, &chosen, &previous), InUse { shortcut: chosen, taken: skipped })
    }

    /// Binds the first free one of `candidates`, asking Hyprland for its binds first.
    pub fn register(&mut self, candidates: &[Shortcut]) -> Result<InUse, HyprError> {
        let binds = socket_path(".socket.sock").ok().and_then(|path| binds_at(&path)).unwrap_or(Value::Null);
        let (lua, in_use) = self.plan(&binds, candidates);
        if candidates.iter().all(|s| taken(&binds, s, &self.exe)) {
            eprintln!("aria: every shortcut Aria can use is taken; binding {} anyway", in_use.shortcut.label());
        }
        match hyprctl_eval(&lua)?.as_str() {
            "ok" => {
                self.bound = Some(in_use.shortcut.clone());
                Ok(in_use)
            }
            other => Err(HyprError::Eval(other.to_string())),
        }
    }
}

/// One request on Hyprland's request socket (`j/monitors`, `j/binds`), its answer as text.
fn request_at(socket: &Path, request: &str) -> Option<String> {
    let mut stream = UnixStream::connect(socket).ok()?;
    stream.write_all(request.as_bytes()).ok()?;
    let mut body = String::new();
    stream.read_to_string(&mut body).ok()?;
    Some(body)
}

/// The output of `hyprctl -j binds`, asked on the request socket.
pub fn binds_at(socket: &Path) -> Option<Value> {
    serde_json::from_str(&request_at(socket, "j/binds")?).ok()
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
    // `reserved` is [left, top, right, bottom], logical pixels; `height` is physical, `scale` makes it logical.
    let scale = focused["scale"].as_f64().filter(|s| *s > 0.0).unwrap_or(1.0);
    let reserved = |side: usize| focused["reserved"][side].as_f64().unwrap_or(0.0);
    let usable_height = focused["height"]
        .as_f64()
        .map(|height| ((height / scale).round() - reserved(1) - reserved(3)) as i32)
        .filter(|height| *height > 0);
    Some(Monitor {
        name: focused["name"].as_str()?.to_string(),
        x: focused["x"].as_i64()? as i32,
        y: focused["y"].as_i64()? as i32,
        usable_height,
    })
}

pub fn focused_monitor_at(socket: &Path) -> Option<Monitor> {
    parse_focused(&request_at(socket, "j/monitors")?)
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Registration {
    /// Hyprland is not reachable yet (or refused the bind): trying again.
    WaitingForHyprland,
    /// The classic configuration: the hotkey cannot be registered; not retried.
    ClassicConfig,
    /// Bound: which shortcut is in use (sent again after every change).
    Registered(InUse),
}

/// Registers the hotkey, retrying every `retry` until Hyprland answers. Returns how it settled:
/// `Registered` or `ClassicConfig`. `on_state` hears about every attempt.
pub fn keep_registered_with(
    detect: impl Fn() -> Result<ConfigKind, HyprError>,
    mut register: impl FnMut() -> Result<InUse, HyprError>,
    retry: Duration,
    on_state: impl Fn(Registration),
) -> Registration {
    loop {
        match detect() {
            Ok(ConfigKind::Classic) => {
                on_state(Registration::ClassicConfig);
                return Registration::ClassicConfig;
            }
            Ok(ConfigKind::Lua) => match register() {
                Ok(in_use) => {
                    on_state(Registration::Registered(in_use.clone()));
                    return Registration::Registered(in_use);
                }
                Err(_) => {
                    on_state(Registration::WaitingForHyprland);
                    std::thread::sleep(retry);
                }
            },
            _ => {
                on_state(Registration::WaitingForHyprland);
                std::thread::sleep(retry);
            }
        }
    }
}

/// Why the hotkey is registered again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wake {
    /// Hyprland reloaded its config, which drops runtime binds: always bind again.
    Reload,
    /// The settings changed: bind again when the `shortcut` setting did.
    Settings,
}

/// The hotkey's whole life, with its pieces passed in: register as soon as Hyprland can take it (the shortcut
/// `source` names, or a fallback), then on every wake bind again (a reload) or bind the new shortcut (a settings change
/// that changed it). `start_watching` runs once, after the first bind. Ends on a classic config, or when every sender
/// of `wakes` is gone.
pub fn run_hotkey(
    detect: impl Fn() -> Result<ConfigKind, HyprError>,
    mut register: impl FnMut(&[Shortcut]) -> Result<InUse, HyprError>,
    source: impl Fn() -> String,
    retry: Duration,
    wakes: Receiver<Wake>,
    start_watching: impl FnOnce(),
    on_state: impl Fn(Registration),
) {
    let mut current = source();
    let settled = keep_registered_with(
        &detect,
        || {
            current = source();
            register(&shortcut::candidates(&current, Platform::Linux))
        },
        retry,
        &on_state,
    );
    if settled == Registration::ClassicConfig {
        return;
    }
    start_watching();
    for wake in wakes {
        let chosen = source();
        if wake == Wake::Settings && chosen == current {
            continue;
        }
        current = chosen;
        match register(&shortcut::candidates(&current, Platform::Linux)) {
            Ok(in_use) => on_state(Registration::Registered(in_use)),
            Err(error) => eprintln!("aria: could not register the shortcut again: {error}"),
        }
    }
}

/// Tells the hotkey thread to look again (see [`Wake`]).
#[derive(Debug, Clone)]
pub struct HotkeyControl(Sender<Wake>);

impl HotkeyControl {
    /// The settings changed: when the `shortcut` setting did, the old bind goes and the new one comes.
    pub fn settings_changed(&self) {
        let _ = self.0.send(Wake::Settings);
    }
}

/// In the background: register the hotkey `source` names (the `shortcut` setting; a taken one falls back) as soon as
/// Hyprland can take it, then register it again after every config reload (a reload drops runtime binds) and after a
/// settings change that changed it. `on_state` hears which shortcut is in use.
pub fn keep_hotkey_registered(
    exe: PathBuf,
    source: impl Fn() -> String + Send + 'static,
    retry: Duration,
    on_state: impl Fn(Registration) + Send + 'static,
) -> HotkeyControl {
    let (sender, wakes) = mpsc::channel();
    let reloads = sender.clone();
    std::thread::spawn(move || {
        let mut hotkey = Hotkey::new(exe);
        run_hotkey(
            detect_config,
            |candidates| hotkey.register(candidates),
            source,
            retry,
            wakes,
            move || {
                let _ = watch_events(move || {
                    let _ = reloads.send(Wake::Reload);
                });
            },
            on_state,
        );
    });
    HotkeyControl(sender)
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
        {"id":1,"name":"DP-2","width":1920,"height":1080,"x":1536,"y":0,"scale":1.25,"reserved":[0,22,0,0],"focused":true}
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

    fn sc(text: &str) -> Shortcut {
        shortcut::parse(text, Platform::Linux).unwrap()
    }

    fn in_use(text: &str) -> InUse {
        InUse { shortcut: sc(text), taken: Vec::new() }
    }

    fn states_until_settled(
        detect: impl Fn() -> Result<ConfigKind, HyprError>,
        register: impl Fn() -> Result<InUse, HyprError>,
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
            || Ok(in_use("SUPER + A")),
        );
        let registered = Registration::Registered(in_use("SUPER + A"));
        assert_eq!(seen, vec![Registration::WaitingForHyprland, Registration::WaitingForHyprland, registered.clone()]);
        assert_eq!(last, registered);
    }

    #[test]
    fn registration_retries_when_the_bind_is_refused_once() {
        let refused = std::cell::Cell::new(true);
        let (seen, last) = states_until_settled(
            || Ok(ConfigKind::Lua),
            || if refused.replace(false) { Err(HyprError::Eval("not ready".into())) } else { Ok(in_use("SUPER + A")) },
        );
        let registered = Registration::Registered(in_use("SUPER + A"));
        assert_eq!(seen, vec![Registration::WaitingForHyprland, registered.clone()]);
        assert_eq!(last, registered);
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
            registration_lua(Path::new("/home/u/.local/bin/aria"), &sc("SUPER + A"), &[]),
            r#"hl.unbind("SUPER + A"); hl.bind("SUPER + A", hl.dsp.exec_cmd("/home/u/.local/bin/aria toggle"), { description = "Toggle Aria" }); hl.layer_rule({ name = "aria-dock", match = { namespace = "^aria-dock$" }, blur = false, no_anim = true })"#
        );
    }

    #[test]
    fn lua_escapes_quotes_and_backslashes() {
        let lua = registration_lua(Path::new(r#"/tmp/a"b\c/aria"#), &sc("SUPER + A"), &[]);
        assert!(lua.contains(r#"exec_cmd("/tmp/a\"b\\c/aria toggle")"#), "{lua}");
    }

    #[test]
    fn the_bind_lua_for_a_shortcut() {
        let exe = Path::new("/home/ana/.local/bin/aria");
        let lua = registration_lua(exe, &sc("super + shift + a"), &[sc("SUPER + A")]);
        assert!(
            lua.starts_with(
                r#"hl.unbind("SUPER + A"); hl.unbind("SUPER + SHIFT + A"); hl.bind("SUPER + SHIFT + A", hl.dsp.exec_cmd("/home/ana/.local/bin/aria toggle"), { description = "Toggle Aria" }); hl.layer_rule("#
            ),
            "{lua}"
        );
        // The previous one equal to the new one (a reload), or repeated: unbound once.
        let lua =
            registration_lua(exe, &sc("SUPER + A"), &[sc("SUPER + A"), sc("CTRL + ALT + K"), sc("CTRL + ALT + K")]);
        assert!(
            lua.starts_with(r#"hl.unbind("CTRL + ALT + K"); hl.unbind("SUPER + A"); hl.bind("SUPER + A""#),
            "{lua}"
        );
        assert_eq!(hypr_combo(&sc("ctrl + alt + space")), "CTRL + ALT + SPACE");
        assert_eq!(hypr_combo(&sc("super + f5")), "SUPER + F5");
    }

    /// Entries in the shape `hyprctl -j binds` gives with the Lua config (every dispatcher is `__lua`).
    fn binds_fixture() -> Value {
        let bind = |modmask: u64, key: &str, keycode: u64, description: &str, submap: &str| {
            serde_json::json!({
                "locked": false, "mouse": false, "release": false, "repeat": false, "longPress": false,
                "non_consuming": false, "auto_consuming": false, "has_description": !description.is_empty(),
                "modmask": modmask, "submap": submap, "submap_universal": "false", "key": key, "keycode": keycode,
                "catch_all": false, "description": description, "allow_input_capture": false,
                "dispatcher": "__lua", "arg": "12"
            })
        };
        let mut mouse = bind(64, "mouse:272", 0, "Move window", "");
        mouse["mouse"] = Value::Bool(true);
        serde_json::json!([
            bind(64, "A", 0, "Toggle Aria", ""),
            bind(64, "SPACE", 0, "Launch apps", ""),
            bind(65, "A", 0, "Something else", ""),
            bind(12, "k", 0, "", ""),
            bind(64, "", 0, "Switch to workspace 1", ""),
            bind(72, "A", 0, "Resize mode", "resize"),
            mouse,
        ])
    }

    #[test]
    fn taken_reads_hyprctl_binds() {
        let (binds, exe) = (binds_fixture(), Path::new("/home/ana/.local/bin/aria"));
        assert!(taken(&binds, &sc("SUPER + SHIFT + A"), exe), "modmask 65 = SUPER + SHIFT");
        assert!(taken(&binds, &sc("SUPER + Space"), exe), "keys compare without case");
        assert!(taken(&binds, &sc("CTRL + ALT + K"), exe), "modmask 12 = CTRL + ALT");
        assert!(taken(&binds, &sc("SUPER + 1"), exe), "a digit bound by keycode: listed with no key");
        assert!(!taken(&binds, &sc("SUPER + CTRL + 1"), exe), "other modifiers");
        assert!(!taken(&binds, &sc("SUPER + Q"), exe), "an empty key only stands for a digit");
        let coded = serde_json::json!([{"modmask": 8, "key": "", "keycode": 12, "submap": "", "description": "x"}]);
        assert!(taken(&coded, &sc("ALT + 5"), exe), "a keycode in 10..=19, when listed, counts for any digit");
        assert!(!taken(&binds, &sc("SUPER + ALT + A"), exe), "only in a submap");
        assert!(!taken(&binds, &sc("SUPER + B"), exe));
        assert!(!taken(&binds, &sc("SUPER + CTRL + A"), exe), "other modifiers");
        assert!(!taken(&Value::Null, &sc("SUPER + SHIFT + A"), exe), "no answer: nothing is taken");
        assert_eq!(modmask(&sc("SUPER + CTRL + ALT + SHIFT + A")), 64 + 4 + 8 + 1);
    }

    #[test]
    fn aria_s_own_bind_is_not_taken() {
        let (binds, exe) = (binds_fixture(), Path::new("/home/ana/.local/bin/aria"));
        assert!(!taken(&binds, &sc("SUPER + A"), exe), "described as Aria's");
        let classic = serde_json::json!([
            {"modmask": 64, "key": "B", "keycode": 0, "submap": "", "dispatcher": "exec", "arg": "/home/ana/.local/bin/aria toggle", "description": ""},
            {"modmask": 64, "key": "C", "keycode": 0, "submap": "", "dispatcher": "exec", "arg": "/usr/bin/other toggle", "description": ""}
        ]);
        assert!(!taken(&classic, &sc("SUPER + B"), exe), "`exec <exe> toggle` is Aria's");
        assert!(taken(&classic, &sc("SUPER + C"), exe));
        assert_eq!(own_binds(&binds, exe), vec![sc("SUPER + A")]);
    }

    #[test]
    fn the_plan_skips_taken_shortcuts_and_unbinds_the_old_one() {
        let exe = PathBuf::from("/home/ana/.local/bin/aria");
        let binds = serde_json::json!([
            {"modmask": 64, "key": "A", "keycode": 0, "submap": "", "dispatcher": "__lua", "arg": "3", "description": "Omarchy thing"},
            {"modmask": 65, "key": "A", "keycode": 0, "submap": "", "dispatcher": "__lua", "arg": "4", "description": "Another"}
        ]);
        let mut hotkey = Hotkey::new(exe);
        let (lua, used) = hotkey.plan(&binds, &shortcut::candidates("", Platform::Linux));
        assert_eq!(
            used,
            InUse { shortcut: sc("SUPER + ALT + A"), taken: vec![sc("SUPER + A"), sc("SUPER + SHIFT + A")] }
        );
        assert!(
            lua.starts_with(r#"hl.unbind("SUPER + ALT + A"); hl.bind("SUPER + ALT + A""#),
            "someone else's binds stay: {lua}"
        );
        hotkey.bound = Some(sc("SUPER + ALT + A"));
        let (lua, used) = hotkey.plan(&Value::Null, &shortcut::candidates("ctrl + alt + k", Platform::Linux));
        assert_eq!(used, in_use("CTRL + ALT + K"));
        assert!(
            lua.starts_with(r#"hl.unbind("SUPER + ALT + A"); hl.unbind("CTRL + ALT + K"); hl.bind("CTRL + ALT + K""#),
            "no list from Hyprland: the last bind is unbound: {lua}"
        );
        // Every candidate taken: the first is bound anyway, and only the others are reported taken.
        let full = serde_json::json!([
            {"modmask": 64, "key": "A", "keycode": 0, "submap": "", "dispatcher": "__lua", "arg": "3", "description": "Omarchy thing"},
            {"modmask": 65, "key": "A", "keycode": 0, "submap": "", "dispatcher": "__lua", "arg": "4", "description": "Another"},
            {"modmask": 72, "key": "A", "keycode": 0, "submap": "", "dispatcher": "__lua", "arg": "5", "description": "A third"}
        ]);
        let (_, used) = Hotkey::new(PathBuf::from("/home/ana/.local/bin/aria"))
            .plan(&full, &shortcut::candidates("", Platform::Linux));
        assert_eq!(
            used,
            InUse { shortcut: sc("SUPER + A"), taken: vec![sc("SUPER + SHIFT + A"), sc("SUPER + ALT + A")] }
        );
        // After a reload Aria's bind is gone and the person's config now binds SUPER + A: it stays.
        hotkey.bound = Some(sc("SUPER + A"));
        let (lua, used) = hotkey.plan(&binds, &shortcut::candidates("", Platform::Linux));
        assert_eq!(used.shortcut, sc("SUPER + ALT + A"));
        assert!(!lua.contains(r#"hl.unbind("SUPER + A")"#), "a foreign SUPER + A is never unbound: {lua}");
        let own =
            serde_json::json!([{"modmask": 64, "key": "A", "keycode": 0, "submap": "", "description": "Toggle Aria"}]);
        let (lua, _) = hotkey.plan(&own, &shortcut::candidates("super + shift + a", Platform::Linux));
        assert!(
            lua.starts_with(r#"hl.unbind("SUPER + A"); hl.unbind("SUPER + SHIFT + A");"#),
            "listed as Aria's: {lua}"
        );
    }

    #[test]
    fn binds_are_asked_on_the_request_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".socket.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let answer = binds_fixture().to_string();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 64];
            let n = stream.read(&mut request).unwrap();
            assert_eq!(&request[..n], b"j/binds");
            stream.write_all(answer.as_bytes()).unwrap();
        });
        assert_eq!(binds_at(&path), Some(binds_fixture()));
        server.join().unwrap();
    }

    #[test]
    fn a_settings_change_rebinds() {
        use std::sync::Mutex;
        let setting = Arc::new(Mutex::new(String::new()));
        let bound = Arc::new(Mutex::new(Vec::<String>::new()));
        let states = Arc::new(Mutex::new(Vec::<Registration>::new()));
        let watching = Arc::new(AtomicUsize::new(0));
        let (wake, wakes) = mpsc::channel();
        let worker = {
            let (setting, bound, states, watching) = (setting.clone(), bound.clone(), states.clone(), watching.clone());
            std::thread::spawn(move || {
                run_hotkey(
                    || Ok(ConfigKind::Lua),
                    // A fake Hyprland where SUPER + SHIFT + A is taken.
                    |candidates| {
                        let (used, taken) = shortcut::first_free(candidates, |s| s.label() == "SUPER + SHIFT + A");
                        bound.lock().unwrap().push(used.label());
                        Ok(InUse { shortcut: used, taken })
                    },
                    move || setting.lock().unwrap().clone(),
                    Duration::from_millis(1),
                    wakes,
                    move || {
                        watching.fetch_add(1, Ordering::SeqCst);
                    },
                    move |state| states.lock().unwrap().push(state),
                )
            })
        };
        let wait_for = |count: usize| {
            let deadline = Instant::now() + Duration::from_secs(5);
            while states.lock().unwrap().len() < count && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        wait_for(1);
        assert_eq!(*bound.lock().unwrap(), ["SUPER + A"], "empty: the default");
        assert_eq!(watching.load(Ordering::SeqCst), 1);

        wake.send(Wake::Settings).unwrap(); // something else in the settings changed
        *setting.lock().unwrap() = "super + shift + a".into();
        wake.send(Wake::Settings).unwrap();
        wait_for(2);
        wake.send(Wake::Reload).unwrap(); // a reload binds again even with nothing changed
        drop(wake);
        worker.join().unwrap();
        assert_eq!(
            *bound.lock().unwrap(),
            ["SUPER + A", "SUPER + A", "SUPER + A"],
            "the chosen one is taken: the default"
        );
        let states = states.lock().unwrap();
        assert_eq!(
            states[1],
            Registration::Registered(InUse { shortcut: sc("SUPER + A"), taken: vec![sc("SUPER + SHIFT + A")] }),
            "the page hears which one is in use, and which was taken"
        );
        assert_eq!(states.len(), 3);
        assert_eq!(watching.load(Ordering::SeqCst), 1, "the reload watcher starts once");
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
        assert_eq!(
            parse_focused(MONITORS),
            Some(Monitor { name: "DP-2".into(), x: 1536, y: 0, usable_height: Some(842) }),
            "1080 physical at 1.25 is 864 logical, less the 22 px bar above"
        );
        let plain =
            parse_focused(r#"[{"name":"DP-1","width":1920,"height":1080,"x":0,"y":0,"scale":1,"focused":true}]"#);
        assert_eq!(plain.and_then(|m| m.usable_height), Some(1080), "nothing reserved, scale 1");
        let no_height = parse_focused(r#"[{"name":"DP-1","x":0,"y":0,"focused":true}]"#);
        assert_eq!(no_height.map(|m| m.usable_height), Some(None), "no height: the compositor decides");
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
        assert_eq!(
            focused_monitor_at(&path),
            Some(Monitor { name: "DP-2".into(), x: 1536, y: 0, usable_height: Some(842) })
        );
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
