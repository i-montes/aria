//! Aria's face in the Omarchy bar: `aria bar install` puts the widget (install/omarchy/aria.face, embedded in the
//! binary) in the Omarchy shell's plugins and places it next to the weather; `aria bar remove` takes it out. The
//! installer runs both. Where there is no Omarchy shell (macOS, another Linux desktop) they do nothing.

use serde_json::Value;
use std::path::{Path, PathBuf};

pub const WIDGET_ID: &str = "aria.face";
const FILES: [(&str, &str); 2] = [
    (
        "manifest.json",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../install/omarchy/aria.face/manifest.json")),
    ),
    ("Face.qml", include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../install/omarchy/aria.face/Face.qml"))),
];

/// What `aria bar install` did.
#[derive(Debug, PartialEq, Eq)]
pub enum Installed {
    /// No Omarchy shell here: nothing was written.
    NoOmarchy,
    /// The widget's files were written or were already current; `placed` says whether it was added to the bar now
    /// (false: it already was).
    Done { changed: bool, placed: bool },
}

/// The Omarchy shell's config directory, `~/.config/omarchy` (or under `$XDG_CONFIG_HOME`).
pub fn omarchy_dir() -> PathBuf {
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute())
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".config"));
    config.join("omarchy")
}

/// The Omarchy shell is there when its config directory exists and its `omarchy` command is on the PATH.
pub fn has_omarchy(dir: &Path) -> bool {
    dir.is_dir()
        && std::env::var_os("PATH")
            .map(|path| std::env::split_paths(&path).any(|bin| bin.join("omarchy").is_file()))
            .unwrap_or(false)
}

/// Whether the bar layout in `shell.json` already holds the widget, in any section.
pub fn in_layout(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("shell.json")) else { return false };
    let Ok(config) = serde_json::from_str::<Value>(&text) else { return false };
    let Some(layout) = config.pointer("/bar/layout").and_then(Value::as_object) else { return false };
    layout
        .values()
        .filter_map(Value::as_array)
        .flatten()
        .any(|entry| entry.get("id").and_then(Value::as_str) == Some(WIDGET_ID))
}

fn center_has(dir: &Path, id: &str) -> bool {
    std::fs::read_to_string(dir.join("shell.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|config| config.pointer("/bar/layout/center").and_then(Value::as_array).cloned())
        .is_some_and(|center| center.iter().any(|entry| entry.get("id").and_then(Value::as_str) == Some(id)))
}

/// Writes the widget's files (only the ones that changed, so a running bar reloads only when there is something new)
/// and, when the bar does not show it yet, backs up `shell.json` the way Omarchy does (`shell.json.bak.<epoch>`) and
/// places it with Omarchy's own command, after the shell has scanned it in: after the weather, else at the end of the
/// centre. `run` runs a command and says whether it succeeded.
pub fn install(
    dir: &Path,
    has_omarchy: bool,
    now: u64,
    run: &mut dyn FnMut(&[&str]) -> bool,
) -> Result<Installed, String> {
    if !has_omarchy {
        return Ok(Installed::NoOmarchy);
    }
    let plugin = dir.join("plugins").join(WIDGET_ID);
    std::fs::create_dir_all(&plugin).map_err(|error| format!("could not create {}: {error}", plugin.display()))?;
    let mut changed = false;
    for (name, text) in FILES {
        let path = plugin.join(name);
        if std::fs::read_to_string(&path).ok().as_deref() != Some(text) {
            std::fs::write(&path, text).map_err(|error| format!("could not write {}: {error}", path.display()))?;
            changed = true;
        }
    }
    if in_layout(dir) {
        return Ok(Installed::Done { changed, placed: false });
    }
    let shell = dir.join("shell.json");
    if shell.is_file() {
        let backup = dir.join(format!("shell.json.bak.{now}"));
        std::fs::copy(&shell, &backup).map_err(|error| format!("could not back up {}: {error}", shell.display()))?;
    }
    // A running shell only knows the plugins it has scanned; a new one has to be scanned in before Omarchy enables it
    // (best effort: with no shell running there is nothing to tell).
    run(&["omarchy-shell", "-q", "shell", "rescanPlugins"]);
    let placed = if center_has(dir, "omarchy.weather") {
        run(&["omarchy", "plugin", "enable", WIDGET_ID, "--after", "omarchy.weather"])
    } else {
        run(&["omarchy", "plugin", "enable", WIDGET_ID, "--section", "center"])
    };
    if !placed {
        return Err("Omarchy did not put the widget in the bar (`omarchy plugin enable aria.face` failed)".into());
    }
    Ok(Installed::Done { changed, placed: true })
}

/// Takes the widget out of the bar (Omarchy's own command, only when the bar holds it) and deletes its files.
/// Returns false when there was nothing to remove.
pub fn remove(dir: &Path, has_omarchy: bool, run: &mut dyn FnMut(&[&str]) -> bool) -> Result<bool, String> {
    let plugin = dir.join("plugins").join(WIDGET_ID);
    let mut removed = false;
    if has_omarchy && in_layout(dir) {
        if !run(&["omarchy", "plugin", "disable", WIDGET_ID]) {
            return Err(
                "Omarchy did not take the widget out of the bar (`omarchy plugin disable aria.face` failed)".into()
            );
        }
        removed = true;
    }
    if plugin.is_dir() {
        std::fs::remove_dir_all(&plugin).map_err(|error| format!("could not delete {}: {error}", plugin.display()))?;
        removed = true;
    }
    Ok(removed)
}

fn run_command(argv: &[&str]) -> bool {
    std::process::Command::new(argv[0])
        .args(&argv[1..])
        .stdout(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// `aria bar install|remove`; returns the exit code.
pub fn bar_command(action: Option<&str>) -> i32 {
    let dir = omarchy_dir();
    let has = has_omarchy(&dir);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let result = match action {
        Some("install") => install(&dir, has, now, &mut run_command).map(|done| match done {
            Installed::NoOmarchy => "No Omarchy shell here; Aria's face in the bar was skipped".to_string(),
            Installed::Done { placed: true, .. } => {
                "Aria's face is in the Omarchy bar, next to the weather".to_string()
            }
            Installed::Done { changed: true, .. } => "Aria's face in the Omarchy bar is updated".to_string(),
            Installed::Done { .. } => "Aria's face in the Omarchy bar is up to date".to_string(),
        }),
        Some("remove") => remove(&dir, has, &mut run_command).map(|removed| {
            if removed { "Aria's face is out of the Omarchy bar" } else { "Aria's face was not in the Omarchy bar" }
                .to_string()
        }),
        _ => {
            eprintln!("usage: aria bar [install|remove]");
            return 2;
        }
    };
    match result {
        Ok(message) => {
            println!("{message}");
            0
        }
        Err(error) => {
            eprintln!("aria: {error}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell_json(center: &[&str]) -> String {
        let center: Vec<Value> = center.iter().map(|id| serde_json::json!({ "id": id })).collect();
        serde_json::json!({ "version": 1, "bar": { "layout": { "left": [], "center": center, "right": [{ "id": "omarchy.tray" }] } } })
            .to_string()
    }

    /// A fake `omarchy plugin enable|disable` that edits shell.json the way the real one does, and records its calls.
    fn fake_omarchy<'a>(dir: &Path, calls: &'a mut Vec<String>) -> impl FnMut(&[&str]) -> bool + 'a {
        let dir = dir.to_path_buf();
        let mut scanned = false;
        move |argv: &[&str]| {
            calls.push(argv.join(" "));
            if argv[0] == "omarchy-shell" {
                scanned = true;
                return true;
            }
            if argv[2] == "enable" && !scanned {
                return false; // the real one: "plugin 'aria.face' is not known; run: omarchy-shell shell rescanPlugins"
            }
            let path = dir.join("shell.json");
            let mut config: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let center = config.pointer_mut("/bar/layout/center").unwrap().as_array_mut().unwrap();
            match argv[2] {
                "enable" => center.push(serde_json::json!({ "id": WIDGET_ID })),
                _ => center.retain(|entry| entry["id"] != WIDGET_ID),
            }
            std::fs::write(&path, config.to_string()).unwrap();
            true
        }
    }

    #[test]
    fn without_omarchy_nothing_is_written() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let mut calls = Vec::new();
        assert_eq!(install(dir, false, 1, &mut fake_omarchy(dir, &mut calls)).unwrap(), Installed::NoOmarchy);
        assert!(!dir.join("plugins").exists());
        assert!(calls.is_empty());
    }

    #[test]
    fn install_writes_the_widget_backs_up_and_places_it_after_the_weather() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("shell.json"), shell_json(&["omarchy.clock", "omarchy.weather"])).unwrap();
        let mut calls = Vec::new();
        let done = install(dir, true, 42, &mut fake_omarchy(dir, &mut calls)).unwrap();
        assert_eq!(done, Installed::Done { changed: true, placed: true });
        for (name, text) in FILES {
            assert_eq!(std::fs::read_to_string(dir.join("plugins/aria.face").join(name)).unwrap(), text);
        }
        assert_eq!(
            calls,
            ["omarchy-shell -q shell rescanPlugins", "omarchy plugin enable aria.face --after omarchy.weather"]
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("shell.json.bak.42")).unwrap(),
            shell_json(&["omarchy.clock", "omarchy.weather"])
        );
        assert!(in_layout(dir));
    }

    #[test]
    fn without_the_weather_it_goes_at_the_end_of_the_centre() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("shell.json"), shell_json(&["omarchy.clock"])).unwrap();
        let mut calls = Vec::new();
        install(dir, true, 1, &mut fake_omarchy(dir, &mut calls)).unwrap();
        assert_eq!(calls, ["omarchy-shell -q shell rescanPlugins", "omarchy plugin enable aria.face --section center"]);
    }

    #[test]
    fn a_second_install_changes_nothing_and_does_not_place_it_twice() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("shell.json"), shell_json(&["omarchy.weather"])).unwrap();
        let mut calls = Vec::new();
        install(dir, true, 1, &mut fake_omarchy(dir, &mut calls)).unwrap();
        let again = install(dir, true, 2, &mut fake_omarchy(dir, &mut calls)).unwrap();
        assert_eq!(again, Installed::Done { changed: false, placed: false });
        assert_eq!(calls.len(), 2, "one rescan and one enable, from the first install only");
        assert!(!dir.join("shell.json.bak.2").exists());
    }

    #[test]
    fn an_old_widget_is_replaced_and_its_place_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("shell.json"), shell_json(&["omarchy.weather", WIDGET_ID])).unwrap();
        std::fs::create_dir_all(dir.join("plugins/aria.face")).unwrap();
        std::fs::write(dir.join("plugins/aria.face/Face.qml"), "old").unwrap();
        let mut calls = Vec::new();
        let done = install(dir, true, 1, &mut fake_omarchy(dir, &mut calls)).unwrap();
        assert_eq!(done, Installed::Done { changed: true, placed: false });
        assert_ne!(std::fs::read_to_string(dir.join("plugins/aria.face/Face.qml")).unwrap(), "old");
        assert!(calls.is_empty());
    }

    #[test]
    fn a_failed_enable_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("shell.json"), shell_json(&["omarchy.weather"])).unwrap();
        assert!(install(dir, true, 1, &mut |_: &[&str]| false).is_err());
    }

    #[test]
    fn remove_takes_it_out_of_the_bar_and_deletes_it() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("shell.json"), shell_json(&["omarchy.weather"])).unwrap();
        let mut calls = Vec::new();
        install(dir, true, 1, &mut fake_omarchy(dir, &mut calls)).unwrap();
        assert!(remove(dir, true, &mut fake_omarchy(dir, &mut calls)).unwrap());
        assert_eq!(calls.last().unwrap(), "omarchy plugin disable aria.face");
        assert!(!in_layout(dir));
        assert!(!dir.join("plugins/aria.face").exists());
        assert!(!remove(dir, true, &mut fake_omarchy(dir, &mut calls)).unwrap(), "a second remove has nothing to do");
    }

    #[test]
    fn the_embedded_widget_is_the_aria_face_plugin() {
        let manifest: Value = serde_json::from_str(FILES[0].1).unwrap();
        assert_eq!(manifest["id"], WIDGET_ID);
        assert_eq!(manifest["entryPoints"]["barWidget"], "Face.qml");
        assert!(FILES[1].1.contains("moduleName: \"aria.face\""));
    }
}
