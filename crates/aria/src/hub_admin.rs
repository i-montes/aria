//! What sets a hub up (docs/superpowers/specs/2026-10-23-hub-and-machines-design.md, "First setup without a page"):
//! the bundle of settings and credentials the owner's computer sends, and the import that writes it on the hub.
//!
//! The pieces are small and pure; the ssh orchestration lives with `aria hub init`.

// `aria hub init` is the caller of the export and of the ssh checks; until it lands only `hub-import` is wired up.
#![allow(dead_code)]

use crate::keyring_cli::secret_value;
use aria_core::json_file::{merge_preserving, read_json, write_bytes_atomic};
use aria_core::paths::AriaPaths;
use aria_core::vault::{self, Vault, SEALED};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::io::BufRead;
use std::net::IpAddr;
use std::path::Path;

/// The only files a bundle carries, in the order they are written.
pub const BUNDLE_FILES: [&str; 4] = ["settings.json", "auth.json", "mcp.json", "mcp-auth.json"];

/// The most one line of a bundle may hold.
pub const MAX_BUNDLE_BYTES: u64 = 16 * 1024 * 1024;

/// Settings and credentials as plain files: `{"files":{"settings.json": "<content>", …}}` on one line.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub files: BTreeMap<String, String>,
}

/// One address a computer listens on, from `ss -ltnH`.
#[derive(Debug, Clone, PartialEq)]
pub struct Listener {
    pub addr: String,
    pub port: u16,
}

/// The bundle the owner's computer sends to a hub: the same settings and credentials Pi reads, as plain files.
///
/// The hub runs with the keyring off, so a key kept in the owner's keyring (a `!aria secret …` command in `auth.json` or
/// `mcp.json`, `mcp-auth.json` sealed whole) is read out and written as its value. Nothing is written or changed here:
/// the keyring and the live copies are only read, and a secret that cannot be read fails the export whole.
pub fn export_bundle(paths: &AriaPaths, hub_home: &str, projects: Option<&str>) -> Result<Bundle, String> {
    export_with(paths, hub_home, projects, &|| {
        crate::keyring_cli::vault_for(&paths.root).map_err(|error| error.to_string())
    })
}

/// `export_bundle` with the way to the keyring given, opened only when a file needs it (a home whose files are plain
/// never asks the keyring anything, and a locked one is not asked for nothing).
pub fn export_with(
    paths: &AriaPaths,
    hub_home: &str,
    projects: Option<&str>,
    open_vault: &dyn Fn() -> Result<Vault, String>,
) -> Result<Bundle, String> {
    let mut files = BTreeMap::new();

    let settings_path = paths.settings();
    let mut settings = match read_json(&settings_path).map_err(|error| error.to_string())? {
        Some(settings @ Value::Object(_)) => settings,
        Some(_) => return Err(format!("{}: expected a JSON object", settings_path.display())),
        None => return Err(format!("{}: there are no settings to send", settings_path.display())),
    };
    let keys = Keys {
        paths,
        open: open_vault,
        opened: OnceCell::new(),
        keyring_on: settings["credentials"]["keyring"] != json!(false),
    };
    let projects_dir =
        projects.map(str::to_string).unwrap_or_else(|| format!("{}/Projects", hub_home.trim_end_matches('/')));
    settings["credentials"] = merge_preserving(settings["credentials"].take(), json!({"keyring": false}));
    settings["sandbox"] = merge_preserving(settings["sandbox"].take(), json!({"projectsDir": projects_dir}));
    files.insert("settings.json".to_string(), render(&settings));

    for name in &BUNDLE_FILES[1..] {
        let sealed = SEALED.contains(name);
        let found = if sealed { keys.sealed(name)? } else { read_object(&paths.root.join(name))? };
        let Some(mut file) = found else { continue };
        if *name != "mcp-auth.json" {
            keys.plain(&mut file).map_err(|why| format!("{name}: {why}"))?;
        }
        files.insert(name.to_string(), render(&file));
    }
    Ok(Bundle { files })
}

fn render(value: &Value) -> String {
    let mut text = serde_json::to_string_pretty(value).unwrap_or_default();
    text.push('\n');
    text
}

/// The JSON object in `path`, or None when there is no file.
fn read_object(path: &Path) -> Result<Option<Value>, String> {
    match read_json(path).map_err(|error| error.to_string())? {
        Some(file @ Value::Object(_)) => Ok(Some(file)),
        Some(_) => Err(format!("{}: expected a JSON object", path.display())),
        None => Ok(None),
    }
}

/// Where the owner's keys are read from, without changing any of them.
struct Keys<'a> {
    paths: &'a AriaPaths,
    open: &'a dyn Fn() -> Result<Vault, String>,
    opened: OnceCell<Result<Vault, String>>,
    /// The owner's `credentials.keyring`: off, the sealed files are only looked for in plain files.
    keyring_on: bool,
}

impl Keys<'_> {
    /// The vault the paths hold (the dock is running), else the system keyring, opened the first time it is needed.
    fn vault(&self) -> Result<&Vault, String> {
        if let Some(vault) = &self.paths.vault {
            return Ok(vault);
        }
        self.opened.get_or_init(self.open).as_ref().map_err(Clone::clone)
    }

    /// A file Pi rewrites itself (`auth.json`, `mcp-auth.json`): the dock's live copy when it is running (the newest),
    /// else the plain file in the home, else the copy sealed in the keyring (not asked when the owner has the keyring
    /// off). None when there is none anywhere, or when the keyring cannot be reached and nothing else holds one (a
    /// computer that never had a keyring).
    fn sealed(&self, name: &str) -> Result<Option<Value>, String> {
        let live = match &self.paths.vault {
            Some(vault) => vault.live.clone(),
            None => vault::live_dir(&self.paths.root),
        };
        for path in [live.join(name), self.paths.root.join(name)] {
            if let Some(file) = read_object(&path)? {
                return Ok(Some(file));
            }
        }
        if !self.keyring_on {
            return Ok(None);
        }
        let Ok(vault) = self.vault() else { return Ok(None) };
        let account = format!("file:{name}");
        match vault.keyring.get(&account).map_err(|error| format!("{name}: {error}"))? {
            Some(text) => match serde_json::from_str(&text) {
                Ok(file @ Value::Object(_)) => Ok(Some(file)),
                _ => Err(format!("{name}: the copy in the keyring is not a JSON object")),
            },
            None => Ok(None),
        }
    }

    /// Every string in `file` that is one of Aria's own commands, replaced by the value it reads. Another command, a
    /// `${VAR}` reference or a literal stays as it is: Pi on the hub reads those as it would here.
    fn plain(&self, file: &mut Value) -> Result<(), String> {
        match file {
            Value::String(text) => {
                if let Some(account) = vault::own_account(text) {
                    let vault = self.vault().map_err(|why| format!("{account} cannot be read: {why}"))?;
                    *text = secret_value(vault.keyring.as_ref(), &account)?;
                }
                Ok(())
            }
            Value::Array(items) => items.iter_mut().try_for_each(|item| self.plain(item)),
            Value::Object(map) => map.values_mut().try_for_each(|item| self.plain(item)),
            _ => Ok(()),
        }
    }
}

/// What `hub-import` runs on the hub with the line it was given: the files of the bundle written into `~/.aria` (mode
/// 600, each atomically, the one it replaces kept as `<name>.bak`), the names in [`BUNDLE_FILES`] order. A bundle that is
/// not the expected shape, holds a name that is not one of the four or a file that is not a JSON object writes nothing.
pub fn import(paths: &AriaPaths, bundle: &str) -> Result<Vec<String>, String> {
    let bundle: Bundle = serde_json::from_str(bundle).map_err(|error| format!("the bundle is not one: {error}"))?;
    if let Some(name) = bundle.files.keys().find(|name| !BUNDLE_FILES.contains(&name.as_str())) {
        return Err(format!("{name:?} is not part of a bundle (only {})", BUNDLE_FILES.join(", ")));
    }
    for (name, content) in &bundle.files {
        match serde_json::from_str::<Value>(content) {
            Ok(Value::Object(_)) => {}
            Ok(_) => return Err(format!("{name} is not a JSON object")),
            Err(error) => return Err(format!("{name} is not JSON: {error}")),
        }
    }
    if bundle.files.is_empty() {
        return Err("the bundle has no files".into());
    }
    paths.ensure_root().map_err(|error| format!("{}: {error}", paths.root.display()))?;
    let mut written = Vec::new();
    for name in BUNDLE_FILES {
        let Some(content) = bundle.files.get(name) else { continue };
        let path = paths.root.join(name);
        let failed = |error: std::io::Error| format!("{}: {error}", path.display());
        match std::fs::read(&path) {
            Ok(old) => write_bytes_atomic(&paths.root.join(format!("{name}.bak")), &old, 0o600).map_err(failed)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(failed(error)),
        }
        write_bytes_atomic(&path, content.as_bytes(), 0o600).map_err(failed)?;
        written.push(name.to_string());
    }
    Ok(written)
}

/// The one line of a bundle from `reader`, without its newline: at most `max` bytes, and not empty.
pub fn read_bundle_line(reader: impl BufRead, max: u64) -> Result<String, String> {
    let mut line = String::new();
    reader.take(max + 1).read_line(&mut line).map_err(|error| format!("could not read the bundle: {error}"))?;
    let line = line.trim_end_matches(['\n', '\r']);
    if line.len() as u64 > max {
        return Err(format!("the bundle is too large (more than {max} bytes)"));
    }
    if line.trim().is_empty() {
        return Err("no bundle on stdin".into());
    }
    Ok(line.to_string())
}

/// The addresses `ss -ltnH` lists: `LISTEN 0 128 0.0.0.0:2222 0.0.0.0:*`. An interface (`100.64.0.9%tailscale0`) is
/// dropped from the address. A line that is not one is skipped.
pub fn parse_listeners(ss_output: &str) -> Vec<Listener> {
    ss_output
        .lines()
        .filter_map(|line| {
            let local = line.split_whitespace().nth(3)?;
            let (addr, port) = local.rsplit_once(':')?;
            let addr = match addr.split_once('%') {
                Some((ip, _)) if addr.ends_with(']') => format!("{ip}]"),
                Some((ip, _)) => ip.to_string(),
                None => addr.to_string(),
            };
            (!addr.is_empty()).then_some(Listener { addr, port: port.parse().ok()? })
        })
        .collect()
}

/// Whether sshd on `port` can be reached from beyond the computer and its private networks: something listens on that port
/// on every address, or on one that is not private. An address that is not one counts as public (it is a warning).
pub fn public_ssh(listeners: &[Listener], port: u16) -> bool {
    listeners.iter().filter(|listener| listener.port == port).any(|listener| {
        let addr = listener.addr.trim_start_matches('[').trim_end_matches(']');
        match addr {
            "*" | "0.0.0.0" | "::" => true,
            _ => addr.parse::<IpAddr>().map_or(true, |ip| !private_address(ip)),
        }
    })
}

/// An address that is the computer's own or its private network's: loopback, RFC 1918, link-local, Tailscale's
/// `100.64.0.0/10` and unique local `fc00::/7` (which holds Tailscale's `fd7a:115c:a1e0::/48`).
pub fn private_address(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_loopback()
                || ip.is_private()
                || ip.is_link_local()
                || (ip.octets()[0] == 100 && ip.octets()[1] & 0xc0 == 64)
        }
        IpAddr::V6(ip) => match ip.to_ipv4_mapped() {
            Some(v4) => private_address(IpAddr::V4(v4)),
            None => ip.is_loopback() || ip.segments()[0] & 0xfe00 == 0xfc00 || ip.segments()[0] & 0xffc0 == 0xfe80,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::keyring::{Backend, MemoryBackend};
    use aria_core::vault::{command_for, Vault};
    use serde_json::{json, Value};
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    fn mode(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    fn write(path: &Path, value: &Value) {
        std::fs::write(path, serde_json::to_string_pretty(value).unwrap()).unwrap();
    }

    fn json_of(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    fn owner_settings() -> Value {
        json!({
            "language": "es",
            "shortcut": "SUPER + K",
            "provider": {"default": "opencode-go"},
            "sandbox": {"projectsDir": "/home/ana/Projects", "extraWritable": ["/home/ana/notes"], "askToLeave": true},
            "credentials": {"keyring": true},
            "somethingNew": {"kept": 1}
        })
    }

    /// An owner's home with plain credential files.
    fn plain_home(dir: &Path) -> AriaPaths {
        let root = dir.join("home");
        std::fs::create_dir_all(&root).unwrap();
        write(&root.join("settings.json"), &owner_settings());
        write(&root.join("auth.json"), &json!({"opencode-go": {"type": "api_key", "key": "sk-test-1234"}}));
        write(
            &root.join("mcp.json"),
            &json!({"mcpServers": {"tavily": {"url": "https://t.example", "headers": {"Authorization": "Bearer tvly-test"}}}}),
        );
        write(&root.join("mcp-auth.json"), &json!({"mcp__notion|u": {"tokens": {"access_token": "n-test"}}}));
        // Not in the bundle: sessions and the like stay on the owner's computer.
        write(&root.join("sessions.json"), &json!({"tabs": []}));
        std::fs::write(root.join("notes.txt"), "x").unwrap();
        AriaPaths::new(root)
    }

    fn vault(dir: &Path, keyring: Arc<MemoryBackend>) -> Vault {
        Vault { keyring, live: dir.join("live"), exe: PathBuf::from("/bin/aria") }
    }

    /// An owner whose keys are in the keyring: the files hold `!aria secret …` commands, `mcp-auth.json` is sealed.
    fn keyring_home(dir: &Path) -> (AriaPaths, Arc<MemoryBackend>) {
        let paths = plain_home(dir);
        let keyring = Arc::new(MemoryBackend::default());
        let exe = Path::new("/bin/aria");
        keyring.set("key:opencode-go", "sk-test-1234").unwrap();
        keyring.set("mcp:tavily:header:Authorization", "Bearer tvly-test").unwrap();
        keyring.set("file:mcp-auth.json", r#"{"mcp__notion|u":{"tokens":{"access_token":"n-test"}}}"#).unwrap();
        write(
            &paths.root.join("auth.json"),
            &json!({"opencode-go": {"type": "api_key", "key": command_for(exe, &paths.root, "key:opencode-go")}}),
        );
        write(
            &paths.root.join("mcp.json"),
            &json!({"mcpServers": {"tavily": {"url": "https://t.example", "headers": {
                "Authorization": command_for(exe, &paths.root, "mcp:tavily:header:Authorization")}}}}),
        );
        std::fs::remove_file(paths.root.join("mcp-auth.json")).unwrap();
        (paths, keyring)
    }

    fn export_plain(paths: &AriaPaths, projects: Option<&str>) -> Result<Bundle, String> {
        export_with(paths, "/home/hub", projects, &|| Err("no keyring here".into()))
    }

    fn files_in(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .map(|entries| {
                entries.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    fn hub(dir: &Path) -> AriaPaths {
        AriaPaths::new(dir.join("hub-home").join(".aria"))
    }

    fn line(files: &[(&str, &str)]) -> String {
        serde_json::to_string(&json!({"files": files.iter().cloned().collect::<BTreeMap<_, _>>()})).unwrap()
    }

    #[test]
    fn export_sets_the_hubs_projects_folder_and_turns_the_keyring_off() {
        let dir = tempfile::tempdir().unwrap();
        let paths = plain_home(dir.path());
        let settings = json_of(&export_plain(&paths, None).unwrap().files["settings.json"]);
        assert_eq!(settings["sandbox"]["projectsDir"], "/home/hub/Projects");
        assert_eq!(settings["credentials"]["keyring"], false);
        // Everything else is as the owner has it, including what this version does not know.
        assert_eq!(settings["language"], "es");
        assert_eq!(settings["shortcut"], "SUPER + K");
        assert_eq!(settings["provider"], json!({"default": "opencode-go"}));
        assert_eq!(settings["sandbox"]["extraWritable"], json!(["/home/ana/notes"]));
        assert_eq!(settings["somethingNew"], json!({"kept": 1}));
        let chosen = json_of(&export_plain(&paths, Some("/srv/work")).unwrap().files["settings.json"]);
        assert_eq!(chosen["sandbox"]["projectsDir"], "/srv/work");
        // The owner's own file is not touched.
        assert_eq!(json_of(&std::fs::read_to_string(paths.settings()).unwrap()), owner_settings());
    }

    #[test]
    fn export_carries_only_the_four_files() {
        let dir = tempfile::tempdir().unwrap();
        let paths = plain_home(dir.path());
        let bundle = export_plain(&paths, None).unwrap();
        assert_eq!(bundle.files.keys().map(String::as_str).collect::<Vec<_>>(), {
            let mut names = BUNDLE_FILES.to_vec();
            names.sort();
            names
        });
        assert_eq!(json_of(&bundle.files["auth.json"])["opencode-go"]["key"], "sk-test-1234");
        std::fs::remove_file(paths.root.join("mcp.json")).unwrap();
        std::fs::remove_file(paths.root.join("mcp-auth.json")).unwrap();
        let bundle = export_plain(&paths, None).unwrap();
        assert!(!bundle.files.contains_key("mcp.json") && !bundle.files.contains_key("mcp-auth.json"));
        assert!(bundle.files.contains_key("auth.json"));
    }

    #[test]
    fn export_never_asks_the_keyring_of_an_owner_who_has_it_off() {
        let dir = tempfile::tempdir().unwrap();
        let paths = plain_home(dir.path());
        let mut settings = owner_settings();
        settings["credentials"]["keyring"] = json!(false);
        write(&paths.settings(), &settings);
        std::fs::remove_file(paths.root.join("mcp-auth.json")).unwrap();
        let bundle = export_with(&paths, "/home/hub", None, &|| panic!("the keyring is off")).unwrap();
        assert!(!bundle.files.contains_key("mcp-auth.json"));
        assert_eq!(json_of(&bundle.files["auth.json"])["opencode-go"]["key"], "sk-test-1234");
    }

    #[test]
    fn export_needs_the_settings_file() {
        let dir = tempfile::tempdir().unwrap();
        let paths = plain_home(dir.path());
        std::fs::remove_file(paths.settings()).unwrap();
        let error = export_plain(&paths, None).unwrap_err();
        assert!(error.contains("settings.json"), "{error}");
    }

    #[test]
    fn export_resolves_what_the_keyring_holds_into_plain_values() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, keyring) = keyring_home(dir.path());
        let shown = vault(dir.path(), keyring.clone());
        let bundle = export_with(&paths, "/home/hub", None, &|| Ok(shown.clone())).unwrap();
        assert_eq!(
            json_of(&bundle.files["auth.json"]),
            json!({"opencode-go": {"type": "api_key", "key": "sk-test-1234"}})
        );
        assert_eq!(
            json_of(&bundle.files["mcp.json"])["mcpServers"]["tavily"]["headers"]["Authorization"],
            "Bearer tvly-test"
        );
        assert_eq!(
            json_of(&bundle.files["mcp-auth.json"]),
            json!({"mcp__notion|u": {"tokens": {"access_token": "n-test"}}})
        );
        assert!(!bundle.files.values().any(|text| text.contains("secret --home")), "no command is left");
    }

    #[test]
    fn export_changes_nothing_on_the_owners_computer() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, keyring) = keyring_home(dir.path());
        let shown = vault(dir.path(), keyring.clone());
        let before_files: Vec<(String, Vec<u8>)> = files_in(&paths.root)
            .into_iter()
            .map(|n| (n.clone(), std::fs::read(paths.root.join(&n)).unwrap()))
            .collect();
        let accounts = ["key:opencode-go", "mcp:tavily:header:Authorization", "file:mcp-auth.json"];
        let before_keys: Vec<_> = accounts.iter().map(|a| keyring.get(a).unwrap()).collect();
        export_with(&paths, "/home/hub", None, &|| Ok(shown.clone())).unwrap();
        let after_files: Vec<(String, Vec<u8>)> = files_in(&paths.root)
            .into_iter()
            .map(|n| (n.clone(), std::fs::read(paths.root.join(&n)).unwrap()))
            .collect();
        assert_eq!(before_files, after_files);
        assert_eq!(before_keys, accounts.iter().map(|a| keyring.get(a).unwrap()).collect::<Vec<_>>());
        assert!(!dir.path().join("live").exists(), "the live folder is never made");
    }

    #[test]
    fn export_prefers_the_live_copy_the_running_dock_keeps() {
        let dir = tempfile::tempdir().unwrap();
        let (mut paths, keyring) = keyring_home(dir.path());
        let shown = vault(dir.path(), keyring);
        std::fs::create_dir_all(&shown.live).unwrap();
        write(&shown.live.join("mcp-auth.json"), &json!({"fresh": true}));
        paths.vault = Some(shown);
        let bundle = export_with(&paths, "/home/hub", None, &|| panic!("the paths hold the vault")).unwrap();
        assert_eq!(json_of(&bundle.files["mcp-auth.json"]), json!({"fresh": true}));
    }

    #[test]
    fn export_fails_whole_when_a_secret_cannot_be_read() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, keyring) = keyring_home(dir.path());
        keyring.delete("mcp:tavily:header:Authorization").unwrap();
        let shown = vault(dir.path(), keyring);
        let error = export_with(&paths, "/home/hub", None, &|| Ok(shown.clone())).unwrap_err();
        assert!(error.contains("mcp:tavily:header:Authorization"), "{error}");
        assert!(!error.contains("sk-test-1234") && !error.contains("tvly-test"), "never a value: {error}");
        let unavailable =
            export_with(&paths, "/home/hub", None, &|| Err("secret-tool is not installed".into())).unwrap_err();
        assert!(unavailable.contains("secret-tool is not installed"), "{unavailable}");
    }

    #[test]
    fn export_refuses_a_credential_file_that_is_not_json() {
        let dir = tempfile::tempdir().unwrap();
        let paths = plain_home(dir.path());
        std::fs::write(paths.root.join("auth.json"), "{ nope").unwrap();
        let error = export_plain(&paths, None).unwrap_err();
        assert!(error.contains("auth.json"), "{error}");
    }

    #[test]
    fn import_writes_the_four_files_with_mode_600() {
        let dir = tempfile::tempdir().unwrap();
        let paths = hub(dir.path());
        let bundle = export_plain(&plain_home(dir.path()), None).unwrap();
        let written = import(&paths, &serde_json::to_string(&bundle).unwrap()).unwrap();
        assert_eq!(written, BUNDLE_FILES);
        for name in BUNDLE_FILES {
            assert_eq!(mode(&paths.root.join(name)), 0o600, "{name}");
            assert_eq!(std::fs::read_to_string(paths.root.join(name)).unwrap(), bundle.files[name], "{name}");
        }
        assert_eq!(mode(&paths.root), 0o700, "the folder is made private");
        assert_eq!(files_in(&paths.root), {
            let mut names: Vec<String> = BUNDLE_FILES.iter().map(|n| n.to_string()).collect();
            names.sort();
            names
        });
    }

    #[test]
    fn import_writes_only_what_the_bundle_has_in_bundle_order() {
        let dir = tempfile::tempdir().unwrap();
        let paths = hub(dir.path());
        let written = import(&paths, &line(&[("mcp.json", "{}"), ("settings.json", "{}")])).unwrap();
        assert_eq!(written, ["settings.json", "mcp.json"]);
        assert!(!paths.root.join("auth.json").exists());
    }

    #[test]
    fn import_refuses_a_name_with_a_path() {
        let dir = tempfile::tempdir().unwrap();
        let paths = hub(dir.path());
        for name in ["../x", "a/b", "settings.json/", "sessions.json", "/etc/passwd", ""] {
            let error = import(&paths, &line(&[("settings.json", "{}"), (name, "{}")])).unwrap_err();
            assert!(error.contains("not part of a bundle"), "{name}: {error}");
        }
        assert!(!dir.path().join("x").exists() && !paths.root.join("settings.json").exists());
    }

    #[test]
    fn import_backs_up_what_it_replaces() {
        let dir = tempfile::tempdir().unwrap();
        let paths = hub(dir.path());
        paths.ensure_root().unwrap();
        std::fs::write(paths.root.join("settings.json"), "hub's own, not even json").unwrap();
        std::fs::set_permissions(paths.root.join("settings.json"), std::fs::Permissions::from_mode(0o644)).unwrap();
        import(&paths, &line(&[("settings.json", r#"{"language":"en"}"#)])).unwrap();
        let backup = paths.root.join("settings.json.bak");
        assert_eq!(std::fs::read_to_string(&backup).unwrap(), "hub's own, not even json");
        assert_eq!(mode(&backup), 0o600);
        assert_eq!(mode(&paths.root.join("settings.json")), 0o600);
        assert_eq!(
            json_of(&std::fs::read_to_string(paths.root.join("settings.json")).unwrap()),
            json!({"language": "en"})
        );
        assert!(!files_in(&paths.root).iter().any(|n| n.ends_with(".tmp")));
    }

    #[test]
    fn import_rejects_a_bundle_that_is_not_json_without_touching_files() {
        let dir = tempfile::tempdir().unwrap();
        let paths = hub(dir.path());
        paths.ensure_root().unwrap();
        std::fs::write(paths.root.join("settings.json"), r#"{"keep":"me"}"#).unwrap();
        for bad in [
            "not json at all".to_string(),
            r#"{"files": ["settings.json"]}"#.to_string(),
            r#"{"files":{"settings.json":"{}"},"extra":1}"#.to_string(),
            line(&[("settings.json", r#"{"language":"en"}"#), ("auth.json", "{ nope")]),
            line(&[("settings.json", "[]")]),
            line(&[]),
        ] {
            assert!(import(&paths, &bad).is_err(), "{bad}");
        }
        assert_eq!(files_in(&paths.root), ["settings.json"]);
        assert_eq!(std::fs::read_to_string(paths.root.join("settings.json")).unwrap(), r#"{"keep":"me"}"#);
    }

    #[test]
    fn a_bundle_line_is_read_up_to_a_limit() {
        assert_eq!(read_bundle_line("abc\nrest".as_bytes(), 100).unwrap(), "abc");
        assert_eq!(read_bundle_line("abc".as_bytes(), 100).unwrap(), "abc");
        assert!(read_bundle_line("".as_bytes(), 100).unwrap_err().contains("no bundle"));
        assert!(read_bundle_line("\n".as_bytes(), 100).unwrap_err().contains("no bundle"));
        assert!(read_bundle_line("0123456789\n".as_bytes(), 5).unwrap_err().contains("too large"));
        assert_eq!(read_bundle_line("01234\n".as_bytes(), 5).unwrap(), "01234");
    }

    const SS: &str = "\
LISTEN 0      128          0.0.0.0:2222       0.0.0.0:*
LISTEN 0      128             [::]:22            [::]:*
LISTEN 0      511        127.0.0.1:3000       0.0.0.0:*
LISTEN 0      4096    100.64.0.9%tailscale0:41641   0.0.0.0:*
LISTEN 0      128               *:80               *:*
LISTEN 0      4096    127.0.0.1%lo:53         0.0.0.0:*    users:((\"x\",pid=1,fd=3))
LISTEN 0      128   [fe80::1%eth0]:631         [::]:*
";

    #[test]
    fn parse_listeners_reads_ss_output() {
        let listeners = parse_listeners(SS);
        let pairs: Vec<(&str, u16)> = listeners.iter().map(|l| (l.addr.as_str(), l.port)).collect();
        assert_eq!(
            pairs,
            [
                ("0.0.0.0", 2222),
                ("[::]", 22),
                ("127.0.0.1", 3000),
                ("100.64.0.9", 41641),
                ("*", 80),
                ("127.0.0.1", 53),
                ("[fe80::1]", 631),
            ]
        );
    }

    #[test]
    fn parse_listeners_ignores_what_it_cannot_read() {
        let text = "State Recv-Q Send-Q Local Peer\n\ngarbage\nLISTEN 0 1 nonsense 0.0.0.0:*\nLISTEN 0 1 1.2.3.4:http *:*\nLISTEN 0 1 :22 *:*\nLISTEN 0 1 10.0.0.2:22 0.0.0.0:*\n";
        assert_eq!(parse_listeners(text), [Listener { addr: "10.0.0.2".into(), port: 22 }]);
        assert!(parse_listeners("").is_empty());
    }

    fn at(addr: &str, port: u16) -> Listener {
        Listener { addr: addr.into(), port }
    }

    #[test]
    fn public_ssh_is_true_for_a_wildcard_and_false_for_loopback_and_private_only() {
        for wildcard in ["0.0.0.0", "*", "::", "[::]"] {
            assert!(public_ssh(&[at(wildcard, 22)], 22), "{wildcard}");
        }
        assert!(public_ssh(&parse_listeners(SS), 2222));
        assert!(public_ssh(&parse_listeners(SS), 22), "[::] on 22");
        assert!(public_ssh(&[at("203.0.113.7", 22)], 22), "a public address");
        assert!(public_ssh(&[at("[2001:db8::1]", 22)], 22));
        assert!(!public_ssh(&[at("127.0.0.1", 22), at("[::1]", 22), at("10.1.2.3", 22), at("100.64.0.9", 22)], 22));
        assert!(!public_ssh(&[at("[fd7a:115c:a1e0::1]", 22), at("192.168.1.5", 22)], 22), "tailscale and LAN only");
        assert!(!public_ssh(&parse_listeners(SS), 3000), "loopback only");
        assert!(!public_ssh(&parse_listeners(SS), 4444), "nothing listens");
        assert!(!public_ssh(&[], 22));
        assert!(!public_ssh(&[at("0.0.0.0", 80)], 22), "another port");
    }

    #[test]
    fn private_address_covers_tailscale_and_rfc1918_and_not_a_public_ip() {
        let ip = |text: &str| text.parse::<IpAddr>().unwrap();
        for private in [
            "127.0.0.1",
            "127.9.9.9",
            "10.0.0.1",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "169.254.1.1",
            "100.64.0.1",
            "100.100.100.100",
            "100.127.255.255",
            "::1",
            "fc00::1",
            "fd12::1",
            "fd7a:115c:a1e0::5",
            "fe80::1",
            "::ffff:10.0.0.1",
        ] {
            assert!(private_address(ip(private)), "{private}");
        }
        for public in [
            "8.8.8.8",
            "203.0.113.7",
            "172.15.0.1",
            "172.32.0.1",
            "100.63.255.255",
            "100.128.0.1",
            "192.169.0.1",
            "2001:db8::1",
            "2606:4700::1",
            "::ffff:8.8.8.8",
            "0.0.0.0",
        ] {
            assert!(!private_address(ip(public)), "{public}");
        }
    }
}
