//! The keyring from the dock's side (docs/superpowers/specs/2026-10-21-keyring-design.md): the vault the dock opens at
//! start, and `aria secret` / `aria keyring`.

use crate::{control, note};
use aria_core::keyring::{self, Backend, KeyringError};
use aria_core::paths::AriaPaths;
use aria_core::settings::{self, Settings};
use aria_core::vault::{self, Migrated, Sync, Vault, SEALED};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The vault the dock holds while it runs, and what it last sealed.
pub struct Open {
    pub vault: Vault,
    pub state: Sync,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
}

/// The `aria` the files' commands run: this binary, by the path it was installed at.
fn exe() -> PathBuf {
    control::binary_path(std::env::current_exe().unwrap_or_else(|_| PathBuf::from("aria")))
}

fn vault_for(root: &Path) -> Result<Vault, KeyringError> {
    let keyring = keyring::system(&keyring::service_for(root, &home()))?;
    Ok(Vault { keyring, live: vault::live_dir(root), exe: exe() })
}

/// The dock's start: the keys move into the keyring (once), the sealed files are written live, and the paths returned
/// point at them. With the setting off, no keyring, or a migration that failed, the dock runs on the plain files and the
/// log says why.
pub fn open_for_dock(paths: AriaPaths, settings: Option<&Settings>) -> (AriaPaths, Option<Open>) {
    let keyring = keyring::system(&keyring::service_for(&paths.root, &home()));
    let live = vault::live_dir(&paths.root);
    open_with(paths, settings, keyring, exe(), live)
}

pub fn open_with(
    paths: AriaPaths,
    settings: Option<&Settings>,
    keyring: Result<Arc<dyn Backend>, KeyringError>,
    exe: PathBuf,
    live: PathBuf,
) -> (AriaPaths, Option<Open>) {
    let off = settings.is_some_and(|s| !s.credentials.keyring);
    let keyring = match keyring {
        Ok(keyring) => keyring,
        Err(error) => {
            if migrated(&paths.root, None) {
                note(&format!("THE KEYS ARE IN THE KEYRING, WHICH IS NOT AVAILABLE: Aria starts without its credentials ({error})"));
            } else {
                note(&format!("keys stay in plain files: {error}"));
            }
            return (paths, None);
        }
    };
    if off {
        // Off by hand on a home whose keys are already in the keyring: falling back to plain files would strand them
        // (and the next sign-in would start a new auth.json). `aria keyring off` is the way back.
        if !migrated(&paths.root, Some(keyring.as_ref())) {
            note("keys stay in plain files (credentials.keyring is off)");
            return (paths, None);
        }
        note("credentials.keyring is off, but the keys are in the keyring: run `aria keyring off` to bring them back");
    }
    let vault = Vault { keyring, live, exe };
    match vault.migrate(&paths.root) {
        Ok(moved) if moved != Migrated::default() => note(&format!(
            "moved into the keyring: {} keys, {} connector values, {} files",
            moved.keys, moved.mcp_values, moved.files
        )),
        Ok(_) => {}
        Err(error) => {
            note(&format!("keys stay in plain files: the move into the keyring stopped ({error})"));
            return (paths, None);
        }
    }
    let sealed = AriaPaths { vault: Some(vault.clone()), ..paths };
    match vault.open().or_else(|_| vault.open()) {
        Ok(state) => {
            if let Err(error) = vault.refresh_commands(&sealed.root) {
                note(&format!("the keyring commands could not be updated: {error}"));
            }
            (sealed, Some(Open { vault, state }))
        }
        Err(error) => {
            note(&format!("THE KEYRING COULD NOT BE READ: Aria starts without its credentials ({error})"));
            (sealed, None)
        }
    }
}

/// Whether the home's keys are already in the keyring: sealed files there, or Aria's commands in its `mcp.json`.
fn migrated(root: &Path, keyring: Option<&dyn Backend>) -> bool {
    let sealed =
        keyring.is_some_and(|k| SEALED.iter().any(|name| k.get(&format!("file:{name}")).ok().flatten().is_some()));
    let mut commands = Vec::new();
    if let Ok(value) =
        serde_json::from_str::<Value>(&std::fs::read_to_string(root.join("mcp.json")).unwrap_or_default())
    {
        own_accounts(&value, &mut commands);
    }
    sealed || !commands.is_empty()
}

/// What `aria secret` prints: the value, or why there is none (for stderr; stdout stays empty).
pub fn secret_value(keyring: &dyn Backend, account: &str) -> Result<String, String> {
    match keyring.get(account) {
        Ok(Some(value)) => Ok(value),
        Ok(None) => Err(format!("no {account} in the keyring")),
        Err(error) => Err(error.to_string()),
    }
}

/// `aria secret --home <root> <account>`: the command Pi runs to read a key.
pub fn secret_command(args: &[String]) -> i32 {
    let [flag, root, account] = args else {
        eprintln!("usage: aria secret --home <root> <account>");
        return 2;
    };
    if flag != "--home" {
        eprintln!("usage: aria secret --home <root> <account>");
        return 2;
    }
    let value = keyring::system(&keyring::service_for(Path::new(root), &home()))
        .map_err(|error| error.to_string())
        .and_then(|keyring| secret_value(keyring.as_ref(), account));
    match value {
        Ok(value) => {
            print!("{value}");
            0
        }
        Err(why) => {
            eprintln!("aria secret: {why}");
            1
        }
    }
}

/// The way back: plain files in `root` again, the keyring emptied of Aria's values, and the setting off.
pub fn turn_off(paths: &AriaPaths, vault: &Vault) -> Result<(), String> {
    vault.unseal_to_plain(&paths.root).map_err(|error| error.to_string())?;
    set_keyring(paths, false)
}

fn set_keyring(paths: &AriaPaths, on: bool) -> Result<(), String> {
    let mut current =
        settings::load(paths).map_err(|error| error.to_string())?.unwrap_or_else(|| Settings::defaults("en"));
    current.credentials.keyring = on;
    settings::save(paths, &current).map_err(|error| error.to_string())
}

fn own_accounts(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(text) => out.extend(vault::own_account(text)),
        Value::Array(items) => items.iter().for_each(|v| own_accounts(v, out)),
        Value::Object(map) => map.values().for_each(|v| own_accounts(v, out)),
        _ => {}
    }
}

/// The accounts Aria keeps in the keyring, from its own files (the keyring is never listed): names only.
pub fn accounts(paths: &AriaPaths, vault: &Vault) -> Vec<String> {
    let mut found = Vec::new();
    for name in SEALED {
        let account = format!("file:{name}");
        let text =
            std::fs::read_to_string(vault.live.join(name)).ok().or_else(|| vault.keyring.get(&account).ok().flatten());
        if let Some(text) = text {
            found.push(account);
            if let Ok(value) = serde_json::from_str::<Value>(&text) {
                own_accounts(&value, &mut found);
            }
        }
    }
    for name in ["auth.json", "mcp.json"] {
        if let Ok(value) =
            serde_json::from_str::<Value>(&std::fs::read_to_string(paths.root.join(name)).unwrap_or_default())
        {
            own_accounts(&value, &mut found);
        }
    }
    found.sort();
    found.dedup();
    found
}

/// `aria keyring status|on|off`.
pub fn keyring_command(what: Option<&str>) -> i32 {
    let paths = AriaPaths::from_env();
    let on = settings::load(&paths).ok().flatten().is_none_or(|s| s.credentials.keyring);
    match what {
        Some("status") | None => {
            println!("keyring: {}", if on { "on" } else { "off (credentials.keyring)" });
            match vault_for(&paths.root) {
                Ok(vault) => {
                    println!("available: yes");
                    for account in accounts(&paths, &vault) {
                        println!("  {account}");
                    }
                }
                Err(error) => println!("available: no ({error})"),
            }
            0
        }
        Some("on") => match set_keyring(&paths, true) {
            Ok(()) => {
                println!("The keyring is on. The dock moves the keys at its next start: aria quit && aria start");
                0
            }
            Err(why) => {
                eprintln!("aria keyring: {why}");
                1
            }
        },
        Some("off") => {
            // The running dock holds the live copies and seals them every second: it has to be stopped first.
            if std::os::unix::net::UnixStream::connect(control::control_socket_path()).is_ok() {
                eprintln!("aria keyring: quit Aria first (aria quit), then run aria keyring off again");
                return 1;
            }
            let result =
                vault_for(&paths.root).map_err(|error| error.to_string()).and_then(|vault| turn_off(&paths, &vault));
            match result {
                Ok(()) => {
                    println!("The keys are back in plain files in {}; the keyring is off.", paths.root.display());
                    0
                }
                Err(why) => {
                    eprintln!("aria keyring: {why}");
                    1
                }
            }
        }
        Some(other) => {
            eprintln!("aria keyring: unknown '{other}' (status, on or off)");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::keyring::{Backend, KeyringError, MemoryBackend};
    use aria_core::settings::Settings;
    use std::sync::Arc;

    fn home(dir: &Path) -> AriaPaths {
        let root = dir.join("home");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("auth.json"), r#"{"opencode-go":{"type":"api_key","key":"K"}}"#).unwrap();
        AriaPaths::new(root)
    }

    fn settings(keyring: bool) -> Settings {
        let mut settings = Settings::defaults("en");
        settings.credentials.keyring = keyring;
        settings
    }

    fn open(
        paths: AriaPaths,
        settings: &Settings,
        keyring: Result<Arc<dyn Backend>, KeyringError>,
        dir: &Path,
    ) -> (AriaPaths, Option<Open>) {
        open_with(paths, Some(settings), keyring, PathBuf::from("/bin/aria"), dir.join("live"))
    }

    #[test]
    fn a_dock_without_the_keyring_keeps_plain_files() {
        let dir = tempfile::tempdir().unwrap();
        let before = std::fs::read(home(dir.path()).auth()).unwrap();
        let (paths, open_vault) =
            open(home(dir.path()), &settings(false), Ok(Arc::new(MemoryBackend::default())), dir.path());
        assert!(paths.vault.is_none() && open_vault.is_none());
        assert_eq!(std::fs::read(paths.auth()).unwrap(), before);
        let (paths, open_vault) =
            open(home(dir.path()), &settings(true), Err(KeyringError::Unavailable("none".into())), dir.path());
        assert!(paths.vault.is_none() && open_vault.is_none());
        assert_eq!(std::fs::read(paths.auth()).unwrap(), before);
        assert!(!dir.path().join("live").exists());
    }

    #[test]
    fn a_dock_with_the_keyring_migrates_and_points_at_the_live_copies() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = Arc::new(MemoryBackend::default());
        let (paths, open_vault) = open(home(dir.path()), &settings(true), Ok(keyring.clone()), dir.path());
        assert!(open_vault.is_some());
        assert_eq!(paths.auth(), dir.path().join("live/auth.json"));
        assert!(!dir.path().join("home/auth.json").exists());
        assert_eq!(keyring.get("key:opencode-go").unwrap().as_deref(), Some("K"));
        assert_eq!(
            aria_core::auth::AuthStore::for_paths(&paths).get("opencode-go").unwrap(),
            Some(aria_core::auth::Credential::ApiKey { key: "K".into() })
        );
    }

    #[test]
    fn a_setting_turned_off_by_hand_on_a_migrated_home_keeps_the_keyring() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = Arc::new(MemoryBackend::default());
        let (_paths, open_vault) = open(home(dir.path()), &settings(true), Ok(keyring.clone()), dir.path());
        let Open { vault, mut state } = open_vault.unwrap();
        vault.close(&mut state);
        let (paths, open_vault) =
            open(AriaPaths::new(dir.path().join("home")), &settings(false), Ok(keyring), dir.path());
        assert!(open_vault.is_some(), "the keys are only in the keyring: off by hand must not strand them");
        assert_eq!(
            aria_core::auth::AuthStore::for_paths(&paths).get("opencode-go").unwrap(),
            Some(aria_core::auth::Credential::ApiKey { key: "K".into() })
        );
    }

    #[test]
    fn a_failed_migration_runs_on_the_plain_files() {
        #[derive(Debug)]
        struct Refusing;
        impl Backend for Refusing {
            fn get(&self, _: &str) -> Result<Option<String>, KeyringError> {
                Ok(None)
            }
            fn set(&self, _: &str, _: &str) -> Result<(), KeyringError> {
                Err(KeyringError::Failed("locked".into()))
            }
            fn delete(&self, _: &str) -> Result<(), KeyringError> {
                Ok(())
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let (paths, open_vault) = open(home(dir.path()), &settings(true), Ok(Arc::new(Refusing)), dir.path());
        assert!(paths.vault.is_none() && open_vault.is_none());
        assert!(dir.path().join("home/auth.json").exists());
    }

    #[test]
    fn secret_prints_only_the_value() {
        let keyring = MemoryBackend::default();
        keyring.set("key:x", "v a l").unwrap();
        assert_eq!(secret_value(&keyring, "key:x"), Ok("v a l".to_string()));
    }

    #[test]
    fn secret_on_a_missing_account_prints_nothing_and_fails() {
        let keyring = MemoryBackend::default();
        assert!(secret_value(&keyring, "key:nope").is_err());
    }

    #[test]
    fn keyring_off_restores_plain_files_and_turns_the_setting_off() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = Arc::new(MemoryBackend::default());
        let plain = home(dir.path());
        aria_core::settings::save(&plain, &settings(true)).unwrap();
        let (_sealed, open_vault) = open(plain.clone(), &settings(true), Ok(keyring.clone()), dir.path());
        let Open { vault, mut state } = open_vault.unwrap();
        vault.close(&mut state);
        turn_off(&plain, &vault).unwrap();
        let auth: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(plain.auth()).unwrap()).unwrap();
        assert_eq!(auth["opencode-go"]["key"], "K");
        assert_eq!(keyring.get("key:opencode-go").unwrap(), None);
        assert!(!aria_core::settings::load(&plain).unwrap().unwrap().credentials.keyring);
    }

    #[test]
    fn status_names_the_accounts_never_the_values() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = Arc::new(MemoryBackend::default());
        let plain = home(dir.path());
        let (_sealed, open_vault) = open(plain.clone(), &settings(true), Ok(keyring.clone()), dir.path());
        let Open { vault, .. } = open_vault.unwrap();
        let accounts = accounts(&plain, &vault);
        assert!(accounts.contains(&"key:opencode-go".to_string()), "{accounts:?}");
        assert!(accounts.contains(&"file:auth.json".to_string()), "{accounts:?}");
        assert!(!accounts.iter().any(|a| a.contains('K') && a.len() == 1));
    }
}
