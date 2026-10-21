//! Keys out of plain files (docs/superpowers/specs/2026-10-21-keyring-design.md).
//!
//! Fixed keys live in the keyring and the files hold `!aria secret …` commands that Pi runs when it needs them. The files
//! Pi rewrites itself (`auth.json`, `mcp-auth.json`) are sealed in the keyring whole; while the dock runs, a live copy of
//! each sits in a private folder outside the home and outside backups, and every change to it is sealed again.

use crate::keyring::{Backend, KeyringError};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The files sealed whole: Pi rewrites them (token refreshes, sign-ins), so they cannot be commands.
pub const SEALED: [&str; 2] = ["auth.json", "mcp-auth.json"];

#[derive(Debug, thiserror::Error)]
pub enum VaultError {
    #[error(transparent)]
    Keyring(#[from] KeyringError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("the keyring gave back something else for {0}")]
    Mismatch(String),
}

/// Where the dock keeps keys: the keyring, the live folder of the sealed files, and the `aria` the commands run.
#[derive(Clone, Debug)]
pub struct Vault {
    pub keyring: Arc<dyn Backend>,
    pub live: PathBuf,
    pub exe: PathBuf,
}

/// What each sealed file held when it was last sealed or written live (its sha256), so only a change is sealed again.
#[derive(Debug, Default)]
pub struct Sync {
    seen: HashMap<&'static str, [u8; 32]>,
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn account(name: &str) -> String {
    format!("file:{name}")
}

/// The private folder of the live copies: in memory on Linux (`$XDG_RUNTIME_DIR`, which the sandbox also hides), the
/// per-user temporary folder on macOS (the same for the LaunchAgent and a terminal, outside backups). One per home.
pub fn live_dir(root: &Path) -> PathBuf {
    let id: String = digest(root.as_os_str().as_encoded_bytes()).iter().take(6).map(|b| format!("{b:02x}")).collect();
    runtime_base().join(format!("aria-secrets-{id}"))
}

#[cfg(target_os = "linux")]
fn runtime_base() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir)
}

#[cfg(target_os = "macos")]
fn runtime_base() -> PathBuf {
    let mut buffer = vec![0u8; 1024];
    // SAFETY: the buffer is as long as the length given; confstr writes at most that much, NUL-terminated.
    let length = unsafe { libc::confstr(libc::_CS_DARWIN_USER_TEMP_DIR, buffer.as_mut_ptr().cast(), buffer.len()) };
    if length == 0 || length > buffer.len() {
        return std::env::temp_dir();
    }
    buffer.truncate(length - 1);
    PathBuf::from(String::from_utf8_lossy(&buffer).into_owned())
}

/// A word for a POSIX shell: `'…'`, with each `'` written `'\''`.
pub fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// The value a file holds in place of a key: Pi runs it and reads the key from its output.
pub fn command_for(exe: &Path, root: &Path, account: &str) -> String {
    format!(
        "!{} secret --home {} {}",
        shell_quote(&exe.to_string_lossy()),
        shell_quote(&root.to_string_lossy()),
        shell_quote(account)
    )
}

/// The words of a command made of single-quoted words and bare words (what `command_for` writes), or None.
fn words(command: &str) -> Option<Vec<String>> {
    let mut out = Vec::new();
    let mut chars = command.chars().peekable();
    loop {
        while chars.peek() == Some(&' ') {
            chars.next();
        }
        if chars.peek().is_none() {
            return Some(out);
        }
        let mut word = String::new();
        while let Some(&c) = chars.peek() {
            match c {
                ' ' => break,
                '\'' => {
                    chars.next();
                    loop {
                        match chars.next()? {
                            '\'' => break,
                            c => word.push(c),
                        }
                    }
                }
                '\\' => {
                    chars.next();
                    word.push(chars.next()?);
                }
                c => {
                    chars.next();
                    word.push(c);
                }
            }
        }
        out.push(word);
    }
}

/// The account when `value` is one of Aria's own commands (whatever `aria` and home it names), else None.
pub fn own_account(value: &str) -> Option<String> {
    match words(value.strip_prefix('!')?)?.as_slice() {
        [_, secret, home, _, account] if secret == "secret" && home == "--home" => Some(account.clone()),
        _ => None,
    }
}

/// What a file's value really is: a literal as it is, Aria's own command read from the keyring, any other command None
/// (Aria never runs a shell to find a key).
pub fn resolve(vault: Option<&Vault>, value: &str) -> Option<String> {
    if !value.starts_with('!') {
        return Some(value.to_string());
    }
    let account = own_account(value)?;
    vault?.keyring.get(&account).ok().flatten()
}

/// Writes `bytes` to `path` with mode 0600 through a temporary file in the same folder, so a reader never sees half.
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&tmp)?;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

impl Vault {
    /// Stores `value` and reads it back: a keyring that gives back something else has not stored it.
    pub fn store(&self, account: &str, value: &str) -> Result<(), VaultError> {
        self.keyring.set(account, value)?;
        match self.keyring.get(account)? {
            Some(back) if back == value => Ok(()),
            _ => Err(VaultError::Mismatch(account.to_string())),
        }
    }

    /// Makes the live folder (0700) and fills it: a live copy already there was left by a crash and is newer than the
    /// keyring (which is synced every second), so it is sealed; otherwise the keyring's copy is written live.
    pub fn open(&self) -> Result<Sync, VaultError> {
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&self.live)?;
        std::fs::set_permissions(&self.live, std::fs::Permissions::from_mode(0o700))?;
        let mut state = Sync::default();
        for name in SEALED {
            let live = self.live.join(name);
            if let Ok(bytes) = std::fs::read(&live) {
                self.store(&account(name), &String::from_utf8_lossy(&bytes))?;
                state.seen.insert(name, digest(&bytes));
            } else if let Some(value) = self.keyring.get(&account(name))? {
                write_private(&live, value.as_bytes())?;
                state.seen.insert(name, digest(value.as_bytes()));
            }
        }
        Ok(state)
    }

    /// Seals every live copy that changed since it was last sealed; one Pi removed (a sign-out) leaves the keyring too.
    /// Returns what failed; a failed file is tried again on the next call.
    pub fn sync(&self, state: &mut Sync) -> Vec<VaultError> {
        let mut errors = Vec::new();
        for name in SEALED {
            let result = match std::fs::read(self.live.join(name)) {
                Ok(bytes) => {
                    let now = digest(&bytes);
                    if state.seen.get(name) == Some(&now) {
                        continue;
                    }
                    self.store(&account(name), &String::from_utf8_lossy(&bytes)).map(|()| Some(now))
                }
                Err(_) if state.seen.contains_key(name) => {
                    self.keyring.delete(&account(name)).map(|()| None).map_err(VaultError::from)
                }
                Err(_) => continue,
            };
            match result {
                Ok(Some(now)) => {
                    state.seen.insert(name, now);
                }
                Ok(None) => {
                    state.seen.remove(name);
                }
                Err(error) => errors.push(error),
            }
        }
        errors
    }

    /// A last sync, then the live copies go. Returns what could not be sealed (its live copy is then kept).
    pub fn close(&self, state: &mut Sync) -> Vec<VaultError> {
        let errors = self.sync(state);
        if errors.is_empty() {
            for name in SEALED {
                let _ = std::fs::remove_file(self.live.join(name));
            }
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyring::{KeyringError, MemoryBackend};
    use crate::paths::AriaPaths;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn vault_in(dir: &Path, keyring: Arc<dyn Backend>) -> Vault {
        Vault { keyring, live: dir.join("live"), exe: PathBuf::from("/bin/aria") }
    }

    fn mode(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    /// Counts the values it is given, and fails them while `fail` is set.
    #[derive(Debug, Default)]
    struct Counting {
        inner: MemoryBackend,
        sets: AtomicUsize,
        fail: std::sync::atomic::AtomicBool,
    }

    impl Backend for Counting {
        fn get(&self, account: &str) -> Result<Option<String>, KeyringError> {
            self.inner.get(account)
        }
        fn set(&self, account: &str, value: &str) -> Result<(), KeyringError> {
            if self.fail.load(Ordering::SeqCst) {
                return Err(KeyringError::Failed("locked".into()));
            }
            self.sets.fetch_add(1, Ordering::SeqCst);
            self.inner.set(account, value)
        }
        fn delete(&self, account: &str) -> Result<(), KeyringError> {
            self.inner.delete(account)
        }
    }

    #[test]
    fn open_writes_the_keyring_copies_live() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = Arc::new(MemoryBackend::default());
        keyring.set("file:mcp-auth.json", "{\"a\":1}").unwrap();
        let vault = vault_in(dir.path(), keyring);
        vault.open().unwrap();
        let live = vault.live.join("mcp-auth.json");
        assert_eq!(std::fs::read_to_string(&live).unwrap(), "{\"a\":1}");
        assert_eq!(mode(&live), 0o600);
        assert_eq!(mode(&vault.live), 0o700);
        assert!(!vault.live.join("auth.json").exists(), "nothing sealed, nothing written");
    }

    #[test]
    fn open_after_a_crash_keeps_the_newer_live_copy() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = Arc::new(MemoryBackend::default());
        keyring.set("file:mcp-auth.json", "{\"a\":1}").unwrap();
        let vault = vault_in(dir.path(), keyring.clone());
        std::fs::create_dir_all(&vault.live).unwrap();
        std::fs::write(vault.live.join("mcp-auth.json"), "{\"a\":2}").unwrap();
        vault.open().unwrap();
        assert_eq!(keyring.get("file:mcp-auth.json").unwrap().as_deref(), Some("{\"a\":2}"));
        assert_eq!(std::fs::read_to_string(vault.live.join("mcp-auth.json")).unwrap(), "{\"a\":2}");
    }

    #[test]
    fn sync_seals_only_a_changed_file() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = Arc::new(Counting::default());
        keyring.inner.set("file:auth.json", "{}").unwrap();
        let vault = vault_in(dir.path(), keyring.clone());
        let mut state = vault.open().unwrap();
        assert!(vault.sync(&mut state).is_empty());
        assert_eq!(keyring.sets.load(Ordering::SeqCst), 0);
        std::fs::write(vault.live.join("auth.json"), "{\"b\":1}").unwrap();
        assert!(vault.sync(&mut state).is_empty());
        assert_eq!(keyring.sets.load(Ordering::SeqCst), 1);
        assert_eq!(keyring.get("file:auth.json").unwrap().as_deref(), Some("{\"b\":1}"));
        vault.sync(&mut state);
        assert_eq!(keyring.sets.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_live_file_pi_removed_is_removed_from_the_keyring() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = Arc::new(MemoryBackend::default());
        keyring.set("file:mcp-auth.json", "{\"a\":1}").unwrap();
        let vault = vault_in(dir.path(), keyring.clone());
        let mut state = vault.open().unwrap();
        std::fs::remove_file(vault.live.join("mcp-auth.json")).unwrap();
        assert!(vault.sync(&mut state).is_empty());
        assert_eq!(keyring.get("file:mcp-auth.json").unwrap(), None);
    }

    #[test]
    fn close_seals_and_removes_the_live_copies() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = Arc::new(MemoryBackend::default());
        keyring.set("file:auth.json", "{}").unwrap();
        let vault = vault_in(dir.path(), keyring.clone());
        let mut state = vault.open().unwrap();
        std::fs::write(vault.live.join("auth.json"), "{\"c\":3}").unwrap();
        vault.close(&mut state);
        assert_eq!(keyring.get("file:auth.json").unwrap().as_deref(), Some("{\"c\":3}"));
        assert!(!vault.live.join("auth.json").exists());
    }

    #[test]
    fn a_failing_sync_keeps_the_live_copy_and_reports() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = Arc::new(Counting::default());
        keyring.inner.set("file:auth.json", "{}").unwrap();
        let vault = vault_in(dir.path(), keyring.clone());
        let mut state = vault.open().unwrap();
        std::fs::write(vault.live.join("auth.json"), "{\"d\":4}").unwrap();
        keyring.fail.store(true, Ordering::SeqCst);
        assert_eq!(vault.sync(&mut state).len(), 1);
        assert_eq!(std::fs::read_to_string(vault.live.join("auth.json")).unwrap(), "{\"d\":4}");
        keyring.fail.store(false, Ordering::SeqCst);
        assert!(vault.sync(&mut state).is_empty(), "retried");
        assert_eq!(keyring.get("file:auth.json").unwrap().as_deref(), Some("{\"d\":4}"));
    }

    #[test]
    fn commands_round_trip() {
        let command = command_for(Path::new("/a b/ar'ia"), Path::new("/r'oot"), "key:x");
        assert_eq!(command, r#"!'/a b/ar'\''ia' secret --home '/r'\''oot' 'key:x'"#);
        assert_eq!(own_account(&command).as_deref(), Some("key:x"));
        assert_eq!(own_account("!security find-generic-password -ws anthropic"), None);
        assert_eq!(own_account("plain"), None);
        let keyring = Arc::new(MemoryBackend::default());
        keyring.set("key:x", "real").unwrap();
        let vault = Vault { keyring, live: PathBuf::from("/nowhere"), exe: PathBuf::from("/bin/aria") };
        assert_eq!(resolve(Some(&vault), "literal").as_deref(), Some("literal"));
        assert_eq!(resolve(Some(&vault), &command).as_deref(), Some("real"));
        assert_eq!(resolve(None, &command), None);
        assert_eq!(resolve(Some(&vault), "!echo hi"), None, "never runs a shell");
    }

    #[test]
    fn the_live_folder_is_per_home() {
        let a = live_dir(Path::new("/h/.aria"));
        let b = live_dir(Path::new("/tmp/other"));
        assert_ne!(a, b);
        assert!(a.file_name().unwrap().to_string_lossy().starts_with("aria-secrets-"));
        assert!(!a.starts_with("/h/.aria"), "never inside the home");
    }

    #[test]
    fn paths_point_at_the_live_copies_only_with_a_vault() {
        let plain = AriaPaths::new(PathBuf::from("/r"));
        assert_eq!(plain.auth(), PathBuf::from("/r/auth.json"));
        assert_eq!(plain.mcp_auth(), PathBuf::from("/r/mcp-auth.json"));
        let vault = Vault {
            keyring: Arc::new(MemoryBackend::default()),
            live: PathBuf::from("/live"),
            exe: PathBuf::from("/bin/aria"),
        };
        let sealed = AriaPaths { vault: Some(vault), ..plain };
        assert_eq!(sealed.auth(), PathBuf::from("/live/auth.json"));
        assert_eq!(sealed.mcp_auth(), PathBuf::from("/live/mcp-auth.json"));
        assert_eq!(sealed.mcp(), PathBuf::from("/r/mcp.json"), "mcp.json stays in the home");
    }
}
