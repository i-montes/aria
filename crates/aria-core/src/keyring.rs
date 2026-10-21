//! The system keyring, where Aria keeps keys instead of plain files (docs/superpowers/specs/2026-10-21-keyring-design.md).
//!
//! One interface over three backends: macOS through Apple's `security` tool (signed by Apple, so the keychain never asks
//! after an update of the unsigned `aria`), Linux through libsecret's `secret-tool`, and memory for tests. Values are
//! stored base64, so JSON and any byte pass through both tools unchanged.

use base64::Engine;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum KeyringError {
    #[error("the keyring is not available: {0}")]
    Unavailable(String),
    #[error("the keyring refused: {0}")]
    Failed(String),
}

/// Where values live. `account` names one value (`key:opencode-go`, `file:auth.json`); values are the plain secrets.
pub trait Backend: Send + Sync + std::fmt::Debug {
    fn get(&self, account: &str) -> Result<Option<String>, KeyringError>;
    fn set(&self, account: &str, value: &str) -> Result<(), KeyringError>;
    /// Removing what is not there is not an error.
    fn delete(&self, account: &str) -> Result<(), KeyringError>;
}

/// `aria` for the default home; `aria:<root>` for any other, so a development or test instance never meets the real keys.
pub fn service_for(root: &Path, home: &Path) -> String {
    if root == home.join(".aria") {
        "aria".to_string()
    } else {
        format!("aria:{}", root.display())
    }
}

fn encode(value: &str) -> String {
    base64::engine::general_purpose::STANDARD.encode(value)
}

fn decode(text: &str) -> Result<String, KeyringError> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(text.trim())
        .map_err(|_| KeyringError::Failed("a value that is not Aria's".into()))?;
    String::from_utf8(bytes).map_err(|_| KeyringError::Failed("a value that is not text".into()))
}

fn failed(what: &str, output: &std::process::Output) -> KeyringError {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    KeyringError::Failed(if stderr.is_empty() { format!("{what} exited with {}", output.status) } else { stderr })
}

/// Runs `program` with `args`, `input` on its stdin (or nothing), and returns its output.
fn run(program: &Path, args: &[&str], input: Option<&str>) -> Result<std::process::Output, KeyringError> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(if input.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| KeyringError::Unavailable(format!("{}: {e}", program.display())))?;
    if let Some(input) = input {
        let mut stdin = child.stdin.take().expect("piped");
        stdin.write_all(input.as_bytes()).map_err(|e| KeyringError::Failed(e.to_string()))?;
    }
    child.wait_with_output().map_err(|e| KeyringError::Failed(e.to_string()))
}

#[derive(Debug, Default)]
pub struct MemoryBackend {
    values: Mutex<BTreeMap<String, String>>,
}

impl Backend for MemoryBackend {
    fn get(&self, account: &str) -> Result<Option<String>, KeyringError> {
        Ok(self.values.lock().unwrap().get(account).cloned())
    }
    fn set(&self, account: &str, value: &str) -> Result<(), KeyringError> {
        self.values.lock().unwrap().insert(account.to_string(), value.to_string());
        Ok(())
    }
    fn delete(&self, account: &str) -> Result<(), KeyringError> {
        self.values.lock().unwrap().remove(account);
        Ok(())
    }
}

/// `security`'s exit code for an item that is not there.
const MAC_MISSING: i32 = 44;
/// `security -i` cuts its input lines at 4095 characters (checked on macOS 26): every line Aria writes stays below.
const MAC_LINE: usize = 4000;
/// The longest base64 text one item holds; a longer value is kept in pieces.
const PIECE: usize = 3000;
/// What an item holds when its value is in pieces: `chunks:<n>`, never base64 (which has no `:`).
const PIECES: &str = "chunks:";

/// The stored texts of a keychain, below the values: base64, or the `chunks:<n>` mark.
pub trait Raw {
    fn raw_get(&self, account: &str) -> Result<Option<String>, KeyringError>;
    fn raw_set(&self, account: &str, text: &str) -> Result<(), KeyringError>;
    fn raw_delete(&self, account: &str) -> Result<(), KeyringError>;
}

fn piece(account: &str, n: usize) -> String {
    format!("{account}#{n}")
}

fn pieces(text: &Option<String>) -> usize {
    text.as_deref().and_then(|t| t.strip_prefix(PIECES)).and_then(|n| n.parse().ok()).unwrap_or(0)
}

/// Stores `value`, in pieces when its base64 is long: the pieces first and the mark last, so a reader never sees a mix;
/// pieces an earlier, longer value left are removed after.
pub fn chunked_set(raw: &dyn Raw, account: &str, value: &str) -> Result<(), KeyringError> {
    let text = encode(value);
    let before = pieces(&raw.raw_get(account)?);
    let now = if text.len() <= PIECE {
        raw.raw_set(account, &text)?;
        0
    } else {
        let parts: Vec<&str> = text.as_bytes().chunks(PIECE).map(|c| std::str::from_utf8(c).expect("base64")).collect();
        for (n, part) in parts.iter().enumerate() {
            raw.raw_set(&piece(account, n), part)?;
        }
        raw.raw_set(account, &format!("{PIECES}{}", parts.len()))?;
        parts.len()
    };
    for n in now..before {
        raw.raw_delete(&piece(account, n))?;
    }
    Ok(())
}

pub fn chunked_get(raw: &dyn Raw, account: &str) -> Result<Option<String>, KeyringError> {
    let Some(text) = raw.raw_get(account)? else { return Ok(None) };
    let count = pieces(&Some(text.clone()));
    if count == 0 {
        return decode(&text).map(Some);
    }
    let mut whole = String::new();
    for n in 0..count {
        let part = raw
            .raw_get(&piece(account, n))?
            .ok_or_else(|| KeyringError::Failed(format!("{account}: a piece is missing")))?;
        whole.push_str(&part);
    }
    decode(&whole).map(Some)
}

pub fn chunked_delete(raw: &dyn Raw, account: &str) -> Result<(), KeyringError> {
    let count = pieces(&raw.raw_get(account)?);
    raw.raw_delete(account)?;
    for n in 0..count {
        raw.raw_delete(&piece(account, n))?;
    }
    Ok(())
}

/// The login keychain through `security`. A write goes through `security -i` with the command on stdin, so the secret is
/// never in a process's arguments. Items carry no access list of their own: their creator, `security`, reads, updates and
/// deletes them without a dialog, while `-T` on an update would ask macOS to change the list and raise one each time.
#[derive(Debug)]
pub struct MacBackend {
    pub program: PathBuf,
    pub service: String,
}

impl MacBackend {
    /// `security -i` reads double-quoted words on one line: a quote, a backslash or a control character in a name could end
    /// the word or the line (and start another command), so such a name is refused.
    fn checked<'a>(&self, account: &'a str) -> Result<&'a str, KeyringError> {
        let unsafe_char = |c: char| c == '"' || c == '\\' || c.is_control();
        if self.service.chars().any(unsafe_char) || account.chars().any(unsafe_char) {
            return Err(KeyringError::Failed("a service or account Aria cannot write safely".into()));
        }
        Ok(account)
    }
}

impl Raw for MacBackend {
    fn raw_get(&self, account: &str) -> Result<Option<String>, KeyringError> {
        let account = self.checked(account)?;
        let output = run(&self.program, &["find-generic-password", "-s", &self.service, "-a", account, "-w"], None)?;
        match output.status.code() {
            Some(0) => Ok(Some(String::from_utf8_lossy(&output.stdout).trim().to_string())),
            Some(MAC_MISSING) => Ok(None),
            _ => Err(failed("security", &output)),
        }
    }

    fn raw_set(&self, account: &str, text: &str) -> Result<(), KeyringError> {
        let account = self.checked(account)?;
        let line = format!(
            "add-generic-password -U -s \"{service}\" -a \"{account}\" -l \"{service}:{account}\" -w {text}\n",
            service = self.service
        );
        if line.len() >= MAC_LINE {
            return Err(KeyringError::Failed("a keychain line security would cut".into()));
        }
        let output = run(&self.program, &["-i"], Some(&line))?;
        if !output.status.success() || !output.stderr.is_empty() {
            return Err(failed("security", &output));
        }
        Ok(())
    }

    fn raw_delete(&self, account: &str) -> Result<(), KeyringError> {
        let account = self.checked(account)?;
        let output = run(&self.program, &["delete-generic-password", "-s", &self.service, "-a", account], None)?;
        match output.status.code() {
            Some(0) | Some(MAC_MISSING) => Ok(()),
            _ => Err(failed("security", &output)),
        }
    }
}

impl Backend for MacBackend {
    fn get(&self, account: &str) -> Result<Option<String>, KeyringError> {
        chunked_get(self, account)
    }
    fn set(&self, account: &str, value: &str) -> Result<(), KeyringError> {
        chunked_set(self, account, value)
    }
    fn delete(&self, account: &str) -> Result<(), KeyringError> {
        chunked_delete(self, account)
    }
}

/// The Secret Service (gnome-keyring on Omarchy) through `secret-tool`, which reads a stored secret from stdin. It cuts a
/// secret at 8192 bytes and still exits 0 (only a word on stderr says so, checked 2026-10-21), so values go in pieces as
/// on macOS, and a store that says anything on stderr failed.
#[derive(Debug)]
pub struct LinuxBackend {
    pub program: PathBuf,
    pub service: String,
}

impl Raw for LinuxBackend {
    fn raw_get(&self, account: &str) -> Result<Option<String>, KeyringError> {
        let output = run(&self.program, &["lookup", "service", &self.service, "account", account], None)?;
        if output.status.success() {
            return Ok(Some(String::from_utf8_lossy(&output.stdout).trim().to_string()));
        }
        // `lookup` exits 1 for "not there" without a word; anything on stderr is the Secret Service failing.
        if output.status.code() == Some(1) && output.stderr.is_empty() {
            return Ok(None);
        }
        Err(failed("secret-tool", &output))
    }

    fn raw_set(&self, account: &str, text: &str) -> Result<(), KeyringError> {
        let label = format!("Aria: {account}");
        let args = ["store", "--label", &label, "service", &self.service, "account", account];
        let output = run(&self.program, &args, Some(text))?;
        if !output.status.success() || !output.stderr.is_empty() {
            return Err(failed("secret-tool", &output));
        }
        Ok(())
    }

    fn raw_delete(&self, account: &str) -> Result<(), KeyringError> {
        let output = run(&self.program, &["clear", "service", &self.service, "account", account], None)?;
        if output.status.success() || (output.status.code() == Some(1) && output.stderr.is_empty()) {
            return Ok(());
        }
        Err(failed("secret-tool", &output))
    }
}

impl Backend for LinuxBackend {
    fn get(&self, account: &str) -> Result<Option<String>, KeyringError> {
        chunked_get(self, account)
    }
    fn set(&self, account: &str, value: &str) -> Result<(), KeyringError> {
        chunked_set(self, account, value)
    }
    fn delete(&self, account: &str) -> Result<(), KeyringError> {
        chunked_delete(self, account)
    }
}

/// The platform's keyring for `service`, or why there is none.
pub fn system(service: &str) -> Result<Arc<dyn Backend>, KeyringError> {
    #[cfg(target_os = "macos")]
    {
        let program = PathBuf::from("/usr/bin/security");
        if !program.exists() {
            return Err(KeyringError::Unavailable("/usr/bin/security is missing".into()));
        }
        Ok(Arc::new(MacBackend { program, service: service.to_string() }))
    }
    #[cfg(target_os = "linux")]
    {
        let program = std::env::var_os("PATH")
            .into_iter()
            .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
            .map(|dir| dir.join("secret-tool"))
            .find(|candidate| candidate.is_file())
            .ok_or_else(|| KeyringError::Unavailable("secret-tool is not installed (the libsecret package)".into()))?;
        let backend = LinuxBackend { program, service: service.to_string() };
        // A lookup of nothing tells a working Secret Service (exit 0, or 1 in silence) from a missing one.
        match backend.get("aria-probe") {
            Ok(_) => Ok(Arc::new(backend)),
            Err(KeyringError::Failed(why)) | Err(KeyringError::Unavailable(why)) => Err(KeyringError::Unavailable(why)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;

    /// A stand-in for `security` or `secret-tool`: logs its arguments and its stdin, then runs `body`.
    fn fake(dir: &Path, body: &str) -> PathBuf {
        let program = dir.join("fake");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{dir}/argv.log'\ncat >> '{dir}/stdin.log'\n{body}\n",
            dir = dir.display()
        );
        std::fs::write(&program, script).unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        program
    }

    fn log(dir: &Path, name: &str) -> String {
        std::fs::read_to_string(dir.join(name)).unwrap_or_default()
    }

    fn b64(value: &str) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(value)
    }

    #[test]
    fn memory_round_trip() {
        let keyring = MemoryBackend::default();
        keyring.set("key:x", "a b'c\n").unwrap();
        assert_eq!(keyring.get("key:x").unwrap().as_deref(), Some("a b'c\n"));
        keyring.delete("key:x").unwrap();
        assert_eq!(keyring.get("key:x").unwrap(), None);
    }

    #[test]
    fn service_for_the_default_home_is_aria() {
        assert_eq!(service_for(Path::new("/h/.aria"), Path::new("/h")), "aria");
    }

    #[test]
    fn service_for_another_home_names_it() {
        assert_eq!(service_for(Path::new("/tmp/x"), Path::new("/h")), "aria:/tmp/x");
    }

    #[test]
    fn mac_writes_through_stdin_never_argv() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = MacBackend { program: fake(dir.path(), "exit 0"), service: "aria:/t".into() };
        keyring.set("key:x", "s3cr3t'").unwrap();
        let argv = log(dir.path(), "argv.log");
        assert_eq!(argv.lines().last(), Some("-i"), "the write is `security -i` (a read comes first, for old pieces)");
        assert!(!argv.contains("s3cr3t"));
        let stdin = log(dir.path(), "stdin.log");
        assert!(stdin.contains(r#"add-generic-password -U -s "aria:/t" -a "key:x""#), "{stdin}");
        // No access list: a `-T` on an item that already exists asks macOS to change its list, which raises a password
        // dialog on every update (found live, 2026-10-21); the creator (`security`) is trusted without it.
        assert!(!stdin.contains("-T"), "{stdin}");
        assert!(stdin.contains(&format!("-w {}", b64("s3cr3t'"))), "{stdin}");
        assert!(!stdin.contains("s3cr3t'"));
    }

    #[test]
    fn mac_get_decodes_and_44_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!("case \"$*\" in *'-a key:x -w'*) echo {} ;; *) exit 44 ;; esac", b64("v"));
        let keyring = MacBackend { program: fake(dir.path(), &body), service: "aria".into() };
        assert_eq!(keyring.get("key:x").unwrap().as_deref(), Some("v"));
        assert_eq!(keyring.get("key:y").unwrap(), None);
        let broken = MacBackend { program: fake(dir.path(), "exit 51"), service: "aria".into() };
        assert!(matches!(broken.get("key:x"), Err(KeyringError::Failed(_))));
    }

    #[test]
    fn mac_delete_of_a_missing_item_is_fine() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = MacBackend { program: fake(dir.path(), "exit 44"), service: "aria".into() };
        keyring.delete("key:x").unwrap();
    }

    #[test]
    fn a_service_or_account_with_a_double_quote_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = MacBackend { program: fake(dir.path(), "exit 0"), service: "aria:/a\"b".into() };
        assert!(matches!(keyring.set("key:x", "v"), Err(KeyringError::Failed(_))));
    }

    /// A keychain in memory that refuses a stored text as long as the `security -i` line `security` would cut.
    #[derive(Debug, Default)]
    struct ShortLines {
        items: Mutex<BTreeMap<String, String>>,
    }

    impl Raw for ShortLines {
        fn raw_get(&self, account: &str) -> Result<Option<String>, KeyringError> {
            Ok(self.items.lock().unwrap().get(account).cloned())
        }
        fn raw_set(&self, account: &str, text: &str) -> Result<(), KeyringError> {
            assert!(text.len() + account.len() + 120 < MAC_LINE, "a line of {} would be cut", text.len());
            self.items.lock().unwrap().insert(account.to_string(), text.to_string());
            Ok(())
        }
        fn raw_delete(&self, account: &str) -> Result<(), KeyringError> {
            self.items.lock().unwrap().remove(account);
            Ok(())
        }
    }

    #[test]
    fn a_large_value_is_stored_in_pieces_and_read_whole() {
        let keyring = ShortLines::default();
        let big: String = (0..20_000).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
        chunked_set(&keyring, "file:mcp-auth.json", &big).unwrap();
        assert!(keyring.items.lock().unwrap().len() > 2, "stored in pieces");
        assert_eq!(chunked_get(&keyring, "file:mcp-auth.json").unwrap().as_deref(), Some(big.as_str()));
        // Smaller again: one item, and the old pieces gone.
        chunked_set(&keyring, "file:mcp-auth.json", "small").unwrap();
        assert_eq!(keyring.items.lock().unwrap().len(), 1);
        assert_eq!(chunked_get(&keyring, "file:mcp-auth.json").unwrap().as_deref(), Some("small"));
        chunked_set(&keyring, "file:mcp-auth.json", &big).unwrap();
        chunked_delete(&keyring, "file:mcp-auth.json").unwrap();
        assert!(keyring.items.lock().unwrap().is_empty());
    }

    #[test]
    fn a_missing_piece_is_an_error_not_a_short_value() {
        let keyring = ShortLines::default();
        let big = "x".repeat(10_000);
        chunked_set(&keyring, "k", &big).unwrap();
        keyring.items.lock().unwrap().remove("k#1");
        assert!(matches!(chunked_get(&keyring, "k"), Err(KeyringError::Failed(_))));
    }

    #[test]
    fn a_name_with_a_control_character_or_backslash_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = MacBackend { program: fake(dir.path(), "exit 0"), service: "aria".into() };
        for account in ["mcp:a\ndelete-keychain x:env:K", "mcp:a\\b:env:K", "mcp:\tx:env:K"] {
            assert!(matches!(keyring.set(account, "v"), Err(KeyringError::Failed(_))), "{account:?}");
        }
        assert_eq!(log(dir.path(), "argv.log"), "", "security never ran");
    }

    #[test]
    fn linux_store_reads_stdin_and_lookup_decodes() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = LinuxBackend { program: fake(dir.path(), "exit 0"), service: "aria".into() };
        keyring.set("key:x", "s3cr3t").unwrap();
        let argv = log(dir.path(), "argv.log");
        assert_eq!(
            argv.lines().last(),
            Some("store --label Aria: key:x service aria account key:x"),
            "the write is a store (a read comes first, for old pieces)"
        );
        assert_eq!(log(dir.path(), "stdin.log"), b64("s3cr3t"));

        let found = LinuxBackend { program: fake(dir.path(), &format!("printf {}", b64("v"))), service: "aria".into() };
        assert_eq!(found.get("key:x").unwrap().as_deref(), Some("v"));
        let missing = LinuxBackend { program: fake(dir.path(), "exit 1"), service: "aria".into() };
        assert_eq!(missing.get("key:x").unwrap(), None);
        let broken =
            LinuxBackend { program: fake(dir.path(), "echo 'no Secret Service' >&2; exit 1"), service: "aria".into() };
        assert!(matches!(broken.get("key:x"), Err(KeyringError::Failed(_))));
    }

    #[test]
    fn linux_store_that_cuts_the_secret_is_a_failure() {
        let dir = tempfile::tempdir().unwrap();
        let body = "case \"$1\" in store) echo 'secret-tool: password is too long' >&2 ;; *) exit 1 ;; esac";
        let keyring = LinuxBackend { program: fake(dir.path(), body), service: "aria".into() };
        assert!(matches!(keyring.set("key:x", "v"), Err(KeyringError::Failed(_))));
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "uses the real Secret Service (its own service, cleared at the end)"]
    fn linux_live_round_trip() {
        let Ok(program) = which("secret-tool") else { panic!("secret-tool is not installed") };
        let keyring = LinuxBackend { program, service: format!("aria-test-{}", std::process::id()) };
        assert_eq!(keyring.get("key:live").unwrap(), None);
        keyring.set("key:live", "a b'c\"d\n{\"json\":1}").unwrap();
        assert_eq!(keyring.get("key:live").unwrap().as_deref(), Some("a b'c\"d\n{\"json\":1}"));
        keyring.set("key:live", "second").unwrap();
        assert_eq!(keyring.get("key:live").unwrap().as_deref(), Some("second"));
        // Longer than the 8192 bytes secret-tool keeps: in pieces, read back whole, every piece removed.
        let big: String = (0..20_000).map(|i| char::from(b'!' + (i % 90) as u8)).collect();
        keyring.set("key:live", &big).unwrap();
        assert_eq!(keyring.get("key:live").unwrap().as_deref(), Some(big.as_str()));
        keyring.delete("key:live").unwrap();
        assert_eq!(keyring.raw_get("key:live#0").unwrap(), None);
        keyring.delete("key:live").unwrap();
        assert_eq!(keyring.get("key:live").unwrap(), None);
    }

    #[cfg(target_os = "linux")]
    fn which(name: &str) -> Result<PathBuf, ()> {
        std::env::split_paths(&std::env::var_os("PATH").ok_or(())?)
            .map(|d| d.join(name))
            .find(|p| p.is_file())
            .ok_or(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "uses the real login keychain (its own service, deleted at the end)"]
    fn mac_live_round_trip() {
        let keyring = MacBackend {
            program: PathBuf::from("/usr/bin/security"),
            service: format!("aria-test-{}", std::process::id()),
        };
        keyring.set("key:live", "a b'c\"d\n{\"json\":1}").unwrap();
        assert_eq!(keyring.get("key:live").unwrap().as_deref(), Some("a b'c\"d\n{\"json\":1}"));
        // An update of an item that exists: it must not raise a dialog (the test would hang).
        keyring.set("key:live", "second").unwrap();
        assert_eq!(keyring.get("key:live").unwrap().as_deref(), Some("second"));
        // A value longer than one `security -i` line: kept in pieces, read back whole, and every piece removed.
        let big: String = (0..10_000).map(|i| char::from(b'!' + (i % 90) as u8)).collect();
        keyring.set("key:live", &big).unwrap();
        assert_eq!(keyring.get("key:live").unwrap().as_deref(), Some(big.as_str()));
        keyring.delete("key:live").unwrap();
        assert_eq!(keyring.raw_get("key:live#0").unwrap(), None);
        keyring.delete("key:live").unwrap();
        assert_eq!(keyring.get("key:live").unwrap(), None);
    }
}
