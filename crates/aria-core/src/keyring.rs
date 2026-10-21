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

/// The login keychain through `security`. A write goes through `security -i` with the command on stdin, so the secret is
/// never in a process's arguments. Items carry no access list of their own: their creator, `security`, reads, updates and
/// deletes them without a dialog, while `-T` on an update would ask macOS to change the list and raise one each time.
#[derive(Debug)]
pub struct MacBackend {
    pub program: PathBuf,
    pub service: String,
}

/// `security`'s exit code for an item that is not there.
const MAC_MISSING: i32 = 44;

impl MacBackend {
    /// `security -i` reads double-quoted words; a service or account with a quote in it cannot be written safely.
    fn quoted(&self, account: &str) -> Result<(String, String), KeyringError> {
        if self.service.contains('"') || account.contains('"') {
            return Err(KeyringError::Failed("a service or account with a double quote".into()));
        }
        Ok((format!("\"{}\"", self.service), format!("\"{account}\"")))
    }
}

impl Backend for MacBackend {
    fn get(&self, account: &str) -> Result<Option<String>, KeyringError> {
        let output = run(&self.program, &["find-generic-password", "-s", &self.service, "-a", account, "-w"], None)?;
        match output.status.code() {
            Some(0) => decode(&String::from_utf8_lossy(&output.stdout)).map(Some),
            Some(MAC_MISSING) => Ok(None),
            _ => Err(failed("security", &output)),
        }
    }

    fn set(&self, account: &str, value: &str) -> Result<(), KeyringError> {
        let (service, quoted) = self.quoted(account)?;
        let label = format!("\"{}:{account}\"", self.service);
        let line = format!(
            "add-generic-password -U -s {service} -a {quoted} -l {label} -w {}\n",
            encode(value)
        );
        let output = run(&self.program, &["-i"], Some(&line))?;
        if !output.status.success() || !output.stderr.is_empty() {
            return Err(failed("security", &output));
        }
        Ok(())
    }

    fn delete(&self, account: &str) -> Result<(), KeyringError> {
        let output = run(&self.program, &["delete-generic-password", "-s", &self.service, "-a", account], None)?;
        match output.status.code() {
            Some(0) | Some(MAC_MISSING) => Ok(()),
            _ => Err(failed("security", &output)),
        }
    }
}

/// The Secret Service (gnome-keyring on Omarchy) through `secret-tool`, which reads a stored secret from stdin.
#[derive(Debug)]
pub struct LinuxBackend {
    pub program: PathBuf,
    pub service: String,
}

impl Backend for LinuxBackend {
    fn get(&self, account: &str) -> Result<Option<String>, KeyringError> {
        let output = run(&self.program, &["lookup", "service", &self.service, "account", account], None)?;
        if output.status.success() {
            return decode(&String::from_utf8_lossy(&output.stdout)).map(Some);
        }
        // `lookup` exits 1 for "not there" without a word; anything on stderr is the Secret Service failing.
        if output.status.code() == Some(1) && output.stderr.is_empty() {
            return Ok(None);
        }
        Err(failed("secret-tool", &output))
    }

    fn set(&self, account: &str, value: &str) -> Result<(), KeyringError> {
        let label = format!("Aria: {account}");
        let args = ["store", "--label", &label, "service", &self.service, "account", account];
        let output = run(&self.program, &args, Some(&encode(value)))?;
        if !output.status.success() {
            return Err(failed("secret-tool", &output));
        }
        Ok(())
    }

    fn delete(&self, account: &str) -> Result<(), KeyringError> {
        let output = run(&self.program, &["clear", "service", &self.service, "account", account], None)?;
        if output.status.success() || (output.status.code() == Some(1) && output.stderr.is_empty()) {
            return Ok(());
        }
        Err(failed("secret-tool", &output))
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
        assert_eq!(argv.trim(), "-i");
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

    #[test]
    fn linux_store_reads_stdin_and_lookup_decodes() {
        let dir = tempfile::tempdir().unwrap();
        let keyring = LinuxBackend { program: fake(dir.path(), "exit 0"), service: "aria".into() };
        keyring.set("key:x", "s3cr3t").unwrap();
        assert_eq!(log(dir.path(), "argv.log").trim(), "store --label Aria: key:x service aria account key:x");
        assert_eq!(log(dir.path(), "stdin.log"), b64("s3cr3t"));

        let found = LinuxBackend { program: fake(dir.path(), &format!("printf {}", b64("v"))), service: "aria".into() };
        assert_eq!(found.get("key:x").unwrap().as_deref(), Some("v"));
        let missing = LinuxBackend { program: fake(dir.path(), "exit 1"), service: "aria".into() };
        assert_eq!(missing.get("key:x").unwrap(), None);
        let broken =
            LinuxBackend { program: fake(dir.path(), "echo 'no Secret Service' >&2; exit 1"), service: "aria".into() };
        assert!(matches!(broken.get("key:x"), Err(KeyringError::Failed(_))));
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
        keyring.delete("key:live").unwrap();
        assert_eq!(keyring.get("key:live").unwrap(), None);
    }
}
