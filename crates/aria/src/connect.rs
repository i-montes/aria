//! `aria connect [user@]host`: the terminal on another computer's service. The transport is ssh and nothing else:
//! `ssh <host> ~/.local/bin/aria client-stdio` is the bridge (`stdio_bridge`) on that computer, and its stdin and stdout
//! are the link. Nothing of Aria listens on a port.

use crate::term::link::Link;
use aria_core::json_file::write_bytes_atomic;
use aria_core::paths::AriaPaths;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// Where the bridge lives on the other computer (a login shell expands the `~`).
const REMOTE_ARIA: &str = "~/.local/bin/aria";

#[derive(Debug)]
pub enum ConnectError {
    /// A host that starts with `-` would be an option of ssh (`-oProxyCommand=...`).
    OptionLikeHost,
    /// There is no ssh to run.
    NoSsh,
    /// ssh ran, but the hub did not open the link (see `LinkError`), or the link broke.
    Failed(io::Error),
}

/// The program that runs ssh commands: the one `ARIA_SSH` names, or `ssh`.
pub(crate) fn ssh_program() -> OsString {
    ssh_program_from(std::env::var_os("ARIA_SSH"))
}

/// `ssh_program` for a given value of `ARIA_SSH` (an empty one is none).
fn ssh_program_from(env: Option<OsString>) -> OsString {
    env.filter(|program| !program.is_empty()).unwrap_or_else(|| "ssh".into())
}

/// A host that starts with `-` would be an option of ssh (`-oProxyCommand=...`): every ssh the program runs checks it first.
pub(crate) fn check_host(host: &str) -> Result<(), ConnectError> {
    if host.starts_with('-') {
        return Err(ConnectError::OptionLikeHost);
    }
    Ok(())
}

/// The hubs `aria hub init` set up from this computer (`~/.aria/hubs.json`): which one `aria connect` opens when it is
/// given no host, and the key each one was linked with, so that nobody has to write an ssh entry for it.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hubs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    #[serde(default)]
    pub hubs: BTreeMap<String, HubEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HubEntry {
    /// The private key `aria hub init` linked this computer with.
    pub key: String,
}

impl Hubs {
    pub fn path(root: &Path) -> PathBuf {
        root.join("hubs.json")
    }

    /// The file's content; no file is no hubs. A file that does not parse is an error, so that `remember` never writes over
    /// something the owner edited.
    pub fn load(root: &Path) -> io::Result<Hubs> {
        match std::fs::read_to_string(Hubs::path(root)) {
            Ok(text) => serde_json::from_str(&text)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, format!("hubs.json: {error}"))),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Hubs::default()),
            Err(error) => Err(error),
        }
    }

    /// Adds (or updates) a hub. The first one becomes the default, and so does one when the default names a hub that is
    /// no longer listed; an existing default is otherwise kept.
    pub fn remember(&mut self, host: &str, key: &Path) {
        self.hubs.insert(host.to_string(), HubEntry { key: key.to_string_lossy().into_owned() });
        if self.default.as_ref().is_none_or(|current| !self.hubs.contains_key(current)) {
            self.default = Some(host.to_string());
        }
    }

    /// Written whole and atomically, readable by its owner alone (it names a key file).
    pub fn save(&self, root: &Path) -> io::Result<()> {
        AriaPaths::new(root.to_path_buf()).ensure_root()?;
        let mut text = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        text.push('\n');
        write_bytes_atomic(&Hubs::path(root), text.as_bytes(), 0o600)
    }

    pub fn key_for(&self, host: &str) -> Option<&Path> {
        self.hubs.get(host).map(|entry| Path::new(entry.key.as_str()))
    }
}

/// The key `hub init` linked this computer to `host` with, when it is still there to use.
fn remembered_key(host: &str) -> Option<PathBuf> {
    let hubs = Hubs::load(&AriaPaths::from_env().root).ok()?;
    hubs.key_for(host).filter(|key| key.is_file()).map(Path::to_path_buf)
}

/// The hub `aria connect` opens when it is given no host.
pub fn default_hub() -> Option<String> {
    let hubs = Hubs::load(&AriaPaths::from_env().root).ok()?;
    hubs.default.filter(|host| hubs.hubs.contains_key(host))
}

/// The command that runs `aria client-stdio` on `host` through `ssh`, with no key of its own (the tests' form of it).
#[cfg(test)]
fn command_for(ssh: OsString, host: &str) -> Result<Command, ConnectError> {
    command_with_key(ssh, host, None)
}

/// `command_for`, with the key the hub was linked with (and no other: an agent full of keys would be tried first, and a
/// server that allows few attempts would refuse before the right one).
fn command_with_key(ssh: OsString, host: &str, key: Option<&Path>) -> Result<Command, ConnectError> {
    check_host(host)?;
    let mut command = Command::new(ssh);
    command.args(["-T", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]);
    if let Some(key) = key {
        command.arg("-i").arg(key).args(["-o", "IdentitiesOnly=yes"]);
    }
    command.arg(host).args([REMOTE_ARIA, "client-stdio"]);
    Ok(command)
}

/// `ssh -T -o ServerAliveInterval=15 -o ServerAliveCountMax=3 [-i <key> -o IdentitiesOnly=yes] <host> ~/.local/bin/aria
/// client-stdio`.
pub fn ssh_command(host: &str) -> Result<Command, ConnectError> {
    command_with_key(ssh_program(), host, remembered_key(host).as_deref())
}

/// Tries to open the hub of `host` as `aria connect` does (ssh, then the hello), and closes it at once: what `hub init`
/// does last, so that "ready" means a connection worked.
pub fn try_open(host: &str, within: Duration) -> Result<(), ConnectError> {
    try_with(&mut ssh_command(host)?, within)
}

fn try_with(command: &mut Command, within: Duration) -> Result<(), ConnectError> {
    Link::spawn_within(command, within).map(drop).map_err(|error| match error.kind() {
        io::ErrorKind::NotFound => ConnectError::NoSsh,
        _ => ConnectError::Failed(error),
    })
}

/// A link to the hub of `host`: ssh runs, the hellos are exchanged. ssh's own errors go to this terminal's stderr.
pub fn open(host: &str) -> Result<Link, ConnectError> {
    let mut command = ssh_command(host)?;
    Link::spawn(&mut command).map_err(|error| match error.kind() {
        io::ErrorKind::NotFound => ConnectError::NoSsh,
        _ => ConnectError::Failed(error),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn a_host_that_looks_like_an_option_is_refused() {
        assert!(matches!(ssh_command("-oProxyCommand=x"), Err(ConnectError::OptionLikeHost)));
    }

    #[test]
    fn the_command_is_ssh_with_keepalive_and_the_remote_binary() {
        let command = command_for(OsString::from("ssh"), "ana@hub.example").unwrap();
        assert_eq!(command.get_program(), "ssh");
        let args: Vec<_> = command.get_args().collect();
        assert_eq!(
            args,
            [
                "-T",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "ana@hub.example",
                "~/.local/bin/aria",
                "client-stdio"
            ]
        );
    }

    #[test]
    fn a_remembered_key_is_the_only_one_ssh_offers() {
        let command =
            command_with_key(OsString::from("ssh"), "ana@hub.example", Some(Path::new("/home/ana/.ssh/aria-hub")))
                .unwrap();
        let args: Vec<_> = command.get_args().collect();
        assert_eq!(
            args,
            [
                "-T",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "-i",
                "/home/ana/.ssh/aria-hub",
                "-o",
                "IdentitiesOnly=yes",
                "ana@hub.example",
                "~/.local/bin/aria",
                "client-stdio"
            ]
        );
        assert!(
            matches!(
                command_with_key(OsString::from("ssh"), "-oProxyCommand=x", Some(Path::new("/k"))),
                Err(ConnectError::OptionLikeHost)
            ),
            "a key does not make an option-like host acceptable"
        );
    }

    fn scratch_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("aria-hubs-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn the_first_hub_is_the_default_and_a_later_one_does_not_take_it_over() {
        let mut hubs = Hubs::default();
        hubs.remember("ana@one", Path::new("/home/ana/.ssh/aria-hub"));
        hubs.remember("ana@two", Path::new("/home/ana/.ssh/aria-two"));
        assert_eq!(hubs.default.as_deref(), Some("ana@one"));
        assert_eq!(hubs.key_for("ana@two"), Some(Path::new("/home/ana/.ssh/aria-two")));
        assert_eq!(hubs.key_for("ana@none"), None);
    }

    #[test]
    fn a_default_that_is_no_longer_listed_is_replaced() {
        let mut hubs = Hubs { default: Some("ana@gone".into()), hubs: BTreeMap::new() };
        hubs.remember("ana@srv", Path::new("/k"));
        assert_eq!(hubs.default.as_deref(), Some("ana@srv"));
    }

    #[test]
    fn hubs_are_written_privately_and_read_back() {
        use std::os::unix::fs::PermissionsExt;
        let root = scratch_root("roundtrip");
        assert_eq!(Hubs::load(&root).unwrap(), Hubs::default(), "no file is no hubs");
        let mut hubs = Hubs::default();
        hubs.remember("ana@srv", Path::new("/home/ana/.ssh/aria-hub"));
        hubs.save(&root).unwrap();
        let mode = std::fs::metadata(Hubs::path(&root)).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let text = std::fs::read_to_string(Hubs::path(&root)).unwrap();
        assert!(
            text.contains("\"default\": \"ana@srv\"") && text.contains("\"key\": \"/home/ana/.ssh/aria-hub\""),
            "{text}"
        );
        assert_eq!(Hubs::load(&root).unwrap(), hubs);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_hubs_file_that_does_not_parse_is_an_error_not_an_empty_list() {
        let root = scratch_root("broken");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(Hubs::path(&root), "{ not json").unwrap();
        assert!(Hubs::load(&root).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_connection_that_never_says_hello_fails() {
        // `true` exits at once: no hello comes.
        assert!(try_with(&mut Command::new("true"), Duration::from_secs(5)).is_err());
        assert!(matches!(
            try_with(&mut Command::new("/nonexistent/aria-test-ssh"), Duration::from_secs(5)),
            Err(ConnectError::NoSsh)
        ));
    }

    #[test]
    #[allow(non_snake_case)]
    fn ARIA_SSH_names_another_program() {
        assert_eq!(ssh_program_from(None), OsString::from("ssh"));
        assert_eq!(ssh_program_from(Some(OsString::new())), OsString::from("ssh"));
        assert_eq!(ssh_program_from(Some(OsString::from("/opt/bin/fake-ssh"))), OsString::from("/opt/bin/fake-ssh"));
        let command = command_for(ssh_program_from(Some(OsString::from("fake-ssh"))), "hub.example").unwrap();
        assert_eq!(command.get_program(), "fake-ssh");
    }
}
