//! What sets a hub up (docs/superpowers/specs/2026-10-23-hub-and-machines-design.md, "First setup without a page"):
//! the bundle of settings and credentials the owner's computer sends, and the import that writes it on the hub.
//!
//! The pieces are small and pure; `aria hub init` (below) orchestrates them over ssh.

use crate::keyring_cli::secret_value;
use aria_core::json_file::{merge_preserving, read_json, write_bytes_atomic};
use aria_core::paths::AriaPaths;
use aria_core::vault::{self, Vault, SEALED};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::io::{self, BufRead, Read, Write};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, Stdio};

/// The only files a bundle carries, in the order they are written.
pub const BUNDLE_FILES: [&str; 4] = ["settings.json", "auth.json", "mcp.json", "mcp-auth.json"];

/// The most one line of a bundle may hold.
pub const MAX_BUNDLE_BYTES: u64 = 16 * 1024 * 1024;

/// Settings and credentials as plain files: `{"files":{"settings.json": "<content>", …}}` on one line.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub files: BTreeMap<String, String>,
    /// What the export could not settle (a credential file left out because the keyring could not be opened to look for
    /// it). Never sent: `aria hub init` prints them before anything is sent. Never holds a secret value.
    #[serde(skip)]
    pub warnings: Vec<String>,
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
    let mut warnings = Vec::new();

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
        let found = if sealed { keys.sealed(name, &mut warnings)? } else { read_object(&paths.root.join(name))? };
        let Some(mut file) = found else { continue };
        if *name != "mcp-auth.json" {
            keys.plain(&mut file).map_err(|why| format!("{name}: {why}"))?;
        }
        files.insert(name.to_string(), render(&file));
    }
    Ok(Bundle { files, warnings })
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
    fn sealed(&self, name: &str, warnings: &mut Vec<String>) -> Result<Option<Value>, String> {
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
        let vault = match self.vault() {
            Ok(vault) => vault,
            Err(why) => {
                warnings.push(format!("{name}: no copy found and the keyring could not be opened ({why})"));
                return Ok(None);
            }
        };
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
/// 600, each atomically, the first one it replaces kept as `<name>.bak`: never overwritten by a later import), the names in [`BUNDLE_FILES`] order. A bundle that is
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
            // The first copy is the hub's own: a second import never replaces it with what the first one wrote.
            Ok(old) => {
                let backup = paths.root.join(format!("{name}.bak"));
                if !backup.exists() {
                    write_bytes_atomic(&backup, &old, 0o600).map_err(failed)?;
                }
            }
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

// ---------------------------------------------------------------------------------------------------------------------
// `aria hub init`: the owner's computer sets a server up as a hub over ssh. The probes before the plan only read; every
// write waits for a yes. Every remote command is one argument to ssh after the host.
// ---------------------------------------------------------------------------------------------------------------------

const USAGE: &str =
    "usage: aria hub init <user@host> [--binary <path>] [--projects <dir>] [--client-key <pubkey file>] \
     [--settings] [--credentials] [--allow-no-sandbox] [--yes]";

/// What `aria hub init` uploads from the checkout: the installer, the service unit and the pins the installer reads
/// (`install/install.sh` reads `../<pin>`, so the tree keeps its shape under `~/.aria/hub-src`).
const UPLOAD_FILES: [&str; 7] = [
    "install/install.sh",
    "install/aria-serve.service",
    "pi-version",
    "pi-packages",
    "engram-version",
    "agent-browser-version",
    "node-version",
];

/// Where the upload lands on the hub (the remote shell expands the `~`) and what the installer is run with.
const HUB_SRC: &str = "~/.aria/hub-src";
const REMOTE_UPLOAD: &str = "umask 077 && mkdir -p ~/.aria/hub-src && tar -xf - -C ~/.aria/hub-src";
const INSTALL_COMMAND: &str = "sh ~/.aria/hub-src/install/install.sh --headless --local ~/.aria/hub-src/aria";
const REMOTE_IMPORT: &str = "~/.local/bin/aria hub-import";
const SERVICE_UNIT: &str = "aria-serve.service";
const RESTART_COMMAND: &str = "systemctl --user restart aria-serve.service";
const IS_ACTIVE_COMMAND: &str = "systemctl --user is-active aria-serve.service";
/// Whether the hub's user keeps its user manager (and Aria) running when nobody is logged in.
const REMOTE_LINGER: &str = r#"loginctl show-user "$USER" -p Linger"#;
/// Which of the four bundle files the hub has: a `test -f` on each, the names of those that exist.
const REMOTE_EXISTING: &str =
    "for f in settings.json auth.json mcp.json mcp-auth.json; do test -f ~/.aria/$f && echo $f; done; true";
const SERVICE_KEEPS_RUNNING: &str =
    "the service keeps running: it takes the new binary when idle and the settings within a second";
const REMOTE_VERSION: &str = "~/.local/bin/aria --version";
/// What the sandbox extension's probe runs (`extensions/sandbox/probe.ts`), which is the real call's mounts: a server that
/// cannot mount `/proc` or `/dev` (a container) fails here and not at the first command.
const REMOTE_BWRAP: &str = "bwrap --unshare-user --ro-bind / / --dev /dev --proc /proc true";
const REMOTE_SSHD_CONFIG: &str = "cat /etc/ssh/sshd_config.d/*.conf /etc/ssh/sshd_config 2>/dev/null";

/// `systemctl --user` needs `XDG_RUNTIME_DIR` (and the user bus beside it), which a non-login ssh command lacks: set to what
/// a login would have given when they are unset and the user's runtime directory exists. In front of every remote command
/// that talks to the user manager.
pub fn user_env(command: &str) -> String {
    format!(
        "if [ -z \"$XDG_RUNTIME_DIR\" ] && [ -d \"/run/user/$(id -u)\" ]; then XDG_RUNTIME_DIR=/run/user/$(id -u); export XDG_RUNTIME_DIR; fi; \
         if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ] && [ -n \"$XDG_RUNTIME_DIR\" ]; then DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus; export DBUS_SESSION_BUS_ADDRESS; fi; {command}"
    )
}

/// What to do when `systemctl --user` fails on the hub, which is nearly always its bus or linger.
fn systemctl_hint(user: &str) -> String {
    format!(
        "If it cannot connect to the bus, the hub needs dbus-user-session (sudo apt-get install -y dbus-user-session) and \
         linger for its user (sudo loginctl enable-linger {user}) so that the user manager runs without a login"
    )
}

/// Reads one validated key line from stdin and adds it to `authorized_keys` as the restricted line, once. sshd uses the
/// first line a key matches, so the blob is looked for in the lines that are not comments: none holds it, the line is
/// added (after a newline, when the file lacks one at its end); the first one starts with `restrict,command=`, it is
/// already allowed; any other is the same key with no restriction, which is reported (exit 3) and left for its owner.
/// Nothing is interpolated: the key arrives on stdin and the home is the remote shell's own `$HOME`. Plain POSIX, with
/// no single quote in it.
const REMOTE_ALLOW_KEY: &str = concat!(
    r#"umask 077; f="$HOME/.ssh/authorized_keys"; mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh" "#,
    r#"&& touch "$f" && chmod 600 "$f" && IFS= read -r key "#,
    r#"&& blob=$(printf "%s\n" "$key" | cut -d" " -f2) "#,
    r#"&& first=$(grep -v "^[[:space:]]*#" "$f" | grep -F " $blob" | head -n 1) "#,
    r#"&& case "$first" in "") if [ -s "$f" ] && [ -n "$(tail -c 1 "$f")" ]; then echo >> "$f"; fi; "#,
    r#"printf "restrict,command=\"%s/.local/bin/aria client-stdio\" %s\n" "$HOME" "$key" >> "$f" && echo "allowed" ;; "#,
    r#""restrict,command="*) echo "already allowed" ;; "#,
    r#"*) echo "unrestricted"; exit 3 ;; esac"#,
);

/// The exit status `REMOTE_ALLOW_KEY` ends with when the key is there without the restriction.
const KEY_UNRESTRICTED: i32 = 3;

/// The key types a bare public key line starts with.
const KEY_TYPES: [&str; 7] = [
    "ssh-ed25519",
    "ssh-rsa",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
    "sk-ssh-ed25519@openssh.com",
    "sk-ecdsa-sha2-nistp256@openssh.com",
];

/// The arguments as typed.
#[derive(Debug, Clone, PartialEq)]
pub struct RawInit {
    pub host: String,
    pub binary: Option<PathBuf>,
    pub projects: Option<String>,
    pub client_key: Option<PathBuf>,
    /// Install the client key even when the sandbox does not work on the hub.
    pub allow_no_sandbox: bool,
    /// Replace the hub's `settings.json` with this computer's (otherwise a hub that has one keeps it).
    pub settings: bool,
    /// Replace the hub's credentials (`auth.json`, `mcp.json`, `mcp-auth.json`) with this computer's.
    pub credentials: bool,
    pub yes: bool,
}

/// The arguments once the binary is found and the key is read and checked.
#[derive(Debug, Clone, PartialEq)]
pub struct InitArgs {
    pub host: String,
    pub binary: PathBuf,
    pub projects: Option<String>,
    /// The bare public key line, validated.
    pub client_key: Option<String>,
    /// The file it was read from.
    pub client_key_file: Option<PathBuf>,
    pub allow_no_sandbox: bool,
    pub settings: bool,
    pub credentials: bool,
    /// When the binary file was last written (UTC, to the minute): a stale cross-build is easy to ship.
    pub binary_built: Option<String>,
    pub yes: bool,
}

/// What the read-only probes learned about the hub.
#[derive(Debug, Clone, PartialEq)]
pub struct Probe {
    /// `uname -m`: `aarch64` or `x86_64`.
    pub arch: String,
    pub os: String,
    pub home: String,
    /// The host name `ssh -G` resolved (what the owner's ssh configuration calls it).
    pub ssh_host: String,
    /// The port the owner's ssh connects to, which is the one sshd listens on.
    pub sshd_port: u16,
    /// The user the owner's ssh connects as.
    pub ssh_user: String,
    /// `aria-serve.service` was active before this run (a hub that is already working).
    pub service_active: bool,
    /// The bundle files `~/.aria` on the hub already has.
    pub existing: Vec<String>,
    pub listeners: Vec<Listener>,
    pub has_xz: bool,
    pub has_bwrap: bool,
    pub has_npm: bool,
    pub has_tailscale: bool,
    pub node_version: Option<String>,
}

/// Something that stops `aria hub init`, with the exit status it ends with.
struct Failure(i32, String);

fn usage_error(message: impl Into<String>) -> Failure {
    Failure(2, message.into())
}

fn failed(message: impl Into<String>) -> Failure {
    Failure(1, message.into())
}

/// `aria hub <command>`: only `init` for now. `args` are what follows `hub`.
pub fn hub_command(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("init") => match parse_init_args(&args[1..]) {
            Ok(raw) => init(raw),
            Err(why) => {
                eprintln!("aria hub init: {why}\n{USAGE}");
                2
            }
        },
        _ => {
            eprintln!("{USAGE}");
            2
        }
    }
}

pub fn parse_init_args(args: &[String]) -> Result<RawInit, String> {
    let mut raw = RawInit {
        host: String::new(),
        binary: None,
        projects: None,
        client_key: None,
        allow_no_sandbox: false,
        settings: false,
        credentials: false,
        yes: false,
    };
    let mut host = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let mut value = |name: &str| rest.next().cloned().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--yes" | "-y" => raw.yes = true,
            "--allow-no-sandbox" => raw.allow_no_sandbox = true,
            "--settings" => raw.settings = true,
            "--credentials" => raw.credentials = true,
            "--binary" => raw.binary = Some(PathBuf::from(value("--binary")?)),
            "--projects" => raw.projects = Some(value("--projects")?),
            "--client-key" => raw.client_key = Some(PathBuf::from(value("--client-key")?)),
            // Anything else is the host, even one that looks like an option: the host check refuses that one the way
            // `aria connect` does.
            _ if host.is_none() => host = Some(arg.clone()),
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}")),
            _ => return Err("only one host".into()),
        }
    }
    raw.host = host.ok_or("a host is needed")?;
    Ok(raw)
}

/// The machine of an ELF binary's header (`e_machine`): `aarch64` or `x86_64`, and None for anything else, a file
/// that is no ELF, or one too short to hold the field.
pub fn elf_arch(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() < 20 || &bytes[..4] != b"\x7fELF" {
        return None;
    }
    let machine = match bytes[5] {
        1 => u16::from_le_bytes([bytes[18], bytes[19]]),
        2 => u16::from_be_bytes([bytes[18], bytes[19]]),
        _ => return None,
    };
    match machine {
        0xB7 => Some("aarch64"),
        0x3E => Some("x86_64"),
        _ => None,
    }
}

/// The key line of a `--client-key` file, checked: exactly one line that is a bare public key (a key type, the base64
/// blob, an optional comment). A line with options of its own, a second line or a quote would let the file say what the
/// restriction does not, so each is refused before anything is sent.
pub fn validate_client_key(text: &str) -> Result<String, String> {
    let line = text.strip_suffix('\n').map_or(text, |line| line.strip_suffix('\r').unwrap_or(line));
    if line.trim().is_empty() {
        return Err("there is no key in it".into());
    }
    if line.chars().any(char::is_control) {
        return Err("it must be one line with no control characters".into());
    }
    if line.contains('"') {
        return Err("a key line holds no quote".into());
    }
    let (kind, rest) = line.split_once(' ').unwrap_or((line, ""));
    if !KEY_TYPES.contains(&kind) {
        return Err(format!(
            "it must be a bare public key (it starts with a key type such as ssh-ed25519), not {kind:?}"
        ));
    }
    let blob = rest.split(' ').next().unwrap_or("");
    if blob.is_empty() || !blob.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=')) {
        return Err("the key is missing or is not base64".into());
    }
    Ok(line.to_string())
}

/// The line `authorized_keys` gets: the key may only run Aria's bridge, with no terminal, forwarding or agent.
pub fn authorized_line(hub_home: &str, key: &str) -> String {
    format!("restrict,command=\"{}/.local/bin/aria client-stdio\" {key}", hub_home.trim_end_matches('/'))
}

/// True when a Node version (`v22.19.0`) is 22.19 or newer, which runs Pi.
pub fn node_is_current(version: &str) -> bool {
    let mut parts = version.trim().trim_start_matches('v').split('.');
    let number = |part: Option<&str>| part.and_then(|part| part.parse::<u32>().ok());
    match (number(parts.next()), number(parts.next())) {
        (Some(major), Some(minor)) => (major, minor) >= (22, 19),
        _ => false,
    }
}

/// The installer needs `xz` to unpack the private Node it falls back to, which it does when the hub's Node is missing,
/// has no npm or is older than 22.19.
pub fn needs_xz(probe: &Probe) -> bool {
    let node_current = probe.has_npm && probe.node_version.as_deref().is_some_and(node_is_current);
    !probe.has_xz && !node_current
}

/// A tar (ustar) of the given files `(path, content, mode)`, each with owner 0 and time 0. Paths are at most 100 bytes.
pub fn tar_archive(entries: &[(&str, &[u8], u32)]) -> Vec<u8> {
    fn octal(field: &mut [u8], value: u64) {
        let width = field.len() - 1;
        field[..width].copy_from_slice(format!("{value:0width$o}").as_bytes());
    }
    let mut out = Vec::new();
    for (name, data, mode) in entries {
        debug_assert!(name.len() <= 100, "{name}");
        let mut header = [0u8; 512];
        let name = &name.as_bytes()[..name.len().min(100)];
        header[..name.len()].copy_from_slice(name);
        octal(&mut header[100..108], u64::from(*mode));
        octal(&mut header[108..116], 0);
        octal(&mut header[116..124], 0);
        octal(&mut header[124..136], data.len() as u64);
        octal(&mut header[136..148], 0);
        header[148..156].fill(b' ');
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        let sum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
        header[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(data);
        out.resize(out.len().next_multiple_of(512), 0);
    }
    out.extend_from_slice(&[0u8; 1024]);
    out
}

/// `secs` since the epoch as `2026-10-22 14:03 UTC`.
pub fn utc_stamp(secs: u64) -> String {
    let (days, rest) = ((secs / 86_400) as i64, secs % 86_400);
    // Days to a civil date (Howard Hinnant's civil_from_days).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 { shifted_month + 3 } else { shifted_month - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02} {:02}:{:02} UTC", rest / 3600, rest % 3600 / 60)
}

/// The user of `ssh -G <host>`: its first `user` line.
pub fn ssh_g_user(output: &str) -> Option<String> {
    output.lines().find_map(|line| line.strip_prefix("user ")).map(|user| user.trim().to_string())
}

/// What the owner adds to `~/.ssh/config` so `aria connect aria-hub` uses the dedicated key (and only it): the host and
/// user from what was typed (`fallback_user` when it names none), the key's file without `.pub`.
pub fn ssh_snippet(host: &str, fallback_user: &str, key_file: &Path) -> Vec<String> {
    let (user, hostname) = host.split_once('@').unwrap_or((fallback_user, host));
    let key_file = std::path::absolute(key_file).unwrap_or_else(|_| key_file.to_path_buf());
    let key = key_file.to_string_lossy();
    let identity = key.strip_suffix(".pub").unwrap_or(&key);
    vec![
        "Add this to ~/.ssh/config on this computer, so that `aria connect aria-hub` uses this key and no other:"
            .to_string(),
        "".to_string(),
        "Host aria-hub".to_string(),
        format!("    HostName {hostname}"),
        format!("    User {user}"),
        format!("    IdentityFile {identity}"),
        "    IdentitiesOnly yes".to_string(),
        "".to_string(),
    ]
}

/// The program names `command -v` printed a path for.
pub fn parse_found(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| Path::new(line.trim()).file_name().map(|name| name.to_string_lossy().into_owned()))
        .collect()
}

/// The port and the host name of `ssh -G <host>` (the options ssh would connect with).
pub fn parse_ssh_g(output: &str) -> (Option<u16>, Option<String>) {
    let mut port = None;
    let mut hostname = None;
    for line in output.lines() {
        match line.split_once(' ') {
            Some(("port", value)) if port.is_none() => port = value.trim().parse().ok(),
            Some(("hostname", value)) if hostname.is_none() => hostname = Some(value.trim().to_string()),
            _ => {}
        }
    }
    (port, hostname)
}

/// Whether sshd's configuration turns password logins on (`Some(true)`) or off, by its first `PasswordAuthentication`
/// (the first one counts). None when it does not say or says something else.
pub fn password_auth(config: &str) -> Option<bool> {
    for line in config.lines() {
        let mut words = line.split_whitespace();
        if words.next().is_some_and(|word| word.eq_ignore_ascii_case("PasswordAuthentication")) {
            return match words.next().map(str::to_ascii_lowercase).as_deref() {
                Some("yes") => Some(true),
                Some("no") => Some(false),
                _ => None,
            };
        }
    }
    None
}

/// Only an explicit yes counts.
pub fn answer_is_yes(answer: &str) -> bool {
    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

/// The bundle files this run sends: each one the hub does not have yet, and the ones it has when the flag for them is
/// given (`--settings` for `settings.json`, `--credentials` for the other three). What a hub already has is its own: it
/// holds what its Pi rotated and what its owner changed there.
pub fn files_to_send(probe: &Probe, args: &InitArgs) -> Vec<&'static str> {
    BUNDLE_FILES
        .into_iter()
        .filter(|name| {
            let flag = if *name == "settings.json" { args.settings } else { args.credentials };
            flag || !probe.existing.iter().any(|have| have == name)
        })
        .collect()
}

/// What `aria hub init` is about to do, as the lines printed before anything runs.
pub fn plan(probe: &Probe, args: &InitArgs) -> Vec<String> {
    let projects = args.projects.clone().unwrap_or_else(|| format!("{}/Projects", probe.home.trim_end_matches('/')));
    let mut lines = vec![
        format!("Hub: {} ({}): {}, {}, home {}", args.host, probe.ssh_host, probe.os, probe.arch, probe.home),
        "Aria will do this, in order, once you say yes:".to_string(),
        format!(
            "1. Upload the Aria tree (installer, service unit, version pins) and the binary {}{} to {HUB_SRC} on the hub",
            args.binary.display(),
            args.binary_built.as_ref().map(|built| format!(" (built {built})")).unwrap_or_default()
        ),
        format!(
            "2. Run the installer there: {INSTALL_COMMAND} (it installs bubblewrap and iproute2 with the hub's package \
             manager or tells you the command, a private Node under ~/.aria/node, the aria-serve.service user unit and \
             linger, with XDG_RUNTIME_DIR and the user bus set as a non-login ssh lacks them; no browser: agent-browser \
             and its Chrome, a large download, stay out of a server unless you run install.sh with --with-browser on the hub)"
        ),
        format!(
            "3. Send your settings and credentials (the ones you have) as one line on the stdin of {REMOTE_IMPORT}; the \
             hub's sandbox projects folder will be {projects}. {}",
            send_detail(probe, args)
        ),
        if probe.service_active {
            format!("4. {SERVICE_KEEPS_RUNNING} ({SERVICE_UNIT} is active: it is not restarted)")
        } else {
            format!("4. Restart the service: {RESTART_COMMAND}")
        },
        format!("5. Check the installed version: {REMOTE_VERSION}"),
    ];
    if let Some(key) = &args.client_key {
        lines.push(format!(
            "6. Allow your key, restricted to Aria's bridge, once: add {} to ~/.ssh/authorized_keys. The sandbox on the hub is \
             tried first: if it does not work, this step is skipped (the key is not installed, and the run ends with an \
             error) unless you pass --allow-no-sandbox, because without a sandbox every command asks first and anyone \
             holding the client key can approve those prompts",
            authorized_line(&probe.home, key)
        ));
    }
    lines.push(
        "Then, read-only: look at whether its ssh port can be reached from beyond its private networks.".to_string(),
    );
    if needs_xz(probe) {
        lines.push(
            "Blocked: the installer needs xz to unpack its own Node (this hub has none, and its Node is missing or older \
             than 22.19). Run this on the hub, then run init again:"
                .to_string(),
        );
        lines.push("    sudo apt-get install -y xz-utils".to_string());
    }
    lines
}

/// Says what step 3 writes on the hub, what it replaces and what it keeps.
fn send_detail(probe: &Probe, args: &InitArgs) -> String {
    let sent = files_to_send(probe, args);
    let (replaced, new): (Vec<&str>, Vec<&str>) =
        sent.iter().partition(|name| probe.existing.iter().any(|have| have == *name));
    let kept: Vec<&str> = BUNDLE_FILES
        .into_iter()
        .filter(|name| !sent.contains(name) && probe.existing.contains(&name.to_string()))
        .collect();
    let mut parts = Vec::new();
    if !new.is_empty() && !probe.existing.is_empty() {
        parts.push(format!("Written on the hub, which has none: {}.", new.join(", ")));
    }
    if !replaced.is_empty() {
        parts
            .push(format!("Replaced on the hub (the first copy it had stays as <name>.bak): {}.", replaced.join(", ")));
    }
    if !kept.is_empty() {
        parts.push(format!(
            "Kept as the hub has them: {} (--settings replaces settings.json; --credentials replaces auth.json, \
             mcp.json and mcp-auth.json).",
            kept.join(", ")
        ));
    }
    if sent.is_empty() {
        parts.push("Nothing is sent.".to_string());
    }
    parts.join(" ")
}

// ---- ssh ------------------------------------------------------------------------------------------------------------

/// `ssh -T <host> <command>` with the program `connect` uses (ARIA_SSH or ssh); `command` is a single argument.
fn ssh_for(host: &str, command: &str) -> io::Result<Command> {
    crate::connect::check_host(host).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "not a host"))?;
    let mut ssh = Command::new(crate::connect::ssh_program());
    ssh.args(["-T", "-o", "ConnectTimeout=20"]).arg(host).arg(command);
    Ok(ssh)
}

/// Runs `command` on `host` with `stdin` (none when None), keeping its output.
pub fn remote(host: &str, command: &str, stdin: Option<&[u8]>) -> io::Result<Output> {
    let mut ssh = ssh_for(host, command)?;
    ssh.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ssh.spawn()?;
    std::thread::scope(|scope| {
        if let (Some(bytes), Some(mut pipe)) = (stdin, child.stdin.take()) {
            // A write that fails (ssh ended early) shows in the exit status.
            scope.spawn(move || {
                let _ = pipe.write_all(bytes);
            });
        }
        child.wait_with_output()
    })
}

/// Runs `command` on `host` with its output going straight to this terminal (the installer's progress).
fn remote_shown(host: &str, command: &str) -> io::Result<ExitStatus> {
    ssh_for(host, command)?.stdin(Stdio::null()).status()
}

fn first_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).lines().map(str::trim).find(|line| !line.is_empty()).unwrap_or("").to_string()
}

fn describe(status: ExitStatus) -> String {
    match status.code() {
        Some(255) => "exit status 255, ssh could not reach or log in to the hub".to_string(),
        Some(code) => format!("exit status {code}"),
        None => "ended by a signal".to_string(),
    }
}

fn describe_error(error: &io::Error) -> String {
    match error.kind() {
        io::ErrorKind::NotFound => "ssh is not installed (or ARIA_SSH names a program that is not there)".to_string(),
        _ => error.to_string(),
    }
}

/// A read-only question: the answer's text, whatever the exit status unless ssh itself failed.
fn probe_text(host: &str, command: &str) -> Result<String, Failure> {
    let output = remote(host, command, None).map_err(|error| failed(describe_error(&error)))?;
    if output.status.code() == Some(255) {
        return Err(failed(format!("ssh could not reach or log in to {host}: {}", first_line(&output.stderr))));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The listeners `ss -ltnH -n` shows, or none when `ss` fails, is missing or says nothing usable: a hub whose sshd cannot
/// be seen is not one that has none, so an empty list means "could not tell" (every reachable hub listens on something).
fn probe_listeners(host: &str) -> Result<Vec<Listener>, Failure> {
    let output = remote(host, "ss -ltnH -n", None).map_err(|error| failed(describe_error(&error)))?;
    match output.status.code() {
        Some(255) => Err(failed(format!("ssh could not reach or log in to {host}: {}", first_line(&output.stderr)))),
        Some(0) => Ok(parse_listeners(&String::from_utf8_lossy(&output.stdout))),
        _ => Ok(Vec::new()),
    }
}

/// What `ssh -G <host>` gives (the owner's ssh configuration, nothing is connected).
fn ssh_config(host: &str) -> Result<(u16, String, String), Failure> {
    crate::connect::check_host(host).map_err(|_| usage_error(format!("{host}: not a host")))?;
    let output = Command::new(crate::connect::ssh_program())
        .arg("-G")
        .arg(host)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| failed(describe_error(&error)))?;
    if !output.status.success() {
        return Err(failed(format!("ssh -G {host}: {}", first_line(&output.stderr))));
    }
    let shown = String::from_utf8_lossy(&output.stdout);
    let (port, hostname) = parse_ssh_g(&shown);
    Ok((port.unwrap_or(22), hostname.unwrap_or_else(|| host.to_string()), ssh_g_user(&shown).unwrap_or_default()))
}

// ---- the run --------------------------------------------------------------------------------------------------------

/// The checkout the installer and the pins are in: the current directory's, else the running binary's.
fn find_repo(starts: &[PathBuf]) -> Option<PathBuf> {
    starts
        .iter()
        .flat_map(|start| start.ancestors())
        .find(|dir| dir.join("install/install.sh").is_file() && dir.join("node-version").is_file())
        .map(Path::to_path_buf)
}

fn read_binary_arch(path: &Path) -> Result<&'static str, Failure> {
    let mut header = [0u8; 64];
    let read = std::fs::File::open(path)
        .and_then(|mut file| file.read(&mut header))
        .map_err(|error| failed(format!("{}: {error}", path.display())))?;
    elf_arch(&header[..read])
        .ok_or_else(|| failed(format!("{} is not a Linux binary for aarch64 or x86_64", path.display())))
}

fn triple_of(arch: &str) -> String {
    format!("{arch}-unknown-linux-gnu")
}

/// `aria hub init`: the exit status of the run.
fn init(raw: RawInit) -> i32 {
    match run_init(raw) {
        Ok(code) => code,
        Err(Failure(code, message)) => {
            eprintln!("aria hub init: {message}");
            code
        }
    }
}

fn run_init(raw: RawInit) -> Result<i32, Failure> {
    crate::connect::check_host(&raw.host).map_err(|_| usage_error(format!("{}: not a host", raw.host)))?;
    let client_key = match &raw.client_key {
        Some(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|error| usage_error(format!("{}: {error}", path.display())))?;
            Some(validate_client_key(&text).map_err(|why| usage_error(format!("{}: {why}", path.display())))?)
        }
        None => None,
    };
    let mut starts: Vec<PathBuf> = std::env::current_dir().into_iter().collect();
    starts.extend(std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf)));
    let repo = find_repo(&starts).ok_or_else(|| {
        failed("the Aria checkout (install/install.sh and the version pins) is not here: run this from it")
    })?;
    let binary = match raw.binary {
        Some(binary) => binary,
        None => {
            let built = repo.join("target").join(triple_of("aarch64")).join("release/aria");
            if !built.is_file() {
                return Err(failed(format!(
                    "there is no binary to install ({} is not built): run `sh tools/build-headless.sh`, or pass --binary <path>",
                    built.display()
                )));
            }
            built
        }
    };
    let binary_arch = read_binary_arch(&binary)?;
    let binary_built = std::fs::metadata(&binary)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|since| utc_stamp(since.as_secs()));
    let args = InitArgs {
        host: raw.host,
        binary,
        projects: raw.projects,
        client_key,
        client_key_file: raw.client_key,
        allow_no_sandbox: raw.allow_no_sandbox,
        settings: raw.settings,
        credentials: raw.credentials,
        binary_built,
        yes: raw.yes,
    };
    let host = args.host.as_str();

    // Read-only probes: nothing on the hub changes until a yes.
    let (sshd_port, ssh_host, ssh_user) = ssh_config(host)?;
    let arch = probe_text(host, "uname -m")?;
    if arch != "aarch64" && arch != "x86_64" {
        return Err(failed(format!("the hub is {arch:?}: only aarch64 and x86_64 are supported")));
    }
    if arch != binary_arch {
        return Err(failed(format!(
            "{} is built for {binary_arch} but the hub is {arch}: build one for it with `sh tools/build-headless.sh {}` \
             and pass it with --binary",
            args.binary.display(),
            triple_of(&arch)
        )));
    }
    let os = probe_text(host, ". /etc/os-release; echo $PRETTY_NAME")?;
    let home = probe_text(host, "echo $HOME")?;
    if !home.starts_with('/') || home.chars().any(|c| c.is_control() || c.is_whitespace() || c == '"') {
        return Err(failed(format!("the hub's home directory ({home:?}) is not a plain absolute path")));
    }
    let listeners = probe_listeners(host)?;
    let found = parse_found(&probe_text(host, "command -v xz bwrap node npm tailscale")?);
    let has = |name: &str| found.iter().any(|found| found == name);
    let node_version = if has("node") {
        Some(probe_text(host, "node --version")?).filter(|version| !version.is_empty())
    } else {
        None
    };
    let service_active =
        probe_text(host, &user_env(IS_ACTIVE_COMMAND))?.lines().next().map(str::trim) == Some("active");
    let existing: Vec<String> = probe_text(host, REMOTE_EXISTING)?
        .lines()
        .map(str::trim)
        .filter(|name| BUNDLE_FILES.contains(name))
        .map(str::to_string)
        .collect();
    let probe = Probe {
        arch,
        os: if os.is_empty() { "unknown OS".to_string() } else { os },
        home,
        ssh_host,
        sshd_port,
        ssh_user,
        service_active,
        existing,
        listeners,
        has_xz: has("xz"),
        has_bwrap: has("bwrap"),
        has_npm: has("npm"),
        has_tailscale: has("tailscale"),
        node_version,
    };

    let mut bundle = export_bundle(&AriaPaths::from_env(), &probe.home, args.projects.as_deref())
        .map_err(|why| failed(format!("cannot build the settings to send: {why}")))?;
    // What the hub already has is its own, unless a flag says to replace it.
    let sending = files_to_send(&probe, &args);
    bundle.files.retain(|name, _| sending.contains(&name.as_str()));
    for warning in &bundle.warnings {
        println!("note: {warning}");
    }
    if !bundle.warnings.is_empty() {
        println!("      (a computer that never had a keyring has nothing there to send; otherwise unlock it and run init again)");
    }
    // Nothing to send (the hub has everything and no flag says to replace it): no bundle line at all.
    let mut line = if bundle.files.is_empty() {
        String::new()
    } else {
        serde_json::to_string(&bundle).map_err(|error| failed(error.to_string()))?
    };
    if !line.is_empty() {
        line.push('\n');
    }
    if line.len() as u64 > MAX_BUNDLE_BYTES {
        return Err(failed("the settings to send are too large"));
    }

    for text in plan(&probe, &args) {
        println!("{text}");
    }
    if needs_xz(&probe) {
        eprintln!("aria hub init: the installer would stop on this hub (see above); nothing was changed");
        return Ok(1);
    }
    if !confirmed(args.yes) {
        println!("nothing was changed");
        return Ok(0);
    }
    write_steps(&args, &probe, &repo, &line)
}

/// A yes: `--yes`, or an answer at a terminal. Without a terminal and without `--yes` it is no.
fn confirmed(yes: bool) -> bool {
    use std::io::IsTerminal;
    if yes {
        return true;
    }
    if !std::io::stdin().is_terminal() {
        return false;
    }
    print!("Go ahead? [y/N] ");
    let _ = std::io::stdout().flush();
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer).is_ok() && answer_is_yes(&answer)
}

/// What is already on the hub when a step fails, said honestly.
fn on_the_hub_after(step: usize) -> &'static str {
    match step {
        1 => "The upload may be partial in `~/.aria/hub-src`; nothing else on the hub was changed",
        2 => {
            "`~/.aria/hub-src` is uploaded and the installer may have partly run; running `aria hub init` again is safe"
        }
        3 => "the installer ran; nothing was imported",
        4 => "the credentials and settings are already on the hub, the service did not restart",
        _ => "everything was installed and imported; only the version check failed",
    }
}

/// The writes, in order. A failing step 1 to 5 stops the run; the key (6) and the checks (7) report and go on.
fn write_steps(args: &InitArgs, probe: &Probe, repo: &Path, bundle_line: &str) -> Result<i32, Failure> {
    let host = args.host.as_str();
    let total = if args.client_key.is_some() { 6 } else { 5 };
    let stopped = |step: usize, what: &str, detail: String| {
        failed(format!("step {step} ({what}) failed: {detail}. {}", on_the_hub_after(step)))
    };

    println!("step 1 of {total}: uploading to {HUB_SRC}");
    let mut files: Vec<(String, Vec<u8>, u32)> = Vec::new();
    for name in UPLOAD_FILES {
        let content = std::fs::read(repo.join(name)).map_err(|error| failed(format!("{name}: {error}")))?;
        files.push((name.to_string(), content, if name.ends_with(".sh") { 0o755 } else { 0o644 }));
    }
    let binary = std::fs::read(&args.binary).map_err(|error| failed(format!("{}: {error}", args.binary.display())))?;
    files.push(("aria".to_string(), binary, 0o755));
    let entries: Vec<(&str, &[u8], u32)> =
        files.iter().map(|(name, content, mode)| (name.as_str(), content.as_slice(), *mode)).collect();
    let output = remote(host, REMOTE_UPLOAD, Some(&tar_archive(&entries)))
        .map_err(|error| stopped(1, "upload", describe_error(&error)))?;
    if !output.status.success() {
        return Err(stopped(1, "upload", format!("{} {}", describe(output.status), first_line(&output.stderr))));
    }

    println!("step 2 of {total}: running the installer on the hub");
    let status = remote_shown(host, &user_env(INSTALL_COMMAND))
        .map_err(|error| stopped(2, "installer", describe_error(&error)))?;
    if !status.success() {
        return Err(stopped(2, "installer", describe(status)));
    }
    // The installer may have put iproute2 on the hub: a `ss` that was missing is asked again before giving up.
    let listeners =
        if probe.listeners.is_empty() { probe_listeners(host).unwrap_or_default() } else { probe.listeners.clone() };

    if bundle_line.is_empty() {
        println!("step 3 of {total}: nothing to send: the hub keeps the settings and credentials it has");
    } else {
        println!("step 3 of {total}: sending settings and credentials");
        let output = remote(host, REMOTE_IMPORT, Some(bundle_line.as_bytes()))
            .map_err(|error| stopped(3, "import", describe_error(&error)))?;
        if !output.status.success() {
            // What hub-import says on stderr can quote a value of the bundle: it is never shown.
            return Err(stopped(3, "import", format!("the hub refused the bundle ({})", describe(output.status))));
        }
        let written: Vec<&str> = std::str::from_utf8(&output.stdout).unwrap_or("").lines().collect();
        println!("      the hub wrote {}", written.join(", "));
    }

    if probe.service_active {
        println!("step 4 of {total}: {SERVICE_KEEPS_RUNNING}");
    } else {
        println!("step 4 of {total}: restarting the service");
        let output = remote(host, &user_env(RESTART_COMMAND), None)
            .map_err(|error| stopped(4, "restart", describe_error(&error)))?;
        if !output.status.success() {
            return Err(stopped(
                4,
                "restart",
                format!(
                    "{} {}. {}",
                    describe(output.status),
                    first_line(&output.stderr),
                    systemctl_hint(&hub_user(host, &probe.ssh_user))
                ),
            ));
        }
    }

    println!("step 5 of {total}: checking the version");
    let output = remote(host, REMOTE_VERSION, None).map_err(|error| stopped(5, "version", describe_error(&error)))?;
    if !output.status.success() {
        return Err(stopped(5, "version", format!("{} {}", describe(output.status), first_line(&output.stderr))));
    }
    println!("      the hub runs {}", first_line(&output.stdout));

    // Without linger the hub's user manager, and Aria with it, stops when the owner logs out of the hub.
    let lingers = hub_lingers(host);
    if !lingers {
        let user = hub_user(host, &probe.ssh_user);
        println!(
            "warning: linger is not on for {user} on the hub (or it could not be checked): the hub stops when you log \
             out of it, and the service with it. An administrator turns it on:"
        );
        println!("      sudo loginctl enable-linger {user}");
    }

    // The sandbox is tried before the key goes in: a key on a hub whose commands all ask is shell access for whoever holds it.
    let sandbox = probe_sandbox(host);
    for text in sandbox_warning(&sandbox) {
        println!("{text}");
    }
    let mut code = 0;
    // Whether the hub now has the key, restricted: only then is it said to be what protects the hub.
    let mut key = KeyState::None;
    if let Some(line) = &args.client_key {
        key = KeyState::Unconfirmed;
        if sandbox != Sandbox::Works && !args.allow_no_sandbox {
            eprintln!(
                "aria hub init: step 6 (key) skipped: the sandbox does not work on the hub, so the client key was not \
                 installed (anyone holding it could approve every command). Fix the sandbox and run init again, or pass \
                 --allow-no-sandbox to install the key anyway"
            );
            code = 1;
        } else {
            println!("step 6 of {total}: allowing your key, restricted to Aria's bridge");
            let sent = format!("{line}\n");
            match remote(host, REMOTE_ALLOW_KEY, Some(sent.as_bytes())) {
                Ok(output) if output.status.success() => {
                    println!("      {}", first_line(&output.stdout));
                    key = KeyState::Restricted;
                    if let Some(file) = &args.client_key_file {
                        for text in ssh_snippet(host, &probe.ssh_user, file) {
                            println!("{text}");
                        }
                    }
                }
                Ok(output) if output.status.code() == Some(KEY_UNRESTRICTED) => {
                    eprintln!(
                        "aria hub init: step 6 (key) failed: that key is already in authorized_keys WITHOUT the \
                         restriction: use a dedicated key, not your login key (see the README), or edit that line \
                         yourself to start with `{} `",
                        authorized_line(&probe.home, "").trim_end()
                    );
                    code = 1;
                }
                Ok(output) => {
                    eprintln!(
                        "aria hub init: step 6 (key) failed: {} {}",
                        describe(output.status),
                        first_line(&output.stderr)
                    );
                    code = 1;
                }
                Err(error) => {
                    eprintln!("aria hub init: step 6 (key) failed: {}", describe_error(&error));
                    code = 1;
                }
            }
        }
    }

    check_public_ssh(host, probe, &listeners, key);
    if code == 0 && !lingers {
        println!("The hub is installed, but it is not ready: turn linger on (see the warning above).");
    } else if code == 0 {
        // With the dedicated key's ssh entry added, `aria-hub` is the way in that uses it.
        let target = if key == KeyState::Restricted { "aria-hub" } else { host };
        println!("The hub is ready. Connect from this computer with: aria connect {target}");
    }
    Ok(code)
}

/// What trying the sandbox on the hub showed.
#[derive(Debug, Clone, PartialEq)]
enum Sandbox {
    Works,
    /// `bwrap: not found`: it is not installed.
    Missing,
    /// Installed, but it could not start (usually AppArmor or a sysctl refusing the user namespace); what it said.
    Blocked(String),
}

/// The user the hub's commands run as: the one typed before the `@`, else the one the owner's ssh uses.
fn hub_user(host: &str, ssh_user: &str) -> String {
    match host.split_once('@') {
        Some((user, _)) => user.to_string(),
        None if !ssh_user.is_empty() => ssh_user.to_string(),
        None => "<user>".to_string(),
    }
}

/// Whether the hub's user lingers (read-only): `Linger=yes`. Anything else, including a `loginctl` that fails, is no.
fn hub_lingers(host: &str) -> bool {
    probe_text(host, REMOTE_LINGER).is_ok_and(|said| said.lines().any(|line| line.trim() == "Linger=yes"))
}

/// Tries the sandbox on the hub (read-only).
fn probe_sandbox(host: &str) -> Sandbox {
    match remote(host, REMOTE_BWRAP, None) {
        Ok(output) if output.status.success() => Sandbox::Works,
        Ok(output) => {
            let said = first_line(&output.stderr);
            if output.status.code() == Some(127) || said.ends_with("bwrap: not found") {
                Sandbox::Missing
            } else if said.is_empty() {
                Sandbox::Blocked(describe(output.status))
            } else {
                Sandbox::Blocked(said)
            }
        }
        Err(error) => Sandbox::Blocked(describe_error(&error)),
    }
}

/// What is said when the sandbox does not work on the hub; the fix is only printed, never run. Nothing for a working one.
fn sandbox_warning(sandbox: &Sandbox) -> Vec<String> {
    let mut lines = match sandbox {
        Sandbox::Works => return Vec::new(),
        Sandbox::Missing => vec![
            "warning: the sandbox cannot start on the hub: bubblewrap is not installed (bwrap: not found).".to_string(),
            "  Install it on the hub:".to_string(),
            "      sudo apt-get install -y bubblewrap".to_string(),
        ],
        Sandbox::Blocked(problem) => vec![
            format!("warning: the sandbox cannot start on the hub ({problem})."),
            "  bwrap is installed but cannot create a user namespace. On Ubuntu 24.04 this is usually AppArmor restricting"
                .to_string(),
            "  unprivileged user namespaces. An administrator can fix it:".to_string(),
            "      sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0".to_string(),
            "  (write kernel.apparmor_restrict_unprivileged_userns=0 in a file under /etc/sysctl.d/ to keep it after a"
                .to_string(),
            "  reboot), or give bwrap its own AppArmor profile that allows user namespaces. Aria did not run any of this."
                .to_string(),
        ],
    };
    lines.push(
        "  Until a sandbox works, Aria asks you before every command it wants to run there, and anyone holding the client \
         key can approve those prompts: treat that key as shell access to the hub user."
            .to_string(),
    );
    lines
}

/// What the hub's `authorized_keys` holds of the owner's `--client-key` after step 6.
#[derive(Clone, Copy, PartialEq)]
enum KeyState {
    /// No `--client-key` was given.
    None,
    /// One was given and step 6 did not end well: the hub is not known to hold it restricted.
    Unconfirmed,
    Restricted,
}

/// Says once when sshd can be reached from beyond the hub's private networks, or that it could not be checked. Nothing
/// is changed.
fn check_public_ssh(host: &str, probe: &Probe, listeners: &[Listener], key: KeyState) {
    if listeners.is_empty() {
        println!(
            "warning: could not check whether sshd is open to the internet (`ss` failed or is missing); check it yourself."
        );
        return;
    }
    if !public_ssh(listeners, probe.sshd_port) {
        return;
    }
    println!(
        "warning: sshd on {} listens on port {} on a public address, and that port stays open until you close it \
         yourself (Aria never edits sshd_config, the firewall or Tailscale).",
        probe.ssh_host, probe.sshd_port
    );
    match key {
        KeyState::Restricted => println!(
            "  Meanwhile the restricted client key (it can only run `aria client-stdio`) and key-only login are the protection."
        ),
        KeyState::Unconfirmed => println!(
            "  Meanwhile the protection is key-only login; your client key is not known to be restricted yet (see step 6)."
        ),
        KeyState::None => {
            println!("  Meanwhile the protection is key-only login; add a restricted client key with --client-key.")
        }
    }
    // Best effort, and only what a user can read: no sudo.
    let config = remote(host, REMOTE_SSHD_CONFIG, None)
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    match password_auth(&config) {
        Some(false) => println!("  sshd_config: password logins look off."),
        Some(true) => println!("  sshd_config: password logins look on; set PasswordAuthentication no."),
        None => {}
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
    fn export_warns_when_the_keyring_cannot_be_opened_to_look_for_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let paths = plain_home(dir.path());
        std::fs::remove_file(paths.root.join("mcp-auth.json")).unwrap();
        let bundle = export_plain(&paths, None).unwrap();
        assert!(!bundle.files.contains_key("mcp-auth.json"), "still skipped");
        assert_eq!(
            bundle.warnings,
            ["mcp-auth.json: no copy found and the keyring could not be opened (no keyring here)"]
        );
        for secret in ["sk-test-1234", "tvly-test", "n-test"] {
            assert!(!bundle.warnings.iter().any(|w| w.contains(secret)), "{secret}");
        }
        let sent = serde_json::to_string(&bundle).unwrap();
        assert!(!sent.contains("warnings"), "warnings are never sent: {sent}");
        assert_eq!(serde_json::from_str::<Bundle>(&sent).unwrap().files, bundle.files);
    }

    #[test]
    fn export_does_not_warn_when_the_keyring_opens_and_the_file_is_simply_missing() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, keyring) = keyring_home(dir.path());
        keyring.delete("file:mcp-auth.json").unwrap();
        let shown = vault(dir.path(), keyring);
        let bundle = export_with(&paths, "/home/hub", None, &|| Ok(shown.clone())).unwrap();
        assert!(!bundle.files.contains_key("mcp-auth.json"));
        assert!(bundle.warnings.is_empty(), "{:?}", bundle.warnings);
        // A file found in the home needs no keyring and warns of nothing.
        let plain = plain_home(&dir.path().join("p"));
        assert!(export_plain(&plain, None).unwrap().warnings.is_empty());
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
    fn import_keeps_the_first_backup() {
        let dir = tempfile::tempdir().unwrap();
        let paths = hub(dir.path());
        std::fs::create_dir_all(&paths.root).unwrap();
        std::fs::write(paths.root.join("auth.json"), r#"{"first":"the hub's own"}"#).unwrap();
        import(&paths, &line(&[("auth.json", r#"{"second":1}"#)])).unwrap();
        import(&paths, &line(&[("auth.json", r#"{"third":2}"#)])).unwrap();
        assert_eq!(
            std::fs::read_to_string(paths.root.join("auth.json.bak")).unwrap(),
            r#"{"first":"the hub's own"}"#,
            "the first backup is never overwritten"
        );
        assert_eq!(std::fs::read_to_string(paths.root.join("auth.json")).unwrap(), r#"{"third":2}"#);
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

    fn probe() -> Probe {
        Probe {
            arch: "aarch64".into(),
            os: "Ubuntu 24.04.2 LTS".into(),
            home: "/home/ana".into(),
            ssh_host: "hub.example".into(),
            sshd_port: 2222,
            ssh_user: "ana".into(),
            service_active: false,
            existing: vec![],
            listeners: parse_listeners(SS),
            has_xz: true,
            has_bwrap: false,
            has_npm: true,
            has_tailscale: true,
            node_version: Some("v18.19.1".into()),
        }
    }

    fn init_args() -> InitArgs {
        InitArgs {
            host: "ana@hub.example".into(),
            binary: PathBuf::from("/src/aria/target/aarch64-unknown-linux-gnu/release/aria"),
            projects: None,
            client_key: None,
            client_key_file: None,
            yes: false,
            allow_no_sandbox: false,
            settings: false,
            credentials: false,
            binary_built: None,
        }
    }

    const KEY: &str = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl ana@laptop";

    fn elf_header(class: u8, data: u8, machine: [u8; 2]) -> Vec<u8> {
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = class;
        bytes[5] = data;
        bytes[18..20].copy_from_slice(&machine);
        bytes
    }

    #[test]
    fn elf_arch_reads_the_machine_of_the_header() {
        assert_eq!(elf_arch(&elf_header(2, 1, [0xB7, 0x00])), Some("aarch64"));
        assert_eq!(elf_arch(&elf_header(2, 1, [0x3E, 0x00])), Some("x86_64"));
        assert_eq!(elf_arch(&elf_header(2, 2, [0x00, 0xB7])), Some("aarch64"), "big endian");
        assert_eq!(elf_arch(&elf_header(1, 1, [0x28, 0x00])), None, "32-bit ARM is neither");
        assert_eq!(elf_arch(&elf_header(2, 1, [0xF3, 0x00])), None, "RISC-V is neither");
    }

    #[test]
    fn elf_arch_refuses_what_is_not_a_full_elf_header() {
        assert_eq!(elf_arch(&[]), None);
        assert_eq!(elf_arch(b"\x7fELF"), None, "too short");
        assert_eq!(elf_arch(&elf_header(2, 1, [0xB7, 0x00])[..19]), None, "one byte short of the machine");
        let mut not_elf = elf_header(2, 1, [0xB7, 0x00]);
        not_elf[0] = b'#';
        assert_eq!(elf_arch(&not_elf), None);
        assert_eq!(
            elf_arch(&[0xCF, 0xFA, 0xED, 0xFE, 0x0C, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            None,
            "Mach-O"
        );
        assert_eq!(elf_arch(&elf_header(2, 7, [0xB7, 0x00])), None, "no such byte order");
    }

    #[test]
    fn a_bare_public_key_is_accepted_with_or_without_a_comment() {
        assert_eq!(validate_client_key(&format!("{KEY}\n")).unwrap(), KEY);
        assert_eq!(validate_client_key(KEY).unwrap(), KEY);
        assert_eq!(validate_client_key(&format!("{KEY}\r\n")).unwrap(), KEY);
        let bare = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC+/=";
        assert_eq!(validate_client_key(bare).unwrap(), bare);
        for kind in [
            "ssh-ed25519",
            "ssh-rsa",
            "ecdsa-sha2-nistp256",
            "ecdsa-sha2-nistp384",
            "ecdsa-sha2-nistp521",
            "sk-ssh-ed25519@openssh.com",
            "sk-ecdsa-sha2-nistp256@openssh.com",
        ] {
            assert!(validate_client_key(&format!("{kind} AAAAC3Nza comment words\n")).is_ok(), "{kind}");
        }
    }

    #[test]
    fn a_key_line_with_options_or_injection_is_refused() {
        for bad in [
            format!("command=\"sh\" {KEY}"),
            format!("no-pty {KEY}"),
            format!("restrict {KEY}"),
            format!("from=\"1.2.3.4\" {KEY}"),
            format!("{KEY}\n{KEY}"),
            format!("{KEY}\n\n{KEY}\n"),
            format!("{KEY}\n# more\n"),
            format!("{KEY}\" ,command=\"x"),
            format!("{KEY}\rcommand=\"x\" {KEY}"),
            "ssh-ed25519 AAAA\"".to_string(),
            "ssh-ed25519 AA AA\0".to_string(),
            "ssh-ed25519 AAAA\x01".to_string(),
            "ssh-ed25519".to_string(),
            "ssh-ed25519 ".to_string(),
            "ssh-ed25519 not*base64".to_string(),
            "ssh-dss AAAA".to_string(),
            "-----BEGIN OPENSSH PRIVATE KEY-----".to_string(),
            String::new(),
            "\n".to_string(),
            "   ".to_string(),
        ] {
            assert!(validate_client_key(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_authorized_keys_line_restricts_the_key_to_the_bridge() {
        assert_eq!(
            authorized_line("/home/ana", KEY),
            format!("restrict,command=\"/home/ana/.local/bin/aria client-stdio\" {KEY}")
        );
        assert!(authorized_line("/home/ana/", KEY)
            .starts_with("restrict,command=\"/home/ana/.local/bin/aria client-stdio\" "));
    }

    #[test]
    fn the_plan_numbers_the_steps_and_names_each_command() {
        let lines = plan(&probe(), &init_args());
        let text = lines.join("\n");
        let numbered: Vec<&String> =
            lines.iter().filter(|line| line.chars().next().is_some_and(|c| c.is_ascii_digit())).collect();
        assert_eq!(numbered.len(), 5, "{text}");
        for (index, line) in numbered.iter().enumerate() {
            assert!(line.starts_with(&format!("{}. ", index + 1)), "{line}");
        }
        assert!(numbered[0].contains("~/.aria/hub-src") && numbered[0].contains("release/aria"), "{text}");
        assert!(
            numbered[1].contains("sh ~/.aria/hub-src/install/install.sh --headless --local ~/.aria/hub-src/aria"),
            "{text}"
        );
        assert!(numbered[2].contains("aria hub-import"), "{text}");
        assert!(numbered[3].contains("systemctl --user restart aria-serve.service"), "{text}");
        assert!(numbered[4].contains("aria --version"), "{text}");
        assert!(
            text.contains("Ubuntu 24.04.2 LTS") && text.contains("aarch64") && text.contains("hub.example"),
            "{text}"
        );
        assert!(!text.contains("client-stdio") && !text.contains("authorized_keys"), "no key, no key step: {text}");
        assert!(!text.contains("xz-utils"), "{text}");
    }

    #[test]
    fn the_plan_has_a_step_for_the_client_key_only_when_there_is_one() {
        let mut args = init_args();
        args.client_key = Some(KEY.to_string());
        let lines = plan(&probe(), &args);
        let text = lines.join("\n");
        let numbered: Vec<&String> =
            lines.iter().filter(|line| line.chars().next().is_some_and(|c| c.is_ascii_digit())).collect();
        assert_eq!(numbered.len(), 6, "{text}");
        assert!(
            numbered[5].contains("authorized_keys")
                && numbered[5].contains("restrict,command=\"/home/ana/.local/bin/aria client-stdio\""),
            "{text}"
        );
        assert!(numbered[5].contains("ana@laptop"), "{text}");
        assert!(
            numbered[5].contains("sandbox")
                && numbered[5].contains("skipped")
                && numbered[5].contains("--allow-no-sandbox"),
            "the plan says the key waits for a working sandbox: {text}"
        );
    }

    #[test]
    fn a_first_init_plans_to_send_everything_and_restart() {
        let text = plan(&probe(), &init_args()).join("\n");
        assert!(text.contains("Restart the service: systemctl --user restart aria-serve.service"), "{text}");
        assert!(!text.contains("kept") && !text.contains("keeps running"), "{text}");
        assert_eq!(files_to_send(&probe(), &init_args()), BUNDLE_FILES);
    }

    #[test]
    fn a_rerun_plans_to_keep_what_the_hub_has_and_not_to_restart() {
        let mut host = probe();
        host.service_active = true;
        host.existing = BUNDLE_FILES.iter().map(|name| name.to_string()).collect();
        let text = plan(&host, &init_args()).join("\n");
        assert!(
            text.contains(
                "the service keeps running: it takes the new binary when idle and the settings within a second"
            ),
            "{text}"
        );
        assert!(!text.contains("Restart the service"), "{text}");
        assert!(text.contains("Nothing is sent") || text.contains("kept"), "{text}");
        assert!(text.contains("--settings") && text.contains("--credentials"), "{text}");
        assert!(files_to_send(&host, &init_args()).is_empty());
        let mut with = init_args();
        with.credentials = true;
        assert_eq!(files_to_send(&host, &with), ["auth.json", "mcp.json", "mcp-auth.json"]);
        let text = plan(&host, &with).join("\n");
        assert!(text.contains("replace") && text.contains("auth.json"), "{text}");
        with.credentials = false;
        with.settings = true;
        assert_eq!(files_to_send(&host, &with), ["settings.json"]);
        // a file the hub lacks is sent without a flag
        host.existing = vec!["auth.json".into()];
        assert_eq!(files_to_send(&host, &init_args()), ["settings.json", "mcp.json", "mcp-auth.json"]);
        // an active service is not restarted even on a first import
        assert!(plan(&host, &init_args()).join("\n").contains("keeps running"));
    }

    #[test]
    fn the_user_environment_prefix_sets_what_a_non_login_ssh_lacks_and_keeps_what_is_there() {
        let uid = String::from_utf8(Command::new("id").arg("-u").output().unwrap().stdout).unwrap();
        let dir = format!("/run/user/{}", uid.trim());
        let show = "printf '%s %s' \"$XDG_RUNTIME_DIR\" \"$DBUS_SESSION_BUS_ADDRESS\"";
        let run = |env: &[(&str, &str)]| {
            let mut sh = Command::new("sh");
            sh.args(["-c", &user_env(show)]).env_clear().env("PATH", std::env::var("PATH").unwrap());
            sh.envs(env.iter().copied());
            String::from_utf8(sh.output().unwrap().stdout).unwrap()
        };
        if Path::new(&dir).is_dir() {
            assert_eq!(run(&[]), format!("{dir} unix:path={dir}/bus"));
        }
        assert_eq!(
            run(&[("XDG_RUNTIME_DIR", "/elsewhere"), ("DBUS_SESSION_BUS_ADDRESS", "unix:path=/mine")]),
            "/elsewhere unix:path=/mine"
        );
    }

    #[test]
    fn a_time_is_said_in_utc_to_the_minute() {
        assert_eq!(utc_stamp(0), "1970-01-01 00:00 UTC");
        assert_eq!(utc_stamp(1_700_000_000), "2023-11-14 22:13 UTC");
        assert_eq!(utc_stamp(1_709_164_799), "2024-02-28 23:59 UTC");
        assert_eq!(utc_stamp(1_709_164_800), "2024-02-29 00:00 UTC", "a leap day");
    }

    #[test]
    fn the_plan_says_when_the_binary_was_built() {
        let mut args = init_args();
        args.binary_built = Some("2023-11-14 22:13 UTC".into());
        let text = plan(&probe(), &args).join("\n");
        assert!(text.contains("release/aria (built 2023-11-14 22:13 UTC)"), "{text}");
        args.binary_built = None;
        assert!(!plan(&probe(), &args).join("\n").contains("built"));
    }

    #[test]
    fn the_plan_leaves_the_browser_out_and_says_how_to_get_it() {
        let text = plan(&probe(), &init_args()).join("\n");
        assert!(text.contains("--with-browser") && text.contains("agent-browser"), "{text}");
    }

    #[test]
    fn the_plan_says_when_the_installer_would_need_xz() {
        let mut host = probe();
        host.has_xz = false;
        assert!(needs_xz(&host), "Node 18 is too old, so the installer unpacks its own");
        assert!(plan(&host, &init_args()).join("\n").contains("sudo apt-get install -y xz-utils"));
        host.node_version = Some("v22.19.0".into());
        assert!(!needs_xz(&host));
        assert!(!plan(&host, &init_args()).join("\n").contains("xz-utils"));
        host.has_npm = false;
        assert!(needs_xz(&host), "Node without npm is not enough");
        host.has_npm = true;
        host.node_version = None;
        assert!(needs_xz(&host), "no Node at all");
        host.has_xz = true;
        assert!(!needs_xz(&host));
    }

    #[test]
    fn a_node_version_is_current_from_22_19() {
        for current in ["v22.19.0", "v22.23.2", "22.19.1", "v23.0.0", "v24.1.0", "v100.0.0"] {
            assert!(node_is_current(current), "{current}");
        }
        for old in ["v18.19.1", "v22.18.9", "v20.99.0", "", "node", "v22", "v22.x.1"] {
            assert!(!node_is_current(old), "{old}");
        }
    }

    #[test]
    fn the_tar_holds_each_file_with_its_mode_and_nothing_else() {
        let entries: [(&str, &[u8], u32); 3] = [
            ("install/install.sh", b"#!/bin/sh\n", 0o755),
            ("pi-version", b"1.2.3\n", 0o644),
            ("aria", &[7u8; 1000], 0o755),
        ];
        let archive = tar_archive(&entries);
        assert_eq!(archive.len() % 512, 0);
        let dir = tempfile::tempdir().unwrap();
        let mut child = std::process::Command::new("tar")
            .args(["-xf", "-", "-C"])
            .arg(dir.path())
            .stdin(std::process::Stdio::piped())
            .spawn()
            .expect("tar");
        std::io::Write::write_all(child.stdin.as_mut().unwrap(), &archive).unwrap();
        assert!(child.wait().unwrap().success());
        assert_eq!(std::fs::read(dir.path().join("install/install.sh")).unwrap(), b"#!/bin/sh\n");
        assert_eq!(std::fs::read(dir.path().join("pi-version")).unwrap(), b"1.2.3\n");
        assert_eq!(std::fs::read(dir.path().join("aria")).unwrap(), vec![7u8; 1000]);
        assert_eq!(mode(&dir.path().join("aria")), 0o755);
        assert_eq!(mode(&dir.path().join("pi-version")), 0o644);
        assert_eq!(files_in(dir.path()), ["aria", "install", "pi-version"]);
    }

    #[test]
    fn the_command_v_probe_is_read_by_the_names_it_found() {
        let found = parse_found("/usr/bin/xz\n/usr/local/bin/node\n\n/usr/sbin/tailscale\n");
        assert_eq!(found, ["xz", "node", "tailscale"]);
        assert!(parse_found("").is_empty());
    }

    #[test]
    fn ssh_g_gives_the_port_and_the_hostname() {
        assert_eq!(
            parse_ssh_g("user ana\nhostname hub.example\nport 2222\nidentityfile x\n"),
            (Some(2222), Some("hub.example".into()))
        );
        assert_eq!(parse_ssh_g("port 22\n"), (Some(22), None));
        assert_eq!(parse_ssh_g("port notaport\nport 99999\n"), (None, None));
        assert_eq!(parse_ssh_g(""), (None, None));
    }

    #[test]
    fn ssh_g_gives_the_user_the_first_time() {
        assert_eq!(ssh_g_user("user ana\nhostname hub.example\nuser bob\n").as_deref(), Some("ana"));
        assert_eq!(ssh_g_user("hostname hub.example\n"), None);
    }

    #[test]
    fn the_ssh_config_snippet_names_the_host_the_user_and_the_key_without_pub() {
        let lines = ssh_snippet("ana@hub.example", "root", Path::new("/home/ana/.ssh/aria-hub.pub")).join("\n");
        assert!(lines.contains(
            "Host aria-hub\n    HostName hub.example\n    User ana\n    IdentityFile /home/ana/.ssh/aria-hub\n    IdentitiesOnly yes"
        ));
        assert!(lines.contains("~/.ssh/config") && lines.contains("aria connect aria-hub"), "{lines}");
        // a host with no user: the one ssh would use
        let lines = ssh_snippet("hub.example", "ana", Path::new("/k/aria-hub.pub")).join("\n");
        assert!(lines.contains("HostName hub.example\n    User ana\n"), "{lines}");
    }

    #[test]
    fn sshd_config_says_whether_passwords_are_off_the_first_word_winning() {
        assert_eq!(password_auth("Port 2222\nPasswordAuthentication no\n"), Some(false));
        assert_eq!(password_auth("  passwordauthentication   yes # why\n"), Some(true));
        assert_eq!(
            password_auth("PasswordAuthentication no\nPasswordAuthentication yes\n"),
            Some(false),
            "the first one wins"
        );
        assert_eq!(password_auth("#PasswordAuthentication yes\nPort 22\n"), None, "commented out");
        assert_eq!(password_auth(""), None);
        assert_eq!(password_auth("PasswordAuthentication maybe\n"), None);
    }

    #[test]
    fn only_a_yes_is_a_yes() {
        for yes in ["y", "Y", "yes", "YES", " yes \n", "y\r\n"] {
            assert!(answer_is_yes(yes), "{yes:?}");
        }
        for no in ["", "n", "no", "\n", "yep", "ye", "sure", "yes please", "1"] {
            assert!(!answer_is_yes(no), "{no:?}");
        }
    }

    #[test]
    fn init_arguments_are_a_host_and_four_options() {
        let args = |list: &[&str]| parse_init_args(&list.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let all = args(&[
            "ana@hub.example",
            "--binary",
            "/b/aria",
            "--projects",
            "/srv/work",
            "--client-key",
            "/k.pub",
            "--allow-no-sandbox",
            "--settings",
            "--credentials",
            "--yes",
        ])
        .unwrap();
        assert!(all.allow_no_sandbox && all.settings && all.credentials);
        assert_eq!(all.host, "ana@hub.example");
        assert_eq!(all.binary, Some(PathBuf::from("/b/aria")));
        assert_eq!(all.projects.as_deref(), Some("/srv/work"));
        assert_eq!(all.client_key, Some(PathBuf::from("/k.pub")));
        assert!(all.yes);
        let few = args(&["--yes", "hub.example"]).unwrap();
        assert!(few.yes && few.binary.is_none() && few.projects.is_none() && few.client_key.is_none());
        assert!(!few.allow_no_sandbox && !few.settings && !few.credentials);
        assert!(args(&[]).is_err(), "a host is needed");
        assert!(args(&["a", "b"]).is_err(), "one host");
        assert!(args(&["a", "--binary"]).is_err(), "an option needs its value");
        assert!(args(&["a", "--nope"]).is_err());
        assert_eq!(
            args(&["-oProxyCommand=x"]).unwrap().host,
            "-oProxyCommand=x",
            "the host check refuses it later, as connect does"
        );
    }

    #[test]
    fn the_checkout_is_found_from_a_folder_inside_it() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("aria");
        std::fs::create_dir_all(repo.join("install")).unwrap();
        std::fs::create_dir_all(repo.join("target/debug")).unwrap();
        let elsewhere = dir.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        assert_eq!(find_repo(&[repo.join("target/debug")]), None, "no installer yet");
        std::fs::write(repo.join("install/install.sh"), "").unwrap();
        assert_eq!(find_repo(&[repo.join("target/debug")]), None, "the pins are part of it");
        std::fs::write(repo.join("node-version"), "").unwrap();
        assert_eq!(find_repo(&[repo.join("target/debug")]), Some(repo.clone()));
        assert_eq!(find_repo(&[elsewhere, repo.clone()]), Some(repo), "the first start that is not in one is skipped");
    }
}
