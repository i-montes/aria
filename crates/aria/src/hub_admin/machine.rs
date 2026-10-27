//! This computer as a **machine** of a hub, from the side of `aria hub`: the step of `hub init` that links it (the key, its
//! line in the hub's `authorized_keys`, `~/.aria/link.json`), and the commands that look at or undo that: `aria hub machines`,
//! `aria hub unlink-machine <name>` and `aria hub status`. A child of `hub_admin` so it uses that module's helpers (the
//! admin ssh, the key checks, the screens) instead of copying them. The link itself, the thread that keeps the connection up, is
//! `link.rs`; the hub's side of it is `machines.rs`.

use super::*;
use crate::link::{self, LinkConfig};
use aria_core::bridge::PROTOCOL;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::Stdio;
use std::time::{Duration, Instant};

/// How long `hub init` waits for this computer to show as linked on the hub: the service here has to notice `link.json` (it
/// looks every ten seconds) and open ssh. `ARIA_LINK_WAIT_MS` is how the tests make it short.
pub fn link_wait() -> Duration {
    std::env::var("ARIA_LINK_WAIT_MS")
        .ok()
        .and_then(|ms| ms.parse().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_secs(20))
}

/// `~/.ssh/aria-machine-<name>.pub`.
pub fn default_machine_key_file(name: &str) -> Result<PathBuf, Failure> {
    let home = std::env::var_os("HOME").filter(|home| !home.is_empty()).ok_or_else(|| failed("HOME is not set"))?;
    Ok(PathBuf::from(home).join(".ssh").join(format!("aria-machine-{name}.pub")))
}

/// The machine `hub init` makes of this computer, or None: not when `--no-machine`, and only with a key step (the client key's)
/// or a `--machine-key` of its own.
pub fn machine_setup(
    name: Option<String>,
    key: Option<&Path>,
    no_machine: bool,
    key_step: bool,
) -> Result<Option<MachineSetup>, Failure> {
    if no_machine || !(key_step || key.is_some()) {
        return Ok(None);
    }
    let name = name.unwrap_or_else(|| link::machine_name_from(&link::this_computer()));
    let (key_file, line, key_new) = match key {
        Some(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|error| usage_error(format!("{}: {error}", path.display())))?;
            let line = validate_client_key(&text).map_err(|why| usage_error(format!("{}: {why}", path.display())))?;
            (path.to_path_buf(), Some(line), false)
        }
        None => {
            let file = default_machine_key_file(&name)?;
            if file.exists() {
                let text = std::fs::read_to_string(&file)
                    .map_err(|error| usage_error(format!("{}: {error}", file.display())))?;
                let line =
                    validate_client_key(&text).map_err(|why| usage_error(format!("{}: {why}", file.display())))?;
                (file, Some(line), false)
            } else {
                (file, None, true)
            }
        }
    };
    Ok(Some(MachineSetup { name, key_file, key: line, key_new }))
}

/// The arguments that make a machine's key: no passphrase (`-N ""`), because the link starts by itself with nobody there to
/// type one; `-q` keeps out the fingerprint and the picture.
pub fn machine_keygen_args(private: &Path, name: &str) -> Vec<std::ffi::OsString> {
    vec![
        "-q".into(),
        "-t".into(),
        "ed25519".into(),
        "-N".into(),
        "".into(),
        "-f".into(),
        private.into(),
        "-C".into(),
        format!("aria-machine-{name}").into(),
    ]
}

/// Makes the key whose public half is `pub_file` and returns that public line, checked. Never overwrites a key. It asks
/// nothing, so it runs with no terminal.
fn make_machine_key(program: &std::ffi::OsStr, pub_file: &Path, name: &str) -> Result<String, Failure> {
    let shown = pub_file.to_string_lossy();
    let private = PathBuf::from(shown.strip_suffix(".pub").unwrap_or(&shown).to_string());
    if private.exists() && !pub_file.exists() {
        return Err(failed(format!(
            "{} exists but {} does not, and Aria never overwrites a key: make the public half with `ssh-keygen -y -f {} > {}`, or pass --machine-key",
            private.display(),
            pub_file.display(),
            private.display(),
            pub_file.display()
        )));
    }
    if !pub_file.exists() {
        if let Some(dir) = private.parent() {
            if !dir.exists() {
                std::fs::create_dir_all(dir).map_err(|error| failed(format!("{}: {error}", dir.display())))?;
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
            }
        }
        let status = Command::new(program)
            .args(machine_keygen_args(&private, name))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|error| failed(format!("could not run ssh-keygen: {}", describe_error(&error))))?;
        if !status.success() {
            return Err(failed(format!(
                "ssh-keygen did not make the machine key ({}); make one yourself and pass it with --machine-key, or pass --no-machine",
                describe(status)
            )));
        }
    }
    let text = std::fs::read_to_string(pub_file).map_err(|error| failed(format!("{}: {error}", pub_file.display())))?;
    validate_client_key(&text).map_err(|why| failed(format!("{}: {why}", pub_file.display())))
}

/// How linking this computer ended.
#[derive(Debug, PartialEq)]
pub enum Linked {
    /// Set up, and the hub lists this computer as online.
    Yes,
    /// Set up, but it does not show as online yet (nothing runs Aria here, or it has not noticed): said in a note.
    Pending,
    /// Something failed: said, and the run ends with an error.
    Failed,
}

/// The step of `hub init` that makes this computer a machine: the key, its line on the hub, `link.json`, and a wait for the hub
/// to list the machine online. Returns how it ended; the failures are said on stderr as the key step's are.
pub fn link_this_computer(
    host: &str,
    hub_home: &str,
    machine: &MachineSetup,
    step: (usize, usize),
    notes: &mut Vec<Note>,
    ui: &mut dyn HubUi,
) -> Linked {
    ui.step_start(
        step.0,
        step.1,
        &format!("Linking this computer as {}", machine.name),
        "So the server can work on projects on this computer, when you ask it to.",
    );
    let key = match &machine.key {
        Some(key) => key.clone(),
        None => match make_machine_key(&keygen_program(), &machine.key_file, &machine.name) {
            Ok(key) => {
                ui.step_ok(&format!("The link's key is {}", tilde(&private_key_of(&machine.key_file))));
                key
            }
            Err(Failure(_, message)) => {
                ui.failed("This computer could not be linked: its key could not be made.");
                eprintln!("aria hub init: the machine key failed: {message}");
                return Linked::Failed;
            }
        },
    };
    let sent = format!("{key}\n{}\n", machine.name);
    match remote(host, REMOTE_ALLOW_MACHINE, Some(sent.as_bytes())) {
        Ok(output) if output.status.success() => {
            ui.step_ok(&if first_line(&output.stdout) == "already allowed" {
                "The link's key was already allowed on the server".to_string()
            } else {
                "The link's key is allowed on the server, to be this computer's link and nothing else".to_string()
            });
        }
        Ok(output) if output.status.code() == Some(KEY_UNRESTRICTED) => {
            let said = first_line(&output.stdout);
            ui.failed("This computer could not be linked: the server already has that key in another form.");
            eprintln!(
                "aria hub init: the machine key failed: that key is already in authorized_keys {}: edit that line yourself to start with `{} ` (or remove it and run init again)",
                match said.as_str() {
                    "other options" => "with this machine's link as its command, but with other options after it",
                    "different command" => "with a different command than this machine's link",
                    _ => "WITHOUT the restriction (use a key of its own, not your login key)",
                },
                authorized_machine_line(hub_home, &machine.name, "").trim_end()
            );
            return Linked::Failed;
        }
        Ok(output) => {
            ui.failed("This computer could not be linked: the key was not allowed.");
            eprintln!(
                "aria hub init: the machine key failed: {} {}",
                describe(output.status),
                first_line(&output.stderr)
            );
            return Linked::Failed;
        }
        Err(error) => {
            ui.failed("This computer could not be linked: the key was not allowed.");
            eprintln!("aria hub init: the machine key failed: {}", describe_error(&error));
            return Linked::Failed;
        }
    }
    let config = LinkConfig {
        hub: host.to_string(),
        name: machine.name.clone(),
        key: private_key_of(&machine.key_file).to_string_lossy().into_owned(),
    };
    if let Err(error) = config.save(&AriaPaths::from_env().root) {
        ui.failed("This computer could not be linked: its link file could not be written.");
        eprintln!("aria hub init: the machine key failed: ~/.aria/link.json: {error}");
        return Linked::Failed;
    }
    ui.step_ok("This computer links by itself whenever Aria runs here (~/.aria/link.json)");
    if wait_for_machine(host, &machine.name, link_wait()) {
        ui.step_ok(&format!("{} is linked", machine.name));
        return Linked::Yes;
    }
    notes.push(
        Note::text(format!(
            "This computer is set up as {} but does not show as linked yet: it links when Aria runs here (the dock, or \
             `aria serve`), within ten seconds of starting. If Aria was already running, restart it. To see:",
            machine.name
        ))
        .then_command("aria hub machines"),
    );
    Linked::Pending
}

/// Whether the hub lists machine `name` online, asking now and again for up to `within`.
fn wait_for_machine(host: &str, name: &str, within: Duration) -> bool {
    let until = Instant::now() + within;
    loop {
        // Each ask may take a while on a real ssh (a handshake), and no longer than the wait itself in a test.
        let ask = within.clamp(Duration::from_secs(1), Duration::from_secs(15));
        if let Ok(machines) = fetch_machines(host, ask) {
            if machines.iter().any(|machine| machine["name"] == name && machine["online"] == true) {
                return true;
            }
        }
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_secs(2).min(until.saturating_duration_since(Instant::now())));
    }
}

// ---- asking the hub ------------------------------------------------------------------------------------------------

/// The machines the hub lists, through the terminal bridge `aria connect` uses (with the key `hub init` linked this computer
/// with, when there is one): `hello`, then `machinesList`, read until the answer.
pub fn fetch_machines(host: &str, within: Duration) -> Result<Vec<Value>, String> {
    let mut command = crate::connect::ssh_command(host).map_err(|error| connection_problem(&error))?;
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("could not run ssh: {error}"))?;
    let (Some(mut stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        let _ = child.kill();
        return Err("ssh has no pipes".to_string());
    };
    let (sender, lines) = std::sync::mpsc::channel::<Value>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(value) = serde_json::from_str::<Value>(&line) {
                if sender.send(value).is_err() {
                    return;
                }
            }
        }
    });
    let result = (|| {
        writeln!(
            stdin,
            "{}",
            json!({"type": "hello", "protocol": PROTOCOL, "version": env!("CARGO_PKG_VERSION"), "remote": true})
        )
        .and_then(|_| writeln!(stdin, "{}", json!({"type": "machinesList", "id": 1})))
        .and_then(|_| stdin.flush())
        .map_err(|_| "the connection closed".to_string())?;
        let until = Instant::now() + within;
        loop {
            let wait = until.saturating_duration_since(Instant::now());
            match lines.recv_timeout(wait) {
                Ok(line) if line["type"] == "machines" => {
                    return Ok(line["machines"].as_array().cloned().unwrap_or_default());
                }
                Ok(line) if line["type"] == "error" && line["code"] == "protocol-mismatch" => {
                    return Err("the server speaks another version of the protocol: run aria hub update".to_string());
                }
                Ok(line) if line["type"] == "error" && line["id"] == 1 => {
                    return Err("this server is too old to list its machines: run aria hub update".to_string());
                }
                Ok(_) => {}
                Err(_) => return Err("the server did not answer".to_string()),
            }
        }
    })();
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
    result
}

/// `HH:MM` of a time given as seconds since 1970, in this computer's time zone.
pub fn clock(epoch: u64) -> String {
    let time = epoch as libc::time_t;
    // SAFETY: localtime_r fills the struct it is given and keeps nothing.
    let mut parts: libc::tm = unsafe { std::mem::zeroed() };
    let filled = unsafe { libc::localtime_r(&time, &mut parts) };
    if filled.is_null() {
        return "?".to_string();
    }
    format!("{:02}:{:02}", parts.tm_hour, parts.tm_min)
}

/// One line of the table `aria hub machines` prints: `laptop  linux 0.22.0  online since 10:42  3 sessions`.
pub fn machine_row(machine: &Value, widths: (usize, usize)) -> String {
    let name = machine["name"].as_str().unwrap_or("?");
    let os = machine["os"].as_str().unwrap_or("?");
    let version = machine["version"].as_str().unwrap_or("?");
    let online = machine["online"] == true;
    let sessions = machine["sessions"].as_u64().unwrap_or(0);
    let state = if online {
        match machine["since"].as_u64() {
            Some(since) => format!("online since {}", clock(since)),
            None => "online".to_string(),
        }
    } else {
        "offline".to_string()
    };
    let software = format!("{os} {version}");
    format!(
        "{name:<w0$}  {software:<w1$}  {state:<20}  {sessions} session{}",
        if sessions == 1 { "" } else { "s" },
        w0 = widths.0,
        w1 = widths.1
    )
    .trim_end()
    .to_string()
}

pub fn machines_table(machines: &[Value]) -> Vec<String> {
    let name_width =
        machines.iter().filter_map(|m| m["name"].as_str()).map(|name| name.chars().count()).max().unwrap_or(0);
    let software_width = machines
        .iter()
        .map(|m| {
            format!("{} {}", m["os"].as_str().unwrap_or("?"), m["version"].as_str().unwrap_or("?")).chars().count()
        })
        .max()
        .unwrap_or(0);
    machines.iter().map(|machine| machine_row(machine, (name_width, software_width))).collect()
}

/// `aria hub machines [<user@host>]`: the machines linked to a hub.
pub fn machines_command(args: &[String]) -> i32 {
    let mut host = None;
    for arg in args {
        match arg.as_str() {
            _ if host.is_none() && !arg.starts_with('-') => host = Some(arg.clone()),
            _ if arg.starts_with('-') => {
                eprintln!("aria hub machines: unknown option {arg}\n{MACHINES_USAGE}");
                return 2;
            }
            _ => {
                eprintln!("aria hub machines: only one host\n{MACHINES_USAGE}");
                return 2;
            }
        }
    }
    let Some(host) = host.or_else(crate::connect::default_hub) else {
        eprintln!("aria hub machines: no hub is linked to this computer: run `aria hub init user@host`");
        return 2;
    };
    match fetch_machines(&host, Duration::from_secs(30)) {
        Ok(machines) if machines.is_empty() => {
            println!("No machine is linked to {host} yet. `aria hub init {host}` makes this computer one.");
            0
        }
        Ok(machines) => {
            println!("Machines linked to {host}:");
            for row in machines_table(&machines) {
                println!("  {row}");
            }
            0
        }
        Err(reason) => {
            eprintln!("aria hub machines: {host}: {reason}");
            1
        }
    }
}

pub const MACHINES_USAGE: &str = "usage: aria hub machines [<user@host>]";

// ---- unlinking -----------------------------------------------------------------------------------------------------

pub const UNLINK_USAGE: &str = "usage: aria hub unlink-machine <name> [<user@host>] [--details] [--plain] [--yes]";

#[derive(Debug, Default, PartialEq)]
pub struct RawUnlink {
    pub name: String,
    pub host: Option<String>,
    pub details: bool,
    pub plain: bool,
    pub yes: bool,
}

pub fn parse_unlink_args(args: &[String]) -> Result<RawUnlink, String> {
    let mut raw = RawUnlink::default();
    let mut name = None;
    for arg in args {
        match arg.as_str() {
            "--yes" | "-y" => raw.yes = true,
            "--details" => raw.details = true,
            "--plain" => raw.plain = true,
            _ if arg.starts_with('-') && name.is_some() && raw.host.is_none() && !arg.contains('@') => {
                return Err(format!("unknown option {arg}"))
            }
            // The host may look like an option: the host check refuses it the way `aria connect` does.
            _ if name.is_none() => name = Some(arg.clone()),
            _ if raw.host.is_none() => raw.host = Some(arg.clone()),
            _ => return Err("one machine name and at most one host".into()),
        }
    }
    raw.name = name.ok_or("a machine name is needed")?;
    if !crate::machines::valid_name(&raw.name) {
        return Err(format!("{:?} is not a machine name (lowercase letters, digits and dashes)", raw.name));
    }
    Ok(raw)
}

/// Removes the lines of `authorized_keys` whose forced command is exactly `aria hub-link machine <name>`, and says how many:
/// `removed N`, or `not linked`. The name is the first line of stdin (checked here too), nothing is interpolated.
const REMOTE_UNLINK_MACHINE: &str = concat!(
    r#"umask 077; f="$HOME/.ssh/authorized_keys"; IFS= read -r name "#,
    r#"&& case "$name" in ""|-*|*[!a-z0-9-]*) echo "bad name"; exit 2 ;; esac "#,
    r#"&& if [ ! -f "$f" ]; then echo "not linked"; exit 0; fi "#,
    r#"&& pre="restrict,command=\"$HOME/.local/bin/aria hub-link machine $name\" " "#,
    r#"&& n=$(grep -cF "$pre" "$f" | head -n 1); "#,
    r#"if [ "${n:-0}" = 0 ]; then echo "not linked"; exit 0; fi; "#,
    r#"tmp=$(mktemp) || exit 1; "#,
    r#"awk -v pre="$pre" 'index($0, pre) == 1 { next } { print }' "$f" > "$tmp" || { rm -f "$tmp"; exit 1; }; "#,
    // The file that lets the owner in is edited: a copy of it as it was stays beside it (same mode), for the one time it matters.
    r#"cp -p "$f" "$f.aria-backup" || { rm -f "$tmp"; exit 1; }; "#,
    r#"cat "$tmp" > "$f"; rm -f "$tmp"; echo "removed $n""#,
);

/// What unlinking does on this computer: the link file when it is this machine's, and the key files it names when they are the
/// ones Aria made (under `~/.ssh`, called `aria-machine-*`). Returns what was removed.
pub fn forget_here(root: &Path, name: &str) -> Vec<String> {
    let mut removed = Vec::new();
    let Ok(Some(config)) = LinkConfig::load(root) else { return removed };
    if config.name != name {
        return removed;
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let ssh_dir = home.map(|home| home.join(".ssh"));
    let private = PathBuf::from(&config.key);
    let ours = |path: &Path| {
        let base = path.file_name().and_then(|base| base.to_str()).unwrap_or("");
        base.starts_with("aria-machine-") && ssh_dir.as_deref().is_some_and(|dir| path.parent() == Some(dir))
    };
    if std::fs::remove_file(LinkConfig::path(root)).is_ok() {
        removed.push("~/.aria/link.json".to_string());
    }
    for path in [private.clone(), PathBuf::from(format!("{}.pub", private.display()))] {
        if ours(&path) && std::fs::remove_file(&path).is_ok() {
            removed.push(tilde(&path));
        }
    }
    removed
}

pub fn unlink_machine(raw: RawUnlink) -> i32 {
    let mut ui = hub_ui::ui_for_terminal(raw.plain);
    match run_unlink(raw, ui.as_mut()) {
        Ok(code) => code,
        Err(Failure(code, message)) => {
            eprintln!("aria hub unlink-machine: {message}");
            code
        }
    }
}

fn run_unlink(raw: RawUnlink, ui: &mut dyn HubUi) -> Result<i32, Failure> {
    let host = match raw.host.clone().or_else(crate::connect::default_hub) {
        Some(host) => host,
        None => {
            return Err(usage_error(
                "no hub is linked to this computer: give the host, or run `aria hub init user@host`",
            ))
        }
    };
    crate::connect::check_host(&host).map_err(|_| usage_error(format!("{host}: not a host")))?;
    let root = AriaPaths::from_env().root;
    let here = matches!(LinkConfig::load(&root), Ok(Some(config)) if config.name == raw.name);
    ui.title("unlinking a machine");
    let mut steps = vec![format!("Remove {}'s key from {host}, so it can no longer link", raw.name)];
    if here {
        steps.push("Stop linking this computer, and remove the link's key made for it".to_string());
    }
    let mut good_to_know = vec![format!(
        "{} stays on the server's list as offline until the server restarts; its projects and sessions stay on it.",
        raw.name
    )];
    if here {
        good_to_know.push("This computer's own Aria keeps working on its own.".to_string());
    }
    if raw.details {
        ui.details(&[format!(
            "On {host}, with your own ssh login: remove from ~/.ssh/authorized_keys the lines that start with \
             restrict,command=\"<hub home>/.local/bin/aria hub-link machine {}\"",
            raw.name
        )]);
    }
    ui.plan(&Plan { host: host.clone(), steps, good_to_know, blocked: Vec::new() });
    if !raw.yes && !ui.confirm() {
        ui.declined();
        return Ok(0);
    }
    ui.step_start(1, if here { 2 } else { 1 }, &format!("Removing {}'s key from the server", raw.name), "");
    let output = remote(&host, REMOTE_UNLINK_MACHINE, Some(format!("{}\n", raw.name).as_bytes()))
        .map_err(|error| failed(format!("{host}: {}", describe_error(&error))))?;
    if !output.status.success() {
        ui.failed("The key was not removed.");
        return Err(failed(format!("{host}: {} {}", describe(output.status), first_line(&output.stderr))));
    }
    let said = first_line(&output.stdout);
    let was_linked = said.starts_with("removed");
    if was_linked {
        ui.step_ok(&format!("{} can no longer link to {host}", raw.name));
    } else {
        ui.step_ok(&format!("{} was not linked to {host}", raw.name));
    }
    if here {
        ui.step_start(2, 2, "Stopping the link on this computer", "");
        let removed = forget_here(&root, &raw.name);
        if removed.is_empty() {
            ui.step_info("Nothing to remove here.");
        } else {
            ui.step_ok(&format!("Removed {}", removed.join(", ")));
        }
    }
    ui.done(&Outcome::Unlinked { name: raw.name.clone(), host, was_linked });
    Ok(0)
}

// ---- status --------------------------------------------------------------------------------------------------------

/// What this computer's link says, in plain words.
pub fn link_summary(root: &Path) -> Vec<String> {
    let config = LinkConfig::load(root);
    let reported = link::reported(root);
    match (config, reported) {
        (Err(reason), _) => vec![format!("This computer's link file is wrong: {reason}")],
        (Ok(None), _) => {
            vec!["This computer is not a machine of any hub (`aria hub init user@host` makes it one).".to_string()]
        }
        (Ok(Some(config)), reported) => {
            let mut lines = vec![format!("This computer is set to link to {} as {}.", config.hub, config.name)];
            lines.push(match reported {
                Some(state) if state.alive && state.state == "connected" => "The link is up.".to_string(),
                Some(state) if state.alive && state.state == "connecting" => "The link is opening.".to_string(),
                Some(state) if state.alive => format!(
                    "The link is down: {}. It tries again by itself.",
                    state.reason.unwrap_or_else(|| "no reason given".to_string())
                ),
                _ => "Aria is not running here, so the link is not up: it links when Aria starts.".to_string(),
            });
            lines
        }
    }
}

/// `aria hub status`: this computer's link and, for each hub it knows, whether it answers and which Aria it runs.
pub fn status_command(args: &[String]) -> i32 {
    if let Some(arg) = args.first() {
        eprintln!("aria hub status: unexpected {arg}\naria hub status");
        return 2;
    }
    let root = AriaPaths::from_env().root;
    for line in link_summary(&root) {
        println!("{line}");
    }
    let hubs = match crate::connect::Hubs::load(&root) {
        Ok(hubs) => hubs,
        Err(error) => {
            eprintln!("aria hub status: {error}");
            return 1;
        }
    };
    if hubs.hubs.is_empty() {
        println!("No hub is remembered here.");
        return 0;
    }
    let mut code = 0;
    for host in hubs.hubs.keys() {
        let default = if hubs.default.as_deref() == Some(host.as_str()) { " (the default)" } else { "" };
        match crate::connect::open_within(host, Duration::from_secs(15)) {
            Ok(link) => println!(
                "{host}{default} answers, and runs {}.",
                link.hub_version()
                    .map(|version| format!("Aria {version}"))
                    .unwrap_or_else(|| "an Aria too old to say".to_string())
            ),
            Err(error) => {
                println!("{host}{default} does not answer: {}", connection_problem(&error));
                code = 1;
            }
        }
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_machines_key_has_no_passphrase_and_a_name_of_its_own() {
        let args = machine_keygen_args(Path::new("/home/ana/.ssh/aria-machine-laptop"), "laptop");
        assert_eq!(
            args,
            ["-q", "-t", "ed25519", "-N", "", "-f", "/home/ana/.ssh/aria-machine-laptop", "-C", "aria-machine-laptop"]
        );
    }

    #[test]
    fn the_authorized_line_of_a_machine_forces_its_one_command() {
        assert_eq!(
            authorized_machine_line("/home/ubuntu/", "laptop", "ssh-ed25519 AAAA aria-machine-laptop"),
            "restrict,command=\"/home/ubuntu/.local/bin/aria hub-link machine laptop\" ssh-ed25519 AAAA aria-machine-laptop"
        );
    }

    #[test]
    fn the_table_lines_up_and_says_what_each_machine_is_doing() {
        let machines = vec![
            json!({"name": "laptop", "os": "linux", "version": "0.22.0", "online": true, "since": 0, "sessions": 3}),
            json!({"name": "mac", "os": "macos", "version": "0.22.0", "online": false, "sessions": 1}),
        ];
        let rows = machines_table(&machines);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].starts_with("laptop  linux 0.22.0  online since "), "{}", rows[0]);
        assert!(rows[0].ends_with("3 sessions"), "{}", rows[0]);
        assert!(rows[1].starts_with("mac     macos 0.22.0  offline"), "{}", rows[1]);
        assert!(rows[1].ends_with("1 session"), "{}", rows[1]);
        assert!(rows.iter().all(|row| row.chars().count() <= 70));
    }

    #[test]
    fn the_clock_is_hours_and_minutes() {
        let text = clock(1_700_000_000);
        assert_eq!(text.len(), 5, "{text}");
        assert_eq!(&text[2..3], ":");
    }

    #[test]
    fn the_unlink_arguments_need_a_valid_name() {
        let args = |items: &[&str]| parse_unlink_args(&items.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(args(&["laptop"]).unwrap().name, "laptop");
        let both = args(&["laptop", "ana@hub", "--yes"]).unwrap();
        assert_eq!((both.host.as_deref(), both.yes), (Some("ana@hub"), true));
        assert!(args(&[]).is_err());
        assert!(args(&["Laptop"]).is_err());
        assert!(args(&["laptop; ls"]).is_err());
        assert!(args(&["laptop", "a", "b"]).is_err());
        assert!(args(&["laptop", "ana@hub", "--nope"]).is_err());
    }

    #[test]
    fn unlinking_here_removes_only_what_aria_made() {
        let dir = std::env::temp_dir().join(format!("aria-unlink-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let ssh = dir.join(".ssh");
        let root = dir.join(".aria");
        std::fs::create_dir_all(&ssh).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        let theirs = ssh.join("id_ed25519");
        let ours = ssh.join("aria-machine-laptop");
        for file in [&theirs, &ours, &ssh.join("aria-machine-laptop.pub")] {
            std::fs::write(file, "k").unwrap();
        }
        // SAFETY of the environment: tests in this module do not run in parallel with ones reading HOME.
        let saved = std::env::var_os("HOME");
        std::env::set_var("HOME", &dir);
        LinkConfig { hub: "ana@hub".into(), name: "laptop".into(), key: ours.to_string_lossy().into_owned() }
            .save(&root)
            .unwrap();
        // another name: nothing is touched
        assert!(forget_here(&root, "desk").is_empty());
        assert!(ours.exists() && LinkConfig::path(&root).exists());
        let removed = forget_here(&root, "laptop");
        assert_eq!(removed.len(), 3, "{removed:?}");
        assert!(!ours.exists() && !LinkConfig::path(&root).exists());
        // a key that is not one Aria made is never removed, even when the link file names it
        LinkConfig { hub: "ana@hub".into(), name: "laptop".into(), key: theirs.to_string_lossy().into_owned() }
            .save(&root)
            .unwrap();
        let removed = forget_here(&root, "laptop");
        assert_eq!(removed, vec!["~/.aria/link.json".to_string()]);
        assert!(theirs.exists(), "a key of the owner's is never touched");
        match saved {
            Some(home) => std::env::set_var("HOME", home),
            None => std::env::remove_var("HOME"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_summary_says_where_the_link_is() {
        let dir = tempfile::tempdir().unwrap();
        assert!(link_summary(dir.path())[0].contains("not a machine"));
        LinkConfig { hub: "ana@hub".into(), name: "laptop".into(), key: "/k".into() }.save(dir.path()).unwrap();
        let lines = link_summary(dir.path());
        assert!(lines[0].contains("link to ana@hub as laptop"), "{lines:?}");
        assert!(lines[1].contains("not running"), "{lines:?}");
        std::fs::write(
            dir.path().join(link::STATE_FILE),
            format!(r#"{{"state":"connected","pid":{}}}"#, std::process::id()),
        )
        .unwrap();
        assert_eq!(link_summary(dir.path())[1], "The link is up.");
        std::fs::write(
            dir.path().join(link::STATE_FILE),
            format!(r#"{{"state":"error","reason":"Permission denied","pid":{}}}"#, std::process::id()),
        )
        .unwrap();
        assert!(link_summary(dir.path())[1].contains("Permission denied"));
        std::fs::write(LinkConfig::path(dir.path()), "{").unwrap();
        assert!(link_summary(dir.path())[0].contains("wrong"));
    }
}
