//! `aria hub update`: brings the hubs this computer is linked to up to this Aria's version. It is the install step of
//! `hub init` and the version check, and nothing else: no sign-ins or settings are sent, no key is made, `authorized_keys`
//! and `hubs.json` are not touched. A child of `hub_admin` so it uses that module's helpers instead of copying them.
//!
//! A hub that was linked with its key (`hubs.json`) and has an Aria that knows `hubUpdate` updates **itself**: this computer
//! opens the same terminal bridge `aria connect` uses, with that restricted key, and asks the service to install this Aria's
//! release (`hub_update.rs` is the service's side). No ssh login of the owner's is used. Everything else keeps the owner's own
//! ssh login, as `hub init` does, since the restricted key cannot run an installer: a build uploaded by hand (`--binary`, or a
//! checkout's), a hub that has no key here or cannot be opened with it, and a hub too old to know `hubUpdate` (said, once).

use super::*;
use crate::connect::{self, Hubs};
use crate::term::link::SSH_HELLO_TIMEOUT;
use crate::version;
use aria_core::bridge::{DockMessage, PageMessage};
use std::cmp::Ordering;
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub const UPDATE_USAGE: &str =
    "usage: aria hub update [<user@host>] [--all] [--binary <path>] [--details] [--plain] [--yes]";

#[derive(Debug, Default, PartialEq)]
pub struct RawUpdate {
    /// The hub to update; none means the default one (or every one with `--all`).
    pub host: Option<String>,
    pub all: bool,
    pub binary: Option<PathBuf>,
    pub details: bool,
    pub plain: bool,
    pub yes: bool,
}

pub fn parse_update_args(args: &[String]) -> Result<RawUpdate, String> {
    let mut raw = RawUpdate::default();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--yes" | "-y" => raw.yes = true,
            "--all" => raw.all = true,
            "--details" => raw.details = true,
            "--plain" => raw.plain = true,
            "--binary" => {
                raw.binary = Some(PathBuf::from(rest.next().ok_or_else(|| "--binary needs a value".to_string())?))
            }
            // Anything else is the host, even one that looks like an option: the host check refuses that one the way
            // `aria connect` does.
            _ if raw.host.is_none() => raw.host = Some(arg.clone()),
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}")),
            _ => return Err("only one host (or --all)".into()),
        }
    }
    if raw.all && raw.host.is_some() {
        return Err("--all and a host contradict each other".into());
    }
    Ok(raw)
}

/// `aria hub update`: the exit status of the run.
pub fn update(raw: RawUpdate) -> i32 {
    let mut ui = hub_ui::ui_for_terminal(raw.plain);
    match run_update(raw, ui.as_mut()) {
        Ok(code) => code,
        Err(Failure(code, message)) => {
            eprintln!("aria hub update: {message}");
            code
        }
    }
}

/// The hubs to update, in order: the one named, every linked one (`--all`) or the default.
fn hubs_to_update(raw: &RawUpdate, hubs: &Hubs) -> Result<Vec<String>, Failure> {
    let nothing = || {
        usage_error(
            "no hub is set up on this computer: run `aria hub init user@host` to set one up, or give the host: \
             aria hub update user@host",
        )
    };
    if let Some(host) = &raw.host {
        return Ok(vec![host.clone()]);
    }
    if raw.all {
        let all: Vec<String> = hubs.hubs.keys().cloned().collect();
        return if all.is_empty() { Err(nothing()) } else { Ok(all) };
    }
    hubs.default.clone().filter(|host| hubs.hubs.contains_key(host)).map(|host| vec![host]).ok_or_else(nothing)
}

/// What a hub is before anything is changed.
#[derive(Debug, PartialEq)]
enum Probed {
    /// Runs this Aria's version.
    Current(String),
    /// Runs a newer one.
    Newer(String),
    /// Runs an older one (or one that cannot be read as a version): it gets updated.
    Older { from: String, active: bool },
}

/// The version as the owner reads it: `0.21.0` when it parses, else what the hub said.
fn shown(raw: &str) -> String {
    version::parse(raw).map(version::show).unwrap_or_else(|| raw.to_string())
}

/// How a hub that runs `hub` compares with `mine`; the service state is read by the caller.
fn classify(hub: &str, mine: &str, active: bool) -> Probed {
    match version::compare(hub, mine) {
        Some(Ordering::Equal) => Probed::Current(shown(hub)),
        Some(Ordering::Greater) => Probed::Newer(shown(hub)),
        _ => Probed::Older { from: shown(hub), active },
    }
}

/// Reads the hub, changing nothing: its CPU (against a binary that is to be uploaded), the Aria it runs and whether its
/// service is up.
fn probe_hub(host: &str, binary_arch: Option<&str>) -> Result<Probed, Failure> {
    let arch = probe_text(host, "uname -m")?;
    if arch != "aarch64" && arch != "x86_64" {
        return Err(failed(format!("{host} is {arch:?}: only aarch64 and x86_64 are supported")));
    }
    if let Some(binary_arch) = binary_arch.filter(|binary_arch| *binary_arch != arch) {
        return Err(failed(format!(
            "the binary you gave is built for {binary_arch} but {host} is {arch}: build one for it with `sh \
             tools/build-headless.sh {}` and pass it with --binary",
            triple_of(&arch)
        )));
    }
    let Some(running) = hub_aria_version(host) else {
        return Err(failed(format!("{host} has no Aria that can be read: run `aria hub init {host}` to set it up")));
    };
    let active = probe_text(host, &user_env(IS_ACTIVE_COMMAND))?.lines().next().map(str::trim) == Some("active");
    Ok(classify(&running, version::mine(), active))
}

/// How a hub gets the new Aria.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Via {
    /// The service installs it itself, asked through the terminal bridge with the key `hub init` linked this computer with.
    Itself,
    /// The owner's own ssh login runs the installer (or uploads a build), as `hub init` does.
    Login,
}

/// One hub that gets updated: its name, the version it runs, whether its service is up and how it is updated.
struct Todo {
    host: String,
    from: String,
    active: bool,
    via: Via,
}

/// How long this computer waits at each point of asking a hub to update itself.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Waits {
    /// For the first word back: a hub that says nothing in this time does not know `hubUpdate`.
    silence: Duration,
    /// For a hub whose connection dropped mid-update (the service restarts into the new binary): how long to try again.
    come_back: Duration,
    /// Between those tries.
    poll: Duration,
    /// For the check after a good result: how long to wait for the running service to be the new one.
    confirm: Duration,
}

impl Waits {
    fn standard() -> Waits {
        Waits {
            silence: Duration::from_secs(20),
            come_back: Duration::from_secs(90),
            poll: Duration::from_secs(3),
            confirm: Duration::from_secs(10),
        }
    }

    /// `ARIA_SELF_UPDATE_WAIT_MS=n` (how the tests make them short) makes every wait a multiple of n milliseconds.
    fn from_env() -> Waits {
        match std::env::var("ARIA_SELF_UPDATE_WAIT_MS").ok().and_then(|ms| ms.parse::<u64>().ok()) {
            Some(ms) => Waits {
                silence: Duration::from_millis(ms),
                come_back: Duration::from_millis(ms * 20),
                poll: Duration::from_millis(ms),
                confirm: Duration::from_millis(ms * 5),
            },
            None => Waits::standard(),
        }
    }
}

/// How long one look at a hub that is restarting waits for its hello (it is tried again after `Waits::poll`).
const CHECK_HELLO: Duration = Duration::from_secs(30);

/// The longest the service may take (its own limit is 15 minutes) and a minute to say so.
const UPDATE_LIMIT: Duration = Duration::from_secs(16 * 60);

/// The plan in plain words: two steps for each hub, and what to know.
fn plan_for(todo: &[Todo], release: bool, mine: &str) -> Plan {
    let many = todo.len() > 1;
    let mut steps = Vec::new();
    for hub in todo {
        steps.push(if release {
            let how = match hub.via {
                Via::Itself => "the server updates itself",
                Via::Login => "through your ssh login",
            };
            format!("Update Aria on {} from {} to {mine} ({how})", hub.host, hub.from)
        } else {
            format!("Update Aria on {} from {} to the build you gave me", hub.host, hub.from)
        });
        steps.push(if many {
            format!("Check the new version on {}", hub.host)
        } else {
            "Check the new version".to_string()
        });
    }
    let mut good_to_know = vec!["Nothing is sent: no sign-ins, no settings, no key.".to_string()];
    if todo.iter().any(|hub| hub.active) {
        good_to_know.push(
            if many {
                "The servers keep running; each takes the new version when idle."
            } else {
                "The server keeps running; it takes the new version when idle."
            }
            .to_string(),
        );
    }
    if todo.iter().any(|hub| hub.via == Via::Login) {
        good_to_know
            .push("The server's installer runs again; its sudo is used only if a package is missing.".to_string());
    }
    if todo.iter().any(|hub| hub.via == Via::Itself) {
        good_to_know.push(
            if many {
                "A server that updates itself does it with the key Aria linked it with: no ssh login of yours is used."
            } else {
                "The server updates itself with the key Aria linked it with: no ssh login of yours is used."
            }
            .to_string(),
        );
    }
    let host = if many { "your servers".to_string() } else { todo[0].host.clone() };
    Plan { host, steps, good_to_know, blocked: Vec::new() }
}

/// The technical plan (`--details`): the commands that run, one paragraph for each hub.
fn details_for(todo: &[Todo], release: bool, mine: &str) -> Vec<String> {
    todo.iter()
        .map(|hub| {
            let install = if release && hub.via == Via::Itself {
                format!(
                    "ask its service to install release v{mine} through the terminal bridge (`aria client-stdio`, the \
                     restricted key: `hubUpdate` {{\"version\": \"{mine}\"}}); the service runs the release's installer \
                     itself: {}",
                    release_install_command(mine)
                )
            } else if release {
                format!("run the installer of release v{mine} on it: {}", release_install_command(mine))
            } else {
                format!("upload the build you gave me and the checkout's installer to {HUB_SRC}, then run: {INSTALL_COMMAND}")
            };
            format!(
                "{}: read its version ({REMOTE_VERSION_BEFORE}), {install}, then read the version again \
                 ({REMOTE_VERSION}). It sends no settings or credentials and changes no key.",
                hub.host
            )
        })
        .collect()
}

/// One line of the installer's output as the screen shows it under a step.
fn shorten(line: &str) -> String {
    let line = line.trim();
    if line.chars().count() <= 70 {
        return line.to_string();
    }
    let mut cut: String = line.chars().take(69).collect();
    cut.push('…');
    cut
}

/// The Aria a hub runs, from the hello of its terminal bridge, opened with the key `hub init` linked this computer with.
/// `Ok(None)`: it opened and said no version (a hub that old cannot update itself).
fn hello_version(host: &str, within: Duration) -> Result<Option<String>, String> {
    let link = connect::open_within(host, within).map_err(|error| connection_problem(&error))?;
    Ok(link.hub_version().map(str::to_string))
}

/// What asking a hub to update itself came to.
#[derive(Debug, PartialEq)]
enum Asked {
    /// The hub has this Aria's version installed.
    Updated,
    /// It could not (or would not) do it: the plain reason, and the owner's ssh login does it instead.
    Fallback(String),
}

/// The hub's connection dropped while it updated (the service restarts into the new binary once it is idle, and the line may
/// go with it): the answer is whether it comes back running this Aria's version.
fn wait_for_the_hub(host: &str, waits: Waits, ui: &mut dyn HubUi) -> Result<Asked, Failure> {
    let mine = version::mine();
    ui.step_info("The connection dropped; waiting for the server to come back.");
    let until = Instant::now() + waits.come_back;
    while Instant::now() < until {
        std::thread::sleep(waits.poll);
        if let Ok(Some(running)) = hello_version(host, CHECK_HELLO) {
            if version::compare(&running, mine) == Some(Ordering::Equal) {
                return Ok(Asked::Updated);
            }
        }
    }
    Err(failed(format!(
        "{host}: the connection was lost during the update and the server does not answer with Aria {mine} yet: \
         running `aria hub update` again is safe"
    )))
}

/// Asks the service of `host` to install this Aria's release, through the restricted key, and shows what its installer
/// prints. Nothing is run over the owner's own login.
fn ask_hub(host: &str, waits: Waits, ui: &mut dyn HubUi) -> Result<Asked, Failure> {
    let mine = version::mine();
    let link = match connect::open_within(host, SSH_HELLO_TIMEOUT) {
        Ok(link) => link,
        Err(error) => {
            return Ok(Asked::Fallback(format!(
                "could not open the server with its key ({}); using your ssh login",
                connection_problem(&error)
            )))
        }
    };
    let events = link.events();
    link.send(&PageMessage::HubUpdate { id: 1, version: mine.to_string() });
    let too_old = || Asked::Fallback("this server is too old to update itself; using your ssh login once".to_string());
    let started = Instant::now();
    let mut heard = false;
    loop {
        // A hub that knows `hubUpdate` answers at once (a line, or a refusal); one that does not says nothing or an error.
        let wait = if heard { UPDATE_LIMIT.saturating_sub(started.elapsed()) } else { waits.silence };
        match events.recv_timeout(wait) {
            Ok(DockMessage::HubUpdateProgress { line, .. }) => {
                heard = true;
                ui.step_info(&shorten(&line));
            }
            Ok(DockMessage::HubUpdateResult { ok: true, version: said, .. }) => {
                return match said.as_deref().map(|said| version::compare(said, mine)) {
                    Some(Some(Ordering::Equal)) => Ok(Asked::Updated),
                    _ => Err(failed(format!(
                        "{host}: the server said it updated but not to Aria {mine} ({}): running `aria hub update` \
                         again may finish it",
                        said.as_deref().unwrap_or("no version")
                    ))),
                };
            }
            Ok(DockMessage::HubUpdateResult { reason, .. }) => {
                let reason = reason.unwrap_or_else(|| "it gave no reason".to_string());
                // Someone updated it between the look and the ask: it is what was wanted.
                if reason == format!("already runs {mine}") {
                    return Ok(Asked::Updated);
                }
                return Err(failed(format!("{host}: the server could not update itself: {reason}")));
            }
            Ok(DockMessage::Error { code, .. }) if code == "not-for-clients" && !heard => return Ok(too_old()),
            Ok(DockMessage::Error { code, .. }) if code == "gone" => {
                drop(link);
                return if heard { wait_for_the_hub(host, waits, ui) } else { Ok(too_old()) };
            }
            Ok(_) => {}
            Err(mpsc::RecvTimeoutError::Timeout) if !heard => return Ok(too_old()),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(failed(format!(
                    "{host}: the update is taking too long: running `aria hub update` again is safe"
                )))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                drop(link);
                return if heard { wait_for_the_hub(host, waits, ui) } else { Ok(too_old()) };
            }
        }
    }
}

/// After a hub updated itself: opens it again and says which Aria its service runs. It runs the new one once it has restarted,
/// which it does by itself when it is idle, so an old one is not a failure: the new Aria is installed.
fn confirm_running(host: &str, waits: Waits, ui: &mut dyn HubUi) {
    let mine = version::mine();
    let until = Instant::now() + waits.confirm;
    let mut seen: Option<String> = None;
    loop {
        if let Ok(Some(running)) = hello_version(host, CHECK_HELLO) {
            if version::compare(&running, mine) == Some(Ordering::Equal) {
                ui.step_ok(&format!("The server runs Aria {mine}"));
                return;
            }
            seen = Some(shown(&running));
        }
        if Instant::now() >= until {
            break;
        }
        std::thread::sleep(waits.poll);
    }
    match seen {
        Some(running) => ui.step_ok(&format!(
            "Aria {mine} is installed; the server runs {running} until it is idle, then restarts into it by itself"
        )),
        None => ui.step_info(&format!("Aria {mine} is installed; the server could not be reached again just now")),
    }
}

/// Updates one hub that the plan listed: the install step and the version check. The version it runs now.
fn update_one(
    hub: &Todo,
    source: &InstallSource<'_>,
    steps: (usize, usize),
    many: bool,
    ui: &mut dyn HubUi,
) -> Result<String, Failure> {
    let host = hub.host.as_str();
    let (number, total) = steps;
    let title = format!("Updating Aria on {host}");
    let checking =
        if many { format!("Checking the new version on {host}") } else { "Checking the new version".to_string() };
    let mut announce = true;
    if hub.via == Via::Itself {
        let waits = Waits::from_env();
        let mine = version::mine();
        ui.step_start(
            number,
            total,
            &title,
            &format!("The server downloads Aria {mine} and sets itself up. Nothing is asked of you."),
        );
        match ask_hub(host, waits, ui)? {
            Asked::Updated => {
                ui.step_ok("Aria is updated on the server");
                ui.step_start(number + 1, total, &checking, "Opens the server again and reads the version it runs.");
                confirm_running(host, waits, ui);
                return Ok(shown(mine));
            }
            Asked::Fallback(reason) => {
                ui.step_info(&reason);
                announce = false;
            }
        }
    }
    let stopped = |step: usize, what: &str, detail: String| {
        failed(format!(
            "{host}: step {} ({what}) failed: {detail}. The installer may have partly run on the hub; running \
             `aria hub update` again is safe",
            if step == 1 { "upload" } else { "installer" }
        ))
    };
    install_on_hub(host, source, &title, (number, total), ui, announce, &stopped)?;
    ui.step_ok("Aria is updated on the server");
    if hub.active {
        ui.step_ok("The service keeps running (it takes the new version when idle)");
    }
    ui.step_start(number + 1, total, &checking, "Reads the version the server runs now.");
    let output = remote(host, REMOTE_VERSION, None)
        .map_err(|error| failed(format!("{host}: the version check failed: {}", describe_error(&error))))?;
    if !output.status.success() {
        return Err(failed(format!(
            "{host}: the version check failed: {} {}",
            describe(output.status),
            first_line(&output.stderr)
        )));
    }
    let said = first_line(&output.stdout);
    let now = shown(said.strip_prefix("aria ").unwrap_or(&said));
    // A release is installed at this Aria's own version: anything else after it means the update did not take. A build
    // uploaded by hand has a version of its own, which this computer cannot read from an aarch64 binary.
    if source.release.is_some() && version::compare(&now, version::mine()) != Some(Ordering::Equal) {
        return Err(failed(format!(
            "{host} runs Aria {now} after the update, not {}: running `aria hub update` again may finish it",
            version::mine()
        )));
    }
    ui.step_ok(&format!("The server runs Aria {now}"));
    Ok(now)
}

fn run_update(raw: RawUpdate, ui: &mut dyn HubUi) -> Result<i32, Failure> {
    let hubs = Hubs::load(&AriaPaths::from_env().root).map_err(|error| failed(error.to_string()))?;
    let hosts = hubs_to_update(&raw, &hubs)?;
    for host in &hosts {
        crate::connect::check_host(host).map_err(|_| usage_error(format!("{host}: not a host")))?;
    }
    let mut starts: Vec<PathBuf> = std::env::current_dir().into_iter().collect();
    starts.extend(std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf)));
    // Where the new Aria comes from, as for `hub init`: a binary built here (with `--binary`, or a checkout holding the
    // headless build), else this Aria's own release.
    let (binary, repo) = choose_source(raw.binary.clone(), find_repo(&starts), |path| path.is_file())?;
    let binary_arch = match &binary {
        Some(binary) => Some(read_binary_arch(binary)?),
        None => None,
    };
    let mine = version::mine();
    let release = binary.is_none();

    ui.title(if hosts.len() > 1 { "updating your servers" } else { "updating your server" });
    let mut failures = 0;
    let mut runs: Vec<(String, String)> = Vec::new();
    let mut todo: Vec<Todo> = Vec::new();
    for host in &hosts {
        ui.looking(host);
        // A hub linked with its key, updated from a release, is read through that key: the hello says the version, and no ssh
        // login of the owner's is asked for. Anything else is read the way `hub init` reads a hub.
        let mut via = Via::Login;
        let mut probed = None;
        if release && connect::remembered_key(host).is_some() {
            match hello_version(host, SSH_HELLO_TIMEOUT) {
                Ok(Some(version)) => {
                    via = Via::Itself;
                    probed = Some(Ok(classify(&version, mine, true)));
                }
                Ok(None) => ui.probe_note(
                    &Note::text(format!("{host} did not say which Aria it runs.")).then_text("Using your ssh login."),
                ),
                Err(problem) => ui.probe_note(
                    &Note::text(format!("Could not open {host} with its key: {problem}"))
                        .then_text("Using your ssh login."),
                ),
            }
        }
        match probed.unwrap_or_else(|| probe_hub(host, binary_arch)) {
            Ok(Probed::Current(now)) => {
                ui.probe_ok(&format!("{host} already runs Aria {now}"));
                runs.push((host.clone(), now));
            }
            Ok(Probed::Newer(theirs)) => {
                let text = format!(
                    "The server runs a newer Aria ({theirs}) than this one ({mine}). Update this computer first."
                );
                eprintln!("aria hub update: {host}: {text}");
                ui.failed(&format!("{host} was not updated."));
                failures += 1;
            }
            Ok(Probed::Older { from, active }) => {
                ui.probe_ok(&format!("{host} runs Aria {from}"));
                todo.push(Todo { host: host.clone(), from, active, via });
            }
            Err(Failure(_, message)) => {
                eprintln!("aria hub update: {host}: {message}");
                ui.failed(&format!("{host} was not updated."));
                failures += 1;
            }
        }
    }

    if !todo.is_empty() {
        if raw.details {
            ui.details(&details_for(&todo, release, mine));
        }
        ui.plan(&plan_for(&todo, release, mine));
        if !raw.yes && !ui.confirm() {
            ui.declined();
            return Ok(if failures > 0 { 1 } else { 0 });
        }
        let many = todo.len() > 1;
        let total = todo.len() * 2;
        let source = InstallSource {
            release: release.then_some(mine),
            repo: repo.as_deref(),
            binary: binary.as_deref().unwrap_or(Path::new("")),
        };
        for (at, hub) in todo.iter().enumerate() {
            match update_one(hub, &source, (at * 2 + 1, total), many, ui) {
                Ok(now) => runs.push((hub.host.clone(), now)),
                Err(Failure(_, message)) => {
                    eprintln!("aria hub update: {message}");
                    ui.failed(&format!("{} was not updated.", hub.host));
                    failures += 1;
                }
            }
        }
    }
    if !runs.is_empty() {
        ui.done(&Outcome::Updated { runs });
    }
    Ok(if failures > 0 { 1 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|arg| arg.to_string()).collect()
    }

    #[test]
    fn the_arguments_are_a_host_or_all_and_the_flags() {
        assert_eq!(parse_update_args(&args(&[])).unwrap(), RawUpdate::default());
        let raw =
            parse_update_args(&args(&["ana@hub", "--yes", "--plain", "--details", "--binary", "/x/aria"])).unwrap();
        assert_eq!(raw.host.as_deref(), Some("ana@hub"));
        assert!(raw.yes && raw.plain && raw.details && !raw.all);
        assert_eq!(raw.binary, Some(PathBuf::from("/x/aria")));
        assert!(parse_update_args(&args(&["--all"])).unwrap().all);
        assert!(parse_update_args(&args(&["-y"])).unwrap().yes);
    }

    #[test]
    fn what_cannot_be_asked_for_is_refused() {
        for bad in [
            &["--all", "ana@hub"][..],
            &["a@b", "c@d"],
            &["ana@hub", "--frobnicate"],
            &["--binary"],
            &["ana@hub", "--all"],
        ] {
            assert!(parse_update_args(&args(bad)).is_err(), "{bad:?}");
        }
        // A host that looks like an option is taken as the host (the host check refuses it later, as `aria connect` does).
        assert_eq!(parse_update_args(&args(&["-oProxyCommand=x"])).unwrap().host.as_deref(), Some("-oProxyCommand=x"));
    }

    fn hubs(list: &[&str], default: Option<&str>) -> Hubs {
        let mut hubs = Hubs::default();
        for host in list {
            hubs.remember(host, Path::new("/k/aria-hub"));
        }
        hubs.default = default.map(str::to_string);
        hubs
    }

    #[test]
    fn the_hubs_are_the_named_one_all_of_them_or_the_default() {
        let two = hubs(&["a@one", "b@two"], Some("b@two"));
        let want =
            |host: Option<&str>, all: bool| RawUpdate { host: host.map(str::to_string), all, ..RawUpdate::default() };
        assert_eq!(
            hubs_to_update(&want(Some("c@other"), false), &two).unwrap(),
            ["c@other"],
            "a hub not remembered can be named"
        );
        assert_eq!(hubs_to_update(&want(None, true), &two).unwrap(), ["a@one", "b@two"]);
        assert_eq!(hubs_to_update(&want(None, false), &two).unwrap(), ["b@two"], "no host is the default hub");
    }

    #[test]
    fn with_no_hub_linked_it_says_to_run_init() {
        let none = Hubs::default();
        for all in [false, true] {
            let Err(Failure(code, message)) = hubs_to_update(&RawUpdate { all, ..RawUpdate::default() }, &none) else {
                panic!("a hub was found")
            };
            assert_eq!(code, 2);
            assert!(message.contains("aria hub init user@host"), "{message}");
        }
        // A default that names a hub no longer listed is no default.
        let stale = Hubs { default: Some("gone@x".into()), ..Hubs::default() };
        assert!(hubs_to_update(&RawUpdate::default(), &stale).is_err());
    }

    #[test]
    fn a_hub_is_classified_by_its_version() {
        assert_eq!(classify("0.21.0", "0.21.0", true), Probed::Current("0.21.0".into()));
        assert_eq!(classify("0.23.0", "0.22.0", true), Probed::Newer("0.23.0".into()));
        assert_eq!(classify("0.21.0", "0.22.0", true), Probed::Older { from: "0.21.0".into(), active: true });
        assert_eq!(classify("0.21.0", "0.22.0", false), Probed::Older { from: "0.21.0".into(), active: false });
        // A version that cannot be read is updated, and said as the hub said it.
        assert_eq!(classify("unknown", "0.22.0", true), Probed::Older { from: "unknown".into(), active: true });
    }

    #[test]
    fn the_plan_has_two_steps_for_each_hub_and_says_nothing_is_sent() {
        let one = [Todo { host: "ana@hub".into(), from: "0.21.0".into(), active: true, via: Via::Login }];
        let plan = plan_for(&one, true, "0.22.0");
        assert_eq!(
            plan.steps,
            ["Update Aria on ana@hub from 0.21.0 to 0.22.0 (through your ssh login)", "Check the new version"]
        );
        assert!(plan.good_to_know[0].contains("Nothing is sent"));
        assert!(plan.good_to_know.iter().any(|line| line.contains("takes the new version when idle")));
        let build = plan_for(&one, false, "0.22.0");
        assert_eq!(build.steps[0], "Update Aria on ana@hub from 0.21.0 to the build you gave me");

        let two = [
            Todo { host: "a@one".into(), from: "0.21.0".into(), active: true, via: Via::Login },
            Todo { host: "b@two".into(), from: "0.20.1".into(), active: false, via: Via::Login },
        ];
        let plan = plan_for(&two, true, "0.22.0");
        assert_eq!(plan.host, "your servers");
        assert_eq!(plan.steps.len(), 4);
        assert_eq!(plan.steps[1], "Check the new version on a@one");
        assert_eq!(plan.steps[2], "Update Aria on b@two from 0.20.1 to 0.22.0 (through your ssh login)");
    }

    #[test]
    fn a_hub_that_updates_itself_says_so_and_asks_for_no_ssh_login() {
        let itself = [Todo { host: "ana@hub".into(), from: "0.21.0".into(), active: true, via: Via::Itself }];
        let plan = plan_for(&itself, true, "0.22.0");
        assert_eq!(
            plan.steps,
            ["Update Aria on ana@hub from 0.21.0 to 0.22.0 (the server updates itself)", "Check the new version"]
        );
        assert!(plan.good_to_know.iter().any(|line| line.contains("no ssh login of yours is used")), "{plan:?}");
        assert!(
            !plan.good_to_know.iter().any(|line| line.contains("its sudo is used")),
            "only a login install runs sudo"
        );
        // A build uploaded by hand never updates itself: its plan has no "how".
        let build = plan_for(&itself, false, "0.22.0");
        assert_eq!(build.steps[0], "Update Aria on ana@hub from 0.21.0 to the build you gave me");

        let mixed = [
            Todo { host: "a@one".into(), from: "0.21.0".into(), active: true, via: Via::Itself },
            Todo { host: "b@two".into(), from: "0.20.1".into(), active: true, via: Via::Login },
        ];
        let plan = plan_for(&mixed, true, "0.22.0");
        assert!(
            plan.steps[0].ends_with("(the server updates itself)")
                && plan.steps[2].ends_with("(through your ssh login)")
        );
        let details = details_for(&itself, true, "0.22.0");
        assert!(details[0].contains("hubUpdate") && details[0].contains("restricted key"), "{details:?}");
    }

    #[test]
    fn the_waits_are_those_of_a_real_server_unless_the_tests_shorten_them() {
        let standard = Waits::standard();
        assert_eq!(standard.silence, Duration::from_secs(20));
        assert!(standard.come_back > standard.poll && standard.confirm >= standard.poll);
    }

    #[test]
    fn a_line_of_the_installer_is_cut_to_fit_under_a_step() {
        assert_eq!(shorten("  - Checking this server  "), "- Checking this server");
        let long = shorten(&"x".repeat(200));
        assert_eq!(long.chars().count(), 70);
        assert!(long.ends_with('…'));
    }

    #[test]
    fn the_technical_plan_names_the_commands() {
        let one = [Todo { host: "ana@hub".into(), from: "0.21.0".into(), active: true, via: Via::Login }];
        let release = details_for(&one, true, "0.22.0");
        assert!(release[0].contains(&release_install_command("0.22.0")), "{release:?}");
        assert!(release[0].contains("no settings or credentials"));
        let build = details_for(&one, false, "0.22.0");
        assert!(build[0].contains(INSTALL_COMMAND) && build[0].contains(HUB_SRC), "{build:?}");
    }
}
