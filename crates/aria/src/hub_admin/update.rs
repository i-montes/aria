//! `aria hub update`: brings the hubs this computer is linked to up to this Aria's version. It is the install step of
//! `hub init` and the version check, and nothing else: no sign-ins or settings are sent, no key is made, `authorized_keys`
//! and `hubs.json` are not touched. A child of `hub_admin` so it uses that module's helpers instead of copying them.
//!
//! The commands run on the hub with the owner's own ssh login, as `hub init` does: the key `hub init` made for `aria connect`
//! is restricted to Aria's terminal bridge, so it cannot run an installer.

use super::*;
use crate::connect::Hubs;
use crate::version;
use std::cmp::Ordering;

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

/// One hub that gets updated: its name, the version it runs and whether its service is up.
struct Todo {
    host: String,
    from: String,
    active: bool,
}

/// The plan in plain words: two steps for each hub, and what to know.
fn plan_for(todo: &[Todo], release: bool, mine: &str) -> Plan {
    let many = todo.len() > 1;
    let mut steps = Vec::new();
    for hub in todo {
        steps.push(if release {
            format!("Update Aria on {} from {} to {mine}", hub.host, hub.from)
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
    good_to_know.push("The server's installer runs again; its sudo is used only if a package is missing.".to_string());
    let host = if many { "your servers".to_string() } else { todo[0].host.clone() };
    Plan { host, steps, good_to_know, blocked: Vec::new() }
}

/// The technical plan (`--details`): the commands that run, one paragraph for each hub.
fn details_for(todo: &[Todo], release: bool, mine: &str) -> Vec<String> {
    todo.iter()
        .map(|hub| {
            let install = if release {
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
    let stopped = |step: usize, what: &str, detail: String| {
        failed(format!(
            "{host}: step {} ({what}) failed: {detail}. The installer may have partly run on the hub; running \
             `aria hub update` again is safe",
            if step == 1 { "upload" } else { "installer" }
        ))
    };
    install_on_hub(host, source, &format!("Updating Aria on {host}"), (number, total), ui, &stopped)?;
    ui.step_ok("Aria is updated on the server");
    if hub.active {
        ui.step_ok("The service keeps running (it takes the new version when idle)");
    }
    ui.step_start(
        number + 1,
        total,
        &if many { format!("Checking the new version on {host}") } else { "Checking the new version".to_string() },
        "Reads the version the server runs now.",
    );
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
        match probe_hub(host, binary_arch) {
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
                todo.push(Todo { host: host.clone(), from, active });
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
        let one = [Todo { host: "ana@hub".into(), from: "0.21.0".into(), active: true }];
        let plan = plan_for(&one, true, "0.22.0");
        assert_eq!(plan.steps, ["Update Aria on ana@hub from 0.21.0 to 0.22.0", "Check the new version"]);
        assert!(plan.good_to_know[0].contains("Nothing is sent"));
        assert!(plan.good_to_know.iter().any(|line| line.contains("takes the new version when idle")));
        let build = plan_for(&one, false, "0.22.0");
        assert_eq!(build.steps[0], "Update Aria on ana@hub from 0.21.0 to the build you gave me");

        let two = [
            Todo { host: "a@one".into(), from: "0.21.0".into(), active: true },
            Todo { host: "b@two".into(), from: "0.20.1".into(), active: false },
        ];
        let plan = plan_for(&two, true, "0.22.0");
        assert_eq!(plan.host, "your servers");
        assert_eq!(plan.steps.len(), 4);
        assert_eq!(plan.steps[1], "Check the new version on a@one");
        assert_eq!(plan.steps[2], "Update Aria on b@two from 0.20.1 to 0.22.0");
    }

    #[test]
    fn the_technical_plan_names_the_commands() {
        let one = [Todo { host: "ana@hub".into(), from: "0.21.0".into(), active: true }];
        let release = details_for(&one, true, "0.22.0");
        assert!(release[0].contains(&release_install_command("0.22.0")), "{release:?}");
        assert!(release[0].contains("no settings or credentials"));
        let build = details_for(&one, false, "0.22.0");
        assert!(build[0].contains(INSTALL_COMMAND) && build[0].contains(HUB_SRC), "{build:?}");
    }
}
