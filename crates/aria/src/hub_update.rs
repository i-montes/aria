//! A hub updating itself: the service's side of `hubUpdate` (`aria hub update` over the restricted key, which cannot run an
//! installer). A client asks for release `X.Y.Z`; the service installs that release as its own user and says, line by line,
//! what the installer printed, then whether it worked.
//!
//! What a client can and cannot make this do, since a stolen client key can send the message:
//! - The version is three numbers and nothing else (`^[0-9]+\.[0-9]+\.[0-9]+$`, at most nine digits each), and it is written
//!   out again from the numbers it parsed to: no text of the client's is ever interpolated into a shell. It reaches the
//!   child only as an argument of its own (`$1` and `$2` of the script below), never inside the script's text.
//! - It must be newer than what runs: no downgrade, and no "update" to the same version.
//! - The repository is fixed (`i-montes/aria`), or the one the hub's own owner put in the service's environment
//!   (`ARIA_REPO`); a client never names one.
//! - One update at a time; the child runs in its own process group with a total time limit and is killed with it.
//! - Only a hub (the build without a window) does it: a dock refuses, since the installer here is the server's (`--headless`).
//!
//! What is left is what the owner's own release does when installed on their own hub: nothing a client chose.

use aria_core::bridge::DockMessage;
use std::io::{BufRead, Read};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

/// The repository releases are installed from, unless the hub's owner names another in the service's environment.
pub const REPO: &str = "i-montes/aria";

/// How long the installer may take in all, downloads included.
pub const TOTAL_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// The longest line of the installer's output that is passed on (a client's screen is not for more).
const MAX_LINE: usize = 300;

/// What runs on the hub: the release's `install.sh`, fetched to a file first (`curl | sh` would say "worked" when curl
/// failed and the shell got nothing), then run with its text on stdin as `curl | sh` does. `$1` is the address, `$2` the
/// release tag: arguments of their own, so the script's text never holds either.
const SCRIPT: &str = r#"set -e
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
curl -fsSL "$1" -o "$tmp"
ARIA_VERSION="$2" ARIA_NESTED=1 sh -s -- --headless --yes < "$tmp""#;

/// `X.Y.Z` as numbers when `text` is exactly that: ASCII digits, three groups, at most nine digits each.
pub fn parse_requested(text: &str) -> Option<(u64, u64, u64)> {
    let mut numbers = [0u64; 3];
    let mut groups = text.split('.');
    for number in &mut numbers {
        let group = groups.next()?;
        if group.is_empty() || group.len() > 9 || !group.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        *number = group.parse().ok()?;
    }
    groups.next().is_none().then_some((numbers[0], numbers[1], numbers[2]))
}

/// The version to install, written from its numbers (`007.1.0` is `7.1.0`), or why it is refused.
pub fn check(requested: &str, running: &str) -> Result<String, String> {
    let Some(wanted) = parse_requested(requested) else {
        return Err("the version must be three numbers like 0.22.0".to_string());
    };
    let shown = crate::version::show(wanted);
    match crate::version::parse(running).map(|running| wanted.cmp(&running)) {
        Some(std::cmp::Ordering::Equal) => Err(format!("already runs {shown}")),
        Some(std::cmp::Ordering::Less) => Err("a downgrade is refused".to_string()),
        // A running version that cannot be read (it always can: it is the crate's) leaves nothing to compare with.
        Some(std::cmp::Ordering::Greater) | None => Ok(shown),
    }
}

/// A repository name as GitHub writes one (`owner/name`), and nothing that could be more than a path in an address.
fn valid_repo(text: &str) -> bool {
    let mut parts = text.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else { return false };
    let plain = |part: &str| {
        !part.is_empty()
            && part.len() <= 100
            && !part.starts_with('.')
            && part.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    };
    plain(owner) && plain(name)
}

/// The address of the release's installer: the service's own `ARIA_REPO` (its owner's), else the fixed repository.
pub fn release_url(version: &str, repo: Option<&str>) -> String {
    let repo = repo.filter(|repo| valid_repo(repo)).unwrap_or(REPO);
    format!("https://github.com/{repo}/releases/download/v{version}/install.sh")
}

/// How the installer ended.
#[derive(Debug, Clone, PartialEq)]
pub enum RunEnd {
    /// It exited with this status.
    Exited(i32),
    /// It ran out of time and was killed.
    TimedOut,
    /// It could not be started.
    Failed(String),
}

/// What runs the installer and reads what it installed; a trait so the tests can stand in for the real thing.
pub trait Runner: Send + Sync {
    /// Runs the installer fetched from `url` for release `tag` (`v0.22.0`), handing every line it prints to `line`, and
    /// gives up after `timeout`.
    fn install(&self, url: &str, tag: &str, timeout: Duration, line: &mut dyn FnMut(&str)) -> RunEnd;
    /// What the installed `aria` says (`aria --version`), when it can be run.
    fn installed_version(&self) -> Option<String>;
}

/// The real one: `sh -c <script> <url> <tag>`, in its own process group.
pub struct ShellRunner {
    script: String,
}

impl ShellRunner {
    pub fn real() -> ShellRunner {
        ShellRunner { script: SCRIPT.to_string() }
    }

    #[cfg(test)]
    fn with_script(script: &str) -> ShellRunner {
        ShellRunner { script: script.to_string() }
    }
}

/// Lines from a pipe, each cut at `MAX_LINE` characters (the rest of a longer one is dropped, never buffered whole).
fn read_lines(reader: impl Read, sender: mpsc::Sender<String>) {
    let mut reader = std::io::BufReader::new(reader);
    loop {
        let mut bytes = Vec::new();
        match reader.by_ref().take(64 * 1024).read_until(b'\n', &mut bytes) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let text = String::from_utf8_lossy(&bytes);
        let text: String = text.trim_end_matches(['\n', '\r']).chars().take(MAX_LINE).collect();
        if sender.send(text).is_err() {
            return;
        }
    }
}

impl Runner for ShellRunner {
    fn install(&self, url: &str, tag: &str, timeout: Duration, line: &mut dyn FnMut(&str)) -> RunEnd {
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(&self.script)
            .arg("aria-hub-update")
            .arg(url)
            .arg(tag)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => return RunEnd::Failed(format!("could not start the installer: {error}")),
        };
        let group = child.id() as i32;
        let (sender, lines) = mpsc::channel::<String>();
        let mut pipes: Vec<Box<dyn Read + Send>> = Vec::new();
        if let Some(pipe) = child.stdout.take() {
            pipes.push(Box::new(pipe));
        }
        if let Some(pipe) = child.stderr.take() {
            pipes.push(Box::new(pipe));
        }
        for pipe in pipes {
            let sender = sender.clone();
            std::thread::spawn(move || read_lines(pipe, sender));
        }
        drop(sender);
        let started = Instant::now();
        loop {
            match lines.recv_timeout(Duration::from_millis(100)) {
                Ok(text) => line(&text),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                // Both pipes are closed: the installer (and anything it started) is done writing.
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if started.elapsed() >= timeout {
                // The whole group: the installer started curl, npm and others, and killing only the shell would leave them.
                unsafe { libc::killpg(group, libc::SIGKILL) };
                let _ = child.wait();
                return RunEnd::TimedOut;
            }
        }
        match child.wait() {
            Ok(status) => RunEnd::Exited(status.code().unwrap_or(-1)),
            Err(error) => RunEnd::Failed(format!("could not wait for the installer: {error}")),
        }
    }

    fn installed_version(&self) -> Option<String> {
        let dir = std::env::var_os("ARIA_BIN_DIR")
            .filter(|dir| !dir.is_empty())
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".local/bin")))?;
        let output = Command::new(dir.join("aria")).arg("--version").stdin(Stdio::null()).output().ok()?;
        output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

/// What a line of the installer's output becomes on its way to a client: no control characters (a colour code, a cursor
/// move) and no spaces at the ends.
fn tidy(line: &str) -> String {
    let plain: String = line.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    plain.trim().to_string()
}

/// The service's updater: whether it may update (a hub, not a dock), one at a time, and what runs the installer.
pub struct Updater {
    enabled: bool,
    busy: AtomicBool,
    runner: Arc<dyn Runner>,
    timeout: Duration,
    /// The version this service runs (the crate's), what a request is compared with.
    running: String,
    /// The repository from the service's own environment, if its owner set one.
    repo: Option<String>,
}

impl Updater {
    /// The service's own: only a build with no window is a hub. (A dock refuses, with its reason.)
    pub fn real() -> Updater {
        Updater {
            enabled: !cfg!(feature = "gui"),
            busy: AtomicBool::new(false),
            runner: Arc::new(ShellRunner::real()),
            timeout: TOTAL_TIMEOUT,
            running: crate::version::mine().to_string(),
            repo: std::env::var("ARIA_REPO").ok().filter(|repo| !repo.is_empty()),
        }
    }

    #[cfg(test)]
    fn with(runner: Arc<dyn Runner>, running: &str, timeout: Duration) -> Updater {
        Updater {
            enabled: true,
            busy: AtomicBool::new(false),
            runner,
            timeout,
            running: running.to_string(),
            repo: None,
        }
    }

    /// Answers `hubUpdate { id, version }`: progress lines while it runs, and one result. Blocks until it ends (the caller
    /// runs it on a thread of its own).
    pub fn handle(&self, id: u64, requested: &str, emit: &dyn Fn(DockMessage)) {
        let result = |ok: bool, version: Option<String>, reason: Option<String>| {
            emit(DockMessage::HubUpdateResult { id, ok, version, reason });
        };
        if !self.enabled {
            return result(false, None, Some("this Aria is a dock, not a hub: it does not update itself".to_string()));
        }
        let version = match check(requested, &self.running) {
            Ok(version) => version,
            Err(reason) => return result(false, None, Some(reason)),
        };
        if self.busy.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            return result(false, None, Some("an update is already running".to_string()));
        }
        // Free again however this ends.
        struct Free<'a>(&'a AtomicBool);
        impl Drop for Free<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _free = Free(&self.busy);

        let url = release_url(&version, self.repo.as_deref());
        let mut last = String::new();
        let end = self.runner.install(&url, &format!("v{version}"), self.timeout, &mut |line| {
            let line = tidy(line);
            if line.is_empty() {
                return;
            }
            last.clone_from(&line);
            emit(DockMessage::HubUpdateProgress { id, line });
        });
        match end {
            RunEnd::Exited(0) => {
                let now = self.runner.installed_version();
                match now.as_deref().and_then(crate::version::parse) {
                    Some(found) if crate::version::show(found) == version => result(true, Some(version), None),
                    Some(found) => result(
                        false,
                        None,
                        Some(format!(
                            "the installer finished but the hub has Aria {} installed, not {version}",
                            crate::version::show(found)
                        )),
                    ),
                    None => result(
                        false,
                        None,
                        Some(format!(
                            "the installer finished but the installed Aria could not be read (wanted {version})"
                        )),
                    ),
                }
            }
            RunEnd::Exited(code) => {
                let reason = if last.is_empty() { format!("the installer failed (exit {code})") } else { last };
                result(false, None, Some(reason))
            }
            RunEnd::TimedOut => result(
                false,
                None,
                Some(format!(
                    "the update took more than {} minutes and was stopped",
                    self.timeout.as_secs().div_ceil(60)
                )),
            ),
            RunEnd::Failed(reason) => result(false, None, Some(reason)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A runner that says what it was asked and what to answer; `gate` makes it wait until the test lets it go.
    struct Fake {
        calls: Mutex<Vec<(String, String)>>,
        lines: Vec<&'static str>,
        end: RunEnd,
        installed: Option<&'static str>,
        gate: Option<Mutex<mpsc::Receiver<()>>>,
    }

    impl Fake {
        fn new(end: RunEnd, installed: Option<&'static str>) -> Fake {
            Fake { calls: Mutex::new(Vec::new()), lines: vec!["- Checking this server"], end, installed, gate: None }
        }
        fn calls(&self) -> Vec<(String, String)> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl Runner for Arc<Fake> {
        fn install(&self, url: &str, tag: &str, _timeout: Duration, line: &mut dyn FnMut(&str)) -> RunEnd {
            self.calls.lock().unwrap().push((url.to_string(), tag.to_string()));
            for text in &self.lines {
                line(text);
            }
            if let Some(gate) = &self.gate {
                let _ = gate.lock().unwrap().recv();
            }
            self.end.clone()
        }
        fn installed_version(&self) -> Option<String> {
            self.installed.map(str::to_string)
        }
    }

    fn collect(updater: &Updater, version: &str) -> Vec<DockMessage> {
        let out = Mutex::new(Vec::new());
        updater.handle(7, version, &|message| out.lock().unwrap().push(message));
        out.into_inner().unwrap()
    }

    fn only_result(messages: &[DockMessage]) -> (bool, Option<String>, Option<String>) {
        match messages.last() {
            Some(DockMessage::HubUpdateResult { id: 7, ok, version, reason }) => (*ok, version.clone(), reason.clone()),
            other => panic!("no result at the end: {other:?}"),
        }
    }

    #[test]
    fn only_three_numbers_are_a_version() {
        assert_eq!(parse_requested("0.22.0"), Some((0, 22, 0)));
        assert_eq!(parse_requested("12.345.6789"), Some((12, 345, 6789)));
        for bad in [
            "",
            "0.22",
            "0.22.0.1",
            "v0.22.0",
            "1.0.0-rc1",
            "0.22.0 ",
            " 0.22.0",
            "0.22.0\n",
            "0.22.0\n1.0.0",
            "0.22.0; rm -rf ~",
            "0.22.0 && x",
            "$(x).1.2",
            "0.22.0`x`",
            "0..1",
            ".1.2",
            "1.2.",
            "1.2.-3",
            "+1.2.3",
            "1.2.3e1",
            "１.２.３",
            "9999999999.0.0",
        ] {
            assert_eq!(parse_requested(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_version_is_written_from_its_numbers() {
        assert_eq!(check("007.01.0", "0.5.0"), Ok("7.1.0".to_string()));
        assert_eq!(check("0.22.0", "0.21.0"), Ok("0.22.0".to_string()));
        assert_eq!(check("0.9.0", "0.10.0"), Err("a downgrade is refused".to_string()), "numbers, not text");
        assert_eq!(check("0.10.0", "0.9.0"), Ok("0.10.0".to_string()));
    }

    #[test]
    fn the_address_is_the_fixed_repository_unless_the_hubs_owner_set_one() {
        assert_eq!(
            release_url("0.22.0", None),
            "https://github.com/i-montes/aria/releases/download/v0.22.0/install.sh"
        );
        assert_eq!(
            release_url("0.22.0", Some("acme/aria-fork")),
            "https://github.com/acme/aria-fork/releases/download/v0.22.0/install.sh"
        );
        for bad in ["", "acme", "a/b/c", "../x", "acme/.hidden", "a b/c", "acme/aria?x=1", "evil.example/x@y/z", "/a"] {
            assert_eq!(release_url("0.22.0", Some(bad)), release_url("0.22.0", None), "{bad:?} falls back");
        }
    }

    #[test]
    fn what_could_inject_a_command_is_refused_before_any_process_starts() {
        let fake = Arc::new(Fake::new(RunEnd::Exited(0), Some("aria 0.99.0")));
        let updater = Updater::with(Arc::new(fake.clone()), "0.21.0", Duration::from_secs(5));
        for bad in [
            "0.22.0; rm -rf ~",
            "0.22.0 && x",
            "$(x)",
            "0.22.0\nx",
            "v0.22.0",
            "0.22",
            "1.0.0-rc1",
            "",
            "0.22.0`id`",
            "0.22.0|sh",
        ] {
            let (ok, version, reason) = only_result(&collect(&updater, bad));
            assert!(!ok && version.is_none(), "{bad:?}");
            assert_eq!(reason.as_deref(), Some("the version must be three numbers like 0.22.0"), "{bad:?}");
        }
        assert!(fake.calls().is_empty(), "no process was started: {:?}", fake.calls());
    }

    #[test]
    fn a_downgrade_and_the_same_version_are_refused_too() {
        let fake = Arc::new(Fake::new(RunEnd::Exited(0), Some("aria 0.20.0")));
        let updater = Updater::with(Arc::new(fake.clone()), "0.21.0", Duration::from_secs(5));
        assert_eq!(only_result(&collect(&updater, "0.20.0")).2.as_deref(), Some("a downgrade is refused"));
        assert_eq!(only_result(&collect(&updater, "0.21.0")).2.as_deref(), Some("already runs 0.21.0"));
        assert!(fake.calls().is_empty());
    }

    #[test]
    fn the_child_gets_only_the_validated_version_and_the_fixed_address() {
        let fake = Arc::new(Fake::new(RunEnd::Exited(0), Some("aria 7.1.0")));
        let updater = Updater::with(Arc::new(fake.clone()), "0.21.0", Duration::from_secs(5));
        let messages = collect(&updater, "007.01.0");
        assert_eq!(
            fake.calls(),
            [(
                "https://github.com/i-montes/aria/releases/download/v7.1.0/install.sh".to_string(),
                "v7.1.0".to_string()
            )]
        );
        assert_eq!(only_result(&messages), (true, Some("7.1.0".to_string()), None));
        assert_eq!(
            messages[0],
            DockMessage::HubUpdateProgress { id: 7, line: "- Checking this server".to_string() },
            "the installer's lines come first, with the request's id"
        );
    }

    #[test]
    fn it_is_ok_only_when_the_installed_version_is_the_requested_one() {
        let updater = |installed| {
            let fake = Arc::new(Fake::new(RunEnd::Exited(0), installed));
            Updater::with(Arc::new(fake), "0.21.0", Duration::from_secs(5))
        };
        assert!(only_result(&collect(&updater(Some("aria 0.22.0")), "0.22.0")).0);
        let (ok, _, reason) = only_result(&collect(&updater(Some("aria 0.21.0")), "0.22.0"));
        assert!(!ok && reason.unwrap().contains("not 0.22.0"));
        let (ok, _, reason) = only_result(&collect(&updater(None), "0.22.0"));
        assert!(!ok && reason.unwrap().contains("could not be read"));
    }

    #[test]
    fn a_failing_installer_says_its_last_line() {
        let mut fake = Fake::new(RunEnd::Exited(22), None);
        fake.lines = vec!["curl: (22) The requested URL returned error: 404", "", "  "];
        let updater = Updater::with(Arc::new(Arc::new(fake)), "0.21.0", Duration::from_secs(5));
        let messages = collect(&updater, "0.22.0");
        assert_eq!(only_result(&messages).2.as_deref(), Some("curl: (22) The requested URL returned error: 404"));
        assert_eq!(messages.len(), 2, "the empty lines are not passed on");
        let silent =
            Updater::with(Arc::new(Arc::new(Fake::new(RunEnd::Exited(3), None))), "0.21.0", Duration::from_secs(5));
        let only = |messages: &[DockMessage]| only_result(messages).2;
        assert_eq!(only(&collect(&silent, "0.22.0")).as_deref(), Some("- Checking this server"));
        let mut mute = Fake::new(RunEnd::Exited(3), None);
        mute.lines = vec![];
        let mute = Updater::with(Arc::new(Arc::new(mute)), "0.21.0", Duration::from_secs(5));
        assert_eq!(only(&collect(&mute, "0.22.0")).as_deref(), Some("the installer failed (exit 3)"));
    }

    #[test]
    fn a_timeout_and_a_failed_start_are_results() {
        let updater = |end| Updater::with(Arc::new(Arc::new(Fake::new(end, None))), "0.21.0", Duration::from_secs(120));
        let (ok, _, reason) = only_result(&collect(&updater(RunEnd::TimedOut), "0.22.0"));
        assert!(!ok && reason.unwrap().contains("took more than 2 minutes"));
        let (ok, _, reason) = only_result(&collect(&updater(RunEnd::Failed("no sh".into())), "0.22.0"));
        assert!(!ok && reason.as_deref() == Some("no sh"));
    }

    #[test]
    fn control_characters_of_the_installer_are_not_passed_on() {
        assert_eq!(tidy("\u{1b}[32m  ok \u{1b}[0m done  "), "[32m  ok  [0m done");
        assert_eq!(tidy("   "), "");
    }

    #[test]
    fn a_second_request_while_one_runs_is_refused() {
        let (release, gate) = mpsc::channel();
        let mut fake = Fake::new(RunEnd::Exited(0), Some("aria 0.22.0"));
        fake.gate = Some(Mutex::new(gate));
        let fake = Arc::new(fake);
        let updater = Arc::new(Updater::with(Arc::new(fake.clone()), "0.21.0", Duration::from_secs(5)));
        let first = {
            let updater = updater.clone();
            std::thread::spawn(move || collect(&updater, "0.22.0"))
        };
        // The first is running once the runner was called.
        let started = Instant::now();
        while fake.calls().is_empty() {
            assert!(started.elapsed() < Duration::from_secs(5), "the first never started");
            std::thread::sleep(Duration::from_millis(5));
        }
        let (ok, _, reason) = only_result(&collect(&updater, "0.23.0"));
        assert!(!ok);
        assert_eq!(reason.as_deref(), Some("an update is already running"));
        assert_eq!(fake.calls().len(), 1, "the second started nothing");
        release.send(()).unwrap();
        assert!(only_result(&first.join().unwrap()).0);
        // The gate stays open from here on (a closed sender makes every later wait return at once).
        drop(release);
        // And free again afterwards.
        assert!(
            !only_result(&collect(&updater, "0.23.0")).0,
            "the second now runs (and the fake says 0.22.0 is installed)"
        );
        assert_eq!(fake.calls().len(), 2);
    }

    #[test]
    fn a_dock_does_not_update_itself() {
        let fake = Arc::new(Fake::new(RunEnd::Exited(0), Some("aria 0.22.0")));
        let mut updater = Updater::with(Arc::new(fake.clone()), "0.21.0", Duration::from_secs(5));
        updater.enabled = false;
        let (ok, _, reason) = only_result(&collect(&updater, "0.22.0"));
        assert!(!ok && reason.unwrap().contains("not a hub"));
        assert!(fake.calls().is_empty());
    }

    // ---- the real runner, with scripts of the tests' own ---------------------------------------------------------------

    #[test]
    fn the_real_runner_passes_the_address_and_the_tag_as_arguments_of_their_own() {
        let runner = ShellRunner::with_script(r#"echo "$1|$2"; echo to-stderr >&2; exit 5"#);
        let mut lines = Vec::new();
        let end = runner.install("https://x.example/a b;c", "v1.2.3", Duration::from_secs(10), &mut |line| {
            lines.push(line.to_string())
        });
        assert_eq!(end, RunEnd::Exited(5));
        lines.sort();
        assert_eq!(
            lines,
            ["https://x.example/a b;c|v1.2.3", "to-stderr"],
            "a value with spaces and a semicolon stays one argument"
        );
    }

    #[test]
    fn the_script_holds_no_place_for_the_client_s_text() {
        // The version and the address come in as $1 and $2 and nothing is formatted into the script.
        assert!(SCRIPT.contains(r#"curl -fsSL "$1""#) && SCRIPT.contains(r#"ARIA_VERSION="$2""#));
        assert!(!SCRIPT.contains('{') && !SCRIPT.contains("github.com"), "{SCRIPT}");
        assert!(SCRIPT.starts_with("set -e"), "a failing download stops it");
    }

    #[test]
    fn the_real_runner_kills_the_whole_group_when_the_time_is_up() {
        // The marker is written by a grandchild that would outlive its shell if only the shell were killed.
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("grandchild-ran-on");
        let script = format!("(sleep 3; touch {}) & sleep 30", marker.display());
        let runner = ShellRunner::with_script(&script);
        let started = Instant::now();
        let end = runner.install("u", "t", Duration::from_millis(300), &mut |_| {});
        assert_eq!(end, RunEnd::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(5), "it came back at the limit, not after the 30 s");
        std::thread::sleep(Duration::from_millis(3500));
        assert!(!marker.exists(), "the grandchild was killed with the group");
    }

    #[test]
    fn the_real_runner_reads_a_long_line_cut_and_the_rest_after_it() {
        let runner = ShellRunner::with_script(r#"head -c 20000 /dev/zero | tr '\0' x; echo; echo after"#);
        let mut lines = Vec::new();
        let end = runner.install("u", "t", Duration::from_secs(10), &mut |line| lines.push(line.to_string()));
        assert_eq!(end, RunEnd::Exited(0));
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].chars().count(), MAX_LINE);
        assert_eq!(lines[1], "after");
    }
}
