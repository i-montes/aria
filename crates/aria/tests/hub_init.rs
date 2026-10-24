//! `aria hub init` as a process, against a fake ssh (tests/fake_ssh.sh) and a scratch repository and home: nothing here
//! touches the owner's `~/.aria`, keyring, ssh configuration or network.
#![cfg(unix)]

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

const PINS: [&str; 5] = ["pi-version", "pi-packages", "engram-version", "agent-browser-version", "node-version"];
const KEY: &str = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl ana@laptop";
const TOOLS: [&str; 14] =
    ["sh", "head", "tail", "cat", "wc", "tr", "mkdir", "tar", "cut", "grep", "touch", "chmod", "rm", "cp"];

/// An owner's computer (a home with settings) and a checkout of the repository, next to a hub that is only the fake ssh.
struct World {
    dir: tempfile::TempDir,
}

fn elf(machine: u8) -> Vec<u8> {
    let mut bytes = vec![0u8; 64];
    bytes[..4].copy_from_slice(b"\x7fELF");
    bytes[4] = 2;
    bytes[5] = 1;
    bytes[18] = machine;
    bytes
}

fn which(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| panic!("{name} is not installed"))
}

impl World {
    fn new() -> World {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        for sub in ["repo/install", "repo/bin", "home/.aria", "hub", "bin"] {
            std::fs::create_dir_all(path.join(sub)).unwrap();
        }
        std::fs::write(path.join("repo/install/install.sh"), "#!/bin/sh\necho fake installer\n").unwrap();
        std::fs::write(path.join("repo/install/aria-serve.service"), "[Unit]\n").unwrap();
        for pin in PINS {
            std::fs::write(path.join("repo").join(pin), format!("{pin} contents\n")).unwrap();
        }
        std::fs::write(path.join("repo/bin/aria-aarch64"), elf(0xB7)).unwrap();
        std::fs::write(path.join("repo/bin/aria-x86_64"), elf(0x3E)).unwrap();
        std::fs::write(path.join("home/.aria/settings.json"), r#"{"language":"en","credentials":{"keyring":false}}"#)
            .unwrap();
        std::fs::write(path.join("home/.aria/auth.json"), r#"{"opencode-go":{"key":"sk-never-in-a-log"}}"#).unwrap();
        for tool in TOOLS {
            std::os::unix::fs::symlink(which(tool), path.join("bin").join(tool)).unwrap();
        }
        World { dir }
    }

    fn path(&self, sub: &str) -> PathBuf {
        self.dir.path().join(sub)
    }

    fn log(&self) -> Vec<String> {
        std::fs::read_to_string(self.path("hub/log")).unwrap_or_default().lines().map(str::to_string).collect()
    }

    /// `aria hub init ana@hub.example <args>` with the fake ssh and no terminal; `envs` pick the fake's variants.
    fn init(&self, args: &[&str], envs: &[(&str, &str)]) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_aria"));
        command
            .args(["hub", "init"])
            .args(args)
            .current_dir(self.path("repo"))
            .env_clear()
            .env("HOME", self.path("home"))
            .env("ARIA_HOME", self.path("home/.aria"))
            .env("PATH", self.path("bin"))
            .env("ARIA_SSH", concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fake_ssh.sh"))
            .env("ARIA_FAKE_SSH_LOG", self.path("hub/log"))
            .env("ARIA_FAKE_SSH_ROOT", self.path("hub"))
            .stdin(Stdio::null());
        for (name, value) in envs {
            command.env(name, value);
        }
        command.output().unwrap()
    }

    fn init_yes(&self, extra: &[&str], envs: &[(&str, &str)]) -> Output {
        let mut args = vec!["ana@hub.example", "--binary", "bin/aria-aarch64", "--yes"];
        args.extend_from_slice(extra);
        self.init(&args, envs)
    }

    fn authorized_keys(&self) -> String {
        std::fs::read_to_string(self.path("hub/home/.ssh/authorized_keys")).unwrap_or_default()
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn commands(log: &[String]) -> Vec<String> {
    log.iter().map(|line| line.split('\t').nth(1).unwrap_or(line).to_string()).collect()
}

/// The remote commands that change something on the hub (every probe reads only).
fn writes(log: &[String]) -> Vec<String> {
    const PROBES: [&str; 9] = [
        "uname -m",
        "cat /etc/os-release",
        "echo $HOME",
        "ss -ltnH -n",
        "command -v",
        "node --version",
        "is-active",
        "test -f",
        "loginctl show-user",
    ];
    commands(log)
        .into_iter()
        .filter(|command| !command.starts_with("-G ") && !PROBES.iter().any(|probe| command.contains(probe)))
        .filter(|command| !command.contains("os-release") && !command.starts_with("cat /etc/ssh"))
        .filter(|command| !command.starts_with("bwrap "))
        .collect()
}

fn position(commands: &[String], needle: &str) -> usize {
    commands
        .iter()
        .position(|command| command.contains(needle))
        .unwrap_or_else(|| panic!("{needle} not in {commands:?}"))
}

#[test]
fn the_plan_is_printed_before_anything_runs() {
    let world = World::new();
    let output = world.init(&["ana@hub.example", "--binary", "bin/aria-aarch64"], &[]);
    let out = text(&output.stdout);
    assert!(output.status.success(), "{out}{}", text(&output.stderr));
    assert!(out.contains("1. "), "{out}");
    assert!(out.contains("install.sh --headless --local"), "{out}");
    assert!(out.contains("aria hub-import"), "{out}");
    assert!(out.contains("aria-serve.service"), "{out}");
    assert!(out.contains("nothing was changed"), "{out}");
    let log = world.log();
    let probes = commands(&log);
    assert!(probes.contains(&"-G ana@hub.example".to_string()), "{probes:?}");
    assert!(probes.contains(&"uname -m".to_string()), "{probes:?}");
    assert!(probes.iter().any(|c| c.contains("/etc/os-release")), "{probes:?}");
    assert!(probes.contains(&"echo $HOME".to_string()), "{probes:?}");
    assert!(probes.contains(&"ss -ltnH -n".to_string()), "{probes:?}");
    assert!(
        probes.iter().any(|c| c.starts_with("command -v ") && c.contains("xz") && c.contains("bwrap")),
        "{probes:?}"
    );
    assert!(writes(&log).is_empty(), "only probes ran: {:?}", writes(&log));
}

#[test]
fn the_plan_comes_before_the_first_write_in_the_output() {
    let world = World::new();
    let output = world.init_yes(&[], &[]);
    let out = text(&output.stdout);
    assert!(output.status.success(), "{out}{}", text(&output.stderr));
    let plan = out.find("1. ").expect("a plan");
    let first_step = out.find("step 1").expect("a first step");
    assert!(plan < first_step, "{out}");
}

#[test]
fn a_no_answer_changes_nothing() {
    let world = World::new();
    let output = world.init(&["ana@hub.example", "--binary", "bin/aria-aarch64"], &[]);
    assert_eq!(output.status.code(), Some(0));
    assert!(text(&output.stdout).contains("nothing was changed"));
    assert!(writes(&world.log()).is_empty());
    assert!(!world.path("hub/home/.aria/hub-src").exists());
}

#[test]
fn no_terminal_and_no_yes_means_no() {
    let world = World::new();
    // Even an answer on a stdin that is not a terminal is no: only `--yes` or a person at a terminal says yes.
    let mut child = Command::new(env!("CARGO_BIN_EXE_aria"))
        .args(["hub", "init", "ana@hub.example", "--binary", "bin/aria-aarch64"])
        .current_dir(world.path("repo"))
        .env_clear()
        .env("HOME", world.path("home"))
        .env("ARIA_HOME", world.path("home/.aria"))
        .env("PATH", world.path("bin"))
        .env("ARIA_SSH", concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fake_ssh.sh"))
        .env("ARIA_FAKE_SSH_LOG", world.path("hub/log"))
        .env("ARIA_FAKE_SSH_ROOT", world.path("hub"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), b"y\n").unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert!(text(&output.stdout).contains("nothing was changed"), "{}", text(&output.stdout));
    assert!(writes(&world.log()).is_empty());
}

#[test]
fn a_yes_uploads_installs_and_imports_in_order() {
    let world = World::new();
    let output = world.init_yes(&[], &[]);
    let out = text(&output.stdout);
    assert!(output.status.success(), "{out}{}", text(&output.stderr));
    let log = world.log();
    let steps = writes(&log);
    assert_eq!(steps.len(), 5, "{steps:?}");
    assert!(position(&steps, "tar -xf") < position(&steps, "install.sh --headless --local"));
    assert!(position(&steps, "install.sh --headless --local") < position(&steps, "aria hub-import"));
    assert!(position(&steps, "aria hub-import") < position(&steps, "systemctl --user restart aria-serve"));
    assert!(position(&steps, "systemctl --user restart aria-serve") < position(&steps, "aria --version"));
    assert!(steps[1].ends_with("install.sh --headless --local ~/.aria/hub-src/aria"), "{}", steps[1]);
    assert!(steps[2].ends_with("~/.local/bin/aria hub-import"), "{}", steps[2]);
    assert!(steps[3].ends_with("systemctl --user restart aria-serve.service"), "{}", steps[3]);
    // Every remote command is one argument after the host, and the host is the one given.
    assert!(log.iter().filter(|line| !line.starts_with("-G ")).all(|line| line.starts_with("ana@hub.example\t")));
    // The tree and the binary arrived where the installer reads them.
    let src = world.path("hub/home/.aria/hub-src");
    assert_eq!(std::fs::read_to_string(src.join("install/install.sh")).unwrap(), "#!/bin/sh\necho fake installer\n");
    assert!(src.join("install/aria-serve.service").is_file());
    for pin in PINS {
        assert_eq!(std::fs::read_to_string(src.join(pin)).unwrap(), format!("{pin} contents\n"), "{pin}");
    }
    assert_eq!(std::fs::read(src.join("aria")).unwrap(), elf(0xB7));
    // The bundle went as stdin of hub-import (its size is logged), and its bytes are in no log or output.
    let import_line = log.iter().find(|line| line.contains("hub-import")).unwrap();
    let length: usize = import_line.rsplit("stdin=").next().unwrap().parse().unwrap();
    assert!(length > 100, "{import_line}");
    let everything =
        format!("{}{}{}", std::fs::read_to_string(world.path("hub/log")).unwrap(), out, text(&output.stderr));
    assert!(!everything.contains("sk-never-in-a-log"));
    assert!(out.contains("aria 0.0.0-fake"), "the hub's version is shown: {out}");
}

#[test]
fn a_public_sshd_is_warned_once_and_does_not_block() {
    let world = World::new();
    let output = world.init_yes(&[], &[]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    let warnings: Vec<&str> = out.lines().filter(|line| line.contains("2222") && line.contains("open")).collect();
    assert_eq!(warnings.len(), 1, "{out}");
    assert!(out.contains("restricted"), "{out}");
    assert!(out.contains("password logins look off"), "{out}");
    let passwords = world.init_yes(&[], &[("ARIA_FAKE_SSH_PASSWORDS", "yes")]);
    assert!(text(&passwords.stdout).contains("password logins look on"), "{}", text(&passwords.stdout));
    // A server that only listens on its private network has nothing to warn of.
    let private = world.init_yes(&[], &[("ARIA_FAKE_SSH_PUBLIC", "0")]);
    assert_eq!(private.status.code(), Some(0));
    assert!(!text(&private.stdout).contains("stays open"), "{}", text(&private.stdout));
}

#[test]
fn a_sandbox_that_cannot_work_is_said_with_the_fix_printed_not_applied() {
    let world = World::new();
    let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_BWRAP_FAIL", "1")]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    assert!(out.contains("kernel.apparmor_restrict_unprivileged_userns"), "{out}");
    assert!(out.contains("Permission denied"), "what bwrap said: {out}");
    assert!(out.contains("ask"), "{out}");
    for command in commands(&world.log()) {
        assert!(!command.contains("sysctl") && !command.contains("apparmor") && !command.contains("sudo"), "{command}");
    }
    // A sandbox that works says nothing of it.
    let fine = world.init_yes(&[], &[]);
    assert!(!text(&fine.stdout).contains("apparmor"), "{}", text(&fine.stdout));
}

#[test]
fn the_sandbox_probe_does_what_the_real_call_does_so_a_server_that_cannot_mount_proc_is_caught() {
    let world = World::new();
    let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_BWRAP_NOPROC", "1")]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    assert!(out.contains("Can't mount proc"), "what bwrap said: {out}");
    assert!(out.contains("kernel.apparmor_restrict_unprivileged_userns"), "the fix is printed: {out}");
    // The probe asks for what the sandbox extension asks for: the disk read-only, /dev and /proc.
    let probe = commands(&world.log()).into_iter().find(|c| c.starts_with("bwrap ")).expect("a probe ran");
    assert_eq!(probe, "bwrap --unshare-user --ro-bind / / --dev /dev --proc /proc true");
    // And no key is installed on such a server without the flag.
    let key = key_file(&world);
    let refused = world.init_yes(&["--client-key", &key], &[("ARIA_FAKE_SSH_BWRAP_NOPROC", "1")]);
    assert_eq!(refused.status.code(), Some(1), "{}{}", text(&refused.stdout), text(&refused.stderr));
    assert_eq!(world.authorized_keys(), "");
}

#[test]
fn a_client_key_is_installed_restricted_and_once() {
    let world = World::new();
    let key_file = world.path("laptop.pub");
    std::fs::write(&key_file, format!("{KEY}\n")).unwrap();
    let key_arg = key_file.to_str().unwrap();
    let output = world.init_yes(&["--client-key", key_arg], &[]);
    assert!(output.status.success(), "{}{}", text(&output.stdout), text(&output.stderr));
    // The hub's `$HOME` is the fake's scratch home here (it is `/home/ana` in what the plan shows).
    let expected =
        format!("restrict,command=\"{}/.local/bin/aria client-stdio\" {KEY}\n", world.path("hub/home").display());
    assert!(
        text(&output.stdout).contains(&format!("restrict,command=\"/home/ana/.local/bin/aria client-stdio\" {KEY}")),
        "the plan shows the line"
    );
    assert_eq!(world.authorized_keys(), expected);
    let mode = |sub: &str| {
        std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(world.path(sub)).unwrap().permissions()) & 0o777
    };
    assert_eq!(mode("hub/home/.ssh"), 0o700);
    assert_eq!(mode("hub/home/.ssh/authorized_keys"), 0o600);
    // The key is installed after the version check, and after the sandbox probe (it is not installed without one).
    let steps = writes(&world.log());
    assert!(position(&steps, "aria --version") < position(&steps, "authorized_keys"));
    let all = commands(&world.log());
    assert!(position(&all, "bwrap --unshare-user") < position(&all, "authorized_keys"), "{all:?}");
    // A second run does not add it again, and another key already there is kept.
    let again = world.init_yes(&["--client-key", key_arg], &[]);
    assert!(again.status.success());
    assert_eq!(world.authorized_keys(), expected);
    let other = format!("{}\n", KEY.replace("ana@laptop", "ana@desktop"));
    std::fs::write(&key_file, other).unwrap();
    let third = world.init_yes(&["--client-key", key_arg], &[]);
    assert!(third.status.success());
    assert_eq!(world.authorized_keys(), expected, "the same key blob under another comment is the same key");
}

#[test]
fn a_pubkey_with_its_own_options_is_refused() {
    let world = World::new();
    let key_file = world.path("evil.pub");
    for bad in [
        format!("command=\"sh\" {KEY}\n"),
        format!("no-pty {KEY}\n"),
        format!("{KEY}\n{KEY}\n"),
        format!("{KEY}\nssh-ed25519 AAAA other\n"),
        format!("{KEY}\" ,command=\"x\n"),
        "ssh-ed25519 AAAA\"\n".to_string(),
        String::new(),
        "\n".to_string(),
    ] {
        std::fs::write(&key_file, &bad).unwrap();
        let output = world.init_yes(&["--client-key", key_file.to_str().unwrap()], &[]);
        assert_eq!(output.status.code(), Some(2), "{bad:?}: {}{}", text(&output.stdout), text(&output.stderr));
        assert!(world.log().is_empty(), "nothing is sent for {bad:?}: {:?}", world.log());
    }
    let missing = world.init_yes(&["--client-key", "/nonexistent/key.pub"], &[]);
    assert_eq!(missing.status.code(), Some(2));
    assert!(world.log().is_empty());
}

#[test]
fn a_binary_of_another_architecture_is_refused() {
    let world = World::new();
    let output = world.init(&["ana@hub.example", "--binary", "bin/aria-x86_64", "--yes"], &[]);
    let err = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{}{err}", text(&output.stdout));
    assert!(err.contains("x86_64") && err.contains("aarch64"), "{err}");
    assert!(writes(&world.log()).is_empty());
    // A file that is no ELF at all.
    std::fs::write(world.path("repo/bin/not-elf"), "#!/bin/sh\n").unwrap();
    let output = world.init(&["ana@hub.example", "--binary", "bin/not-elf", "--yes"], &[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(writes(&world.log()).is_empty());
    // The same binary suits a host of its own architecture.
    let output =
        world.init(&["ana@hub.example", "--binary", "bin/aria-x86_64", "--yes"], &[("ARIA_FAKE_SSH_ARCH", "x86_64")]);
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
}

#[test]
fn without_a_binary_the_cross_build_is_used_or_asked_for() {
    let world = World::new();
    let output = world.init(&["ana@hub.example", "--yes"], &[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(text(&output.stderr).contains("sh tools/build-headless.sh"), "{}", text(&output.stderr));
    assert!(world.log().is_empty());
    let built = world.path("repo/target/aarch64-unknown-linux-gnu/release");
    std::fs::create_dir_all(&built).unwrap();
    std::fs::write(built.join("aria"), elf(0xB7)).unwrap();
    let output = world.init(&["ana@hub.example", "--yes"], &[]);
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
    assert_eq!(std::fs::read(world.path("hub/home/.aria/hub-src/aria")).unwrap(), elf(0xB7));
}

#[test]
fn an_option_like_host_is_refused() {
    let world = World::new();
    for host in ["-oProxyCommand=touch-pwned", "-F/tmp/evil"] {
        let output = world.init(&[host, "--binary", "bin/aria-aarch64", "--yes"], &[]);
        assert_eq!(output.status.code(), Some(2), "{host}");
        assert!(text(&output.stderr).contains("not a host"), "{}", text(&output.stderr));
    }
    assert!(world.log().is_empty(), "ssh never ran: {:?}", world.log());
}

#[test]
fn the_bundle_warnings_are_printed_before_anything_is_sent() {
    let world = World::new();
    // The keyring is on and the sealed file is nowhere: the keyring cannot be opened (this PATH has no secret-tool).
    std::fs::write(world.path("home/.aria/settings.json"), r#"{"language":"en"}"#).unwrap();
    let output = world.init(&["ana@hub.example", "--binary", "bin/aria-aarch64"], &[]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    let note = out.find("note: mcp-auth.json").expect("the warning is printed");
    let plan = out.find("1. ").unwrap();
    assert!(note < plan, "before the plan: {out}");
    assert!(out.contains("never had a keyring"), "{out}");
    assert!(writes(&world.log()).is_empty());
}

#[test]
fn what_the_hub_says_about_a_refused_bundle_is_never_shown() {
    let world = World::new();
    let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_IMPORT_FAIL", "1")]);
    assert_eq!(output.status.code(), Some(1));
    let all = format!("{}{}", text(&output.stdout), text(&output.stderr));
    assert!(all.contains("the hub refused the bundle"), "{all}");
    assert!(all.contains("exit status 1"), "{all}");
    assert!(!all.contains("invalid value"), "stderr of hub-import is never shown: {all}");
    assert!(!all.contains("files"), "{all}");
    // It stops there: the service is not restarted.
    let steps = writes(&world.log());
    assert!(!steps.iter().any(|c| c.contains("systemctl")), "{steps:?}");
}

#[test]
fn a_hub_that_would_need_xz_is_not_touched_and_told_the_command() {
    let world = World::new();
    let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_MISSING", "xz")]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(1), "{out}{}", text(&output.stderr));
    assert!(out.contains("sudo apt-get install -y xz-utils"), "{out}");
    assert!(writes(&world.log()).is_empty(), "nothing is changed");
    for command in commands(&world.log()) {
        assert!(!command.contains("apt-get"), "{command}");
    }
    // With a Node that is new enough the installer needs no xz.
    let fine = world.init_yes(&[], &[("ARIA_FAKE_SSH_MISSING", "xz"), ("ARIA_FAKE_SSH_NODE", "v22.23.2")]);
    assert_eq!(fine.status.code(), Some(0), "{}{}", text(&fine.stdout), text(&fine.stderr));
}

/// An ssh that fails for the commands matching `pattern` (a shell `case` pattern) with `status`, and is the fake otherwise.
fn failing_ssh(world: &World, pattern: &str, status: u8) -> String {
    let failing = world.path("bin/failing-ssh");
    std::fs::write(
        &failing,
        format!(
            "#!/bin/sh\ncase \"$*\" in {pattern}) echo 'the hub said no' >&2; exit {status} ;; esac\nexec {} \"$@\"\n",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fake_ssh.sh")
        ),
    )
    .unwrap();
    std::fs::set_permissions(&failing, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    failing.to_str().unwrap().to_string()
}

#[test]
fn a_failing_step_stops_and_says_which_and_what_is_already_on_the_hub() {
    let world = World::new();
    let ssh = failing_ssh(&world, "*install.sh*", 3);
    let output = world.init_yes(&[], &[("ARIA_SSH", &ssh)]);
    let err = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{}{err}", text(&output.stdout));
    assert!(err.contains("step 2"), "{err}");
    assert!(
        err.contains(
            "`~/.aria/hub-src` is uploaded and the installer may have partly run; running `aria hub init` again is safe"
        ),
        "{err}"
    );
    let steps = writes(&world.log());
    assert!(!steps.iter().any(|c| c.contains("hub-import")), "{steps:?}");

    let world = World::new();
    let ssh = failing_ssh(&world, "*hub-import*", 1);
    let err = text(&world.init_yes(&[], &[("ARIA_SSH", &ssh)]).stderr);
    assert!(err.contains("step 3") && err.contains("the installer ran; nothing was imported"), "{err}");

    let world = World::new();
    let ssh = failing_ssh(&world, "*systemctl*", 1);
    let output = world.init_yes(&[], &[("ARIA_SSH", &ssh)]);
    let err = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1));
    assert!(err.contains("step 4"), "{err}");
    assert!(err.contains("the credentials and settings are already on the hub, the service did not restart"), "{err}");
    assert!(!writes(&world.log()).iter().any(|c| c.contains("--version")), "it stops there");

    let world = World::new();
    let ssh = failing_ssh(&world, "*\"aria --version\"*", 1);
    let err = text(&world.init_yes(&[], &[("ARIA_SSH", &ssh)]).stderr);
    assert!(
        err.contains("step 5") && err.contains("everything was installed and imported; only the version check failed"),
        "{err}"
    );
}

/// The key as a person keeps it in `authorized_keys` of the hub, written where the fake hub's `$HOME` is.
fn seed_authorized_keys(world: &World, content: &str) {
    let ssh = world.path("hub/home/.ssh");
    std::fs::create_dir_all(&ssh).unwrap();
    std::fs::write(ssh.join("authorized_keys"), content).unwrap();
}

fn key_file(world: &World) -> String {
    let file = world.path("laptop.pub");
    std::fs::write(&file, format!("{KEY}\n")).unwrap();
    file.to_str().unwrap().to_string()
}

fn restricted(world: &World) -> String {
    format!("restrict,command=\"{}/.local/bin/aria client-stdio\" {KEY}", world.path("hub/home").display())
}

#[test]
fn a_file_with_no_final_newline_does_not_get_the_new_line_glued_on() {
    let world = World::new();
    let other = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOtherKeyBlobOtherKeyBlobOtherKeyBlob10 ana@desktop";
    seed_authorized_keys(&world, other);
    let output = world.init_yes(&["--client-key", &key_file(&world)], &[]);
    assert!(output.status.success(), "{}{}", text(&output.stdout), text(&output.stderr));
    assert_eq!(world.authorized_keys(), format!("{other}\n{}\n", restricted(&world)));
    // A second run finds the key as it is now: still two lines.
    let again = world.init_yes(&["--client-key", &key_file(&world)], &[]);
    assert!(again.status.success());
    assert_eq!(world.authorized_keys(), format!("{other}\n{}\n", restricted(&world)));
}

#[test]
fn a_key_already_there_without_the_restriction_is_reported_not_duplicated() {
    let world = World::new();
    for existing in [KEY.to_string(), format!("no-pty {KEY}"), format!("command=\"sh\" {KEY}")] {
        seed_authorized_keys(&world, &format!("{existing}\n"));
        let output = world.init_yes(&["--client-key", &key_file(&world)], &[]);
        let (out, err) = (text(&output.stdout), text(&output.stderr));
        assert_eq!(output.status.code(), Some(1), "{existing}: {out}{err}");
        assert!(err.contains("step 6"), "{err}");
        assert!(err.contains("already in authorized_keys WITHOUT the restriction"), "{err}");
        assert!(err.contains("restrict,command=\"/home/ana/.local/bin/aria client-stdio\""), "{err}");
        assert_eq!(world.authorized_keys(), format!("{existing}\n"), "the file is not touched");
        // The steps before it ran, and no claim is made that the key protects the hub.
        assert!(writes(&world.log()).iter().any(|c| c.contains("systemctl")));
        assert!(!out.contains("restricted client key (it can only run"), "{out}");
        assert!(!out.contains("The hub is ready"), "{out}");
        assert!(out.contains("stays open"), "the public sshd is still warned of: {out}");
    }
}

#[test]
fn a_key_already_there_with_the_restriction_is_left_as_it_is() {
    let world = World::new();
    seed_authorized_keys(&world, &format!("# mine\n{}\n", restricted(&world)));
    let output = world.init_yes(&["--client-key", &key_file(&world)], &[]);
    assert!(output.status.success(), "{}{}", text(&output.stdout), text(&output.stderr));
    assert_eq!(world.authorized_keys(), format!("# mine\n{}\n", restricted(&world)));
    assert!(text(&output.stdout).contains("already allowed"), "{}", text(&output.stdout));
}

#[test]
fn a_commented_out_copy_of_the_key_does_not_count() {
    let world = World::new();
    for comment in [format!("# {KEY}"), format!("   #{KEY}")] {
        seed_authorized_keys(&world, &format!("{comment}\n"));
        let output = world.init_yes(&["--client-key", &key_file(&world)], &[]);
        assert!(output.status.success(), "{}{}", text(&output.stdout), text(&output.stderr));
        assert_eq!(world.authorized_keys(), format!("{comment}\n{}\n", restricted(&world)));
    }
}

#[test]
fn an_ss_that_fails_is_said_not_taken_for_a_private_hub() {
    let could_not = "could not check whether sshd is open to the internet";
    // ss fails before and after the install, or answers nothing: one line says it, the run still succeeds.
    for variant in ["fail", "empty"] {
        let world = World::new();
        let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_SS", variant)]);
        let out = text(&output.stdout);
        assert_eq!(output.status.code(), Some(0), "{variant}: {out}{}", text(&output.stderr));
        assert_eq!(out.lines().filter(|line| line.contains(could_not)).count(), 1, "{variant}: {out}");
        assert!(out.contains("check it yourself"), "{out}");
        assert!(!out.contains("stays open"), "{out}");
        let asked = commands(&world.log()).iter().filter(|c| *c == "ss -ltnH -n").count();
        assert_eq!(asked, 2, "asked again after the installer: {variant}");
    }
    // ss missing until the installer installs iproute2: the second try sees the public sshd.
    let world = World::new();
    let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_SS", "fail-until-install")]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    assert!(out.contains("stays open") && !out.contains(could_not), "{out}");
    // The ordinary cases say what they always did, and ask once.
    let world = World::new();
    let public = text(&world.init_yes(&[], &[]).stdout);
    assert!(public.contains("stays open") && !public.contains(could_not), "{public}");
    assert_eq!(commands(&world.log()).iter().filter(|c| *c == "ss -ltnH -n").count(), 1);
    let private = text(&world.init_yes(&[], &[("ARIA_FAKE_SSH_PUBLIC", "0")]).stdout);
    assert!(!private.contains("stays open") && !private.contains(could_not), "{private}");
}

const NO_PROMPT_WARNING: &str = "anyone holding the client key can approve those prompts";

#[test]
fn without_a_working_sandbox_the_key_is_not_installed_unless_allowed() {
    for (variant, fix) in
        [("missing", "sudo apt-get install -y bubblewrap"), ("1", "kernel.apparmor_restrict_unprivileged_userns")]
    {
        let world = World::new();
        let key = key_file(&world);
        let output = world.init_yes(&["--client-key", &key], &[("ARIA_FAKE_SSH_BWRAP_FAIL", variant)]);
        let (out, err) = (text(&output.stdout), text(&output.stderr));
        assert_eq!(output.status.code(), Some(1), "{variant}: {out}{err}");
        assert!(err.contains("step 6") && err.contains("skipped"), "{variant}: {err}");
        assert!(err.contains("--allow-no-sandbox"), "{variant}: {err}");
        assert!(out.contains(fix), "{variant}: {out}");
        assert!(out.contains("warning: the sandbox cannot start"), "{variant}: {out}");
        assert!(out.contains(NO_PROMPT_WARNING) && out.contains("shell access to the hub user"), "{variant}: {out}");
        assert!(!out.contains("The hub is ready"), "{out}");
        assert_eq!(world.authorized_keys(), "", "{variant}: the key was not installed");
        assert!(!commands(&world.log()).iter().any(|c| c.contains("client-stdio")), "{variant}");
        // The other steps ran.
        assert!(writes(&world.log()).iter().any(|c| c.contains("systemctl")), "{variant}");

        // With the flag the key is installed and the same warning is printed.
        let output =
            world.init_yes(&["--client-key", &key, "--allow-no-sandbox"], &[("ARIA_FAKE_SSH_BWRAP_FAIL", variant)]);
        let (out, err) = (text(&output.stdout), text(&output.stderr));
        assert_eq!(output.status.code(), Some(0), "{variant}: {out}{err}");
        assert!(out.contains(NO_PROMPT_WARNING), "{variant}: {out}");
        assert_eq!(world.authorized_keys(), format!("{}\n", restricted(&world)), "{variant}");
    }
}

#[test]
fn a_missing_bwrap_and_a_blocked_one_are_told_apart() {
    let world = World::new();
    let missing = text(&world.init_yes(&[], &[("ARIA_FAKE_SSH_BWRAP_FAIL", "missing")]).stdout);
    assert!(missing.contains("not found") && missing.contains("sudo apt-get install -y bubblewrap"), "{missing}");
    assert!(!missing.contains("apparmor"), "{missing}");
    let blocked = text(&world.init_yes(&[], &[("ARIA_FAKE_SSH_BWRAP_FAIL", "1")]).stdout);
    assert!(blocked.contains("kernel.apparmor_restrict_unprivileged_userns"), "{blocked}");
    assert!(!blocked.contains("apt-get install -y bubblewrap"), "{blocked}");
    // No key asked for: nothing is skipped, and the run still succeeds, as before.
    assert!(world.init_yes(&[], &[("ARIA_FAKE_SSH_BWRAP_FAIL", "missing")]).status.success());
}

#[test]
fn with_a_working_sandbox_the_key_goes_in_with_no_flag() {
    let world = World::new();
    let output = world.init_yes(&["--client-key", &key_file(&world)], &[]);
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
    assert_eq!(world.authorized_keys(), format!("{}\n", restricted(&world)));
    assert!(!text(&output.stdout).contains("warning: the sandbox cannot start"));
}

#[test]
fn a_restricted_key_that_went_in_comes_with_the_ssh_config_to_use_it() {
    let world = World::new();
    let key = key_file(&world);
    let output = world.init_yes(&["--client-key", &key], &[]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    let private = key.trim_end_matches(".pub");
    // The host and user are the ones given; the identity is the key's file without `.pub`.
    let snippet = format!(
        "Host aria-hub\n    HostName hub.example\n    User ana\n    IdentityFile {private}\n    IdentitiesOnly yes\n"
    );
    assert!(out.contains(&snippet), "{out}");
    assert!(out.contains("~/.ssh/config") && out.contains("aria connect aria-hub"), "{out}");
    // A key that was already there, restricted, is as good: same snippet.
    let again = text(&world.init_yes(&["--client-key", &key], &[]).stdout);
    assert!(again.contains(&snippet), "{again}");
    assert!(again.contains("The hub is ready. Connect from this computer with: aria connect aria-hub"), "{again}");
    // No key asked for, no snippet.
    let plain = text(&World::new().init_yes(&[], &[]).stdout);
    assert!(!plain.contains("IdentityFile") && !plain.contains("Host aria-hub"), "{plain}");
}

#[test]
fn a_login_key_already_in_authorized_keys_is_told_to_be_a_dedicated_key() {
    let world = World::new();
    seed_authorized_keys(&world, &format!("{KEY}\n"));
    let output = world.init_yes(&["--client-key", &key_file(&world)], &[]);
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "{out}{err}");
    assert!(err.contains("use a dedicated key, not your login key"), "{err}");
    assert!(!out.contains("IdentityFile"), "no snippet for a key that is not restricted: {out}");
}

/// The hub's files, as an owner who already ran `hub init` has them: each holds its own name, so a change shows.
fn seed_hub_files(world: &World) {
    let dir = world.path("hub/home/.aria");
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["settings.json", "auth.json", "mcp.json", "mcp-auth.json"] {
        std::fs::write(dir.join(name), format!("{{\"hub's own\":\"{name}\"}}")).unwrap();
    }
}

/// The names the hub was last sent in a bundle (none when nothing was).
fn last_bundle(world: &World) -> Vec<String> {
    let bundle = std::fs::read_to_string(world.path("hub/last-bundle")).unwrap_or_default();
    ["settings.json", "auth.json", "mcp.json", "mcp-auth.json"]
        .into_iter()
        .filter(|name| bundle.contains(&format!("\"{name}\":")))
        .map(str::to_string)
        .collect()
}

fn owner_has_all_files(world: &World) {
    std::fs::write(world.path("home/.aria/mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::fs::write(world.path("home/.aria/mcp-auth.json"), r#"{"mcp__notion|u":{}}"#).unwrap();
}

#[test]
fn a_rerun_sends_nothing_and_leaves_the_running_service_alone() {
    let world = World::new();
    owner_has_all_files(&world);
    seed_hub_files(&world);
    let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_ACTIVE", "1")]);
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(0), "{out}{err}");
    let steps = writes(&world.log());
    assert!(!steps.iter().any(|c| c.contains("hub-import")), "no bundle was sent: {steps:?}");
    assert!(!steps.iter().any(|c| c.contains("restart")), "no restart: {steps:?}");
    assert!(
        out.contains("the service keeps running: it takes the new binary when idle and the settings within a second"),
        "{out}"
    );
    assert!(
        out.contains("Kept as the hub has them") && out.contains("--credentials") && out.contains("--settings"),
        "the plan says what is kept: {out}"
    );
    assert_eq!(last_bundle(&world), Vec::<String>::new());
    // the hub's own files were not touched
    assert_eq!(
        std::fs::read_to_string(world.path("hub/home/.aria/auth.json")).unwrap(),
        r#"{"hub's own":"auth.json"}"#
    );
    // the installer and the version check still ran
    assert!(steps.iter().any(|c| c.contains("install.sh")) && steps.iter().any(|c| c.contains("--version")));
}

#[test]
fn a_flag_sends_only_the_files_it_names() {
    let world = World::new();
    owner_has_all_files(&world);
    seed_hub_files(&world);
    let output = world.init_yes(&["--credentials"], &[("ARIA_FAKE_SSH_ACTIVE", "1")]);
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
    assert_eq!(last_bundle(&world), ["auth.json", "mcp.json", "mcp-auth.json"]);
    assert!(!writes(&world.log()).iter().any(|c| c.contains("restart")), "the service was active: no restart");
    let world = World::new();
    owner_has_all_files(&world);
    seed_hub_files(&world);
    let output = world.init_yes(&["--settings"], &[("ARIA_FAKE_SSH_ACTIVE", "1")]);
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
    assert_eq!(last_bundle(&world), ["settings.json"]);
    // both flags: all four
    let world = World::new();
    owner_has_all_files(&world);
    seed_hub_files(&world);
    world.init_yes(&["--settings", "--credentials"], &[("ARIA_FAKE_SSH_ACTIVE", "1")]);
    assert_eq!(last_bundle(&world), ["settings.json", "auth.json", "mcp.json", "mcp-auth.json"]);
}

#[test]
fn a_first_init_sends_everything_and_restarts_the_service() {
    let world = World::new();
    owner_has_all_files(&world);
    let output = world.init_yes(&[], &[]);
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
    assert_eq!(last_bundle(&world), ["settings.json", "auth.json", "mcp.json", "mcp-auth.json"]);
    assert!(writes(&world.log()).iter().any(|c| c.contains("restart")));
    assert!(!text(&output.stdout).contains("the service keeps running"));
}

#[test]
fn a_hub_with_only_some_files_gets_the_missing_ones() {
    let world = World::new();
    owner_has_all_files(&world);
    let dir = world.path("hub/home/.aria");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("auth.json"), "{}").unwrap();
    let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_ACTIVE", "1")]);
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
    assert_eq!(
        last_bundle(&world),
        ["settings.json", "mcp.json", "mcp-auth.json"],
        "auth.json stays as the hub has it"
    );
    assert!(!writes(&world.log()).iter().any(|c| c.contains("restart")), "still no restart: it was active");
}

const XDG: &str = "XDG_RUNTIME_DIR=/run/user/$(id -u)";
const BUS: &str = "DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus";

#[test]
fn the_remote_commands_that_use_the_user_manager_set_its_runtime_directory() {
    let world = World::new();
    let output = world.init_yes(&[], &[]);
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
    let all = commands(&world.log());
    for needle in ["install.sh --headless", "systemctl --user restart", "systemctl --user is-active"] {
        let command = &all[position(&all, needle)];
        assert!(command.contains(XDG) && command.contains(BUS), "{needle}: {command}");
        assert!(
            command.contains("[ -z \"$XDG_RUNTIME_DIR\" ]") && command.contains("-d \"/run/user/$(id -u)\""),
            "{command}"
        );
    }
    // the reads that do not need the user manager are left alone
    for needle in ["uname -m", "echo $HOME", "ss -ltnH -n", "command -v"] {
        assert!(!all[position(&all, needle)].contains("XDG_RUNTIME_DIR"), "{needle}");
    }
}

#[test]
fn a_failing_systemctl_names_dbus_user_session_and_linger() {
    let world = World::new();
    let ssh = failing_ssh(&world, "*systemctl*restart*", 1);
    let output = world.init_yes(&[], &[("ARIA_SSH", &ssh)]);
    let err = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{}{err}", text(&output.stdout));
    assert!(err.contains("step 4") && err.contains("dbus-user-session") && err.contains("linger"), "{err}");
    assert!(err.contains("sudo loginctl enable-linger ana"), "{err}");
}

#[test]
fn the_hub_is_ready_only_when_its_user_lingers() {
    let world = World::new();
    let output = world.init_yes(&[], &[]);
    let out = text(&output.stdout);
    assert!(out.contains("The hub is ready"), "{out}");
    assert!(!out.contains("linger is not on"), "{out}");
    let probe = commands(&world.log()).into_iter().find(|c| c.contains("loginctl")).expect("a linger probe");
    assert_eq!(probe, "loginctl show-user \"$USER\" -p Linger");
    // it is asked after the version check
    let all = commands(&world.log());
    assert!(position(&all, "aria --version") < position(&all, "loginctl show-user"), "{all:?}");
    for variant in ["no", "fail"] {
        let world = World::new();
        let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_LINGER", variant)]);
        let (out, err) = (text(&output.stdout), text(&output.stderr));
        assert_eq!(output.status.code(), Some(0), "{variant}: {out}{err}");
        assert!(!out.contains("The hub is ready"), "{variant}: {out}");
        assert!(out.contains("stops when you log out"), "{variant}: {out}");
        assert!(out.contains("sudo loginctl enable-linger ana"), "{variant}: {out}");
        for command in commands(&world.log()) {
            assert!(!command.contains("sudo"), "nothing is run with sudo: {command}");
        }
    }
}

#[test]
fn the_plan_says_the_browser_is_left_out_of_a_server() {
    let world = World::new();
    let out = text(&world.init(&["ana@hub.example", "--binary", "bin/aria-aarch64"], &[]).stdout);
    assert!(out.contains("--with-browser") && out.contains("agent-browser"), "{out}");
}

#[test]
fn the_plan_says_when_the_binary_was_built() {
    let world = World::new();
    let file = std::fs::File::options().write(true).open(world.path("repo/bin/aria-aarch64")).unwrap();
    file.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000)).unwrap();
    let output = world.init(&["ana@hub.example", "--binary", "bin/aria-aarch64"], &[]);
    let out = text(&output.stdout);
    assert!(output.status.success(), "{out}{}", text(&output.stderr));
    assert!(out.contains("bin/aria-aarch64 (built 2023-11-14 22:13 UTC)"), "{out}");
    // the built binary of the default path is shown the same way, and an old one is not refused
    let built = world.path("repo/target/aarch64-unknown-linux-gnu/release");
    std::fs::create_dir_all(&built).unwrap();
    std::fs::write(built.join("aria"), elf(0xB7)).unwrap();
    let default = std::fs::File::options().write(true).open(built.join("aria")).unwrap();
    default.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000)).unwrap();
    let output = world.init(&["ana@hub.example"], &[]);
    let out = text(&output.stdout);
    assert!(output.status.success(), "{out}{}", text(&output.stderr));
    assert!(out.contains("release/aria (built 2001-09-09 01:46 UTC)"), "{out}");
}
