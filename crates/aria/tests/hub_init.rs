//! `aria hub init` as a process, against a fake ssh (tests/fake_ssh.sh) and a scratch repository and home: nothing here
//! touches the owner's `~/.aria`, keyring, ssh configuration or network.
#![cfg(unix)]

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

const PINS: [&str; 5] = ["pi-version", "pi-packages", "engram-version", "agent-browser-version", "node-version"];
const KEY: &str = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl ana@laptop";
const TOOLS: [&str; 11] = ["sh", "cat", "wc", "tr", "mkdir", "tar", "cut", "grep", "touch", "chmod", "rm"];

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
    const PROBES: [&str; 6] =
        ["uname -m", "cat /etc/os-release", "echo $HOME", "ss -ltnH -n", "command -v", "node --version"];
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
    assert_eq!(steps[3], "systemctl --user restart aria-serve.service");
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
    // The key is installed after the version check and before the sandbox probe.
    let steps = writes(&world.log());
    assert!(position(&steps, "aria --version") < position(&steps, "authorized_keys"));
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

#[test]
fn a_failing_step_stops_and_says_which() {
    let world = World::new();
    // A hub whose installer dies: the fake's `install.sh` answer is replaced by a failure through a ssh that fails.
    let failing = world.path("bin/failing-ssh");
    std::fs::write(
        &failing,
        format!(
            "#!/bin/sh\ncase \"$*\" in *install.sh*) echo 'installer said no' >&2; exit 3 ;; esac\nexec {} \"$@\"\n",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fake_ssh.sh")
        ),
    )
    .unwrap();
    std::fs::set_permissions(&failing, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    let output = world.init_yes(&[], &[("ARIA_SSH", failing.to_str().unwrap())]);
    let err = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{}{err}", text(&output.stdout));
    assert!(err.contains("step 2"), "{err}");
    let steps = writes(&world.log());
    assert!(!steps.iter().any(|c| c.contains("hub-import")), "{steps:?}");
}
