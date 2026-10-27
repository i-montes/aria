//! `aria hub init` as a process, against a fake ssh (tests/fake_ssh.sh) and a scratch repository and home: nothing here
//! touches the owner's `~/.aria`, keyring, ssh configuration or network.
#![cfg(unix)]

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

const PINS: [&str; 5] = ["pi-version", "pi-packages", "engram-version", "agent-browser-version", "node-version"];
const KEY: &str = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl ana@laptop";
const TOOLS: [&str; 15] =
    ["sh", "head", "tail", "cat", "wc", "tr", "mkdir", "tar", "cut", "grep", "touch", "chmod", "rm", "cp", "sleep"];

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
        // Most tests are about the hub's install, not the key: without a key flag they leave the default key out, which would
        // be made on this computer (the tests that want it name `--client-key` or say so with `init_with_default_key`).
        let keyless =
            !args.iter().any(|arg| *arg == "--client-key" || *arg == "--no-client-key" || *arg == "--default-key");
        command
            .args(["hub", "init"])
            .args(args.iter().filter(|arg| **arg != "--default-key"))
            .args(if keyless { vec!["--no-client-key"] } else { vec![] })
            .current_dir(self.path("repo"))
            .env_clear()
            .env("HOME", self.path("home"))
            .env("ARIA_HOME", self.path("home/.aria"))
            .env("PATH", self.path("bin"))
            .env("ARIA_SSH", concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fake_ssh.sh"))
            .env("ARIA_FAKE_SSH_LOG", self.path("hub/log"))
            .env("ARIA_FAKE_SSH_ROOT", self.path("hub"))
            .env("ARIA_CONNECT_WORD", "aria")
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

/// The words of a text with every run of white space made one space: a sentence the screen wrapped is one line again.
fn flat(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// No line the screen prints on stdout is wider than a narrow terminal.
fn assert_fits(out: &str) {
    for line in out.lines() {
        assert!(line.chars().count() <= 78, "{} columns: {line:?}\n{out}", line.chars().count());
    }
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
        .filter(|command| command != "~/.local/bin/aria client-stdio")
        .filter(|command| !command.ends_with("aria --version 2>/dev/null"))
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
    let output = world.init(&["ana@hub.example", "--binary", "bin/aria-aarch64", "--details"], &[]);
    let out = text(&output.stdout);
    assert!(output.status.success(), "{out}{}", text(&output.stderr));
    // The plan in plain words, and with --details every command before it.
    assert!(out.contains("What I will do on ana@hub.example"), "{out}");
    assert!(out.contains("1  Install Aria on the server (the build you gave me)"), "{out}");
    assert!(!out.contains("Press Enter"), "nobody is asked without a terminal: {out}");
    let details = flat(&out);
    assert!(details.contains("install.sh --headless --local"), "{out}");
    assert!(details.contains("aria hub-import"), "{out}");
    assert!(details.contains("aria-serve.service"), "{out}");
    assert!(out.contains("nothing was changed"), "{out}");
    assert_fits(&out);
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
    let plan = out.find("What I will do on").expect("a plan");
    let first_step = out.find("[1/").expect("a first step");
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
    assert!(out.contains("The server runs Aria 0.0.0-fake"), "the hub's version is shown: {out}");
}

#[test]
fn a_public_sshd_is_warned_once_and_does_not_block() {
    let world = World::new();
    let output = world.init_yes(&[], &[]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    let said = flat(&out);
    assert_eq!(said.matches("is open to the internet").count(), 1, "{out}");
    assert!(said.contains("port 2222"), "{out}");
    assert!(said.contains("password logins look off"), "{out}");
    assert_fits(&out);
    let passwords = world.init_yes(&[], &[("ARIA_FAKE_SSH_PASSWORDS", "yes")]);
    assert!(flat(&text(&passwords.stdout)).contains("password logins look on"), "{}", text(&passwords.stdout));
    // A server that only listens on its private network has nothing to warn of.
    let private = world.init_yes(&[], &[("ARIA_FAKE_SSH_PUBLIC", "0")]);
    assert_eq!(private.status.code(), Some(0));
    assert!(!text(&private.stdout).contains("is open to the internet"), "{}", text(&private.stdout));
}

#[test]
fn a_sandbox_that_cannot_work_is_said_with_the_fix_printed_not_applied() {
    let world = World::new();
    let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_BWRAP_FAIL", "1")]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    let said = flat(&out);
    assert!(said.contains("kernel.apparmor_restrict_unprivileged_userns"), "{out}");
    assert!(said.contains("Permission denied"), "what bwrap said: {out}");
    assert!(said.contains("Aria asks you before every command"), "{out}");
    assert!(out.contains("Heads up"), "the sandbox is a note after the steps: {out}");
    assert_fits(&out);
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
    let said = flat(&out);
    assert!(said.contains("Can't mount proc"), "what bwrap said: {out}");
    // It says the mount failed, not that the namespace could not be made, and the AppArmor fixes are not offered for it.
    assert!(said.contains("mount of /proc or /dev"), "{out}");
    assert!(said.contains("do not help"), "{out}");
    assert!(
        !said.contains("kernel.apparmor_restrict_unprivileged_userns")
            && !said.contains("cannot create a user namespace"),
        "{out}"
    );
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
fn the_keyring_is_announced_before_it_is_read_and_projects_without_settings_is_said_to_do_nothing() {
    let world = World::new();
    owner_has_all_files(&world);
    let output = world.init_yes(&[], &[]);
    let out = text(&output.stdout);
    let said = "Reading your sign-ins from this computer's keyring (nothing is sent yet)";
    assert!(out.contains(said), "{out}");
    assert!(out.find(said).unwrap() < out.find("What I will do on").unwrap(), "before the plan: {out}");
    // A rerun that keeps the hub's settings: --projects does nothing, and the plan says so instead of naming a folder.
    let world = World::new();
    owner_has_all_files(&world);
    seed_hub_files(&world);
    let output = world.init_yes(&["--projects", "/srv/work", "--details"], &[("ARIA_FAKE_SSH_ACTIVE", "1")]);
    let out = flat(&text(&output.stdout));
    assert!(out.contains("--projects has no effect without --settings"), "{out}");
    assert!(!out.contains("projects folder will be"), "{out}");
    assert_eq!(last_bundle(&world), Vec::<String>::new());
    let with =
        world.init_yes(&["--projects", "/srv/work", "--settings", "--details"], &[("ARIA_FAKE_SSH_ACTIVE", "1")]);
    let out = flat(&text(&with.stdout));
    assert!(out.contains("sandbox projects folder will be /srv/work"), "{out}");
    assert!(!out.contains("no effect"), "{out}");
}

#[test]
fn a_client_key_is_installed_restricted_and_once() {
    let world = World::new();
    let key_file = world.path("laptop.pub");
    std::fs::write(&key_file, format!("{KEY}\n")).unwrap();
    let key_arg = key_file.to_str().unwrap();
    let output = world.init_yes(&["--client-key", key_arg, "--details"], &[]);
    assert!(output.status.success(), "{}{}", text(&output.stdout), text(&output.stderr));
    // The hub's `$HOME` is the fake's scratch home here (it is `/home/ana` in what the plan shows).
    let expected =
        format!("restrict,command=\"{}/.local/bin/aria client-stdio\" {KEY}\n", world.path("hub/home").display());
    assert!(
        flat(&text(&output.stdout))
            .contains(&format!("restrict,command=\"/home/ana/.local/bin/aria client-stdio\" {KEY}")),
        "the detailed plan shows the line"
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
fn without_a_binary_the_cross_build_is_used_or_the_release_is_installed() {
    let world = World::new();
    // No binary and no headless build in the checkout: the hub is installed from this Aria's release, by the hub itself.
    let output = world.init(&["ana@hub.example", "--yes"], &[]);
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
    let commands = commands(&world.log());
    assert!(!world.path("hub/home/.aria/hub-src").exists(), "nothing is uploaded");
    let version = env!("CARGO_PKG_VERSION");
    let installer = commands.iter().find(|command| command.contains("install.sh")).expect("the installer ran");
    assert!(
        installer.contains(&format!("https://github.com/i-montes/aria/releases/download/v{version}/install.sh"))
            && installer.contains(&format!("ARIA_VERSION=v{version} ARIA_NESTED=1 sh -s -- --headless --yes")),
        "{installer}"
    );
    // With the headless build made in the checkout, the upload is used as before.
    let world = World::new();
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
    let note = out.find("! mcp-auth.json").expect("the warning is printed");
    let plan = out.find("What I will do on").unwrap();
    assert!(note < plan, "before the plan: {out}");
    assert!(flat(&out).contains("never had a keyring"), "{out}");
    assert_fits(&out);
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
fn a_key_there_with_another_forced_command_is_reported_like_an_unrestricted_one() {
    let world = World::new();
    let home = world.path("hub/home").display().to_string();
    for existing in [
        format!("restrict,command=\"sh\" {KEY}"),
        format!("restrict,command=\"{home}/.local/bin/aria client-stdio --extra\" {KEY}"),
        format!("restrict,command=\"/home/other/.local/bin/aria client-stdio\" {KEY}"),
        format!("restrict,command=\"{home}/.local/bin/aria client-stdio\"{KEY}"),
    ] {
        seed_authorized_keys(&world, &format!("{existing}\n"));
        let output = world.init_yes(&["--client-key", &key_file(&world)], &[]);
        let (out, err) = (text(&output.stdout), text(&output.stderr));
        assert_eq!(output.status.code(), Some(1), "{existing}: {out}{err}");
        assert!(err.contains("step 6") && err.contains("different command"), "{existing}: {err}");
        assert!(!err.contains("WITHOUT the restriction"), "the message says the command differs: {err}");
        assert!(err.contains("restrict,command=\"/home/ana/.local/bin/aria client-stdio\""), "{err}");
        assert_eq!(world.authorized_keys(), format!("{existing}\n"), "the file is not touched");
        assert!(!out.contains("restricted client key (it can only run"), "no claim of protection: {out}");
    }
}

#[test]
fn only_tightening_options_after_the_forced_command_keep_the_line_allowed() {
    let world = World::new();
    let command = restricted(&world).trim_end_matches(&format!(" {KEY}")).to_string();
    for options in [
        ",from=\"203.0.113.0/24\"",
        ",from=\"203.0.113.1,198.51.100.0/24\"",
        ",expiry-time=\"20990101\"",
        ",verify-required",
        ",verify-required,from=\"10.0.0.0/8\",expiry-time=\"20990101\"",
    ] {
        let line = format!("{command}{options} {KEY}");
        seed_authorized_keys(&world, &format!("{line}\n"));
        let output = world.init_yes(&["--client-key", &key_file(&world)], &[]);
        assert!(output.status.success(), "{options}: {}{}", text(&output.stdout), text(&output.stderr));
        assert!(text(&output.stdout).contains("already allowed"), "{options}: {}", text(&output.stdout));
        assert_eq!(world.authorized_keys(), format!("{line}\n"), "{options}");
    }
}

#[test]
fn other_options_after_the_forced_command_are_reported_not_accepted() {
    let world = World::new();
    let command = restricted(&world).trim_end_matches(&format!(" {KEY}")).to_string();
    for options in [
        ",pty",
        ",port-forwarding",
        ",permitopen=\"10.0.0.1:22\"",
        ",command=\"sh\"",
        ",environment=\"A=b\"",
        ",from=\"10.0.0.0/8\",pty",
        ",no-pty,port-forwarding",
        ",no-pty",
        ",no-touch-required",
        ",from=\"10.0.0.0/8\",no-agent-forwarding",
        ",from=\"10.0.0.0/8 192.0.2.0/24\"",
        ",verify-required,pty",
        ",from=\"10.0.0.0/8\"x",
        ",",
    ] {
        let line = format!("{command}{options} {KEY}");
        seed_authorized_keys(&world, &format!("{line}\n"));
        let output = world.init_yes(&["--client-key", &key_file(&world)], &[]);
        let (out, err) = (text(&output.stdout), text(&output.stderr));
        assert_eq!(output.status.code(), Some(1), "{options}: {out}{err}");
        assert!(err.contains("other options") || err.contains("options after it"), "{options}: {err}");
        assert!(err.contains("only from=") && err.contains("expiry-time=") && err.contains("verify-required"), "{err}");
        assert!(err.contains("containing a space is not recognised"), "{err}");
        assert!(!err.contains("switch ssh features back on"), "{err}");
        assert!(!err.contains("already allowed") && !out.contains("already allowed"), "{options}: {out}{err}");
        assert_eq!(world.authorized_keys(), format!("{line}\n"), "{options}: untouched");
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
    let could_not = "Could not check whether ssh is open to the internet";
    // ss fails before and after the install, or answers nothing: one line says it, the run still succeeds.
    for variant in ["fail", "empty"] {
        let world = World::new();
        let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_SS", variant)]);
        let out = text(&output.stdout);
        assert_eq!(output.status.code(), Some(0), "{variant}: {out}{}", text(&output.stderr));
        assert_eq!(flat(&out).matches(could_not).count(), 1, "{variant}: {out}");
        assert!(flat(&out).contains("check it yourself"), "{out}");
        assert!(!out.contains("stays open"), "{out}");
        let asked = commands(&world.log()).iter().filter(|c| *c == "ss -ltnH -n").count();
        assert_eq!(asked, 2, "asked again after the installer: {variant}");
    }
    // ss missing until the installer installs iproute2: the second try sees the public sshd.
    let world = World::new();
    let output = world.init_yes(&[], &[("ARIA_FAKE_SSH_SS", "fail-until-install")]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    let said = flat(&out);
    assert!(said.contains("stays open until you close it") && !said.contains(could_not), "{out}");
    // The ordinary cases say what they always did, and ask once.
    let world = World::new();
    let public = flat(&text(&world.init_yes(&[], &[]).stdout));
    assert!(public.contains("stays open until you close it") && !public.contains(could_not), "{public}");
    assert_eq!(commands(&world.log()).iter().filter(|c| *c == "ss -ltnH -n").count(), 1);
    let private = flat(&text(&world.init_yes(&[], &[("ARIA_FAKE_SSH_PUBLIC", "0")]).stdout));
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
        let said = flat(&out);
        assert!(said.contains(fix), "{variant}: {out}");
        assert!(said.contains("The sandbox cannot start on the server"), "{variant}: {out}");
        assert!(said.contains(NO_PROMPT_WARNING) && said.contains("shell access to the hub user"), "{variant}: {out}");
        assert!(!out.contains("Your server is ready"), "{out}");
        assert_fits(&out);
        assert_eq!(world.authorized_keys(), "", "{variant}: the key was not installed");
        assert!(!commands(&world.log()).iter().any(|c| c.contains("client-stdio")), "{variant}");
        // The other steps ran.
        assert!(writes(&world.log()).iter().any(|c| c.contains("systemctl")), "{variant}");

        // With the flag the key is installed and the same warning is printed.
        let output =
            world.init_yes(&["--client-key", &key, "--allow-no-sandbox"], &[("ARIA_FAKE_SSH_BWRAP_FAIL", variant)]);
        let (out, err) = (text(&output.stdout), text(&output.stderr));
        assert_eq!(output.status.code(), Some(0), "{variant}: {out}{err}");
        assert!(flat(&out).contains(NO_PROMPT_WARNING), "{variant}: {out}");
        assert_eq!(world.authorized_keys(), format!("{}\n", restricted(&world)), "{variant}");
    }
}

#[test]
fn a_missing_bwrap_and_a_blocked_one_are_told_apart() {
    let world = World::new();
    let missing = text(&world.init_yes(&[], &[("ARIA_FAKE_SSH_BWRAP_FAIL", "missing")]).stdout);
    let missing = flat(&missing);
    assert!(missing.contains("not found") && missing.contains("sudo apt-get install -y bubblewrap"), "{missing}");
    assert!(!missing.contains("apparmor"), "{missing}");
    let blocked = flat(&text(&world.init_yes(&[], &[("ARIA_FAKE_SSH_BWRAP_FAIL", "1")]).stdout));
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
    assert!(!text(&output.stdout).contains("The sandbox cannot start"));
}

#[test]
fn a_restricted_key_that_went_in_is_remembered_and_needs_no_ssh_config() {
    let world = World::new();
    let key = key_file(&world);
    let output = world.init_yes(&["--client-key", &key], &[]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    let private = key.trim_end_matches(".pub");
    // The hub is remembered with the key's file (without `.pub`), and is the default: `aria connect` needs no host.
    let hubs: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(world.path("home/.aria/hubs.json")).unwrap()).unwrap();
    assert_eq!(hubs["default"], "ana@hub.example");
    assert_eq!(hubs["hubs"]["ana@hub.example"]["key"], private);
    let mode = std::os::unix::fs::PermissionsExt::mode(
        &std::fs::metadata(world.path("home/.aria/hubs.json")).unwrap().permissions(),
    );
    assert_eq!(mode & 0o777, 0o600);
    assert!(!out.contains("IdentityFile") && !out.contains("Host aria-hub") && !out.contains("~/.ssh/config"), "{out}");
    assert!(out.contains("It works"), "{out}");
    assert!(out.contains("Your server is ready.") && out.contains("To use it, type:  aria connect\n"), "{out}");
    assert_fits(&out);
    // The connection that was tried used the key and only it.
    let tried: Vec<String> =
        world.log().into_iter().filter(|line| line.contains("client-stdio") && !line.contains("restrict")).collect();
    assert_eq!(tried.len(), 1, "{tried:?}");
    // A key that was already there, restricted, is as good.
    let again = text(&world.init_yes(&["--client-key", &key], &[]).stdout);
    assert!(again.contains("To use it, type:  aria connect"), "{again}");
    assert!(again.contains("The key was already allowed on the server"), "{again}");
    // No key asked for, no snippet and no hub remembered.
    let other = World::new();
    let plain = text(&other.init_yes(&[], &[]).stdout);
    assert!(!plain.contains("IdentityFile") && !plain.contains("Host aria-hub"), "{plain}");
    assert!(!other.path("home/.aria/hubs.json").exists());
    assert!(plain.contains("To use it, type:  aria connect ana@hub.example"), "{plain}");
}

#[test]
fn a_hub_that_cannot_be_remembered_still_comes_with_the_ssh_config() {
    let world = World::new();
    let key = key_file(&world);
    // A folder where the file should be: the write fails, and the old way in is printed.
    std::fs::create_dir_all(world.path("home/.aria/hubs.json")).unwrap();
    let output = world.init_yes(&["--client-key", &key], &[]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    let private = key.trim_end_matches(".pub");
    assert!(flat(&out).contains("could not remember the server"), "{out}");
    for line in [
        "Host aria-hub".to_string(),
        "HostName hub.example".to_string(),
        "User ana".to_string(),
        format!("IdentityFile {private}"),
        "IdentitiesOnly yes".to_string(),
    ] {
        assert!(out.lines().any(|each| each.trim() == line), "{line}: {out}");
    }
    assert!(out.contains("To use it, type:  aria connect aria-hub"), "{out}");
}

/// A stand-in for ssh-keygen that makes the pair the way the real one does (`-q -t ed25519 -f <private>`), or fails when asked to.
fn fake_keygen(world: &World, body: &str) -> String {
    let program = world.path("fake-keygen");
    std::fs::write(&program, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&program, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    program.to_str().unwrap().to_string()
}

fn makes_the_default_key() -> String {
    format!("printf private > \"$5\"; printf '%s\\n' '{KEY}' > \"$5.pub\"")
}

#[test]
fn the_default_key_is_made_allowed_remembered_and_tried() {
    let world = World::new();
    let keygen = fake_keygen(&world, &makes_the_default_key());
    let output = world.init_yes(&["--default-key"], &[("ARIA_SSH_KEYGEN", &keygen)]);
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(0), "{out}{err}");
    // Said before the install, in plain words, and made in a folder only its owner reads.
    let said = out.find("[1/5] Making a key only for Aria").expect("the key is the first step");
    assert!(out.find("What I will do on").unwrap() < said && said < out.find("[2/5]").unwrap(), "{out}");
    assert!(flat(&out).contains("It asks for a passphrase"), "{out}");
    assert!(out.contains("The key is ~/.ssh/aria-hub"), "the path is written with a tilde: {out}");
    assert_fits(&out);
    assert!(world.path("home/.ssh/aria-hub").is_file() && world.path("home/.ssh/aria-hub.pub").is_file());
    let mode =
        std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(world.path("home/.ssh")).unwrap().permissions());
    assert_eq!(mode & 0o777, 0o700);
    assert_eq!(world.authorized_keys(), format!("{}\n", restricted(&world)));
    let hubs: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(world.path("home/.aria/hubs.json")).unwrap()).unwrap();
    assert_eq!(hubs["hubs"]["ana@hub.example"]["key"], world.path("home/.ssh/aria-hub").to_str().unwrap());
    assert!(out.contains("[4/5] Allowing the key on the server"), "{out}");
    assert!(out.contains("[5/5] Testing the connection"), "{out}");
    assert!(out.contains("It works") && out.contains("To use it, type:  aria connect"), "{out}");
    // A second run finds the key and never runs ssh-keygen again (this one would fail).
    let broken = fake_keygen(&world, "exit 9");
    let again = world.init_yes(&["--default-key"], &[("ARIA_SSH_KEYGEN", &broken)]);
    assert_eq!(again.status.code(), Some(0), "{}{}", text(&again.stdout), text(&again.stderr));
}

#[test]
fn a_keygen_that_fails_stops_before_anything_is_changed_on_the_hub() {
    let world = World::new();
    let keygen = fake_keygen(&world, "exit 1");
    let output = world.init_yes(&["--default-key"], &[("ARIA_SSH_KEYGEN", &keygen)]);
    assert_eq!(output.status.code(), Some(1));
    assert!(text(&output.stderr).contains("--no-client-key"), "{}", text(&output.stderr));
    assert!(writes(&world.log()).is_empty(), "{:?}", writes(&world.log()));
}

#[test]
fn a_connection_that_does_not_open_ends_the_run_with_the_reason() {
    let world = World::new();
    let key = key_file(&world);
    let output = world.init_yes(&["--client-key", &key], &[("ARIA_FAKE_SSH_HELLO", "fail")]);
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "{out}{err}");
    assert!(err.contains("connection tried: it did not work"), "{err}");
    assert!(!out.contains("Your server is ready"), "{out}");
}

#[test]
fn the_screen_of_a_run_in_a_pipe_has_no_colour_codes_and_ascii_marks() {
    let world = World::new();
    let key = key_file(&world);
    let out = text(&world.init_yes(&["--client-key", &key], &[]).stdout);
    assert!(!out.contains('\x1b'), "{out}");
    assert!(out.contains("[2/4] Copying your settings and sign-ins"), "{out}");
    assert!(out.contains("  ok The server runs Aria 0.0.0-fake"), "{out}");
    assert_fits(&out);
}

#[test]
fn a_release_hub_has_one_step_fewer_and_names_its_key_step_by_that_number() {
    let world = World::new();
    let key = key_file(&world);
    let output = world.init(&["ana@hub.example", "--yes", "--client-key", &key], &[]);
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    assert!(
        out.contains("[1/4] Installing Aria on the server") && out.contains("[4/4] Testing the connection"),
        "{out}"
    );
    assert!(!out.contains("[5/"), "{out}");
    assert!(flat(&out).contains("Install Aria") && flat(&out).contains("on the server"), "{out}");
    assert_fits(&out);
    // The key's failure names the same number.
    let seeded = World::new();
    seed_authorized_keys(&seeded, &format!("{KEY}\n"));
    let key = key_file(&seeded);
    let failed = seeded.init(&["ana@hub.example", "--yes", "--client-key", &key], &[]);
    assert!(text(&failed.stderr).contains("step 5 (key) failed"), "{}", text(&failed.stderr));
}

#[test]
fn a_hub_whose_only_sign_in_is_a_claude_plan_is_told_it_will_have_no_model() {
    let world = World::new();
    std::fs::write(
        world.path("home/.aria/auth.json"),
        r#"{"anthropic":{"type":"oauth","access":"a","refresh":"r","expires":1}}"#,
    )
    .unwrap();
    let out = text(&world.init_yes(&[], &[]).stdout);
    let said = flat(&out);
    assert!(
        said.contains("a hub cannot use a Claude plan") && said.contains("Anthropic API key or an OpenCode Go key"),
        "{out}"
    );
    assert!(said.contains("aria hub init ana@hub.example --credentials"), "{out}");
    assert_fits(&out);
    // An API key is no reason to warn.
    let other = World::new();
    let out = text(&other.init_yes(&[], &[]).stdout);
    assert!(!out.contains("Claude plan"), "{out}");
}

#[test]
fn connect_with_no_host_and_no_hub_says_how_to_set_one_up() {
    let world = World::new();
    let output = Command::new(env!("CARGO_BIN_EXE_aria"))
        .arg("connect")
        .env_clear()
        .env("HOME", world.path("home"))
        .env("ARIA_HOME", world.path("home/.aria"))
        .env("PATH", world.path("bin"))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains("aria hub init user@host"), "{}", text(&output.stderr));
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
    let output = world.init_yes(&["--details"], &[("ARIA_FAKE_SSH_ACTIVE", "1")]);
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(0), "{out}{err}");
    let steps = writes(&world.log());
    assert!(!steps.iter().any(|c| c.contains("hub-import")), "no bundle was sent: {steps:?}");
    assert!(!steps.iter().any(|c| c.contains("restart")), "no restart: {steps:?}");
    let said = flat(&out);
    assert!(said.contains("The service keeps running (it takes the new version when idle)"), "{out}");
    assert!(said.contains("Aria 0.0.0-fake already runs there"), "the hub's version is read first: {out}");
    assert!(
        said.contains("Kept as the hub has them") && said.contains("--credentials") && said.contains("--settings"),
        "the detailed plan says what is kept: {out}"
    );
    assert!(!out.contains("Copying your"), "nothing to copy, no copy step: {out}");
    assert!(
        flat(&text(&world.init_yes(&[], &[("ARIA_FAKE_SSH_ACTIVE", "1")]).stdout))
            .contains("Nothing to copy: the server keeps the settings and sign-ins it has"),
        "{out}"
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
    assert!(out.contains("Your server is ready"), "{out}");
    assert!(!out.contains("Linger is off"), "{out}");
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
        assert!(!out.contains("Your server is ready"), "{variant}: {out}");
        assert!(flat(&out).contains("installed, but not ready"), "{variant}: {out}");
        assert!(flat(&out).contains("stops when you log out"), "{variant}: {out}");
        assert!(out.contains("sudo loginctl enable-linger ana"), "{variant}: {out}");
        assert_fits(&out);
        for command in commands(&world.log()) {
            assert!(!command.contains("sudo"), "nothing is run with sudo: {command}");
        }
    }
}

#[test]
fn the_plan_says_the_browser_is_left_out_of_a_server() {
    let world = World::new();
    let out = flat(&text(&world.init(&["ana@hub.example", "--binary", "bin/aria-aarch64", "--details"], &[]).stdout));
    assert!(out.contains("--with-browser") && out.contains("agent-browser"), "{out}");
}

#[test]
fn the_plan_says_when_the_binary_was_built() {
    let world = World::new();
    let file = std::fs::File::options().write(true).open(world.path("repo/bin/aria-aarch64")).unwrap();
    file.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000)).unwrap();
    let output = world.init(&["ana@hub.example", "--binary", "bin/aria-aarch64", "--details"], &[]);
    let out = flat(&text(&output.stdout));
    assert!(output.status.success(), "{out}{}", text(&output.stderr));
    assert!(out.contains("bin/aria-aarch64 (built 2023-11-14 22:13 UTC)"), "{out}");
    // the built binary of the default path is shown the same way, and an old one is not refused
    let built = world.path("repo/target/aarch64-unknown-linux-gnu/release");
    std::fs::create_dir_all(&built).unwrap();
    std::fs::write(built.join("aria"), elf(0xB7)).unwrap();
    let default = std::fs::File::options().write(true).open(built.join("aria")).unwrap();
    default.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000)).unwrap();
    let output = world.init(&["ana@hub.example", "--details"], &[]);
    let out = flat(&text(&output.stdout));
    assert!(output.status.success(), "{out}{}", text(&output.stderr));
    assert!(out.contains("release/aria (built 2001-09-09 01:46 UTC)"), "{out}");
}

#[test]
fn every_line_of_a_whole_run_fits_78_columns() {
    // From a checkout, with a key to allow and every command listed.
    let world = World::new();
    let key = key_file(&world);
    let output = world.init_yes(&["--client-key", &key, "--details"], &[]);
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
    assert_fits(&text(&output.stdout));
    // From the release, with the key made, a public ssh port and a user that does not linger: the most the screen says.
    let world = World::new();
    let keygen = fake_keygen(&world, &makes_the_default_key());
    let output = world.init(
        &["ana@hub.example", "--yes", "--default-key", "--details"],
        &[("ARIA_SSH_KEYGEN", &keygen), ("ARIA_FAKE_SSH_LINGER", "no"), ("ARIA_FAKE_SSH_PASSWORDS", "yes")],
    );
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    assert_fits(&out);
    assert!(flat(&out).contains("is open to the internet") && flat(&out).contains("Linger is off"), "{out}");
    // Runs that go wrong: a sandbox that does not work (with and without a key), a keyring that cannot be read, a refused
    // bundle, a connection that does not open.
    for (envs, extra) in [
        (vec![("ARIA_FAKE_SSH_BWRAP_FAIL", "1")], vec![]),
        (vec![("ARIA_FAKE_SSH_BWRAP_FAIL", "missing")], vec!["--client-key"]),
        (vec![("ARIA_FAKE_SSH_BWRAP_NOPROC", "1")], vec!["--client-key"]),
        (vec![("ARIA_FAKE_SSH_HELLO", "fail")], vec!["--client-key"]),
        (vec![("ARIA_FAKE_SSH_SS", "fail")], vec![]),
        (vec![("ARIA_FAKE_SSH_IMPORT_FAIL", "1")], vec![]),
    ] {
        let world = World::new();
        let key = key_file(&world);
        let mut args: Vec<&str> = extra.clone();
        if !extra.is_empty() {
            args.push(&key);
        }
        let output = world.init_yes(&args, &envs);
        assert_fits(&text(&output.stdout));
        assert_fits(&text(&output.stderr).lines().filter(|line| line.starts_with("  ")).collect::<Vec<_>>().join("\n"));
    }
}

#[test]
fn the_plan_lists_only_the_steps_that_will_happen() {
    // No key: install and copy, and nothing about a key or a connection.
    let world = World::new();
    let out = text(&world.init_yes(&[], &[]).stdout);
    let plan = &out[out.find("What I will do on").unwrap()..out.find("[1/").unwrap()];
    assert!(plan.contains("1  Install Aria") && plan.contains("2  Copy your settings and sign-ins"), "{plan}");
    assert!(
        !plan.contains("Allow") && !plan.contains("Test the connection") && !plan.contains("key can open"),
        "{plan}"
    );
    assert!(out.contains("[1/2] Installing Aria") && out.contains("[2/2] Copying your"), "{out}");
    assert!(!out.contains("Allowing the key") && !out.contains("Testing the connection"), "{out}");
    // A key to allow adds the two steps for it, numbered the same way in the headers.
    let world = World::new();
    let key = key_file(&world);
    let out = text(&world.init_yes(&["--client-key", &key], &[]).stdout);
    let plan = &out[out.find("What I will do on").unwrap()..out.find("[1/").unwrap()];
    assert!(plan.contains("3  Allow your Aria key on the server") && plan.contains("4  Test the connection"), "{plan}");
    assert!(plan.contains("it is not a login"), "{plan}");
    assert!(
        out.contains("[3/4] Allowing the key on the server") && out.contains("[4/4] Testing the connection"),
        "{out}"
    );
    // A hub that has everything already has nothing to copy: one step.
    let world = World::new();
    owner_has_all_files(&world);
    seed_hub_files(&world);
    let out = text(&world.init_yes(&[], &[]).stdout);
    let plan = &out[out.find("What I will do on").unwrap()..out.find("[1/").unwrap()];
    assert!(plan.contains("1  Install Aria") && !plan.contains("2  "), "{plan}");
    assert!(!plan.contains("sign-ins are copied"), "{plan}");
    assert!(out.contains("[1/1] Installing Aria"), "{out}");
}

#[test]
fn the_technical_plan_is_printed_only_with_details_and_comes_first() {
    let world = World::new();
    let plain = text(&world.init_yes(&[], &[]).stdout);
    assert!(!plain.contains("Aria will do this, in order"), "{plain}");
    assert!(!flat(&plain).contains("hub-src"), "no command in the default screen: {plain}");
    let detailed = text(&world.init_yes(&["--details"], &[]).stdout);
    let said = flat(&detailed);
    assert!(said.contains("Aria will do this, in order") && said.contains("hub-src"), "{detailed}");
    assert!(detailed.find("Aria will do this").unwrap() < detailed.find("What I will do on").unwrap(), "{detailed}");
    assert_fits(&detailed);
}

#[test]
fn plain_is_accepted_and_changes_nothing_in_the_text() {
    let world = World::new();
    let with = world.init_yes(&["--plain"], &[]);
    assert_eq!(with.status.code(), Some(0), "{}{}", text(&with.stdout), text(&with.stderr));
    let without = World::new().init_yes(&[], &[]);
    assert_eq!(text(&with.stdout), text(&without.stdout));
}

#[test]
fn the_probe_says_what_it_found_without_the_technical_details() {
    let world = World::new();
    let out = text(&world.init_yes(&[], &[]).stdout);
    let head = &out[..out.find("What I will do on").unwrap()];
    assert!(head.contains("Aria · linking this computer to your server") || head.contains("Aria - linking"), "{head}");
    assert!(head.contains("Looking at ana@hub.example..."), "{head}");
    assert!(head.contains("Ubuntu 24.04.2 LTS, aarch64"), "{head}");
    assert!(head.contains("Your settings and sign-ins are ready to send"), "{head}");
    for private in ["/home/ana", "hub.example)", "2222", "ssh -G"] {
        assert!(!head.contains(private), "{private} is not for the probe lines: {head}");
    }
    assert!(!head.contains("Aria already runs there"), "a hub with no service says nothing of one: {head}");
}

// ---- aria hub update ------------------------------------------------------------------------------------------------

/// This Aria's version: what a hub that is up to date says.
const MINE: &str = env!("CARGO_PKG_VERSION");

impl World {
    /// `aria hub update <args>` with the fake ssh and no terminal.
    fn update(&self, args: &[&str], envs: &[(&str, &str)]) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_aria"));
        command
            .args(["hub", "update"])
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

    /// The hubs this computer is linked to (`~/.aria/hubs.json`), the first of `hosts` being the default.
    fn link(&self, hosts: &[&str]) {
        let entries: Vec<String> = hosts.iter().map(|host| format!(r#""{host}": {{"key": "/k/aria-hub"}}"#)).collect();
        std::fs::write(
            self.path("home/.aria/hubs.json"),
            format!(r#"{{"default": "{}", "hubs": {{{}}}}}"#, hosts[0], entries.join(", ")),
        )
        .unwrap();
    }

    fn hubs_json(&self) -> String {
        std::fs::read_to_string(self.path("home/.aria/hubs.json")).unwrap_or_default()
    }
}

/// What a hub that runs `before` says, and says after the installer ran.
fn versions<'a>(before: &'a str, after: &'a str) -> [(&'a str, &'a str); 2] {
    [("ARIA_FAKE_SSH_VERSION", before), ("ARIA_FAKE_SSH_VERSION_AFTER", after)]
}

/// Nothing a `hub update` must never do: send settings, make or allow a key, upload.
fn assert_sent_nothing(world: &World) {
    let log = commands(&world.log());
    for forbidden in ["hub-import", "authorized_keys", "ssh-keygen"] {
        assert!(!log.iter().any(|command| command.contains(forbidden)), "{forbidden} in {log:?}");
    }
    assert!(world.authorized_keys().is_empty(), "no key was allowed");
    assert!(!world.path("home/.ssh").exists(), "no key was made on this computer");
}

#[test]
fn an_older_hub_is_updated_with_the_release_installer_and_gets_nothing_else() {
    let world = World::new();
    world.link(&["ana@hub.example"]);
    let hubs_before = world.hubs_json();
    let mut envs = versions("0.0.1", MINE).to_vec();
    envs.push(("ARIA_FAKE_SSH_ACTIVE", "1"));
    let output = world.update(&["ana@hub.example", "--yes"], &envs);
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(0), "{out}{err}");
    for expected in [
        "updating your server",
        "ana@hub.example runs Aria 0.0.1",
        format!("Update Aria on ana@hub.example from 0.0.1 to {MINE}").as_str(),
        "Check the new version",
        "[1/2] Updating Aria on ana@hub.example",
        "[2/2] Checking the new version",
        format!("The server runs Aria {MINE}").as_str(),
        format!("ana@hub.example runs Aria {MINE}").as_str(),
        "Nothing is sent: no sign-ins, no settings, no key.",
    ] {
        assert!(flat(&out).contains(expected), "{expected:?} not in\n{out}");
    }
    assert_fits(&out);
    // The installer of this version's release, nested, with --yes; and the version read again afterwards.
    let log = commands(&world.log());
    let install = log.iter().find(|command| command.contains("install.sh")).expect("the installer ran");
    for part in [
        format!("releases/download/v{MINE}/install.sh").as_str(),
        format!("ARIA_VERSION=v{MINE}").as_str(),
        "ARIA_NESTED=1",
        "--headless --yes",
    ] {
        assert!(install.contains(part), "{part} not in {install}");
    }
    assert!(!log.iter().any(|command| command.contains("tar -xf")), "a release uploads nothing: {log:?}");
    assert!(
        log.iter().rposition(|c| c.ends_with("aria --version")).unwrap()
            > log.iter().position(|c| c.contains("install.sh")).unwrap(),
        "the version is read again after the installer: {log:?}"
    );
    assert_sent_nothing(&world);
    assert_eq!(world.hubs_json(), hubs_before, "hubs.json is not touched");
}

#[test]
fn a_hub_that_runs_this_version_is_left_alone() {
    let world = World::new();
    world.link(&["ana@hub.example"]);
    let output = world.update(&["ana@hub.example", "--yes"], &[("ARIA_FAKE_SSH_VERSION", MINE)]);
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(0), "{out}{err}");
    assert!(out.contains(&format!("ana@hub.example already runs Aria {MINE}")), "{out}");
    assert!(out.contains(&format!("ana@hub.example runs Aria {MINE}")), "{out}");
    assert!(!out.contains("What I will do"), "nothing to plan: {out}");
    assert_fits(&out);
    let log = commands(&world.log());
    assert!(!log.iter().any(|c| c.contains("install.sh") || c.contains("tar -xf")), "nothing was installed: {log:?}");
    assert_sent_nothing(&world);
}

#[test]
fn a_hub_that_runs_a_newer_version_is_refused() {
    let world = World::new();
    world.link(&["ana@hub.example"]);
    let output = world.update(&["ana@hub.example", "--yes"], &[("ARIA_FAKE_SSH_VERSION", "999.0.0")]);
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "{out}{err}");
    assert!(
        flat(&err).contains(&format!(
            "The server runs a newer Aria (999.0.0) than this one ({MINE}). Update this computer first."
        )),
        "{err}"
    );
    assert!(!commands(&world.log()).iter().any(|c| c.contains("install.sh")), "nothing was installed");
    assert_fits(&out);
}

#[test]
fn a_build_given_by_hand_is_uploaded_and_installed_as_init_does() {
    let world = World::new();
    world.link(&["ana@hub.example"]);
    let output = world.update(
        &["ana@hub.example", "--binary", "bin/aria-aarch64", "--yes", "--details"],
        &[("ARIA_FAKE_SSH_VERSION", "0.0.1")],
    );
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(0), "{out}{err}");
    assert!(flat(&out).contains("from 0.0.1 to the build you gave me"), "{out}");
    assert!(flat(&out).contains("hub-src"), "--details names the commands: {out}");
    let log = world.log();
    assert!(
        log.iter().any(|line| line.contains("tar -xf") && !line.ends_with("stdin=0")),
        "the build was uploaded: {log:?}"
    );
    assert!(
        commands(&log).iter().any(|c| c.contains("install.sh --headless --local") && c.contains("ARIA_NESTED=1")),
        "{log:?}"
    );
    assert_fits(&out);
    assert_sent_nothing(&world);
}

#[test]
fn an_update_that_did_not_take_is_said() {
    let world = World::new();
    world.link(&["ana@hub.example"]);
    // The hub still says the old version after the installer: the release was not what it runs.
    let output = world.update(&["ana@hub.example", "--yes"], &[("ARIA_FAKE_SSH_VERSION", "0.0.1")]);
    assert_eq!(output.status.code(), Some(1), "{}{}", text(&output.stdout), text(&output.stderr));
    assert!(flat(&text(&output.stderr)).contains(&format!("runs Aria 0.0.1 after the update, not {MINE}")));
}

#[test]
fn all_updates_every_linked_hub_and_one_that_fails_does_not_stop_the_others() {
    let world = World::new();
    world.link(&["a@one", "b@two"]);
    let mut envs = versions("0.0.1", MINE).to_vec();
    envs.push(("ARIA_FAKE_SSH_DOWN", "b@two"));
    let output = world.update(&["--all", "--yes"], &envs);
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "one failed: {out}{err}");
    assert!(out.contains("updating your servers"), "{out}");
    assert!(flat(&out).contains(&format!("a@one runs Aria {MINE}")), "the first was updated: {out}");
    assert!(err.contains("b@two"), "the second is reported: {err}");
    let log = world.log();
    assert!(log.iter().any(|line| line.starts_with("a@one\t") && line.contains("install.sh")), "{log:?}");
    assert!(!log.iter().any(|line| line.starts_with("b@two\t") && line.contains("install.sh")), "{log:?}");
    assert_fits(&out);
    assert_sent_nothing(&world);
}

#[test]
fn with_two_hubs_that_need_it_one_plan_lists_them_all() {
    let world = World::new();
    world.link(&["a@one", "b@two"]);
    let output = world.update(&["--all", "--yes"], &versions("0.0.1", MINE));
    let (out, err) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(0), "{out}{err}");
    let said = flat(&out);
    for step in [
        format!("1 Update Aria on a@one from 0.0.1 to {MINE}").as_str(),
        "2 Check the new version on a@one",
        format!("3 Update Aria on b@two from 0.0.1 to {MINE}").as_str(),
        "4 Check the new version on b@two",
        "[3/4] Updating Aria on b@two",
    ] {
        assert!(said.contains(step), "{step} not in {out}");
    }
    assert!(
        said.contains(&format!("a@one runs Aria {MINE}")) && said.contains(&format!("b@two runs Aria {MINE}")),
        "{out}"
    );
    assert_fits(&out);
}

#[test]
fn with_no_host_the_default_hub_is_the_one() {
    let world = World::new();
    world.link(&["a@one", "b@two"]);
    let output = world.update(&["--yes"], &versions("0.0.1", MINE));
    assert_eq!(output.status.code(), Some(0), "{}{}", text(&output.stdout), text(&output.stderr));
    let log = world.log();
    assert!(log.iter().all(|line| !line.starts_with("b@two\t")), "only the default is touched: {log:?}");
    assert!(log.iter().any(|line| line.starts_with("a@one\t") && line.contains("install.sh")), "{log:?}");
}

#[test]
fn with_no_hub_linked_it_says_to_run_init() {
    let world = World::new();
    for args in [&["--yes"][..], &["--all", "--yes"]] {
        let output = world.update(args, &[]);
        assert_eq!(output.status.code(), Some(2), "{}", text(&output.stderr));
        assert!(text(&output.stderr).contains("aria hub init user@host"), "{}", text(&output.stderr));
    }
    assert!(world.log().is_empty(), "nothing was asked of any server");
}

#[test]
fn without_a_yes_and_without_a_terminal_nothing_changes() {
    let world = World::new();
    world.link(&["ana@hub.example"]);
    let output = world.update(&["ana@hub.example"], &versions("0.0.1", MINE));
    let out = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{out}{}", text(&output.stderr));
    assert!(out.contains("nothing was changed"), "{out}");
    assert!(!out.contains("Press Enter"), "{out}");
    assert!(!commands(&world.log()).iter().any(|c| c.contains("install.sh")));
}

#[test]
fn plain_and_details_are_accepted_and_the_arguments_are_checked() {
    let world = World::new();
    world.link(&["ana@hub.example"]);
    let with = world.update(&["ana@hub.example", "--yes", "--plain", "--details"], &versions("0.0.1", MINE));
    assert_eq!(with.status.code(), Some(0), "{}{}", text(&with.stdout), text(&with.stderr));
    let said = flat(&text(&with.stdout));
    assert!(said.contains(&format!("releases/download/v{MINE}/install.sh")), "--details names the installer: {said}");
    for bad in [&["--all", "ana@hub.example"][..], &["a@b", "c@d"], &["--frobnicate"], &["-oProxyCommand=x", "--yes"]] {
        let output = world.update(bad, &[]);
        assert_eq!(output.status.code(), Some(2), "{bad:?}: {}", text(&output.stderr));
    }
}
